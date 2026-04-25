#!/usr/bin/env python
"""Sweep fork pseudo-label thresholds on ALL reads (train+val) to count
how many ORIs would be flanked at each threshold.

Runs v2 model inference once, then sweeps fork thresholds to measure:
  - n_left_fork / n_right_fork pseudo-labels
  - n_flanked_oris (pseudo + real, flanked by those forks within 100kb)

Usage (from /replication-analyzer/):
  CUDA_VISIBLE_DEVICES="" conda run -n ONT python CODEX/scripts/sweep_fork_thresholds_training.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer.models.base import SelfAttention
from replication_analyzer_codex.losses import (
    MaskedClassPrecision, MaskedClassRecall, MaskedMacroF1,
    SparseCategoricalFocalLoss,
)
from replication_analyzer_codex.evaluation import (
    load_xy_for_prediction, predict_reads, windows_to_events,
)

BASE = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")
CONFIG_PATH  = BASE / "CODEX/configs/forte_v2.yaml"
MODEL_PATH   = BASE / "CODEX/models/forte_v2.keras"
PSEUDO_ORI   = BASE / "CODEX/results/forte_v2/pseudo_labels/combined_origin.bed"

FORK_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
FLANK_BP = 100_000


def load_bed(path):
    df = pd.read_csv(path, sep="\t", header=None, low_memory=False)
    df.columns = ["chr", "start", "end", "read_id"] + [f"c{i}" for i in range(4, len(df.columns))]
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    return df


def count_flanked(ori_df, lf_df, rf_df, flank_bp):
    """Returns (n_flanked_oris, n_lf_flanking, n_rf_flanking).

    n_lf_flanking = left forks that are within flank_bp of at least one ORI start
    n_rf_flanking = right forks that are within flank_bp of at least one ORI end
    These are "useful" forks that participate in a triplet.
    """
    lf_by = {rid: grp for rid, grp in lf_df.groupby("read_id")} if len(lf_df) else {}
    rf_by = {rid: grp for rid, grp in rf_df.groupby("read_id")} if len(rf_df) else {}
    ori_by = {rid: grp for rid, grp in ori_df.groupby("read_id")}

    flanked = 0
    lf_flanking_idx = set()
    rf_flanking_idx = set()

    for _, ori in ori_df.iterrows():
        rid, os, oe = ori["read_id"], int(ori["start"]), int(ori["end"])
        has_left = has_right = False

        if rid in lf_by:
            lf = lf_by[rid]
            cands = lf[(lf["end"] >= os - flank_bp) & (lf["end"] <= os + flank_bp)]
            if len(cands):
                has_left = True
                lf_flanking_idx.update(cands.index.tolist())

        if rid in rf_by:
            rf = rf_by[rid]
            cands = rf[(rf["start"] >= oe - flank_bp) & (rf["start"] <= oe + flank_bp)]
            if len(cands):
                has_right = True
                rf_flanking_idx.update(cands.index.tolist())

        if has_left and has_right:
            flanked += 1

    return flanked, len(lf_flanking_idx), len(rf_flanking_idx)


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print("Loading model...")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={
            "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
            "MaskedMacroF1": MaskedMacroF1,
            "MaskedClassPrecision": MaskedClassPrecision,
            "MaskedClassRecall": MaskedClassRecall,
            "SelfAttention": SelfAttention,
        },
        compile=False,
        safe_mode=False,
    )

    print("Loading reads (all — train + val)...")
    xy_data = load_xy_for_prediction(cfg)
    max_length = model.input_shape[1]
    read_ids = xy_data["read_id"].unique().tolist()
    print(f"  Loaded {len(read_ids):,} reads")

    pred_cache = BASE / "CODEX/results/fork_threshold_sweep_predictions.tsv"
    if pred_cache.exists():
        print(f"Loading cached predictions from {pred_cache}...")
        predictions = pd.read_csv(pred_cache, sep="\t")
        print(f"  Loaded {len(predictions):,} rows")
    else:
        print("Running inference (this may take a few minutes)...")
        predictions = predict_reads(
            model=model,
            xy_data=xy_data,
            read_ids=read_ids,
            max_length=max_length,
            preprocessing_config=cfg["preprocessing"],
        )
        print(f"  Predicted {len(predictions):,} rows in predictions table")
        predictions.to_csv(pred_cache, sep="\t", index=False)
        print(f"  Cached → {pred_cache}")

    print("\nLoading ORI annotations (combined real+pseudo)...")
    ori_df = load_bed(PSEUDO_ORI)
    print(f"  {len(ori_df):,} ORIs")

    print(f"\nSweep results (flank=±100kb):")
    print("-" * 90)

    LF_CLASS = 1  # left_fork class id
    RF_CLASS = 2  # right_fork class id

    rows = []
    for thr in FORK_THRESHOLDS:
        lf_events = windows_to_events(
            predictions=predictions, class_id=LF_CLASS,
            prob_threshold=thr, min_windows=1, max_gap=5000,
        )
        rf_events = windows_to_events(
            predictions=predictions, class_id=RF_CLASS,
            prob_threshold=thr, min_windows=1, max_gap=5000,
        )

        def events_to_df(evs):
            if isinstance(evs, pd.DataFrame):
                if evs.empty:
                    return pd.DataFrame(columns=["chr","start","end","read_id"])
                return evs[["chr","start","end","read_id"]].copy()
            if not evs:
                return pd.DataFrame(columns=["chr","start","end","read_id"])
            return pd.DataFrame([{"chr": e["chr"], "start": e["start"],
                                   "end": e["end"], "read_id": e["read_id"]} for e in evs])

        lf_df = events_to_df(lf_events)
        rf_df = events_to_df(rf_events)

        n_flanked, n_lf_useful, n_rf_useful = count_flanked(ori_df, lf_df, rf_df, FLANK_BP)
        pct_ori    = 100 * n_flanked  / max(len(ori_df), 1)
        pct_lf_use = 100 * n_lf_useful / max(len(lf_df), 1)
        pct_rf_use = 100 * n_rf_useful / max(len(rf_df), 1)

        print(f"  thr={thr:.2f}  LF={len(lf_df):>7,} (useful {pct_lf_use:>4.1f}%)  "
              f"RF={len(rf_df):>7,} (useful {pct_rf_use:>4.1f}%)  "
              f"flanked_ORI={n_flanked:>6,} ({pct_ori:.1f}%)")
        rows.append(dict(fork_thr=thr,
                         n_lf=len(lf_df), n_lf_useful=n_lf_useful, pct_lf_useful=round(pct_lf_use,1),
                         n_rf=len(rf_df), n_rf_useful=n_rf_useful, pct_rf_useful=round(pct_rf_use,1),
                         n_flanked_ori=n_flanked, pct_flanked_ori=round(pct_ori,1)))

    out = BASE / "CODEX/results/fork_threshold_sweep_training.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
