#!/usr/bin/env python
"""Evaluate model separately on human-annotated vs AI pseudo-label ground truth.

For each class the GT events (from the training BED files) are split into:
  - human   : intervals that overlap the original human-annotated source files
  - pseudo  : intervals that do NOT overlap any human-annotated source

The model is evaluated (precision / recall / F1 / event IoU) against each
subset independently, allowing a clean comparison.

Human source files (hardcoded to the canonical Nerea/fork annotation paths):
  left_fork:  data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed
  right_fork: data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed
  origin:     data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed

Usage:
  CUDA_VISIBLE_DEVICES="" python CODEX/scripts/evaluate_human_vs_pseudo.py \\
      --config CODEX/configs/forte_v5.0.yaml \\
      --model  CODEX/models/forte_v5.0.keras \\
      --split  test
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import tensorflow as tf
import yaml

ROOT      = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer.models.base import SelfAttention
from replication_analyzer_codex.annotations import load_annotations_for_codex
from replication_analyzer_codex.evaluation import (
    load_xy_for_prediction,
    predict_reads,
    resolve_active_class_ids,
    run_threshold_sweep,
)
from replication_analyzer_codex.losses import (
    MaskedClassPrecision, MaskedClassRecall, MaskedMacroF1,
    MaskedMeanIoU, SparseCategoricalFocalLoss,
)
from replication_analyzer_codex.splits import load_split_manifest
from replication_analyzer_codex.constants import CLASS_NAME_TO_ID

DATA_DIR = ROOT / "data/case_study_jan2026/combined/annotations"

HUMAN_SOURCES = {
    "left_fork":  DATA_DIR / "leftForks_ALL_combined.bed",
    "right_fork": DATA_DIR / "rightForks_ALL_combined.bed",
    "origin":     DATA_DIR / "ORIs_combined_cleaned.bed",
}


def load_bed4(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def _load_model(model_path):
    custom_objects = {
        "SelfAttention":               SelfAttention,
        "SparseCategoricalFocalLoss":  SparseCategoricalFocalLoss,
        "MaskedMacroF1":               MaskedMacroF1,
        "MaskedMeanIoU":               MaskedMeanIoU,
        "MaskedClassPrecision":        MaskedClassPrecision,
        "MaskedClassRecall":           MaskedClassRecall,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects,
                                      safe_mode=False)


def build_human_index(human_df):
    """Build a per-read sorted interval list for fast overlap queries."""
    idx = {}
    for row in human_df.itertuples(index=False):
        idx.setdefault(row.read_id, []).append((int(row.start), int(row.end)))
    return idx


def overlaps_human(read_id, start, end, human_idx):
    for hs, he in human_idx.get(read_id, []):
        if start < he and end > hs:
            return True
    return False


def split_gt_human_pseudo(gt_df, human_idx):
    """Split GT dataframe into human and pseudo subsets."""
    mask = gt_df.apply(
        lambda r: overlaps_human(r["read_id"], int(r["start"]), int(r["end"]), human_idx),
        axis=1,
    )
    return gt_df[mask].copy(), gt_df[~mask].copy()


def summarise(sweep_df, label):
    """Print best-threshold row per event type."""
    best = (
        sweep_df.sort_values(["event_type", "f1"], ascending=[True, False])
        .groupby("event_type", as_index=False).first()
    )
    print(f"\n── {label} ──")
    cols = ["event_type", "prob_threshold", "iou_threshold",
            "precision", "recall", "f1", "event_iou", "num_ground_truth"]
    available = [c for c in cols if c in best.columns]
    print(best[available].to_string(index=False))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model",  required=True)
    parser.add_argument("--split",  default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # ── Load split manifest ───────────────────────────────────────────────────
    manifest_path = config["data"].get("split_manifest")
    if not manifest_path:
        # Fall back to the auto-generated manifest in the results dir
        results_dir = config["output"]["results_dir"]
        experiment  = config.get("experiment_name", "model")
        manifest_path = str(Path(results_dir) / f"preprocessed_{experiment}.split_manifest.tsv")
        if not Path(manifest_path).exists():
            # Last resort: any split_manifest.tsv in results dir
            candidates = list(Path(results_dir).glob("*.split_manifest.tsv")) + \
                         list(Path(results_dir).glob("split_manifest.tsv"))
            if candidates:
                manifest_path = str(sorted(candidates)[-1])
            else:
                raise FileNotFoundError(
                    f"No split manifest found for {experiment}. "
                    f"Set data.split_manifest in config or pre-generate it."
                )
        print(f"  Using split manifest: {manifest_path}")
    manifest   = load_split_manifest(manifest_path)
    read_ids   = manifest.loc[manifest["split"] == args.split, "read_id"].tolist()
    print(f"Split '{args.split}': {len(read_ids):,} reads")

    # ── Load model and predict ────────────────────────────────────────────────
    print("Loading model…")
    model      = _load_model(args.model)
    max_length = model.input_shape[1]

    print("Loading XY data…")
    xy_data = load_xy_for_prediction(config)

    print("Running predictions…")
    predictions = predict_reads(
        model=model,
        xy_data=xy_data,
        read_ids=read_ids,
        max_length=max_length,
        preprocessing_config=config["preprocessing"],
    )

    # ── Load GT (training BEDs, filtered to this split) ───────────────────────
    print("Loading ground truth annotations…")
    left_forks, right_forks, origins, _ = load_annotations_for_codex(config)
    gt_all = {
        CLASS_NAME_TO_ID["left_fork"]:  left_forks,
        CLASS_NAME_TO_ID["right_fork"]: right_forks,
        CLASS_NAME_TO_ID["origin"]:     origins,
    }
    for cid in list(gt_all):
        gt_all[cid] = gt_all[cid][gt_all[cid]["read_id"].isin(read_ids)].copy()

    # ── Load human source files and build indices ─────────────────────────────
    print("Loading human-annotated source BEDs…")
    human_idx = {}
    for event_type, path in HUMAN_SOURCES.items():
        df  = load_bed4(path)
        idx = build_human_index(df[df["read_id"].isin(read_ids)])
        human_idx[CLASS_NAME_TO_ID[event_type]] = idx
        print(f"  {event_type}: {sum(len(v) for v in idx.values()):,} human intervals on split reads")

    # ── Split GT into human vs pseudo ─────────────────────────────────────────
    print("\nSplitting GT into human / pseudo…")
    gt_human  = {}
    gt_pseudo = {}
    for cid, df in gt_all.items():
        h, p = split_gt_human_pseudo(df, human_idx[cid])
        gt_human[cid]  = h
        gt_pseudo[cid] = p
        name = {v: k for k, v in CLASS_NAME_TO_ID.items()}[cid]
        print(f"  {name}: {len(df):,} total → {len(h):,} human + {len(p):,} pseudo")

    # ── Evaluation settings ───────────────────────────────────────────────────
    prob_thresholds = config["evaluation"].get("probability_thresholds", [0.3, 0.4, 0.5])
    iou_thresholds  = config["evaluation"].get("iou_thresholds",         [0.2, 0.3, 0.5])
    min_windows     = config["evaluation"].get("min_windows", 1)
    max_gap         = config["evaluation"].get("max_gap", 5000)
    active_ids      = resolve_active_class_ids(
        event_types=config.get("evaluation", {}).get("active_event_types"),
        n_classes=config["model"].get("n_classes", 4),
    )

    output_dir = Path(config["output"]["results_dir"]) / f"evaluation_{args.split}_human_vs_pseudo"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)

    # ── Run sweeps ────────────────────────────────────────────────────────────
    results = {}
    for label, gt in [("ALL", gt_all), ("HUMAN", gt_human), ("PSEUDO", gt_pseudo)]:
        print(f"\nRunning threshold sweep: {label}…")
        sweep = run_threshold_sweep(
            predictions=predictions,
            ground_truth_by_class=gt,
            probability_thresholds=prob_thresholds,
            iou_thresholds=iou_thresholds,
            min_windows=min_windows,
            max_gap=max_gap,
            active_class_ids=active_ids,
        )
        sweep["gt_subset"] = label
        sweep.to_csv(output_dir / f"threshold_sweep_{label.lower()}.tsv", sep="\t", index=False)
        results[label] = sweep

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY (best threshold per event type per subset)")
    print("="*60)
    bests = []
    for label in ["ALL", "HUMAN", "PSEUDO"]:
        best = summarise(results[label], label)
        best["gt_subset"] = label
        bests.append(best)

    summary = pd.concat(bests, ignore_index=True)
    summary.to_csv(output_dir / "summary_human_vs_pseudo.tsv", sep="\t", index=False)
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
