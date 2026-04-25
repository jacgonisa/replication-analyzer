#!/usr/bin/env python
"""Call flanked vs unflanked ORIs from FORTE predictions.

Biological constraint: a replication origin must be flanked by
  - a left_fork event upstream (within --flank-kb kb)
  - a right_fork event downstream (within --flank-kb kb)

This script takes per-window predictions from a FORTE model, converts them
to events, then applies the flanking rule to classify each predicted ORI as:
  - flanked   (high-confidence, biologically validated)
  - unflanked (ambiguous — predicted ORI but no flanking fork support)

Outputs:
  <output-dir>/events_left_fork.bed
  <output-dir>/events_right_fork.bed
  <output-dir>/events_ori_all.bed         all predicted ORIs
  <output-dir>/events_ori_flanked.bed     flanked ORIs only
  <output-dir>/events_ori_unflanked.bed   unflanked ORIs
  <output-dir>/flanking_summary.tsv       per-read stats
  <output-dir>/flanking_iou.tsv           IoU evaluation: all vs flanked ORIs

Usage:
  python call_flanked_oris.py \\
      --model   CODEX/models/forte_v2.keras \\
      --config  CODEX/configs/forte_v2.yaml \\
      --split-manifest CODEX/results/forte_v1_conservative/preprocessed_forte_v1_conservative.split_manifest.tsv \\
      --output-dir CODEX/results/forte_v2/flanked_oris \\
      --prob-threshold 0.40 \\
      --flank-kb 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))


# ── IoU helpers ───────────────────────────────────────────────────────────────

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def evaluate_iou(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                 iou_threshold: float = 0.2) -> dict:
    if len(pred_df) == 0 and len(gt_df) == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0, n_pred=0, n_gt=0)
    if len(pred_df) == 0:
        return dict(tp=0, fp=0, fn=len(gt_df), precision=0.0, recall=0.0, f1=0.0, n_pred=0, n_gt=len(gt_df))
    if len(gt_df) == 0:
        return dict(tp=0, fp=len(pred_df), fn=0, precision=0.0, recall=0.0, f1=0.0, n_pred=len(pred_df), n_gt=0)

    tp, fp = 0, 0
    gt_by_read = {rid: grp for rid, grp in gt_df.groupby("read_id")}
    gt_matched = set()

    for _, p in pred_df.iterrows():
        rid = p["read_id"]
        if rid not in gt_by_read:
            fp += 1
            continue
        gt_read = gt_by_read[rid]
        best_iou, best_idx = 0.0, None
        for idx, g in gt_read.iterrows():
            iou = compute_iou(int(p["start"]), int(p["end"]),
                              int(g["start"]), int(g["end"]))
            if iou > best_iou and idx not in gt_matched:
                best_iou, best_idx = iou, idx
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            gt_matched.add(best_idx)
        else:
            fp += 1

    fn = len(gt_df) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_pred=len(pred_df), n_gt=len(gt_df))


# ── Fork-pair flanking logic ──────────────────────────────────────────────────

def apply_flanking_filter(
    ori_events: pd.DataFrame,
    left_fork_events: pd.DataFrame,
    right_fork_events: pd.DataFrame,
    flank_bp: int,
) -> pd.DataFrame:
    """Label each ORI event as flanked or unflanked.

    Flanked = has a left_fork ending within flank_bp upstream of ORI start
              AND a right_fork starting within flank_bp downstream of ORI end.

    Returns ori_events with added columns:
      has_left_fork  : bool
      has_right_fork : bool
      flanked        : bool
      nearest_left_dist  : int (bp from left_fork end to ORI start; NaN if none)
      nearest_right_dist : int (bp from ORI end to right_fork start; NaN if none)
    """
    if len(ori_events) == 0:
        ori_events = ori_events.copy()
        for col in ["has_left_fork", "has_right_fork", "flanked",
                    "nearest_left_dist", "nearest_right_dist"]:
            ori_events[col] = pd.Series(dtype=object)
        return ori_events

    lf_by_read = {rid: grp for rid, grp in left_fork_events.groupby("read_id")} \
        if len(left_fork_events) > 0 else {}
    rf_by_read = {rid: grp for rid, grp in right_fork_events.groupby("read_id")} \
        if len(right_fork_events) > 0 else {}

    rows = []
    for _, ori in ori_events.iterrows():
        rid = ori["read_id"]
        ori_start = int(ori["start"])
        ori_end   = int(ori["end"])

        # Search for left_fork ending upstream (left_fork.end ≤ ori.start + flank_bp)
        has_left = False
        nearest_left_dist = np.nan
        if rid in lf_by_read:
            lf = lf_by_read[rid]
            # left fork should end before (or just inside) the ORI start
            candidates = lf[lf["end"] <= ori_start + flank_bp]
            candidates = candidates[candidates["end"] >= ori_start - flank_bp]
            if len(candidates) > 0:
                # closest one: minimise distance from fork end to ORI start
                dists = (ori_start - candidates["end"]).clip(lower=0)
                nearest_left_dist = int(dists.min())
                has_left = True

        # Search for right_fork starting downstream (right_fork.start ≥ ori.end - flank_bp)
        has_right = False
        nearest_right_dist = np.nan
        if rid in rf_by_read:
            rf = rf_by_read[rid]
            candidates = rf[rf["start"] >= ori_end - flank_bp]
            candidates = candidates[candidates["start"] <= ori_end + flank_bp]
            if len(candidates) > 0:
                dists = (candidates["start"] - ori_end).clip(lower=0)
                nearest_right_dist = int(dists.min())
                has_right = True

        rows.append({
            **ori.to_dict(),
            "has_left_fork": has_left,
            "has_right_fork": has_right,
            "flanked": has_left and has_right,
            "nearest_left_dist": nearest_left_dist,
            "nearest_right_dist": nearest_right_dist,
        })

    return pd.DataFrame(rows)


# ── BED output ────────────────────────────────────────────────────────────────

def events_to_bed(events: pd.DataFrame, path: Path, score_col: str = "max_prob"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, e in events.iterrows():
        score = int(float(e.get(score_col, 0)) * 1000)
        rows.append(f"{e['chr']}\t{int(e['start'])}\t{int(e['end'])}\t{e['read_id']}\t{score}")
    with open(path, "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"  Wrote {len(rows):,} events → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--config",  required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prob-threshold", type=float, default=0.40,
                        help="Probability threshold for ORI event calling (default 0.40)")
    parser.add_argument("--fork-prob-threshold", type=float, default=None,
                        help="Probability threshold for fork event calling (default: same as --prob-threshold). "
                             "Set lower (e.g. 0.20) to find more candidate flanking forks.")
    parser.add_argument("--flank-kb", type=float, default=100.0,
                        help="Max distance (kb) to search for flanking fork events (default 100)")
    parser.add_argument("--max-gap", type=int, default=5000,
                        help="Max gap (bp) between windows for event merging (default 5000)")
    parser.add_argument("--iou-thresholds", nargs="+", type=float,
                        default=[0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    flank_bp = int(args.flank_kb * 1000)
    fork_threshold = args.fork_prob_threshold if args.fork_prob_threshold is not None else args.prob_threshold
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Load val read IDs ─────────────────────────────────────────────────────
    manifest = pd.read_csv(args.split_manifest, sep="\t")
    val_read_ids = manifest[manifest["split"] == "val"]["read_id"].tolist()
    print(f"Val reads: {len(val_read_ids):,}")

    # ── Load GT annotations ───────────────────────────────────────────────────
    def load_bed_gt(path):
        if not path or not Path(path).exists():
            return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
        return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                           names=["chr", "start", "end", "read_id"])

    gt_ori = load_bed_gt(config["data"].get("ori_annotations_bed"))
    gt_ori_val = gt_ori[gt_ori["read_id"].isin(val_read_ids)].copy()
    print(f"GT ORIs in val split: {len(gt_ori_val):,}")

    # ── Load XY cache + run predictions ──────────────────────────────────────
    import tensorflow as tf
    from replication_analyzer_codex.losses import (
        SparseCategoricalFocalLoss, MaskedMacroF1,
        MaskedClassPrecision, MaskedClassRecall,
    )
    from replication_analyzer_codex.evaluation import predict_reads, windows_to_events
    from replication_analyzer.models.base import SelfAttention
    from replication_analyzer.models.losses import MultiClassFocalLoss
    from replication_analyzer.training.callbacks import MultiClassF1Score
    from replication_analyzer.data.loaders import load_all_xy_data

    CUSTOM_OBJECTS = {
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
        "MaskedMacroF1": MaskedMacroF1,
        "MaskedClassPrecision": MaskedClassPrecision,
        "MaskedClassRecall": MaskedClassRecall,
        "SelfAttention": SelfAttention,
        "MultiClassFocalLoss": MultiClassFocalLoss,
        "MultiClassF1Score": MultiClassF1Score,
    }

    print(f"\nLoading model: {args.model}")
    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(args.model, custom_objects=CUSTOM_OBJECTS)
    max_length = model.input_shape[1]
    print(f"  max_length = {max_length}")

    print("\nLoading XY data...")
    xy_cache = config["data"].get("xy_cache_path")
    if xy_cache and Path(xy_cache).exists():
        import pickle
        with open(xy_cache, "rb") as fh:
            xy_data = pickle.load(fh)
        print(f"  Loaded from cache: {xy_cache}")
    else:
        xy_data = load_all_xy_data(
            base_dir=config["data"]["base_dir"],
            run_dirs=config["data"].get("run_dirs"),
        )

    print(f"\nRunning predictions on {len(val_read_ids):,} val reads...")
    preprocessing_config = config["preprocessing"]
    predictions = predict_reads(model, xy_data, val_read_ids, max_length, preprocessing_config)
    print(f"  {len(predictions):,} prediction rows")

    # ── Convert windows → events ──────────────────────────────────────────────
    print(f"\nConverting to events (ori_threshold={args.prob_threshold}, fork_threshold={fork_threshold}, max_gap={args.max_gap})...")
    left_fork_events  = windows_to_events(predictions, class_id=1,
                                          prob_threshold=fork_threshold,
                                          max_gap=args.max_gap)
    right_fork_events = windows_to_events(predictions, class_id=2,
                                          prob_threshold=fork_threshold,
                                          max_gap=args.max_gap)
    ori_events        = windows_to_events(predictions, class_id=3,
                                          prob_threshold=args.prob_threshold,
                                          max_gap=args.max_gap)

    print(f"  Left fork events:  {len(left_fork_events):,}")
    print(f"  Right fork events: {len(right_fork_events):,}")
    print(f"  ORI events (all):  {len(ori_events):,}")

    # ── Apply flanking filter ─────────────────────────────────────────────────
    print(f"\nApplying flanking filter (flank_bp={flank_bp:,})...")
    ori_labeled = apply_flanking_filter(
        ori_events, left_fork_events, right_fork_events, flank_bp
    )

    ori_flanked   = ori_labeled[ori_labeled["flanked"]].copy()
    ori_unflanked = ori_labeled[~ori_labeled["flanked"]].copy()

    n_flanked   = len(ori_flanked)
    n_unflanked = len(ori_unflanked)
    pct_flanked = 100 * n_flanked / max(len(ori_labeled), 1)

    print(f"  Flanked ORIs:   {n_flanked:,}  ({pct_flanked:.1f}%)")
    print(f"  Unflanked ORIs: {n_unflanked:,}  ({100-pct_flanked:.1f}%)")

    # Per-read summary
    summary_rows = []
    for rid in val_read_ids:
        n_lf  = len(left_fork_events[left_fork_events["read_id"] == rid])
        n_rf  = len(right_fork_events[right_fork_events["read_id"] == rid])
        n_ori = len(ori_events[ori_events["read_id"] == rid]) if len(ori_events) > 0 else 0
        n_fl  = len(ori_flanked[ori_flanked["read_id"] == rid]) if len(ori_flanked) > 0 else 0
        summary_rows.append(dict(read_id=rid, n_left_fork=n_lf, n_right_fork=n_rf,
                                 n_ori_all=n_ori, n_ori_flanked=n_fl))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "flanking_summary.tsv", sep="\t", index=False)

    # ── Save BED files ────────────────────────────────────────────────────────
    print("\nSaving BED files...")
    events_to_bed(left_fork_events,  output_dir / "events_left_fork.bed")
    events_to_bed(right_fork_events, output_dir / "events_right_fork.bed")
    events_to_bed(ori_events,        output_dir / "events_ori_all.bed")
    events_to_bed(ori_flanked,       output_dir / "events_ori_flanked.bed")
    events_to_bed(ori_unflanked,     output_dir / "events_ori_unflanked.bed")

    # ── IoU evaluation: all ORIs vs flanked ORIs ─────────────────────────────
    print("\nIoU evaluation...")
    iou_rows = []
    for iou_thr in args.iou_thresholds:
        for label, pred_df in [("all_oris", ori_events), ("flanked_oris", ori_flanked)]:
            m = evaluate_iou(pred_df, gt_ori_val, iou_threshold=iou_thr)
            iou_rows.append(dict(
                label=label,
                iou_threshold=iou_thr,
                prob_threshold=args.prob_threshold,
                flank_kb=args.flank_kb,
                **m,
            ))
        print(f"  IoU≥{iou_thr}  all={iou_rows[-2]['recall']:.3f} recall  "
              f"flanked={iou_rows[-1]['recall']:.3f} recall  "
              f"(flanked prec={iou_rows[-1]['precision']:.3f})")

    iou_df = pd.DataFrame(iou_rows)
    iou_df.to_csv(output_dir / "flanking_iou.tsv", sep="\t", index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FLANKED-ORI SUMMARY")
    print("=" * 60)
    print(f"  Model:            {args.model}")
    print(f"  ORI threshold:    {args.prob_threshold}")
    print(f"  Fork threshold:   {fork_threshold}")
    print(f"  Flank window:     ±{args.flank_kb} kb")
    print(f"  Val reads:        {len(val_read_ids):,}")
    print(f"  GT ORIs (val):    {len(gt_ori_val):,}")
    print(f"  Predicted left forks:  {len(left_fork_events):,}")
    print(f"  Predicted right forks: {len(right_fork_events):,}")
    print(f"  Predicted ORIs (all):  {len(ori_events):,}")
    print(f"  Flanked ORIs:          {n_flanked:,}  ({pct_flanked:.1f}%)")
    print(f"  Unflanked ORIs:        {n_unflanked:,}  ({100-pct_flanked:.1f}%)")
    print(f"\n  IoU ≥ 0.2:")
    r_all = iou_df[(iou_df["label"] == "all_oris") & (iou_df["iou_threshold"] == 0.2)].iloc[0]
    r_fl  = iou_df[(iou_df["label"] == "flanked_oris") & (iou_df["iou_threshold"] == 0.2)].iloc[0]
    print(f"    All ORIs:     recall={r_all['recall']:.3f}  precision={r_all['precision']:.3f}  "
          f"F1={r_all['f1']:.3f}  n={int(r_all['n_pred'])}")
    print(f"    Flanked ORIs: recall={r_fl['recall']:.3f}  precision={r_fl['precision']:.3f}  "
          f"F1={r_fl['f1']:.3f}  n={int(r_fl['n_pred'])}")
    print(f"\n  Outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
