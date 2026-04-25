#!/usr/bin/env python
"""Fork-pairing post-processing for FORTE predictions.

Biology: every origin lies between a left_fork (to its left) and a right_fork (to its right).
FORTE v1 under-predicts left_forks (928 vs 3,617 right_fork windows) despite balanced training
labels, because the BiLSTM learned a strong origin→right_fork contextual cue.

Fix: for each predicted origin event, if a fork is missing on one side, search within
`--max-gap` bp for the window with the highest fork probability and promote it as an event
(using a lower probability floor `--min-fork-prob` than the main threshold).

Outputs
-------
CODEX/results/forte_v1/fork_pairing/
  pairing_iou_comparison.png   — recall/precision/F1 bars before vs after pairing
  pairing_summary.tsv          — numeric results table

Usage
-----
  CUDA_VISIBLE_DEVICES="" python evaluate_fork_pairing.py
  CUDA_VISIBLE_DEVICES="" python evaluate_fork_pairing.py --min-fork-prob 0.10 --max-gap 75000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import yaml

ROOT       = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

import tensorflow as tf
from replication_analyzer_codex.losses import (
    SparseCategoricalFocalLoss, MaskedMacroF1,
    MaskedClassPrecision, MaskedClassRecall,
)
from replication_analyzer_codex.evaluation import (
    predict_reads, windows_to_events, evaluate_event_predictions,
)
from replication_analyzer.models.base import SelfAttention
from replication_analyzer.models.losses import MultiClassFocalLoss
from replication_analyzer.training.callbacks import MultiClassF1Score

CUSTOM_OBJECTS = {
    "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
    "MaskedMacroF1": MaskedMacroF1,
    "MaskedClassPrecision": MaskedClassPrecision,
    "MaskedClassRecall": MaskedClassRecall,
    "SelfAttention": SelfAttention,
    "MultiClassFocalLoss": MultiClassFocalLoss,
    "MultiClassF1Score": MultiClassF1Score,
}

BASE = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")

CLASS_NAMES = {1: "left_fork", 2: "right_fork", 3: "origin"}
CLASS_COLORS = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}


# ── GT loading ────────────────────────────────────────────────────────────────

def load_bed4(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def gt_to_events(bed_df, event_type):
    if len(bed_df) == 0:
        return pd.DataFrame(columns=["read_id", "chr", "start", "end", "event_type"])
    df = bed_df.copy()
    df["event_type"] = event_type
    return df[["read_id", "chr", "start", "end", "event_type"]]


# ── Core pairing algorithm ────────────────────────────────────────────────────

def pair_forks_around_origins(
    preds: pd.DataFrame,
    threshold: float = 0.40,
    min_fork_prob: float = 0.12,
    max_gap: int = 60_000,
    min_windows: int = 1,
    merge_gap: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baseline: windows_to_events at `threshold` for left/right forks.
    Paired:   for each predicted origin event, if the left or right fork is absent
              within `max_gap`, find the best candidate window (prob >= min_fork_prob)
              and add it as a synthetic event.

    Returns (baseline_fork_events, paired_fork_events).
    Both DataFrames contain all three event types (left_fork, right_fork, origin).
    """
    # ── baseline events ───────────────────────────────────────────────────────
    baseline_parts = []
    for cls_id, cls_name in CLASS_NAMES.items():
        evs = windows_to_events(preds, cls_id, threshold,
                                min_windows=min_windows, max_gap=merge_gap)
        if len(evs) > 0:
            evs["event_type"] = cls_name
        baseline_parts.append(evs)
    baseline_all = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()

    # ── origin events (same in both) ──────────────────────────────────────────
    ori_events = windows_to_events(preds, 3, threshold,
                                   min_windows=min_windows, max_gap=merge_gap)

    # ── per-read lookup tables ─────────────────────────────────────────────────
    reads_by_id = {rid: grp for rid, grp in preds.groupby("read_id")}

    # ── pairing logic ─────────────────────────────────────────────────────────
    synthetic_rows = []   # new events to add

    for _, ori in ori_events.iterrows():
        read_id   = ori["read_id"]
        ori_start = int(ori["start"])
        ori_end   = int(ori["end"])
        read_preds = reads_by_id.get(read_id)
        if read_preds is None:
            continue

        # existing fork events for this read (from baseline)
        bl_read = baseline_all[baseline_all["read_id"] == read_id] if len(baseline_all) else pd.DataFrame()

        for fork_cls, fork_name, prob_col, side in [
            (1, "left_fork",  "prob_left_fork",  "left"),
            (2, "right_fork", "prob_right_fork", "right"),
        ]:
            # Check if a fork event already exists on the correct side
            if len(bl_read):
                fork_events_this = bl_read[bl_read["event_type"] == fork_name]
                if side == "left":
                    existing = fork_events_this[fork_events_this["end"] <= ori_start + merge_gap]
                    existing = existing[existing["end"] >= ori_start - max_gap]
                else:
                    existing = fork_events_this[fork_events_this["start"] >= ori_end - merge_gap]
                    existing = existing[existing["start"] <= ori_end + max_gap]
                if len(existing) > 0:
                    continue   # fork already present, nothing to add

            # Search for candidate windows
            if side == "left":
                candidates = read_preds[
                    (read_preds["end"] <= ori_start + merge_gap) &
                    (read_preds["start"] >= ori_start - max_gap)
                ]
            else:
                candidates = read_preds[
                    (read_preds["start"] >= ori_end - merge_gap) &
                    (read_preds["end"] <= ori_end + max_gap)
                ]

            if len(candidates) == 0:
                continue

            # Pick best window by fork probability
            best_idx = candidates[prob_col].idxmax()
            best_prob = candidates.loc[best_idx, prob_col]
            if best_prob < min_fork_prob:
                continue

            # Add as synthetic event (single window → will merge with nearby promoted windows)
            row = candidates.loc[best_idx]
            synthetic_rows.append({
                "read_id":    read_id,
                "chr":        row["chr"],
                "start":      int(row["start"]),
                "end":        int(row["end"]),
                "event_type": fork_name,
                "synthetic":  True,
                "prob":       float(best_prob),
            })

    # ── build paired event set ─────────────────────────────────────────────────
    if synthetic_rows:
        synth_df = pd.DataFrame(synthetic_rows)
        paired_all = pd.concat([baseline_all, synth_df[
            ["read_id", "chr", "start", "end", "event_type"]
        ]], ignore_index=True)
    else:
        paired_all = baseline_all.copy()

    print(f"  Synthetic fork events added: {len(synthetic_rows)}")
    return baseline_all, paired_all


# ── Evaluation wrapper ─────────────────────────────────────────────────────────

def eval_events(events_df, gt_left, gt_right, gt_ori, iou_thr, val_read_ids=None):
    """Evaluate events. If val_read_ids is given, restrict GT to those reads only
    (prevents train-set GT annotations from inflating false-negative count)."""
    results = {}
    for cls_name, gt_df in [("left_fork", gt_left), ("right_fork", gt_right), ("origin", gt_ori)]:
        pred = events_df[events_df["event_type"] == cls_name] if len(events_df) else pd.DataFrame()
        gt   = gt_to_events(gt_df, cls_name)
        if val_read_ids is not None and len(gt):
            gt = gt[gt["read_id"].isin(val_read_ids)]
        r    = evaluate_event_predictions(pred, gt, iou_thr)
        results[cls_name] = r
    return results


def print_results(label, results):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  {'class':<14} {'recall':>7} {'precision':>10} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    for cls, r in results.items():
        print(f"  {cls:<14} {r['recall']:>7.3f} {r['precision']:>10.3f} {r['f1']:>7.3f} "
              f"{r['true_positives']:>6} {r['false_positives']:>6} {r['false_negatives']:>6}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_comparison(baseline, paired, iou_thr, out_path):
    classes = list(CLASS_NAMES.values())
    metrics = ["recall", "precision", "f1"]
    x = np.arange(len(classes))
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.35})

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        bl_vals = [baseline[c][metric] for c in classes]
        pa_vals = [paired[c][metric]   for c in classes]
        bars1 = ax.bar(x - width/2, bl_vals, width, label="Baseline",
                       color=[CLASS_COLORS[i+1] for i in range(len(classes))],
                       alpha=0.55, edgecolor="white")
        bars2 = ax.bar(x + width/2, pa_vals, width, label="Fork-paired",
                       color=[CLASS_COLORS[i+1] for i in range(len(classes))],
                       alpha=1.0, edgecolor="white")

        # delta labels
        for i, (b, p) in enumerate(zip(bl_vals, pa_vals)):
            delta = p - b
            if abs(delta) > 0.001:
                col = "green" if delta > 0 else "red"
                ax.text(x[i] + width/2, p + 0.01, f"{delta:+.3f}",
                        ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")

        ax.set_ylim(0, min(1.05, max(max(bl_vals), max(pa_vals)) + 0.15))
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(metric.capitalize(), fontsize=9)
        ax.set_title(f"{metric.capitalize()} @ IoU≥{iou_thr}", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if mi == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Fork-pairing post-processing: Baseline vs Paired (FORTE v1)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="CODEX/configs/forte_v1.yaml")
    parser.add_argument("--model",    default="CODEX/models/forte_v1.keras")
    parser.add_argument("--xy-cache", default="CODEX/results/cache/xy_data.pkl")
    parser.add_argument("--val-info",
        default="CODEX/results/forte_v1/preprocessed_forte_v1.val_info.tsv")
    parser.add_argument("--gt-left",
        default="data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed")
    parser.add_argument("--gt-right",
        default="data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed")
    parser.add_argument("--gt-ori",
        default="data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed")
    parser.add_argument("--n-reads",      type=int,   default=3068,
        help="Val reads to evaluate (use 3068 for full val set)")
    parser.add_argument("--threshold",    type=float, default=0.40,
        help="Main probability threshold for event calling")
    parser.add_argument("--min-fork-prob",type=float, default=0.12,
        help="Minimum fork probability to promote a candidate window")
    parser.add_argument("--max-gap",      type=int,   default=60_000,
        help="Max distance (bp) from origin boundary to search for missing forks")
    parser.add_argument("--iou-thr",      type=float, default=0.20)
    parser.add_argument("--output-dir",
        default="CODEX/results/forte_v1/fork_pairing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = BASE / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── Load GT ───────────────────────────────────────────────────────────────
    gt_left  = load_bed4(str(BASE / args.gt_left))
    gt_right = load_bed4(str(BASE / args.gt_right))
    gt_ori   = load_bed4(str(BASE / args.gt_ori))
    print(f"GT: {len(gt_left)} left-forks | {len(gt_right)} right-forks | {len(gt_ori)} origins")

    # ── Load model + run inference ─────────────────────────────────────────────
    val_info = pd.read_csv(str(BASE / args.val_info), sep="\t")
    rng = np.random.default_rng(args.seed)
    read_ids = val_info["read_id"].tolist()
    if len(read_ids) > args.n_reads:
        read_ids = [read_ids[i] for i in rng.choice(len(read_ids), args.n_reads, replace=False)]

    print(f"Loading XY cache…")
    with open(str(BASE / args.xy_cache), "rb") as fh:
        xy_data = pickle.load(fh)
    xy_data = xy_data[xy_data["read_id"].isin(set(read_ids))].copy()
    actual_ids = list(xy_data["read_id"].unique())
    print(f"  {len(actual_ids):,} reads loaded")

    with open(str(BASE / args.config)) as f:
        config = yaml.safe_load(f)

    print("Loading model…")
    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(str(BASE / args.model),
                                        custom_objects=CUSTOM_OBJECTS)
    max_length = model.input_shape[1]

    print(f"Running inference on {len(actual_ids):,} reads…")
    preds = predict_reads(model, xy_data, actual_ids, max_length,
                          config["preprocessing"])
    tf.keras.backend.clear_session()
    print(f"  {len(preds):,} windows predicted")
    print(f"  Predicted class dist: {dict(preds['predicted_class'].value_counts().sort_index())}")

    # ── Apply pairing ─────────────────────────────────────────────────────────
    print(f"\nApplying fork pairing (min_fork_prob={args.min_fork_prob}, max_gap={args.max_gap} bp)…")
    baseline_events, paired_events = pair_forks_around_origins(
        preds,
        threshold=args.threshold,
        min_fork_prob=args.min_fork_prob,
        max_gap=args.max_gap,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_read_ids = set(actual_ids)
    # Filter GT report to val reads so train-set annotations don't inflate FN
    for df_name, df in [("gt_left", gt_left), ("gt_right", gt_right), ("gt_ori", gt_ori)]:
        n_before = len(df)
        df_filt = df[df["read_id"].isin(val_read_ids)]
        print(f"  GT {df_name}: {n_before} → {len(df_filt)} (restricted to val reads)")
    gt_left_val  = gt_left[gt_left["read_id"].isin(val_read_ids)]
    gt_right_val = gt_right[gt_right["read_id"].isin(val_read_ids)]
    gt_ori_val   = gt_ori[gt_ori["read_id"].isin(val_read_ids)]
    print(f"GT (val only): {len(gt_left_val)} left-forks | {len(gt_right_val)} right-forks | {len(gt_ori_val)} origins")

    print(f"\nEvaluating @ IoU ≥ {args.iou_thr}…")
    baseline_results = eval_events(baseline_events, gt_left_val, gt_right_val, gt_ori_val, args.iou_thr)
    paired_results   = eval_events(paired_events,   gt_left_val, gt_right_val, gt_ori_val, args.iou_thr)

    print_results(f"Baseline (threshold={args.threshold})", baseline_results)
    print_results(f"Fork-paired (min_fork_prob={args.min_fork_prob})", paired_results)

    # ── Event count summary ───────────────────────────────────────────────────
    print(f"\nEvent counts:")
    for cls in CLASS_NAMES.values():
        n_bl = len(baseline_events[baseline_events["event_type"] == cls]) if len(baseline_events) else 0
        n_pa = len(paired_events[paired_events["event_type"] == cls])   if len(paired_events)   else 0
        print(f"  {cls:<14}  baseline={n_bl:,}  paired={n_pa:,}  (+{n_pa - n_bl})")

    # ── Save TSV ──────────────────────────────────────────────────────────────
    rows = []
    for method, results in [("baseline", baseline_results), ("fork_paired", paired_results)]:
        for cls, r in results.items():
            rows.append({
                "method": method,
                "class": cls,
                "recall": r["recall"],
                "precision": r["precision"],
                "f1": r["f1"],
                "TP": r["true_positives"],
                "FP": r["false_positives"],
                "FN": r["false_negatives"],
                "n_predictions": r["num_predictions"],
                "n_gt": r["num_ground_truth"],
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "pairing_summary.tsv", sep="\t", index=False)
    print(f"\nSaved: {out / 'pairing_summary.tsv'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(baseline_results, paired_results, args.iou_thr,
                    out / "pairing_iou_comparison.png")

    print(f"\nAll outputs: {out}")


if __name__ == "__main__":
    main()
