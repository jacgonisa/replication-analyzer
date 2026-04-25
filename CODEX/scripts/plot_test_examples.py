#!/usr/bin/env python
"""Plot good and bad prediction examples from the test set evaluation.

Selects reads automatically from evaluation_test/ outputs:
  - Good: reads where ALL predicted events have high IoU (≥0.6) with GT annotations
  - Bad:  reads with false positives (is_match=0) or very low IoU matched events

Each panel shows:
  - BrdU signal (raw XY data)
  - GT annotation spans (LF=blue, RF=red, ORI=green)
  - Predicted probability tracks (one line per class)
  - Predicted event spans (shaded, from windows_to_events)

Usage (from /replication-analyzer/):
  /home/jg2070/miniforge3/envs/ONT/bin/python -u \\
      CODEX/scripts/plot_test_examples.py \\
      --config CODEX/configs/forte_v4.4.yaml \\
      --predictions CODEX/results/forte_v4.4/evaluation_test/predictions.tsv \\
      --events-lf   CODEX/results/forte_v4.4/evaluation_test/events_left_fork.tsv \\
      --events-rf   CODEX/results/forte_v4.4/evaluation_test/events_right_fork.tsv \\
      --events-ori  CODEX/results/forte_v4.4/evaluation_test/events_origin.tsv \\
      --output-dir  CODEX/results/forte_v4.4/example_plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(CODEX_ROOT))

# ── colours ──────────────────────────────────────────────────────────────────
COL_LF_GT   = "#1f77b4"   # blue  — GT left fork
COL_RF_GT   = "#d62728"   # red   — GT right fork
COL_ORI_GT  = "#2ca02c"   # green — GT origin

COL_LF_PRED  = "#74b9ff"  # light blue  — predicted LF prob
COL_RF_PRED  = "#ff7675"  # light red   — predicted RF prob
COL_ORI_PRED = "#55efc4"  # light green — predicted ORI prob
COL_BG_PRED  = "#b2bec3"  # grey        — predicted BG prob

PROB_THR = 0.40


# ── helpers ───────────────────────────────────────────────────────────────────

def load_xy(base_dir, run_dirs, read_id):
    rows = []
    for run_dir in run_dirs:
        f = Path(base_dir) / run_dir / f"plot_data_{read_id}.txt"
        if f.exists():
            df = pd.read_csv(f, sep="\t", header=None,
                             names=["chr", "start", "end", "signal"])
            rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True).sort_values("start").reset_index(drop=True)


def load_bed(path):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def add_annotation_spans(ax, bed_df, read_id, color, label, alpha=0.30):
    sub = bed_df[bed_df["read_id"] == read_id]
    used = False
    for row in sub.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), color=color, alpha=alpha,
                   label=label if not used else "_")
        used = True


def add_predicted_spans(ax, events_df, read_id, color, alpha=0.55):
    sub = events_df[events_df["read_id"] == read_id]
    for row in sub.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), color=color, alpha=alpha,
                   linewidth=1.5, edgecolor=color)


def select_reads(events_lf, events_rf, events_ori, n_good=5, n_bad=5):
    """Return (good_reads, bad_reads) based on event matching quality."""
    all_events = pd.concat([events_lf, events_rf, events_ori], ignore_index=True)

    # Per-read stats
    def read_stats(df):
        if df.empty:
            return pd.DataFrame(columns=["read_id", "n_events", "n_match", "mean_iou"])
        g = df.groupby("read_id").agg(
            n_events=("is_match", "count"),
            n_match=("is_match", "sum"),
            mean_iou=("best_iou", "mean"),
        ).reset_index()
        return g

    stats_lf  = read_stats(events_lf)
    stats_rf  = read_stats(events_rf)
    stats_ori = read_stats(events_ori)

    # Reads that have at least one predicted event from each class
    has_lf  = set(events_lf["read_id"])
    has_rf  = set(events_rf["read_id"])
    has_ori = set(events_ori["read_id"])
    has_all = has_lf & has_rf & has_ori

    # Good: reads with all events matched at high IoU
    good_candidates = []
    for rid in has_all:
        lf_iou  = events_lf[events_lf["read_id"] == rid]["best_iou"].mean()
        rf_iou  = events_rf[events_rf["read_id"] == rid]["best_iou"].mean()
        ori_iou = events_ori[events_ori["read_id"] == rid]["best_iou"].mean()
        min_iou = min(lf_iou, rf_iou, ori_iou)
        lf_fp   = (events_lf[events_lf["read_id"] == rid]["is_match"] == 0).sum()
        rf_fp   = (events_rf[events_rf["read_id"] == rid]["is_match"] == 0).sum()
        ori_fp  = (events_ori[events_ori["read_id"] == rid]["is_match"] == 0).sum()
        total_fp = lf_fp + rf_fp + ori_fp
        good_candidates.append((rid, min_iou, total_fp))

    good_candidates.sort(key=lambda x: (-x[1], x[2]))  # max min_IoU, min FP
    good_reads = [r[0] for r in good_candidates[:n_good]]

    # Bad: reads with FPs or low IoU — pick diverse failure modes
    bad_candidates = []
    for rid in (has_lf | has_rf | has_ori):
        sub = all_events[all_events["read_id"] == rid]
        n_fp = (sub["is_match"] == 0).sum()
        mean_iou = sub["best_iou"].mean()
        # Avoid reads already selected as good
        if rid in good_reads:
            continue
        bad_candidates.append((rid, n_fp, mean_iou))

    # Sort by n_fp desc, then by mean_iou asc (worst first)
    bad_candidates.sort(key=lambda x: (-x[1], x[2]))
    bad_reads = [r[0] for r in bad_candidates[:n_bad]]

    return good_reads, bad_reads


def plot_read(read_id, xy_df, preds_df,
              lf_gt, rf_gt, ori_gt,
              lf_events, rf_events, ori_events,
              title_suffix, out_path):
    """Plot one read: signal + GT spans + prob tracks + predicted spans."""
    read_preds = preds_df[preds_df["read_id"] == read_id].sort_values("start")
    if read_preds.empty or xy_df is None:
        print(f"  Skipping {read_id}: no data")
        return

    positions = ((read_preds["start"] + read_preds["end"]) / 2).values

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5]})

    # ── Top panel: BrdU signal + GT annotation spans ──────────────────────────
    ax_sig = axes[0]
    # Rectangular step signal — black outline with light grey fill (consistent style)
    x = xy_df["start"].tolist() + [xy_df["end"].iloc[-1]]
    y = xy_df["signal"].tolist() + [xy_df["signal"].iloc[-1]]
    ax_sig.step(x, y, where="post", color="black", linewidth=1.2, zorder=3)
    ax_sig.fill_between(x, y, step="post", alpha=0.12, color="gray", zorder=2)

    add_annotation_spans(ax_sig, lf_gt,  read_id, COL_LF_GT,  "GT Left Fork")
    add_annotation_spans(ax_sig, rf_gt,  read_id, COL_RF_GT,  "GT Right Fork")
    add_annotation_spans(ax_sig, ori_gt, read_id, COL_ORI_GT, "GT Origin")

    # Predicted spans on top of signal (slightly more opaque, hatched)
    sub_lf  = lf_events[lf_events["read_id"]   == read_id]
    sub_rf  = rf_events[rf_events["read_id"]   == read_id]
    sub_ori = ori_events[ori_events["read_id"] == read_id]
    for row in sub_lf.itertuples():
        ax_sig.axvspan(row.start, row.end, color=COL_LF_GT,
                       alpha=0.18, hatch="//", edgecolor=COL_LF_GT, linewidth=0)
    for row in sub_rf.itertuples():
        ax_sig.axvspan(row.start, row.end, color=COL_RF_GT,
                       alpha=0.18, hatch="\\\\", edgecolor=COL_RF_GT, linewidth=0)
    for row in sub_ori.itertuples():
        ax_sig.axvspan(row.start, row.end, color=COL_ORI_GT,
                       alpha=0.18, hatch="xx", edgecolor=COL_ORI_GT, linewidth=0)

    ax_sig.set_ylabel("BrdU signal", fontsize=9)
    ax_sig.set_ylim(-0.05, 1.05)

    # Legend: GT solid + pred hatched
    patches = [
        mpatches.Patch(color=COL_LF_GT,  alpha=0.35, label="GT Left Fork"),
        mpatches.Patch(color=COL_RF_GT,  alpha=0.35, label="GT Right Fork"),
        mpatches.Patch(color=COL_ORI_GT, alpha=0.35, label="GT Origin"),
        mpatches.Patch(facecolor="white", edgecolor=COL_LF_GT,  hatch="//",  label="Pred LF"),
        mpatches.Patch(facecolor="white", edgecolor=COL_RF_GT,  hatch="\\\\", label="Pred RF"),
        mpatches.Patch(facecolor="white", edgecolor=COL_ORI_GT, hatch="xx",  label="Pred ORI"),
    ]
    ax_sig.legend(handles=patches, fontsize=7, loc="upper right", ncol=3)

    # ── Bottom panel: predicted probability tracks ────────────────────────────
    ax_prob = axes[1]
    ax_prob.fill_between(positions, read_preds["prob_left_fork"].values,
                         color=COL_LF_PRED, alpha=0.7, label="P(LF)")
    ax_prob.fill_between(positions, read_preds["prob_right_fork"].values,
                         color=COL_RF_PRED, alpha=0.7, label="P(RF)")
    ax_prob.fill_between(positions, read_preds["prob_origin"].values,
                         color=COL_ORI_PRED, alpha=0.85, label="P(ORI)")
    ax_prob.axhline(PROB_THR, color="black", linewidth=0.8, linestyle="--",
                    alpha=0.6, label=f"thr={PROB_THR}")

    ax_prob.set_ylabel("Probability", fontsize=9)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_xlabel("Genomic position (bp)", fontsize=9)
    ax_prob.legend(fontsize=7, loc="upper right", ncol=4)

    # ── Title ─────────────────────────────────────────────────────────────────
    # Compute per-event IoUs for title annotation
    def _iou_str(df, rid):
        sub = df[df["read_id"] == rid]
        if sub.empty:
            return "—"
        ious = sub["best_iou"].values
        matched = sub["is_match"].values
        parts = [f"{v:.2f}{'✓' if m else '✗'}" for v, m in zip(ious, matched)]
        return ", ".join(parts)

    lf_str  = _iou_str(lf_events,  read_id)
    rf_str  = _iou_str(rf_events,  read_id)
    ori_str = _iou_str(ori_events, read_id)

    read_len_kb = (xy_df["end"].max() - xy_df["start"].min()) / 1000
    fig.suptitle(
        f"{title_suffix}  |  {read_id[:16]}…  ({read_len_kb:.0f} kb)\n"
        f"IoU — LF: [{lf_str}]   RF: [{rf_str}]   ORI: [{ori_str}]",
        fontsize=9, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--events-lf",   required=True)
    parser.add_argument("--events-rf",   required=True)
    parser.add_argument("--events-ori",  required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--n-good",      type=int, default=5)
    parser.add_argument("--n-bad",       type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_dir  = cfg["data"]["base_dir"]
    run_dirs  = cfg["data"]["run_dirs"]
    lf_bed    = cfg["data"].get("left_forks_bed", "")
    rf_bed    = cfg["data"].get("right_forks_bed", "")
    ori_bed   = cfg["data"].get("ori_annotations_bed", "")

    print("Loading predictions and events...")
    preds      = pd.read_csv(args.predictions, sep="\t")
    events_lf  = pd.read_csv(args.events_lf,  sep="\t")
    events_rf  = pd.read_csv(args.events_rf,  sep="\t")
    events_ori = pd.read_csv(args.events_ori, sep="\t")

    print("Loading GT annotations...")
    lf_gt  = load_bed(lf_bed)
    rf_gt  = load_bed(rf_bed)
    ori_gt = load_bed(ori_bed)

    print(f"  GT LF:  {len(lf_gt):,} regions")
    print(f"  GT RF:  {len(rf_gt):,} regions")
    print(f"  GT ORI: {len(ori_gt):,} regions")

    print("Selecting reads...")
    good_reads, bad_reads = select_reads(
        events_lf, events_rf, events_ori,
        n_good=args.n_good, n_bad=args.n_bad,
    )
    print(f"  Good reads: {len(good_reads)}")
    print(f"  Bad reads:  {len(bad_reads)}")

    def plot_set(read_ids, label):
        for i, rid in enumerate(read_ids):
            print(f"  Plotting {label} [{i+1}/{len(read_ids)}]: {rid[:20]}...")
            xy = load_xy(base_dir, run_dirs, rid)
            if xy is None:
                print(f"    XY data not found, skipping.")
                continue
            out_path = out_dir / f"{label}_{i+1:02d}_{rid[:12]}.png"
            plot_read(
                read_id=rid,
                xy_df=xy,
                preds_df=preds,
                lf_gt=lf_gt, rf_gt=rf_gt, ori_gt=ori_gt,
                lf_events=events_lf,
                rf_events=events_rf,
                ori_events=events_ori,
                title_suffix=f"{'✅ GOOD' if label == 'good' else '❌ BAD'} prediction #{i+1}",
                out_path=out_path,
            )

    print("\nPlotting good reads...")
    plot_set(good_reads, "good")

    print("\nPlotting bad reads...")
    plot_set(bad_reads, "bad")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
