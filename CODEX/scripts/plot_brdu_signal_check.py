#!/usr/bin/env python
"""Quick visual check for mean_brdu_signal on predicted ORI events.

Plots a few reads showing:
  - Raw BrdU signal trace
  - Predicted event spans (LF/RF/ORI) shaded
  - mean_brdu_signal annotated as a dashed horizontal line + label on each event

This is a sanity check to verify mean_brdu_signal = mean of raw signal
across the ORI windows (the valley) — not the model's ORI class probability.

Usage (from /replication-analyzer/):
  python CODEX/scripts/plot_brdu_signal_check.py \\
      --config   CODEX/configs/forte_v4.4.yaml \\
      --segments CODEX/results/forte_v4.4/reannotation_all/reannotated_segments.tsv \\
      --output   CODEX/results/forte_v4.4/brdu_signal_check/
"""
from __future__ import annotations

import argparse
import pickle
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

from replication_analyzer_codex.evaluation import windows_to_events
from replication_analyzer_codex.constants import CLASS_NAME_TO_ID

PROB_THR = 0.40
MAX_GAP  = 5000

COL = {
    "left_fork":  "#1f77b4",
    "right_fork": "#d62728",
    "origin":     "#2ca02c",
}


def load_xy_cache(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def pick_reads_with_ori(events_ori, n=6):
    """Pick reads that have at least one predicted ORI event."""
    reads = events_ori["read_id"].value_counts()
    return reads.index[:n].tolist()


def load_bed(path):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def plot_read(read_id, xy_data, segments, events_lf, events_rf, events_ori,
              gt_lf, gt_rf, gt_ori, out_path):
    """Plot signal + predicted events + mean_brdu_signal annotations."""
    # Raw signal for this read
    xy = xy_data[xy_data["read_id"] == read_id].sort_values("start")
    if xy.empty:
        print(f"  No XY data for {read_id[:16]}, skipping")
        return

    seg = segments[segments["read_id"] == read_id].sort_values("start")
    if seg.empty:
        print(f"  No segments for {read_id[:16]}, skipping")
        return

    # Join signal onto segments for mean_brdu_signal computation
    seg = seg.merge(xy[["start", "signal"]], on="start", how="left")

    # Recompute events with signal attached
    def get_events(class_name):
        sub = seg.copy()
        # windows_to_events expects a 'signal' column
        return windows_to_events(
            predictions=sub,
            class_id=CLASS_NAME_TO_ID[class_name],
            prob_threshold=PROB_THR,
            min_windows=1,
            max_gap=MAX_GAP,
        )

    ev_lf  = get_events("left_fork")
    ev_rf  = get_events("right_fork")
    ev_ori = get_events("origin")

    fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 0.6, 1.2]})

    # ── Top: BrdU signal + predicted event spans + mean_brdu_signal annotations ─
    ax = axes[0]
    x = xy["start"].tolist() + [xy["end"].iloc[-1]]
    y = xy["signal"].tolist() + [xy["signal"].iloc[-1]]
    ax.step(x, y, where="post", color="black", linewidth=1.0, zorder=3)
    ax.fill_between(x, y, step="post", alpha=0.10, color="gray", zorder=2)

    def shade_events(ev_df, class_name):
        col = COL[class_name]
        for row in ev_df.itertuples(index=False):
            ax.axvspan(row.start, row.end, color=col, alpha=0.20)
            if class_name == "origin":
                val = getattr(row, "mean_brdu_signal", None)
                if val is not None and not np.isnan(float(val)):
                    mid = (row.start + row.end) / 2
                    ax.hlines(float(val), row.start, row.end,
                              colors=col, linewidths=2.0, linestyles="--", zorder=4)
                    ax.text(mid, float(val) + 0.03, f"μ={float(val):.3f}",
                            ha="center", va="bottom", fontsize=7,
                            color=col, fontweight="bold")
            else:
                val = getattr(row, "brdu_slope", None)
                if val is not None and not np.isnan(float(val)):
                    mid = (row.start + row.end) / 2
                    slope_per_kb = float(val) * 1000
                    ax.text(mid, 0.92, f"slope={slope_per_kb:.4f}/kb",
                            ha="center", va="top", fontsize=7,
                            color=col, fontweight="bold",
                            transform=ax.get_xaxis_transform())

    shade_events(ev_lf,  "left_fork")
    shade_events(ev_rf,  "right_fork")
    shade_events(ev_ori, "origin")

    ax.set_ylabel("BrdU signal", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    pred_patches = [mpatches.Patch(color=COL[k], alpha=0.5,
                                   label=f"Pred {k.replace('_',' ').title()}")
                    for k in COL]
    ax.legend(handles=pred_patches, fontsize=7, loc="upper right")

    # ── Middle: GT annotation spans ────────────────────────────────────────────
    ax_gt = axes[1]
    ax_gt.set_ylabel("GT", fontsize=9)
    ax_gt.set_yticks([])
    ax_gt.set_ylim(0, 1)

    def add_gt_spans(bed_df, class_name):
        col = COL[class_name]
        sub = bed_df[bed_df["read_id"] == read_id]
        used = False
        for row in sub.itertuples(index=False):
            ax_gt.axvspan(int(row.start), int(row.end), color=col, alpha=0.55,
                          label=f"GT {class_name.replace('_',' ').title()}" if not used else "_")
            used = True

    add_gt_spans(gt_lf,  "left_fork")
    add_gt_spans(gt_rf,  "right_fork")
    add_gt_spans(gt_ori, "origin")

    gt_patches = [mpatches.Patch(color=COL[k], alpha=0.55,
                                 label=f"GT {k.replace('_',' ').title()}")
                  for k in COL]
    ax_gt.legend(handles=gt_patches, fontsize=7, loc="upper right")

    # ── Bottom: class probabilities ────────────────────────────────────────────
    ax2 = axes[2]
    pos = ((seg["start"] + seg["end"]) / 2).values
    ax2.fill_between(pos, seg["prob_origin"].values, color=COL["origin"], alpha=0.7, label="P(ORI)")
    ax2.fill_between(pos, seg["prob_left_fork"].values, color=COL["left_fork"], alpha=0.5, label="P(LF)")
    ax2.fill_between(pos, seg["prob_right_fork"].values, color=COL["right_fork"], alpha=0.5, label="P(RF)")
    ax2.axhline(PROB_THR, color="black", linewidth=0.8, linestyle="--", alpha=0.6,
                label=f"thr={PROB_THR}")
    ax2.set_ylabel("Probability", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Genomic position (bp)", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right", ncol=4)

    # ── Summary in title ───────────────────────────────────────────────────────
    ori_means = [row.mean_brdu_signal for row in ev_ori.itertuples()
                 if hasattr(row, "mean_brdu_signal") and not np.isnan(float(row.mean_brdu_signal))]
    lf_slopes = [float(row.brdu_slope) * 1000 for row in ev_lf.itertuples()
                 if hasattr(row, "brdu_slope") and not np.isnan(float(row.brdu_slope))]
    rf_slopes = [float(row.brdu_slope) * 1000 for row in ev_rf.itertuples()
                 if hasattr(row, "brdu_slope") and not np.isnan(float(row.brdu_slope))]

    def fmt_mean(vals, label):
        if not vals: return f"{label}: —"
        return f"{label} μ-BrdU={np.mean(vals):.3f} (n={len(vals)})"

    def fmt_slope(vals, label):
        if not vals: return f"{label}: —"
        return f"{label} slope={np.mean(vals):.4f}/kb (n={len(vals)})"

    read_len_kb = (xy["end"].max() - xy["start"].min()) / 1000
    fig.suptitle(
        f"{read_id[:20]}…  ({read_len_kb:.0f} kb)\n"
        f"{fmt_mean(ori_means,'ORI')}   {fmt_slope(lf_slopes,'LF')}   {fmt_slope(rf_slopes,'RF')}\n"
        f"(ORI: dashed line = mean BrdU signal; Forks: slope in BrdU/kb)",
        fontsize=8, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--segments", required=True, help="reannotated_segments.tsv")
    parser.add_argument("--output",   required=True)
    parser.add_argument("--n-reads",  type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    cache_path = cfg["data"].get("xy_cache_path")
    print(f"Loading XY cache: {cache_path}")
    xy_data = load_xy_cache(cache_path)

    print(f"Loading segments: {args.segments}")
    segments = pd.read_csv(args.segments, sep="\t")

    # Get events to pick representative reads
    events_ori = windows_to_events(segments, CLASS_NAME_TO_ID["origin"],  PROB_THR, 1, MAX_GAP)
    events_lf  = windows_to_events(segments, CLASS_NAME_TO_ID["left_fork"],  PROB_THR, 1, MAX_GAP)
    events_rf  = windows_to_events(segments, CLASS_NAME_TO_ID["right_fork"], PROB_THR, 1, MAX_GAP)

    read_ids = pick_reads_with_ori(events_ori, n=args.n_reads)
    print(f"Plotting {len(read_ids)} reads with ORI events...")

    print("Loading GT annotations...")
    gt_lf  = load_bed(cfg["data"].get("left_forks_bed", ""))
    gt_rf  = load_bed(cfg["data"].get("right_forks_bed", ""))
    gt_ori = load_bed(cfg["data"].get("ori_annotations_bed", ""))
    print(f"  GT LF: {len(gt_lf):,}  RF: {len(gt_rf):,}  ORI: {len(gt_ori):,}")

    for i, rid in enumerate(read_ids):
        out_path = out_dir / f"brdu_check_{i+1:02d}_{rid[:12]}.png"
        print(f"  [{i+1}/{len(read_ids)}] {rid[:20]}...")
        plot_read(rid, xy_data, segments, events_lf, events_rf, events_ori,
                  gt_lf, gt_rf, gt_ori, out_path)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
