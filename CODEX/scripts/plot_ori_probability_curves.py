#!/usr/bin/env python
"""
Plot per-window ORI probability curves for reads with broad predicted ORIs.
Shows the flat plateau problem: model outputs a wide, undifferentiated probability
mass instead of sharp peaks at individual ORI locations.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_COLORS = {
    "left_fork":  "#2196F3",
    "right_fork": "#FF5722",
    "origin":     "#4CAF50",
    "background": "#9E9E9E",
}

PROB_COLORS = {
    "prob_left_fork":  "#2196F3",
    "prob_right_fork": "#FF5722",
    "prob_origin":     "#4CAF50",
    "prob_background": "#9E9E9E",
}


def load_human_oris(bed_path, read_id):
    if not bed_path or not Path(bed_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(bed_path, sep="\t", header=None,
                     names=["chr", "start", "end", "read_id"])
    return df[df["read_id"] == read_id]


def load_predicted_events(events_path, read_id):
    if not events_path or not Path(events_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(events_path, sep="\t")
    return df[df["read_id"] == read_id]


def plot_read(read_id, seg, human_oris, pred_events, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [1.5, 2, 1.5]},
                             sharex=True)

    pos = (seg["start"] + seg["end"]) / 2 / 1000  # kb

    # ── Panel 1: BrdU signal ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot(pos, seg["signal"], color="#333333", lw=0.6, alpha=0.8)
    ax.set_ylabel("BrdU signal", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{read_id[:8]}…  |  {len(seg)} windows  |  "
                 f"{(seg['end'].max() - seg['start'].min()) / 1000:.0f} kb",
                 fontsize=10)

    # shade human ORI regions
    for _, row in human_oris[human_oris["event_type"] == "origin"].iterrows() if "event_type" in human_oris.columns else human_oris.iterrows():
        s, e = row["start"] / 1000, row["end"] / 1000
        ax.axvspan(s, e, alpha=0.25, color=CLASS_COLORS["origin"], label="Human ORI")

    # ── Panel 2: per-window probability curves ────────────────────────────
    ax = axes[1]
    for col, color in PROB_COLORS.items():
        if col in seg.columns:
            label = col.replace("prob_", "").replace("_", " ")
            ax.plot(pos, seg[col], color=color, lw=1.2, alpha=0.85, label=label)

    # threshold line
    ax.axhline(0.4, color="black", lw=0.8, ls="--", alpha=0.5, label="threshold 0.4")

    # shade human ORI regions
    for _, row in human_oris.iterrows():
        s, e = row["start"] / 1000, row["end"] / 1000
        ax.axvspan(s, e, alpha=0.15, color=CLASS_COLORS["origin"])

    ax.set_ylabel("Window probability", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=8, ncol=5)

    # ── Panel 3: predicted vs human events (annotation bars) ─────────────
    ax = axes[2]
    ax.set_ylim(0, 4)
    ax.set_yticks([1, 3])
    ax.set_yticklabels(["Human GT", "Predicted"], fontsize=8)

    for _, row in human_oris.iterrows():
        s, e = row["start"] / 1000, row["end"] / 1000
        ax.barh(1, e - s, left=s, height=0.6, color=CLASS_COLORS["origin"],
                alpha=0.8)

    for _, row in pred_events[pred_events["event_type"] == "origin"].iterrows() if "event_type" in pred_events.columns else pred_events.iterrows():
        s, e = row["start"] / 1000, row["end"] / 1000
        label = f"p={row.get('mean_prob', 0):.2f}"
        ax.barh(3, e - s, left=s, height=0.6, color=CLASS_COLORS["origin"],
                alpha=0.8)
        ax.text((s + e) / 2, 3.4, label, ha="center", va="bottom",
                fontsize=7, color=CLASS_COLORS["origin"])

    ax.set_xlabel("Genomic position (kb)", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--human-ori-bed", required=True)
    parser.add_argument("--read-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading segments...")
    seg_df = pd.read_csv(args.segments, sep="\t")
    print("Loading events...")
    events_df = pd.read_csv(args.events, sep="\t")
    print("Loading human ORI BED...")
    ori_bed = pd.read_csv(args.human_ori_bed, sep="\t", header=None,
                          names=["chr", "start", "end", "read_id"])

    for read_id in args.read_ids:
        seg = seg_df[seg_df["read_id"] == read_id].sort_values("start")
        if len(seg) == 0:
            print(f"  WARNING: no segments for {read_id[:8]}")
            continue

        human_oris = ori_bed[ori_bed["read_id"] == read_id]
        pred_events = events_df[events_df["read_id"] == read_id]

        # add event_type column for uniform handling
        if "event_type" not in pred_events.columns and "class" in pred_events.columns:
            pred_events = pred_events.rename(columns={"class": "event_type"})

        out_path = out_dir / f"prob_curve_{read_id[:8]}.png"
        plot_read(read_id, seg, human_oris, pred_events, out_path)


if __name__ == "__main__":
    main()
