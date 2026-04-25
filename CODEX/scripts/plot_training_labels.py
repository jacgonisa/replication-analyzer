#!/usr/bin/env python
"""Plot final training labels (LF / RF / ORI) on example reads to verify cleaning.

Human-annotated intervals are drawn solid; AI pseudo-label intervals are drawn
transparent so you can visually distinguish the two sources.

Picks reads that have all three annotation types so we can check:
  - ORIs are flanked by forks
  - No overlaps between classes
  - Label boundaries look biologically sensible

Usage:
  python CODEX/scripts/plot_training_labels.py \
      --config  CODEX/configs/forte_v5.0.yaml \
      --output  CODEX/results/forte_v5.0/label_check/
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).resolve().parents[2]

COL = {
    "left_fork":  "#3a86ff",
    "right_fork": "#e63946",
    "origin":     "#2dc653",
}
LANE_KEYS   = ["left_fork", "right_fork", "origin"]
LANE_LABELS = ["Left Fork", "Right Fork", "Origin"]

ALPHA_HUMAN  = 0.85   # solid — human-annotated
ALPHA_PSEUDO = 0.25   # transparent — AI pseudo-label


def load_bed4(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def build_human_sets(human_beds):
    """Return dict of key → set of (read_id, start, end) for fast overlap lookup."""
    human_sets = {}
    for key, df in human_beds.items():
        by_read = {}
        for row in df.itertuples(index=False):
            by_read.setdefault(row.read_id, []).append((int(row.start), int(row.end)))
        human_sets[key] = by_read
    return human_sets


def is_human(read_id, start, end, human_by_read):
    """True if interval overlaps any human-annotated interval on this read."""
    for hs, he in human_by_read.get(read_id, []):
        if start < he and end > hs:   # any overlap
            return True
    return False


def load_xy_cache(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def pick_reads(lf, rf, ori, n=12, mode="all3"):
    """Pick reads by mode:
      'all3'   — reads that have all three annotation types (best for checking flanking)
      'ori'    — reads with most ORI events
      'dense'  — reads with most total annotations
    """
    if mode == "all3":
        lf_reads  = set(lf["read_id"])
        rf_reads  = set(rf["read_id"])
        ori_reads = set(ori["read_id"])
        candidates = list(lf_reads & rf_reads & ori_reads)
        # Sort by number of ORI events (most interesting first)
        ori_counts = ori[ori["read_id"].isin(candidates)].groupby("read_id").size()
        candidates_sorted = ori_counts.sort_values(ascending=False).index.tolist()
        return candidates_sorted[:n]
    elif mode == "ori":
        return ori.groupby("read_id").size().sort_values(ascending=False).index[:n].tolist()
    else:
        total = (lf.groupby("read_id").size().add(
                 rf.groupby("read_id").size(), fill_value=0).add(
                 ori.groupby("read_id").size(), fill_value=0))
        return total.sort_values(ascending=False).index[:n].tolist()


def plot_read(read_id, xy_data, beds, human_sets, out_path):
    xy = xy_data[xy_data["read_id"] == read_id].sort_values("start")
    if xy.empty:
        print(f"  No XY data for {read_id[:16]}, skipping")
        return

    fig, axes = plt.subplots(2, 1, figsize=(20, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.8]})
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("white")

    # ── Panel 1: BrdU signal ──────────────────────────────────────────────────
    ax_sig = axes[0]
    x = xy["start"].tolist() + [xy["end"].iloc[-1]]
    y = xy["signal"].tolist() + [xy["signal"].iloc[-1]]
    ax_sig.step(x, y, where="post", color="#222222", linewidth=1.0, zorder=4)
    ax_sig.fill_between(x, y, step="post", alpha=0.07, color="#888888", zorder=2)

    for key in LANE_KEYS:
        sub = beds[key][beds[key]["read_id"] == read_id]
        for row in sub.itertuples(index=False):
            human = is_human(read_id, int(row.start), int(row.end), human_sets[key])
            ax_sig.axvspan(int(row.start), int(row.end),
                           color=COL[key],
                           alpha=0.18 if human else 0.07, zorder=1)

    ax_sig.set_ylabel("BrdU signal", fontsize=10)
    ax_sig.set_ylim(-0.05, 1.12)
    ax_sig.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_sig.tick_params(labelsize=8)
    ax_sig.spines[["top", "right"]].set_visible(False)

    legend_handles = []
    for k in LANE_KEYS:
        label = k.replace("_", " ").title()
        legend_handles.append(mpatches.Patch(color=COL[k], alpha=0.75, label=f"{label} (human)"))
        legend_handles.append(mpatches.Patch(color=COL[k], alpha=0.25, label=f"{label} (pseudo)"))
    ax_sig.legend(handles=legend_handles, fontsize=7, loc="upper right",
                  framealpha=0.85, ncol=3)

    # ── Panel 2: Annotation lanes ─────────────────────────────────────────────
    ax_ann = axes[1]
    ax_ann.set_ylim(0, 3)
    ax_ann.set_yticks([0.5, 1.5, 2.5])
    ax_ann.set_yticklabels(LANE_LABELS, fontsize=9)
    ax_ann.yaxis.set_tick_params(length=0)
    ax_ann.spines[["top", "right", "bottom"]].set_visible(False)
    ax_ann.axhline(1, color="#cccccc", linewidth=0.6)
    ax_ann.axhline(2, color="#cccccc", linewidth=0.6)

    for lane_idx, key in enumerate(LANE_KEYS):
        sub = beds[key][beds[key]["read_id"] == read_id]
        pad = 0.08
        for row in sub.itertuples(index=False):
            human = is_human(read_id, int(row.start), int(row.end), human_sets[key])
            alpha = ALPHA_HUMAN if human else ALPHA_PSEUDO
            ax_ann.fill_betweenx(
                [lane_idx + pad, lane_idx + 1 - pad],
                int(row.start), int(row.end),
                color=COL[key], alpha=alpha, linewidth=0,
            )

    ax_ann.set_xlabel("Genomic position (bp)", fontsize=10)
    ax_ann.set_ylabel("Annotation", fontsize=10)
    ax_ann.tick_params(axis="x", labelsize=8)

    # ── Title ─────────────────────────────────────────────────────────────────
    n_lf  = (beds["left_fork"]["read_id"]  == read_id).sum()
    n_rf  = (beds["right_fork"]["read_id"] == read_id).sum()
    n_ori = (beds["origin"]["read_id"]     == read_id).sum()
    read_len_kb = (xy["end"].max() - xy["start"].min()) / 1000

    fig.suptitle(
        f"Read: {read_id[:28]}   ({read_len_kb:.0f} kb)\n"
        f"LF: {n_lf}   RF: {n_rf}   ORI: {n_ori}   "
        f"(solid = human-annotated, transparent = AI pseudo-label)",
        fontsize=9, fontweight="bold", color="#333333",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--n-reads", type=int, default=12)
    parser.add_argument("--mode",    choices=["all3", "ori", "dense"], default="all3",
                        help="all3=reads with LF+RF+ORI; ori=most ORI events; dense=most total annotations")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print("Loading XY cache...")
    xy_data = load_xy_cache(cfg["data"]["xy_cache_path"])
    print(f"  {len(xy_data):,} rows, {xy_data['read_id'].nunique():,} reads")

    print("Loading annotation BEDs (training labels)...")
    beds = {
        "left_fork":  load_bed4(cfg["data"]["left_forks_bed"]),
        "right_fork": load_bed4(cfg["data"]["right_forks_bed"]),
        "origin":     load_bed4(cfg["data"]["ori_annotations_bed"]),
    }
    for k, df in beds.items():
        print(f"  {k}: {len(df):,} intervals on {df['read_id'].nunique():,} reads")

    # Human-annotated source files — used to distinguish human vs pseudo
    DATA = Path(__file__).resolve().parents[2] / "data/case_study_jan2026/combined/annotations"
    print("\nLoading human-annotated source BEDs...")
    human_beds = {
        "left_fork":  load_bed4(DATA / "leftForks_ALL_combined.bed"),
        "right_fork": load_bed4(DATA / "rightForks_ALL_combined.bed"),
        "origin":     load_bed4(DATA / "ORIs_combined_cleaned.bed"),
    }
    for k, df in human_beds.items():
        print(f"  {k}: {len(df):,} human intervals")
    human_sets = build_human_sets(human_beds)

    print(f"\nPicking reads (mode={args.mode})...")
    read_ids = pick_reads(beds["left_fork"], beds["right_fork"], beds["origin"],
                          n=args.n_reads, mode=args.mode)
    print(f"  Selected {len(read_ids)} reads")

    for i, rid in enumerate(read_ids):
        out_path = out_dir / f"labels_{i+1:02d}_{rid[:12]}.png"
        print(f"  [{i+1}/{len(read_ids)}] {rid[:24]}...")
        plot_read(rid, xy_data, beds, human_sets, out_path)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
