#!/usr/bin/env python
"""Plot N manual annotations / Gb of data in each read-length bin.

Same aggregate approach as plot_readlen_vs_features_aggregate.py but uses
only Nerea's hand-annotated BED files — no model predictions.

3 panels: Left Forks, Right Forks, ORIs.

Y = total annotated events in bin / total Gb of sequencing data in bin.

Usage (from /replication-analyzer/):
  python CODEX/scripts/plot_readlen_vs_manual_annotations.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE       = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")
CODEX_ROOT = BASE / "CODEX"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(CODEX_ROOT))

XY_CACHE  = CODEX_ROOT / "results/cache/xy_data.pkl"
ANNOT_DIR = BASE / "data/case_study_jan2026/combined/annotations"

LF_BED  = ANNOT_DIR / "leftForks_ALL_combined.bed"
RF_BED  = ANNOT_DIR / "rightForks_ALL_combined.bed"
ORI_BED = ANNOT_DIR / "ORIs_combined_cleaned.bed"

OUT_DIR     = CODEX_ROOT / "results/readlen_analysis_density"
BIN_STEP_KB = 20
MIN_KB      = 30
MAX_KB      = 300


def make_bins(min_kb=30, max_kb=300, step=20):
    edges  = list(range(min_kb, max_kb + step, step)) + [int(1e6)]
    labels = [f"{e}–{e+step}" for e in range(min_kb, max_kb, step)] + [f">{max_kb}"]
    return edges, labels


def get_read_lengths(xy_data):
    g = xy_data.groupby("read_id")
    return (g["end"].max() - g["start"].min()).rename("read_length_bp")


def load_bed4(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def count_events_per_read(event_df, all_read_ids):
    counts = event_df.groupby("read_id").size()
    return counts.reindex(all_read_ids, fill_value=0)


def aggregate_per_bin(event_counts, read_lengths_bp, read_lengths_kb,
                      bin_edges, bin_labels):
    df = pd.DataFrame({
        "length_kb": read_lengths_kb,
        "length_bp": read_lengths_bp,
        "events":    event_counts.reindex(read_lengths_kb.index, fill_value=0),
    })
    full_edges  = [0] + bin_edges
    full_labels = ["<30"] + list(bin_labels)
    df["bin"] = pd.cut(df["length_kb"], bins=full_edges, labels=full_labels,
                       right=False, include_lowest=True)
    df = df[df["bin"].isin(bin_labels)]

    result = {}
    for lbl in bin_labels:
        grp = df[df["bin"] == lbl]
        n_reads      = len(grp)
        total_gb     = grp["length_bp"].sum() / 1e9
        total_events = int(grp["events"].sum())
        value = total_events / total_gb if total_gb > 0 else 0.0
        result[lbl] = (value, n_reads, total_gb, total_events)
    return result


def plot_aggregate(ax, agg_dict, bin_labels, color, title, ylabel="Events / Gb"):
    xs      = list(range(len(bin_labels)))
    ys      = [agg_dict[lbl][0] for lbl in bin_labels]
    total_gb= [agg_dict[lbl][2] for lbl in bin_labels]
    total_ev= [agg_dict[lbl][3] for lbl in bin_labels]

    ax.bar(xs, ys, color=color, alpha=0.75, edgecolor="black",
           linewidth=0.7, width=0.6)
    ax.plot(xs, ys, color=color, linewidth=1.5, marker="o",
            markersize=5, markerfacecolor="white", markeredgecolor=color,
            zorder=5)

    y_max = max(ys) if max(ys) > 0 else 1
    for xi, (yi, tg, te) in enumerate(zip(ys, total_gb, total_ev)):
        ax.text(xi, yi + 0.01 * y_max,
                f"{te:,} ev\n{tg:.2f} Gb",
                ha="center", va="bottom", fontsize=6.5, color="#333")

    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.7)
    ax.set_ylim(0, y_max * 1.25)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_edges, bin_labels = make_bins(MIN_KB, MAX_KB, BIN_STEP_KB)

    # ── Read lengths ──────────────────────────────────────────────────────────
    print("Loading XY cache for read lengths...")
    with open(XY_CACHE, "rb") as fh:
        xy_data = pickle.load(fh)
    read_lengths_bp = get_read_lengths(xy_data)
    read_lengths_kb = read_lengths_bp / 1000
    all_read_ids    = read_lengths_bp.index.tolist()
    del xy_data
    print(f"  {len(all_read_ids):,} reads  (all >30 kb + BrdU signal)")
    print(f"  Total data: {read_lengths_bp.sum()/1e9:.1f} Gb")

    # ── Manual annotations ────────────────────────────────────────────────────
    print("Loading manual annotations...")
    lf_bed  = load_bed4(LF_BED)
    rf_bed  = load_bed4(RF_BED)
    ori_bed = load_bed4(ORI_BED)
    print(f"  Manual LF:  {len(lf_bed):,}")
    print(f"  Manual RF:  {len(rf_bed):,}")
    print(f"  Manual ORI: {len(ori_bed):,}")

    lf_counts  = count_events_per_read(lf_bed,  all_read_ids)
    rf_counts  = count_events_per_read(rf_bed,  all_read_ids)
    ori_counts = count_events_per_read(ori_bed, all_read_ids)

    sources = [
        ("Manual Left Forks\n(Nerea annotations)",  lf_counts,  "#1f77b4"),
        ("Manual Right Forks\n(Nerea annotations)", rf_counts,  "#d62728"),
        ("Manual ORIs\n(Nerea annotations)",         ori_counts, "#2c3e50"),
    ]

    agg_results = {}
    for title, counts, _ in sources:
        agg_results[title] = aggregate_per_bin(
            counts, read_lengths_bp, read_lengths_kb, bin_edges, bin_labels)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(28, 7))
    fig.suptitle(
        "Manual replication annotations per Gb of sequencing data vs read length\n"
        f"(reads >30 kb with BrdU signal, n={len(all_read_ids):,};  {BIN_STEP_KB} kb bins)\n"
        "Y = total annotated events in bin / total Gb of data in bin  —  "
        "one absolute value per bin, no per-read averaging",
        fontsize=11, fontweight="bold",
    )

    for ax, (title, counts, color) in zip(axes, sources):
        plot_aggregate(ax, agg_results[title], bin_labels, color, title,
                       ylabel="Events / Gb of data")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = OUT_DIR / "readlen_vs_manual_annotations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ── Summary TSV ───────────────────────────────────────────────────────────
    rows = []
    for title, counts, _ in sources:
        agg = agg_results[title]
        for lbl_bin in bin_labels:
            val, nr, tg, te = agg[lbl_bin]
            rows.append({
                "source":          title.replace("\n", " "),
                "bin_kb":          lbl_bin,
                "n_reads_in_bin":  nr,
                "total_Gb_in_bin": round(tg, 4),
                "total_events":    te,
                "events_per_Gb":   round(val, 4),
            })
    tsv_path = OUT_DIR / "readlen_vs_manual_annotations.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved TSV: {tsv_path}")


if __name__ == "__main__":
    main()
