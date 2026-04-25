#!/usr/bin/env python
"""Plot N features / Gb of data in each read-length bin.

One absolute value per bin — no distribution, no per-read computation.
Y = total events in bin / total Gb of sequencing data in bin.

As requested by collaborator:
  "absolute value of N features per amount of data in the allocated window
   of lengths. Si hay 50 ORIs en la ventana de 20-30kb y hay 5 Gb de datos
   en esa ventana, 50/5."

Sources:
  - Predicted forks + ORIs: any reannotated_segments.tsv (--pred-tsv)
  - Manual ORIs:            Nerea ORIs_combined_cleaned.bed

Usage (from /replication-analyzer/):
  # v4.3 run3 (default)
  python CODEX/scripts/plot_readlen_vs_features_aggregate.py

  # v4.4 (once reannotation_all is done)
  python CODEX/scripts/plot_readlen_vs_features_aggregate.py \\
      --pred-tsv CODEX/results/forte_v4.4/reannotation_all/reannotated_segments.tsv \\
      --model-label "FORTE v4.4"
"""
from __future__ import annotations

import argparse
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

from replication_analyzer_codex.evaluation import windows_to_events

XY_CACHE      = CODEX_ROOT / "results/cache/xy_data.pkl"
PRED_V43      = CODEX_ROOT / "results/forte_v4.3/reannotation_run3/reannotated_segments.tsv"
NEREA_ORI_BED = BASE / "data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed"
OUT_DIR       = CODEX_ROOT / "results/readlen_analysis_density"

PRED_FORK_THR = 0.40
PRED_ORI_THR  = 0.40
BIN_STEP_KB   = 20
MIN_KB        = 30
MAX_KB        = 300


def make_bins(min_kb=30, max_kb=300, step=20):
    edges  = list(range(min_kb, max_kb + step, step)) + [int(1e6)]
    labels = [f"{e}–{e+step}" for e in range(min_kb, max_kb, step)] + [f">{max_kb}"]
    return edges, labels


def get_read_lengths(xy_data):
    g = xy_data.groupby("read_id")
    return (g["end"].max() - g["start"].min()).rename("read_length_bp")


def count_events_per_read(event_df, all_read_ids):
    counts = event_df.groupby("read_id").size()
    return counts.reindex(all_read_ids, fill_value=0)


def aggregate_per_bin(event_counts, read_lengths_bp, read_lengths_kb,
                      bin_edges, bin_labels):
    """
    For each bin: total_events / total_Gb_of_data_in_bin.
    Returns dict {label: (value, n_reads, total_gb, total_events)}.
    """
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-tsv",     default=str(PRED_V43),
                        help="reannotated_segments.tsv from any FORTE model")
    parser.add_argument("--model-label",  default="FORTE v4.3 run3",
                        help="Label shown in plot titles (e.g. 'FORTE v4.4')")
    parser.add_argument("--out-suffix",   default="",
                        help="Optional suffix for output filenames (e.g. '_v4.4')")
    args = parser.parse_args()

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

    # ── Model predictions ─────────────────────────────────────────────────────
    print(f"Loading predictions: {args.pred_tsv}")
    preds = pd.read_csv(args.pred_tsv, sep="\t")

    lf_events  = windows_to_events(preds, class_id=1, prob_threshold=PRED_FORK_THR,
                                   min_windows=1, max_gap=5000)
    rf_events  = windows_to_events(preds, class_id=2, prob_threshold=PRED_FORK_THR,
                                   min_windows=1, max_gap=5000)
    ori_events = windows_to_events(preds, class_id=3, prob_threshold=PRED_ORI_THR,
                                   min_windows=1, max_gap=5000)

    lf_counts  = count_events_per_read(lf_events,  all_read_ids)
    rf_counts  = count_events_per_read(rf_events,  all_read_ids)
    ori_counts = count_events_per_read(ori_events, all_read_ids)

    print(f"  LF: {len(lf_events):,}  RF: {len(rf_events):,}  ORI: {len(ori_events):,}")

    # ── Nerea manual ORIs ─────────────────────────────────────────────────────
    print("Loading Nerea ORIs...")
    nerea_ori = pd.read_csv(NEREA_ORI_BED, sep="\t", header=None,
                             usecols=[0, 1, 2, 3],
                             names=["chr", "start", "end", "read_id"])
    nerea_counts = count_events_per_read(nerea_ori, all_read_ids)
    print(f"  Nerea ORIs: {len(nerea_ori):,}")

    # ── Aggregate per bin ─────────────────────────────────────────────────────
    lbl = args.model_label
    sources = [
        (f"Predicted Left Forks\n({lbl})",  lf_counts,    "#3498db"),
        (f"Predicted Right Forks\n({lbl})", rf_counts,    "#e74c3c"),
        (f"Predicted ORIs\n({lbl})",         ori_counts,   "#2ecc71"),
        ("Manual ORIs\n(Nerea annotations)", nerea_counts, "#2c3e50"),
    ]

    agg_results = {}
    for title, counts, color in sources:
        agg_results[title] = aggregate_per_bin(
            counts, read_lengths_bp, read_lengths_kb, bin_edges, bin_labels)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle(
        f"Replication features per Gb of sequencing data vs read length\n"
        f"(reads >30 kb with BrdU signal, n={len(all_read_ids):,};  "
        f"prob thr={PRED_FORK_THR};  {BIN_STEP_KB} kb bins)\n"
        "Y = total events in bin / total Gb of data in bin  —  "
        "one absolute value per bin, no per-read averaging",
        fontsize=11, fontweight="bold",
    )

    for ax, (title, counts, color) in zip(axes.flat, sources):
        plot_aggregate(ax, agg_results[title], bin_labels, color, title,
                       ylabel="Events / Gb of data")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    suffix = args.out_suffix
    out_path = OUT_DIR / f"readlen_vs_features_aggregate{suffix}.png"
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
    tsv_path = OUT_DIR / f"readlen_vs_features_aggregate{suffix}.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved TSV: {tsv_path}")


if __name__ == "__main__":
    main()
