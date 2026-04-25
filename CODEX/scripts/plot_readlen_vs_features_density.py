#!/usr/bin/env python
"""Plot N events per Mb of data vs read length — ForkML-style boxplots.

Y-axis is normalised by read length (features / Mb of read) rather than
raw count, so panels across read-length bins are directly comparable.

Sources used (as requested by collaborator):
  - Predicted forks (LF + RF): FORTE v4.3 run3 predicted events
  - Predicted ORIs:            FORTE v4.3 run3 predicted events
  - Manual ORIs:               Nerea ORIs_combined_cleaned.bed

Filtered reads: >30 kb with BrdU signal (all reads in the XY cache are
already filtered to minLen30000_nascent40, so no extra filtering needed).

Usage (from /replication-analyzer/):
  /home/jg2070/miniforge3/envs/ONT/bin/python -u \\
      CODEX/scripts/plot_readlen_vs_features_density.py
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


# ── helpers ───────────────────────────────────────────────────────────────────

def make_bins(min_kb=30, max_kb=300, step=10):
    edges  = list(range(min_kb, max_kb + step, step)) + [int(1e6)]
    labels = [f"{e}–{e+step}" for e in range(min_kb, max_kb, step)] + [f">{max_kb}"]
    return edges, labels


def get_read_lengths(xy_data):
    g = xy_data.groupby("read_id")
    return (g["end"].max() - g["start"].min()).rename("read_length_bp")


def count_events_per_read(event_df, all_read_ids):
    counts = event_df.groupby("read_id").size()
    return counts.reindex(all_read_ids, fill_value=0)


def density_per_read(counts, read_lengths_bp):
    """N events / Mb of read. Returns Series aligned to counts.index."""
    length_mb = read_lengths_bp.reindex(counts.index) / 1e6
    return counts / length_mb.replace(0, np.nan)


def forkml_boxplot_density(ax, density_series, read_lengths_kb,
                            bin_edges, bin_labels, color, title):
    """ForkML-style boxplot with y = events / Mb of read."""
    df = pd.DataFrame({
        "length_kb": read_lengths_kb.reindex(density_series.index),
        "density":   density_series,
    })
    full_edges  = [0] + bin_edges
    full_labels = ["<30"] + list(bin_labels)
    df["bin"] = pd.cut(
        df["length_kb"],
        bins=full_edges,
        labels=full_labels,
        right=False, include_lowest=True,
    )
    df = df[df["bin"].isin(bin_labels)]

    xs, data_all, means_det, means_all, n_labels = [], [], [], [], []
    for i, lbl in enumerate(bin_labels):
        grp = df[df["bin"] == lbl]["density"].dropna()
        n_total = len(df[df["bin"] == lbl])
        grp_det = grp[grp > 0]
        n_det   = len(grp_det)
        xs.append(i)
        data_all.append(grp_det.values if n_det > 0 else np.array([0.0]))
        means_det.append(grp_det.mean() if n_det > 0 else 0.0)
        means_all.append(grp.mean() if len(grp) > 0 else 0.0)
        n_labels.append(f"{n_det}\n/{n_total}")

    ax.boxplot(
        data_all, positions=xs, widths=0.55, patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        boxprops=dict(facecolor=color, alpha=0.55),
    )
    ax.scatter(xs, means_det, color="red",    zorder=5, s=40,
               label="Mean (detected reads only)")
    ax.scatter(xs, means_all, color="orange", zorder=5, s=40, marker="D",
               label="Mean (all reads incl. 0-event)")
    # Cap y-axis at 95th percentile of all detected densities to avoid
    # the short-read bins (very high density) stretching the scale
    all_det = np.concatenate([d for d in data_all if len(d) > 0])
    if len(all_det) > 0:
        y_cap = np.percentile(all_det, 95) * 1.15
        ax.set_ylim(0, max(y_cap, 1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
    y_lo, y_hi = ax.get_ylim()
    for xi, lbl in zip(xs, n_labels):
        ax.text(xi, y_lo - 0.12 * (y_hi - y_lo),
                lbl, ha="center", va="top", fontsize=6.5, color="#333")
    ax.set_ylabel("Events / Mb of read", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.7)
    ax.legend(fontsize=7, loc="upper right")


# ── main ──────────────────────────────────────────────────────────────────────

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
    print(f"  Median length: {read_lengths_kb.median():.1f} kb")

    # ── v4.3 run3 predictions ─────────────────────────────────────────────────
    print("Loading v4.3 run3 predictions...")
    preds = pd.read_csv(PRED_V43, sep="\t")

    lf_events  = windows_to_events(preds, class_id=1,
                                   prob_threshold=PRED_FORK_THR,
                                   min_windows=1, max_gap=5000)
    rf_events  = windows_to_events(preds, class_id=2,
                                   prob_threshold=PRED_FORK_THR,
                                   min_windows=1, max_gap=5000)
    ori_events = windows_to_events(preds, class_id=3,
                                   prob_threshold=PRED_ORI_THR,
                                   min_windows=1, max_gap=5000)

    lf_counts  = count_events_per_read(lf_events,  all_read_ids)
    rf_counts  = count_events_per_read(rf_events,  all_read_ids)
    ori_counts = count_events_per_read(ori_events, all_read_ids)

    print(f"  LF events: {len(lf_events):,}  on {int((lf_counts>0).sum()):,} reads")
    print(f"  RF events: {len(rf_events):,}  on {int((rf_counts>0).sum()):,} reads")
    print(f"  ORI events: {len(ori_events):,}  on {int((ori_counts>0).sum()):,} reads")

    # ── Nerea manual ORIs ─────────────────────────────────────────────────────
    print("Loading Nerea ORIs...")
    nerea_ori = pd.read_csv(NEREA_ORI_BED, sep="\t", header=None,
                             usecols=[0, 1, 2, 3],
                             names=["chr", "start", "end", "read_id"])
    nerea_ori_counts = count_events_per_read(nerea_ori, all_read_ids)
    print(f"  Nerea ORIs: {len(nerea_ori):,}  on {int((nerea_ori_counts>0).sum()):,} reads")

    # ── Density (events / Mb) ─────────────────────────────────────────────────
    sources = {
        "Predicted Left Forks\n(v4.3 run3)":   (density_per_read(lf_counts,  read_lengths_bp), "#3498db"),
        "Predicted Right Forks\n(v4.3 run3)":  (density_per_read(rf_counts,  read_lengths_bp), "#e74c3c"),
        "Predicted ORIs\n(v4.3 run3)":          (density_per_read(ori_counts, read_lengths_bp), "#2ecc71"),
        "Manual ORIs\n(Nerea annotations)":     (density_per_read(nerea_ori_counts, read_lengths_bp), "#2c3e50"),
    }

    # ── Figure: 2×2 grid ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle(
        "Replication features per Mb of data vs read length\n"
        f"(reads >30 kb with BrdU signal, n={len(all_read_ids):,};  prob thr={PRED_FORK_THR};  20 kb bins)\n"
        "Boxes = distribution over detected reads only  ·  "
        "Red dot = mean (detected reads)  ·  "
        "Orange diamond = mean (all reads incl. 0-event)  —  "
        "gap between them reflects detection rate",
        fontsize=11, fontweight="bold",
    )

    for ax, (title, (density, color)) in zip(axes.flat, sources.items()):
        forkml_boxplot_density(
            ax=ax,
            density_series=density,
            read_lengths_kb=read_lengths_kb,
            bin_edges=bin_edges,
            bin_labels=bin_labels,
            color=color,
            title=title,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUT_DIR / "readlen_vs_features_per_Mb.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ── Summary TSV ───────────────────────────────────────────────────────────
    full_edges  = [0] + bin_edges
    full_labels = ["<30"] + list(bin_labels)
    len_bins = pd.cut(read_lengths_kb, bins=full_edges, labels=full_labels,
                      right=False, include_lowest=True)

    rows = []
    for title, (density, _) in sources.items():
        df = pd.DataFrame({"bin": len_bins, "density": density,
                           "read_length_kb": read_lengths_kb})
        df = df[df["bin"].isin(bin_labels)]
        for lbl in bin_labels:
            grp = df[df["bin"] == lbl]["density"].dropna()
            grp_det = grp[grp > 0]
            n_total_in_bin = int((df["bin"] == lbl).sum())
            total_mb_in_bin = float(
                df.loc[df["bin"] == lbl, "read_length_kb"].sum() / 1000
            )
            total_events_in_bin = int(
                (density.reindex(df.index)[df["bin"] == lbl]).fillna(0)
                .mul(df.loc[df["bin"] == lbl, "read_length_kb"] / 1000).sum()
            )
            rows.append({
                "source": title.replace("\n", " "),
                "bin_kb": lbl,
                "n_reads_total": n_total_in_bin,
                "n_reads_detected": len(grp_det),
                "total_Mb_in_bin": round(total_mb_in_bin, 2),
                "total_events_in_bin": total_events_in_bin,
                "aggregate_events_per_Mb": round(
                    total_events_in_bin / total_mb_in_bin
                    if total_mb_in_bin > 0 else 0, 4),
                "mean_events_per_Mb": round(float(grp.mean()), 4) if len(grp) > 0 else 0,
                "median_events_per_Mb": round(float(grp.median()), 4) if len(grp) > 0 else 0,
            })

    tsv_path = OUT_DIR / "readlen_vs_features_per_Mb_summary.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved summary TSV: {tsv_path}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
