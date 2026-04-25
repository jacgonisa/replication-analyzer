#!/usr/bin/env python
"""Plot precision/recall/F1 heatmaps (prob_threshold × iou_threshold) per event type.

Usage (run from replication-analyzer/ root):
  python CODEX/scripts/plot_threshold_heatmaps.py \
      --sweep-all    CODEX/results/forte_v5.1/evaluation_test_human_vs_pseudo/threshold_sweep_all.tsv \
      --sweep-human  CODEX/results/forte_v5.1/evaluation_test_human_vs_pseudo/threshold_sweep_human.tsv \
      --sweep-pseudo CODEX/results/forte_v5.1/evaluation_test_human_vs_pseudo/threshold_sweep_pseudo.tsv \
      --output-dir   CODEX/results/forte_v5.1
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


EVENT_LABELS = {
    "left_fork":  "Left Fork",
    "right_fork": "Right Fork",
    "origin":     "Origin",
}
METRICS = ["precision", "recall", "f1"]
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1"}


def load_sweep(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  Warning: {path} not found, skipping.")
        return None
    return pd.read_csv(p, sep="\t")


def pivot_metric(df: pd.DataFrame, event_type: str, metric: str) -> tuple[pd.DataFrame, list, list]:
    sub = df[df["event_type"] == event_type].copy()
    if sub.empty:
        return None, [], []
    prob_vals = sorted(sub["prob_threshold"].unique())
    iou_vals  = sorted(sub["iou_threshold"].unique())
    mat = sub.pivot_table(index="prob_threshold", columns="iou_threshold",
                          values=metric, aggfunc="mean")
    mat = mat.reindex(index=prob_vals, columns=iou_vals)
    return mat, prob_vals, iou_vals


def plot_heatmaps_for_subset(df: pd.DataFrame, subset_label: str, output_dir: Path):
    """One figure per GT subset with a 3×3 grid: rows=event_types, cols=metrics."""
    event_types = [et for et in ["left_fork", "right_fork", "origin"] if et in df["event_type"].values]
    n_events = len(event_types)
    n_metrics = len(METRICS)

    fig, axes = plt.subplots(n_events, n_metrics,
                             figsize=(5 * n_metrics, 4 * n_events),
                             squeeze=False)

    for r, event_type in enumerate(event_types):
        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            mat, prob_vals, iou_vals = pivot_metric(df, event_type, metric)
            if mat is None:
                ax.set_visible(False)
                continue

            cmap = "RdYlGn"
            vmin, vmax = 0.0, 1.0
            im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate cells
            for ri, pth in enumerate(prob_vals):
                for ci, ith in enumerate(iou_vals):
                    val = mat.values[ri, ci]
                    if np.isnan(val):
                        continue
                    color = "black" if 0.3 < val < 0.85 else "white"
                    ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")

            # Mark best F1 cell
            if metric == "f1":
                best_idx = np.nanargmax(mat.values)
                br, bc = np.unravel_index(best_idx, mat.values.shape)
                ax.add_patch(plt.Rectangle((bc - 0.5, br - 0.5), 1, 1,
                                           fill=False, edgecolor="blue", lw=2))

            ax.set_xticks(range(len(iou_vals)))
            ax.set_xticklabels([f"{v:.1f}" for v in iou_vals], fontsize=8)
            ax.set_yticks(range(len(prob_vals)))
            ax.set_yticklabels([f"{v:.2f}" for v in prob_vals], fontsize=8)
            ax.set_xlabel("IoU threshold", fontsize=9)
            ax.set_ylabel("Prob threshold", fontsize=9)

            title = f"{EVENT_LABELS.get(event_type, event_type)} — {METRIC_LABELS[metric]}"
            ax.set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle(f"Threshold sweep · GT subset: {subset_label}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fname = output_dir / f"threshold_heatmap_{subset_label.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-all",    default=None)
    parser.add_argument("--sweep-human",  default=None)
    parser.add_argument("--sweep-pseudo", default=None)
    parser.add_argument("--output-dir",   required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweeps = {
        "ALL":    load_sweep(args.sweep_all),
        "HUMAN":  load_sweep(args.sweep_human),
        "PSEUDO": load_sweep(args.sweep_pseudo),
    }

    for label, df in sweeps.items():
        if df is None:
            print(f"Skipping {label} (no data).")
            continue
        print(f"Plotting {label}…")
        plot_heatmaps_for_subset(df, label, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
