#!/usr/bin/env python
"""Plot reads showing real ORIs + ORI-validated pseudo-forks with optional probability curves."""

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

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

COL_ORI      = "#2ca02c"
COL_PSEUDO_L = "#1f77b4"
COL_PSEUDO_R = "#d62728"


def load_bed(path):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                     names=["chr", "start", "end", "read_id"])
    return df.drop_duplicates(subset=["chr", "start", "end", "read_id"]).reset_index(drop=True)


def load_xy(base_dir, run_dirs, read_id):
    rows = []
    for run_dir in run_dirs:
        f = Path(base_dir) / run_dir / f"plot_data_{read_id}.txt"
        if f.exists():
            df = pd.read_csv(f, sep="\t", header=None,
                             names=["chr", "start", "end", "signal"])
            df["read_id"] = read_id
            rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True).sort_values("start").reset_index(drop=True)


def run_inference(model, xy, read_id, preprocessing_config):
    from replication_analyzer_codex.evaluation import predict_reads
    max_length = model.input_shape[1]
    return predict_reads(model, xy, [read_id], max_length, preprocessing_config)


def plot_read(read_id, xy, ori_real, pseudo_left, pseudo_right,
              output_path, preds=None, thresholds=(0.45, 0.50)):
    ori_sub = ori_real[ori_real["read_id"] == read_id]
    pl_sub  = pseudo_left[pseudo_left["read_id"] == read_id]
    pr_sub  = pseudo_right[pseudo_right["read_id"] == read_id]

    has_probs = preds is not None and len(preds) > 0
    n_rows = 4 if has_probs else 3
    height_ratios = ([3, 1.2, 0.8, 0.8] if has_probs else [3, 0.8, 0.8])

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 + 1.8 * n_rows),
                              sharex=True, height_ratios=height_ratios,
                              gridspec_kw={"hspace": 0.05})

    x = xy["start"].to_numpy()
    y = xy["signal"].to_numpy()

    # ── Panel 0: BrdU signal + ORI shading ───────────────────────────────────
    ax = axes[0]
    ax.step(x, y, where="post", color="black", linewidth=1.2)
    ax.fill_between(x, y, step="post", alpha=0.12, color="gray")
    for row in ori_sub.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), color=COL_ORI, alpha=0.20)
    ax.set_ylabel("BrdU signal")
    n_ori = len(ori_sub)
    ax.set_title(
        f"{read_id}\n"
        f"{n_ori} real ORIs  |  {len(pl_sub)} pseudo left-forks  |  {len(pr_sub)} pseudo right-forks  "
        f"(no real forks annotated)",
        fontsize=9,
    )
    ax.grid(alpha=0.2)
    handles = [
        mpatches.Patch(color=COL_ORI,      alpha=0.4, label="Real ORI"),
        mpatches.Patch(color=COL_PSEUDO_L, alpha=0.7, label="Pseudo left-fork"),
        mpatches.Patch(color=COL_PSEUDO_R, alpha=0.7, label="Pseudo right-fork"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=8)

    # ── Panel 1: probability curves (optional) ───────────────────────────────
    prob_ax_idx = 1
    if has_probs:
        ax = axes[1]
        read_preds = preds[preds["read_id"] == read_id].sort_values("start")
        xp = read_preds["start"].to_numpy()
        ax.step(xp, read_preds["prob_left_fork"].to_numpy(),
                where="post", color=COL_PSEUDO_L, linewidth=1.1, label="prob left-fork", alpha=0.85)
        ax.step(xp, read_preds["prob_right_fork"].to_numpy(),
                where="post", color=COL_PSEUDO_R, linewidth=1.1, label="prob right-fork", alpha=0.85)
        ax.step(xp, read_preds["prob_origin"].to_numpy(),
                where="post", color=COL_ORI, linewidth=1.1, label="prob origin", alpha=0.85)
        for thr, ls in zip(thresholds, ["--", ":"]):
            ax.axhline(thr, color="gray", linewidth=0.8, linestyle=ls, label=f"thresh {thr}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability", fontsize=8)
        ax.legend(loc="upper right", ncol=5, fontsize=7, framealpha=0.7)
        ax.grid(alpha=0.15)
        prob_ax_idx = 2

    # ── Panel: real ORIs bar ─────────────────────────────────────────────────
    ax = axes[prob_ax_idx]
    ax.set_ylabel("Real", fontsize=8, rotation=0, labelpad=30, va="center")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.15, axis="x")
    for row in ori_sub.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), ymin=0.1, ymax=0.9,
                   color=COL_ORI, alpha=0.6)

    # ── Panel: pseudo-fork bar ───────────────────────────────────────────────
    ax = axes[prob_ax_idx + 1]
    ax.set_ylabel("Pseudo\nforks", fontsize=8, rotation=0, labelpad=30, va="center")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.15, axis="x")
    for row in pl_sub.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), ymin=0.52, ymax=0.95,
                   color=COL_PSEUDO_L, alpha=0.65)
    for row in pr_sub.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), ymin=0.05, ymax=0.48,
                   color=COL_PSEUDO_R, alpha=0.65)

    axes[-1].set_xlabel("Genomic position (bp)")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ori-validated-left", required=True)
    parser.add_argument("--ori-validated-right", required=True)
    parser.add_argument("--read-id", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", help="Optional model for probability curves")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.45, 0.50],
                        help="Threshold lines to draw on probability panel")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ori_real     = load_bed(config["data"]["ori_annotations_bed"])
    pseudo_left  = load_bed(args.ori_validated_left)
    pseudo_right = load_bed(args.ori_validated_right)

    model = None
    preprocessing_config = config["preprocessing"]
    if args.model:
        import tensorflow as tf
        from replication_analyzer_codex.losses import (
            SparseCategoricalFocalLoss, MaskedMacroF1,
            MaskedClassPrecision, MaskedClassRecall,
        )
        from replication_analyzer.models.base import SelfAttention
        from replication_analyzer.models.losses import MultiClassFocalLoss
        from replication_analyzer.training.callbacks import MultiClassF1Score
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(args.model, custom_objects={
            "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
            "MaskedMacroF1": MaskedMacroF1,
            "MaskedClassPrecision": MaskedClassPrecision,
            "MaskedClassRecall": MaskedClassRecall,
            "SelfAttention": SelfAttention,
            "MultiClassFocalLoss": MultiClassFocalLoss,
            "MultiClassF1Score": MultiClassF1Score,
        })
        print(f"Loaded model: {args.model}  (max_length={model.input_shape[1]})")

    for read_id in args.read_id:
        xy = load_xy(config["data"]["base_dir"], config["data"]["run_dirs"], read_id)
        if xy is None:
            print(f"No XY data for {read_id}, skipping")
            continue

        preds = None
        if model is not None:
            from replication_analyzer_codex.evaluation import predict_reads
            preds = predict_reads(model, xy, [read_id],
                                  model.input_shape[1], preprocessing_config)

        plot_read(
            read_id=read_id,
            xy=xy,
            ori_real=ori_real,
            pseudo_left=pseudo_left,
            pseudo_right=pseudo_right,
            output_path=Path(args.output_dir) / f"ori_validated_{read_id[:8]}.png",
            preds=preds,
            thresholds=tuple(args.thresholds),
        )


if __name__ == "__main__":
    main()
