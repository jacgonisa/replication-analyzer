#!/usr/bin/env python
"""Plot per-window probability curves from multiple FORTE models on the same reads.

For each read: signal panel + real-annotation bar + one probability panel per model.
Useful for judging whether high-probability windows are real events or noise,
even when annotations are incomplete.

Usage:
  python plot_multimodel_probs.py \
      --config   CODEX/configs/forte_v1.yaml \
      --models   v1:CODEX/models/forte_v1.keras \
                 v1_cons:CODEX/models/forte_v1_conservative.keras:CODEX/configs/forte_v1_conservative.yaml \
                 v2:CODEX/models/forte_v2.keras:CODEX/configs/forte_v2.yaml \
                 v3:CODEX/models/forte_v3.keras:CODEX/configs/forte_v3.yaml \
                 v4:CODEX/models/forte_v4.keras:CODEX/configs/forte_v4.yaml \
      --read-ids  READ1 READ2 READ3 \
      --threshold 0.40 \
      --output-dir CODEX/results/multimodel_probs
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

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

import tensorflow as tf
from replication_analyzer_codex.losses import (
    SparseCategoricalFocalLoss, MaskedMacroF1,
    MaskedClassPrecision, MaskedClassRecall,
)
from replication_analyzer_codex.evaluation import predict_reads
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

COL_LEFT   = "#1f77b4"
COL_RIGHT  = "#d62728"
COL_ORI    = "#2ca02c"
COL_SIGNAL = "#555555"


def load_xy_for_read(base_dir, run_dirs, read_id):
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


def load_bed(path):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def add_gt_spans(ax, df, read_id, color, label, ymin=0.0, ymax=1.0, alpha=0.25):
    subset = df[df["read_id"] == read_id]
    used = False
    for row in subset.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), ymin=ymin, ymax=ymax,
                   color=color, alpha=alpha, label=label if not used else None)
        used = True


def plot_read(read_id, xy, left_gt, right_gt, ori_gt,
              model_predictions,   # list of (name, preds_df)
              threshold, output_path):
    """One figure per read: signal + GT bar + N model probability panels."""
    n_models = len(model_predictions)
    n_rows = 2 + n_models
    height_ratios = [2.5] + [0.5] + [1.8] * n_models

    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 2.5 + 2.0 * n_models),
                              sharex=True, height_ratios=height_ratios,
                              gridspec_kw={"hspace": 0.08})

    x_sig = xy["start"].to_numpy()
    y_sig = xy["signal"].to_numpy()

    # ── Panel 0: BrdU signal ─────────────────────────────────────────────────
    ax = axes[0]
    ax.step(x_sig, y_sig, where="post", color=COL_SIGNAL, linewidth=0.9)
    ax.fill_between(x_sig, y_sig, step="post", alpha=0.08, color=COL_SIGNAL)
    # shade GT annotations on signal panel too
    add_gt_spans(ax, left_gt,  read_id, COL_LEFT,  "GT left fork",  alpha=0.18)
    add_gt_spans(ax, right_gt, read_id, COL_RIGHT, "GT right fork", alpha=0.18)
    add_gt_spans(ax, ori_gt,   read_id, COL_ORI,   "GT origin",     alpha=0.18)
    ax.set_ylabel("BrdU signal", fontsize=8)
    ax.set_title(f"Read  {read_id}   |   prob threshold = {threshold}",
                 fontsize=9, fontweight="bold")
    ax.grid(alpha=0.15, linewidth=0.4)
    ax.legend(loc="upper right", ncol=3, fontsize=7, framealpha=0.75)
    ax.set_ylim(bottom=0)

    # ── Panel 1: GT annotation bar ───────────────────────────────────────────
    ax = axes[1]
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel("GT", fontsize=8, rotation=0, labelpad=28, va="center")
    ax.grid(alpha=0.12, axis="x", linewidth=0.4)
    for df, col, y0, y1 in [
        (left_gt,  COL_LEFT,  0.68, 0.98),
        (right_gt, COL_RIGHT, 0.34, 0.64),
        (ori_gt,   COL_ORI,   0.02, 0.30),
    ]:
        subset = df[df["read_id"] == read_id]
        for row in subset.itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), ymin=y0, ymax=y1,
                       color=col, alpha=0.6)
    handles = [
        mpatches.Patch(color=COL_LEFT,  label="L"),
        mpatches.Patch(color=COL_RIGHT, label="R"),
        mpatches.Patch(color=COL_ORI,   label="ORI"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=7, framealpha=0.7)

    # ── Panels 2+: per-model probability curves ──────────────────────────────
    for m_idx, (model_name, preds) in enumerate(model_predictions):
        ax = axes[2 + m_idx]
        ax.set_ylabel(model_name, fontsize=8, rotation=0, labelpad=38, va="center")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15, linewidth=0.4)
        ax.axhline(threshold, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)

        if preds is None or len(preds) == 0:
            ax.text(0.5, 0.5, "no predictions", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            continue

        read_preds = preds[preds["read_id"] == read_id].copy()
        if len(read_preds) == 0:
            ax.text(0.5, 0.5, "no predictions", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            continue

        read_preds = read_preds.sort_values("start")
        xp = read_preds["start"].to_numpy()

        for col_name, color, label in [
            ("prob_left_fork",  COL_LEFT,  "p(left)"),
            ("prob_right_fork", COL_RIGHT, "p(right)"),
            ("prob_origin",     COL_ORI,   "p(ori)"),
        ]:
            if col_name in read_preds.columns:
                yp = read_preds[col_name].to_numpy()
                ax.fill_between(xp, yp, alpha=0.18, color=color, step="post")
                ax.step(xp, yp, where="post", color=color,
                        linewidth=1.0, label=label)

        # shade GT annotations faintly in probability panels
        add_gt_spans(ax, left_gt,  read_id, COL_LEFT,  None, alpha=0.10)
        add_gt_spans(ax, right_gt, read_id, COL_RIGHT, None, alpha=0.10)
        add_gt_spans(ax, ori_gt,   read_id, COL_ORI,   None, alpha=0.10)

        ax.legend(loc="upper right", ncol=3, fontsize=6, framealpha=0.7)

    axes[-1].set_xlabel("Genomic position (bp)", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Base config yaml (for data + annotation paths)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="name:model.keras[:config.yaml]  (config optional)")
    parser.add_argument("--read-ids", nargs="+",
                        help="Specific read IDs to plot")
    parser.add_argument("--split-manifest",
                        help="TSV split manifest; if given, samples N val reads with GT")
    parser.add_argument("--n-reads", type=int, default=20,
                        help="Number of val reads to sample (used with --split-manifest)")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # Parse model specs
    model_specs = []
    for spec in args.models:
        parts = spec.split(":")
        name = parts[0]
        model_path = parts[1]
        cfg_path = parts[2] if len(parts) > 2 else None
        model_specs.append((name, model_path, cfg_path))

    # Resolve read IDs
    read_ids = args.read_ids
    if not read_ids:
        assert args.split_manifest, "Provide --read-ids or --split-manifest"
        manifest = pd.read_csv(args.split_manifest, sep="\t")
        val_reads = manifest[manifest["split"] == "val"]["read_id"].tolist()
        # Load GT to restrict to reads with annotations
        left_gt  = load_bed(base_config["data"].get("left_forks_bed"))
        right_gt = load_bed(base_config["data"].get("right_forks_bed"))
        ori_gt   = load_bed(base_config["data"].get("ori_annotations_bed"))
        annotated = (
            set(left_gt["read_id"]) | set(right_gt["read_id"]) | set(ori_gt["read_id"])
        )
        val_annotated = [r for r in val_reads if r in annotated]
        rng = np.random.default_rng(args.seed)
        read_ids = list(rng.choice(val_annotated,
                                   size=min(args.n_reads, len(val_annotated)),
                                   replace=False))
        print(f"Sampled {len(read_ids)} annotated val reads")
    else:
        left_gt  = load_bed(base_config["data"].get("left_forks_bed"))
        right_gt = load_bed(base_config["data"].get("right_forks_bed"))
        ori_gt   = load_bed(base_config["data"].get("ori_annotations_bed"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all models + run predictions for all reads at once
    all_predictions = []  # list of (model_name, preds_df_for_all_reads)
    for model_name, model_path, cfg_path in model_specs:
        print(f"\nLoading {model_name} from {model_path} ...")
        if cfg_path:
            with open(cfg_path) as f:
                mc = yaml.safe_load(f)
            preproc = mc["preprocessing"]
        else:
            preproc = base_config["preprocessing"]

        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        max_length = model.input_shape[1]
        print(f"  max_length={max_length}  predicting {len(read_ids)} reads...")

        # Load XY for all reads
        xy_rows = []
        for rid in read_ids:
            xy = load_xy_for_read(base_config["data"]["base_dir"],
                                  base_config["data"]["run_dirs"], rid)
            if xy is not None:
                xy_rows.append(xy)
        if not xy_rows:
            all_predictions.append((model_name, None))
            tf.keras.backend.clear_session()
            continue

        xy_all = pd.concat(xy_rows, ignore_index=True)
        preds = predict_reads(model, xy_all, read_ids, max_length, preproc)
        all_predictions.append((model_name, preds))
        tf.keras.backend.clear_session()
        print(f"  Done — {len(preds)} prediction rows")

    # Plot each read
    print(f"\nPlotting {len(read_ids)} reads...")
    for read_id in read_ids:
        print(f"  {read_id}")
        xy = load_xy_for_read(base_config["data"]["base_dir"],
                               base_config["data"]["run_dirs"], read_id)
        if xy is None:
            print(f"    XY data not found, skipping")
            continue

        model_preds_for_read = [(name, preds) for name, preds in all_predictions]
        out_path = output_dir / f"{read_id}.png"
        plot_read(read_id, xy, left_gt, right_gt, ori_gt,
                  model_preds_for_read, args.threshold, out_path)

    print(f"\nDone. Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
