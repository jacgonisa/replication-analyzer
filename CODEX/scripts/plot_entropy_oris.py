#!/usr/bin/env python
"""Find high-entropy vs low-entropy ORI predictions from FORTE v1
and plot the reads they live in.

High entropy ORI = model predicts origin but is uncertain (spread probability)
Low entropy ORI  = model predicts origin confidently

Usage:
  CUDA_VISIBLE_DEVICES="" python plot_entropy_oris.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

ROOT       = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

import tensorflow as tf
from replication_analyzer_codex.losses import (
    SparseCategoricalFocalLoss, MaskedMacroF1,
    MaskedClassPrecision, MaskedClassRecall,
)
from replication_analyzer_codex.evaluation import predict_reads, windows_to_events
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

BASE       = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")
COL_LEFT   = "#1f77b4"
COL_RIGHT  = "#d62728"
COL_ORI_GT = "#2ca02c"
COL_ORI_AI = "#ff7f0e"
COL_HI_ENT = "#e41a1c"   # red  — high entropy ORI window
COL_LO_ENT = "#377eb8"   # blue — low entropy ORI window


def load_bed4(path):
    if not Path(path).exists():
        return pd.DataFrame(columns=["chr","start","end","read_id"])
    return pd.read_csv(path, sep="\t", header=None, usecols=[0,1,2,3],
                       names=["chr","start","end","read_id"])


def plot_read(ax_sig, ax_gt, ax_ai, read_id, xy_data, gt_left, gt_right, gt_ori,
              preds_read, entropy_windows, entropy_label, entropy_col,
              threshold=0.40, max_gap=5000):
    """Draw one read across three axes: signal, GT bar, AI prediction bar."""
    bins = xy_data[xy_data["read_id"] == read_id].sort_values("start")
    x, y = bins["start"].values, bins["signal"].values

    # ── Signal panel ──────────────────────────────────────────────────────────
    ax_sig.step(x, y, where="post", color="black", linewidth=0.9)
    ax_sig.fill_between(x, y, step="post", color="gray", alpha=0.08)
    for df, col in [(gt_left, COL_LEFT), (gt_right, COL_RIGHT), (gt_ori, COL_ORI_GT)]:
        for row in df[df["read_id"] == read_id].itertuples(index=False):
            ax_sig.axvspan(int(row.start), int(row.end), color=col, alpha=0.18)
    # Highlight the entropy windows
    for _, w in entropy_windows[entropy_windows["read_id"] == read_id].iterrows():
        ax_sig.axvspan(int(w["start"]), int(w["end"]),
                       color=entropy_col, alpha=0.55, zorder=3,
                       label=entropy_label)
    ax_sig.set_xticks([]); ax_sig.tick_params(labelsize=6)
    ax_sig.grid(alpha=0.15)
    ax_sig.set_title(f"{read_id[:14]}…", fontsize=8, fontweight="bold")

    # ── GT bar ────────────────────────────────────────────────────────────────
    ax_gt.set_ylim(0,1); ax_gt.set_yticks([]); ax_gt.set_xticks([])
    for df, col, y0, y1 in [(gt_left,  COL_LEFT,  0.68, 0.98),
                              (gt_right, COL_RIGHT, 0.36, 0.66),
                              (gt_ori,   COL_ORI_GT,0.03, 0.33)]:
        for row in df[df["read_id"] == read_id].itertuples(index=False):
            ax_gt.axvspan(int(row.start), int(row.end), ymin=y0, ymax=y1,
                          color=col, alpha=0.75)
    ax_gt.set_xlim(x[0], x[-1])

    # ── AI prediction bar ─────────────────────────────────────────────────────
    ax_ai.set_ylim(0,1); ax_ai.set_yticks([]); ax_ai.tick_params(axis="x", labelsize=6)
    ax_ai.set_xlim(x[0], x[-1])
    if len(preds_read) > 0:
        for class_id, col, y0, y1 in [(1, COL_LEFT,  0.68, 0.98),
                                        (2, COL_RIGHT, 0.36, 0.66),
                                        (3, COL_ORI_AI,0.03, 0.33)]:
            evs = windows_to_events(preds_read, class_id, threshold,
                                    min_windows=1, max_gap=max_gap)
            evs = evs[evs["read_id"] == read_id]
            for row in evs.itertuples(index=False):
                ax_ai.axvspan(int(row.start), int(row.end), ymin=y0, ymax=y1,
                              color=col, alpha=0.75)
    # Mark the entropy windows on the AI bar too
    for _, w in entropy_windows[entropy_windows["read_id"] == read_id].iterrows():
        ax_ai.axvspan(int(w["start"]), int(w["end"]),
                      color=entropy_col, alpha=0.6, zorder=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="CODEX/configs/forte_v1.yaml")
    parser.add_argument("--model",     default="CODEX/models/forte_v1.keras")
    parser.add_argument("--xy-cache",  default="CODEX/results/cache/xy_data.pkl")
    parser.add_argument("--val-info",
        default="CODEX/results/forte_v1/preprocessed_forte_v1.val_info.tsv")
    parser.add_argument("--gt-left",
        default="data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed")
    parser.add_argument("--gt-right",
        default="data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed")
    parser.add_argument("--gt-ori",
        default="data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed")
    parser.add_argument("--n-reads",      type=int, default=3000)
    parser.add_argument("--n-examples",   type=int, default=4,
                        help="Number of reads to show per group")
    parser.add_argument("--entropy-pct",  type=float, default=10.0,
                        help="Percentile cutoff for high/low entropy")
    parser.add_argument("--threshold",    type=float, default=0.40)
    parser.add_argument("--output-dir",
        default="CODEX/results/entropy_ori_examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = BASE / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── Load GT ───────────────────────────────────────────────────────────────
    gt_left  = load_bed4(str(BASE / args.gt_left))
    gt_right = load_bed4(str(BASE / args.gt_right))
    gt_ori   = load_bed4(str(BASE / args.gt_ori))

    # ── Sample val reads and run inference ────────────────────────────────────
    val_info = pd.read_csv(str(BASE / args.val_info), sep="\t")
    rng = np.random.default_rng(args.seed)
    read_ids = val_info["read_id"].tolist()
    if len(read_ids) > args.n_reads:
        read_ids = [read_ids[i] for i in rng.choice(len(read_ids), args.n_reads, replace=False)]

    print(f"Loading XY cache…")
    with open(str(BASE / args.xy_cache), "rb") as fh:
        xy_data = pickle.load(fh)
    xy_data = xy_data[xy_data["read_id"].isin(set(read_ids))].copy()
    actual_ids = list(xy_data["read_id"].unique())
    print(f"  {len(actual_ids):,} reads loaded")

    with open(str(BASE / args.config)) as f:
        config = yaml.safe_load(f)

    print(f"Loading model…")
    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(str(BASE / args.model),
                                        custom_objects=CUSTOM_OBJECTS)
    max_length = model.input_shape[1]

    print(f"Running inference on {len(actual_ids):,} reads…")
    preds = predict_reads(model, xy_data, actual_ids, max_length,
                          config["preprocessing"])
    tf.keras.backend.clear_session()
    print(f"  {len(preds):,} windows predicted")

    # ── Find ORI windows and split by entropy ─────────────────────────────────
    ori_windows = preds[preds["predicted_class"] == 3].copy()
    print(f"  ORI-predicted windows: {len(ori_windows):,}")

    lo_thr = np.percentile(ori_windows["entropy"], args.entropy_pct)
    hi_thr = np.percentile(ori_windows["entropy"], 100 - args.entropy_pct)

    lo_ent = ori_windows[ori_windows["entropy"] <= lo_thr].copy()
    hi_ent = ori_windows[ori_windows["entropy"] >= hi_thr].copy()
    print(f"  Low  entropy ORI (≤ p{args.entropy_pct:.0f} = {lo_thr:.3f}): {len(lo_ent):,} windows")
    print(f"  High entropy ORI (≥ p{100-args.entropy_pct:.0f} = {hi_thr:.3f}): {len(hi_ent):,} windows")

    # Pick reads that have multiple entropy windows so plots are informative
    def pick_reads(ent_df, n, prefer_gt=True):
        counts = ent_df.groupby("read_id").size().sort_values(ascending=False)
        if prefer_gt:
            # prefer reads that also have GT ORI annotations
            gt_ids = set(gt_ori["read_id"])
            with_gt  = [r for r in counts.index if r in gt_ids]
            without  = [r for r in counts.index if r not in gt_ids]
            ordered  = with_gt + without
        else:
            ordered = list(counts.index)
        return ordered[:n]

    lo_reads = pick_reads(lo_ent, args.n_examples)
    hi_reads = pick_reads(hi_ent, args.n_examples)
    print(f"  Example reads — low entropy:  {lo_reads}")
    print(f"  Example reads — high entropy: {hi_reads}")

    # ── Plot function ─────────────────────────────────────────────────────────
    def make_figure(read_list, ent_df, entropy_col, entropy_label, filename):
        n = len(read_list)
        # 3 rows per read: signal (tall) | GT bar | AI bar
        n_rows = 3
        height_ratios = [3.5, 0.8, 0.8]

        fig, axes = plt.subplots(
            n_rows, n,
            figsize=(6.5 * n, sum(height_ratios) + 0.5),
            squeeze=False,
            gridspec_kw={"hspace": 0.07, "wspace": 0.05,
                         "height_ratios": height_ratios},
        )

        for ci, rid in enumerate(read_list):
            preds_read = preds[preds["read_id"] == rid].copy()
            ent_win    = ent_df[ent_df["read_id"] == rid].copy()

            plot_read(
                ax_sig=axes[0][ci],
                ax_gt=axes[1][ci],
                ax_ai=axes[2][ci],
                read_id=rid,
                xy_data=xy_data,
                gt_left=gt_left, gt_right=gt_right, gt_ori=gt_ori,
                preds_read=preds_read,
                entropy_windows=ent_win,
                entropy_label=entropy_label,
                entropy_col=entropy_col,
                threshold=args.threshold,
            )

            # Row labels (left-most column only)
            if ci == 0:
                axes[0][ci].set_ylabel("BrdU signal", fontsize=8)
                axes[1][ci].set_ylabel("GT", fontsize=8, rotation=0,
                                        labelpad=30, va="center")
                axes[2][ci].set_ylabel("FORTE v1", fontsize=8, rotation=0,
                                        labelpad=38, va="center")

        # Legend
        handles = [
            mpatches.Patch(color=COL_LEFT,      label="Left fork"),
            mpatches.Patch(color=COL_RIGHT,     label="Right fork"),
            mpatches.Patch(color=COL_ORI_GT,    label="ORI (GT)"),
            mpatches.Patch(color=COL_ORI_AI,    label="ORI (AI pred)"),
            mpatches.Patch(color=entropy_col,   label=entropy_label, alpha=0.7),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
                   bbox_to_anchor=(0.5, -0.04), framealpha=0.85)

        ent_str = f"entropy ≥ p{100-args.entropy_pct:.0f} ({hi_thr:.3f})" \
                  if "high" in entropy_label.lower() \
                  else f"entropy ≤ p{args.entropy_pct:.0f} ({lo_thr:.3f})"
        fig.suptitle(
            f"FORTE v1 — {entropy_label} origin windows  ({ent_str})\n"
            f"Highlighted windows = ORI predictions in the {entropy_label.split()[0]} entropy group",
            fontsize=10, fontweight="bold",
        )

        path = out / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ── Entropy distribution plot ─────────────────────────────────────────────
    fig_ent, ax_ent = plt.subplots(figsize=(8, 4))
    ax_ent.hist(ori_windows["entropy"], bins=80, color="#9467bd", alpha=0.75,
                edgecolor="white", linewidth=0.4)
    ax_ent.axvline(lo_thr, color=COL_LO_ENT, linewidth=2,
                   label=f"Low entropy cutoff (p{args.entropy_pct:.0f} = {lo_thr:.3f})")
    ax_ent.axvline(hi_thr, color=COL_HI_ENT, linewidth=2,
                   label=f"High entropy cutoff (p{100-args.entropy_pct:.0f} = {hi_thr:.3f})")
    ax_ent.set_xlabel("Entropy", fontsize=10)
    ax_ent.set_ylabel("# ORI windows", fontsize=10)
    ax_ent.set_title("Entropy distribution of ORI-predicted windows (FORTE v1)", fontsize=11)
    ax_ent.legend(fontsize=9)
    ax_ent.grid(alpha=0.25)
    fig_ent.tight_layout()
    fig_ent.savefig(out / "entropy_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ent)
    print(f"Saved: {out / 'entropy_distribution.png'}")

    # ── Make the two example figures ──────────────────────────────────────────
    make_figure(lo_reads, lo_ent, COL_LO_ENT,
                "Low entropy ORI (confident)", "low_entropy_oris.png")
    make_figure(hi_reads, hi_ent, COL_HI_ENT,
                "High entropy ORI (uncertain)", "high_entropy_oris.png")

    print(f"\nAll outputs: {out}")


if __name__ == "__main__":
    main()
