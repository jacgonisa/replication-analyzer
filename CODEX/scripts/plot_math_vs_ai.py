#!/usr/bin/env python
"""Compare GT annotations vs AI (FORTE v1) vs math methods on the same reads.

Layout (one column per read):
  Row 0 — raw BrdU signal with GT spans overlaid (tall)
  Row 1 — GT annotation bar  (left / right / ORI)
  Row 2 — FORTE v1 (AI) bar
  Row 3 — v2 GradPeak sensitive bar
  Row 4 — v3 LoG multiscale bar
  Row 5 — v3 Viterbi HMM (σ=5kb) bar

Usage:
  python CODEX/scripts/plot_math_vs_ai.py \
      --config CODEX/configs/forte_v1.yaml \
      --output CODEX/results/comparison_plots/math_vs_ai.png
"""

from __future__ import annotations

import argparse
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

from scipy.ndimage import gaussian_filter1d, label as nd_label
from mathematical_pipeline_v3 import multiscale_log, viterbi_hmm


CUSTOM_OBJECTS = {
    "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
    "MaskedMacroF1": MaskedMacroF1,
    "MaskedClassPrecision": MaskedClassPrecision,
    "MaskedClassRecall": MaskedClassRecall,
    "SelfAttention": SelfAttention,
    "MultiClassFocalLoss": MultiClassFocalLoss,
    "MultiClassF1Score": MultiClassF1Score,
}

COL_LEFT  = "#1f77b4"
COL_RIGHT = "#d62728"
COL_ORI   = "#2ca02c"
ORI_COLS  = ["#ff7f0e", "#9467bd", "#17becf", "#e377c2"]   # per-method ORI colour


# ── data loading helpers ───────────────────────────────────────────────────────

def load_xy_read(base_dir, run_dirs, read_id):
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


def load_bed4(path):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def pick_example_reads(gt_left, gt_right, gt_ori, n=4, seed=0):
    """Pick reads that have all three annotation types — most informative."""
    with_all = (set(gt_left["read_id"])
                & set(gt_right["read_id"])
                & set(gt_ori["read_id"]))
    if len(with_all) < n:
        # fall back: at least 2 annotation types
        with_two = ((set(gt_left["read_id"]) | set(gt_right["read_id"]))
                    & set(gt_ori["read_id"]))
        candidates = sorted(with_two)
    else:
        candidates = sorted(with_all)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    return [candidates[i] for i in sorted(idx)]


# ── v2 GradPeak sensitive (self-contained, no v2 module import) ───────────────

def _expand_rect(bins_df, target_len=10000):
    widths  = (bins_df["end"].values - bins_df["start"].values).astype(np.float64)
    signals = bins_df["signal"].values.astype(np.float64)
    total   = widths.sum()
    cum     = np.cumsum(widths)
    fracs   = np.linspace(0, total, target_len)
    idx     = np.searchsorted(cum, fracs, side="right").clip(0, len(signals) - 1)
    gen_pos = (bins_df["start"].iloc[0]
               + fracs * (bins_df["end"].iloc[-1] - bins_df["start"].iloc[0]) / total)
    return signals[idx], gen_pos


def _merge_mask(mask, pos, merge_gap_bp, min_len_bp):
    if not mask.any():
        return []
    read_len = float(pos[-1] - pos[0] + 1)
    gap_u    = max(1, int(merge_gap_bp * len(pos) / read_len))
    dilated  = np.zeros_like(mask)
    for i in np.where(mask)[0]:
        dilated[max(0, i - gap_u): i + gap_u + 1] = True
    labeled, n = nd_label(dilated)
    events = []
    for lab in range(1, n + 1):
        idxs = np.where(labeled == lab)[0]
        if not mask[idxs].any():
            continue
        s_bp = float(pos[idxs[0]]); e_bp = float(pos[idxs[-1]])
        if e_bp - s_bp >= min_len_bp:
            events.append((s_bp, e_bp))
    return events


def _segs_to_df(segs, chrom, read_id):
    if not segs:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.DataFrame([{"chr": chrom, "start": int(s), "end": int(e),
                           "read_id": read_id} for s, e in segs])


def _origins_from_forks(left_df, right_df, read_id, chrom, max_dist=150_000):
    if len(left_df) == 0 or len(right_df) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    origins = []
    for _, L in left_df.sort_values("end").iterrows():
        cands = right_df[right_df["start"] >= L["end"]]
        if len(cands) == 0:
            continue
        R = cands.iloc[0]
        if R["start"] - L["end"] <= max_dist:
            origins.append({"chr": chrom, "start": int(L["end"]),
                             "end": int(R["start"]), "read_id": read_id})
    return pd.DataFrame(origins) if origins else pd.DataFrame(
        columns=["chr", "start", "end", "read_id"])


def gradpeak_sensitive(bins_df: pd.DataFrame,
                        smooth_sigma_kb: float = 3.0,
                        grad_rel_threshold: float = 0.15,
                        merge_gap_bp: int = 8000,
                        min_fork_len_bp: int = 2000,
                        target_len: int = 10000) -> dict:
    sig, pos  = _expand_rect(bins_df, target_len)
    read_id   = bins_df["read_id"].iloc[0]
    chrom     = bins_df["chr"].iloc[0]
    read_len  = float(pos[-1] - pos[0] + 1)
    sig_n     = (sig - sig.mean()) / (sig.std() + 1e-8)
    sigma     = max(1.0, smooth_sigma_kb * 1000 * target_len / read_len)
    smoothed  = gaussian_filter1d(sig_n, sigma=sigma)
    grad      = np.gradient(smoothed)
    p95       = np.percentile(np.abs(grad), 95)
    threshold = grad_rel_threshold * max(p95, 1e-9)
    left_events  = _merge_mask(grad >  threshold, pos, merge_gap_bp, min_fork_len_bp)
    right_events = _merge_mask(grad < -threshold, pos, merge_gap_bp, min_fork_len_bp)
    left_df  = _segs_to_df(left_events,  chrom, read_id)
    right_df = _segs_to_df(right_events, chrom, read_id)
    ori_df   = _origins_from_forks(left_df, right_df, read_id, chrom)
    return {"left_fork": left_df, "right_fork": right_df, "origin": ori_df}


# ── plotting helpers ───────────────────────────────────────────────────────────

def _spans(ax, df, read_id, color, ymin, ymax, alpha=0.75):
    for row in df[df["read_id"] == read_id].itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), ymin=ymin, ymax=ymax,
                   color=color, alpha=alpha)


def draw_annotation_bar(ax, read_id, left_df, right_df, ori_df,
                         ori_color=COL_ORI, label=None):
    ax.set_ylim(0, 1); ax.set_yticks([])
    _spans(ax, left_df,  read_id, COL_LEFT,  0.68, 0.98)
    _spans(ax, right_df, read_id, COL_RIGHT, 0.36, 0.66)
    _spans(ax, ori_df,   read_id, ori_color, 0.03, 0.33)
    if label:
        ax.set_ylabel(label, fontsize=7.5, rotation=0, labelpad=48, va="center",
                      fontweight="bold")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="CODEX/configs/forte_v1.yaml")
    parser.add_argument("--model",
                        default="CODEX/models/forte_v1.keras")
    parser.add_argument("--model-name", default="FORTE v1 (AI)")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--max-gap",   type=int,   default=5000)
    parser.add_argument("--n-reads",   type=int,   default=5,
                        help="Number of example reads to plot")
    parser.add_argument("--read-ids",  nargs="+",  default=None,
                        help="Specific read IDs (overrides auto-selection)")
    parser.add_argument("--gt-left",
        default="data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed")
    parser.add_argument("--gt-right",
        default="data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed")
    parser.add_argument("--gt-ori",
        default="data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed")
    parser.add_argument("--output",
                        default="CODEX/results/comparison_plots/math_vs_ai.png")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")

    with open(base / args.config) as f:
        config = yaml.safe_load(f)

    # GT annotations
    gt_left  = load_bed4(str(base / args.gt_left))
    gt_right = load_bed4(str(base / args.gt_right))
    gt_ori   = load_bed4(str(base / args.gt_ori))

    # Select reads
    if args.read_ids:
        read_ids = args.read_ids
    else:
        read_ids = pick_example_reads(gt_left, gt_right, gt_ori,
                                      n=args.n_reads, seed=args.seed)
    print(f"Example reads: {read_ids}")

    # Load XY signal for each read
    run_dirs = config["data"]["run_dirs"]
    base_dir = config["data"]["base_dir"]
    reads_xy = {}
    for rid in read_ids:
        df = load_xy_read(base_dir, run_dirs, rid)
        if df is not None:
            reads_xy[rid] = df
    read_ids = [r for r in read_ids if r in reads_xy]
    if not read_ids:
        raise RuntimeError("No XY data found for any example read")
    print(f"Loaded {len(read_ids)} reads")

    # ── Run math methods ───────────────────────────────────────────────────────
    math_methods = [
        ("v2 GradPeak\nsensitive",  gradpeak_sensitive,  {}),
        ("v3 LoG\nmultiscale",      multiscale_log,
            dict(scales_kb=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                 log_prominence=0.03, grad_rel_threshold=0.15)),
        ("v3 Viterbi HMM\n(σ=5kb)", viterbi_hmm,
            dict(smooth_sigma_kb=5.0,
                 expected_fork_len_bp=30_000,
                 expected_ori_len_bp=5_000,
                 min_len_bp=2000)),
    ]

    math_preds = {mname: {} for mname, _, _ in math_methods}
    for rid in read_ids:
        bins = reads_xy[rid]
        for mname, mfunc, mkw in math_methods:
            try:
                res = mfunc(bins, **mkw)
            except Exception as exc:
                empty = pd.DataFrame(columns=["chr","start","end","read_id"])
                res   = {"left_fork": empty, "right_fork": empty, "origin": empty}
                print(f"  [WARN] {mname} on {rid[:8]}: {exc}")
            math_preds[mname][rid] = res

    # ── Run AI inference ───────────────────────────────────────────────────────
    print(f"\nLoading AI model: {args.model_name} …")
    # Concat all reads into one dataframe for predict_reads
    all_xy = pd.concat([reads_xy[r] for r in read_ids], ignore_index=True)

    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(
        str(base / args.model), custom_objects=CUSTOM_OBJECTS)
    max_length = model.input_shape[1]
    print(f"  max_length={max_length}")

    preds_df = predict_reads(model, all_xy, read_ids, max_length,
                             config["preprocessing"])
    tf.keras.backend.clear_session()

    ai_preds = {}
    for rid in read_ids:
        ai_preds[rid] = {}
        for class_id, class_name in [(1, "left_fork"), (2, "right_fork"), (3, "origin")]:
            ev = windows_to_events(preds_df, class_id, args.threshold,
                                   min_windows=1, max_gap=args.max_gap)
            ai_preds[rid][class_name] = ev[ev["read_id"] == rid]

    # ── Build figure ───────────────────────────────────────────────────────────
    n_reads  = len(read_ids)
    n_math   = len(math_methods)
    # rows: signal | GT | AI | math1 | math2 | math3
    n_rows = 2 + 1 + n_math
    height_ratios = [3.5] + [0.85] * (n_rows - 1)

    fig, axes = plt.subplots(
        n_rows, n_reads,
        figsize=(6.5 * n_reads, 2.5 * n_rows),
        squeeze=False,
        gridspec_kw={"hspace": 0.08, "wspace": 0.04, "height_ratios": height_ratios},
    )

    for ci, rid in enumerate(read_ids):
        bins  = reads_xy[rid]
        x_sig = bins["start"].values
        y_sig = bins["signal"].values

        # ── Panel 0: signal ──────────────────────────────────────────────────
        ax = axes[0][ci]
        ax.step(x_sig, y_sig, where="post", color="black", linewidth=0.9)
        ax.fill_between(x_sig, y_sig, step="post", color="gray", alpha=0.10)
        for df, col in [(gt_left, COL_LEFT), (gt_right, COL_RIGHT), (gt_ori, COL_ORI)]:
            for row in df[df["read_id"] == rid].itertuples(index=False):
                ax.axvspan(int(row.start), int(row.end), color=col, alpha=0.20)
        ax.set_title(f"{rid[:12]}…", fontsize=8.5, fontweight="bold")
        ax.set_xticks([]); ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)
        if ci == 0:
            ax.set_ylabel("BrdU\nsignal", fontsize=8, va="center")
            handles = [
                mpatches.Patch(color=COL_LEFT,  alpha=0.5, label="GT left"),
                mpatches.Patch(color=COL_RIGHT, alpha=0.5, label="GT right"),
                mpatches.Patch(color=COL_ORI,   alpha=0.5, label="GT ORI"),
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=3,
                      framealpha=0.7)

        # ── Panel 1: GT bar ──────────────────────────────────────────────────
        ax = axes[1][ci]
        ax.set_xlim(x_sig[0], x_sig[-1])
        draw_annotation_bar(ax, rid, gt_left, gt_right, gt_ori,
                             label="Ground\ntruth" if ci == 0 else None)
        if ci == 0:
            _add_bar_legend(ax, COL_ORI, "ORI (GT)")

        # ── Panel 2: AI bar ──────────────────────────────────────────────────
        ax = axes[2][ci]
        ax.set_xlim(x_sig[0], x_sig[-1])
        ai_l = ai_preds[rid].get("left_fork",  pd.DataFrame())
        ai_r = ai_preds[rid].get("right_fork", pd.DataFrame())
        ai_o = ai_preds[rid].get("origin",     pd.DataFrame())
        draw_annotation_bar(ax, rid, ai_l, ai_r, ai_o,
                             ori_color=ORI_COLS[0],
                             label=args.model_name if ci == 0 else None)
        if ci == 0:
            _add_bar_legend(ax, ORI_COLS[0], "ORI (AI)")

        # ── Panels 3+: math bars ─────────────────────────────────────────────
        for mi, (mname, _, _) in enumerate(math_methods):
            ax = axes[3 + mi][ci]
            ax.set_xlim(x_sig[0], x_sig[-1])
            res = math_preds[mname].get(rid, {})
            m_l = res.get("left_fork",  pd.DataFrame(columns=["chr","start","end","read_id"]))
            m_r = res.get("right_fork", pd.DataFrame(columns=["chr","start","end","read_id"]))
            m_o = res.get("origin",     pd.DataFrame(columns=["chr","start","end","read_id"]))
            oc  = ORI_COLS[(mi + 1) % len(ORI_COLS)]
            draw_annotation_bar(ax, rid, m_l, m_r, m_o,
                                 ori_color=oc,
                                 label=mname if ci == 0 else None)
            if ci == 0:
                _add_bar_legend(ax, oc, "ORI")
            # x-axis on last row only
            if mi == n_math - 1:
                ax.tick_params(axis="x", labelsize=6)
                ax.set_xlabel("Position (bp)", fontsize=7)
            else:
                ax.set_xticks([])

    # Global legend (classes)
    ghandles = [
        mpatches.Patch(color=COL_LEFT,    label="Left fork"),
        mpatches.Patch(color=COL_RIGHT,   label="Right fork"),
        mpatches.Patch(color=COL_ORI,     label="ORI (GT / v2 GradPeak)"),
        mpatches.Patch(color=ORI_COLS[0], label=f"ORI ({args.model_name})"),
        mpatches.Patch(color=ORI_COLS[1], label="ORI (v3 LoG)"),
        mpatches.Patch(color=ORI_COLS[2], label="ORI (v3 Viterbi)"),
    ]
    fig.legend(handles=ghandles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.03),
               framealpha=0.85)

    fig.suptitle(
        f"AI vs Mathematical methods  |  threshold={args.threshold}  |  "
        f"rows: GT | {args.model_name} | v2 GradPeak | v3 LoG | v3 Viterbi",
        fontsize=10, fontweight="bold",
    )

    out = base / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def _add_bar_legend(ax, ori_color, ori_label):
    handles = [
        mpatches.Patch(color=COL_LEFT,  label="L"),
        mpatches.Patch(color=COL_RIGHT, label="R"),
        mpatches.Patch(color=ori_color, label=ori_label),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, fontsize=6,
              framealpha=0.65, handlelength=0.8, handletextpad=0.4,
              borderpad=0.3, columnspacing=0.5)


if __name__ == "__main__":
    main()
