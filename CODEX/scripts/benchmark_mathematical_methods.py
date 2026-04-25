#!/usr/bin/env python
"""Benchmark mathematical (non-AI) signal processing methods for fork/origin detection.

Evaluation: IoU matching at multiple thresholds — identical to AI model evaluation.
Methods tested:
  1. Gaussian gradient      — smooth + differentiate, detect rising/falling edges
  2. Multiscale Gaussian    — vote across multiple bandwidths
  3. Wavelet gradient       — wavelet approximation + differentiate
  4. High-signal threshold  — just threshold the raw BrdU signal (naive baseline)

Output:
  - summary_table.tsv       — IoU F1/precision/recall per method × class
  - comparison_bar.png      — grouped bar chart vs AI reference
  - read_examples.png       — 4 reads showing signal + GT + each method's detections
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

# ── colours ───────────────────────────────────────────────────────────────────
COL_GT_LEFT  = "#1f77b4"
COL_GT_RIGHT = "#d62728"
COL_GT_ORI   = "#2ca02c"
METHOD_COLS  = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
METHOD_NAMES = ["Gaussian\ngradient", "Multiscale\nGaussian", "Wavelet\ngradient",
                "Raw signal\nthreshold"]


# ── IoU evaluation ────────────────────────────────────────────────────────────

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def evaluate_iou(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                 iou_threshold: float = 0.3) -> dict:
    """Evaluate predicted regions against ground truth using IoU matching."""
    if len(pred_df) == 0 and len(gt_df) == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0,
                    n_pred=0, n_gt=0)
    if len(pred_df) == 0:
        return dict(tp=0, fp=0, fn=len(gt_df), precision=0.0, recall=0.0, f1=0.0,
                    n_pred=0, n_gt=len(gt_df))
    if len(gt_df) == 0:
        return dict(tp=0, fp=len(pred_df), fn=0, precision=0.0, recall=0.0, f1=0.0,
                    n_pred=len(pred_df), n_gt=0)

    tp, fp, fn = 0, 0, 0
    gt_matched = set()

    # group by read_id for speed
    gt_by_read = {rid: grp for rid, grp in gt_df.groupby("read_id")}

    for _, p in pred_df.iterrows():
        rid = p["read_id"]
        if rid not in gt_by_read:
            fp += 1
            continue
        gt_read = gt_by_read[rid]
        best_iou = max(
            compute_iou(int(p["start"]), int(p["end"]), int(g["start"]), int(g["end"]))
            for _, g in gt_read.iterrows()
        )
        matched_idx = None
        for idx, g in gt_read.iterrows():
            iou = compute_iou(int(p["start"]), int(p["end"]), int(g["start"]), int(g["end"]))
            if iou >= iou_threshold and iou == best_iou and idx not in gt_matched:
                matched_idx = idx
                break
        if matched_idx is not None:
            tp += 1
            gt_matched.add(matched_idx)
        else:
            fp += 1

    fn = len(gt_df) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_pred=len(pred_df), n_gt=len(gt_df))


# ── XY data loading ───────────────────────────────────────────────────────────

def load_xy(base_dir: str, run_dirs: List[str],
            read_ids: List[str] | None = None) -> Dict[str, pd.DataFrame]:
    reads = {}
    for run_dir in run_dirs:
        run_path = Path(base_dir) / run_dir
        if not run_path.exists():
            continue
        files = list(run_path.glob("plot_data_*.txt"))
        for f in files:
            rid = f.stem.replace("plot_data_", "")
            if read_ids is not None and rid not in read_ids:
                continue
            if rid not in reads:
                try:
                    df = pd.read_csv(f, sep="\t", header=None,
                                     names=["chr", "start", "end", "signal"])
                    df["read_id"] = rid
                    reads[rid] = df
                except Exception:
                    pass
    return reads


def load_bed4(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                     names=["chr", "start", "end", "read_id"])
    return df


# ── Signal processing methods ─────────────────────────────────────────────────

def _expand_rectangular(bins_df: pd.DataFrame, target_len: int = 10000
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Expand binned signal to rectangular step-function (same as encoding_rectangular.py)."""
    widths = (bins_df["end"].values - bins_df["start"].values).astype(np.float64)
    signals = bins_df["signal"].values.astype(np.float32)
    total = widths.sum()
    if total <= 0:
        return np.zeros(target_len), np.linspace(bins_df["start"].iloc[0],
                                                  bins_df["end"].iloc[-1], target_len)
    cum = np.cumsum(widths)
    positions = np.linspace(0, total, target_len)
    idx = np.searchsorted(cum, positions, side="right").clip(0, len(signals) - 1)
    # Convert positions back to genomic coordinates
    genomic_positions = (bins_df["start"].iloc[0]
                         + positions * (bins_df["end"].iloc[-1] - bins_df["start"].iloc[0]) / total)
    return signals[idx].astype(np.float32), genomic_positions.astype(np.float32)


def _merge_positions_to_events(positions: np.ndarray, read_id: str, chrom: str,
                                merge_gap: int = 5000, min_len: int = 500) -> pd.DataFrame:
    """Merge nearby genomic positions into event regions."""
    if len(positions) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    positions = np.sort(positions)
    events = []
    s, e = positions[0], positions[0]
    for p in positions[1:]:
        if p - e <= merge_gap:
            e = p
        else:
            if e - s >= min_len:
                events.append({"chr": chrom, "start": int(s), "end": int(e), "read_id": read_id})
            s, e = p, p
    if e - s >= min_len:
        events.append({"chr": chrom, "start": int(s), "end": int(e), "read_id": read_id})
    return pd.DataFrame(events) if events else pd.DataFrame(columns=["chr", "start", "end", "read_id"])


def _origins_from_forks(left_df, right_df, read_id, chrom, max_dist=80000):
    """Call origin = region between a left-fork end and the next right-fork start."""
    if len(left_df) == 0 or len(right_df) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    origins = []
    for _, L in left_df.iterrows():
        candidates = right_df[right_df["start"] >= L["end"]]
        if len(candidates) == 0:
            continue
        R = candidates.iloc[0]
        if R["start"] - L["end"] <= max_dist:
            origins.append({"chr": chrom,
                             "start": int(L["end"]),
                             "end": int(R["start"]),
                             "read_id": read_id})
    return pd.DataFrame(origins) if origins else pd.DataFrame(columns=["chr", "start", "end", "read_id"])


def gaussian_gradient(bins_df: pd.DataFrame, bandwidth_kb: float = 5.0,
                      grad_threshold: float = 0.02, target_len: int = 8000) -> dict:
    """Method 1: smooth rectangular signal + detect rising/falling edges."""
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id = bins_df["read_id"].iloc[0]
    chrom   = bins_df["chr"].iloc[0]

    sigma = bandwidth_kb * 1000 * target_len / (pos[-1] - pos[0] + 1)
    smoothed = gaussian_filter1d(sig, sigma=max(1, sigma))
    grad = np.diff(smoothed, prepend=smoothed[0])

    left_pos  = pos[grad >  grad_threshold]
    right_pos = pos[grad < -grad_threshold]

    left_df  = _merge_positions_to_events(left_pos,  read_id, chrom)
    right_df = _merge_positions_to_events(right_pos, read_id, chrom)
    ori_df   = _origins_from_forks(left_df, right_df, read_id, chrom)

    return {"left_fork": left_df, "right_fork": right_df, "origin": ori_df,
            "smoothed": smoothed, "grad": grad, "pos": pos,
            "method": f"Gaussian grad (bw={bandwidth_kb}kb, thr={grad_threshold})"}


def multiscale_gaussian(bins_df: pd.DataFrame,
                        bandwidths_kb=(1.0, 3.0, 5.0, 10.0),
                        grad_threshold: float = 0.015,
                        vote_min: int = 2,
                        target_len: int = 8000) -> dict:
    """Method 2: vote across multiple smoothing scales."""
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id = bins_df["read_id"].iloc[0]
    chrom   = bins_df["chr"].iloc[0]
    read_len = pos[-1] - pos[0] + 1

    left_votes  = np.zeros(target_len, dtype=np.int16)
    right_votes = np.zeros(target_len, dtype=np.int16)

    for bw in bandwidths_kb:
        sigma = bw * 1000 * target_len / read_len
        smoothed = gaussian_filter1d(sig, sigma=max(1, sigma))
        grad = np.diff(smoothed, prepend=smoothed[0])
        left_votes[ grad >  grad_threshold] += 1
        right_votes[grad < -grad_threshold] += 1

    left_pos  = pos[left_votes  >= vote_min]
    right_pos = pos[right_votes >= vote_min]

    left_df  = _merge_positions_to_events(left_pos,  read_id, chrom)
    right_df = _merge_positions_to_events(right_pos, read_id, chrom)
    ori_df   = _origins_from_forks(left_df, right_df, read_id, chrom)

    # Return vote heatmap for visualization
    combined_votes = (left_votes.astype(np.float32) - right_votes.astype(np.float32))
    return {"left_fork": left_df, "right_fork": right_df, "origin": ori_df,
            "smoothed": combined_votes, "grad": combined_votes, "pos": pos,
            "method": f"Multiscale Gaussian (vote≥{vote_min})"}


def wavelet_gradient(bins_df: pd.DataFrame, wavelet: str = "db4", level: int = 3,
                     grad_threshold: float = 0.015, target_len: int = 8000) -> dict:
    """Method 3: wavelet approximation + gradient detection."""
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id = bins_df["read_id"].iloc[0]
    chrom   = bins_df["chr"].iloc[0]

    # Normalize
    sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)

    # Wavelet approximation (low-freq trend)
    coeffs = pywt.wavedec(sig_n, wavelet, level=level)
    approx = coeffs[0]
    approx_up = np.repeat(approx, len(sig_n) // len(approx) + 1)[:len(sig_n)]

    grad = np.diff(approx_up, prepend=approx_up[0])

    left_pos  = pos[grad >  grad_threshold]
    right_pos = pos[grad < -grad_threshold]

    left_df  = _merge_positions_to_events(left_pos,  read_id, chrom)
    right_df = _merge_positions_to_events(right_pos, read_id, chrom)
    ori_df   = _origins_from_forks(left_df, right_df, read_id, chrom)

    return {"left_fork": left_df, "right_fork": right_df, "origin": ori_df,
            "smoothed": approx_up, "grad": grad, "pos": pos,
            "method": f"Wavelet grad ({wavelet} L{level}, thr={grad_threshold})"}


def signal_threshold(bins_df: pd.DataFrame, low_thresh: float = -0.5,
                     high_thresh: float = 0.5, target_len: int = 8000) -> dict:
    """Method 4: naive — high signal → origin, low signal → fork region."""
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id = bins_df["read_id"].iloc[0]
    chrom   = bins_df["chr"].iloc[0]

    sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)

    # High BrdU signal → origin candidate
    ori_pos  = pos[sig_n >= high_thresh]
    # Low BrdU → low replication (flanks of origin)
    low_pos  = pos[sig_n <= low_thresh]

    ori_df   = _merge_positions_to_events(ori_pos, read_id, chrom)
    # Split low regions arbitrarily into left/right by position relative to ori
    # (this is the fundamental limitation of raw-signal methods: can't distinguish direction)
    mid = (bins_df["start"].iloc[0] + bins_df["end"].iloc[-1]) / 2
    left_pos  = pos[(sig_n <= low_thresh) & (pos < mid)]
    right_pos = pos[(sig_n <= low_thresh) & (pos >= mid)]

    left_df  = _merge_positions_to_events(left_pos,  read_id, chrom)
    right_df = _merge_positions_to_events(right_pos, read_id, chrom)

    return {"left_fork": left_df, "right_fork": right_df, "origin": ori_df,
            "smoothed": sig_n, "grad": sig_n, "pos": pos,
            "method": f"Raw threshold (lo={low_thresh}, hi={high_thresh})"}


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_read_examples(reads_data, gt_left, gt_right, gt_ori,
                       method_results, output_path, n_reads=4):
    """For each example read, show signal + GT + detections from each method."""
    example_ids = list(reads_data.keys())[:n_reads]
    n_methods = len(method_results)
    n_rows = 2 + n_methods   # signal row, GT row, one row per method

    fig, axes = plt.subplots(n_rows, len(example_ids),
                              figsize=(6 * len(example_ids), 2.5 * n_rows),
                              squeeze=False,
                              gridspec_kw={"hspace": 0.08, "wspace": 0.05})

    height_ratios_signal = [3] + [0.8] * (n_rows - 1)

    for ci, read_id in enumerate(example_ids):
        bins = reads_data[read_id]
        x = bins["start"].values
        y = bins["signal"].values

        # ── Row 0: signal ───────────────────────────────────────────────────
        ax = axes[0][ci]
        ax.step(x, y, where="post", color="black", linewidth=0.9)
        ax.fill_between(x, y, step="post", alpha=0.10, color="gray")

        # GT shading on signal
        for row in gt_left[gt_left["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), color=COL_GT_LEFT, alpha=0.18)
        for row in gt_right[gt_right["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), color=COL_GT_RIGHT, alpha=0.18)
        for row in gt_ori[gt_ori["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), color=COL_GT_ORI, alpha=0.25)

        ax.set_ylabel("BrdU signal" if ci == 0 else "", fontsize=7)
        ax.set_title(f"{read_id[:8]}…", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.set_xticks([])

        if ci == 0:
            handles = [
                mpatches.Patch(color=COL_GT_LEFT,  alpha=0.5, label="GT left-fork"),
                mpatches.Patch(color=COL_GT_RIGHT, alpha=0.5, label="GT right-fork"),
                mpatches.Patch(color=COL_GT_ORI,   alpha=0.5, label="GT origin"),
            ]
            ax.legend(handles=handles, fontsize=6, loc="upper right", ncol=3)

        # ── Row 1: ground truth bar ─────────────────────────────────────────
        ax = axes[1][ci]
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_xticks([])
        if ci == 0:
            ax.set_ylabel("Ground\ntruth", fontsize=7, rotation=0, labelpad=35, va="center")
        for row in gt_left[gt_left["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), ymin=0.66, ymax=0.98,
                       color=COL_GT_LEFT, alpha=0.7)
        for row in gt_right[gt_right["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), ymin=0.33, ymax=0.65,
                       color=COL_GT_RIGHT, alpha=0.7)
        for row in gt_ori[gt_ori["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), ymin=0.02, ymax=0.32,
                       color=COL_GT_ORI, alpha=0.7)
        ax.set_xlim(axes[0][ci].get_xlim())

        # ── Rows 2+: one per method ─────────────────────────────────────────
        for mi, (method_name, all_preds) in enumerate(method_results.items()):
            ax = axes[2 + mi][ci]
            ax.set_ylim(0, 1); ax.set_yticks([])
            col = METHOD_COLS[mi % len(METHOD_COLS)]
            if ci == 0:
                ax.set_ylabel(method_name, fontsize=6.5, rotation=0, labelpad=35, va="center")

            # Check if per-read result is available
            read_preds = all_preds.get(read_id, {})
            for df_key, ymin, ymax, color in [
                ("left_fork",  0.66, 0.98, COL_GT_LEFT),
                ("right_fork", 0.33, 0.65, COL_GT_RIGHT),
                ("origin",     0.02, 0.32, col),
            ]:
                df = read_preds.get(df_key, pd.DataFrame())
                for row in df.itertuples(index=False):
                    ax.axvspan(int(row.start), int(row.end), ymin=ymin, ymax=ymax,
                               color=color, alpha=0.65)

            if mi < len(method_results) - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Genomic position (bp)", fontsize=7)
                ax.tick_params(axis="x", labelsize=6)
            ax.set_xlim(axes[0][ci].get_xlim())

    # Legend for methods
    handles2 = [mpatches.Patch(color=COL_GT_LEFT,  label="pred left-fork"),
                mpatches.Patch(color=COL_GT_RIGHT, label="pred right-fork")]
    for mi, (mname, _) in enumerate(method_results.items()):
        handles2.append(mpatches.Patch(color=METHOD_COLS[mi], label=f"{mname} origin"))
    fig.legend(handles=handles2, loc="lower center", ncol=len(handles2),
               fontsize=7, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Mathematical methods: signal + ground truth + predictions", fontsize=11)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison_bar(summary_df: pd.DataFrame, ai_f1: dict, output_path: Path,
                        iou_threshold: float = 0.3):
    """Grouped bar chart: each method × class, with AI reference line."""
    methods = summary_df["method"].unique()
    classes = ["left_fork", "right_fork", "origin"]
    sub = summary_df[summary_df["iou_threshold"] == iou_threshold]

    x = np.arange(len(classes))
    width = 0.18
    n = len(methods)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        row = sub[sub["method"] == method]
        vals = [row[row["class"] == cl]["f1"].values[0]
                if len(row[row["class"] == cl]) > 0 else 0.0
                for cl in classes]
        ax.bar(x + offsets[i], vals, width * 0.9, label=method,
               color=METHOD_COLS[i % len(METHOD_COLS)], alpha=0.8, edgecolor="white")

    # AI reference lines
    ai_class_map = {"left_fork": "left_fork", "right_fork": "right_fork", "origin": "origin"}
    ref_labels = {"left_fork": ai_f1.get("left_fork", 0),
                  "right_fork": ai_f1.get("right_fork", 0),
                  "origin": ai_f1.get("origin", 0)}
    for xi, cl in enumerate(classes):
        v = ref_labels[cl]
        ax.hlines(v, xi - 0.45, xi + 0.45, colors="#333", linewidths=2.5,
                  linestyles="--", label="FORTE v1 (AI)" if xi == 0 else None)
        ax.text(xi, v + 0.015, f"AI: {v:.2f}", ha="center", fontsize=8,
                fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"F1  (IoU ≥ {iou_threshold})", fontsize=11)
    ax.set_title("Mathematical methods vs AI (FORTE v1)\nWindow-level IoU matching",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="YAML config (reads base_dir, run_dirs, GT BEDs)")
    parser.add_argument("--base-dir",
                        default="/mnt/ssd-4tb/crisanto_project/data_2025Oct/"
                                "data_reads_minLen30000_nascent40")
    parser.add_argument("--gt-left",
                        default="data/case_study_jan2026/combined/annotations/"
                                "leftForks_ALL_combined.bed")
    parser.add_argument("--gt-right",
                        default="data/case_study_jan2026/combined/annotations/"
                                "rightForks_ALL_combined.bed")
    parser.add_argument("--gt-ori",
                        default="data/case_study_jan2026/combined/annotations/"
                                "ORIs_combined_cleaned.bed")
    parser.add_argument("--output-dir",
                        default="CODEX/results/mathematical_benchmark")
    parser.add_argument("--max-reads", type=int, default=300,
                        help="Max reads to evaluate (annotation-overlapping only)")
    parser.add_argument("--example-reads", nargs="+",
                        default=["05ac4325-04b3-4cb9-b59d-3bd26d1042ca",
                                 "c0e60f34-fb83-4315-b2ed-40842e85171e",
                                 "c4dfc355-98e8-4fca-83f7-d666f17a4eb1",
                                 "d8f2ca7c-bd7b-4c40-9a7d-17c6737664d7"],
                        help="Read IDs to show in visual examples panel")
    parser.add_argument("--ai-f1-left",  type=float, default=0.35,
                        help="FORTE v1 val F1 for left_fork (reference line)")
    parser.add_argument("--ai-f1-right", type=float, default=0.45)
    parser.add_argument("--ai-f1-ori",   type=float, default=0.60)
    parser.add_argument("--iou-eval",    type=float, default=0.3,
                        help="IoU threshold for primary bar chart")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = [
        "NM30_Col0/NM30_plot_data_1strun_xy",
        "NM30_Col0/NM30_plot_data_2ndrun_xy",
        "NM31_orc1b2/NM31_plot_data_1strun_xy",
        "NM31_orc1b2/NM31_plot_data_2ndrun_xy",
    ]

    # ── Load GT ────────────────────────────────────────────────────────────────
    base = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")
    gt_left  = load_bed4(str(base / args.gt_left))
    gt_right = load_bed4(str(base / args.gt_right))
    gt_ori   = load_bed4(str(base / args.gt_ori))
    print(f"GT: {len(gt_left):,} left-forks | {len(gt_right):,} right-forks | {len(gt_ori):,} origins")

    # ── Pick annotated reads ──────────────────────────────────────────────────
    annotated_ids = (set(gt_left["read_id"]) | set(gt_right["read_id"])
                     | set(gt_ori["read_id"]))
    print(f"Annotated reads: {len(annotated_ids):,}")

    # Load all XY for annotated reads (up to max_reads)
    all_reads_to_load = list(annotated_ids)[:args.max_reads]
    example_ids = args.example_reads
    all_ids_needed = list(set(all_reads_to_load) | set(example_ids))

    print(f"Loading XY for {len(all_ids_needed):,} reads…")
    reads_data = load_xy(args.base_dir, run_dirs, read_ids=all_ids_needed)
    print(f"  Loaded: {len(reads_data):,} reads")

    # Filter GT to loaded reads
    loaded_ids = set(reads_data.keys())
    gt_left_eval  = gt_left[gt_left["read_id"].isin(loaded_ids)]
    gt_right_eval = gt_right[gt_right["read_id"].isin(loaded_ids)]
    gt_ori_eval   = gt_ori[gt_ori["read_id"].isin(loaded_ids)]
    print(f"GT after filtering: {len(gt_left_eval):,} left | {len(gt_right_eval):,} right | {len(gt_ori_eval):,} ori")

    # ── Run methods on all reads ───────────────────────────────────────────────
    # Method configs (best typical parameters)
    methods_cfg = [
        ("Gaussian gradient",   gaussian_gradient,
         {"bandwidth_kb": 5.0, "grad_threshold": 0.015}),
        ("Multiscale Gaussian", multiscale_gaussian,
         {"bandwidths_kb": (1.0, 3.0, 5.0, 10.0), "grad_threshold": 0.012, "vote_min": 2}),
        ("Wavelet gradient",    wavelet_gradient,
         {"wavelet": "db4", "level": 3, "grad_threshold": 0.012}),
        ("Raw signal threshold", signal_threshold,
         {"low_thresh": -0.4, "high_thresh": 0.5}),
    ]

    # per-read predictions: {method_name: {read_id: {class: df}}}
    per_read_preds = {m[0]: {} for m in methods_cfg}
    all_preds = {m[0]: {"left_fork": [], "right_fork": [], "origin": []}
                 for m in methods_cfg}

    print("\nRunning methods…")
    for i, (rid, bins) in enumerate(reads_data.items()):
        if i % 50 == 0:
            print(f"  {i}/{len(reads_data)}")
        for mname, mfunc, mkwargs in methods_cfg:
            try:
                res = mfunc(bins, **mkwargs)
            except Exception as e:
                res = {"left_fork": pd.DataFrame(columns=["chr","start","end","read_id"]),
                       "right_fork": pd.DataFrame(columns=["chr","start","end","read_id"]),
                       "origin": pd.DataFrame(columns=["chr","start","end","read_id"])}
            per_read_preds[mname][rid] = res
            for cl in ["left_fork", "right_fork", "origin"]:
                if len(res[cl]) > 0:
                    all_preds[mname][cl].append(res[cl])

    # Combine predictions across all reads
    combined_preds = {}
    for mname in [m[0] for m in methods_cfg]:
        combined_preds[mname] = {}
        for cl in ["left_fork", "right_fork", "origin"]:
            dfs = all_preds[mname][cl]
            combined_preds[mname][cl] = (
                pd.concat(dfs, ignore_index=True) if dfs
                else pd.DataFrame(columns=["chr", "start", "end", "read_id"])
            )

    # ── Evaluate with IoU matching ────────────────────────────────────────────
    iou_thresholds = [0.2, 0.3, 0.4, 0.5]
    gt_map = {"left_fork": gt_left_eval, "right_fork": gt_right_eval, "origin": gt_ori_eval}

    print("\nIoU evaluation:")
    rows = []
    for mname in [m[0] for m in methods_cfg]:
        for iou_thr in iou_thresholds:
            for cl in ["left_fork", "right_fork", "origin"]:
                pred_df = combined_preds[mname][cl]
                gt_df   = gt_map[cl]
                metrics = evaluate_iou(pred_df, gt_df, iou_threshold=iou_thr)
                rows.append({"method": mname, "class": cl, "iou_threshold": iou_thr,
                              **metrics})
                if iou_thr == args.iou_eval:
                    print(f"  {mname:25s} {cl:12s} IoU={iou_thr}  "
                          f"F1={metrics['f1']:.3f}  "
                          f"prec={metrics['precision']:.3f}  "
                          f"rec={metrics['recall']:.3f}  "
                          f"(pred={metrics['n_pred']:,} gt={metrics['n_gt']:,})")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "summary_table.tsv", sep="\t", index=False)
    print(f"\nSaved: {out / 'summary_table.tsv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")

    ai_f1 = {"left_fork": args.ai_f1_left, "right_fork": args.ai_f1_right,
              "origin": args.ai_f1_ori}

    short_names = {m[0]: m[0].split()[0] + "\n" + m[0].split()[1]
                   if len(m[0].split()) > 1 else m[0] for m in methods_cfg}
    plot_comparison_bar(summary_df, ai_f1, out / "comparison_bar.png",
                        iou_threshold=args.iou_eval)

    # Visual examples
    example_reads_data = {rid: reads_data[rid] for rid in example_ids if rid in reads_data}
    short_preds = {m[0].replace(" ", "\n"): per_read_preds[m[0]]
                   for m in methods_cfg}
    plot_read_examples(example_reads_data, gt_left, gt_right, gt_ori,
                       short_preds, out / "read_examples.png", n_reads=4)

    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
