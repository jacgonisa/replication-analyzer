#!/usr/bin/env python
"""Mathematical replication event detector v2.

Improvement over benchmark_mathematical_methods.py:
  The original methods used gradient thresholding with a fixed absolute threshold,
  and inferred ORIs indirectly (between a left-fork end and right-fork start).
  Problems:
    - Fixed threshold misses forks with gradual slopes
    - ORI positions are wrong when forks are fragmented or mis-detected
    - No structural reasoning about read topology

This pipeline keeps gradient-sign fork detection (which works at read boundaries)
but adds:
  1. Adaptive gradient threshold per read (based on signal amplitude)
  2. ORI placement at actual signal peaks (find_peaks), not inferred from fork pairs
  3. Multi-scale consensus for robustness

Signal structure:
  - Left fork  = RISING region (gradient > 0) → fork moving toward ORI from the left
  - Right fork = FALLING region (gradient < 0) → fork moving away from ORI to the right
  - Origin     = local MAXIMUM of BrdU signal (peak where replication initiated)

Two main tries:
  1. gradient_peak_single : single Gaussian smooth + adaptive gradient threshold + peak ORIs
  2. gradient_peak_multi  : three scales, position-level vote + peak ORIs from all scales

Plus two parameter variants ("standard" and "sensitive") for each.

Evaluation: IoU matching at IoU ≥ 0.2 (recall is primary metric).
Output: CODEX/results/mathematical_benchmark_v2/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, label as nd_label
from scipy.signal import find_peaks

# ── colours ───────────────────────────────────────────────────────────────────
COL_GT_LEFT  = "#1f77b4"
COL_GT_RIGHT = "#d62728"
COL_GT_ORI   = "#2ca02c"
METHOD_COLS  = ["#ff7f0e", "#9467bd", "#17becf", "#e377c2"]


# ── IoU evaluation ─────────────────────────────────────────────────────────────

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def evaluate_iou(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                 iou_threshold: float = 0.2) -> dict:
    if len(pred_df) == 0 and len(gt_df) == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0,
                    n_pred=0, n_gt=0)
    if len(pred_df) == 0:
        return dict(tp=0, fp=0, fn=len(gt_df), precision=0.0, recall=0.0, f1=0.0,
                    n_pred=0, n_gt=len(gt_df))
    if len(gt_df) == 0:
        return dict(tp=0, fp=len(pred_df), fn=0, precision=0.0, recall=0.0, f1=0.0,
                    n_pred=len(pred_df), n_gt=0)

    tp, fp = 0, 0
    gt_matched = set()
    gt_by_read = {rid: grp for rid, grp in gt_df.groupby("read_id")}

    for _, p in pred_df.iterrows():
        rid = p["read_id"]
        if rid not in gt_by_read:
            fp += 1
            continue
        gt_read = gt_by_read[rid]
        best_iou, best_idx = 0.0, None
        for idx, g in gt_read.iterrows():
            iou = compute_iou(int(p["start"]), int(p["end"]),
                              int(g["start"]), int(g["end"]))
            if iou > best_iou and idx not in gt_matched:
                best_iou, best_idx = iou, idx
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            gt_matched.add(best_idx)
        else:
            fp += 1

    fn   = len(gt_df) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_pred=len(pred_df), n_gt=len(gt_df))


# ── data loading ──────────────────────────────────────────────────────────────

def load_xy(base_dir: str, run_dirs: List[str],
            read_ids=None) -> Dict[str, pd.DataFrame]:
    reads = {}
    for run_dir in run_dirs:
        run_path = Path(base_dir) / run_dir
        if not run_path.exists():
            continue
        for f in run_path.glob("plot_data_*.txt"):
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
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


# ── signal utilities ──────────────────────────────────────────────────────────

def _expand_rectangular(bins_df: pd.DataFrame,
                         target_len: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Expand binned signal to rectangular step-function."""
    widths  = (bins_df["end"].values - bins_df["start"].values).astype(np.float64)
    signals = bins_df["signal"].values.astype(np.float32)
    total   = widths.sum()
    if total <= 0:
        return (np.zeros(target_len),
                np.linspace(float(bins_df["start"].iloc[0]),
                            float(bins_df["end"].iloc[-1]), target_len))
    cum     = np.cumsum(widths)
    fracs   = np.linspace(0, total, target_len)
    idx     = np.searchsorted(cum, fracs, side="right").clip(0, len(signals) - 1)
    gen_pos = (bins_df["start"].iloc[0]
               + fracs * (bins_df["end"].iloc[-1] - bins_df["start"].iloc[0]) / total)
    return signals[idx].astype(np.float64), gen_pos.astype(np.float64)


def _merge_to_events(mask: np.ndarray, pos: np.ndarray,
                      merge_gap_bp: int, min_len_bp: int) -> List[Tuple[float, float]]:
    """
    Convert a boolean mask (array units) to a list of (start_bp, end_bp) events.
    Adjacent True regions separated by <= merge_gap_bp are merged.
    Events shorter than min_len_bp are dropped.
    """
    if not mask.any():
        return []
    # dilate mask by merge gap
    gap_units = max(1, int(merge_gap_bp * len(pos) / (pos[-1] - pos[0] + 1)))
    # simple dilation: mark positions within gap_units of a True position
    dilated = np.zeros_like(mask, dtype=bool)
    true_idx = np.where(mask)[0]
    for i in true_idx:
        dilated[max(0, i - gap_units): i + gap_units + 1] = True
    # restrict back to original True regions plus gaps
    merged = dilated
    labeled, n = nd_label(merged)
    events = []
    for lab in range(1, n + 1):
        indices = np.where(labeled == lab)[0]
        # keep only if any original mask position is within this label
        if not mask[indices].any():
            continue
        s_bp = float(pos[indices[0]])
        e_bp = float(pos[indices[-1]])
        if e_bp - s_bp >= min_len_bp:
            events.append((s_bp, e_bp))
    return events


def _origins_from_forks(left_df: pd.DataFrame, right_df: pd.DataFrame,
                         read_id: str, chrom: str,
                         max_dist: int = 150_000) -> pd.DataFrame:
    """
    Infer ORI regions as the gap between a left-fork end and the next right-fork start.
    This is the biologically motivated approach: the ORI is between the point where
    the rising slope ends (left-fork end) and the point where the falling slope begins
    (right-fork start).  Pairs are constrained to be within max_dist bp.
    """
    if len(left_df) == 0 or len(right_df) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    origins = []
    for _, L in left_df.sort_values("end").iterrows():
        candidates = right_df[right_df["start"] >= L["end"]]
        if len(candidates) == 0:
            continue
        R = candidates.iloc[0]
        if R["start"] - L["end"] <= max_dist:
            origins.append({"chr": chrom,
                             "start": int(L["end"]),
                             "end":   int(R["start"]),
                             "read_id": read_id})
    return pd.DataFrame(origins) if origins else pd.DataFrame(
        columns=["chr", "start", "end", "read_id"])


def _segs_to_df(segs, chrom, read_id) -> pd.DataFrame:
    if not segs:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.DataFrame([{"chr": chrom, "start": int(s), "end": int(e),
                           "read_id": read_id} for s, e in segs])


# ── Core detection logic ───────────────────────────────────────────────────────

def _detect_at_scale(sig_n: np.ndarray, pos: np.ndarray, read_len: float,
                      smooth_sigma_kb: float,
                      grad_rel_threshold: float,
                      ori_prominence: float,
                      ori_half_width_bp: int,
                      merge_gap_bp: int,
                      min_fork_len_bp: int,
                      min_ori_len_bp: int,
                      target_len: int) -> Tuple[List, List, List, np.ndarray]:
    """
    Run gradient-based fork detection + peak-based ORI detection at one scale.

    Fork detection:
      - Smooth signal at smooth_sigma_kb
      - Compute gradient
      - Left fork:  positions where gradient > adaptive_threshold
      - Right fork: positions where gradient < -adaptive_threshold
      - Adaptive threshold = grad_rel_threshold * (signal amplitude / read_len_factor)
        This makes threshold proportional to signal amplitude → robust across reads

    ORI detection:
      - Find local maxima (peaks) in the smoothed signal
      - Minimum prominence = ori_prominence (in z-score units)
      - ORI region = peak ± ori_half_width_bp

    Returns: (left_events, right_events, ori_events, smoothed)
    """
    sigma    = smooth_sigma_kb * 1000 * target_len / read_len
    smoothed = gaussian_filter1d(sig_n, sigma=max(1.0, sigma))

    # Gradient + adaptive threshold
    # Threshold relative to gradient magnitude (NOT signal amplitude).
    # grad_rel_threshold is a multiplier on the 95th percentile of |gradient|.
    # E.g. grad_rel_threshold=0.25 → detect where |gradient| > 25% of max observed gradient.
    grad = np.gradient(smoothed)
    grad_p95 = np.percentile(np.abs(grad), 95)
    threshold = grad_rel_threshold * max(grad_p95, 1e-9)

    left_mask  = grad >  threshold
    right_mask = grad < -threshold

    left_events  = _merge_to_events(left_mask,  pos, merge_gap_bp, min_fork_len_bp)
    right_events = _merge_to_events(right_mask, pos, merge_gap_bp, min_fork_len_bp)

    # Peak-based ORI detection
    hw_units  = max(1, int(ori_half_width_bp * target_len / read_len))
    width_min = max(1, int(1000 * target_len / read_len))
    peaks, _  = find_peaks(smoothed, prominence=ori_prominence, width=width_min)

    ori_events = []
    for pk in peaks:
        s_bp = max(float(pos[0]),  float(pos[pk]) - ori_half_width_bp)
        e_bp = min(float(pos[-1]), float(pos[pk]) + ori_half_width_bp)
        if e_bp - s_bp >= min_ori_len_bp:
            ori_events.append((s_bp, e_bp))

    return left_events, right_events, ori_events, smoothed


# ── METHOD 1: Gradient + peak ORIs, single scale ─────────────────────────────

def gradient_peak_single(bins_df: pd.DataFrame,
                          smooth_sigma_kb: float = 5.0,
                          grad_rel_threshold: float = 0.06,
                          ori_prominence: float = 0.25,
                          ori_half_width_bp: int = 5000,
                          merge_gap_bp: int = 8000,
                          min_fork_len_bp: int = 3000,
                          target_len: int = 10000) -> dict:
    """
    Single-scale gradient + peak-based ORI detection.

    Improvements over the original Gaussian gradient method:
      1. Adaptive gradient threshold (relative to signal amplitude) → works for
         both high-amplitude and low-amplitude signals.
      2. ORIs placed at actual signal peaks (find_peaks) with a prominence filter,
         NOT inferred indirectly from fork-pair proximity.
      3. np.gradient instead of np.diff → smoother, symmetric derivative estimate.
    """
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id  = bins_df["read_id"].iloc[0]
    chrom    = bins_df["chr"].iloc[0]
    read_len = float(pos[-1] - pos[0] + 1)
    sig_n    = (sig - sig.mean()) / (sig.std() + 1e-8)

    left_ev, right_ev, _, smoothed = _detect_at_scale(
        sig_n, pos, read_len,
        smooth_sigma_kb=smooth_sigma_kb,
        grad_rel_threshold=grad_rel_threshold,
        ori_prominence=ori_prominence,
        ori_half_width_bp=ori_half_width_bp,
        merge_gap_bp=merge_gap_bp,
        min_fork_len_bp=min_fork_len_bp,
        min_ori_len_bp=1000,
        target_len=target_len,
    )

    left_df  = _segs_to_df(left_ev,  chrom, read_id)
    right_df = _segs_to_df(right_ev, chrom, read_id)
    ori_df   = _origins_from_forks(left_df, right_df, read_id, chrom)

    return {
        "left_fork":  left_df,
        "right_fork": right_df,
        "origin":     ori_df,
        "smoothed": smoothed, "pos": pos,
        "method": f"GradPeak-single (σ={smooth_sigma_kb}kb)",
    }


# ── METHOD 2: Multi-scale gradient + peak ORIs with consensus ─────────────────

def gradient_peak_multi(bins_df: pd.DataFrame,
                         smooth_scales_kb: Tuple[float, ...] = (2.0, 5.0, 10.0),
                         grad_rel_threshold: float = 0.06,
                         ori_prominence: float = 0.20,
                         ori_half_width_bp: int = 5000,
                         merge_gap_bp: int = 8000,
                         min_fork_len_bp: int = 2000,
                         min_votes: int = 2,
                         target_len: int = 10000) -> dict:
    """
    Multi-scale gradient + peak-based ORI detection with consensus voting.

    Algorithm:
      1. Run gradient_peak detection at each scale (2, 5, 10 kb smoothing).
      2. Build position-level vote maps for left_fork, right_fork, origin.
      3. Keep positions voted by >= min_votes scales.
      4. Find connected components → consensus events.

    Benefits:
      - Events that appear consistently across scales are more reliable.
      - Noisy single-scale detections are suppressed.
      - ORIs detected as peaks at any prominent scale.
    """
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id  = bins_df["read_id"].iloc[0]
    chrom    = bins_df["chr"].iloc[0]
    read_len = float(pos[-1] - pos[0] + 1)
    sig_n    = (sig - sig.mean()) / (sig.std() + 1e-8)

    left_votes  = np.zeros(target_len, dtype=np.int16)
    right_votes = np.zeros(target_len, dtype=np.int16)
    ori_votes   = np.zeros(target_len, dtype=np.int16)
    smoothed_mid = None

    for si, scale_kb in enumerate(smooth_scales_kb):
        left_ev, right_ev, ori_ev, smoothed = _detect_at_scale(
            sig_n, pos, read_len,
            smooth_sigma_kb=scale_kb,
            grad_rel_threshold=grad_rel_threshold,
            ori_prominence=ori_prominence,
            ori_half_width_bp=ori_half_width_bp,
            merge_gap_bp=merge_gap_bp,
            min_fork_len_bp=min_fork_len_bp,
            min_ori_len_bp=1000,
            target_len=target_len,
        )
        if si == len(smooth_scales_kb) // 2:
            smoothed_mid = smoothed

        # Build vote maps
        for s_bp, e_bp in left_ev:
            si_u = int(np.searchsorted(pos, s_bp))
            ei_u = int(np.searchsorted(pos, e_bp))
            left_votes[si_u:ei_u + 1] += 1

        for s_bp, e_bp in right_ev:
            si_u = int(np.searchsorted(pos, s_bp))
            ei_u = int(np.searchsorted(pos, e_bp))
            right_votes[si_u:ei_u + 1] += 1

        hw_u = max(1, int(ori_half_width_bp * target_len / read_len))
        for s_bp, e_bp in ori_ev:
            si_u = max(0, int(np.searchsorted(pos, s_bp)))
            ei_u = min(target_len - 1, int(np.searchsorted(pos, e_bp)))
            ori_votes[si_u:ei_u + 1] += 1

    # Consensus events
    def _vote_to_events(votes, min_len_bp):
        labeled, n = nd_label(votes >= min_votes)
        evs = []
        for lab in range(1, n + 1):
            idxs = np.where(labeled == lab)[0]
            s_bp = float(pos[idxs[0]])
            e_bp = float(pos[idxs[-1]])
            if e_bp - s_bp >= min_len_bp:
                evs.append((s_bp, e_bp))
        return evs

    left_ev  = _vote_to_events(left_votes,  min_fork_len_bp)
    right_ev = _vote_to_events(right_votes, min_fork_len_bp)

    left_df  = _segs_to_df(left_ev,  chrom, read_id)
    right_df = _segs_to_df(right_ev, chrom, read_id)
    ori_df   = _origins_from_forks(left_df, right_df, read_id, chrom)

    return {
        "left_fork":  left_df,
        "right_fork": right_df,
        "origin":     ori_df,
        "smoothed": smoothed_mid, "pos": pos,
        "method": f"GradPeak-multi (scales={smooth_scales_kb}, vote>={min_votes})",
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_read_examples(reads_data, gt_left, gt_right, gt_ori,
                        method_results, output_path, n_reads=4):
    example_ids = list(reads_data.keys())[:n_reads]
    n_methods   = len(method_results)
    n_rows      = 2 + n_methods

    fig, axes = plt.subplots(n_rows, len(example_ids),
                              figsize=(6 * len(example_ids), 2.8 * n_rows),
                              squeeze=False,
                              gridspec_kw={"hspace": 0.06, "wspace": 0.04})

    for ci, read_id in enumerate(example_ids):
        bins = reads_data[read_id]
        x    = bins["start"].values
        y    = bins["signal"].values

        ax = axes[0][ci]
        ax.step(x, y, where="post", color="black", linewidth=0.9)
        ax.fill_between(x, y, step="post", alpha=0.08, color="gray")
        for row in gt_left[gt_left["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), color=COL_GT_LEFT,  alpha=0.22)
        for row in gt_right[gt_right["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), color=COL_GT_RIGHT, alpha=0.22)
        for row in gt_ori[gt_ori["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), color=COL_GT_ORI,   alpha=0.30)
        ax.set_title(f"{read_id[:10]}…", fontsize=8, fontweight="bold")
        ax.set_ylabel("BrdU signal" if ci == 0 else "", fontsize=7)
        ax.set_xticks([])
        ax.tick_params(labelsize=6)
        if ci == 0:
            handles = [mpatches.Patch(color=COL_GT_LEFT,  alpha=0.5, label="GT left"),
                       mpatches.Patch(color=COL_GT_RIGHT, alpha=0.5, label="GT right"),
                       mpatches.Patch(color=COL_GT_ORI,   alpha=0.5, label="GT ORI")]
            ax.legend(handles=handles, fontsize=6, loc="upper right", ncol=3)

        ax = axes[1][ci]
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([])
        if ci == 0:
            ax.set_ylabel("GT", fontsize=7, rotation=0, labelpad=20, va="center")
        for row in gt_left[gt_left["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), ymin=0.67, ymax=0.99,
                       color=COL_GT_LEFT, alpha=0.8)
        for row in gt_right[gt_right["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), ymin=0.34, ymax=0.66,
                       color=COL_GT_RIGHT, alpha=0.8)
        for row in gt_ori[gt_ori["read_id"] == read_id].itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end), ymin=0.01, ymax=0.33,
                       color=COL_GT_ORI, alpha=0.8)
        ax.set_xlim(axes[0][ci].get_xlim())

        for mi, (mname, all_preds) in enumerate(method_results.items()):
            ax = axes[2 + mi][ci]
            ax.set_ylim(0, 1); ax.set_yticks([])
            col = METHOD_COLS[mi % len(METHOD_COLS)]
            if ci == 0:
                ax.set_ylabel(mname, fontsize=6, rotation=0, labelpad=42, va="center")
            rp = all_preds.get(read_id, {})
            for df_key, ymin, ymax, color in [
                ("left_fork",  0.67, 0.99, COL_GT_LEFT),
                ("right_fork", 0.34, 0.66, COL_GT_RIGHT),
                ("origin",     0.01, 0.33, col),
            ]:
                for row in rp.get(df_key, pd.DataFrame()).itertuples(index=False):
                    ax.axvspan(int(row.start), int(row.end), ymin=ymin, ymax=ymax,
                               color=color, alpha=0.70)
            if mi < n_methods - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Genomic position (bp)", fontsize=7)
                ax.tick_params(axis="x", labelsize=6)
            ax.set_xlim(axes[0][ci].get_xlim())

    handles2 = [mpatches.Patch(color=COL_GT_LEFT,  label="pred left"),
                mpatches.Patch(color=COL_GT_RIGHT, label="pred right")]
    for mi, mname in enumerate(method_results):
        handles2.append(mpatches.Patch(color=METHOD_COLS[mi], label=f"{mname} ORI"))
    fig.legend(handles=handles2, loc="lower center", ncol=min(6, len(handles2)),
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Mathematical pipeline v2: signal + GT + predictions", fontsize=11)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison_bars(summary_df: pd.DataFrame, ai_ref: dict,
                          output_path: Path, iou_threshold: float = 0.2):
    methods = summary_df["method"].unique().tolist()
    classes = ["left_fork", "right_fork", "origin"]
    sub     = summary_df[summary_df["iou_threshold"] == iou_threshold]
    x       = np.arange(len(classes))
    n       = len(methods)
    w       = 0.14
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, metric in zip(axes, ["recall", "f1"]):
        for i, method in enumerate(methods):
            row  = sub[sub["method"] == method]
            vals = [float(row[row["class"] == cl][metric].values[0])
                    if len(row[row["class"] == cl]) > 0 else 0.0
                    for cl in classes]
            ax.bar(x + offsets[i], vals, w * 0.88, label=method,
                   color=METHOD_COLS[i % len(METHOD_COLS)], alpha=0.82,
                   edgecolor="white", linewidth=0.5)

        for xi, cl in enumerate(classes):
            v = ai_ref.get(metric, {}).get(cl, 0)
            ax.hlines(v, xi - 0.44, xi + 0.44, colors="#222", linewidths=2.5,
                      linestyles="--", label="FORTE v1 (AI)" if xi == 0 else "")
            ax.text(xi, v + 0.012, f"AI:{v:.2f}", ha="center", fontsize=7.5,
                    fontweight="bold", color="#222")

        ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1)
        label = "Recall (primary metric)" if metric == "recall" else "F1"
        ax.set_ylabel(f"{label}  (IoU >= {iou_threshold})", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if metric == "recall":
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(f"Mathematical pipeline v2  vs  FORTE v1 (AI)  |  IoU >= {iou_threshold}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",
        default="/mnt/ssd-4tb/crisanto_project/data_2025Oct/"
                "data_reads_minLen30000_nascent40")
    parser.add_argument("--gt-left",
        default="data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed")
    parser.add_argument("--gt-right",
        default="data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed")
    parser.add_argument("--gt-ori",
        default="data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed")
    parser.add_argument("--output-dir",
        default="CODEX/results/mathematical_benchmark_v2")
    parser.add_argument("--max-reads", type=int, default=400)
    parser.add_argument("--iou-primary", type=float, default=0.2)
    parser.add_argument("--example-reads", nargs="+", default=[
        "05ac4325-04b3-4cb9-b59d-3bd26d1042ca",
        "c0e60f34-fb83-4315-b2ed-40842e85171e",
        "c4dfc355-98e8-4fca-83f7-d666f17a4eb1",
        "d8f2ca7c-bd7b-4c40-9a7d-17c6737664d7",
    ])
    # AI reference at IoU >= 0.2
    parser.add_argument("--ai-recall-left",  type=float, default=0.509)
    parser.add_argument("--ai-recall-right", type=float, default=0.702)
    parser.add_argument("--ai-recall-ori",   type=float, default=0.604)
    parser.add_argument("--ai-f1-left",      type=float, default=0.295)
    parser.add_argument("--ai-f1-right",     type=float, default=0.325)
    parser.add_argument("--ai-f1-ori",       type=float, default=0.628)
    args = parser.parse_args()

    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")

    run_dirs = [
        "NM30_Col0/NM30_plot_data_1strun_xy",
        "NM30_Col0/NM30_plot_data_2ndrun_xy",
        "NM31_orc1b2/NM31_plot_data_1strun_xy",
        "NM31_orc1b2/NM31_plot_data_2ndrun_xy",
    ]

    # Load GT
    gt_left  = load_bed4(str(base / args.gt_left))
    gt_right = load_bed4(str(base / args.gt_right))
    gt_ori   = load_bed4(str(base / args.gt_ori))
    print(f"GT: {len(gt_left):,} left | {len(gt_right):,} right | {len(gt_ori):,} ori")

    annotated_ids = set(gt_left["read_id"]) | set(gt_right["read_id"]) | set(gt_ori["read_id"])
    all_ids = list(set(list(annotated_ids)[:args.max_reads]) | set(args.example_reads))

    print(f"Loading XY for up to {args.max_reads:,} annotated reads…")
    reads_data = load_xy(args.base_dir, run_dirs, read_ids=all_ids)
    print(f"  Loaded: {len(reads_data):,}")

    loaded_ids    = set(reads_data.keys())
    gt_left_eval  = gt_left[gt_left["read_id"].isin(loaded_ids)]
    gt_right_eval = gt_right[gt_right["read_id"].isin(loaded_ids)]
    gt_ori_eval   = gt_ori[gt_ori["read_id"].isin(loaded_ids)]
    print(f"GT (eval): {len(gt_left_eval):,} left | {len(gt_right_eval):,} right | {len(gt_ori_eval):,} ori")

    # ── Method configs ────────────────────────────────────────────────────────
    # Try 1: standard (σ=5kb, moderate sensitivity)
    # Try 2: sensitive (σ=3kb, lower threshold → higher recall)
    # Try 3: multi-scale (2/5/10 kb, consensus≥2)
    # Try 4: multi-scale sensitive (1/3/7 kb, lower threshold)
    methods_cfg = [
        # grad_rel_threshold = fraction of p95(|gradient|) to use as threshold
        # 0.30 = detect where gradient is in the top ~5-30% of observed magnitudes (standard)
        # 0.15 = more sensitive, catches gradual slopes too
        # GT ORI median size = 1,180 bp → use ori_half_width_bp ~ 600-750 bp
        # so prediction window (1,200-1,500 bp) ≈ median GT ORI → IoU >= 0.2 when peak is inside GT
        ("GradPeak\nstandard", gradient_peak_single,
         dict(smooth_sigma_kb=5.0, grad_rel_threshold=0.30,
              ori_prominence=0.25, ori_half_width_bp=700,
              merge_gap_bp=8000, min_fork_len_bp=3000)),

        ("GradPeak\nsensitive", gradient_peak_single,
         dict(smooth_sigma_kb=3.0, grad_rel_threshold=0.15,
              ori_prominence=0.15, ori_half_width_bp=700,
              merge_gap_bp=10000, min_fork_len_bp=2000)),

        ("GradPeak\nmulti(2/5/10)", gradient_peak_multi,
         dict(smooth_scales_kb=(2.0, 5.0, 10.0), grad_rel_threshold=0.30,
              ori_prominence=0.20, ori_half_width_bp=700,
              merge_gap_bp=8000, min_fork_len_bp=2000, min_votes=2)),

        ("GradPeak\nmulti-sens", gradient_peak_multi,
         dict(smooth_scales_kb=(1.5, 3.0, 7.0), grad_rel_threshold=0.15,
              ori_prominence=0.15, ori_half_width_bp=700,
              merge_gap_bp=10000, min_fork_len_bp=1500, min_votes=2)),
    ]

    per_read_preds = {m[0]: {} for m in methods_cfg}
    all_preds      = {m[0]: {"left_fork": [], "right_fork": [], "origin": []}
                      for m in methods_cfg}

    print(f"\nRunning {len(methods_cfg)} methods on {len(reads_data):,} reads…")
    for i, (rid, bins) in enumerate(reads_data.items()):
        if i % 50 == 0:
            print(f"  {i}/{len(reads_data)}")
        for mname, mfunc, mkw in methods_cfg:
            try:
                res = mfunc(bins, **mkw)
            except Exception as exc:
                empty = pd.DataFrame(columns=["chr", "start", "end", "read_id"])
                res   = {"left_fork": empty, "right_fork": empty, "origin": empty}
                print(f"  [WARN] {mname} on {rid}: {exc}")
            per_read_preds[mname][rid] = res
            for cl in ["left_fork", "right_fork", "origin"]:
                if len(res[cl]) > 0:
                    all_preds[mname][cl].append(res[cl])

    combined_preds = {}
    for mname in [m[0] for m in methods_cfg]:
        combined_preds[mname] = {}
        for cl in ["left_fork", "right_fork", "origin"]:
            dfs = all_preds[mname][cl]
            combined_preds[mname][cl] = (
                pd.concat(dfs, ignore_index=True) if dfs
                else pd.DataFrame(columns=["chr", "start", "end", "read_id"])
            )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    iou_thresholds = [0.2, 0.3, 0.4, 0.5]
    gt_map = {"left_fork": gt_left_eval, "right_fork": gt_right_eval,
              "origin": gt_ori_eval}

    print(f"\nIoU evaluation (primary: IoU >= {args.iou_primary}):")
    print(f"{'Method':<30} {'Class':<12} {'Recall':>7} {'Prec':>7} {'F1':>7} {'#pred':>6}")
    print("-" * 72)
    rows = []
    for mname in [m[0] for m in methods_cfg]:
        for iou_thr in iou_thresholds:
            for cl in ["left_fork", "right_fork", "origin"]:
                m = evaluate_iou(combined_preds[mname][cl], gt_map[cl], iou_thr)
                rows.append({"method": mname.replace("\n", " "),
                              "class": cl, "iou_threshold": iou_thr, **m})
                if iou_thr == args.iou_primary:
                    print(f"  {mname.replace(chr(10),' '):<28} {cl:<12} "
                          f"{m['recall']:>7.3f} {m['precision']:>7.3f} "
                          f"{m['f1']:>7.3f} {int(m['n_pred']):>6,}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "summary_table.tsv", sep="\t", index=False)
    print(f"\nSaved: {out / 'summary_table.tsv'}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    ai_ref = {
        "recall": {"left_fork": args.ai_recall_left, "right_fork": args.ai_recall_right,
                   "origin": args.ai_recall_ori},
        "f1":     {"left_fork": args.ai_f1_left, "right_fork": args.ai_f1_right,
                   "origin": args.ai_f1_ori},
    }

    plot_comparison_bars(summary_df, ai_ref, out / "comparison_bar.png",
                         iou_threshold=args.iou_primary)

    example_data = {rid: reads_data[rid]
                    for rid in args.example_reads if rid in reads_data}
    plot_read_examples(example_data, gt_left, gt_right, gt_ori,
                       per_read_preds, out / "read_examples.png", n_reads=4)

    print(f"\nAll outputs: {out}")


if __name__ == "__main__":
    main()
