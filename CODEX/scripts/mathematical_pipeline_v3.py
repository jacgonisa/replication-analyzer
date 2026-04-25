#!/usr/bin/env python
"""Mathematical replication event detector v3 — sequential context methods.

This module adds two methods that incorporate sequential/structural context
without any machine learning:

METHOD 1: Multiscale Laplacian-of-Gaussian (LoG) fork/ORI detection
  The LoG is the "blob detector" of signal processing.  At scale σ, the
  scale-normalized LoG (σ² · ∇²G_σ * signal) responds maximally to a peak
  (ORI-like region) whose half-width ≈ σ√2.  By sweeping scales [0.5–20 kb]
  we find ORIs at their natural resolution.  The sign change of the LoG
  (positive → negative going left to right) marks the left → ORI → right
  transition without needing to threshold a gradient.

METHOD 2: HMM with Viterbi decoding
  States:  {bg, left_fork, origin, right_fork}
  Emissions are computed per-read from (gradient, signal-level) features.
  Transitions encode biological constraints:
      bg ──→ left_fork ──→ origin ──→ right_fork ──→ bg
                                                     ↑
       bg ──────────────────────────────────────────(can restart cycle)
  The Viterbi algorithm finds the globally optimal state sequence given
  the emission likelihoods.

  Key advantage over gradient thresholding:
    - Forbids left_fork → right_fork without passing through origin
    - Forbids right_fork → left_fork without background (termination gap)
    - Explicitly models the plateau at origin (gradient ≈ 0, signal high)
    - Transition probabilities encode expected event durations

Both methods are evaluated at IoU ≥ 0.2 (recall is primary metric).
Output: CODEX/results/mathematical_benchmark_v3/
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
from scipy.ndimage import gaussian_filter1d, gaussian_laplace, label as nd_label
from scipy.signal import find_peaks
from scipy.stats import norm as sp_norm

# ── shared utilities (same as v2) ─────────────────────────────────────────────

COL_GT_LEFT  = "#1f77b4"
COL_GT_RIGHT = "#d62728"
COL_GT_ORI   = "#2ca02c"
METHOD_COLS  = ["#ff7f0e", "#9467bd", "#17becf", "#e377c2", "#bcbd22"]


def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def evaluate_iou(pred_df, gt_df, iou_threshold=0.2):
    if len(pred_df) == 0 and len(gt_df) == 0:
        return dict(tp=0,fp=0,fn=0,precision=1.,recall=1.,f1=1.,n_pred=0,n_gt=0)
    if len(pred_df) == 0:
        return dict(tp=0,fp=0,fn=len(gt_df),precision=0.,recall=0.,f1=0.,n_pred=0,n_gt=len(gt_df))
    if len(gt_df) == 0:
        return dict(tp=0,fp=len(pred_df),fn=0,precision=0.,recall=0.,f1=0.,n_pred=len(pred_df),n_gt=0)
    tp, fp = 0, 0
    gt_matched = set()
    gt_by_read = {rid: grp for rid, grp in gt_df.groupby("read_id")}
    for _, p in pred_df.iterrows():
        rid = p["read_id"]
        if rid not in gt_by_read:
            fp += 1; continue
        best_iou, best_idx = 0.0, None
        for idx, g in gt_by_read[rid].iterrows():
            iou = compute_iou(int(p["start"]), int(p["end"]), int(g["start"]), int(g["end"]))
            if iou > best_iou and idx not in gt_matched:
                best_iou, best_idx = iou, idx
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1; gt_matched.add(best_idx)
        else:
            fp += 1
    fn   = len(gt_df) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tp=tp,fp=fp,fn=fn,precision=prec,recall=rec,f1=f1,
                n_pred=len(pred_df),n_gt=len(gt_df))


def load_xy(base_dir, run_dirs, read_ids=None):
    reads = {}
    for run_dir in run_dirs:
        run_path = Path(base_dir) / run_dir
        if not run_path.exists(): continue
        for f in run_path.glob("plot_data_*.txt"):
            rid = f.stem.replace("plot_data_", "")
            if read_ids is not None and rid not in read_ids: continue
            if rid not in reads:
                try:
                    df = pd.read_csv(f, sep="\t", header=None,
                                     names=["chr","start","end","signal"])
                    df["read_id"] = rid
                    reads[rid] = df
                except Exception: pass
    return reads


def load_bed4(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0,1,2,3],
                       names=["chr","start","end","read_id"])


def _expand_rectangular(bins_df, target_len=10000):
    widths  = (bins_df["end"].values - bins_df["start"].values).astype(np.float64)
    signals = bins_df["signal"].values.astype(np.float64)
    total   = widths.sum()
    if total <= 0:
        return (np.zeros(target_len),
                np.linspace(float(bins_df["start"].iloc[0]),
                            float(bins_df["end"].iloc[-1]), target_len))
    cum     = np.cumsum(widths)
    fracs   = np.linspace(0, total, target_len)
    idx     = np.searchsorted(cum, fracs, side="right").clip(0, len(signals)-1)
    gen_pos = (bins_df["start"].iloc[0]
               + fracs * (bins_df["end"].iloc[-1] - bins_df["start"].iloc[0]) / total)
    return signals[idx], gen_pos


def _merge_mask(mask, pos, merge_gap_bp, min_len_bp):
    """Merge True regions separated by <= merge_gap_bp, drop regions < min_len_bp."""
    if not mask.any():
        return []
    read_len = float(pos[-1] - pos[0] + 1)
    gap_u = max(1, int(merge_gap_bp * len(pos) / read_len))
    # dilate
    dilated = np.zeros_like(mask)
    for i in np.where(mask)[0]:
        dilated[max(0, i-gap_u): i+gap_u+1] = True
    labeled, n = nd_label(dilated)
    events = []
    for lab in range(1, n+1):
        idxs = np.where(labeled == lab)[0]
        if not mask[idxs].any(): continue
        s_bp = float(pos[idxs[0]]); e_bp = float(pos[idxs[-1]])
        if e_bp - s_bp >= min_len_bp:
            events.append((s_bp, e_bp))
    return events


def _segs_to_df(segs, chrom, read_id):
    if not segs:
        return pd.DataFrame(columns=["chr","start","end","read_id"])
    return pd.DataFrame([{"chr": chrom, "start": int(s), "end": int(e),
                           "read_id": read_id} for s, e in segs])


def _origins_from_forks(left_df, right_df, read_id, chrom, max_dist=150_000):
    if len(left_df) == 0 or len(right_df) == 0:
        return pd.DataFrame(columns=["chr","start","end","read_id"])
    origins = []
    for _, L in left_df.sort_values("end").iterrows():
        cands = right_df[right_df["start"] >= L["end"]]
        if len(cands) == 0: continue
        R = cands.iloc[0]
        if R["start"] - L["end"] <= max_dist:
            origins.append({"chr": chrom, "start": int(L["end"]),
                             "end": int(R["start"]), "read_id": read_id})
    return pd.DataFrame(origins) if origins else pd.DataFrame(
        columns=["chr","start","end","read_id"])


# ── METHOD 1: Multiscale LoG ──────────────────────────────────────────────────

def multiscale_log(bins_df: pd.DataFrame,
                    scales_kb: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                    log_prominence: float = 0.04,
                    grad_rel_threshold: float = 0.15,
                    merge_gap_bp: int = 8000,
                    min_fork_len_bp: int = 2000,
                    target_len: int = 10000) -> dict:
    """
    Multiscale Laplacian-of-Gaussian (LoG) replication event detector.

    CONCEPT:
      The scale-normalized LoG at scale σ is:
          L(x, σ) = -σ² · ∇²(G_σ * signal)(x)
      It is positive where the signal has a peak of spatial extent ≈ σ√2.
      By taking the MAX over scales (scale-space maximum), we get a signal that
      responds to peaks regardless of their width.  This detects ORIs without
      knowing their scale in advance.

    ALGORITHM:
      1. Compute scale-normalized LoG at scales [0.5, 1, 2, 5, 10, 20] kb.
      2. Scale-space maximum = max_σ L(x, σ) — a single trace with peaks at ORI positions.
      3. Find peaks in the scale-space maximum → ORI candidates.
         ORI region width = 2 * σ_max(x) * √2 (the scale that gave the max response).
      4. Forks: gradient-sign method (same as v2 sensitive) at the scale with best LoG.

    ADVANTAGES over single-scale peak detection:
      - Finds small ORIs (0.5 kb) AND large ORIs (20 kb) simultaneously.
      - No need to tune a single smoothing scale.
      - Peak positions in scale-space are more stable than in the raw signal.
    """
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id  = bins_df["read_id"].iloc[0]
    chrom    = bins_df["chr"].iloc[0]
    read_len = float(pos[-1] - pos[0] + 1)
    sig_n    = (sig - sig.mean()) / (sig.std() + 1e-8)

    # ── Scale-space LoG stack ─────────────────────────────────────────────────
    log_stack = []
    sigma_vals = []
    for scale_kb in scales_kb:
        sigma = scale_kb * 1000 * target_len / read_len
        sigma = max(1.0, sigma)
        # Scale-normalized LoG: positive for bright blobs (peaks)
        log_resp = -(sigma ** 2) * gaussian_laplace(sig_n, sigma=sigma)
        log_stack.append(log_resp)
        sigma_vals.append(sigma)

    log_stack   = np.stack(log_stack, axis=0)          # (n_scales, target_len)
    best_scale  = np.argmax(log_stack, axis=0)          # which scale has max LoG
    log_max     = log_stack.max(axis=0)                 # scale-space maximum

    # ── Detect ORIs as peaks in scale-space maximum ───────────────────────────
    # Width in array units from the scale with max response per position
    # ORI size estimate: 2 * sigma(best_scale) * sqrt(2)
    width_u  = max(1, int(200 * target_len / read_len))   # min 200 bp in array units
    peaks, peak_props = find_peaks(log_max,
                                    prominence=log_prominence,
                                    width=width_u)

    ori_events = []
    for pk in peaks:
        # Estimate ORI half-width from the LoG scale at this peak
        sc      = sigma_vals[int(best_scale[pk])]
        hw_u    = max(int(sc * 1.5), width_u)
        s_bp    = float(pos[max(0, pk - hw_u)])
        e_bp    = float(pos[min(target_len - 1, pk + hw_u)])
        if e_bp - s_bp >= 100:
            ori_events.append((s_bp, e_bp))

    # ── Forks by adaptive gradient on the mid-scale smoothed signal ───────────
    mid_scale_kb = scales_kb[len(scales_kb) // 2]
    sigma_mid    = max(1.0, mid_scale_kb * 1000 * target_len / read_len)
    smoothed_mid = gaussian_filter1d(sig_n, sigma=sigma_mid)
    grad         = np.gradient(smoothed_mid)
    grad_p95     = np.percentile(np.abs(grad), 95)
    threshold    = grad_rel_threshold * max(grad_p95, 1e-9)

    left_events  = _merge_mask(grad >  threshold, pos, merge_gap_bp, min_fork_len_bp)
    right_events = _merge_mask(grad < -threshold, pos, merge_gap_bp, min_fork_len_bp)

    return {
        "left_fork":  _segs_to_df(left_events,  chrom, read_id),
        "right_fork": _segs_to_df(right_events, chrom, read_id),
        "origin":     _segs_to_df(ori_events,   chrom, read_id),
        "smoothed": smoothed_mid, "log_max": log_max, "pos": pos,
        "method": f"LoG-multiscale ({len(scales_kb)} scales)",
    }


# ── METHOD 2: HMM with Viterbi ────────────────────────────────────────────────

def viterbi_hmm(bins_df: pd.DataFrame,
                 smooth_sigma_kb: float = 5.0,
                 expected_fork_len_bp: int = 30_000,
                 expected_ori_len_bp: int = 5_000,
                 min_len_bp: int = 2000,
                 target_len: int = 10000) -> dict:
    """
    HMM with Viterbi decoding for sequential replication event segmentation.

    STATES:   0 = background  (low BrdU, neither fork nor ORI)
              1 = left_fork   (rising BrdU, gradient > 0)
              2 = origin      (gradient ≈ 0, near signal maximum)
              3 = right_fork  (falling BrdU, gradient < 0)

    TRANSITIONS enforce the biological structure:
        bg ──(rare)──→ left_fork ──(rare)──→ origin ──(rare)──→ right_fork ──(rare)──→ bg
         ↑                                                                               |
         └───────────────────────────────────────────────────────────────────────────────┘
                              (cycle repeats for multiple ORIs per read)

        bg ──(rare)──→ right_fork   (read may start inside a right fork, ORI off-read)
        left_fork ──(rare)──→ bg    (fork ends without ORI, read edge)

    EMISSIONS are per-read Gaussians on two features:
        f1 = gradient of Gaussian-smoothed signal     (primary feature)
        f2 = smoothed signal value                    (secondary, for origin)

        bg:         f1 ~ N(0,    σ_g),      f2 = uninformative
        left_fork:  f1 ~ N(+μ_g, σ_g*0.7), f2 = uninformative
        origin:     f1 ~ N(0,    σ_g*0.25), f2 ~ N(μ_high, σ_g) [high signal]
        right_fork: f1 ~ N(-μ_g, σ_g*0.7), f2 = uninformative

    The emission parameters (μ_g, σ_g, μ_high) are estimated from the
    READ ITSELF (unsupervised), not from any training labels.

    VITERBI ALGORITHM finds the globally optimal state sequence in O(S² T) time.

    ADVANTAGE over gradient thresholding:
      • Enforces left→ori→right ordering (no direct left→right transitions)
      • Transition probabilities encode expected event durations
      • The origin state is identified by BOTH flat gradient AND high signal
      • A single fork can span a noisy region without breaking
    """
    sig, pos = _expand_rectangular(bins_df, target_len)
    read_id  = bins_df["read_id"].iloc[0]
    chrom    = bins_df["chr"].iloc[0]
    read_len = float(pos[-1] - pos[0] + 1)

    # Normalize + smooth
    sig_n    = (sig - sig.mean()) / (sig.std() + 1e-8)
    sigma    = smooth_sigma_kb * 1000 * target_len / read_len
    smoothed = gaussian_filter1d(sig_n, sigma=max(1.0, sigma))
    grad     = np.gradient(smoothed)

    # ── Per-read emission parameters (unsupervised) ───────────────────────────
    grad_std  = max(grad.std(), 1e-9)
    pos_g     = grad[grad > 0]
    neg_g     = grad[grad < 0]
    mu_pos    = pos_g.mean() if len(pos_g) > 10 else grad_std * 0.5
    mu_neg    = neg_g.mean() if len(neg_g) > 10 else -grad_std * 0.5
    mu_high   = np.percentile(smoothed, 80)   # "high signal" threshold for origin
    sig_high  = max(smoothed.std() * 0.4, 0.1)

    # ── Log-emission matrix (n_states × target_len) ───────────────────────────
    # Each column is the log-likelihood of that timestep under each state
    log_emit = np.empty((4, target_len), dtype=np.float64)
    log_emit[0] = sp_norm.logpdf(grad, 0,     grad_std)          # background
    log_emit[1] = sp_norm.logpdf(grad, mu_pos, grad_std * 0.7)   # left_fork
    # origin: flat gradient (tight) + high signal
    log_emit[2] = (sp_norm.logpdf(grad, 0,       grad_std * 0.25) +
                   sp_norm.logpdf(smoothed, mu_high, sig_high))
    log_emit[3] = sp_norm.logpdf(grad, mu_neg, grad_std * 0.7)   # right_fork

    # ── Transition matrix ─────────────────────────────────────────────────────
    # p_stay for each state is derived from expected duration (geometric distribution)
    # Expected duration in array units ≈ expected_len_bp * target_len / read_len
    def p_stay_from_len(expected_len_bp):
        steps = max(5, int(expected_len_bp * target_len / read_len))
        return 1.0 - 1.0 / steps

    ps_bg    = p_stay_from_len(expected_fork_len_bp)   # bg = termination zone
    ps_left  = p_stay_from_len(expected_fork_len_bp)
    ps_ori   = p_stay_from_len(expected_ori_len_bp)
    ps_right = p_stay_from_len(expected_fork_len_bp)

    eps = 1e-9
    # Rows = from, cols = to (must sum to 1)
    #                bg                   left                  origin                right
    trans = np.array([
        [ps_bg,        (1-ps_bg)*0.45,     (1-ps_bg)*0.05,      (1-ps_bg)*0.50  ],  # bg
        [(1-ps_left)*0.20, ps_left,        (1-ps_left)*0.80,    0.0             ],  # left
        [0.0,           0.0,               ps_ori,              (1-ps_ori)       ],  # origin
        [(1-ps_right),  0.0,               0.0,                 ps_right         ],  # right
    ]) + eps
    trans      = trans / trans.sum(axis=1, keepdims=True)
    log_trans  = np.log(trans)   # (4, 4)

    # ── Viterbi forward pass ──────────────────────────────────────────────────
    # log_prob[s, t] = log P(best path ending in state s at time t)
    log_prob  = np.full((4, target_len), -np.inf)
    backtrack = np.zeros((4, target_len), dtype=np.int32)

    log_init        = np.log(np.array([0.50, 0.20, 0.05, 0.25]) + eps)  # prior: can start in any state
    log_prob[:, 0]  = log_init + log_emit[:, 0]

    for t in range(1, target_len):
        # scores[i, j] = log_prob[i, t-1] + log_trans[i, j]
        # shape: (4, 4); rows = prev state, cols = next state
        scores         = log_prob[:, t - 1, None] + log_trans    # (4, 4)
        backtrack[:, t] = np.argmax(scores, axis=0)               # best prev for each next
        log_prob[:, t]  = (scores[backtrack[:, t], np.arange(4)]
                           + log_emit[:, t])

    # ── Backtrack ─────────────────────────────────────────────────────────────
    states       = np.zeros(target_len, dtype=np.int32)
    states[-1]   = int(np.argmax(log_prob[:, -1]))
    for t in range(target_len - 2, -1, -1):
        states[t] = backtrack[states[t + 1], t + 1]

    # ── Extract events ────────────────────────────────────────────────────────
    results = {}
    for state_id, cls in [(1, "left_fork"), (2, "origin"), (3, "right_fork")]:
        segs = []
        labeled, n = nd_label(states == state_id)
        for lab in range(1, n + 1):
            idxs  = np.where(labeled == lab)[0]
            s_bp  = float(pos[idxs[0]])
            e_bp  = float(pos[idxs[-1]])
            if e_bp - s_bp >= min_len_bp:
                segs.append((s_bp, e_bp))
        results[cls] = _segs_to_df(segs, chrom, read_id)

    return {
        **results,
        "states":   states,
        "smoothed": smoothed,
        "grad":     grad,
        "pos":      pos,
        "method":   f"Viterbi-HMM (σ={smooth_sigma_kb}kb)",
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_read_examples(reads_data, gt_left, gt_right, gt_ori,
                        method_results, output_path, n_reads=4):
    example_ids = list(reads_data.keys())[:n_reads]
    n_methods   = len(method_results)
    n_rows      = 2 + n_methods

    fig, axes = plt.subplots(n_rows, len(example_ids),
                              figsize=(6.5 * len(example_ids), 3.0 * n_rows),
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
        ax.set_ylabel("signal" if ci == 0 else "", fontsize=7)
        ax.set_xticks([]); ax.tick_params(labelsize=6)
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
                ax.set_ylabel(mname, fontsize=6, rotation=0, labelpad=45, va="center")
            rp = all_preds.get(read_id, {})
            for df_key, ymin, ymax, color in [
                ("left_fork",  0.67, 0.99, COL_GT_LEFT),
                ("right_fork", 0.34, 0.66, COL_GT_RIGHT),
                ("origin",     0.01, 0.33, col),
            ]:
                for row in rp.get(df_key, pd.DataFrame()).itertuples(index=False):
                    ax.axvspan(int(row.start), int(row.end), ymin=ymin, ymax=ymax,
                               color=color, alpha=0.75)
            ax.set_xticks([]) if mi < n_methods - 1 else ax.tick_params(axis="x", labelsize=6)
            ax.set_xlim(axes[0][ci].get_xlim())

    fig.legend(handles=[mpatches.Patch(color=COL_GT_LEFT, label="pred left"),
                        mpatches.Patch(color=COL_GT_RIGHT, label="pred right"),
                        mpatches.Patch(color=METHOD_COLS[0], label="LoG ORI"),
                        mpatches.Patch(color=METHOD_COLS[1], label="Viterbi ORI")],
               loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Mathematical pipeline v3: sequential context methods", fontsize=11)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison(summary_df, ai_ref, output_path, iou_threshold=0.2):
    methods = summary_df["method"].unique().tolist()
    classes = ["left_fork", "right_fork", "origin"]
    sub     = summary_df[summary_df["iou_threshold"] == iou_threshold]
    x       = np.arange(len(classes))
    n       = len(methods)
    w       = 0.16
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * w

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
        label = "Recall (primary)" if metric == "recall" else "F1"
        ax.set_ylabel(f"{label}  (IoU >= {iou_threshold})", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if metric == "recall":
            ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(f"Mathematical pipeline v3  vs  FORTE v1 (AI)  |  IoU >= {iou_threshold}",
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
        default="CODEX/results/mathematical_benchmark_v3")
    parser.add_argument("--max-reads", type=int, default=400)
    parser.add_argument("--iou-primary", type=float, default=0.2)
    parser.add_argument("--example-reads", nargs="+", default=[
        "05ac4325-04b3-4cb9-b59d-3bd26d1042ca",
        "c0e60f34-fb83-4315-b2ed-40842e85171e",
        "c4dfc355-98e8-4fca-83f7-d666f17a4eb1",
        "d8f2ca7c-bd7b-4c40-9a7d-17c6737664d7",
    ])
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

    gt_left  = load_bed4(str(base / args.gt_left))
    gt_right = load_bed4(str(base / args.gt_right))
    gt_ori   = load_bed4(str(base / args.gt_ori))
    print(f"GT: {len(gt_left):,} left | {len(gt_right):,} right | {len(gt_ori):,} ori")

    annotated_ids = set(gt_left["read_id"]) | set(gt_right["read_id"]) | set(gt_ori["read_id"])
    all_ids = list(set(list(annotated_ids)[:args.max_reads]) | set(args.example_reads))

    print(f"Loading XY for up to {args.max_reads} reads…")
    reads_data = load_xy(args.base_dir, run_dirs, read_ids=all_ids)
    print(f"  Loaded: {len(reads_data):,}")

    loaded_ids    = set(reads_data.keys())
    gt_left_eval  = gt_left[gt_left["read_id"].isin(loaded_ids)]
    gt_right_eval = gt_right[gt_right["read_id"].isin(loaded_ids)]
    gt_ori_eval   = gt_ori[gt_ori["read_id"].isin(loaded_ids)]
    print(f"GT (eval): {len(gt_left_eval):,} left | {len(gt_right_eval):,} right | {len(gt_ori_eval):,} ori")

    methods_cfg = [
        ("LoG multiscale\n(0.5-20 kb)", multiscale_log,
         dict(scales_kb=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
              log_prominence=0.03, grad_rel_threshold=0.15,
              merge_gap_bp=8000, min_fork_len_bp=2000)),

        ("Viterbi HMM\n(σ=5kb)", viterbi_hmm,
         dict(smooth_sigma_kb=5.0,
              expected_fork_len_bp=30_000,
              expected_ori_len_bp=5_000,
              min_len_bp=2000)),

        ("Viterbi HMM\n(σ=3kb)", viterbi_hmm,
         dict(smooth_sigma_kb=3.0,
              expected_fork_len_bp=20_000,
              expected_ori_len_bp=3_000,
              min_len_bp=1500)),
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
                empty = pd.DataFrame(columns=["chr","start","end","read_id"])
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
            combined_preds[mname][cl] = (pd.concat(dfs, ignore_index=True) if dfs
                else pd.DataFrame(columns=["chr","start","end","read_id"]))

    iou_thresholds = [0.2, 0.3, 0.4, 0.5]
    gt_map = {"left_fork": gt_left_eval, "right_fork": gt_right_eval,
              "origin": gt_ori_eval}

    print(f"\nIoU evaluation @ IoU >= {args.iou_primary}:")
    print(f"{'Method':<32} {'Class':<12} {'Recall':>7} {'Prec':>7} {'F1':>7} {'#pred':>6}")
    print("-" * 75)
    rows = []
    for mname in [m[0] for m in methods_cfg]:
        for iou_thr in iou_thresholds:
            for cl in ["left_fork", "right_fork", "origin"]:
                m = evaluate_iou(combined_preds[mname][cl], gt_map[cl], iou_thr)
                rows.append({"method": mname.replace("\n"," "), "class": cl,
                              "iou_threshold": iou_thr, **m})
                if iou_thr == args.iou_primary:
                    print(f"  {mname.replace(chr(10),' '):<30} {cl:<12} "
                          f"{m['recall']:>7.3f} {m['precision']:>7.3f} "
                          f"{m['f1']:>7.3f} {int(m['n_pred']):>6,}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "summary_table.tsv", sep="\t", index=False)
    print(f"\nSaved: {out / 'summary_table.tsv'}")

    ai_ref = {
        "recall": {"left_fork": args.ai_recall_left, "right_fork": args.ai_recall_right,
                   "origin": args.ai_recall_ori},
        "f1":     {"left_fork": args.ai_f1_left,     "right_fork": args.ai_f1_right,
                   "origin": args.ai_f1_ori},
    }

    plot_comparison(summary_df, ai_ref, out / "comparison_bar.png",
                    iou_threshold=args.iou_primary)

    example_data = {rid: reads_data[rid] for rid in args.example_reads if rid in reads_data}
    plot_read_examples(example_data, gt_left, gt_right, gt_ori,
                       per_read_preds, out / "read_examples.png", n_reads=4)

    print(f"\nAll outputs: {out}")


if __name__ == "__main__":
    main()
