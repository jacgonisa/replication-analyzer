#!/usr/bin/env python
"""Benchmark each of the 9 signal feature channels as standalone classifiers.

Tests whether any single feature — or simple combination — can replace the AI model.
Uses the preprocessed validation set (already encoded, already split).

Output:
  - best_f1_per_feature.tsv  — best per-class F1 achievable with each channel
  - feature_heatmap.png       — heatmap: channel × class → best F1
  - feature_roc_curves.png    — threshold sweep curves per channel per class
  - summary.txt               — human-readable summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

CHANNEL_NAMES = [
    "norm_signal",
    "approx_upsampled",
    "detail1_upsampled",
    "detail2_upsampled",
    "local_mean",
    "local_std",
    "z_score",
    "cumsum",
    "envelope",
]
CLASS_NAMES = {1: "left_fork", 2: "right_fork", 3: "origin"}
N_THRESHOLDS = 200


# ── helpers ──────────────────────────────────────────────────────────────────

def load_val(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    x = d["val_x"]   # (N, L, 9)
    y = d["val_y"]   # (N, L)
    w = d["val_w"]   # (N, L)  mask: 0 = ignored
    return x, y, w


def flatten_masked(x, y, w):
    """Return (N_valid, 9) features and (N_valid,) labels, excluding masked windows."""
    mask = w.ravel() > 0
    return x.reshape(-1, x.shape[-1])[mask], y.ravel()[mask]


def sweep_threshold(feat_vals: np.ndarray, y_true: np.ndarray, class_id: int,
                    n_thresh: int = N_THRESHOLDS):
    """
    Sweep thresholds for a single feature channel classifying `class_id` (one-vs-all).

    Tests both directions (feat >= T → predict class, and feat <= T → predict class).
    Returns (best_f1, best_thresh, best_direction, precision_arr, recall_arr, f1_arr, thresh_arr).
    """
    y_bin = (y_true == class_id).astype(np.float32)
    n_pos = y_bin.sum()
    if n_pos == 0:
        return 0.0, 0.0, ">=", np.array([]), np.array([]), np.array([]), np.array([])

    vmin, vmax = feat_vals.min(), feat_vals.max()
    thresholds = np.linspace(vmin, vmax, n_thresh)

    best_f1, best_thresh, best_dir = 0.0, thresholds[0], ">="
    precs, recs, f1s = [], [], []

    for T in thresholds:
        for direction in [">=", "<="]:
            pred = feat_vals >= T if direction == ">=" else feat_vals <= T
            tp = (pred & (y_bin == 1)).sum()
            fp = (pred & (y_bin == 0)).sum()
            fn = ((~pred) & (y_bin == 1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_thresh, best_dir = float(f1), float(T), direction
        # Record for the >= direction (for plot)
        pred = feat_vals >= T
        tp = (pred & (y_bin == 1)).sum()
        fp = (pred & (y_bin == 0)).sum()
        fn = ((~pred) & (y_bin == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        precs.append(float(prec)); recs.append(float(rec)); f1s.append(float(f1))

    return best_f1, best_thresh, best_dir, np.array(precs), np.array(recs), np.array(f1s), thresholds


def naive_argmax_classifier(x_flat: np.ndarray, y_true: np.ndarray,
                             channels: list[int] | None = None):
    """
    Simplest multi-class classifier: for each window pick the class with the
    highest response from a set of channels, using hand-crafted sign assumptions.

    channel → class heuristic (tuned for rectangular wavelet encoding):
      norm_signal  high → origin  (BrdU signal high)
      detail1      pos  → left_fork, neg → right_fork  (rising/falling edge)
      z_score      sign → directional fork
    This is purely heuristic — just a sanity-check upper bound for simple rules.
    """
    if channels is None:
        channels = list(range(x_flat.shape[1]))
    x = x_flat[:, channels] if channels else x_flat

    n_samples = x.shape[0]
    scores = np.zeros((n_samples, 4), dtype=np.float32)  # bg, L, R, ORI

    # channel 0 = norm_signal: high → origin
    scores[:, 3] += x[:, 0] if 0 in channels else 0.0
    # channel 2 = detail1: positive peak → left, negative → right
    if 2 in channels:
        idx = channels.index(2)
        scores[:, 1] += np.clip( x[:, idx], 0, None)
        scores[:, 2] += np.clip(-x[:, idx], 0, None)
    # channel 6 = z_score: high local z → origin
    if 6 in channels:
        idx = channels.index(6)
        scores[:, 3] += np.clip(x[:, idx], 0, None)

    pred = np.argmax(scores, axis=1)
    results = {}
    for class_id, class_name in CLASS_NAMES.items():
        tp = ((pred == class_id) & (y_true == class_id)).sum()
        fp = ((pred == class_id) & (y_true != class_id)).sum()
        fn = ((pred != class_id) & (y_true == class_id)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        results[class_name] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                                "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(best_f1_df: pd.DataFrame, output_path: Path):
    """Heatmap: rows = channels, columns = classes."""
    classes = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)]
    channels = CHANNEL_NAMES
    matrix = np.array([[best_f1_df.loc[(best_f1_df["channel"] == ch) &
                                       (best_f1_df["class"] == cl), "best_f1"].values[0]
                         if len(best_f1_df.loc[(best_f1_df["channel"] == ch) &
                                               (best_f1_df["class"] == cl)]) > 0 else 0.0
                         for cl in classes] for ch in channels])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=10)
    ax.set_title("Single-feature best window-level F1 (one-vs-all)\n"
                 "Higher = feature alone can predict this class", fontsize=11)
    plt.colorbar(im, ax=ax, label="Best F1")
    for r in range(len(channels)):
        for c in range(len(classes)):
            v = matrix[r, c]
            color = "white" if v > 0.6 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_f1_curves(curve_data: dict, output_path: Path):
    """F1 vs threshold curves for each channel × class."""
    n_ch = len(CHANNEL_NAMES)
    n_cl = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_ch, n_cl, figsize=(4 * n_cl, 2.5 * n_ch),
                              sharex=False, sharey=True,
                              gridspec_kw={"hspace": 0.55, "wspace": 0.15})
    colors = {"left_fork": "#1f77b4", "right_fork": "#d62728", "origin": "#2ca02c"}

    for ci, ch_name in enumerate(CHANNEL_NAMES):
        for cj, (class_id, class_name) in enumerate(sorted(CLASS_NAMES.items())):
            ax = axes[ci][cj]
            key = (ch_name, class_name)
            if key not in curve_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            precs, recs, f1s, thresholds, best_f1, best_thresh = curve_data[key]
            ax.plot(thresholds, f1s, color=colors[class_name], linewidth=1.2)
            ax.axvline(best_thresh, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylim(0, 1)
            ax.set_title(f"{ch_name}\n→ {class_name}", fontsize=7)
            ax.text(0.98, 0.95, f"F1={best_f1:.2f}", transform=ax.transAxes,
                    fontsize=7, ha="right", va="top",
                    color="darkgreen" if best_f1 > 0.4 else "darkred")
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
            ax.tick_params(labelsize=6)

    fig.supxlabel("Threshold (feature value)", fontsize=9)
    fig.supylabel("F1 score", fontsize=9)
    fig.suptitle("Single-channel threshold sweep — window-level F1 (one-vs-all)", fontsize=10)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_summary_bars(best_f1_df: pd.DataFrame, output_path: Path):
    """Grouped bar chart: for each class, compare all 9 channels side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(CHANNEL_NAMES)))

    for ax, (class_id, class_name) in zip(axes, sorted(CLASS_NAMES.items())):
        sub = best_f1_df[best_f1_df["class"] == class_name].set_index("channel")
        vals = [sub.loc[ch, "best_f1"] if ch in sub.index else 0.0 for ch in CHANNEL_NAMES]
        bars = ax.barh(CHANNEL_NAMES, vals, color=colors, edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_title(class_name, fontweight="bold", fontsize=12)
        ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", label="F1=0.5")
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)
        ax.set_xlabel("Best F1 (window-level, one-vs-all)")
        ax.invert_yaxis()

    fig.suptitle("Single-feature classifier performance — can AI be replaced?", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark individual signal feature channels")
    parser.add_argument("--npz", required=True,
                        help="Preprocessed .npz file (must have val_x, val_y, val_w)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap number of (unmasked) windows for speed (0 = all)")
    parser.add_argument("--n-thresh", type=int, default=200,
                        help="Number of threshold values to sweep (default 200)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading {args.npz}…")
    x_raw, y_raw, w_raw = load_val(args.npz)
    print(f"  val_x: {x_raw.shape}  val_y: {y_raw.shape}  val_w: {w_raw.shape}")

    x_flat, y_flat = flatten_masked(x_raw, y_raw, w_raw)
    print(f"  Unmasked windows: {len(x_flat):,}")

    if args.max_samples and args.max_samples < len(x_flat):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_flat), size=args.max_samples, replace=False)
        x_flat, y_flat = x_flat[idx], y_flat[idx]
        print(f"  Subsampled to: {len(x_flat):,} windows")

    # Class distribution
    print("\nLabel distribution (unmasked windows):")
    for label, name in [(0, "background"), (1, "left_fork"), (2, "right_fork"), (3, "origin")]:
        n = (y_flat == label).sum()
        print(f"  class {label} ({name}): {n:,}  ({100*n/len(y_flat):.1f}%)")

    # ── Per-channel threshold sweep ───────────────────────────────────────────
    print(f"\nSweeping {args.n_thresh} thresholds per channel per class…")
    rows = []
    curve_data = {}

    for ci, ch_name in enumerate(CHANNEL_NAMES):
        feat = x_flat[:, ci]
        for class_id, class_name in sorted(CLASS_NAMES.items()):
            best_f1, best_thresh, best_dir, precs, recs, f1s, thresholds = \
                sweep_threshold(feat, y_flat, class_id, n_thresh=args.n_thresh)

            # Also record precision and recall at the best threshold
            y_bin = (y_flat == class_id)
            pred = feat >= best_thresh if best_dir == ">=" else feat <= best_thresh
            tp = (pred & y_bin).sum()
            fp = (pred & ~y_bin).sum()
            fn = (~pred & y_bin).sum()
            prec_at_best = float(tp / (tp + fp + 1e-9))
            rec_at_best  = float(tp / (tp + fn + 1e-9))

            rows.append({
                "channel": ch_name,
                "channel_idx": ci,
                "class": class_name,
                "class_id": class_id,
                "best_f1": best_f1,
                "best_threshold": best_thresh,
                "direction": best_dir,
                "precision_at_best": prec_at_best,
                "recall_at_best": rec_at_best,
                "n_positives": int((y_flat == class_id).sum()),
            })
            curve_data[(ch_name, class_name)] = (precs, recs, f1s, thresholds,
                                                  best_f1, best_thresh)
            print(f"  {ch_name:22s} → {class_name:12s}: best F1={best_f1:.3f} "
                  f"@ {best_dir}{best_thresh:.4f}  "
                  f"(prec={prec_at_best:.3f}, rec={rec_at_best:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(out / "best_f1_per_feature.tsv", sep="\t", index=False)
    print(f"\nSaved: {out / 'best_f1_per_feature.tsv'}")

    # ── Naive multi-class heuristic ───────────────────────────────────────────
    print("\nNaive argmax rule-based multi-class classifier (all 9 channels):")
    heuristic_results = naive_argmax_classifier(x_flat, y_flat, list(range(9)))
    for class_name, metrics in heuristic_results.items():
        print(f"  {class_name:12s}: F1={metrics['f1']:.3f}  "
              f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_heatmap(df, out / "feature_heatmap.png")
    plot_f1_curves(curve_data, out / "feature_f1_curves.png")
    plot_summary_bars(df, out / "feature_summary_bars.png")

    # ── Summary text ─────────────────────────────────────────────────────────
    lines = ["=" * 70,
             "SINGLE-FEATURE CLASSIFIER BENCHMARK",
             "=" * 70, "",
             f"Preprocessed NPZ: {args.npz}",
             f"Windows evaluated: {len(x_flat):,}",
             ""]

    for class_id, class_name in sorted(CLASS_NAMES.items()):
        sub = df[df["class"] == class_name].sort_values("best_f1", ascending=False)
        lines.append(f"── {class_name.upper()} ──")
        for _, row in sub.iterrows():
            lines.append(f"  {row['channel']:22s}  F1={row['best_f1']:.3f}  "
                         f"(prec={row['precision_at_best']:.3f}  "
                         f"rec={row['recall_at_best']:.3f}  "
                         f"thresh {row['direction']}{row['best_threshold']:.4f})")
        lines.append("")

    lines += ["── NAIVE MULTI-CLASS HEURISTIC (all channels) ──"]
    for class_name, m in heuristic_results.items():
        lines.append(f"  {class_name:12s}  F1={m['f1']:.3f}  "
                     f"prec={m['precision']:.3f}  rec={m['recall']:.3f}")
    lines += ["",
              "Interpretation:",
              "  F1 < 0.10 : feature carries essentially NO class-specific information",
              "  F1 0.10–0.30 : weak signal — useful only in combination",
              "  F1 0.30–0.50 : moderate — could support a simple rule-based detector",
              "  F1 > 0.50 : strong — feature alone carries significant class information",
              "  AI model (FORTE v1 val): left_fork ~0.30-0.40, right_fork ~0.40-0.55, origin ~0.55-0.65",
              ""]

    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary)
    print(summary)
    print(f"Saved: {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
