#!/usr/bin/env python
"""UMAP visualisation of per-window features from the CODEX model.

Two complementary views:
  1. UMAP on 9-channel signal features  — does the raw signal cluster by class?
  2. UMAP on 4D probability vectors     — how does the model separate classes in output space?

Both use a stratified subsample of windows from the val split predictions.

Usage:
  CUDA_VISIBLE_DEVICES="" python umap_windows_codex.py \\
      --predictions  CODEX/results/weak5_rectangular_v4/evaluation_val/predictions.tsv \\
      --config       CODEX/configs/weak5_rectangular_v4.yaml \\
      --output       CODEX/results/weak5_rectangular_v4/umap \\
      --n-windows    30000

Optional:
  --all-predictions  CODEX/results/forte_v1/pseudo_labels/all_predictions.tsv
                     (use the full-dataset predictions once pseudo-labels are done)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import umap
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.representation import encode_read_dataframe

CLASS_NAMES = {0: "background", 1: "left_fork", 2: "right_fork", 3: "origin"}
CLASS_COLORS = {
    0: "#aaaaaa",   # background — grey
    1: "#2196F3",   # left_fork  — blue
    2: "#F44336",   # right_fork — red
    3: "#4CAF50",   # origin     — green
}


def _stratified_sample(df: pd.DataFrame, n: int, class_col: str = "predicted_class",
                        random_seed: int = 42) -> pd.DataFrame:
    """Sample up to n rows, stratified by class_col."""
    rng = np.random.default_rng(random_seed)
    classes = df[class_col].unique()
    n_per_class = max(1, n // len(classes))
    parts = []
    for cls in sorted(classes):
        sub = df[df[class_col] == cls]
        take = min(len(sub), n_per_class)
        idx = rng.choice(len(sub), size=take, replace=False)
        parts.append(sub.iloc[idx])
    sampled = pd.concat(parts, ignore_index=True)
    # Shuffle so class order doesn't dominate scatter plot layering
    sampled = sampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return sampled


def _build_signal_features(predictions: pd.DataFrame, xy_cache_path: str,
                            preprocessing_config: dict) -> np.ndarray:
    """Build (N, 9) matrix of per-window encoded signal features."""
    print("  Loading XY cache for signal encoding…")
    with open(xy_cache_path, "rb") as fh:
        xy_data = pickle.load(fh)

    xy_data = xy_data.sort_values(["read_id", "start"]).reset_index(drop=True)
    xy_by_read = {rid: grp.reset_index(drop=True)
                  for rid, grp in xy_data.groupby("read_id", sort=False)}
    del xy_data

    print(f"  Encoding signal features for {len(predictions):,} windows…")
    feature_rows = []
    missing = 0
    reads_done = 0

    # Encode per read; pick only the windows that are in the sample
    for read_id, win_df in predictions.groupby("read_id"):
        read_xy = xy_by_read.get(read_id)
        if read_xy is None or len(read_xy) == 0:
            feature_rows.extend([[np.nan] * 9] * len(win_df))
            missing += len(win_df)
            continue

        encoded = encode_read_dataframe(read_xy, preprocessing_config)  # (N_bins, 9)

        # Map each sampled window to its bin index by start position
        start_to_idx = {int(row.start): i for i, row in read_xy.iterrows()}
        for _, win in win_df.iterrows():
            idx = start_to_idx.get(int(win["start"]))
            if idx is not None and idx < len(encoded):
                feature_rows.append(encoded[idx].tolist())
            else:
                feature_rows.append([np.nan] * 9)
                missing += 1

        reads_done += 1
        if reads_done % 1000 == 0:
            print(f"    {reads_done:,} reads done…", flush=True)

    if missing > 0:
        print(f"  ⚠  {missing} windows had no matching bin (set to NaN, will be dropped)")

    return np.array(feature_rows, dtype=np.float32)


def _plot_umap(embedding: np.ndarray, labels: np.ndarray, title: str,
               output_path: Path, color_by: str = "predicted_class",
               extra_col: np.ndarray | None = None, extra_label: str = ""):
    fig, axes = plt.subplots(1, 2 if extra_col is not None else 1,
                              figsize=(14 if extra_col is not None else 7, 6))
    if extra_col is None:
        axes = [axes]

    ax = axes[0]
    for cls_id, cls_name in CLASS_NAMES.items():
        mask = labels == cls_id
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=CLASS_COLORS[cls_id], s=2, alpha=0.4, linewidths=0,
                   label=f"{cls_name} (n={mask.sum():,})")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    legend = ax.legend(markerscale=4, fontsize=8, loc="best",
                       handler_map={plt.Line2D: None})
    ax.set_xticks([])
    ax.set_yticks([])

    if extra_col is not None:
        ax2 = axes[1]
        sc = ax2.scatter(embedding[:, 0], embedding[:, 1],
                         c=extra_col, s=2, alpha=0.4, linewidths=0,
                         cmap="viridis")
        plt.colorbar(sc, ax=ax2, shrink=0.8)
        ax2.set_title(extra_label, fontsize=11)
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP of CODEX window features")
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions.tsv (per-window probabilities)")
    parser.add_argument("--config", required=True, help="CODEX YAML config")
    parser.add_argument("--output", required=True, help="Output directory for plots")
    parser.add_argument("--n-windows", type=int, default=30000,
                        help="Total windows to sample (stratified by class, default 30000)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-signal", action="store_true",
                        help="Skip the signal-feature UMAP (faster, prob-space only)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    print(f"Loading predictions: {args.predictions}")
    preds = pd.read_csv(args.predictions, sep="\t")
    print(f"  Total windows: {len(preds):,}")
    print(f"  Class distribution:\n{preds['predicted_class'].value_counts().sort_index().to_string()}")

    print(f"\nStratified sampling → {args.n_windows:,} windows…")
    sample = _stratified_sample(preds, args.n_windows, random_seed=args.random_seed)
    print(f"  Sampled: {len(sample):,}")
    print(f"  Sampled class dist:\n{sample['predicted_class'].value_counts().sort_index().to_string()}")

    labels = sample["predicted_class"].values
    prob_cols = ["prob_background", "prob_left_fork", "prob_right_fork", "prob_origin"]
    prob_features = sample[prob_cols].values.astype(np.float32)

    # ── UMAP 1: probability space ───────────────────────────────────────────
    print("\n[1/2] UMAP on 4D probability vectors…")
    reducer_prob = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        random_state=args.random_seed,
        verbose=True,
    )
    emb_prob = reducer_prob.fit_transform(prob_features)
    print("  Done.")

    _plot_umap(
        embedding=emb_prob,
        labels=labels,
        title=f"UMAP — model probability space (n={len(sample):,})",
        output_path=output_dir / "umap_probability_space.png",
        extra_col=sample["entropy"].values,
        extra_label="Entropy (prediction uncertainty)",
    )

    np.save(output_dir / "umap_prob_embedding.npy", emb_prob)

    # ── UMAP 2: signal feature space ────────────────────────────────────────
    if not args.skip_signal:
        print("\n[2/2] UMAP on 9-channel signal features…")
        xy_cache_path = config["data"].get("xy_cache_path")
        if not xy_cache_path or not Path(xy_cache_path).exists():
            print(f"  ⚠  xy_cache_path not found ({xy_cache_path}), skipping signal UMAP.")
        else:
            signal_features = _build_signal_features(
                predictions=sample,
                xy_cache_path=xy_cache_path,
                preprocessing_config=config["preprocessing"],
            )

            # Drop rows with NaN (unmatched windows)
            valid = ~np.isnan(signal_features).any(axis=1)
            signal_features = signal_features[valid]
            labels_valid = labels[valid]
            top_prob_valid = sample["top_prob"].values[valid]
            print(f"  Valid windows after NaN filter: {valid.sum():,}")

            reducer_sig = umap.UMAP(
                n_neighbors=args.umap_n_neighbors,
                min_dist=args.umap_min_dist,
                n_components=2,
                random_state=args.random_seed,
                verbose=True,
            )
            emb_sig = reducer_sig.fit_transform(signal_features)
            print("  Done.")

            _plot_umap(
                embedding=emb_sig,
                labels=labels_valid,
                title=f"UMAP — 9-channel signal features (n={valid.sum():,})",
                output_path=output_dir / "umap_signal_features.png",
                extra_col=top_prob_valid,
                extra_label="Top class probability",
            )

            np.save(output_dir / "umap_signal_embedding.npy", emb_sig)
    else:
        print("\n[2/2] Signal UMAP skipped (--skip-signal).")

    # Save the sample with embeddings for downstream exploration
    sample_out = sample.copy()
    sample_out["umap_prob_x"] = emb_prob[:, 0]
    sample_out["umap_prob_y"] = emb_prob[:, 1]
    sample_out.to_csv(output_dir / "umap_sample.tsv", sep="\t", index=False)
    print(f"\nSaved sample with coordinates: {output_dir / 'umap_sample.tsv'}")
    print("Done.")


if __name__ == "__main__":
    main()
