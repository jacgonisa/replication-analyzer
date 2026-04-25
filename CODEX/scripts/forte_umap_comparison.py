#!/usr/bin/env python
"""UMAP comparison across all FORTE models.

Produces a single comparison figure:
  - Column 1 (Signal space):  UMAP on raw 9-channel signal features — same for all
                               models since the input encoding is identical.
                               This is the "pre-training" view.
  - Column 2 (Prob space):    UMAP on each model's 4D probability output vector.
                               This is the "post-training" view.

One row per model, so you can visually compare how each model carves up the
same underlying signal.

Usage (CPU, ~20 min for 5 models × 25k windows):
  CUDA_VISIBLE_DEVICES="" python forte_umap_comparison.py

Or run specific models only:
  CUDA_VISIBLE_DEVICES="" python forte_umap_comparison.py \
      --models forte_v1 forte_v1_conservative
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
import umap
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
from replication_analyzer_codex.evaluation import predict_reads
from replication_analyzer_codex.representation import encode_read_dataframe
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

CLASS_NAMES  = {0: "background", 1: "left_fork", 2: "right_fork", 3: "origin"}
CLASS_COLORS = {
    0: "#aaaaaa",   # background
    1: "#2196F3",   # left_fork
    2: "#F44336",   # right_fork
    3: "#4CAF50",   # origin
}

BASE = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")

MODEL_SPECS = {
    "forte_v1": dict(
        config="CODEX/configs/forte_v1.yaml",
        model="CODEX/models/forte_v1.keras",
        val_info="CODEX/results/forte_v1/preprocessed_forte_v1.val_info.tsv",
    ),
    "forte_v1_conservative": dict(
        config="CODEX/configs/forte_v1_conservative.yaml",
        model="CODEX/models/forte_v1_conservative.keras",
        val_info="CODEX/results/forte_v1_conservative/preprocessed_forte_v1_conservative.val_info.tsv",
    ),
    "forte_v2": dict(
        config="CODEX/configs/forte_v2.yaml",
        model="CODEX/models/forte_v2.keras",
        val_info="CODEX/results/forte_v2/preprocessed_forte_v2.val_info.tsv",
    ),
    "forte_v2_conservative": dict(
        config="CODEX/configs/forte_v2_conservative.yaml",
        model="CODEX/models/forte_v2_conservative.keras",
        val_info="CODEX/results/forte_v2_conservative/preprocessed_forte_v2_conservative.val_info.tsv",
    ),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _stratified_sample(df: pd.DataFrame, n: int, class_col="predicted_class",
                        seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = sorted(df[class_col].unique())
    n_per = max(1, n // len(classes))
    parts = []
    for cls in classes:
        sub = df[df[class_col] == cls]
        take = min(len(sub), n_per)
        parts.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _run_inference(model_path, config_path, val_info_path, xy_cache_path,
                   n_reads, seed=42):
    """Load model, run inference on a sample of val reads, return predictions df."""
    val_info = pd.read_csv(str(BASE / val_info_path), sep="\t")
    rng = np.random.default_rng(seed)
    read_ids = val_info["read_id"].tolist()
    if len(read_ids) > n_reads:
        idx = rng.choice(len(read_ids), n_reads, replace=False)
        read_ids = [read_ids[i] for i in idx]

    print(f"  Loading XY cache…")
    with open(str(BASE / xy_cache_path), "rb") as fh:
        xy_data = pickle.load(fh)
    # Filter to sampled reads
    xy_data = xy_data[xy_data["read_id"].isin(set(read_ids))].copy()

    with open(str(BASE / config_path)) as f:
        config = yaml.safe_load(f)

    print(f"  Loading model {Path(model_path).name}…")
    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(str(BASE / model_path),
                                        custom_objects=CUSTOM_OBJECTS)
    max_length = model.input_shape[1]

    actual_ids = list(xy_data["read_id"].unique())
    print(f"  Running inference on {len(actual_ids):,} reads…")
    preds = predict_reads(model, xy_data, actual_ids, max_length,
                          config["preprocessing"])
    tf.keras.backend.clear_session()
    return preds, config


def _build_signal_features(preds: pd.DataFrame, xy_cache_path: str,
                             preprocessing_config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (features array, valid label array) after NaN removal."""
    print("  Building signal features…")
    with open(str(BASE / xy_cache_path), "rb") as fh:
        xy_data = pickle.load(fh)

    xy_by_read = {rid: grp.reset_index(drop=True)
                  for rid, grp in xy_data.groupby("read_id", sort=False)}
    del xy_data

    feature_rows = []
    labels_out   = []
    for read_id, win_df in preds.groupby("read_id"):
        read_xy = xy_by_read.get(read_id)
        if read_xy is None:
            continue
        encoded = encode_read_dataframe(read_xy, preprocessing_config)
        start_to_idx = {int(r.start): i for i, r in read_xy.iterrows()}
        for _, win in win_df.iterrows():
            idx = start_to_idx.get(int(win["start"]))
            if idx is not None and idx < len(encoded):
                feature_rows.append(encoded[idx].tolist())
                labels_out.append(int(win["predicted_class"]))

    feat = np.array(feature_rows, dtype=np.float32)
    labs = np.array(labels_out, dtype=np.int32)
    valid = ~np.isnan(feat).any(axis=1)
    return feat[valid], labs[valid]


def _run_umap(data: np.ndarray, n_neighbors=15, min_dist=0.1, seed=42) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         n_components=2, random_state=seed, verbose=False)
    return reducer.fit_transform(data)


def _scatter(ax, emb, labels, title, extra=None, extra_cmap="viridis",
             extra_label="", alpha=0.35, s=2):
    if extra is not None:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=extra, s=s, alpha=alpha,
                        linewidths=0, cmap=extra_cmap)
        plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
    else:
        for cls_id, cls_name in CLASS_NAMES.items():
            mask = labels == cls_id
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=CLASS_COLORS[cls_id], s=s, alpha=alpha,
                       linewidths=0, label=f"{cls_name} ({mask.sum():,})")
        ax.legend(markerscale=5, fontsize=7, loc="best",
                  framealpha=0.7, handletextpad=0.3)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=7); ax.set_ylabel("UMAP 2", fontsize=7)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=list(MODEL_SPECS.keys()),
                        help="Which models to include")
    parser.add_argument("--n-reads", type=int, default=2000,
                        help="Val reads per model for inference")
    parser.add_argument("--n-windows", type=int, default=25000,
                        help="Windows per model after stratified sampling")
    parser.add_argument("--xy-cache", type=str,
                        default="CODEX/results/cache/xy_data.pkl")
    parser.add_argument("--output-dir", type=str,
                        default="CODEX/results/umap_forte_comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = BASE / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    model_names = [m for m in args.models if m in MODEL_SPECS
                   and (BASE / MODEL_SPECS[m]["model"]).exists()]
    if not model_names:
        raise RuntimeError("No valid models found")
    print(f"Models: {model_names}")

    # ── Step 1: run inference for each model ──────────────────────────────────
    all_preds   = {}
    all_configs = {}
    for mname in model_names:
        spec = MODEL_SPECS[mname]
        print(f"\n{'='*60}")
        print(f"Model: {mname}")
        print(f"{'='*60}")
        preds, config = _run_inference(
            model_path=spec["model"],
            config_path=spec["config"],
            val_info_path=spec["val_info"],
            xy_cache_path=args.xy_cache,
            n_reads=args.n_reads,
            seed=args.seed,
        )
        all_preds[mname]   = preds
        all_configs[mname] = config
        preds.to_csv(out / f"{mname}_predictions.tsv", sep="\t", index=False)
        print(f"  {len(preds):,} windows predicted")
        print(f"  Class dist: {dict(preds['predicted_class'].value_counts().sort_index())}")

    # ── Step 2: build UMAPs ───────────────────────────────────────────────────
    # Signal UMAP — compute once using the first model's config
    # (all models share the same signal encoding, so this is model-independent)
    first = model_names[0]
    sample_first = _stratified_sample(all_preds[first], args.n_windows, seed=args.seed)

    print(f"\nBuilding signal features for pre-training UMAP (from {first})…")
    sig_feat, sig_labels = _build_signal_features(
        sample_first, args.xy_cache,
        all_configs[first]["preprocessing"])
    print(f"  {len(sig_feat):,} windows with signal features")

    print("Running UMAP on signal features…")
    emb_signal = _run_umap(sig_feat, seed=args.seed)
    np.save(out / "umap_signal_embedding.npy", emb_signal)
    print("  Done.")

    # Probability UMAPs — one per model
    prob_embeddings = {}
    prob_samples    = {}
    for mname in model_names:
        print(f"\nRunning UMAP on probability space: {mname}…")
        sample = _stratified_sample(all_preds[mname], args.n_windows, seed=args.seed)
        prob_samples[mname] = sample
        prob_cols = ["prob_background", "prob_left_fork", "prob_right_fork", "prob_origin"]
        prob_feat = sample[prob_cols].values.astype(np.float32)
        emb = _run_umap(prob_feat, seed=args.seed)
        prob_embeddings[mname] = emb
        np.save(out / f"umap_prob_{mname}.npy", emb)
        print("  Done.")

    # ── Step 3: comparison figure ─────────────────────────────────────────────
    # Layout:
    #   Row 0: Signal space (pre-training, shared) | Signal entropy
    #   Rows 1…N: prob space per model (class color) | entropy
    n_models = len(model_names)
    n_rows   = 1 + n_models

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(12, 5 * n_rows),
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )

    # Row 0: signal space
    ax_sig_class   = axes[0][0]
    ax_sig_entropy = axes[0][1]
    _scatter(ax_sig_class, emb_signal, sig_labels,
             f"Signal features — pre-training (n={len(emb_signal):,})")
    # top_prob as proxy for "confidence" in signal space
    top_prob_sig = sample_first["top_prob"].values[: len(sig_feat)]
    _scatter(ax_sig_entropy, emb_signal, sig_labels,
             "Top class probability (signal space)",
             extra=top_prob_sig, extra_label="top prob")

    # Rows 1…N: each model's probability space
    for row_i, mname in enumerate(model_names, start=1):
        emb   = prob_embeddings[mname]
        samp  = prob_samples[mname]
        labs  = samp["predicted_class"].values

        ax_class   = axes[row_i][0]
        ax_entropy = axes[row_i][1]

        label_map = {
            "forte_v1":              "FORTE v1",
            "forte_v1_conservative": "FORTE v1 conservative",
            "forte_v2":              "FORTE v2",
            "forte_v2_conservative": "FORTE v2 conservative",
        }
        display_name = label_map.get(mname, mname)
        _scatter(ax_class, emb, labs,
                 f"{display_name} — prob space (n={len(emb):,})")
        _scatter(ax_entropy, emb, labs,
                 f"{display_name} — entropy",
                 extra=samp["entropy"].values, extra_label="Entropy")

    fig.suptitle("UMAP: pre-training (signal features) vs post-training (probability space)\n"
                 "per FORTE model variant", fontsize=12, fontweight="bold")

    out_png = out / "forte_umap_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison: {out_png}")

    # ── Step 4: individual per-model figures (same style as weak5 UMAP) ───────
    for mname in model_names:
        emb   = prob_embeddings[mname]
        samp  = prob_samples[mname]
        labs  = samp["predicted_class"].values
        display_name = label_map.get(mname, mname)

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        _scatter(axes2[0], emb, labs,
                 f"UMAP — model probability space (n={len(emb):,})\n{display_name}")
        _scatter(axes2[1], emb, labs,
                 "Entropy (prediction uncertainty)",
                 extra=samp["entropy"].values)
        fig2.tight_layout()
        fig2.savefig(out / f"umap_prob_{mname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out / f'umap_prob_{mname}.png'}")

    # individual signal UMAP
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(axes3[0], emb_signal, sig_labels,
             f"UMAP — 9-channel signal features (n={len(emb_signal):,})\npre-training / model-independent")
    _scatter(axes3[1], emb_signal, sig_labels,
             "Top class probability (signal space)",
             extra=top_prob_sig)
    fig3.tight_layout()
    fig3.savefig(out / "umap_signal_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out / 'umap_signal_features.png'}")

    print(f"\nAll outputs: {out}")


if __name__ == "__main__":
    main()
