"""CODEX-only training pipeline for weakly supervised 5-event detection."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from replication_analyzer.models.fork_model import build_4class_fork_ori_model
from replication_analyzer.training.callbacks import create_callbacks, TrainingProgressLogger

from .annotations import load_annotations_for_codex
from .constants import CLASS_ID_TO_NAME
from .data_cache import load_xy_data_cached
from .event_iou_callback import EventLevelIoUCallback
from .losses import MaskedClassPrecision, MaskedClassRecall, MaskedMacroF1, MaskedMeanIoU, SparseCategoricalFocalLoss
from .representation import encode_read_dataframe
from .splits import build_read_metadata, create_split_manifest, load_split_manifest, save_split_manifest
from .weak_labels import (
    build_annotation_dict,
    build_weak_labels_for_read,
)


def _sample_partition_read_ids(
    partition_manifest: pd.DataFrame,
    split_name: str,
    oversample_ratio: float,
    random_seed: int,
    small_ori_read_ids: List[str] | None = None,
    small_ori_oversample_ratio: float = 0.0,
    large_ori_read_ids: List[str] | None = None,
    large_ori_oversample_ratio: float = 0.0,
) -> List[str]:
    read_ids = partition_manifest["read_id"].tolist()
    if split_name != "train":
        return read_ids

    rng = np.random.default_rng(random_seed)
    feature_ids = partition_manifest.loc[partition_manifest["has_any_event"], "read_id"].tolist()
    background_ids = partition_manifest.loc[~partition_manifest["has_any_event"], "read_id"].tolist()
    sampled = list(feature_ids)

    if feature_ids and oversample_ratio > 0:
        n_extra = int(len(feature_ids) * oversample_ratio)
        sampled.extend(rng.choice(feature_ids, size=n_extra, replace=True).tolist())

    # Extra oversampling pass for reads with small ORIs — stacked on top of the
    # general oversample_ratio so that small ORI reads see proportionally more
    # gradient updates than large-ORI reads.
    if small_ori_read_ids and small_ori_oversample_ratio > 0:
        split_ids = set(partition_manifest["read_id"])
        small_in_split = [r for r in small_ori_read_ids if r in split_ids]
        if small_in_split:
            n_extra_small = int(len(small_in_split) * small_ori_oversample_ratio)
            sampled.extend(rng.choice(small_in_split, size=n_extra_small, replace=True).tolist())
            print(f"  Small ORI oversampling: +{n_extra_small} copies from {len(small_in_split)} reads "
                  f"(ratio={small_ori_oversample_ratio})")

    # Extra oversampling pass for reads with large ORIs — helps model learn the
    # null-BrdU-flanked-by-forks pattern which is rare in gradient updates otherwise.
    if large_ori_read_ids and large_ori_oversample_ratio > 0:
        split_ids = set(partition_manifest["read_id"])
        large_in_split = [r for r in large_ori_read_ids if r in split_ids]
        if large_in_split:
            n_extra_large = int(len(large_in_split) * large_ori_oversample_ratio)
            sampled.extend(rng.choice(large_in_split, size=n_extra_large, replace=True).tolist())
            print(f"  Large ORI oversampling: +{n_extra_large} copies from {len(large_in_split)} reads "
                  f"(ratio={large_ori_oversample_ratio})")

    if background_ids:
        target_background = min(len(background_ids), max(len(sampled), 1))
        sampled.extend(rng.choice(background_ids, size=target_background, replace=False).tolist())

    rng.shuffle(sampled)
    return sampled


def _pad_sequences_with_weights(
    x_sequences: List[np.ndarray],
    y_sequences: List[np.ndarray],
    weights: List[np.ndarray],
    max_length: int | None = None,
    percentile: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    lengths = np.array([len(seq) for seq in x_sequences], dtype=int)
    if max_length is None:
        max_length = int(np.percentile(lengths, percentile))

    n_samples = len(x_sequences)
    n_channels = x_sequences[0].shape[1]

    x_padded = np.zeros((n_samples, max_length, n_channels), dtype=np.float32)
    y_padded = np.zeros((n_samples, max_length), dtype=np.int32)
    w_padded = np.zeros((n_samples, max_length), dtype=np.float32)

    for idx, (x_seq, y_seq, w_seq) in enumerate(zip(x_sequences, y_sequences, weights)):
        use_len = min(len(x_seq), max_length)
        x_padded[idx, :use_len, :] = x_seq[:use_len, :]
        y_padded[idx, :use_len] = y_seq[:use_len]
        w_padded[idx, :use_len] = w_seq[:use_len]

    return x_padded, y_padded, w_padded, max_length


def prepare_partition_data(
    xy_data: pd.DataFrame,
    manifest: pd.DataFrame,
    split_name: str,
    annotation_dict: Dict[str, pd.DataFrame],
    preprocessing_config: dict,
    labeling_config: dict,
    random_seed: int,
    small_ori_read_ids: List[str] | None = None,
    large_ori_read_ids: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, pd.DataFrame]:
    """Prepare one split using weak labels and shared representation logic."""
    partition_manifest = manifest[manifest["split"] == split_name].copy()
    print("\n" + "=" * 72)
    print(f"PREPARING SPLIT: {split_name.upper()}")
    print("=" * 72)
    print(f"Reads in manifest split: {len(partition_manifest):,}")
    print(f"Reads with any event: {int(partition_manifest['has_any_event'].sum()):,}")
    print(f"Reads without events: {int((~partition_manifest['has_any_event']).sum()):,}")
    sampled_read_ids = _sample_partition_read_ids(
        partition_manifest=partition_manifest,
        split_name=split_name,
        oversample_ratio=preprocessing_config.get("oversample_ratio", 0.0),
        random_seed=random_seed,
        small_ori_read_ids=small_ori_read_ids,
        small_ori_oversample_ratio=preprocessing_config.get("small_ori_oversample_ratio", 0.0),
        large_ori_read_ids=large_ori_read_ids,
        large_ori_oversample_ratio=preprocessing_config.get("large_ori_oversample_ratio", 0.0),
    )
    print(f"Reads processed after sampling policy: {len(sampled_read_ids):,}")

    x_sequences = []
    y_sequences = []
    weights = []
    info_rows = []

    flip_augment = preprocessing_config.get("flip_augment", False) and split_name == "train"
    n_flipped = 0

    for read_id in sampled_read_ids:
        read_df = xy_data[xy_data["read_id"] == read_id].copy()
        read_df = read_df.sort_values("start").reset_index(drop=True)
        if len(read_df) == 0:
            continue

        x_encoded = encode_read_dataframe(read_df, preprocessing_config)
        labels, sample_weight, stats = build_weak_labels_for_read(
            read_df=read_df,
            annotation_dict=annotation_dict,
            background_margin_bp=labeling_config.get("trusted_negative_margin_bp", 1000),
        )

        x_sequences.append(x_encoded)
        y_sequences.append(stats["y_safe"])
        weights.append(sample_weight)
        info_rows.append(stats)

        # Flip augmentation: reverse signal + swap left_fork↔right_fork labels.
        # Only applied to reads that actually have fork annotations.
        if flip_augment and (stats["n_left"] > 0 or stats["n_right"] > 0):
            x_flip = x_encoded[::-1, :].copy()
            y_orig = stats["y_safe"]
            y_flip = y_orig[::-1].copy()
            mask_l = (y_flip == 1)
            mask_r = (y_flip == 2)
            y_flip[mask_l] = 2
            y_flip[mask_r] = 1
            w_flip = sample_weight[::-1].copy()
            x_sequences.append(x_flip)
            y_sequences.append(y_flip)
            weights.append(w_flip)
            flip_stats = dict(stats)
            flip_stats["read_id"] = str(read_id) + "_flip"
            flip_stats["n_left"]  = stats["n_right"]
            flip_stats["n_right"] = stats["n_left"]
            info_rows.append(flip_stats)
            n_flipped += 1

        if len(info_rows) % 2000 == 0:
            print(f"  Encoded {len(info_rows):,} reads (incl. {n_flipped} flipped)...")

    if flip_augment:
        print(f"  Flip augmentation: {n_flipped} reads flipped and added to training set")

    info_df = pd.DataFrame(info_rows)
    x_padded, y_padded, w_padded, max_length = _pad_sequences_with_weights(
        x_sequences=x_sequences,
        y_sequences=y_sequences,
        weights=weights,
        max_length=preprocessing_config.get("max_length"),
        percentile=preprocessing_config.get("percentile", 100),
    )
    if not info_df.empty:
        print(f"Completed split: {split_name}")
        print(f"  Encoded reads: {len(info_df):,}")
        print(f"  Tensor shape X: {x_padded.shape}")
        print(f"  Tensor shape y: {y_padded.shape}")
        print(f"  Tensor shape weights: {w_padded.shape}")
        print(f"  Unknown windows masked: {int(info_df['n_unknown'].sum()):,}")
        print(f"  Background windows supervised: {int(info_df['n_background'].sum()):,}")
        print(f"  Left fork windows: {int(info_df['n_left'].sum()):,}")
        print(f"  Right fork windows: {int(info_df['n_right'].sum()):,}")
        print(f"  Origin windows: {int(info_df['n_origin'].sum()):,}")
        print(f"  Termination windows: {int(info_df['n_termination'].sum()):,}")
    return x_padded, y_padded, w_padded, max_length, info_df


def train_weak5_model(config: dict):
    """Train the CODEX weakly supervised 5-event model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("\n" + "=" * 72)
    print("CODEX WEAK 5-EVENT TRAINING")
    print("=" * 72)
    print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    print(f"TensorFlow CPU-only mode: {os.environ['CUDA_VISIBLE_DEVICES']}")

    preprocessed_path = config["data"].get("preprocessed_dataset_path")
    if preprocessed_path:
        return train_weak5_model_from_preprocessed(config, preprocessed_path)

    print("\n[1/7] Loading XY signal data...")
    xy_data = load_xy_data_cached(config)
    print("\n[2/7] Loading annotations and deriving terminations...")
    left_forks, right_forks, origins, terminations = load_annotations_for_codex(config)
    print(f"  Left forks: {len(left_forks):,}")
    print(f"  Right forks: {len(right_forks):,}")
    print(f"  Origins: {len(origins):,}")
    print(f"  Terminations: {len(terminations):,}")
    annotation_dict = build_annotation_dict(left_forks, right_forks, origins, terminations)

    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.tsv"

    print("\n[3/7] Building or loading split manifest...")
    if config["data"].get("split_manifest"):
        manifest = load_split_manifest(config["data"]["split_manifest"])
        print(f"  Loaded existing split manifest: {config['data']['split_manifest']}")
    elif manifest_path.exists():
        manifest = load_split_manifest(str(manifest_path))
        print(f"  Reusing split manifest: {manifest_path}")
    else:
        metadata = build_read_metadata(xy_data, left_forks, right_forks, origins, terminations)
        manifest = create_split_manifest(
            metadata=metadata,
            val_fraction=config["training"].get("val_fraction", 0.2),
            test_fraction=config["training"].get("test_fraction", 0.0),
            random_seed=config["training"].get("random_seed", 42),
        )
        save_split_manifest(manifest, str(manifest_path))
        print(f"  Created new split manifest: {manifest_path}")
    print(manifest["split"].value_counts().to_string())

    # Identify reads with small and large ORIs for targeted oversampling
    small_ori_max_bp = config["preprocessing"].get("small_ori_max_bp", 2000)
    large_ori_min_bp = config["preprocessing"].get("large_ori_min_bp", 20000)
    small_ori_reads: List[str] = []
    large_ori_reads: List[str] = []
    if len(origins) > 0:
        ori_lens = origins["end"] - origins["start"]
        if config["preprocessing"].get("small_ori_oversample_ratio", 0.0) > 0:
            small_ori_reads = list(origins.loc[ori_lens <= small_ori_max_bp, "read_id"].unique())
            print(f"\n  Small ORI reads (≤{small_ori_max_bp}bp): {len(small_ori_reads):,} "
                  f"(oversample ratio: {config['preprocessing']['small_ori_oversample_ratio']}×)")
        if config["preprocessing"].get("large_ori_oversample_ratio", 0.0) > 0:
            large_ori_reads = list(origins.loc[ori_lens >= large_ori_min_bp, "read_id"].unique())
            print(f"  Large ORI reads (≥{large_ori_min_bp}bp): {len(large_ori_reads):,} "
                  f"(oversample ratio: {config['preprocessing']['large_ori_oversample_ratio']}×)")

    print("\n[4/7] Preparing training tensors...")
    train_x, train_y, train_w, max_length, train_info = prepare_partition_data(
        xy_data=xy_data,
        manifest=manifest,
        split_name="train",
        annotation_dict=annotation_dict,
        preprocessing_config=config["preprocessing"],
        labeling_config=config["labeling"],
        random_seed=config["training"].get("random_seed", 42),
        small_ori_read_ids=small_ori_reads,
        large_ori_read_ids=large_ori_reads,
    )
    print("\n[5/7] Preparing validation tensors...")
    val_x, val_y, val_w, _, val_info = prepare_partition_data(
        xy_data=xy_data,
        manifest=manifest,
        split_name="val",
        annotation_dict=annotation_dict,
        preprocessing_config={**config["preprocessing"], "oversample_ratio": 0.0,
                              "small_ori_oversample_ratio": 0.0,
                              "large_ori_oversample_ratio": 0.0, "max_length": max_length},
        labeling_config=config["labeling"],
        random_seed=config["training"].get("random_seed", 42),
    )

    n_classes = config["model"].get("n_classes", 5)
    # Optionally save encoded tensors so future runs can skip re-encoding.
    save_path = config["data"].get("save_preprocessed_dataset_path")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n  Saving preprocessed tensors to: {save_path}")
        np.savez_compressed(
            save_path,
            train_x=train_x, train_y=train_y, train_w=train_w,
            val_x=val_x,   val_y=val_y,   val_w=val_w,
            max_length=np.array([max_length]),
        )
        # Save matching metadata so train_weak5_model_from_preprocessed can load it
        import yaml as _yaml
        meta_path = save_path.with_suffix(".metadata.yaml")
        with open(meta_path, "w") as _fh:
            _yaml.dump({
                "max_length": max_length,
                "n_channels": int(train_x.shape[2]),
                "class_names": CLASS_ID_TO_NAME,
                "split_manifest": str(config["data"].get("split_manifest", "")),
            }, _fh)
        print(f"  Saved. Point preprocessed_dataset_path at this file to skip re-encoding.")

    print("\n[6/7] Building and compiling model...")
    model = build_4class_fork_ori_model(
        max_length=max_length,
        n_channels=config["model"].get("n_channels", 9),
        n_classes=n_classes,
        cnn_filters=config["model"].get("cnn_filters", 64),
        lstm_units=config["model"].get("lstm_units", 128),
        dropout_rate=config["model"].get("dropout_rate", 0.3),
    )
    print(f"  Model input length: {max_length}")
    print(f"  Model channels: {config['model'].get('n_channels', 9)}")
    print(f"  Model classes: {n_classes}")
    print(f"  Total params: {model.count_params():,}")

    alpha = config["training"]["loss"].get("alpha", [1.0, 2.0, 2.0, 2.5, 2.5])
    loss = SparseCategoricalFocalLoss(
        alpha=alpha,
        gamma=config["training"]["loss"].get("gamma", 2.0),
    )
    # class indices: 0=background, 1=left_fork, 2=right_fork, 3=origin
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="masked_accuracy"),
        MaskedMeanIoU(n_classes=n_classes, exclude_background=True),
        MaskedMacroF1(n_classes=n_classes),
        MaskedClassPrecision(1, n_classes, name="masked_precision_left_fork"),
        MaskedClassRecall(1, n_classes, name="masked_recall_left_fork"),
        MaskedClassPrecision(2, n_classes, name="masked_precision_right_fork"),
        MaskedClassRecall(2, n_classes, name="masked_recall_right_fork"),
        MaskedClassPrecision(3, n_classes, name="masked_precision_origin"),
        MaskedClassRecall(3, n_classes, name="masked_recall_origin"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"].get("learning_rate", 5e-4),
            clipnorm=config["training"].get("clipnorm", None),
        ),
        loss=loss,
        metrics=metrics,
    )

    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config["output"]["model_filename"]
    callbacks = create_callbacks(config["training"], model_path=str(model_path))
    callbacks.append(TrainingProgressLogger(log_every=1))

    # Event-level IoU callback (ForkML-style) — added before EarlyStopping
    # so val_event_iou is in logs when checkpoint/early_stopping inspect them
    event_iou_cfg = config["training"].get("event_iou", {})
    if event_iou_cfg.get("enabled", True):
        callbacks.insert(0, EventLevelIoUCallback(
            val_x=val_x,
            val_y=val_y,
            val_w=val_w,
            prob_threshold=event_iou_cfg.get("prob_threshold", 0.4),
            max_gap_windows=event_iou_cfg.get("max_gap_windows", 5),
            n_classes=n_classes,
            batch_size=config["training"].get("batch_size", 128),
            compute_every=event_iou_cfg.get("compute_every", 1),
        ))

    # LR warmup: linearly ramp from start_lr to target_lr over warmup_epochs
    warmup_cfg = config["training"].get("lr_warmup", {})
    if warmup_cfg.get("enabled", False):
        target_lr = config["training"].get("learning_rate", 5e-4)
        warmup_epochs = warmup_cfg.get("epochs", 5)
        start_lr = target_lr * warmup_cfg.get("start_fraction", 0.1)

        class _LRWarmup(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                if epoch < warmup_epochs:
                    lr = start_lr + (target_lr - start_lr) * (epoch / warmup_epochs)
                    self.model.optimizer.learning_rate = lr
                    print(f"\n  [LR warmup] epoch {epoch+1}/{warmup_epochs}: lr={lr:.6f}")

        callbacks.append(_LRWarmup())
        print(f"\n  LR warmup: {warmup_epochs} epochs  {start_lr:.6f} → {target_lr:.6f}")

    print("\n[7/7] Starting model.fit()...")
    print(f"  Batch size: {config['training'].get('batch_size', 16)}")
    print(f"  Epochs: {config['training'].get('epochs', 100)}")
    print(f"  Learning rate: {config['training'].get('learning_rate', 5e-4)}")
    print(f"  Checkpoint path: {model_path}")
    history = model.fit(
        train_x,
        train_y,
        sample_weight=train_w,
        validation_data=(val_x, val_y, val_w),
        epochs=config["training"].get("epochs", 100),
        batch_size=config["training"].get("batch_size", 16),
        callbacks=callbacks,
        verbose=1,
    )

    train_info.to_csv(output_dir / "train_dataset_info.tsv", sep="\t", index=False)
    val_info.to_csv(output_dir / "val_dataset_info.tsv", sep="\t", index=False)
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.tsv", sep="\t", index=False)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False)

    metadata = {
        "max_length": max_length,
        "class_names": CLASS_ID_TO_NAME,
        "model_path": str(model_path),
        "split_manifest": str(manifest_path),
    }
    with open(output_dir / "training_metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(metadata, handle, default_flow_style=False)

    print("\nTraining artifacts written to:")
    print(f"  {output_dir / 'train_dataset_info.tsv'}")
    print(f"  {output_dir / 'val_dataset_info.tsv'}")
    print(f"  {output_dir / 'training_history.tsv'}")
    print(f"  {output_dir / 'training_metadata.yaml'}")

    return {
        "model": model,
        "history": history,
        "max_length": max_length,
        "train_info": train_info,
        "val_info": val_info,
        "split_manifest": manifest,
        "annotation_dict": annotation_dict,
    }


def _make_chunk_dataset(
    chunk_paths: list,
    max_length: int,
    n_channels: int,
    flip_augment: bool = False,
) -> "tf.data.Dataset":
    """Build a tf.data.Dataset that streams (x, y, weight) from chunk pickle files.

    Yields one sample at a time so only one chunk (~15 MB) is in RAM at once,
    avoiding the ~570 MB peak from loading the full train_x tensor.

    If flip_augment=True, each fork-containing sample is followed immediately by
    its time-reversed counterpart with left_fork↔right_fork labels swapped.
    This doubles the effective training set size for annotated reads and fixes
    the BiLSTM directional bias (left_fork under-prediction).
    """
    import gc
    import pickle

    def generator():
        for path in chunk_paths:
            with open(str(path), "rb") as fh:
                payload = pickle.load(fh)
            x_seqs = payload["x_sequences"]
            y_seqs = payload["y_sequences"]
            w_seqs = payload["weights"]
            del payload
            gc.collect()
            for x_seq, y_seq, w_seq in zip(x_seqs, y_seqs, w_seqs):
                use_len = min(len(x_seq), max_length)
                x_pad = np.zeros((max_length, n_channels), dtype=np.float32)
                y_pad = np.zeros((max_length,), dtype=np.int32)
                w_pad = np.zeros((max_length,), dtype=np.float32)
                x_pad[:use_len] = x_seq[:use_len]
                y_pad[:use_len] = y_seq[:use_len]
                w_pad[:use_len] = w_seq[:use_len]
                yield x_pad, y_pad, w_pad

                if flip_augment and (np.any(y_pad == 1) or np.any(y_pad == 2)):
                    x_flip = x_pad[::-1, :].copy()
                    y_flip = y_pad[::-1].copy()
                    mask_l = (y_flip == 1)
                    mask_r = (y_flip == 2)
                    y_flip[mask_l] = 2
                    y_flip[mask_r] = 1
                    w_flip = w_pad[::-1].copy()
                    yield x_flip, y_flip, w_flip

    output_sig = (
        tf.TensorSpec(shape=(max_length, n_channels), dtype=tf.float32),
        tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_length,), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_sig)


def train_weak5_model_from_preprocessed(config: dict, preprocessed_path: str):
    """Train from a saved preprocessed dataset.

    Preferred path: reads the companion .metadata.yaml to locate per-chunk
    pickle files and streams them one at a time, keeping peak RAM near the TF
    base footprint (~400 MB) instead of loading the full ~866 MB .npz.

    Fallback: if no metadata yaml is found, loads the full .npz (legacy path).
    """
    print("\n" + "=" * 70)
    print("CODEX TRAINING FROM PREPROCESSED DATA")
    print("=" * 70)
    print(f"✅ CPU cores: {os.cpu_count()}")
    print(f"✅ GPU disabled: {len(tf.config.list_physical_devices('GPU')) == 0}")

    preprocessed_path = Path(preprocessed_path)
    n_channels = config["model"].get("n_channels", 9)

    # ------------------------------------------------------------------
    # Determine data source: chunk streaming (preferred) or full npz.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING PREPROCESSED DATA")
    print("=" * 70)

    metadata_path = preprocessed_path.with_suffix(".metadata.yaml")
    use_chunks = metadata_path.exists()

    if use_chunks:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
        max_length = int(meta["max_length"])
        chunk_root = Path(meta["chunk_root"])
        if not chunk_root.is_absolute():
            # chunk_root was saved relative to the repo root (CWD at preprocess time).
            # Walk up from the metadata file until the path resolves to an existing dir.
            for base in metadata_path.parents:
                candidate = base / chunk_root
                if candidate.is_dir():
                    chunk_root = candidate
                    break
            else:
                chunk_root = Path.cwd() / chunk_root  # best-effort fallback
        train_chunk_paths = sorted((chunk_root / "train").glob("chunk_*.pkl"))
        val_chunk_paths = sorted((chunk_root / "val").glob("chunk_*.pkl"))
        if not train_chunk_paths:
            raise FileNotFoundError(
                f"No train chunk files found under {chunk_root / 'train'}. "
                "Run preprocess_weak4_codex.py first, or remove the metadata yaml "
                "to fall back to loading the full .npz."
            )
        print(f"📂 Chunk-to-memory mode (shuffle-friendly)")
        print(f"   Train chunks: {len(train_chunk_paths)}")
        print(f"   Val chunks:   {len(val_chunk_paths)}")
        print(f"   Max length:   {max_length} bins")
        print(f"   Channels:     {n_channels}")
    else:
        data = np.load(str(preprocessed_path))
        train_x = data["train_x"]
        train_y = data["train_y"]
        train_w = data["train_w"]
        val_x = data["val_x"]
        val_y = data["val_y"]
        val_w = data["val_w"]
        max_length = int(data["max_length"])
        print(f"📂 Full npz loaded (fallback mode)")
        print(f"   Train tensor: {train_x.shape}")
        print(f"   Val tensor:   {val_x.shape}")
        print(f"   Max length:   {max_length} bins")

    # ------------------------------------------------------------------
    # Build model.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    n_classes = config["model"].get("n_classes", 4)
    model = build_4class_fork_ori_model(
        max_length=max_length,
        n_channels=n_channels,
        n_classes=n_classes,
        cnn_filters=config["model"].get("cnn_filters", 64),
        lstm_units=config["model"].get("lstm_units", 128),
        dropout_rate=config["model"].get("dropout_rate", 0.3),
    )
    print(f"✅ Model built")
    print(f"   Input length: {max_length} bins × {n_channels} channels")
    print(f"   Classes:      {n_classes}")
    print(f"   Total params: {model.count_params():,}")

    alpha = config["training"]["loss"].get("alpha", [1.0, 2.0, 2.0, 2.5])
    loss = SparseCategoricalFocalLoss(
        alpha=alpha,
        gamma=config["training"]["loss"].get("gamma", 2.0),
    )
    # class indices: 0=background, 1=left_fork, 2=right_fork, 3=origin
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="masked_accuracy"),
        MaskedMeanIoU(n_classes=n_classes, exclude_background=True),
        MaskedMacroF1(n_classes=n_classes),
        MaskedClassPrecision(1, n_classes, name="masked_precision_left_fork"),
        MaskedClassRecall(1, n_classes, name="masked_recall_left_fork"),
        MaskedClassPrecision(2, n_classes, name="masked_precision_right_fork"),
        MaskedClassRecall(2, n_classes, name="masked_recall_right_fork"),
        MaskedClassPrecision(3, n_classes, name="masked_precision_origin"),
        MaskedClassRecall(3, n_classes, name="masked_recall_origin"),
    ]
    lr = config["training"].get("learning_rate", 5e-4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            clipnorm=config["training"].get("clipnorm", None),
        ),
        loss=loss,
        metrics=metrics,
    )
    print(f"✅ Model compiled  (lr={lr}, focal γ={config['training']['loss'].get('gamma', 2.0)}, α={alpha})")

    # ------------------------------------------------------------------
    # Callbacks — BackupAndRestore goes first so it restores model/optimizer
    # state before EarlyStopping or ReduceLROnPlateau inspect any metrics.
    # ------------------------------------------------------------------
    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config["output"]["model_filename"]
    backup_dir = str(model_dir / (Path(config["output"]["model_filename"]).stem + "_backup"))
    callbacks = create_callbacks(
        config["training"],
        model_path=str(model_path),
        backup_dir=backup_dir,
    )
    callbacks.append(TrainingProgressLogger(log_every=1))

    # ------------------------------------------------------------------
    # Train.
    # (EventLevelIoUCallback inserted after val data is loaded — see below)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    batch_size = config["training"].get("batch_size", 16)
    epochs = config["training"].get("epochs", 100)

    print(f"📊 Training config:")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {epochs}")
    print(f"   Model checkpoint: {model_path}")
    print(f"   Backup dir:       {backup_dir}")

    flip_augment = config["preprocessing"].get("flip_augment", False)
    if flip_augment:
        print(f"✅ Flip augmentation: ON (fork-containing reads will be doubled with reversed signal + swapped labels)")
    else:
        print(f"ℹ️  Flip augmentation: OFF")

    if use_chunks:
        # Load all chunks into memory so Keras can shuffle every epoch.
        # This eliminates the epoch-to-epoch oscillation caused by fixed chunk order.
        def _load_chunks(paths, do_flip=False):
            x_list, y_list, w_list = [], [], []
            for p in paths:
                with open(str(p), "rb") as fh:
                    payload = pickle.load(fh)
                for x_seq, y_seq, w_seq in zip(
                    payload["x_sequences"], payload["y_sequences"], payload["weights"]
                ):
                    use_len = min(len(x_seq), max_length)
                    x_pad = np.zeros((max_length, n_channels), dtype=np.float32)
                    y_pad = np.zeros((max_length,), dtype=np.int32)
                    w_pad = np.zeros((max_length,), dtype=np.float32)
                    x_pad[:use_len] = x_seq[:use_len]
                    y_pad[:use_len] = y_seq[:use_len]
                    w_pad[:use_len] = w_seq[:use_len]
                    x_list.append(x_pad)
                    y_list.append(y_pad)
                    w_list.append(w_pad)
                    if do_flip and (np.any(y_pad == 1) or np.any(y_pad == 2)):
                        x_flip = x_pad[::-1, :].copy()
                        y_flip = y_pad[::-1].copy()
                        mask_l, mask_r = (y_flip == 1), (y_flip == 2)
                        y_flip[mask_l] = 2
                        y_flip[mask_r] = 1
                        x_list.append(x_flip)
                        y_list.append(y_flip)
                        w_list.append(w_pad[::-1].copy())
            return (np.array(x_list, dtype=np.float32),
                    np.array(y_list, dtype=np.int32),
                    np.array(w_list, dtype=np.float32))

        print("   Loading train chunks into memory…")
        train_x, train_y, train_w = _load_chunks(train_chunk_paths, do_flip=flip_augment)
        print(f"   Train tensor: {train_x.shape}  ({train_x.nbytes / 1e9:.2f} GB)")

        print("   Loading val chunks into memory…")
        val_x, val_y, val_w = _load_chunks(val_chunk_paths, do_flip=False)
        print(f"   Val tensor:   {val_x.shape}  ({val_x.nbytes / 1e9:.2f} GB)")

        # Event-level IoU callback — inserted here so val_x is guaranteed loaded
        event_iou_cfg = config["training"].get("event_iou", {})
        if event_iou_cfg.get("enabled", True):
            callbacks.insert(0, EventLevelIoUCallback(
                val_x=val_x, val_y=val_y, val_w=val_w,
                prob_threshold=event_iou_cfg.get("prob_threshold", 0.4),
                max_gap_windows=event_iou_cfg.get("max_gap_windows", 5),
                n_classes=n_classes,
                batch_size=config["training"].get("batch_size", 128),
                compute_every=event_iou_cfg.get("compute_every", 1),
            ))

        history = model.fit(
            train_x, train_y,
            sample_weight=train_w,
            validation_data=(val_x, val_y, val_w),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,   # Keras re-shuffles every epoch → no more oscillation
            callbacks=callbacks,
            verbose=1,
        )
    else:
        # Event-level IoU callback — val_x already loaded from npz above
        event_iou_cfg = config["training"].get("event_iou", {})
        if event_iou_cfg.get("enabled", True):
            callbacks.insert(0, EventLevelIoUCallback(
                val_x=val_x, val_y=val_y, val_w=val_w,
                prob_threshold=event_iou_cfg.get("prob_threshold", 0.4),
                max_gap_windows=event_iou_cfg.get("max_gap_windows", 5),
                n_classes=n_classes,
                batch_size=config["training"].get("batch_size", 128),
                compute_every=event_iou_cfg.get("compute_every", 1),
            ))

        history = model.fit(
            train_x,
            train_y,
            sample_weight=train_w,
            validation_data=(val_x, val_y, val_w),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    pd.DataFrame(history.history).to_csv(output_dir / "training_history.tsv", sep="\t", index=False)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(config, handle, default_flow_style=False)

    return {
        "model": model,
        "history": history,
        "max_length": max_length,
        "train_info": None,
        "val_info": None,
        "split_manifest": None,
        "annotation_dict": None,
    }
