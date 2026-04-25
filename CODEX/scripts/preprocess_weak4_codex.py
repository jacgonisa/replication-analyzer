#!/usr/bin/env python
"""Preprocess the CODEX weak 4-class dataset once and save it for fast retries."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.annotations import load_annotations_for_codex
from replication_analyzer_codex.data_cache import load_xy_data_cached
from replication_analyzer_codex.representation import encode_read_dataframe
from replication_analyzer_codex.splits import (
    build_read_metadata,
    create_split_manifest,
    save_split_manifest,
)
from replication_analyzer_codex.weak_labels import (
    build_annotation_dict,
    build_weak_labels_for_read,
)


def _rss_mb() -> str:
    """Return current process RSS in MB as a short string for logging."""
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return f"{int(line.split()[1]) // 1024} MB RSS"
    except Exception:
        pass
    return "RSS unknown"


def _sample_partition_read_ids(
    partition_manifest: pd.DataFrame,
    split_name: str,
    oversample_ratio: float,
    random_seed: int,
) -> list[str]:
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

    if background_ids:
        target_background = min(len(background_ids), max(len(sampled), 1))
        sampled.extend(rng.choice(background_ids, size=target_background, replace=False).tolist())

    rng.shuffle(sampled)
    return sampled


def _pad_sequences_with_weights(
    x_sequences: list[np.ndarray],
    y_sequences: list[np.ndarray],
    weights: list[np.ndarray],
    max_length: int | None = None,
    percentile: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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


def _chunk_path(base_dir: Path, split_name: str, chunk_index: int, start: int, end: int) -> Path:
    return base_dir / split_name / f"chunk_{chunk_index:04d}_{start:06d}_{end - 1:06d}.pkl"


def _encode_partition_chunks(
    xy_by_read: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    split_name: str,
    annotation_dict: dict[str, pd.DataFrame],
    preprocessing_config: dict,
    labeling_config: dict,
    random_seed: int,
    chunk_root: Path,
    chunk_size: int,
) -> tuple[list[Path], pd.DataFrame]:
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
    )
    print(f"Reads processed after sampling policy: {len(sampled_read_ids):,}")

    split_dir = chunk_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    info_rows: list[dict] = []
    total_chunks = (len(sampled_read_ids) + chunk_size - 1) // chunk_size

    for chunk_index, chunk_start in enumerate(range(0, len(sampled_read_ids), chunk_size), start=1):
        chunk_end = min(len(sampled_read_ids), chunk_start + chunk_size)
        path = _chunk_path(split_dir.parent, split_name, chunk_index, chunk_start, chunk_end)
        chunk_paths.append(path)

        if path.exists():
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            info_rows.extend(payload["info_rows"])
            # Free x_sequences/y_sequences/weights immediately; only info_rows needed.
            del payload
            gc.collect()
            print(
                f"  Reused {split_name} chunk {chunk_index}/{total_chunks}: "
                f"reads {chunk_start:,}-{chunk_end - 1:,}"
                f"  [{_rss_mb()}]"
            )
            continue

        flip_augment = preprocessing_config.get("flip_augment", False)
        x_sequences = []
        y_sequences = []
        weights = []
        chunk_info_rows = []
        for read_id in sampled_read_ids[chunk_start:chunk_end]:
            read_df = xy_by_read.get(read_id)
            if read_df is None or len(read_df) == 0:
                continue

            x_encoded = encode_read_dataframe(read_df, preprocessing_config)
            _, sample_weight, stats = build_weak_labels_for_read(
                read_df=read_df,
                annotation_dict=annotation_dict,
                background_margin_bp=labeling_config.get("trusted_negative_margin_bp", 1000),
            )

            x_sequences.append(x_encoded)
            y_sequences.append(stats["y_safe"])
            weights.append(sample_weight)
            chunk_info_rows.append(stats)

            # Flip augmentation: reverse the read and swap left_fork ↔ right_fork.
            # Only applied to reads that have at least one fork window, and only
            # during training (not val — val is never augmented by the caller).
            if flip_augment and (stats["n_left"] > 0 or stats["n_right"] > 0):
                x_flip = x_encoded[::-1, :].copy()
                y_flip = stats["y_safe"][::-1].copy()
                w_flip = sample_weight[::-1].copy()
                # Swap class IDs 1 (left_fork) ↔ 2 (right_fork)
                y_swap = y_flip.copy()
                y_swap[y_flip == 1] = 2
                y_swap[y_flip == 2] = 1
                x_sequences.append(x_flip)
                y_sequences.append(y_swap)
                weights.append(w_flip)
                flip_stats = dict(stats)
                flip_stats["read_id"] = str(read_id) + "_flip"
                flip_stats["n_left"] = stats["n_right"]
                flip_stats["n_right"] = stats["n_left"]
                chunk_info_rows.append(flip_stats)

        payload = {
            "x_sequences": x_sequences,
            "y_sequences": y_sequences,
            "weights": weights,
            "info_rows": chunk_info_rows,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        info_rows.extend(chunk_info_rows)
        del x_sequences, y_sequences, weights, chunk_info_rows, payload
        gc.collect()
        print(
            f"  Saved {split_name} chunk {chunk_index}/{total_chunks}: "
            f"reads {chunk_start:,}-{chunk_end - 1:,} ({len(info_rows):,} total encoded)"
            f"  [{_rss_mb()}]",
            flush=True,
        )

    info_df = pd.DataFrame(info_rows)
    if not info_df.empty:
        print(f"Completed split chunking: {split_name}")
        print(f"  Encoded reads: {len(info_df):,}")
        print(f"  Unknown windows masked: {int(info_df['n_unknown'].sum()):,}")
        print(f"  Background windows supervised: {int(info_df['n_background'].sum()):,}")
        print(f"  Left fork windows: {int(info_df['n_left'].sum()):,}")
        print(f"  Right fork windows: {int(info_df['n_right'].sum()):,}")
        print(f"  Origin windows: {int(info_df['n_origin'].sum()):,}")
    return chunk_paths, info_df


def _load_partition_from_chunks(
    chunk_paths: list[Path],
    max_length: int | None,
    percentile: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x_sequences: list[np.ndarray] = []
    y_sequences: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for path in chunk_paths:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        x_sequences.extend(payload["x_sequences"])
        y_sequences.extend(payload["y_sequences"])
        weights.extend(payload["weights"])
        del payload  # free info_rows (y_safe arrays etc.) immediately
        gc.collect()

    return _pad_sequences_with_weights(
        x_sequences=x_sequences,
        y_sequences=y_sequences,
        weights=weights,
        max_length=max_length,
        percentile=percentile,
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess CODEX weak 4-class dataset")
    parser.add_argument("--config", required=True, help="Path to CODEX YAML config")
    parser.add_argument("--output", required=True, help="Output .npz path")
    args = parser.parse_args()

    print(f"[{datetime.now().isoformat(timespec='seconds')}] preprocess_weak4_codex.py starting", flush=True)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Config loaded", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_root = output_path.parent / f"{output_path.stem}_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)

    xy_data = load_xy_data_cached(config)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] XY data loaded  [{_rss_mb()}]", flush=True)
    xy_data = xy_data.sort_values(["read_id", "start"]).reset_index(drop=True)
    xy_by_read = {
        read_id: group.reset_index(drop=True)
        for read_id, group in xy_data.groupby("read_id", sort=False)
    }
    print(f"[{datetime.now().isoformat(timespec='seconds')}] xy_by_read built  [{_rss_mb()}]", flush=True)

    left_forks, right_forks, origins, terminations = load_annotations_for_codex(config)
    annotation_dict = build_annotation_dict(left_forks, right_forks, origins, terminations)

    metadata = build_read_metadata(xy_data, left_forks, right_forks, origins, terminations)
    manifest = create_split_manifest(
        metadata=metadata,
        val_fraction=config["training"].get("val_fraction", 0.2),
        test_fraction=config["training"].get("test_fraction", 0.0),
        random_seed=config["training"].get("random_seed", 42),
    )
    save_split_manifest(manifest, str(output_path.with_suffix(".split_manifest.tsv")))

    # xy_data is no longer needed; xy_by_read holds per-read copies.
    # Keeping both simultaneously doubles memory for the entire dataset.
    del xy_data
    gc.collect()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] xy_data freed  [{_rss_mb()}]", flush=True)

    chunk_size = int(config["preprocessing"].get("encode_chunk_size", 1000))
    random_seed = config["training"].get("random_seed", 42)

    train_chunk_paths, train_info = _encode_partition_chunks(
        xy_by_read=xy_by_read,
        manifest=manifest,
        split_name="train",
        annotation_dict=annotation_dict,
        preprocessing_config=config["preprocessing"],
        labeling_config=config["labeling"],
        random_seed=random_seed,
        chunk_root=chunk_root,
        chunk_size=chunk_size,
    )
    val_chunk_paths, val_info = _encode_partition_chunks(
        xy_by_read=xy_by_read,
        manifest=manifest,
        split_name="val",
        annotation_dict=annotation_dict,
        preprocessing_config={**config["preprocessing"], "oversample_ratio": 0.0},
        labeling_config=config["labeling"],
        random_seed=random_seed,
        chunk_root=chunk_root,
        chunk_size=chunk_size,
    )

    # All chunking done — xy_by_read and related objects no longer needed.
    # Free ~700 MB before the assembly phase allocates padded tensors.
    del xy_by_read, annotation_dict, manifest
    gc.collect()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Freed xy_by_read before assembly  [{_rss_mb()}]", flush=True)

    print(f"Assembling padded tensors from chunk checkpoints...  [{_rss_mb()}]", flush=True)
    train_x, train_y, train_w, max_length = _load_partition_from_chunks(
        chunk_paths=train_chunk_paths,
        max_length=config["preprocessing"].get("max_length"),
        percentile=config["preprocessing"].get("percentile", 100),
    )
    print(f"Train tensors assembled: shape={train_x.shape}  [{_rss_mb()}]", flush=True)
    val_x, val_y, val_w, _ = _load_partition_from_chunks(
        chunk_paths=val_chunk_paths,
        max_length=max_length,
        percentile=config["preprocessing"].get("percentile", 100),
    )
    print(f"Val tensors assembled: shape={val_x.shape}  [{_rss_mb()}]", flush=True)

    print(f"Saving npz...  [{_rss_mb()}]", flush=True)
    np.savez_compressed(
        output_path,
        train_x=train_x,
        train_y=train_y,
        train_w=train_w,
        val_x=val_x,
        val_y=val_y,
        val_w=val_w,
        max_length=max_length,
    )
    train_info.to_csv(output_path.with_suffix(".train_info.tsv"), sep="\t", index=False)
    val_info.to_csv(output_path.with_suffix(".val_info.tsv"), sep="\t", index=False)

    metadata_out = {
        "created_at": datetime.now().isoformat(),
        "config": args.config,
        "output": str(output_path),
        "chunk_root": str(chunk_root),
        "train_shape": list(train_x.shape),
        "val_shape": list(val_x.shape),
        "max_length": int(max_length),
    }
    with open(output_path.with_suffix(".metadata.yaml"), "w", encoding="utf-8") as handle:
        yaml.dump(metadata_out, handle, default_flow_style=False)

    print(f"Saved preprocessed dataset: {output_path}")
    print(f"Train shape: {train_x.shape}")
    print(f"Val shape: {val_x.shape}")


if __name__ == "__main__":
    main()
