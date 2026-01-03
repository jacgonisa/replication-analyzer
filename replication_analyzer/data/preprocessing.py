"""
Data preprocessing utilities for training deep learning models.

This module handles:
- Hybrid balancing (oversampling + undersampling)
- Sequence padding
- Label creation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from .encoding import encode_signal_enhanced


def create_ori_labels(read_df: pd.DataFrame, ori_annotations: pd.DataFrame) -> np.ndarray:
    """
    Create binary labels for ORI detection (segment-level).

    Parameters
    ----------
    read_df : pd.DataFrame
        Read segments with columns: start, end, read_id
    ori_annotations : pd.DataFrame
        ORI annotations with columns: start, end, read_id

    Returns
    -------
    np.ndarray
        Binary labels (1 = ORI, 0 = non-ORI)
    """
    labels = np.zeros(len(read_df), dtype=np.float32)
    read_id = read_df['read_id'].iloc[0]

    oris_in_read = ori_annotations[ori_annotations['read_id'] == read_id]

    for idx, (i, seg) in enumerate(read_df.iterrows()):
        seg_start, seg_end = seg['start'], seg['end']
        for _, ori in oris_in_read.iterrows():
            overlap_start = max(seg_start, ori['start'])
            overlap_end = min(seg_end, ori['end'])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                labels[idx] = 1.0
                break

    return labels


def create_fork_labels(read_df: pd.DataFrame,
                       left_forks: pd.DataFrame,
                       right_forks: pd.DataFrame) -> np.ndarray:
    """
    Create 3-class labels for fork detection (segment-level).

    Parameters
    ----------
    read_df : pd.DataFrame
        Read segments with columns: start, end, read_id
    left_forks : pd.DataFrame
        Left fork annotations
    right_forks : pd.DataFrame
        Right fork annotations

    Returns
    -------
    np.ndarray
        3-class labels (0 = background, 1 = left_fork, 2 = right_fork)
    """
    labels = np.zeros(len(read_df), dtype=np.int32)
    read_id = read_df['read_id'].iloc[0]

    # Label left forks (class 1)
    left_forks_read = left_forks[left_forks['read_id'] == read_id]
    for _, fork in left_forks_read.iterrows():
        for idx, (i, seg) in enumerate(read_df.iterrows()):
            seg_start, seg_end = seg['start'], seg['end']
            overlap = max(0, min(seg_end, fork['end']) - max(seg_start, fork['start']))
            if overlap > 0:
                labels[idx] = 1  # Left fork

    # Label right forks (class 2) - only if not already left fork
    right_forks_read = right_forks[right_forks['read_id'] == read_id]
    for _, fork in right_forks_read.iterrows():
        for idx, (i, seg) in enumerate(read_df.iterrows()):
            seg_start, seg_end = seg['start'], seg['end']
            overlap = max(0, min(seg_end, fork['end']) - max(seg_start, fork['start']))
            if overlap > 0 and labels[idx] == 0:
                labels[idx] = 2  # Right fork

    return labels


def prepare_ori_data_hybrid(xy_data: pd.DataFrame,
                            ori_annotations: pd.DataFrame,
                            oversample_ratio: float = 0.5,
                            use_enhanced_encoding: bool = True,
                            random_seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], pd.DataFrame]:
    """
    Prepare ORI detection data with hybrid balancing.

    Strategy:
    1. Oversample ORI reads (minority) by duplicating
    2. Undersample non-ORI reads (majority) to match
    3. Result: ~1:1 balance

    Parameters
    ----------
    xy_data : pd.DataFrame
        XY signal data
    ori_annotations : pd.DataFrame
        ORI annotations
    oversample_ratio : float
        Fraction of ORI reads to duplicate (0.5 = add 50% more)
    use_enhanced_encoding : bool
        If True, use 9-channel encoding; if False, use 6-channel
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X_sequences, y_sequences, info_df)
    """
    from .encoding import encode_signal_enhanced, encode_signal_basic

    print("="*60)
    print("PREPARING ORI DATA - HYBRID BALANCING")
    print("="*60)

    # Identify ORI and non-ORI reads
    ori_read_ids = set(ori_annotations['read_id'].unique())
    all_read_ids = xy_data['read_id'].unique()
    non_ori_read_ids = [rid for rid in all_read_ids if rid not in ori_read_ids]

    n_ori_initial = len(ori_read_ids)

    print(f"\n📊 Initial dataset:")
    print(f"   Reads with ORIs: {n_ori_initial:,}")
    print(f"   Reads without ORIs: {len(non_ori_read_ids):,}")

    # Hybrid sampling
    np.random.seed(random_seed)

    # 1. Oversample ORI reads
    n_ori_target = int(n_ori_initial * (1 + oversample_ratio))
    reads_to_process = list(ori_read_ids)

    if n_ori_target > n_ori_initial:
        n_to_duplicate = n_ori_target - n_ori_initial
        reads_to_duplicate = np.random.choice(
            list(ori_read_ids),
            size=min(n_to_duplicate, n_ori_initial),
            replace=True
        )
        reads_to_process.extend(list(reads_to_duplicate))

    # 2. Undersample non-ORI reads
    n_non_ori_target = len(reads_to_process)
    sampled_non_ori_ids = np.random.choice(
        non_ori_read_ids,
        size=min(n_non_ori_target, len(non_ori_read_ids)),
        replace=False
    )

    reads_to_process.extend(list(sampled_non_ori_ids))
    np.random.shuffle(reads_to_process)

    print(f"\n⚡ Hybrid balanced dataset:")
    print(f"   Total reads: {len(reads_to_process):,}")
    print(f"   ORI reads (with oversampling): {len(reads_to_process) - len(sampled_non_ori_ids):,}")
    print(f"   Non-ORI reads (undersampled): {len(sampled_non_ori_ids):,}")

    # Process reads
    encoder = encode_signal_enhanced if use_enhanced_encoding else encode_signal_basic
    X_sequences = []
    y_sequences = []
    read_info = []

    processed = 0
    for read_id in reads_to_process:
        processed += 1
        if processed % 2000 == 0:
            print(f"  Processed {processed}/{len(reads_to_process)} reads...")

        read_df = xy_data[xy_data['read_id'] == read_id].copy()
        read_df = read_df.sort_values('start').reset_index(drop=True)

        if len(read_df) == 0:
            continue

        # Encode signal
        signal = read_df['signal'].values
        X_encoded = encoder(signal)

        # Create labels
        y_labels = create_ori_labels(read_df, ori_annotations)

        X_sequences.append(X_encoded)
        y_sequences.append(y_labels)

        read_info.append({
            'read_id': read_id,
            'length': len(read_df),
            'has_ori': y_labels.sum() > 0,
            'n_ori_segments': int(y_labels.sum()),
            'n_oris': len(ori_annotations[ori_annotations['read_id'] == read_id])
        })

    info_df = pd.DataFrame(read_info)

    print(f"\n✅ Data prepared!")
    print(f"   Total reads: {len(X_sequences):,}")
    print(f"   Reads with ORIs: {info_df['has_ori'].sum():,}")
    print(f"   Reads without ORIs: {(~info_df['has_ori']).sum():,}\n")

    return X_sequences, y_sequences, info_df


def prepare_fork_data_hybrid(xy_data: pd.DataFrame,
                             left_forks: pd.DataFrame,
                             right_forks: pd.DataFrame,
                             oversample_ratio: float = 0.5,
                             use_enhanced_encoding: bool = True,
                             random_seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray], pd.DataFrame]:
    """
    Prepare fork detection data with hybrid balancing (3-class).

    Parameters
    ----------
    xy_data : pd.DataFrame
        XY signal data
    left_forks : pd.DataFrame
        Left fork annotations
    right_forks : pd.DataFrame
        Right fork annotations
    oversample_ratio : float
        Fraction of fork reads to duplicate
    use_enhanced_encoding : bool
        If True, use 9-channel encoding
    random_seed : int
        Random seed

    Returns
    -------
    tuple
        (X_sequences, y_sequences, info_df)
    """
    from .encoding import encode_signal_enhanced, encode_signal_basic

    print("="*60)
    print("PREPARING FORK DATA - HYBRID BALANCING (3-CLASS)")
    print("="*60)

    # Identify fork and non-fork reads
    left_fork_reads = set(left_forks['read_id'].unique())
    right_fork_reads = set(right_forks['read_id'].unique())
    fork_reads = left_fork_reads | right_fork_reads

    all_read_ids = xy_data['read_id'].unique()
    non_fork_reads = [rid for rid in all_read_ids if rid not in fork_reads]

    n_fork_initial = len(fork_reads)

    print(f"\n📊 Initial dataset:")
    print(f"   Reads with forks: {n_fork_initial:,}")
    print(f"     - Only left: {len(left_fork_reads - right_fork_reads):,}")
    print(f"     - Only right: {len(right_fork_reads - left_fork_reads):,}")
    print(f"     - Both: {len(left_fork_reads & right_fork_reads):,}")
    print(f"   Reads without forks: {len(non_fork_reads):,}")

    # Hybrid sampling
    np.random.seed(random_seed)

    # 1. Oversample fork reads
    n_fork_target = int(n_fork_initial * (1 + oversample_ratio))
    reads_to_process = list(fork_reads)

    if n_fork_target > n_fork_initial:
        n_to_duplicate = n_fork_target - n_fork_initial
        reads_to_duplicate = np.random.choice(
            list(fork_reads),
            size=min(n_to_duplicate, n_fork_initial),
            replace=True
        )
        reads_to_process.extend(list(reads_to_duplicate))

    # 2. Undersample non-fork reads
    n_non_fork_target = len(reads_to_process)
    sampled_non_fork_ids = np.random.choice(
        non_fork_reads,
        size=min(n_non_fork_target, len(non_fork_reads)),
        replace=False
    )

    reads_to_process.extend(list(sampled_non_fork_ids))
    np.random.shuffle(reads_to_process)

    print(f"\n⚡ Hybrid balanced dataset:")
    print(f"   Total reads: {len(reads_to_process):,}")

    # Process reads
    encoder = encode_signal_enhanced if use_enhanced_encoding else encode_signal_basic
    X_sequences = []
    y_sequences = []
    read_info = []

    processed = 0
    for read_id in reads_to_process:
        processed += 1
        if processed % 2000 == 0:
            print(f"  Processed {processed}/{len(reads_to_process)} reads...")

        read_df = xy_data[xy_data['read_id'] == read_id].copy()
        read_df = read_df.sort_values('start').reset_index(drop=True)

        if len(read_df) == 0:
            continue

        # Encode signal
        signal = read_df['signal'].values
        X_encoded = encoder(signal)

        # Create 3-class labels
        y_labels = create_fork_labels(read_df, left_forks, right_forks)

        X_sequences.append(X_encoded)
        y_sequences.append(y_labels)

        read_info.append({
            'read_id': read_id,
            'length': len(read_df),
            'has_fork': (y_labels > 0).any(),
            'has_left': (y_labels == 1).any(),
            'has_right': (y_labels == 2).any(),
            'n_background': np.sum(y_labels == 0),
            'n_left': np.sum(y_labels == 1),
            'n_right': np.sum(y_labels == 2)
        })

    info_df = pd.DataFrame(read_info)

    print(f"\n✅ Data prepared!")
    print(f"   Total reads: {len(X_sequences):,}")
    print(f"   With forks: {info_df['has_fork'].sum():,}")
    print(f"   Without forks: {(~info_df['has_fork']).sum():,}\n")

    return X_sequences, y_sequences, info_df


def pad_sequences(X_sequences: List[np.ndarray],
                  y_sequences: List[np.ndarray],
                  percentile: int = 95,
                  max_length: int = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Pad variable-length sequences to uniform length.

    Parameters
    ----------
    X_sequences : list of np.ndarray
        List of encoded sequences
    y_sequences : list of np.ndarray
        List of label sequences
    percentile : int
        Use this percentile of lengths as max_length (if max_length not specified)
    max_length : int, optional
        Explicit max length (overrides percentile)

    Returns
    -------
    tuple
        (X_padded, y_padded, max_length)
    """
    lengths = np.array([len(x) for x in X_sequences])

    if max_length is None:
        max_length = int(np.percentile(lengths, percentile))

    print(f"\n📊 Sequence length stats:")
    print(f"   Min: {lengths.min()}, Median: {int(np.median(lengths))}")
    print(f"   {percentile}th percentile: {int(np.percentile(lengths, percentile))}, Max: {lengths.max()}")
    print(f"   Using max_length = {max_length}")

    n_samples = len(X_sequences)
    n_channels = X_sequences[0].shape[1]

    X_padded = np.zeros((n_samples, max_length, n_channels), dtype=np.float32)

    # Determine dtype for y_padded based on first sequence
    y_dtype = y_sequences[0].dtype
    y_padded = np.zeros((n_samples, max_length), dtype=y_dtype)

    for i, (x, y) in enumerate(zip(X_sequences, y_sequences)):
        length = min(len(x), max_length)
        X_padded[i, :length, :] = x[:length, :]
        y_padded[i, :length] = y[:length]

    return X_padded, y_padded, max_length
