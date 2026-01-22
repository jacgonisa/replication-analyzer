#!/usr/bin/env python
"""
Plot best and worst ORIGIN predictions to understand model performance.

This script visualizes complete origins (L→R fork patterns), not individual forks.

Selects reads based on:
- Best: Origins with high Jaccard overlap with curated origins
- Worst: False positive origins (no match to curated) with long length
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.data.loaders import load_all_xy_data
from replication_analyzer.evaluation.predictors import predict_on_read
from replication_analyzer.visualization.read_plots import plot_read_prediction
from replication_analyzer.models.base import SelfAttention
from replication_analyzer.evaluation.bed_utils import compute_jaccard, compute_overlap
import tensorflow as tf


def load_origins_bed(bed_path):
    """Load origin BED file."""
    df = pd.read_csv(bed_path, sep='\t', header=None)

    # Handle variable column formats
    if df.shape[1] >= 4:
        df = df.iloc[:, :4].copy()
        df.columns = ['chr', 'start', 'end', 'read_id']
    else:
        raise ValueError(f"BED file must have at least 4 columns, got {df.shape[1]}")

    return df


def load_curated_origins(bed_path):
    """Load curated origins BED file."""
    df = pd.read_csv(bed_path, sep='\t', header=None)

    # Curated origins format: chr, start, end, name, score, strand
    if df.shape[1] >= 3:
        df = df.iloc[:, :3].copy()
        df.columns = ['chr', 'start', 'end']
    else:
        raise ValueError(f"Curated BED file must have at least 3 columns")

    return df


def score_origin_quality(pred_row, curated_df):
    """
    Score a predicted origin by finding best match with curated origins.
    Returns Jaccard score (0-1, higher is better match).
    """
    # Find curated origins on same chromosome
    curated_chr = curated_df[curated_df['chr'] == pred_row['chr']]

    if len(curated_chr) == 0:
        return 0.0  # No curated origins on this chromosome

    # Compute Jaccard with all curated origins, take max
    best_jaccard = 0.0
    for _, cur_row in curated_chr.iterrows():
        jaccard = compute_jaccard(
            (pred_row['start'], pred_row['end']),
            (cur_row['start'], cur_row['end'])
        )
        if jaccard > best_jaccard:
            best_jaccard = jaccard

    return best_jaccard


def main():
    print("=" * 70)
    print("PLOTTING BEST AND WORST ORIGIN PREDICTIONS")
    print("=" * 70)

    # Load data
    print("\n1. Loading XY data...")
    xy_data = load_all_xy_data(
        base_dir="/mnt/ssd-4tb/crisanto_project/data_2025Oct/data_reads_minLen30000_nascent40/NM30_Col0",
        run_dirs=["NM30_plot_data_1strun_xy", "NM30_plot_data_2ndrun_xy"]
    )
    print(f"   Loaded {xy_data['read_id'].nunique():,} reads")

    # Load predicted origins
    print("\n2. Loading AI-predicted origins...")
    pred_origins = load_origins_bed("results/ori_calling_ai_pipeline/predicted_origins.bed")
    print(f"   Loaded {len(pred_origins):,} predicted origins from {pred_origins['read_id'].nunique():,} reads")

    # Load curated origins
    print("\n3. Loading curated origins...")
    curated_origins = load_curated_origins(
        "/mnt/ssd-4tb/crisanto_project/data_2025Oct/DNAscent_Col0_NM30_ColCEN_ORIs_curated_final.bed"
    )
    print(f"   Loaded {len(curated_origins):,} curated origins")

    # Score all predicted origins
    print("\n4. Scoring origin predictions...")
    pred_origins['jaccard'] = pred_origins.apply(
        lambda row: score_origin_quality(row, curated_origins), axis=1
    )
    pred_origins['length'] = pred_origins['end'] - pred_origins['start']

    print(f"   Mean Jaccard score: {pred_origins['jaccard'].mean():.3f}")
    print(f"   Origins with Jaccard > 0: {(pred_origins['jaccard'] > 0).sum():,} ({100*(pred_origins['jaccard'] > 0).sum()/len(pred_origins):.1f}%)")

    # Find best origins (high Jaccard, reasonable length)
    print("\n5. Finding best origin predictions...")
    best_origins = pred_origins[pred_origins['jaccard'] > 0.3].nlargest(10, 'jaccard')
    print(f"\n   Top 10 origins (Jaccard > 0.3):")
    print(best_origins[['read_id', 'chr', 'start', 'end', 'length', 'jaccard']].to_string(index=False))

    # Find worst origins (false positives: Jaccard = 0, long length)
    print("\n6. Finding worst origin predictions (false positives)...")
    false_positives = pred_origins[pred_origins['jaccard'] == 0]
    worst_origins = false_positives.nlargest(10, 'length')
    print(f"\n   Bottom 10 origins (Jaccard = 0, longest):")
    print(worst_origins[['read_id', 'chr', 'start', 'end', 'length', 'jaccard']].to_string(index=False))

    # Load model for prediction
    print("\n7. Loading model...")
    custom_objects = {'SelfAttention': SelfAttention}
    model = tf.keras.models.load_model(
        "models/case_study_jan2026/combined_fork_detector.keras",
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )
    print("   ✅ Model loaded")

    # Plot best 5 origins
    print("\n8. Plotting best origin predictions...")
    output_dir = Path("results/origin_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (idx, row) in enumerate(best_origins.head(5).iterrows()):
        read_id = row['read_id']
        print(f"\n   Plotting best origin {i+1}/5: {read_id}")
        print(f"      Chr={row['chr']}, Position={row['start']}-{row['end']}, Length={row['length']:,}bp, Jaccard={row['jaccard']:.3f}")

        # Get read data
        read_data = xy_data[xy_data['read_id'] == read_id].copy()
        if len(read_data) == 0:
            print(f"      ⚠️  No XY data found for read {read_id}, skipping...")
            continue

        read_data = read_data.sort_values('start').reset_index(drop=True)
        read_data['center'] = (read_data['start'] + read_data['end']) / 2

        # Get predictions
        read_df, y_pred = predict_on_read(model, read_data, 411, use_enhanced_encoding=True)

        # Get curated origins overlapping this origin
        curated_chr = curated_origins[curated_origins['chr'] == row['chr']]
        overlapping_curated = curated_chr[
            (curated_chr['start'] < row['end']) & (curated_chr['end'] > row['start'])
        ]

        # Plot
        plot_read_prediction(
            read_df, y_pred, annotations=overlapping_curated,
            save_path=output_dir / f"best_origin_{i+1}_{read_id[:8]}.png",
            title=f"Best Origin #{i+1}: {read_id[:16]}... (Jaccard={row['jaccard']:.3f}, Len={row['length']:,}bp)"
        )
        print(f"      ✅ Saved plot")

    # Plot worst 5 origins (false positives)
    print("\n9. Plotting worst origin predictions (false positives)...")
    for i, (idx, row) in enumerate(worst_origins.head(5).iterrows()):
        read_id = row['read_id']
        print(f"\n   Plotting worst origin {i+1}/5: {read_id}")
        print(f"      Chr={row['chr']}, Position={row['start']}-{row['end']}, Length={row['length']:,}bp, Jaccard={row['jaccard']:.3f}")

        # Get read data
        read_data = xy_data[xy_data['read_id'] == read_id].copy()
        if len(read_data) == 0:
            print(f"      ⚠️  No XY data found for read {read_id}, skipping...")
            continue

        read_data = read_data.sort_values('start').reset_index(drop=True)
        read_data['center'] = (read_data['start'] + read_data['end']) / 2

        # Get predictions
        read_df, y_pred = predict_on_read(model, read_data, 411, use_enhanced_encoding=True)

        # No curated origins overlap (Jaccard = 0)
        overlapping_curated = pd.DataFrame(columns=['chr', 'start', 'end'])

        # Plot
        plot_read_prediction(
            read_df, y_pred, annotations=overlapping_curated,
            save_path=output_dir / f"worst_origin_{i+1}_{read_id[:8]}.png",
            title=f"False Positive #{i+1}: {read_id[:16]}... (Jaccard=0.0, Len={row['length']:,}bp)"
        )
        print(f"      ✅ Saved plot")

    print("\n" + "=" * 70)
    print(f"✅ DONE! Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
