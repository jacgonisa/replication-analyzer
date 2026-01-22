#!/usr/bin/env python
"""
Plot best and worst fork predictions to understand model performance.

Selects reads based on:
- Best: High confidence predictions matching DNAscent annotations
- Worst: False positives or false negatives with high confidence
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
import tensorflow as tf


def load_dnascent_forks(left_bed, right_bed):
    """Load DNAscent fork annotations."""
    left_df = pd.read_csv(left_bed, sep='\t', header=None, usecols=[0,1,2,3])
    left_df.columns = ['chr', 'start', 'end', 'read_id']
    left_df['type'] = 'left'

    right_df = pd.read_csv(right_bed, sep='\t', header=None, usecols=[0,1,2,3])
    right_df.columns = ['chr', 'start', 'end', 'read_id']
    right_df['type'] = 'right'

    return pd.concat([left_df, right_df], ignore_index=True)


def compute_overlap(pred_start, pred_end, ann_start, ann_end):
    """Compute overlap between prediction and annotation."""
    overlap_start = max(pred_start, ann_start)
    overlap_end = min(pred_end, ann_end)
    overlap = max(0, overlap_end - overlap_start)
    return overlap


def score_read_predictions(read_id, xy_data, predictions_df, annotations_df):
    """
    Score how well predictions match annotations for a read.
    Returns dict with metrics.
    """
    read_preds = predictions_df[predictions_df['read_id'] == read_id]
    read_anns = annotations_df[annotations_df['read_id'] == read_id]

    if len(read_anns) == 0:
        return None  # Skip reads without annotations

    # Count predicted forks
    left_preds = read_preds[read_preds['predicted_class'] == 1]
    right_preds = read_preds[read_preds['predicted_class'] == 2]

    # Count true forks
    left_anns = read_anns[read_anns['type'] == 'left']
    right_anns = read_anns[read_anns['type'] == 'right']

    # Compute matches (any overlap)
    left_matches = 0
    for _, pred in left_preds.iterrows():
        for _, ann in left_anns.iterrows():
            if compute_overlap(pred['start'], pred['end'], ann['start'], ann['end']) > 0:
                left_matches += 1
                break

    right_matches = 0
    for _, pred in right_preds.iterrows():
        for _, ann in right_anns.iterrows():
            if compute_overlap(pred['start'], pred['end'], ann['start'], ann['end']) > 0:
                right_matches += 1
                break

    total_preds = len(left_preds) + len(right_preds)
    total_anns = len(left_anns) + len(right_anns)
    total_matches = left_matches + right_matches

    # Compute metrics
    precision = total_matches / total_preds if total_preds > 0 else 0
    recall = total_matches / total_anns if total_anns > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Confidence (mean probability of predicted forks)
    if total_preds > 0:
        left_conf = left_preds['class_1_prob'].mean() if len(left_preds) > 0 else 0
        right_conf = right_preds['class_2_prob'].mean() if len(right_preds) > 0 else 0
        confidence = (left_conf * len(left_preds) + right_conf * len(right_preds)) / total_preds
    else:
        confidence = 0

    return {
        'read_id': read_id,
        'n_left_preds': len(left_preds),
        'n_right_preds': len(right_preds),
        'n_left_anns': len(left_anns),
        'n_right_anns': len(right_anns),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confidence': confidence
    }


def main():
    print("=" * 70)
    print("PLOTTING BEST AND WORST FORK PREDICTIONS")
    print("=" * 70)

    # Load data
    print("\n1. Loading XY data...")
    xy_data = load_all_xy_data(
        base_dir="/mnt/ssd-4tb/crisanto_project/data_2025Oct/data_reads_minLen30000_nascent40/NM30_Col0",
        run_dirs=["NM30_plot_data_1strun_xy", "NM30_plot_data_2ndrun_xy"]
    )
    print(f"   Loaded {xy_data['read_id'].nunique():,} reads")

    # Load AI-predicted forks (from BED files)
    print("\n2. Loading AI-predicted forks...")
    left_forks = pd.read_csv("results/ori_calling_ai_pipeline/predicted_left_forks.bed",
                              sep='\t', header=None, usecols=[0,1,2,3])
    left_forks.columns = ['chr', 'start', 'end', 'read_id']
    left_forks['predicted_class'] = 1  # left fork

    right_forks = pd.read_csv("results/ori_calling_ai_pipeline/predicted_right_forks.bed",
                               sep='\t', header=None, usecols=[0,1,2,3])
    right_forks.columns = ['chr', 'start', 'end', 'read_id']
    right_forks['predicted_class'] = 2  # right fork

    predictions_df = pd.concat([left_forks, right_forks], ignore_index=True)

    # Add dummy probabilities for scoring (we'll recompute these during plotting)
    predictions_df['class_1_prob'] = 0.9  # Placeholder
    predictions_df['class_2_prob'] = 0.9  # Placeholder

    print(f"   Loaded {len(predictions_df):,} fork predictions ({len(left_forks):,} left, {len(right_forks):,} right)")

    # Load DNAscent annotations
    print("\n3. Loading DNAscent fork annotations...")
    annotations_df = load_dnascent_forks(
        "/mnt/ssd-4tb/crisanto_project/data_2025Oct/DNAscent_Col0_NM30_left_forks.bed",
        "/mnt/ssd-4tb/crisanto_project/data_2025Oct/DNAscent_Col0_NM30_right_forks.bed"
    )
    print(f"   Loaded {len(annotations_df):,} annotations from {annotations_df['read_id'].nunique():,} reads")

    # Score all reads with annotations
    print("\n4. Scoring predictions...")
    annotated_reads = annotations_df['read_id'].unique()
    scores = []

    for i, read_id in enumerate(annotated_reads):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(annotated_reads)} reads...")

        score = score_read_predictions(read_id, xy_data, predictions_df, annotations_df)
        if score:
            scores.append(score)

    scores_df = pd.DataFrame(scores)
    print(f"\n   Scored {len(scores_df):,} reads")

    # Find best reads (high F1, high confidence)
    print("\n5. Finding best predictions...")
    best_reads = scores_df.nlargest(10, 'f1')
    print("\n   Top 10 reads:")
    print(best_reads[['read_id', 'f1', 'precision', 'recall', 'confidence']].to_string(index=False))

    # Find worst reads (low F1, high confidence - confident mistakes)
    print("\n6. Finding worst predictions...")
    worst_reads = scores_df[(scores_df['confidence'] > 0.5) & (scores_df['f1'] < 0.5)].nsmallest(10, 'f1')
    print("\n   Bottom 10 reads (high confidence, low F1):")
    print(worst_reads[['read_id', 'f1', 'precision', 'recall', 'confidence']].to_string(index=False))

    # Load model for re-prediction
    print("\n7. Loading model...")
    custom_objects = {'SelfAttention': SelfAttention}
    model = tf.keras.models.load_model(
        "models/case_study_jan2026/combined_fork_detector.keras",
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )
    print("   ✅ Model loaded")

    # Plot best 5 reads
    print("\n8. Plotting best predictions...")
    output_dir = Path("results/prediction_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (idx, row) in enumerate(best_reads.head(5).iterrows()):
        read_id = row['read_id']
        print(f"\n   Plotting best read {i+1}/5: {read_id}")
        print(f"      F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")

        # Get read data
        read_data = xy_data[xy_data['read_id'] == read_id].copy()
        read_data = read_data.sort_values('start').reset_index(drop=True)
        read_data['center'] = (read_data['start'] + read_data['end']) / 2

        # Get predictions
        read_df, y_pred = predict_on_read(model, read_data, 411, use_enhanced_encoding=True)

        # Get annotations
        anns = annotations_df[annotations_df['read_id'] == read_id]

        # Plot
        plot_read_prediction(
            read_df, y_pred, annotations=anns,
            save_path=output_dir / f"best_{i+1}_{read_id[:8]}.png",
            title=f"Best Prediction #{i+1}: {read_id[:16]}... (F1={row['f1']:.3f})"
        )

    # Plot worst 5 reads
    print("\n9. Plotting worst predictions...")
    for i, (idx, row) in enumerate(worst_reads.head(5).iterrows()):
        read_id = row['read_id']
        print(f"\n   Plotting worst read {i+1}/5: {read_id}")
        print(f"      F1={row['f1']:.3f}, Precision={row['precision']:.3f}, Recall={row['recall']:.3f}")

        # Get read data
        read_data = xy_data[xy_data['read_id'] == read_id].copy()
        read_data = read_data.sort_values('start').reset_index(drop=True)
        read_data['center'] = (read_data['start'] + read_data['end']) / 2

        # Get predictions
        read_df, y_pred = predict_on_read(model, read_data, 411, use_enhanced_encoding=True)

        # Get annotations
        anns = annotations_df[annotations_df['read_id'] == read_id]

        # Plot
        plot_read_prediction(
            read_df, y_pred, annotations=anns,
            save_path=output_dir / f"worst_{i+1}_{read_id[:8]}.png",
            title=f"Worst Prediction #{i+1}: {read_id[:16]}... (F1={row['f1']:.3f})"
        )

    print("\n" + "=" * 70)
    print(f"✅ DONE! Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
