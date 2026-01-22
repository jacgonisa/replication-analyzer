#!/usr/bin/env python
"""
Step 1: Predict forks using trained AI model

This script loads XY signal data, runs the trained fork detection model,
and saves predicted left/right forks to BED files.

Usage:
    python scripts/predict_forks.py --config configs/fork_prediction.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.data.loaders import load_all_xy_data
from replication_analyzer.evaluation.predictors import predict_on_all_reads
from replication_analyzer.models.base import SelfAttention
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(
        description='Predict forks using trained AI model'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory from config')

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}", flush=True)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"FORK PREDICTION: {config['experiment_name']}")
    print("=" * 70)

    # Load XY data
    print("\nLoading XY signal data...", flush=True)
    xy_data = load_all_xy_data(
        base_dir=config['data']['base_dir'],
        run_dirs=config['data']['run_dirs']
    )
    print(f"✅ Loaded {len(xy_data):,} segments from {xy_data['read_id'].nunique():,} reads")

    # Load model
    print(f"\nLoading fork detection model: {config['model']['fork_model_path']}", flush=True)
    custom_objects = {'SelfAttention': SelfAttention}
    model = tf.keras.models.load_model(
        config['model']['fork_model_path'],
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )
    print("✅ Model loaded successfully", flush=True)

    # Predict on all reads
    print(f"\nPredicting on {xy_data['read_id'].nunique():,} reads...", flush=True)
    predictions_df = predict_on_all_reads(
        model, xy_data,
        config['model']['max_length'],
        use_enhanced_encoding=config['model'].get('use_enhanced_encoding', True),
        verbose=True
    )

    if predictions_df is None:
        print("❌ No predictions generated!")
        return 1

    print(f"✅ Predictions complete: {len(predictions_df):,} segments")

    # Extract forks with threshold
    threshold = config['prediction'].get('fork_threshold', 0.5)
    print(f"\nExtracting fork regions (threshold={threshold})...", flush=True)

    # Left forks (class 1)
    left_mask = predictions_df['predicted_class'] == 1
    left_forks = predictions_df[left_mask & (predictions_df['class_1_prob'] > threshold)].copy()
    left_forks['gradient'] = left_forks['class_1_prob']

    # Right forks (class 2)
    right_mask = predictions_df['predicted_class'] == 2
    right_forks = predictions_df[right_mask & (predictions_df['class_2_prob'] > threshold)].copy()
    right_forks['gradient'] = right_forks['class_2_prob']

    print(f"  → {len(left_forks):,} left fork segments")
    print(f"  → {len(right_forks):,} right fork segments")

    # Save to BED files
    left_bed = output_dir / 'predicted_left_forks.bed'
    right_bed = output_dir / 'predicted_right_forks.bed'

    # BED format: chr, start, end, read_id, gradient
    left_forks[['chr', 'start', 'end', 'read_id', 'gradient']].to_csv(
        left_bed, sep='\t', header=False, index=False
    )
    right_forks[['chr', 'start', 'end', 'read_id', 'gradient']].to_csv(
        right_bed, sep='\t', header=False, index=False
    )

    print(f"\n✅ Fork predictions saved:")
    print(f"  Left forks:  {left_bed}")
    print(f"  Right forks: {right_bed}")

    # Save full predictions (optional)
    predictions_csv = output_dir / 'all_predictions.csv'
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"  All predictions: {predictions_csv}")

    print("\n" + "=" * 70)
    print("✅ FORK PREDICTION COMPLETE!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
