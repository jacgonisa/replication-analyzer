#!/usr/bin/env python
"""
End-to-end pipeline: Predict forks with AI → Call origins from forks → Benchmark against curated data

This script provides a complete workflow for:
1. Using trained fork detection model to predict left/right forks
2. Calling origins from the predicted forks
3. Benchmarking predicted origins against curated dataset

Usage:
    python scripts/predict_forks_and_call_oris.py --config configs/ori_calling_pipeline.yaml

"""

import argparse
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.data.loaders import load_all_xy_data
from replication_analyzer.evaluation.predictors import predict_on_all_reads
from replication_analyzer.evaluation.ori_caller import (
    parse_fork_bed, infer_events, write_bed6
)
from replication_analyzer.evaluation.benchmark import (
    benchmark_ori_predictions, plot_benchmark_results, save_benchmark_report
)
from replication_analyzer.evaluation.bed_utils import write_bed_file

# Import custom layers for model loading
from replication_analyzer.models.base import SelfAttention

import tensorflow as tf


def predict_forks(model_path, xy_data, max_length, use_enhanced_encoding=True,
                 threshold=0.5, output_dir=None):
    """
    Predict left and right forks using trained model.

    Parameters
    ----------
    model_path : str
        Path to trained fork detection model
    xy_data : pd.DataFrame
        XY signal data
    max_length : int
        Maximum sequence length
    use_enhanced_encoding : bool
        Use 9-channel encoding
    threshold : float
        Probability threshold for fork calling
    output_dir : str, optional
        Directory to save fork BED files

    Returns
    -------
    tuple
        (left_forks_df, right_forks_df) DataFrames with fork predictions
    """
    print("\n" + "=" * 70)
    print("STEP 1: PREDICTING FORKS WITH AI MODEL")
    print("=" * 70)

    # Load model with custom objects and safe_mode=False for Lambda layers
    print(f"\nLoading fork detection model: {model_path}", flush=True)
    custom_objects = {'SelfAttention': SelfAttention}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
    print("✅ Model loaded successfully", flush=True)

    # Predict on all reads
    print(f"\nPredicting on {xy_data['read_id'].nunique():,} reads...")
    predictions_df = predict_on_all_reads(
        model, xy_data, max_length,
        use_enhanced_encoding=use_enhanced_encoding,
        verbose=True
    )

    if predictions_df is None:
        print("❌ No predictions generated!")
        return None, None

    # Extract forks (class 1 = left, class 2 = right)
    print(f"\nExtracting fork regions (threshold={threshold})...")

    # Left forks (class 1)
    left_mask = predictions_df['predicted_class'] == 1
    left_forks = predictions_df[left_mask & (predictions_df['class_1_prob'] > threshold)].copy()
    left_forks['gradient'] = left_forks['class_1_prob']  # Use probability as gradient

    # Right forks (class 2)
    right_mask = predictions_df['predicted_class'] == 2
    right_forks = predictions_df[right_mask & (predictions_df['class_2_prob'] > threshold)].copy()
    right_forks['gradient'] = right_forks['class_2_prob']  # Use probability as gradient

    print(f"  → {len(left_forks):,} left fork segments")
    print(f"  → {len(right_forks):,} right fork segments")

    # Save to BED files if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        left_bed = output_dir / 'predicted_left_forks.bed'
        right_bed = output_dir / 'predicted_right_forks.bed'

        # Prepare BED format (chr, start, end, read_id, gradient)
        left_forks[['chr', 'start', 'end', 'read_id', 'gradient']].to_csv(
            left_bed, sep='\t', header=False, index=False
        )
        right_forks[['chr', 'start', 'end', 'read_id', 'gradient']].to_csv(
            right_bed, sep='\t', header=False, index=False
        )

        print(f"\n✅ Fork predictions saved:")
        print(f"  Left forks:  {left_bed}")
        print(f"  Right forks: {right_bed}")

    return left_forks, right_forks


def call_origins_from_forks(left_forks_df, right_forks_df, min_len=0, output_dir=None):
    """
    Call origins from predicted left and right forks.

    Parameters
    ----------
    left_forks_df : pd.DataFrame
        Left fork predictions
    right_forks_df : pd.DataFrame
        Right fork predictions
    min_len : int
        Minimum origin length in bp
    output_dir : str, optional
        Directory to save origin BED files

    Returns
    -------
    pd.DataFrame
        Called origins
    """
    print("\n" + "=" * 70)
    print("STEP 2: CALLING ORIGINS FROM FORKS")
    print("=" * 70)

    # Convert DataFrames to ForkSeg objects
    print("\nProcessing fork data...")
    from replication_analyzer.evaluation.ori_caller import ForkSeg

    left_segs = [
        ForkSeg(
            chrom=row['chr'],
            start=int(row['start']),
            end=int(row['end']),
            read_id=row['read_id'],
            grad=float(row['gradient']),
            kind='L'
        )
        for _, row in left_forks_df.iterrows()
    ]

    right_segs = [
        ForkSeg(
            chrom=row['chr'],
            start=int(row['start']),
            end=int(row['end']),
            read_id=row['read_id'],
            grad=float(row['gradient']),
            kind='R'
        )
        for _, row in right_forks_df.iterrows()
    ]

    # Call origins
    print(f"Inferring origins (min_len={min_len}bp)...")
    from replication_analyzer.evaluation.ori_caller import infer_events

    origins, terminations, stats = infer_events(left_segs, right_segs, min_len=min_len)

    print(f"\n✅ Origins called: {len(origins):,}")
    print(f"✅ Terminations called: {len(terminations):,}")
    print(f"\nStatistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:,}")

    # Convert to DataFrame
    origins_df = pd.DataFrame(origins, columns=['chr', 'start', 'end', 'read_id', 'grad_left', 'grad_right'])

    # Save to BED files if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        origins_bed = output_dir / 'predicted_origins.bed'
        terminations_bed = output_dir / 'predicted_terminations.bed'

        # Save using ori_caller's write_bed6
        from replication_analyzer.evaluation.ori_caller import write_bed6
        write_bed6(str(origins_bed), origins)
        write_bed6(str(terminations_bed), terminations)

        print(f"\n✅ Origins and terminations saved:")
        print(f"  Origins:       {origins_bed}")
        print(f"  Terminations:  {terminations_bed}")

    return origins_df


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end pipeline: Predict forks → Call origins → Benchmark'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='Skip fork prediction, use existing fork BED files')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip benchmarking step')

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"ORIGIN CALLING PIPELINE: {config['experiment_name']}")
    print("=" * 70)

    # STEP 1: Predict forks (or load existing)
    if args.skip_prediction:
        print("\n⏭️  Skipping fork prediction (using existing BED files)")
        left_bed = config['forks']['left_forks_bed']
        right_bed = config['forks']['right_forks_bed']

        print(f"Loading left forks: {left_bed}")
        left_forks_df = pd.read_csv(left_bed, sep='\t', header=None)
        # Take only first 4 columns: chr, start, end, read_id
        left_forks_df = left_forks_df.iloc[:, :4]
        left_forks_df.columns = ['chr', 'start', 'end', 'read_id']
        left_forks_df['gradient'] = -1.0  # Default gradient for left forks

        print(f"Loading right forks: {right_bed}")
        right_forks_df = pd.read_csv(right_bed, sep='\t', header=None)
        # Take only first 4 columns: chr, start, end, read_id
        right_forks_df = right_forks_df.iloc[:, :4]
        right_forks_df.columns = ['chr', 'start', 'end', 'read_id']
        right_forks_df['gradient'] = 1.0  # Default gradient for right forks

        print(f"✅ Loaded {len(left_forks_df):,} left forks")
        print(f"✅ Loaded {len(right_forks_df):,} right forks")

    else:
        # Load XY data
        print("\nLoading XY signal data...")
        xy_data = load_all_xy_data(
            base_dir=config['data']['base_dir'],
            run_dirs=config['data']['run_dirs']
        )
        print(f"✅ Loaded {len(xy_data):,} segments from {xy_data['read_id'].nunique():,} reads")

        # Predict forks
        left_forks_df, right_forks_df = predict_forks(
            model_path=config['model']['fork_model_path'],
            xy_data=xy_data,
            max_length=config['model']['max_length'],
            use_enhanced_encoding=config['model'].get('use_enhanced_encoding', True),
            threshold=config['prediction'].get('fork_threshold', 0.5),
            output_dir=output_dir
        )

        if left_forks_df is None or right_forks_df is None:
            print("❌ Fork prediction failed!")
            return 1

    # STEP 2: Call origins
    origins_df = call_origins_from_forks(
        left_forks_df, right_forks_df,
        min_len=config['ori_calling'].get('min_length', 0),
        output_dir=output_dir
    )

    if len(origins_df) == 0:
        print("❌ No origins called!")
        return 1

    # STEP 3: Benchmark (optional)
    if not args.skip_benchmark and 'curated_ori_bed' in config['data']:
        print("\n" + "=" * 70)
        print("STEP 3: BENCHMARKING AGAINST CURATED DATASET")
        print("=" * 70)

        predicted_bed = output_dir / 'predicted_origins.bed'
        curated_bed = config['data']['curated_ori_bed']

        results = benchmark_ori_predictions(
            predicted_bed=str(predicted_bed),
            curated_bed=curated_bed,
            min_overlap=config['benchmark'].get('min_overlap', 1),
            jaccard_threshold=config['benchmark'].get('jaccard_threshold', 0.0)
        )

        # Generate plots
        print("\nGenerating benchmark plots...")
        plot_benchmark_results(results, output_dir / 'benchmark_plots')

        # Save detailed report
        print("Saving benchmark report...")
        save_benchmark_report(results, output_dir / 'benchmark_report.txt')

        # Save overlap details
        if len(results['high_quality_overlaps']) > 0:
            overlaps_file = output_dir / 'origin_overlaps.tsv'
            results['high_quality_overlaps'].to_csv(overlaps_file, sep='\t', index=False)
            print(f"✅ Overlap details saved to: {overlaps_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nOutput files:")
    print(f"  - predicted_left_forks.bed")
    print(f"  - predicted_right_forks.bed")
    print(f"  - predicted_origins.bed")
    print(f"  - predicted_terminations.bed")
    if not args.skip_benchmark:
        print(f"  - benchmark_report.txt")
        print(f"  - benchmark_plots/")
        print(f"  - origin_overlaps.tsv")

    print("\n" + "=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
