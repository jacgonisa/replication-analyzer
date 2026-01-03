#!/usr/bin/env python
"""
Executable script for annotating new data with trained models.

This is the key script for analyzing new fork/ORI data from collaborators!

Usage:
    # Annotate ORIs
    python scripts/annotate_new_data.py \\
        --model models/ori_expert_model.keras \\
        --type ori \\
        --data-dir data/raw/new_experiment \\
        --output results/new_experiment_oris

    # Annotate forks
    python scripts/annotate_new_data.py \\
        --model models/fork_detector.keras \\
        --type fork \\
        --data-dir data/raw/new_fork_data \\
        --output results/new_fork_annotations

"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.data.loaders import load_all_xy_data
from replication_analyzer.training.train_ori import load_trained_ori_model
from replication_analyzer.training.train_fork import load_trained_fork_model
from replication_analyzer.evaluation.predictors import predict_on_all_reads, call_peaks_from_predictions, export_peaks_to_bed
from replication_analyzer.visualization.read_plots import plot_multiple_reads


def annotate_oris(model, xy_data, output_dir, threshold=0.5, min_peak_length=100,
                  max_length=None, plot_examples=True):
    """
    Annotate ORIs in new data.

    Parameters
    ----------
    model : tf.keras.Model
        Trained ORI model
    xy_data : pd.DataFrame
        XY signal data
    output_dir : Path
        Output directory
    threshold : float
        Probability threshold for calling ORIs
    min_peak_length : int
        Minimum ORI length in bp
    max_length : int, optional
        Model max length (auto-detected if None)
    plot_examples : bool
        Generate example plots

    Returns
    -------
    tuple
        (predictions_df, called_oris_df)
    """
    print("\n" + "="*70)
    print("ANNOTATING ORIs")
    print("="*70)

    # Get max_length
    if max_length is None:
        max_length = model.input_shape[1]

    print(f"\nModel max_length: {max_length}")
    print(f"Threshold: {threshold}")
    print(f"Min peak length: {min_peak_length} bp")

    # Predict on all reads
    print(f"\nPredicting on {xy_data['read_id'].nunique()} reads...")
    predictions_df = predict_on_all_reads(model, xy_data, max_length, verbose=True)

    # Call peaks
    print("\n" + "="*70)
    print("CALLING ORI PEAKS")
    print("="*70)

    called_oris = call_peaks_from_predictions(
        predictions_df,
        threshold=threshold,
        min_length=min_peak_length
    )

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # 1. Save all segment predictions
    predictions_file = output_dir / 'ori_segment_predictions.tsv'
    predictions_df.to_csv(predictions_file, sep='\t', index=False)
    print(f"✅ Segment predictions saved: {predictions_file}")

    # 2. Save called ORIs
    if len(called_oris) > 0:
        oris_file = output_dir / 'called_oris.tsv'
        called_oris.to_csv(oris_file, sep='\t', index=False)
        print(f"✅ Called ORIs saved: {oris_file}")

        # 3. Export to BED format
        bed_file = output_dir / 'called_oris.bed'
        export_peaks_to_bed(called_oris, bed_file)

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"  Total reads processed: {predictions_df['read_id'].nunique():,}")
        print(f"  Total segments: {len(predictions_df):,}")
        print(f"  Called ORIs: {len(called_oris):,}")
        print(f"\n  ORI size distribution:")
        print(called_oris['length'].describe())

        # Plot examples if requested
        if plot_examples:
            print("\n" + "="*70)
            print("GENERATING EXAMPLE PLOTS")
            print("="*70)

            plots_dir = output_dir / 'example_plots'
            plots_dir.mkdir(parents=True, exist_ok=True)

            plot_multiple_reads(
                xy_data, predictions_df,
                annotations=called_oris,
                n_reads=min(6, predictions_df['read_id'].nunique()),
                save_dir=str(plots_dir)
            )

            print(f"✅ Example plots saved: {plots_dir}")

    else:
        print("⚠️ No ORIs called with current parameters!")

    return predictions_df, called_oris


def annotate_forks(model, xy_data, output_dir, threshold=0.5, min_peak_length=100,
                   max_length=None, plot_examples=True):
    """
    Annotate forks in new data.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Fork model
    xy_data : pd.DataFrame
        XY signal data
    output_dir : Path
        Output directory
    threshold : float
        Probability threshold for calling forks
    min_peak_length : int
        Minimum fork length in bp
    max_length : int, optional
        Model max length
    plot_examples : bool
        Generate example plots

    Returns
    -------
    tuple
        (predictions_df, left_forks_df, right_forks_df)
    """
    print("\n" + "="*70)
    print("ANNOTATING FORKS")
    print("="*70)

    # Get max_length
    if max_length is None:
        max_length = model.input_shape[1]

    print(f"\nModel max_length: {max_length}")
    print(f"Threshold: {threshold}")
    print(f"Min fork length: {min_peak_length} bp")

    # Predict on all reads
    print(f"\nPredicting on {xy_data['read_id'].nunique()} reads...")
    predictions_df = predict_on_all_reads(model, xy_data, max_length, verbose=True)

    # Call forks separately for left and right
    print("\n" + "="*70)
    print("CALLING FORK REGIONS")
    print("="*70)

    # Left forks
    predictions_left = predictions_df.copy()
    predictions_left['peak_prob'] = predictions_left['class_1_prob']
    left_forks = call_peaks_from_predictions(
        predictions_left,
        threshold=threshold,
        min_length=min_peak_length
    )
    left_forks['fork_type'] = 'left'

    # Right forks
    predictions_right = predictions_df.copy()
    predictions_right['peak_prob'] = predictions_right['class_2_prob']
    right_forks = call_peaks_from_predictions(
        predictions_right,
        threshold=threshold,
        min_length=min_peak_length
    )
    right_forks['fork_type'] = 'right'

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # 1. Save all segment predictions
    predictions_file = output_dir / 'fork_segment_predictions.tsv'
    predictions_df.to_csv(predictions_file, sep='\t', index=False)
    print(f"✅ Segment predictions saved: {predictions_file}")

    # 2. Save called forks
    if len(left_forks) > 0:
        left_file = output_dir / 'called_left_forks.tsv'
        left_forks.to_csv(left_file, sep='\t', index=False)
        print(f"✅ Left forks saved: {left_file}")

        # Export to BED
        left_bed = output_dir / 'called_left_forks.bed'
        export_peaks_to_bed(left_forks, left_bed)

    if len(right_forks) > 0:
        right_file = output_dir / 'called_right_forks.tsv'
        right_forks.to_csv(right_file, sep='\t', index=False)
        print(f"✅ Right forks saved: {right_file}")

        # Export to BED
        right_bed = output_dir / 'called_right_forks.bed'
        export_peaks_to_bed(right_forks, right_bed)

    # Combined forks
    if len(left_forks) > 0 or len(right_forks) > 0:
        all_forks = pd.concat([left_forks, right_forks], ignore_index=True)
        all_forks_file = output_dir / 'all_called_forks.tsv'
        all_forks.to_csv(all_forks_file, sep='\t', index=False)
        print(f"✅ All forks saved: {all_forks_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Total reads processed: {predictions_df['read_id'].nunique():,}")
    print(f"  Total segments: {len(predictions_df):,}")
    print(f"  Called left forks: {len(left_forks):,}")
    print(f"  Called right forks: {len(right_forks):,}")
    print(f"  Total forks: {len(left_forks) + len(right_forks):,}")

    # Plot examples if requested
    if plot_examples and (len(left_forks) > 0 or len(right_forks) > 0):
        print("\n" + "="*70)
        print("GENERATING EXAMPLE PLOTS")
        print("="*70)

        plots_dir = output_dir / 'example_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        all_forks_for_plot = pd.concat([left_forks, right_forks]) if len(left_forks) > 0 and len(right_forks) > 0 else left_forks if len(left_forks) > 0 else right_forks

        plot_multiple_reads(
            xy_data, predictions_df,
            annotations=all_forks_for_plot,
            n_reads=min(6, predictions_df['read_id'].nunique()),
            save_dir=str(plots_dir)
        )

        print(f"✅ Example plots saved: {plots_dir}")

    return predictions_df, left_forks, right_forks


def main():
    parser = argparse.ArgumentParser(description='Annotate new data with trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.keras)')
    parser.add_argument('--type', type=str, required=True, choices=['ori', 'fork'],
                       help='Model type: ori or fork')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing XY plot data')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for annotations')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for calling peaks (default: 0.5)')
    parser.add_argument('--min-length', type=int, default=100,
                       help='Minimum peak length in bp (default: 100)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating example plots')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"ANNOTATING NEW DATA")
    print("="*70)
    print(f"  Model: {args.model}")
    print(f"  Type: {args.type}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output: {output_dir}")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    xy_data = load_all_xy_data(base_dir=args.data_dir)

    # Load model and annotate
    if args.type == 'ori':
        model = load_trained_ori_model(args.model)
        predictions_df, called_peaks = annotate_oris(
            model, xy_data, output_dir,
            threshold=args.threshold,
            min_peak_length=args.min_length,
            plot_examples=not args.no_plots
        )
    else:
        model = load_trained_fork_model(args.model)
        predictions_df, left_forks, right_forks = annotate_forks(
            model, xy_data, output_dir,
            threshold=args.threshold,
            min_peak_length=args.min_length,
            plot_examples=not args.no_plots
        )

    print("\n" + "="*70)
    print("✅ ANNOTATION COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nOutput files:")
    print(f"  - Segment predictions: {output_dir}/*_segment_predictions.tsv")
    print(f"  - Called peaks: {output_dir}/called_*.tsv")
    print(f"  - BED files: {output_dir}/called_*.bed")
    if not args.no_plots:
        print(f"  - Example plots: {output_dir}/example_plots/")

    print("\n💡 Pro tip: Load the BED files into a genome browser to visualize!")


if __name__ == '__main__':
    main()
