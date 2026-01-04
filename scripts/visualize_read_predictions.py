#!/usr/bin/env python
"""
Visualize read predictions matching notebook style.

Uses twin axes to show:
- Main axis: XY signal with shaded true fork regions
- Twin axis: Prediction probabilities for left/right forks
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_xy_signal(read_id, run, xy_base_dir):
    """Load XY signal for a read."""
    run_map = {
        'NM30_1strun': 'NM30_Col0/NM30_plot_data_1strun_xy',
        'NM30_2ndrun': 'NM30_Col0/NM30_plot_data_2ndrun_xy',
        'NM31_1strun': 'NM31_orc1b2/NM31_plot_data_1strun_xy',
        'NM31_2ndrun': 'NM31_orc1b2/NM31_plot_data_2ndrun_xy'
    }

    if run not in run_map:
        return None, None

    xy_file = Path(xy_base_dir) / run_map[run] / f"plot_data_{read_id}.txt"

    if not xy_file.exists():
        return None, None

    try:
        # Load XY data
        data = pd.read_csv(xy_file, sep='\t', header=None, names=['chr', 'start', 'end', 'signal'])
        chrom = data['chr'].iloc[0]
        return data, chrom
    except Exception as e:
        print(f"Error loading {xy_file}: {e}")
        return None, None


def plot_fork_prediction_single_read(read_id, predictions_df, xy_base_dir, ax):
    """
    Plot predictions on a single read matching notebook style.

    Uses twin axes:
    - Main axis: Signal + shaded true fork regions
    - Twin axis: Prediction probabilities
    """
    # Get predictions for this read
    read_preds = predictions_df[predictions_df['read_id'] == read_id].copy()

    if len(read_preds) == 0:
        print(f"No predictions found for read {read_id}")
        return False

    # Get run and load XY data
    run = read_preds['run'].iloc[0]
    xy_data, chrom = load_xy_signal(read_id, run, xy_base_dir)

    if xy_data is None:
        print(f"Could not load XY signal for {read_id} from run {run}")
        return False

    # Sort read_preds by start position to match signal
    read_preds = read_preds.sort_values('start').reset_index(drop=True)

    # Plot signal on main axis
    ax_signal = ax
    ax_signal.plot(xy_data['start'], xy_data['signal'], 'k-',
                  linewidth=1, alpha=0.5, label='Signal')

    # Overlay true fork regions (using BED regions, not segment-wise)
    # Find contiguous regions for each class
    def find_contiguous_regions(df, class_id):
        """Find contiguous regions of a class from segment predictions."""
        class_mask = df['y_true'] == class_id
        regions = []

        if not class_mask.any():
            return regions

        start_idx = None
        for idx in range(len(df)):
            if class_mask.iloc[idx]:
                if start_idx is None:
                    start_idx = idx
            else:
                if start_idx is not None:
                    regions.append((df.iloc[start_idx]['start'], df.iloc[idx-1]['end']))
                    start_idx = None

        # Handle case where region extends to end
        if start_idx is not None:
            regions.append((df.iloc[start_idx]['start'], df.iloc[-1]['end']))

        return regions

    # Overlay true left forks (blue)
    left_regions = find_contiguous_regions(read_preds, 1)
    for i, (start, end) in enumerate(left_regions):
        ax_signal.axvspan(start, end, alpha=0.3,
                        color='blue', label='True Left Fork' if i == 0 else '')

    # Overlay true right forks (orange)
    right_regions = find_contiguous_regions(read_preds, 2)
    for i, (start, end) in enumerate(right_regions):
        ax_signal.axvspan(start, end, alpha=0.3,
                        color='orange', label='True Right Fork' if i == 0 else '')

    # Create twin axis for predictions
    ax_pred = ax_signal.twinx()

    # Extract probabilities (columns: class_0_prob, class_1_prob, class_2_prob)
    prob_left = read_preds['class_1_prob'].values
    prob_right = read_preds['class_2_prob'].values
    positions = read_preds['start'].values

    # Plot probabilities for left and right
    ax_pred.plot(positions, prob_left, 'b-',
                linewidth=2, label='P(Left Fork)', alpha=0.7)
    ax_pred.plot(positions, prob_right, 'orange',
                linewidth=2, label='P(Right Fork)', alpha=0.7)
    ax_pred.axhline(y=0.5, color='red', linestyle='--',
                   linewidth=1, alpha=0.5, label='Threshold')

    # Highlight predicted regions using fill_between
    is_left_pred = read_preds['predicted_class'] == 1
    is_right_pred = read_preds['predicted_class'] == 2

    ax_pred.fill_between(positions, 0, 1,
                        where=is_left_pred, alpha=0.2,
                        color='blue', label='Predicted Left')
    ax_pred.fill_between(positions, 0, 1,
                        where=is_right_pred, alpha=0.2,
                        color='orange', label='Predicted Right')

    # Labels
    ax_signal.set_xlabel('Genomic Position', fontsize=11)
    ax_signal.set_ylabel('Signal Intensity', fontsize=11, color='black')
    ax_pred.set_ylabel('Fork Probability', fontsize=11, color='black')
    ax_pred.set_ylim([0, 1])

    # Title with stats
    n_left_true = len(left_regions)
    n_right_true = len(right_regions)
    n_left_pred = is_left_pred.sum()
    n_right_pred = is_right_pred.sum()

    ax_signal.set_title(
        f'Read: {read_id} | True: L={n_left_true} R={n_right_true} | '
        f'Pred segments: L={n_left_pred} R={n_right_pred}',
        fontweight='bold', fontsize=11
    )

    # Legends
    lines1, labels1 = ax_signal.get_legend_handles_labels()
    lines2, labels2 = ax_pred.get_legend_handles_labels()
    ax_signal.legend(lines1 + lines2, labels1 + labels2,
                    loc='upper left', fontsize=8, ncol=2)

    ax_signal.grid(True, alpha=0.3)

    return True


def find_interesting_reads(predictions_df):
    """Find example reads showing different patterns."""
    read_stats = []
    for read_id in predictions_df['read_id'].unique():
        read_df = predictions_df[predictions_df['read_id'] == read_id]

        n_total = len(read_df)
        n_errors = (read_df['y_true'] != read_df['predicted_class']).sum()
        accuracy = 1 - (n_errors / n_total)

        has_left = (read_df['y_true'] == 1).any()
        has_right = (read_df['y_true'] == 2).any()

        read_stats.append({
            'read_id': read_id,
            'n_segments': n_total,
            'n_errors': n_errors,
            'accuracy': accuracy,
            'has_left': has_left,
            'has_right': has_right
        })

    stats_df = pd.DataFrame(read_stats)

    examples = []

    # Perfect prediction with a fork
    perfect_with_fork = stats_df[(stats_df['accuracy'] == 1.0) & (stats_df['has_left'] | stats_df['has_right'])]
    if len(perfect_with_fork) > 0:
        both = perfect_with_fork[perfect_with_fork['has_left'] & perfect_with_fork['has_right']]
        if len(both) > 0:
            examples.append(both.iloc[0]['read_id'])
        else:
            examples.append(perfect_with_fork.iloc[0]['read_id'])

    # Has both fork types
    both_forks = stats_df[stats_df['has_left'] & stats_df['has_right']]
    if len(both_forks) > 0:
        good_both = both_forks[(both_forks['accuracy'] > 0.9) & (both_forks['accuracy'] < 1.0)]
        if len(good_both) > 0:
            examples.append(good_both.iloc[0]['read_id'])
        else:
            examples.append(both_forks.iloc[0]['read_id'])

    # One with some errors but still good
    some_errors = stats_df[(stats_df['n_errors'] >= 5) & (stats_df['n_errors'] <= 10) & (stats_df['accuracy'] > 0.85)]
    if len(some_errors) > 0:
        examples.append(some_errors.iloc[0]['read_id'])

    # Fallback: just get any reads with forks
    while len(examples) < 3 and len(stats_df) > 0:
        with_forks = stats_df[stats_df['has_left'] | stats_df['has_right']]
        if len(with_forks) > 0:
            # Get one not already in examples
            for _, row in with_forks.iterrows():
                if row['read_id'] not in examples:
                    examples.append(row['read_id'])
                    break
        break

    return examples


def main():
    parser = argparse.ArgumentParser(description='Generate read prediction visualizations (notebook style)')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions TSV file')
    parser.add_argument('--xy-base-dir', type=str, required=True,
                       help='Base directory containing XY data subdirectories')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--read-ids', type=str, nargs='*',
                       help='Specific read IDs to visualize (optional)')
    parser.add_argument('--n-examples', type=int, default=5,
                       help='Number of example reads to generate')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING FORK PREDICTION EXAMPLES (NOTEBOOK STYLE)")
    print("="*70)

    # Load predictions
    print(f"\n📂 Loading predictions from: {args.predictions}")
    predictions_df = pd.read_csv(args.predictions, sep='\t')
    print(f"✅ Loaded {len(predictions_df):,} predictions for {predictions_df['read_id'].nunique()} reads")

    # Determine which reads to visualize
    if args.read_ids:
        read_ids = args.read_ids
        print(f"\n📊 Visualizing {len(read_ids)} specified reads...")
    else:
        print("\n📊 Finding interesting example reads...")
        read_ids = find_interesting_reads(predictions_df)
        read_ids = read_ids[:args.n_examples]
        print(f"✅ Selected {len(read_ids)} example reads")

    # Create figure with n_examples subplots
    n_examples = len(read_ids)
    fig, axes = plt.subplots(n_examples, 1, figsize=(16, 4*n_examples))

    if n_examples == 1:
        axes = [axes]

    # Generate visualizations
    print(f"\n🎨 Generating visualizations...\n")
    success_count = 0
    for idx, read_id in enumerate(read_ids):
        success = plot_fork_prediction_single_read(
            read_id, predictions_df, args.xy_base_dir, axes[idx]
        )
        if success:
            success_count += 1

    plt.suptitle('Fork Predictions on Individual Reads',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save combined figure
    output_path = output_dir / 'fork_prediction_examples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()

    print("\n" + "="*70)
    print(f"✅ COMPLETE! Generated {success_count}/{n_examples} visualizations")
    print("="*70)
    print(f"\n📁 Saved to: {output_path}")


if __name__ == '__main__':
    main()
