#!/usr/bin/env python
"""
Visualize read predictions matching notebook style.

Shows XY signal with:
- Top panel: Original signal with ground truth annotations
- Bottom panel: Original signal with model predictions
Direct visual comparison between truth and predictions.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

sns.set_style("whitegrid")


def load_xy_signal(read_id, run, xy_base_dir):
    """Load XY signal for a read."""
    run_map = {
        'NM30_1strun': 'NM30_Col0/NM30_plot_data_1strun_xy',
        'NM30_2ndrun': 'NM30_Col0/NM30_plot_data_2ndrun_xy',
        'NM31_1strun': 'NM31_orc1b2/NM31_plot_data_1strun_xy',
        'NM31_2ndrun': 'NM31_orc1b2/NM31_plot_data_2ndrun_xy'
    }

    if run not in run_map:
        return None, None, None

    xy_file = Path(xy_base_dir) / run_map[run] / f"plot_data_{read_id}.txt"

    if not xy_file.exists():
        return None, None, None

    try:
        # Load XY data
        data = pd.read_csv(xy_file, sep='\t', header=None, names=['chr', 'start', 'end', 'signal'])
        chrom = data['chr'].iloc[0]
        return data, chrom
    except Exception as e:
        print(f"Error loading {xy_file}: {e}")
        return None, None


def plot_read_comparison(read_id, predictions_df, xy_base_dir, save_path):
    """
    Plot read with ground truth vs predictions side-by-side.

    Top panel: Ground truth annotations
    Bottom panel: Model predictions
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

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    # Class colors matching the notebook
    colors = {0: '#808080', 1: '#FF4444', 2: '#44AA44'}  # gray, red, green
    class_names = {0: 'Background', 1: 'Left Fork', 2: 'Right Fork'}

    # Calculate stats
    n_total = len(read_preds)
    n_correct = (read_preds['y_true'] == read_preds['predicted_class']).sum()
    accuracy = n_correct / n_total * 100

    # ============= PANEL 1: GROUND TRUTH =============
    # Plot XY signal
    ax1.plot(xy_data['start'], xy_data['signal'], 'k-', linewidth=1, alpha=0.5, zorder=1)

    # Overlay ground truth regions
    for idx, row in read_preds.iterrows():
        color = colors[row['y_true']]
        ax1.axvspan(row['start'], row['end'], alpha=0.3, color=color, zorder=2)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.3, label='Background'),
        Patch(facecolor=colors[1], alpha=0.3, label='Left Fork'),
        Patch(facecolor=colors[2], alpha=0.3, label='Right Fork')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    ax1.set_ylabel('Signal Intensity', fontsize=12, fontweight='bold')
    ax1.set_title(f'{chrom} - Read: {read_id}\nGround Truth Annotations',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # ============= PANEL 2: PREDICTIONS =============
    # Plot XY signal
    ax2.plot(xy_data['start'], xy_data['signal'], 'k-', linewidth=1, alpha=0.5, zorder=1)

    # Overlay predictions
    for idx, row in read_preds.iterrows():
        color = colors[row['predicted_class']]
        ax2.axvspan(row['start'], row['end'], alpha=0.3, color=color, zorder=2)

    # Mark errors with red outlines
    errors = read_preds[read_preds['y_true'] != read_preds['predicted_class']]
    for idx, row in errors.iterrows():
        ax2.axvspan(row['start'], row['end'],
                   alpha=0, edgecolor='red', linewidth=3, zorder=3)

    # Add legend
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    ax2.set_ylabel('Signal Intensity', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Genomic Position (bp)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Model Predictions (Accuracy: {accuracy:.1f}% - {n_correct}/{n_total} correct)',
                  fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Format x-axis
    ax2.ticklabel_format(style='plain', axis='x')

    # Add stats box
    stats_text = f'Read Statistics:\n'
    stats_text += f'Total segments: {n_total}\n'
    stats_text += f'Correct: {n_correct}\n'
    stats_text += f'Errors: {n_total - n_correct}\n'
    stats_text += f'Accuracy: {accuracy:.1f}%\n\n'

    # Count by class
    true_counts = read_preds['y_true'].value_counts().to_dict()
    pred_counts = read_preds['predicted_class'].value_counts().to_dict()

    stats_text += 'Ground Truth:\n'
    for cls in [0, 1, 2]:
        if cls in true_counts:
            stats_text += f'  {class_names[cls]}: {true_counts[cls]}\n'

    stats_text += '\nPredicted:\n'
    for cls in [0, 1, 2]:
        if cls in pred_counts:
            stats_text += f'  {class_names[cls]}: {pred_counts[cls]}\n'

    fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

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

    examples = {}

    # Perfect prediction with a fork
    perfect_with_fork = stats_df[(stats_df['accuracy'] == 1.0) & (stats_df['has_left'] | stats_df['has_right'])]
    if len(perfect_with_fork) > 0:
        # Prefer one with both forks
        both = perfect_with_fork[perfect_with_fork['has_left'] & perfect_with_fork['has_right']]
        if len(both) > 0:
            examples['perfect'] = both.iloc[0]['read_id']
        else:
            examples['perfect'] = perfect_with_fork.iloc[0]['read_id']

    # Has both fork types
    both_forks = stats_df[stats_df['has_left'] & stats_df['has_right']]
    if len(both_forks) > 0:
        # Get one with high but not perfect accuracy
        good_both = both_forks[(both_forks['accuracy'] > 0.9) & (both_forks['accuracy'] < 1.0)]
        if len(good_both) > 0:
            examples['both_forks'] = good_both.iloc[0]['read_id']
        else:
            examples['both_forks'] = both_forks.iloc[0]['read_id']

    # One with some errors but still good
    some_errors = stats_df[(stats_df['n_errors'] >= 5) & (stats_df['n_errors'] <= 10) & (stats_df['accuracy'] > 0.85)]
    if len(some_errors) > 0:
        examples['minor_errors'] = some_errors.iloc[0]['read_id']

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
    parser.add_argument('--n-examples', type=int, default=3,
                       help='Number of example reads to generate')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING READ PREDICTION VISUALIZATIONS (NOTEBOOK STYLE)")
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
        examples = find_interesting_reads(predictions_df)
        read_ids = list(examples.values())[:args.n_examples]
        print(f"✅ Selected {len(read_ids)} example reads")

    # Generate visualizations
    print(f"\n🎨 Generating visualizations...\n")
    success_count = 0
    for i, read_id in enumerate(read_ids, 1):
        save_path = output_dir / f'prediction_example_{i}.png'
        success = plot_read_comparison(
            read_id, predictions_df, args.xy_base_dir, save_path
        )
        if success:
            success_count += 1

    print("\n" + "="*70)
    print(f"✅ COMPLETE! Generated {success_count}/{len(read_ids)} visualizations")
    print("="*70)
    print(f"\n📁 Saved to: {output_dir}/")


if __name__ == '__main__':
    main()
