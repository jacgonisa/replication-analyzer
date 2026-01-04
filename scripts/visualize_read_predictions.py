#!/usr/bin/env python
"""
Generate example read visualizations with predictions.

Shows:
- Original XY signal
- Ground truth annotations
- Model predictions
- Confidence scores
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
    # Map run to directory name
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
        # Load as chr, start, end, signal
        data = pd.read_csv(xy_file, sep='\\t', header=None, names=['chr', 'start', 'end', 'signal'])
        # Use midpoint for position
        positions = (data['start'] + data['end']) / 2
        signals = data['signal'].values
        chrom = data['chr'].iloc[0]
        return positions.values, signals, chrom
    except Exception as e:
        print(f"Error loading {xy_file}: {e}")
        return None, None, None


def plot_read_with_predictions(read_id, predictions_df, xy_base_dir, save_path):
    """
    Create visualization for a single read showing:
    - XY signal
    - Ground truth annotations
    - Model predictions with confidence
    """
    # Get predictions for this read
    read_preds = predictions_df[predictions_df['read_id'] == read_id].copy()

    if len(read_preds) == 0:
        print(f"No predictions found for read {read_id}")
        return False

    # Get run for this read
    run = read_preds['run'].iloc[0]

    # Load XY signal
    positions, signals, chrom = load_xy_signal(read_id, run, xy_base_dir)
    if positions is None:
        print(f"Could not load XY signal for {read_id} from run {run}")
        return False
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Normalize positions to match predictions (0-based indexing)
    min_pos = positions.min()
    norm_positions = positions - min_pos
    
    # Class colors
    class_colors = {0: '#3498db', 1: '#e74c3c', 2: '#2ecc71'}
    class_names = {0: 'Background', 1: 'Left Fork', 2: 'Right Fork'}
    
    # Plot 1: XY Signal with ground truth
    ax1 = axes[0]
    ax1.plot(norm_positions, signals, 'k-', linewidth=0.5, alpha=0.7, label='Signal')
    
    # Overlay ground truth regions
    for class_id in [0, 1, 2]:
        class_mask = read_preds['y_true'].values == class_id
        if class_mask.any():
            for idx in read_preds[class_mask].index:
                row = read_preds.loc[idx]
                ax1.axvspan(row['start'], row['end'], 
                           alpha=0.3, color=class_colors[class_id],
                           label=f'True: {class_names[class_id]}' if idx == read_preds[class_mask].index[0] else '')
    
    ax1.set_ylabel('Signal Intensity', fontsize=11)
    ax1.set_title(f'Read: {read_id} ({chrom}) - Ground Truth Annotations',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted Classes
    ax2 = axes[1]
    ax2.plot(norm_positions, signals, 'k-', linewidth=0.5, alpha=0.3)
    
    for class_id in [0, 1, 2]:
        class_mask = read_preds['predicted_class'].values == class_id
        if class_mask.any():
            for idx in read_preds[class_mask].index:
                row = read_preds.loc[idx]
                ax2.axvspan(row['start'], row['end'], 
                           alpha=0.4, color=class_colors[class_id],
                           label=f'Predicted: {class_names[class_id]}' if idx == read_preds[class_mask].index[0] else '')
    
    ax2.set_ylabel('Signal Intensity', fontsize=11)
    ax2.set_title('Model Predictions', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction Confidence
    ax3 = axes[2]
    
    # Get max probability for each segment
    max_probs = read_preds[['class_0_prob', 'class_1_prob', 'class_2_prob']].max(axis=1).values
    segment_centers = (read_preds['start'].values + read_preds['end'].values) / 2
    
    # Color by predicted class
    colors = [class_colors[c] for c in read_preds['predicted_class'].values]
    ax3.scatter(segment_centers, max_probs, c=colors, alpha=0.6, s=20)
    ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold: 0.5')
    
    # Mark errors
    errors = read_preds[read_preds['y_true'] != read_preds['predicted_class']]
    if len(errors) > 0:
        error_centers = (errors['start'].values + errors['end'].values) / 2
        error_probs = errors[['class_0_prob', 'class_1_prob', 'class_2_prob']].max(axis=1).values
        ax3.scatter(error_centers, error_probs, facecolors='none', edgecolors='red', 
                   s=100, linewidth=2, label=f'Errors ({len(errors)})')
    
    ax3.set_xlabel('Position (bases)', fontsize=11)
    ax3.set_ylabel('Confidence', fontsize=11)
    ax3.set_title('Prediction Confidence', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Add summary statistics
    accuracy = (read_preds['y_true'] == read_preds['predicted_class']).mean()
    n_total = len(read_preds)
    n_errors = len(errors)
    
    fig.text(0.02, 0.98, 
             f'Read Stats:\nTotal segments: {n_total}\nErrors: {n_errors}\nAccuracy: {accuracy:.1%}',
             transform=fig.transFigure, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    return True


def find_interesting_reads(predictions_df):
    """Find example reads showing different patterns."""
    
    # Group by read
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
    
    # Find interesting examples
    examples = {}
    
    # Perfect prediction
    perfect = stats_df[stats_df['accuracy'] == 1.0]
    if len(perfect) > 0:
        examples['perfect'] = perfect.iloc[0]['read_id']
    
    # Has both forks
    both = stats_df[stats_df['has_left'] & stats_df['has_right']]
    if len(both) > 0:
        examples['both_forks'] = both.sort_values('accuracy', ascending=False).iloc[0]['read_id']
    
    # Some errors but good
    good_errors = stats_df[(stats_df['n_errors'] > 0) & (stats_df['accuracy'] > 0.9)]
    if len(good_errors) > 0:
        examples['minor_errors'] = good_errors.iloc[0]['read_id']
    
    return examples


def main():
    parser = argparse.ArgumentParser(description='Generate read prediction visualizations')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions TSV file')
    parser.add_argument('--xy-base-dir', type=str, required=True,
                       help='Base directory containing XY data subdirectories')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--read-ids', type=str, nargs='*',
                       help='Specific read IDs to visualize (optional)')

    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING READ PREDICTION VISUALIZATIONS")
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
        read_ids = list(examples.values())
        print(f"✅ Selected {len(examples)} example reads:")
        for name, read_id in examples.items():
            print(f"   - {name}: {read_id[:16]}...")
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...\n")
    success_count = 0
    for i, read_id in enumerate(read_ids, 1):
        save_path = output_dir / f'read_example_{i}_{read_id[:16]}.png'
        success = plot_read_with_predictions(
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
