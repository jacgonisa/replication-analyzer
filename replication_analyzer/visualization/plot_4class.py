"""
Visualization utilities for 4-class model predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")


def plot_4class_prediction(read_df, predictions, annotations=None,
                           save_path=None, title=None):
    """
    Plot a single read with 4-class predictions and optional annotations.

    Parameters
    ----------
    read_df : pd.DataFrame
        Read data with columns: start, end, signal, center
    predictions : np.ndarray
        4-class predictions shape (n_segments, 4)
        [background, left_fork, right_fork, origin]
    annotations : dict, optional
        Dictionary with 'left_forks', 'right_forks', 'origins' DataFrames
        Each with columns: start, end
    save_path : str, optional
        Path to save figure
    title : str, optional
        Plot title
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    positions = read_df['center'].values
    signal = read_df['signal'].values

    # Panel 1: Raw signal
    axes[0].plot(positions, signal, 'b-', linewidth=2, alpha=0.7, label='BrdU Signal')
    axes[0].fill_between(positions, 0, signal, alpha=0.3, color='blue')

    # Overlay annotations if provided
    if annotations:
        if 'left_forks' in annotations and len(annotations['left_forks']) > 0:
            for _, ann in annotations['left_forks'].iterrows():
                axes[0].axvspan(ann['start'], ann['end'], alpha=0.2, color='orange', label='_'*10)
        if 'right_forks' in annotations and len(annotations['right_forks']) > 0:
            for _, ann in annotations['right_forks'].iterrows():
                axes[0].axvspan(ann['start'], ann['end'], alpha=0.2, color='purple', label='_'*10)
        if 'origins' in annotations and len(annotations['origins']) > 0:
            for _, ann in annotations['origins'].iterrows():
                axes[0].axvspan(ann['start'], ann['end'], alpha=0.3, color='red')
                axes[0].axvline((ann['start'] + ann['end'])/2, color='darkred',
                              linestyle='--', linewidth=2, label='_'*10)

    axes[0].set_ylabel('BrdU Signal', fontsize=12, fontweight='bold')
    axes[0].legend(['Signal', 'True Origin'], loc='upper right')
    axes[0].grid(True, alpha=0.3)

    if title:
        axes[0].set_title(title, fontsize=14, fontweight='bold')

    # Panel 2: Fork Probabilities
    axes[1].plot(positions, predictions[:, 1], color='orange', linewidth=2.5,
                label='Left Fork', alpha=0.8)
    axes[1].plot(positions, predictions[:, 2], color='purple', linewidth=2.5,
                label='Right Fork', alpha=0.8)
    axes[1].fill_between(positions, 0, predictions[:, 1], alpha=0.3, color='orange')
    axes[1].fill_between(positions, 0, predictions[:, 2], alpha=0.3, color='purple')
    axes[1].axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_ylabel('Fork Probability', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Panel 3: Origin Probability
    axes[2].plot(positions, predictions[:, 3], color='red', linewidth=2.5,
                label='Origin', alpha=0.8)
    axes[2].fill_between(positions, 0, predictions[:, 3], alpha=0.3, color='red')
    axes[2].axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    axes[2].set_ylabel('Origin Probability', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])

    # Panel 4: Predicted Classes
    predicted_class = np.argmax(predictions, axis=1)

    # Create separate masks for each class
    left_mask = (predicted_class == 1).astype(float)
    right_mask = (predicted_class == 2).astype(float)
    origin_mask = (predicted_class == 3).astype(float)

    axes[3].fill_between(positions, 0, left_mask, alpha=0.6, color='orange',
                        label='Left Fork', step='mid')
    axes[3].fill_between(positions, 0, right_mask, alpha=0.6, color='purple',
                        label='Right Fork', step='mid')
    axes[3].fill_between(positions, 0, origin_mask, alpha=0.6, color='red',
                        label='Origin', step='mid')

    # Overlay true annotations
    if annotations:
        if 'left_forks' in annotations and len(annotations['left_forks']) > 0:
            for _, ann in annotations['left_forks'].iterrows():
                axes[3].axvspan(ann['start'], ann['end'], alpha=0.2,
                              facecolor='none', edgecolor='orange', linewidth=2)
        if 'right_forks' in annotations and len(annotations['right_forks']) > 0:
            for _, ann in annotations['right_forks'].iterrows():
                axes[3].axvspan(ann['start'], ann['end'], alpha=0.2,
                              facecolor='none', edgecolor='purple', linewidth=2)
        if 'origins' in annotations and len(annotations['origins']) > 0:
            for _, ann in annotations['origins'].iterrows():
                axes[3].axvspan(ann['start'], ann['end'], alpha=0.2,
                              facecolor='none', edgecolor='darkred', linewidth=3)

    axes[3].set_ylabel('Predicted Class', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Genomic Position (bp)', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim([-0.1, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Read plot saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_multiple_4class_reads(xy_data, predictions_df,
                               annotations=None,
                               n_reads=6,
                               random_seed=42,
                               save_dir=None,
                               filter_by='any'):
    """
    Plot 4-class predictions for multiple reads.

    Parameters
    ----------
    xy_data : pd.DataFrame
        XY signal data
    predictions_df : pd.DataFrame
        Predictions with columns: read_id, prob_background, prob_left_fork,
        prob_right_fork, prob_origin, predicted_class
    annotations : dict, optional
        Dictionary with 'left_forks', 'right_forks', 'origins' DataFrames
    n_reads : int
        Number of reads to plot
    random_seed : int
        Random seed for reproducibility
    save_dir : str, optional
        Directory to save plots
    filter_by : str
        Filter reads: 'any', 'origins', 'forks', 'all' (with all features)
    """
    np.random.seed(random_seed)

    # Filter reads based on criteria
    read_ids = predictions_df['read_id'].unique()

    if filter_by == 'origins':
        # Only reads with predicted origins
        origin_reads = predictions_df[predictions_df['predicted_class'] == 3]['read_id'].unique()
        read_ids = origin_reads
        print(f"🔍 Filtering to {len(read_ids)} reads with predicted origins")
    elif filter_by == 'forks':
        # Only reads with predicted forks
        fork_reads = predictions_df[
            (predictions_df['predicted_class'] == 1) |
            (predictions_df['predicted_class'] == 2)
        ]['read_id'].unique()
        read_ids = fork_reads
        print(f"🔍 Filtering to {len(read_ids)} reads with predicted forks")
    elif filter_by == 'all':
        # Only reads with all feature types
        reads_with_features = []
        for read_id in read_ids:
            read_preds = predictions_df[predictions_df['read_id'] == read_id]
            has_left = (read_preds['predicted_class'] == 1).any()
            has_right = (read_preds['predicted_class'] == 2).any()
            has_origin = (read_preds['predicted_class'] == 3).any()
            if has_left and has_right and has_origin:
                reads_with_features.append(read_id)
        read_ids = np.array(reads_with_features)
        print(f"🔍 Filtering to {len(read_ids)} reads with all feature types")

    if len(read_ids) == 0:
        print("⚠️ No reads match the filter criteria!")
        return

    # Randomly select reads
    selected_reads = np.random.choice(read_ids, size=min(n_reads, len(read_ids)), replace=False)

    for idx, read_id in enumerate(selected_reads):
        print(f"📊 Plotting read {idx+1}/{len(selected_reads)}: {read_id}")

        read_df = xy_data[xy_data['read_id'] == read_id].sort_values('center')
        read_preds = predictions_df[predictions_df['read_id'] == read_id].sort_values('start')

        # Get 4-class predictions
        preds = read_preds[[
            'prob_background',
            'prob_left_fork',
            'prob_right_fork',
            'prob_origin'
        ]].values

        # Get annotations for this read
        read_anns = None
        if annotations:
            read_anns = {}
            for key in ['left_forks', 'right_forks', 'origins']:
                if key in annotations and len(annotations[key]) > 0:
                    read_anns[key] = annotations[key][annotations[key]['read_id'] == read_id]
                else:
                    read_anns[key] = pd.DataFrame()

        # Plot
        save_path = None
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / f'read_{read_id}_4class_prediction.png'

        plot_4class_prediction(
            read_df, preds, read_anns,
            save_path=save_path,
            title=f'4-Class Prediction - Read: {read_id}'
        )

    print(f"\n✅ Plotted {len(selected_reads)} reads")
    if save_dir:
        print(f"📁 Plots saved to: {save_dir}")


def plot_class_distribution(predictions_df, save_path=None):
    """
    Plot distribution of predicted classes.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with 'predicted_class' column
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count by class
    class_counts = predictions_df['predicted_class'].value_counts().sort_index()
    class_names = ['Background', 'Left Fork', 'Right Fork', 'Origin']
    colors = ['gray', 'orange', 'purple', 'red']

    # Bar plot
    axes[0].bar(range(len(class_counts)), class_counts.values, color=colors, alpha=0.7)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylabel('Number of Segments', fontsize=12, fontweight='bold')
    axes[0].set_title('Segment-Level Class Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add percentages
    total = class_counts.sum()
    for i, count in enumerate(class_counts.values):
        pct = count / total * 100
        axes[0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Pie chart
    axes[1].pie(class_counts.values, labels=class_names, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Class Proportions', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Distribution plot saved: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_probability_heatmap(predictions_df, n_reads=20, random_seed=42, save_path=None):
    """
    Plot heatmap of class probabilities across multiple reads.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with prob_* columns
    n_reads : int
        Number of reads to include
    random_seed : int
        Random seed
    save_path : str, optional
        Path to save figure
    """
    np.random.seed(random_seed)

    # Select random reads
    read_ids = predictions_df['read_id'].unique()
    selected_reads = np.random.choice(read_ids, size=min(n_reads, len(read_ids)), replace=False)

    # Create figure with 4 subplots (one per class)
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    class_names = ['Background', 'Left Fork', 'Right Fork', 'Origin']
    prob_cols = ['prob_background', 'prob_left_fork', 'prob_right_fork', 'prob_origin']
    cmaps = ['Greys', 'Oranges', 'Purples', 'Reds']

    for class_idx, (class_name, prob_col, cmap) in enumerate(zip(class_names, prob_cols, cmaps)):
        # Prepare data matrix
        matrices = []
        for read_id in selected_reads:
            read_data = predictions_df[predictions_df['read_id'] == read_id].sort_values('start')
            if len(read_data) > 0:
                matrices.append(read_data[prob_col].values)

        # Pad to same length
        max_len = max(len(m) for m in matrices)
        padded = np.zeros((len(matrices), max_len))
        for i, m in enumerate(matrices):
            padded[i, :len(m)] = m

        # Plot heatmap
        im = axes[class_idx].imshow(padded, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        axes[class_idx].set_ylabel(f'{class_name}\nRead Index', fontsize=11, fontweight='bold')
        axes[class_idx].set_yticks(range(0, len(selected_reads), max(1, len(selected_reads)//10)))

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[class_idx], fraction=0.046, pad=0.04)
        cbar.set_label('Probability', rotation=270, labelpad=20, fontweight='bold')

    axes[-1].set_xlabel('Segment Index', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Class Probabilities Across {len(selected_reads)} Reads',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Heatmap saved: {save_path}")
        plt.close()
    else:
        plt.show()
