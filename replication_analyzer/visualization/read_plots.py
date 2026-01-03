"""
Visualization utilities for individual read predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")


def plot_read_prediction(read_df, predictions, annotations=None,
                         save_path=None, title=None):
    """
    Plot a single read with predictions and optional annotations.

    Parameters
    ----------
    read_df : pd.DataFrame
        Read data with columns: start, end, signal, center
    predictions : np.ndarray
        Predictions (probabilities)
    annotations : pd.DataFrame, optional
        True annotations (ORIs or forks) with columns: start, end
    save_path : str, optional
        Path to save figure
    title : str, optional
        Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    positions = read_df['center'].values
    signal = read_df['signal'].values

    # Panel 1: Raw signal
    axes[0].plot(positions, signal, 'b-', linewidth=2, alpha=0.7, label='BrdU Signal')
    axes[0].fill_between(positions, 0, signal, alpha=0.3)

    if annotations is not None:
        for _, ann in annotations.iterrows():
            axes[0].axvspan(ann['start'], ann['end'], alpha=0.3, color='red')
            axes[0].axvline((ann['start'] + ann['end'])/2, color='darkred',
                          linestyle='--', linewidth=2)

    axes[0].set_ylabel('BrdU Signal', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if title:
        axes[0].set_title(title, fontsize=14, fontweight='bold')

    # Panel 2: Predictions
    if len(predictions.shape) == 1:
        # Binary predictions
        axes[1].plot(positions, predictions, 'g-', linewidth=2, label='ORI Probability')
        axes[1].fill_between(positions, 0, predictions, alpha=0.3, color='green')
        axes[1].axhline(0.5, color='red', linestyle='--', linewidth=1, label='Threshold=0.5')
        axes[1].set_ylabel('ORI Probability', fontsize=12, fontweight='bold')
    else:
        # Multi-class predictions
        axes[1].plot(positions, predictions[:, 1], 'orange', linewidth=2, label='Left Fork', alpha=0.7)
        axes[1].plot(positions, predictions[:, 2], 'purple', linewidth=2, label='Right Fork', alpha=0.7)
        axes[1].set_ylabel('Fork Probability', fontsize=12, fontweight='bold')

    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Panel 3: Predicted regions
    if len(predictions.shape) == 1:
        # Binary
        predicted = (predictions > 0.5).astype(int)
        axes[2].fill_between(positions, 0, predicted, alpha=0.5, color='green', label='Predicted ORI')
    else:
        # Multi-class
        predicted_class = np.argmax(predictions, axis=1)
        left_mask = predicted_class == 1
        right_mask = predicted_class == 2

        axes[2].fill_between(positions, 0, left_mask, alpha=0.5, color='orange', label='Left Fork')
        axes[2].fill_between(positions, 0, right_mask, alpha=0.5, color='purple', label='Right Fork')

    if annotations is not None:
        for _, ann in annotations.iterrows():
            axes[2].axvspan(ann['start'], ann['end'], alpha=0.3, color='red')

    axes[2].set_ylabel('Prediction', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Genomic Position (bp)', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-0.1, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Read plot saved: {save_path}")

    plt.show()


def plot_multiple_reads(xy_data, predictions_df, annotations=None,
                        n_reads=6, random_seed=42, save_dir=None):
    """
    Plot predictions for multiple randomly selected reads.

    Parameters
    ----------
    xy_data : pd.DataFrame
        XY signal data
    predictions_df : pd.DataFrame
        Predictions with columns: read_id, ori_prob or class_*_prob
    annotations : pd.DataFrame, optional
        True annotations
    n_reads : int
        Number of reads to plot
    random_seed : int
        Random seed for reproducibility
    save_dir : str, optional
        Directory to save plots
    """
    np.random.seed(random_seed)

    read_ids = predictions_df['read_id'].unique()
    selected_reads = np.random.choice(read_ids, size=min(n_reads, len(read_ids)), replace=False)

    for read_id in selected_reads:
        read_df = xy_data[xy_data['read_id'] == read_id].sort_values('center')
        read_preds = predictions_df[predictions_df['read_id'] == read_id].sort_values('start')

        # Get predictions
        if 'ori_prob' in read_preds.columns:
            preds = read_preds['ori_prob'].values
        else:
            preds = read_preds[['class_0_prob', 'class_1_prob', 'class_2_prob']].values

        # Get annotations for this read
        read_anns = None
        if annotations is not None:
            read_anns = annotations[annotations['read_id'] == read_id]

        # Plot
        save_path = None
        if save_dir:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir) / f'read_{read_id}_prediction.png'

        plot_read_prediction(read_df, preds, read_anns, save_path=save_path, title=f'Read: {read_id}')
