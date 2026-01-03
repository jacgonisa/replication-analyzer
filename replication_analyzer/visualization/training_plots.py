"""
Visualization utilities for training history and progress.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)


def plot_training_history(history, save_path=None, model_name='Model'):
    """
    Plot comprehensive training history.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history object
    save_path : str, optional
        Path to save figure
    model_name : str
        Model name for title
    """
    history_dict = history.history if hasattr(history, 'history') else history

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Loss
    axes[0, 0].plot(history_dict['loss'], label='Train', linewidth=2, color='blue')
    axes[0, 0].plot(history_dict['val_loss'], label='Val', linewidth=2, color='red')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[0, 1].plot(history_dict['accuracy'], label='Train', linewidth=2, color='blue')
    axes[0, 1].plot(history_dict['val_accuracy'], label='Val', linewidth=2, color='red')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: F1-Score (if available)
    if 'f1' in history_dict or 'f1_macro' in history_dict:
        f1_key = 'f1' if 'f1' in history_dict else 'f1_macro'
        axes[0, 2].plot(history_dict[f1_key], label='Train', linewidth=2, color='purple')
        axes[0, 2].plot(history_dict[f'val_{f1_key}'], label='Val', linewidth=2, color='darkviolet')
        axes[0, 2].set_xlabel('Epoch', fontsize=11)
        axes[0, 2].set_ylabel('F1-Score', fontsize=11)
        axes[0, 2].set_title('F1-Score ⭐', fontsize=12, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'F1-Score not tracked', ha='center', va='center')
        axes[0, 2].set_title('F1-Score', fontsize=12, fontweight='bold')

    # Plot 4: Precision & Recall
    if 'precision' in history_dict:
        axes[1, 0].plot(history_dict['val_precision'], label='Precision', linewidth=2, color='orange')
        axes[1, 0].plot(history_dict['val_recall'], label='Recall', linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Score', fontsize=11)
        axes[1, 0].set_title('Precision vs Recall', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Precision/Recall not tracked', ha='center', va='center')
        axes[1, 0].set_title('Precision vs Recall', fontsize=12, fontweight='bold')

    # Plot 5: AUC (if available)
    if 'auc' in history_dict:
        axes[1, 1].plot(history_dict['auc'], label='Train', linewidth=2, color='blue')
        axes[1, 1].plot(history_dict['val_auc'], label='Val', linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('AUC', fontsize=11)
        axes[1, 1].set_title('ROC-AUC', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'AUC not tracked', ha='center', va='center')
        axes[1, 1].set_title('ROC-AUC', fontsize=12, fontweight='bold')

    # Plot 6: Summary
    axes[1, 2].axis('off')

    # Find best epoch
    monitor_key = 'val_f1' if 'val_f1' in history_dict else 'val_f1_macro' if 'val_f1_macro' in history_dict else 'val_loss'

    if 'loss' in monitor_key:
        best_idx = np.argmin(history_dict[monitor_key])
        best_value = history_dict[monitor_key][best_idx]
    else:
        best_idx = np.argmax(history_dict[monitor_key])
        best_value = history_dict[monitor_key][best_idx]

    summary = f"""
TRAINING SUMMARY

Best Epoch: {best_idx + 1}
Total Epochs: {len(history_dict['loss'])}

Best Metrics:
"""
    if 'val_f1' in history_dict:
        summary += f"  F1:        {history_dict['val_f1'][best_idx]:.4f} ⭐\n"
    if 'val_f1_macro' in history_dict:
        summary += f"  F1-Macro:  {history_dict['val_f1_macro'][best_idx]:.4f} ⭐\n"
    if 'val_accuracy' in history_dict:
        summary += f"  Accuracy:  {history_dict['val_accuracy'][best_idx]:.4f}\n"
    if 'val_precision' in history_dict:
        summary += f"  Precision: {history_dict['val_precision'][best_idx]:.4f}\n"
    if 'val_recall' in history_dict:
        summary += f"  Recall:    {history_dict['val_recall'][best_idx]:.4f}\n"

    axes[1, 2].text(0.1, 0.5, summary, fontsize=10, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(f'{model_name} Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training plot saved: {save_path}")

    plt.show()


def plot_loss_curves(history, save_path=None):
    """
    Simple loss curves plot (train vs val).

    Parameters
    ----------
    history : dict or keras.callbacks.History
        Training history
    save_path : str, optional
        Path to save figure
    """
    history_dict = history.history if hasattr(history, 'history') else history

    plt.figure(figsize=(10, 6))
    plt.plot(history_dict['loss'], label='Training Loss', linewidth=2)
    plt.plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Loss curves saved: {save_path}")

    plt.show()


def plot_learning_rate_schedule(history, save_path=None):
    """
    Plot learning rate schedule over training.

    Parameters
    ----------
    history : dict or keras.callbacks.History
        Training history
    save_path : str, optional
        Path to save figure
    """
    history_dict = history.history if hasattr(history, 'history') else history

    if 'lr' not in history_dict:
        print("⚠️ Learning rate not tracked in history")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history_dict['lr'], linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ LR schedule plot saved: {save_path}")

    plt.show()
