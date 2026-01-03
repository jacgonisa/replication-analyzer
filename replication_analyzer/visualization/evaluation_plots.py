"""
Visualization utilities for model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Set style
sns.set_style("whitegrid")


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix with annotations.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Class names for labels
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names,
               cbar_kws={'label': 'Count'})

    # Add percentages
    total = cm.sum()
    for i in range(len(cm)):
        for j in range(len(cm)):
            text = plt.gca().texts[i*len(cm) + j]
            text.set_text(f'{cm[i,j]:,}\n({cm[i,j]/total*100:.1f}%)')

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")

    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None, title='ROC Curve'):
    """
    Plot ROC curve with AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved: {save_path}")

    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None, title='Precision-Recall Curve'):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    save_path : str, optional
        Path to save figure
    title : str
        Plot title
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ PR curve saved: {save_path}")

    plt.show()


def plot_comprehensive_evaluation(y_true, y_pred, y_pred_proba=None,
                                   class_names=None, save_dir=None):
    """
    Create comprehensive evaluation plot (all metrics in one figure).

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities
    class_names : list, optional
        Class names
    save_dir : str, optional
        Directory to save plots
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    fig = plt.figure(figsize=(18, 12))

    # Layout: 2x3 grid
    # Row 1: Confusion Matrix, ROC Curve, PR Curve
    # Row 2: Metrics Bar Chart, Class Distribution, Summary

    # Plot 1: Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=class_names, yticklabels=class_names)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    ax1.set_title('Confusion Matrix', fontweight='bold')

    # Plot 2: ROC Curve (if probabilities available)
    ax2 = plt.subplot(2, 3, 2)
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'ROC Curve\n(Binary classification only)',
                ha='center', va='center')
        ax2.set_title('ROC Curve', fontweight='bold')

    # Plot 3: Precision-Recall Curve
    ax3 = plt.subplot(2, 3, 3)
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ax3.plot(recall, precision, color='blue', lw=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'PR Curve\n(Binary classification only)',
                ha='center', va='center')
        ax3.set_title('Precision-Recall Curve', fontweight='bold')

    # Plot 4: Metrics Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    ax4.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green', 'purple'], alpha=0.7)
    ax4.set_ylabel('Score')
    ax4.set_title('Overall Metrics', fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (k, v) in enumerate(metrics.items()):
        ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Plot 5: Class Distribution
    ax5 = plt.subplot(2, 3, 5)
    unique, counts_true = np.unique(y_true, return_counts=True)
    _, counts_pred = np.unique(y_pred, return_counts=True)
    x = np.arange(len(unique))
    width = 0.35
    ax5.bar(x - width/2, counts_true, width, label='True', alpha=0.7)
    ax5.bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.7)
    ax5.set_xlabel('Class')
    ax5.set_ylabel('Count')
    ax5.set_title('Class Distribution', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(class_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary = f"""
EVALUATION SUMMARY

Total Samples: {len(y_true):,}

Metrics:
  Accuracy:  {metrics['Accuracy']:.4f}
  Precision: {metrics['Precision']:.4f}
  Recall:    {metrics['Recall']:.4f}
  F1-Score:  {metrics['F1-Score']:.4f} ⭐

Class Distribution:
"""
    for i, name in enumerate(class_names):
        summary += f"  {name}: {counts_true[i]:,}\n"

    ax6.text(0.1, 0.5, summary, fontsize=9, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / 'comprehensive_evaluation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive evaluation plot saved: {save_path}")

    plt.show()


def plot_regional_comparison(metrics_df, save_path=None):
    """
    Plot regional performance comparison.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Regional metrics dataframe with columns: region, f1_score, precision, recall
    save_path : str, optional
        Path to save figure
    """
    # Filter out overall
    region_metrics = metrics_df[metrics_df['region'] != 'overall'].copy()

    if len(region_metrics) == 0:
        print("⚠️ No regional data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Grouped bar chart
    metrics_to_plot = ['f1_score', 'precision', 'recall']
    x = np.arange(len(region_metrics))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        axes[0].bar(x + i*width, region_metrics[metric],
                   width, label=metric.replace('_', ' ').title())

    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Regional Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(region_metrics['region'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])

    # Plot 2: Heatmap
    heatmap_data = region_metrics[['region'] + metrics_to_plot].set_index('region').T
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
               vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Score'})
    axes[1].set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Regional comparison plot saved: {save_path}")

    plt.show()
