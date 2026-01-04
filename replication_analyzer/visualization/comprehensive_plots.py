"""
Comprehensive visualization utilities matching notebook style.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    accuracy_score, cohen_kappa_score
)

sns.set_style("whitegrid")


# ============================================
# 1. TRAINING HISTORY PLOTS
# ============================================

def plot_fork_training_history(history, model_name='Fork_Detector'):
    """
    Comprehensive training history visualization for fork model
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    axes[0, 0].plot(history['loss'], label='Train', lw=2, color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val', lw=2, color='red')
    axes[0, 0].set_title('Loss', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1-Score (macro)
    if 'f1_macro' in history:
        axes[0, 1].plot(history['f1_macro'], label='Train', lw=2, color='purple')
        axes[0, 1].plot(history['val_f1_macro'], label='Val', lw=2, color='darkviolet')
        axes[0, 1].set_title('Macro F1-Score ⭐', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        best_f1 = max(history['val_f1_macro'])
        axes[0, 1].axhline(y=best_f1, color='green',
                          linestyle='--', alpha=0.5, label=f'Best: {best_f1:.4f}')
        axes[0, 1].legend()

    # Accuracy
    axes[0, 2].plot(history['accuracy'], label='Train', lw=2, color='blue')
    axes[0, 2].plot(history['val_accuracy'], label='Val', lw=2, color='red')
    axes[0, 2].set_title('Accuracy', fontweight='bold', fontsize=12)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Categorical Accuracy (if available)
    if 'categorical_accuracy' in history:
        axes[1, 0].plot(history['categorical_accuracy'],
                       label='Train', lw=2, color='green')
        axes[1, 0].plot(history['val_categorical_accuracy'],
                       label='Val', lw=2, color='darkgreen')
        axes[1, 0].set_title('Categorical Accuracy', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Categorical Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')

    # Loss (Log Scale)
    axes[1, 1].plot(history['loss'], label='Train Loss', lw=2, color='blue')
    axes[1, 1].plot(history['val_loss'], label='Val Loss', lw=2, color='red')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Loss (Log Scale)', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Summary
    axes[1, 2].axis('off')

    if 'val_f1_macro' in history:
        best_idx = np.argmax(history['val_f1_macro'])
        best_f1 = history['val_f1_macro'][best_idx]
    else:
        best_idx = np.argmax(history['val_accuracy'])
        best_f1 = None

    summary = f"""
TRAINING SUMMARY
{'='*35}

Best Epoch: {best_idx + 1}

Validation Metrics:
  Accuracy:  {history['val_accuracy'][best_idx]:.4f}
"""

    if best_f1 is not None:
        summary += f"  F1-Macro:  {best_f1:.4f} ⭐\n"

    summary += f"""
Training:
  Total epochs: {len(history['loss'])}
  Final loss:   {history['loss'][-1]:.4f}
  Val loss:     {history['val_loss'][-1]:.4f}
    """

    axes[1, 2].text(0.05, 0.5, summary, fontsize=9, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================
# 2. DETAILED EVALUATION PLOTS (3-CLASS)
# ============================================

def plot_fork_evaluation_comprehensive(y_true, y_pred, y_proba, model_name='Fork_Detector'):
    """
    Comprehensive evaluation visualization for 3-class fork model

    Parameters:
    -----------
    y_true : array-like
        True class labels (0=background, 1=left, 2=right)
    y_pred : array-like
        Predicted class labels
    y_proba : array-like
        Predicted probabilities, shape (n_samples, 3)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if y_proba.ndim == 1:
        # If only 1D, assume binary and create 3-class proba
        y_proba_3d = np.zeros((len(y_proba), 3))
        y_proba_3d[:, y_pred] = 1.0
        y_proba = y_proba_3d

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    class_names = ['Background', 'Left Fork', 'Right Fork']

    # 1. Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # 2. Per-class metrics
    ax = axes[0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    x = np.arange(3)
    width = 0.25

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='blue')
    ax.bar(x, recall, width, label='Recall', alpha=0.8, color='green')
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='red')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Add values on bars
    for i in range(3):
        ax.text(i - width, precision[i] + 0.02, f'{precision[i]:.2f}',
               ha='center', fontsize=8)
        ax.text(i, recall[i] + 0.02, f'{recall[i]:.2f}',
               ha='center', fontsize=8)
        ax.text(i + width, f1[i] + 0.02, f'{f1[i]:.2f}',
               ha='center', fontsize=8)

    # 3. Class distribution
    ax = axes[0, 2]
    true_counts = [np.sum(y_true == i) for i in range(3)]
    pred_counts = [np.sum(y_pred == i) for i in range(3)]

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='coral')

    ax.set_ylabel('Count')
    ax.set_title('Class Distribution', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add counts on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}', ha='center', va='bottom', fontsize=8)

    # 4. Normalized confusion matrix
    ax = axes[0, 3]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion'})
    ax.set_title('Normalized Confusion Matrix', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    # 5. Probability distributions for each class
    ax = axes[1, 0]
    for i, (name, color) in enumerate(zip(class_names, ['gray', 'blue', 'orange'])):
        # Get probabilities for samples that are truly class i
        mask = y_true == i
        if mask.sum() > 0:
            probs_i = y_proba[mask, i]
            ax.hist(probs_i, bins=30, alpha=0.5, color=color,
                   label=f'{name} (n={mask.sum():,})', density=True)

    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax.set_xlabel('Predicted Probability (for true class)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Probability Distribution by True Class', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Per-class probability heatmap
    ax = axes[1, 1]
    # Create matrix: rows = true class, cols = predicted prob for each class
    prob_matrix = np.zeros((3, 3))
    for true_class in range(3):
        mask = y_true == true_class
        if mask.sum() > 0:
            for pred_class in range(3):
                prob_matrix[true_class, pred_class] = np.mean(y_proba[mask, pred_class])

    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                vmin=0, vmax=1,
                cbar_kws={'label': 'Mean Probability'})
    ax.set_title('Mean Predicted Probabilities', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Probability For')

    # 7. Error Analysis
    ax = axes[1, 2]

    # Count errors by type
    correct = y_true == y_pred
    n_correct = correct.sum()
    n_errors = (~correct).sum()

    # Breakdown of errors
    error_types = []
    error_counts = []

    for true_class in range(3):
        for pred_class in range(3):
            if true_class != pred_class:
                n = ((y_true == true_class) & (y_pred == pred_class)).sum()
                if n > 0:
                    error_types.append(f'{class_names[true_class][:4]}\n→{class_names[pred_class][:4]}')
                    error_counts.append(n)

    colors_err = ['red'] * len(error_types)
    bars = ax.bar(range(len(error_types)), error_counts, color=colors_err, alpha=0.7)
    ax.set_xticks(range(len(error_types)))
    ax.set_xticklabels(error_types, rotation=0, fontsize=9)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Types', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add counts
    for bar, count in zip(bars, error_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count:,}\n({count/len(y_true)*100:.1f}%)',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 8. Metrics Summary
    ax = axes[1, 3]
    ax.axis('off')

    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    # Calculate weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)

    metrics_summary = f"""
EVALUATION METRICS
{'='*35}

Overall:
  Accuracy:   {accuracy_score(y_true, y_pred):.4f}
  Kappa:      {cohen_kappa_score(y_true, y_pred):.4f}

Macro Average:
  Precision:  {macro_precision:.4f}
  Recall:     {macro_recall:.4f}
  F1-Score:   {macro_f1:.4f} ⭐

Weighted Average:
  Precision:  {weighted_precision:.4f}
  Recall:     {weighted_recall:.4f}
  F1-Score:   {weighted_f1:.4f}

Per-Class F1:
  Background: {f1[0]:.4f}
  Left Fork:  {f1[1]:.4f} ⭐
  Right Fork: {f1[2]:.4f} ⭐

Dataset:
  Total:      {len(y_true):,} segments
  Correct:    {n_correct:,} ({n_correct/len(y_true)*100:.1f}%)
  Errors:     {n_errors:,} ({n_errors/len(y_true)*100:.1f}%)
    """

    ax.text(0.05, 0.5, metrics_summary, fontsize=9, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(f'{model_name} - Comprehensive Evaluation',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================
# 3. COMPLETE VISUALIZATION PIPELINE
# ============================================

def generate_comprehensive_plots(history_csv, predictions_tsv,
                                model_name='Fork_Detector',
                                save_dir='.'):
    """
    Generate all visualization plots for fork model

    Parameters
    ----------
    history_csv : str
        Path to training history CSV
    predictions_tsv : str
        Path to predictions TSV
    model_name : str
        Name for plot titles
    save_dir : str
        Directory to save plots

    Returns
    -------
    dict
        Dictionary of generated figures
    """
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATION PLOTS")
    print("="*70)

    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Load data
    print("\n📂 Loading data...")
    history_df = pd.read_csv(history_csv)
    history = {col: history_df[col].values for col in history_df.columns}
    print(f"   ✅ Loaded training history: {len(history_df)} epochs")

    predictions_df = pd.read_csv(predictions_tsv, sep='\t')
    y_true = predictions_df['y_true'].values
    y_pred = predictions_df['predicted_class'].values
    y_proba = predictions_df[['class_0_prob', 'class_1_prob', 'class_2_prob']].values
    print(f"   ✅ Loaded predictions: {len(predictions_df):,} segments")

    # 1. Training history
    print("\n📊 Generating training history plots...")
    fig1 = plot_fork_training_history(history, model_name)
    figures['training_history'] = fig1
    save_file = save_path / f'{model_name.lower()}_training_history.png'
    fig1.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_file}")

    # 2. Comprehensive evaluation
    print("\n📊 Generating comprehensive evaluation plots...")
    fig2 = plot_fork_evaluation_comprehensive(y_true, y_pred, y_proba, model_name)
    figures['evaluation'] = fig2
    save_file = save_path / f'{model_name.lower()}_evaluation.png'
    fig2.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_file}")

    print("\n✅ All plots generated successfully!")
    print(f"   Total figures: {len(figures)}")

    return figures
