#!/usr/bin/env python
"""
Advanced evaluation script with multi-class ROC, PR curves, calibration, and threshold analysis.

Usage:
    python scripts/advanced_evaluation.py \
        --predictions results/case_study_jan2026/combined/evaluation/predictions.tsv \
        --output results/case_study_jan2026/combined/advanced_evaluation
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

sns.set_style("whitegrid")


def plot_multiclass_roc_curves(y_true, y_pred_proba, class_names, save_path):
    """
    Plot ROC curves for each class using one-vs-rest approach.
    """
    n_classes = len(class_names)

    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (ax, class_name, color) in enumerate(zip(axes, class_names, colors)):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{class_name}\nROC Curve', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                label=f'Optimal threshold: {optimal_threshold:.3f}')
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Multi-class ROC curves saved: {save_path}")
    plt.close()


def plot_multiclass_pr_curves(y_true, y_pred_proba, class_names, save_path):
    """
    Plot Precision-Recall curves for each class.
    """
    n_classes = len(class_names)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (ax, class_name, color) in enumerate(zip(axes, class_names, colors)):
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(
            y_true_bin[:, i], y_pred_proba[:, i]
        )
        avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

        # Plot
        ax.plot(recall, precision, color=color, lw=2,
                label=f'AP = {avg_precision:.4f}')

        # Baseline (random classifier)
        baseline = y_true_bin[:, i].sum() / len(y_true_bin)
        ax.axhline(y=baseline, color='k', linestyle='--', lw=2, alpha=0.3,
                   label=f'Random (AP = {baseline:.4f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'{class_name}\nPrecision-Recall Curve', fontsize=12, fontweight='bold')
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add F1-optimal point
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        ax.plot(recall[optimal_idx], precision[optimal_idx], 'ro', markersize=8,
                label=f'Best F1: {f1_scores[optimal_idx]:.3f}')
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Multi-class PR curves saved: {save_path}")
    plt.close()


def plot_calibration_curves(y_true, y_pred_proba, class_names, save_path):
    """
    Plot calibration (reliability) diagrams for each class.
    """
    n_classes = len(class_names)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (ax, class_name, color) in enumerate(zip(axes, class_names, colors)):
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_bin[:, i], y_pred_proba[:, i], n_bins=10, strategy='uniform'
        )

        # Calculate Brier score
        brier = brier_score_loss(y_true_bin[:, i], y_pred_proba[:, i])

        # Plot
        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                color=color, lw=2, markersize=8,
                label=f'{class_name} (Brier: {brier:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Perfect calibration')

        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_title(f'{class_name}\nCalibration Curve', fontsize=12, fontweight='bold')
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Calibration curves saved: {save_path}")
    plt.close()


def plot_confidence_distributions(y_true, y_pred, y_pred_proba, class_names, save_path):
    """
    Plot distribution of prediction confidences for correct vs incorrect predictions.
    """
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))

    colors_correct = ['#2ecc71', '#27ae60', '#229954']
    colors_incorrect = ['#e74c3c', '#c0392b', '#a93226']

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Get confidences for this class
        class_probs = y_pred_proba[:, i]

        # Separate correct vs incorrect
        correct_mask = (y_pred == y_true)
        correct_probs = class_probs[correct_mask]
        incorrect_probs = class_probs[~correct_mask]

        # Plot histograms
        ax.hist(correct_probs, bins=50, alpha=0.6, color=colors_correct[i],
                label=f'Correct (n={len(correct_probs):,})', density=True)
        ax.hist(incorrect_probs, bins=50, alpha=0.6, color=colors_incorrect[i],
                label=f'Incorrect (n={len(incorrect_probs):,})', density=True)

        ax.set_xlabel(f'{class_name} Probability', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{class_name}\nConfidence Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean lines
        ax.axvline(correct_probs.mean(), color=colors_correct[i],
                   linestyle='--', lw=2, alpha=0.8,
                   label=f'Mean correct: {correct_probs.mean():.3f}')
        ax.axvline(incorrect_probs.mean(), color=colors_incorrect[i],
                   linestyle='--', lw=2, alpha=0.8,
                   label=f'Mean incorrect: {incorrect_probs.mean():.3f}')
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confidence distributions saved: {save_path}")
    plt.close()


def plot_threshold_analysis(y_true, y_pred_proba, class_names, save_path):
    """
    Analyze precision, recall, F1 at different confidence thresholds.
    """
    n_classes = len(class_names)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))

    thresholds = np.linspace(0, 1, 100)

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        precisions = []
        recalls = []
        f1_scores = []

        for thresh in thresholds:
            # Predict based on threshold
            y_pred_thresh = (y_pred_proba[:, i] >= thresh).astype(int)

            # Calculate metrics
            tp = ((y_pred_thresh == 1) & (y_true_bin[:, i] == 1)).sum()
            fp = ((y_pred_thresh == 1) & (y_true_bin[:, i] == 0)).sum()
            fn = ((y_pred_thresh == 0) & (y_true_bin[:, i] == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Plot
        ax.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
        ax.plot(thresholds, recalls, 'r-', lw=2, label='Recall')
        ax.plot(thresholds, f1_scores, 'g-', lw=2, label='F1-Score')

        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_thresh = thresholds[optimal_idx]
        ax.axvline(optimal_thresh, color='purple', linestyle='--', lw=2,
                   label=f'Optimal: {optimal_thresh:.3f}')
        ax.plot(optimal_thresh, f1_scores[optimal_idx], 'ro', markersize=10)

        ax.set_xlabel('Confidence Threshold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{class_name}\nThreshold Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Threshold analysis saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Advanced multi-class evaluation')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions TSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ADVANCED MULTI-CLASS EVALUATION")
    print("="*70)

    # Load predictions
    print(f"\n📂 Loading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions, sep='\t')

    # Extract data
    y_true = df['y_true'].values
    y_pred = df['predicted_class'].values
    y_pred_proba = df[['class_0_prob', 'class_1_prob', 'class_2_prob']].values

    class_names = ['Background', 'Left Fork', 'Right Fork']

    print(f"✅ Loaded {len(y_true):,} predictions")
    print(f"   Classes: {class_names}")

    # Generate plots
    print("\n📊 Generating advanced evaluation plots...\n")

    print("1. Multi-class ROC curves (one-vs-rest)...")
    plot_multiclass_roc_curves(
        y_true, y_pred_proba, class_names,
        output_dir / 'roc_curves_multiclass.png'
    )

    print("2. Multi-class Precision-Recall curves...")
    plot_multiclass_pr_curves(
        y_true, y_pred_proba, class_names,
        output_dir / 'pr_curves_multiclass.png'
    )

    print("3. Calibration curves (reliability diagrams)...")
    plot_calibration_curves(
        y_true, y_pred_proba, class_names,
        output_dir / 'calibration_curves.png'
    )

    print("4. Confidence distributions...")
    plot_confidence_distributions(
        y_true, y_pred, y_pred_proba, class_names,
        output_dir / 'confidence_distributions.png'
    )

    print("5. Threshold analysis...")
    plot_threshold_analysis(
        y_true, y_pred_proba, class_names,
        output_dir / 'threshold_analysis.png'
    )

    print("\n" + "="*70)
    print("✅ ADVANCED EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n📁 All plots saved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  - roc_curves_multiclass.png")
    print("  - pr_curves_multiclass.png")
    print("  - calibration_curves.png")
    print("  - confidence_distributions.png")
    print("  - threshold_analysis.png")


if __name__ == '__main__':
    main()
