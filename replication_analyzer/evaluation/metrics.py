"""
Evaluation metrics for ORI and Fork detection models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    matthews_corrcoef, classification_report, confusion_matrix
)


def calculate_binary_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive metrics for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities

    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    # Add probability-based metrics if available
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)

    return metrics


def calculate_multiclass_metrics(y_true, y_pred, y_pred_proba=None, class_names=None):
    """
    Calculate comprehensive metrics for multi-class classification.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (n_samples, n_classes)
    class_names : list, optional
        Names of classes

    Returns
    -------
    dict
        Dictionary of metrics
    """
    if class_names is None:
        class_names = [f'class_{i}' for i in range(len(np.unique(y_true)))]

    # Overall metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    metrics['per_class'] = {}
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': int(support[i])
        }

    # Macro averages
    metrics['precision_macro'] = np.mean(precision)
    metrics['recall_macro'] = np.mean(recall)
    metrics['f1_macro'] = np.mean(f1)

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def print_metrics(metrics, title="Model Metrics"):
    """
    Pretty print metrics.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary
    title : str
        Title to display
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)

    # Print overall metrics
    for key, value in metrics.items():
        if key not in ['per_class', 'confusion_matrix']:
            print(f"  {key:20s}: {value:.4f}")

    # Print per-class metrics if available
    if 'per_class' in metrics:
        print("\n" + "-"*60)
        print("Per-Class Metrics:")
        print("-"*60)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-"*60)

        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<15} "
                  f"{class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} "
                  f"{class_metrics['f1_score']:<12.4f} "
                  f"{class_metrics['support']:<12}")

    print("="*60 + "\n")


def find_optimal_threshold(y_true, y_pred_proba, target_recall=0.9):
    """
    Find optimal probability threshold for a target recall.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    target_recall : float
        Target recall value

    Returns
    -------
    tuple
        (optimal_threshold, achieved_recall, precision_at_threshold)
    """
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Find threshold closest to target recall
    idx = np.argmin(np.abs(recalls - target_recall))
    optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    print(f"\n🎯 Threshold optimization for {target_recall:.1%} recall:")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   Achieved recall: {recalls[idx]:.4f}")
    print(f"   Precision at this recall: {precisions[idx]:.4f}")

    return optimal_threshold, recalls[idx], precisions[idx]


from sklearn.metrics import precision_recall_fscore_support
