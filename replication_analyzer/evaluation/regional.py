"""
Per-region analysis (centromere, pericentromere, arms).
"""

import numpy as np
import pandas as pd
from .metrics import calculate_binary_metrics, print_metrics


def assign_genomic_regions(annotated_data, regions_dict):
    """
    Assign each segment to a genomic region.

    Parameters
    ----------
    annotated_data : pd.DataFrame
        Data with predictions and columns: chr, start, end
    regions_dict : dict
        Dictionary with 'centromere' and 'pericentromere' DataFrames

    Returns
    -------
    pd.DataFrame
        Original data with 'genomic_region' column added
    """
    print("\n🗺️  Assigning genomic regions to segments...")

    annotated_data = annotated_data.copy()
    annotated_data['genomic_region'] = 'arm'  # Default to arm

    total_segments = len(annotated_data)

    # Assign centromeres
    if 'centromere' in regions_dict:
        print("  Assigning centromeres...")
        for _, region in regions_dict['centromere'].iterrows():
            mask = (
                (annotated_data['chr'] == region['chr']) &
                (annotated_data['start'] >= region['start']) &
                (annotated_data['end'] <= region['end'])
            )
            annotated_data.loc[mask, 'genomic_region'] = 'centromere'

    # Assign pericentromeres
    if 'pericentromere' in regions_dict:
        print("  Assigning pericentromeres...")
        for _, region in regions_dict['pericentromere'].iterrows():
            mask = (
                (annotated_data['chr'] == region['chr']) &
                (annotated_data['start'] >= region['start']) &
                (annotated_data['end'] <= region['end']) &
                (annotated_data['genomic_region'] == 'arm')  # Don't override centromeres
            )
            annotated_data.loc[mask, 'genomic_region'] = 'pericentromere'

    # Summary
    print(f"\n  ✅ Region assignment complete!")
    print(f"\n  📊 Segment distribution:")
    for region in ['centromere', 'pericentromere', 'arm']:
        count = (annotated_data['genomic_region'] == region).sum()
        pct = count / total_segments * 100
        print(f"    {region:15s}: {count:8,} ({pct:5.2f}%)")

    return annotated_data


def calculate_regional_metrics(annotated_data, y_true_col='y_true',
                               y_pred_col='y_pred', y_proba_col=None):
    """
    Calculate metrics for each genomic region.

    Parameters
    ----------
    annotated_data : pd.DataFrame
        Data with 'genomic_region', y_true, y_pred columns
    y_true_col : str
        Column name for true labels
    y_pred_col : str
        Column name for predictions
    y_proba_col : str, optional
        Column name for probabilities

    Returns
    -------
    pd.DataFrame
        Metrics for each region
    """
    print("\n📊 Calculating metrics by region...")

    regions = ['centromere', 'pericentromere', 'arm', 'overall']
    metrics_list = []

    for region in regions:
        if region == 'overall':
            region_data = annotated_data
        else:
            region_data = annotated_data[annotated_data['genomic_region'] == region]

        if len(region_data) == 0:
            print(f"  ⚠️  No data for {region}")
            continue

        y_true = region_data[y_true_col].values
        y_pred = region_data[y_pred_col].values
        y_proba = region_data[y_proba_col].values if y_proba_col else None

        # Calculate metrics
        region_metrics = calculate_binary_metrics(y_true, y_pred, y_proba)
        region_metrics['region'] = region
        region_metrics['n_segments'] = len(region_data)
        region_metrics['n_positive_true'] = int(y_true.sum())
        region_metrics['n_positive_pred'] = int(y_pred.sum())

        metrics_list.append(region_metrics)

        print(f"\n  {region.upper()}:")
        print(f"    Segments: {region_metrics['n_segments']:,}")
        print(f"    F1-Score: {region_metrics['f1_score']:.4f}")
        print(f"    Precision: {region_metrics['precision']:.4f}")
        print(f"    Recall: {region_metrics['recall']:.4f}")

    metrics_df = pd.DataFrame(metrics_list)

    # Reorder columns
    cols = ['region', 'n_segments', 'n_positive_true', 'n_positive_pred',
            'accuracy', 'precision', 'recall', 'f1_score', 'kappa', 'mcc']
    if 'roc_auc' in metrics_df.columns:
        cols.extend(['roc_auc', 'average_precision'])

    metrics_df = metrics_df[cols]

    return metrics_df


def compare_regional_performance(metrics_df, output_file=None):
    """
    Create a comparison table of regional performance.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Regional metrics dataframe
    output_file : str, optional
        Path to save comparison table

    Returns
    -------
    pd.DataFrame
        Formatted comparison table
    """
    # Select key metrics for comparison
    comparison = metrics_df[['region', 'f1_score', 'precision', 'recall', 'accuracy']].copy()

    # Format as percentages
    for col in ['f1_score', 'precision', 'recall', 'accuracy']:
        comparison[col] = (comparison[col] * 100).round(2)

    # Rename columns
    comparison.columns = ['Region', 'F1-Score (%)', 'Precision (%)', 'Recall (%)', 'Accuracy (%)']

    print("\n" + "="*70)
    print("REGIONAL PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison.to_string(index=False))
    print("="*70 + "\n")

    if output_file:
        comparison.to_csv(output_file, index=False)
        print(f"✅ Comparison table saved: {output_file}")

    return comparison


def analyze_region_specific_errors(annotated_data, y_true_col='y_true', y_pred_col='y_pred'):
    """
    Analyze error patterns by genomic region.

    Parameters
    ----------
    annotated_data : pd.DataFrame
        Annotated data with regions and predictions
    y_true_col : str
        True labels column
    y_pred_col : str
        Predicted labels column

    Returns
    -------
    pd.DataFrame
        Error analysis summary
    """
    regions = ['centromere', 'pericentromere', 'arm']
    error_stats = []

    for region in regions:
        region_data = annotated_data[annotated_data['genomic_region'] == region]

        if len(region_data) == 0:
            continue

        y_true = region_data[y_true_col].values
        y_pred = region_data[y_pred_col].values

        # Calculate error types
        true_pos = np.sum((y_true == 1) & (y_pred == 1))
        true_neg = np.sum((y_true == 0) & (y_pred == 0))
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        false_neg = np.sum((y_true == 1) & (y_pred == 0))

        total = len(region_data)

        error_stats.append({
            'region': region,
            'total': total,
            'true_positive': true_pos,
            'true_negative': true_neg,
            'false_positive': false_pos,
            'false_negative': false_neg,
            'tp_rate': true_pos / total * 100,
            'tn_rate': true_neg / total * 100,
            'fp_rate': false_pos / total * 100,
            'fn_rate': false_neg / total * 100
        })

    error_df = pd.DataFrame(error_stats)

    print("\n" + "="*70)
    print("ERROR ANALYSIS BY REGION")
    print("="*70)
    for _, row in error_df.iterrows():
        print(f"\n{row['region'].upper()}:")
        print(f"  True Positives:  {row['true_positive']:6,} ({row['tp_rate']:5.2f}%)")
        print(f"  True Negatives:  {row['true_negative']:6,} ({row['tn_rate']:5.2f}%)")
        print(f"  False Positives: {row['false_positive']:6,} ({row['fp_rate']:5.2f}%)")
        print(f"  False Negatives: {row['false_negative']:6,} ({row['fn_rate']:5.2f}%)")

    print("="*70 + "\n")

    return error_df
