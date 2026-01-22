"""
Benchmarking utilities for comparing predicted origins against curated datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .bed_utils import (
    read_bed_file,
    find_overlapping_intervals,
    compute_overlap,
    compute_jaccard
)


def benchmark_ori_predictions(predicted_bed: str,
                              curated_bed: str,
                              min_overlap: int = 1,
                              jaccard_threshold: float = 0.0) -> Dict:
    """
    Benchmark predicted origins against a curated dataset.

    Parameters
    ----------
    predicted_bed : str
        Path to predicted origins BED file
    curated_bed : str
        Path to curated origins BED file
    min_overlap : int
        Minimum overlap in bp to consider a match
    jaccard_threshold : float
        Minimum Jaccard index to consider a true positive

    Returns
    -------
    dict
        Benchmarking metrics and statistics
    """
    print("=" * 70)
    print("BENCHMARKING ORIGIN PREDICTIONS")
    print("=" * 70)

    # Load BED files
    print(f"\nLoading predicted origins: {predicted_bed}")
    predicted_df = read_bed_file(predicted_bed)
    print(f"  → {len(predicted_df):,} predicted origins")

    print(f"\nLoading curated origins: {curated_bed}")
    curated_df = read_bed_file(curated_bed)
    print(f"  → {len(curated_df):,} curated origins")

    # Find overlaps
    print(f"\nFinding overlaps (min_overlap={min_overlap}bp, jaccard≥{jaccard_threshold})...")
    overlaps_df = find_overlapping_intervals(
        predicted_df, curated_df,
        min_overlap=min_overlap,
        same_chr=True
    )

    if len(overlaps_df) == 0:
        print("⚠️ No overlaps found!")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'n_predicted': len(predicted_df),
            'n_curated': len(curated_df),
            'n_true_positives': 0,
            'n_false_positives': len(predicted_df),
            'n_false_negatives': len(curated_df),
        }

    # Apply Jaccard threshold
    high_quality_overlaps = overlaps_df[overlaps_df['jaccard'] >= jaccard_threshold]

    # Count true positives (unique predicted intervals with at least one match)
    matched_predicted = high_quality_overlaps['query_idx'].unique()
    n_true_positives = len(matched_predicted)

    # Count false positives (predicted intervals with no match)
    n_false_positives = len(predicted_df) - n_true_positives

    # Count false negatives (curated intervals with no match)
    matched_curated = high_quality_overlaps['ref_idx'].unique()
    n_false_negatives = len(curated_df) - len(matched_curated)

    # Calculate metrics
    precision = n_true_positives / len(predicted_df) if len(predicted_df) > 0 else 0.0
    recall = n_true_positives / len(curated_df) if len(curated_df) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate overlap statistics
    overlap_stats = {
        'mean_jaccard': high_quality_overlaps['jaccard'].mean(),
        'median_jaccard': high_quality_overlaps['jaccard'].median(),
        'mean_overlap_bp': high_quality_overlaps['overlap_bp'].mean(),
        'median_overlap_bp': high_quality_overlaps['overlap_bp'].median(),
    }

    # Length statistics
    predicted_lengths = predicted_df['end'] - predicted_df['start']
    curated_lengths = curated_df['end'] - curated_df['start']

    length_stats = {
        'predicted_mean_length': predicted_lengths.mean(),
        'predicted_median_length': predicted_lengths.median(),
        'curated_mean_length': curated_lengths.mean(),
        'curated_median_length': curated_lengths.median(),
    }

    # Per-chromosome statistics
    per_chr_stats = compute_per_chromosome_metrics(
        predicted_df, curated_df, high_quality_overlaps
    )

    # Compile results
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_predicted': len(predicted_df),
        'n_curated': len(curated_df),
        'n_true_positives': n_true_positives,
        'n_false_positives': n_false_positives,
        'n_false_negatives': n_false_negatives,
        'overlap_stats': overlap_stats,
        'length_stats': length_stats,
        'per_chr_stats': per_chr_stats,
        'overlaps_df': overlaps_df,
        'high_quality_overlaps': high_quality_overlaps,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARKING RESULTS")
    print("=" * 70)
    print(f"\n📊 Overall Metrics:")
    print(f"  Precision:  {precision:.3f} ({n_true_positives}/{len(predicted_df)})")
    print(f"  Recall:     {recall:.3f} ({n_true_positives}/{len(curated_df)})")
    print(f"  F1 Score:   {f1:.3f}")
    print(f"\n📈 Confusion Matrix:")
    print(f"  True Positives:   {n_true_positives:,}")
    print(f"  False Positives:  {n_false_positives:,}")
    print(f"  False Negatives:  {n_false_negatives:,}")
    print(f"\n📏 Overlap Quality:")
    print(f"  Mean Jaccard:     {overlap_stats['mean_jaccard']:.3f}")
    print(f"  Median Jaccard:   {overlap_stats['median_jaccard']:.3f}")
    print(f"  Mean Overlap:     {overlap_stats['mean_overlap_bp']:.0f} bp")
    print(f"  Median Overlap:   {overlap_stats['median_overlap_bp']:.0f} bp")
    print(f"\n📐 Length Statistics:")
    print(f"  Predicted (mean/median): {length_stats['predicted_mean_length']:.0f} / {length_stats['predicted_median_length']:.0f} bp")
    print(f"  Curated (mean/median):   {length_stats['curated_mean_length']:.0f} / {length_stats['curated_median_length']:.0f} bp")

    return results


def compute_per_chromosome_metrics(predicted_df: pd.DataFrame,
                                   curated_df: pd.DataFrame,
                                   overlaps_df: pd.DataFrame) -> Dict:
    """
    Compute precision, recall, F1 per chromosome.
    """
    per_chr = {}

    all_chrs = set(predicted_df['chr'].unique()) | set(curated_df['chr'].unique())

    for chr_name in sorted(all_chrs):
        pred_chr = predicted_df[predicted_df['chr'] == chr_name]
        cur_chr = curated_df[curated_df['chr'] == chr_name]
        ov_chr = overlaps_df[overlaps_df['query_chr'] == chr_name]

        n_pred = len(pred_chr)
        n_cur = len(cur_chr)

        if len(ov_chr) > 0:
            n_tp = len(ov_chr['query_idx'].unique())
        else:
            n_tp = 0

        n_fp = n_pred - n_tp
        n_fn = n_cur - n_tp

        precision = n_tp / n_pred if n_pred > 0 else 0.0
        recall = n_tp / n_cur if n_cur > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_chr[chr_name] = {
            'n_predicted': n_pred,
            'n_curated': n_cur,
            'n_tp': n_tp,
            'n_fp': n_fp,
            'n_fn': n_fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    return per_chr


def plot_benchmark_results(results: Dict, output_dir: str):
    """
    Generate visualization plots for benchmarking results.

    Parameters
    ----------
    results : dict
        Results from benchmark_ori_predictions
    output_dir : str
        Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall metrics bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [results['precision'], results['recall'], results['f1']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Origin Prediction Benchmark: Overall Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    confusion_data = [
        [results['n_true_positives'], results['n_false_positives']],
        [results['n_false_negatives'], 0]
    ]
    labels = [['TP', 'FP'], ['FN', '']]

    im = ax.imshow([[results['n_true_positives'], results['n_false_positives']],
                    [results['n_false_negatives'], 0]],
                   cmap='Blues', alpha=0.7)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Positive', 'Predicted Negative'])
    ax.set_yticklabels(['Actually Positive', 'Actually Negative'])
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            if not (i == 1 and j == 1):
                text = ax.text(j, i, f"{labels[i][j]}\n{confusion_data[i][j]:,}",
                             ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Per-chromosome metrics
    if 'per_chr_stats' in results:
        per_chr = results['per_chr_stats']
        chrs = sorted(per_chr.keys())

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        for idx, metric in enumerate(['precision', 'recall', 'f1']):
            ax = axes[idx]
            values = [per_chr[c][metric] for c in chrs]
            colors = ['#3498db' if metric == 'precision' else '#2ecc71' if metric == 'recall' else '#e74c3c' for _ in chrs]

            ax.bar(chrs, values, color=colors[0], alpha=0.7, edgecolor='black')
            ax.set_ylim(0, 1.0)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} by Chromosome', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'per_chromosome_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Jaccard distribution
    if 'high_quality_overlaps' in results and len(results['high_quality_overlaps']) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        jaccard_values = results['high_quality_overlaps']['jaccard']

        ax.hist(jaccard_values, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.axvline(jaccard_values.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {jaccard_values.mean():.3f}')
        ax.axvline(jaccard_values.median(), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {jaccard_values.median():.3f}')

        ax.set_xlabel('Jaccard Index', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Jaccard Index for Matched Origins', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'jaccard_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Length comparison
    if 'high_quality_overlaps' in results and len(results['high_quality_overlaps']) > 0:
        overlaps = results['high_quality_overlaps']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        ax = axes[0]
        ax.scatter(overlaps['query_length'], overlaps['ref_length'],
                  alpha=0.5, s=50, color='#3498db', edgecolors='black', linewidth=0.5)
        max_val = max(overlaps['query_length'].max(), overlaps['ref_length'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
        ax.set_xlabel('Predicted Origin Length (bp)', fontsize=11)
        ax.set_ylabel('Curated Origin Length (bp)', fontsize=11)
        ax.set_title('Length Comparison: Predicted vs Curated', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Box plot comparison
        ax = axes[1]
        data = [overlaps['query_length'], overlaps['ref_length']]
        bp = ax.boxplot(data, labels=['Predicted', 'Curated'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Origin Length (bp)', fontsize=11)
        ax.set_title('Length Distribution Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'length_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Benchmark plots saved to: {output_dir}")


def save_benchmark_report(results: Dict, output_file: str):
    """
    Save detailed benchmark report to text file.

    Parameters
    ----------
    results : dict
        Results from benchmark_ori_predictions
    output_file : str
        Output file path
    """
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ORIGIN PREDICTION BENCHMARK REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Precision:        {results['precision']:.4f}\n")
        f.write(f"Recall:           {results['recall']:.4f}\n")
        f.write(f"F1 Score:         {results['f1']:.4f}\n")
        f.write(f"\n")
        f.write(f"Predicted Origins:    {results['n_predicted']:,}\n")
        f.write(f"Curated Origins:      {results['n_curated']:,}\n")
        f.write(f"True Positives:       {results['n_true_positives']:,}\n")
        f.write(f"False Positives:      {results['n_false_positives']:,}\n")
        f.write(f"False Negatives:      {results['n_false_negatives']:,}\n")
        f.write("\n")

        # Overlap stats
        if 'overlap_stats' in results:
            f.write("OVERLAP QUALITY\n")
            f.write("-" * 70 + "\n")
            stats = results['overlap_stats']
            f.write(f"Mean Jaccard:         {stats['mean_jaccard']:.4f}\n")
            f.write(f"Median Jaccard:       {stats['median_jaccard']:.4f}\n")
            f.write(f"Mean Overlap (bp):    {stats['mean_overlap_bp']:.0f}\n")
            f.write(f"Median Overlap (bp):  {stats['median_overlap_bp']:.0f}\n")
            f.write("\n")

        # Length stats
        if 'length_stats' in results:
            f.write("LENGTH STATISTICS\n")
            f.write("-" * 70 + "\n")
            stats = results['length_stats']
            f.write(f"Predicted Origins:\n")
            f.write(f"  Mean Length:    {stats['predicted_mean_length']:.0f} bp\n")
            f.write(f"  Median Length:  {stats['predicted_median_length']:.0f} bp\n")
            f.write(f"Curated Origins:\n")
            f.write(f"  Mean Length:    {stats['curated_mean_length']:.0f} bp\n")
            f.write(f"  Median Length:  {stats['curated_median_length']:.0f} bp\n")
            f.write("\n")

        # Per-chromosome stats
        if 'per_chr_stats' in results:
            f.write("PER-CHROMOSOME METRICS\n")
            f.write("-" * 70 + "\n")
            for chr_name in sorted(results['per_chr_stats'].keys()):
                stats = results['per_chr_stats'][chr_name]
                f.write(f"\n{chr_name}:\n")
                f.write(f"  Predicted: {stats['n_predicted']:,}  |  Curated: {stats['n_curated']:,}\n")
                f.write(f"  TP: {stats['n_tp']:,}  |  FP: {stats['n_fp']:,}  |  FN: {stats['n_fn']:,}\n")
                f.write(f"  Precision: {stats['precision']:.4f}  |  Recall: {stats['recall']:.4f}  |  F1: {stats['f1']:.4f}\n")

    print(f"✅ Benchmark report saved to: {output_file}")
