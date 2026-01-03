#!/usr/bin/env python
"""
Executable script for evaluating trained models.

Usage:
    python scripts/evaluate_model.py --model models/ori_expert_model.keras --type ori --config configs/ori_model_default.yaml
    python scripts/evaluate_model.py --model models/fork_detector.keras --type fork --config configs/fork_model_default.yaml

"""

import argparse
import yaml
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.data.loaders import load_all_xy_data, load_curated_origins, load_fork_data, load_genomic_regions
from replication_analyzer.training.train_ori import load_trained_ori_model
from replication_analyzer.training.train_fork import load_trained_fork_model
from replication_analyzer.evaluation.predictors import predict_on_all_reads
from replication_analyzer.evaluation.metrics import calculate_binary_metrics, calculate_multiclass_metrics, print_metrics
from replication_analyzer.evaluation.regional import assign_genomic_regions, calculate_regional_metrics, compare_regional_performance
from replication_analyzer.visualization.evaluation_plots import plot_comprehensive_evaluation, plot_regional_comparison


def evaluate_ori_model(model, config, output_dir):
    """Evaluate ORI model"""
    print("\n" + "="*70)
    print("EVALUATING ORI MODEL")
    print("="*70)

    # Load data
    xy_data = load_all_xy_data(
        base_dir=config['data']['base_dir'],
        run_dirs=config['data'].get('run_dirs')
    )

    ori_annotations = load_curated_origins(config['data']['ori_bed'])

    # Get max_length from model input shape
    max_length = model.input_shape[1]
    print(f"\nModel max_length: {max_length}")

    # Predict on all reads with ORIs
    read_ids = ori_annotations['read_id'].unique()
    print(f"\nPredicting on {len(read_ids)} reads...")

    predictions_df = predict_on_all_reads(
        model, xy_data, max_length,
        read_ids=read_ids,
        use_enhanced_encoding=config['preprocessing'].get('use_enhanced_encoding', True)
    )

    # Create ground truth labels
    print("\nCreating ground truth labels...")
    predictions_df['y_true'] = 0

    for read_id, read_df_group in predictions_df.groupby('read_id'):
        ori_df = ori_annotations[ori_annotations['read_id'] == read_id]
        for idx, row in read_df_group.iterrows():
            for _, ori in ori_df.iterrows():
                overlap = max(0, min(row['end'], ori['end']) - max(row['start'], ori['start']))
                if overlap > 0:
                    predictions_df.at[idx, 'y_true'] = 1
                    break

    # Calculate metrics
    y_true = predictions_df['y_true'].values
    y_pred_proba = predictions_df['ori_prob'].values
    y_pred = predictions_df['is_ori'].values

    metrics = calculate_binary_metrics(y_true, y_pred, y_pred_proba)
    print_metrics(metrics, title="ORI Model Performance")

    # Save metrics
    import pandas as pd
    metrics_file = output_dir / 'overall_metrics.csv'
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"✅ Metrics saved: {metrics_file}")

    # Plot comprehensive evaluation
    plot_comprehensive_evaluation(
        y_true, y_pred, y_pred_proba,
        class_names=['Non-ORI', 'ORI'],
        save_dir=str(output_dir / 'plots')
    )

    # Regional analysis (if regions provided)
    if 'centromere_bed' in config['data'] and 'pericentromere_bed' in config['data']:
        print("\n" + "="*70)
        print("REGIONAL ANALYSIS")
        print("="*70)

        regions_dict = load_genomic_regions(
            config['data']['centromere_bed'],
            config['data']['pericentromere_bed']
        )

        predictions_df = assign_genomic_regions(predictions_df, regions_dict)

        regional_metrics = calculate_regional_metrics(
            predictions_df,
            y_true_col='y_true',
            y_pred_col='is_ori',
            y_proba_col='ori_prob'
        )

        # Save regional metrics
        regional_file = output_dir / 'regional_metrics.csv'
        regional_metrics.to_csv(regional_file, index=False)
        print(f"\n✅ Regional metrics saved: {regional_file}")

        # Plot regional comparison
        plot_regional_comparison(
            regional_metrics,
            save_path=output_dir / 'plots' / 'regional_comparison.png'
        )

        # Save comparison table
        compare_regional_performance(
            regional_metrics,
            output_file=output_dir / 'regional_comparison.csv'
        )

    return predictions_df, metrics


def evaluate_fork_model(model, config, output_dir):
    """Evaluate Fork model"""
    print("\n" + "="*70)
    print("EVALUATING FORK MODEL")
    print("="*70)

    # Load data
    xy_data = load_all_xy_data(
        base_dir=config['data']['base_dir'],
        run_dirs=config['data'].get('run_dirs')
    )

    left_forks, right_forks = load_fork_data(
        config['data']['left_forks_bed'],
        config['data']['right_forks_bed']
    )

    # Get max_length from model
    max_length = model.input_shape[1]
    print(f"\nModel max_length: {max_length}")

    # Get reads with forks
    fork_read_ids = set(left_forks['read_id'].unique()) | set(right_forks['read_id'].unique())
    print(f"\nPredicting on {len(fork_read_ids)} reads with forks...")

    predictions_df = predict_on_all_reads(
        model, xy_data, max_length,
        read_ids=list(fork_read_ids),
        use_enhanced_encoding=config['preprocessing'].get('use_enhanced_encoding', True)
    )

    # Create ground truth labels
    print("\nCreating ground truth labels...")
    predictions_df['y_true'] = 0  # Background

    for read_id, read_df_group in predictions_df.groupby('read_id'):
        # Label left forks
        left_forks_read = left_forks[left_forks['read_id'] == read_id]
        for idx, row in read_df_group.iterrows():
            for _, fork in left_forks_read.iterrows():
                overlap = max(0, min(row['end'], fork['end']) - max(row['start'], fork['start']))
                if overlap > 0:
                    predictions_df.at[idx, 'y_true'] = 1  # Left fork
                    break

        # Label right forks (only if not already left fork)
        right_forks_read = right_forks[right_forks['read_id'] == read_id]
        for idx, row in read_df_group.iterrows():
            if predictions_df.at[idx, 'y_true'] == 0:  # Only if background
                for _, fork in right_forks_read.iterrows():
                    overlap = max(0, min(row['end'], fork['end']) - max(row['start'], fork['start']))
                    if overlap > 0:
                        predictions_df.at[idx, 'y_true'] = 2  # Right fork
                        break

    # Calculate metrics
    y_true = predictions_df['y_true'].values
    y_pred = predictions_df['predicted_class'].values
    y_pred_proba = predictions_df[['class_0_prob', 'class_1_prob', 'class_2_prob']].values

    metrics = calculate_multiclass_metrics(
        y_true, y_pred, y_pred_proba,
        class_names=['Background', 'Left Fork', 'Right Fork']
    )
    print_metrics(metrics, title="Fork Model Performance")

    # Save metrics
    import pandas as pd
    metrics_file = output_dir / 'overall_metrics.csv'
    pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'kappa': metrics['kappa'],
        'mcc': metrics['mcc'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro']
    }]).to_csv(metrics_file, index=False)
    print(f"✅ Metrics saved: {metrics_file}")

    # Plot evaluation
    plot_comprehensive_evaluation(
        y_true, y_pred, None,
        class_names=['Background', 'Left Fork', 'Right Fork'],
        save_dir=str(output_dir / 'plots')
    )

    return predictions_df, metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.keras)')
    parser.add_argument('--type', type=str, required=True, choices=['ori', 'fork'],
                       help='Model type: ori or fork')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results/<model_name>_eval)')

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set output directory
    if args.output is None:
        model_name = Path(args.model).stem
        output_dir = Path(f"results/{model_name}_eval")
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    # Load model and evaluate
    if args.type == 'ori':
        model = load_trained_ori_model(args.model)
        predictions_df, metrics = evaluate_ori_model(model, config, output_dir)
    else:
        model = load_trained_fork_model(args.model)
        predictions_df, metrics = evaluate_fork_model(model, config, output_dir)

    # Save predictions
    predictions_file = output_dir / 'predictions.tsv'
    predictions_df.to_csv(predictions_file, sep='\t', index=False)
    print(f"\n✅ Predictions saved: {predictions_file}")

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics: {output_dir}/overall_metrics.csv")
    print(f"  - Predictions: {predictions_file}")
    print(f"  - Plots: {output_dir}/plots/")


if __name__ == '__main__':
    main()
