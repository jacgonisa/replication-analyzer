#!/usr/bin/env python
"""
Executable script for training ORI detection model.

Usage:
    python scripts/train_ori_model.py --config configs/ori_model_default.yaml

"""

import argparse
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.training.train_ori import train_ori_model
from replication_analyzer.visualization.training_plots import plot_training_history


def main():
    parser = argparse.ArgumentParser(description='Train ORI detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate training plots after training')

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*70)
    print(f"TRAINING: {config['experiment_name']}")
    print("="*70)

    # Train model
    model, history, max_length, info = train_ori_model(config)

    # Generate plots if requested
    if args.plot:
        print("\n" + "="*70)
        print("GENERATING TRAINING PLOTS")
        print("="*70)

        plots_dir = Path(config['output']['results_dir']) / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_training_history(
            history,
            save_path=plots_dir / 'training_history.png',
            model_name=config['experiment_name']
        )

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {config['output']['model_dir']}/{config['output']['model_filename']}")
    print(f"Results saved to: {config['output']['results_dir']}")

    if args.plot:
        print(f"Plots saved to: {plots_dir}")

    print("\nNext steps:")
    print(f"  1. Evaluate: python scripts/evaluate_model.py --model {config['output']['model_dir']}/{config['output']['model_filename']} --type ori")
    print(f"  2. Predict: python scripts/annotate_new_data.py --model {config['output']['model_dir']}/{config['output']['model_filename']} --type ori")


if __name__ == '__main__':
    main()
