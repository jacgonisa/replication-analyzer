#!/usr/bin/env python
"""
Generate comprehensive visualization plots matching notebook style.

Usage:
    python scripts/generate_comprehensive_plots.py \
        --history results/case_study_jan2026/combined/training_history.csv \
        --predictions results/case_study_jan2026/combined/evaluation/predictions.tsv \
        --output wiki/images/comprehensive \
        --model-name "Fork_Detector_Combined"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.visualization.comprehensive_plots import generate_comprehensive_plots


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive evaluation plots'
    )
    parser.add_argument('--history', type=str, required=True,
                       help='Path to training history CSV')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions TSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--model-name', type=str, default='Fork_Detector',
                       help='Model name for plot titles')

    args = parser.parse_args()

    # Generate plots
    figures = generate_comprehensive_plots(
        history_csv=args.history,
        predictions_tsv=args.predictions,
        model_name=args.model_name,
        save_dir=args.output
    )

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(figures)} comprehensive visualization figures")
    print(f"Saved to: {args.output}/")


if __name__ == '__main__':
    main()
