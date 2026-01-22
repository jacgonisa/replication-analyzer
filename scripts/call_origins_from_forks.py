#!/usr/bin/env python
"""
Step 2: Call origins from predicted forks

This script takes predicted fork BED files (from predict_forks.py or DNAscent),
infers origins, and optionally benchmarks against curated dataset.

Usage:
    # From AI predictions
    python scripts/call_origins_from_forks.py --config configs/ori_calling.yaml

    # From DNAscent forks
    python scripts/call_origins_from_forks.py --config configs/ori_calling.yaml \\
        --left-forks path/to/left_forks.bed \\
        --right-forks path/to/right_forks.bed
"""

import argparse
import yaml
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.evaluation.ori_caller import (
    ForkSeg, infer_events, write_bed6
)
from replication_analyzer.evaluation.benchmark import (
    benchmark_ori_predictions, plot_benchmark_results, save_benchmark_report
)


def load_fork_bed(bed_path, fork_type='L'):
    """
    Load fork BED file and convert to ForkSeg objects.

    Handles variable column formats (4, 5, or 8+ columns).
    """
    print(f"Loading {fork_type} forks from: {bed_path}")

    df = pd.read_csv(bed_path, sep='\t', header=None)

    # Take first 4 columns: chr, start, end, read_id
    # If 5th column exists and is numeric, use as gradient, else default
    df_core = df.iloc[:, :4].copy()
    df_core.columns = ['chr', 'start', 'end', 'read_id']

    if df.shape[1] >= 5:
        # Try to convert column 4 to float, if fails use default
        try:
            df_core['gradient'] = pd.to_numeric(df.iloc[:, 4], errors='coerce').fillna(
                -1.0 if fork_type == 'L' else 1.0
            )
        except:
            # Default gradient: -1 for left, +1 for right
            df_core['gradient'] = -1.0 if fork_type == 'L' else 1.0
    else:
        # Default gradient: -1 for left, +1 for right
        df_core['gradient'] = -1.0 if fork_type == 'L' else 1.0

    print(f"  Loaded {len(df_core):,} {fork_type} forks")

    # Convert to ForkSeg objects
    fork_segs = [
        ForkSeg(
            chrom=row['chr'],
            start=int(row['start']),
            end=int(row['end']),
            read_id=row['read_id'],
            grad=float(row['gradient']),
            kind=fork_type
        )
        for _, row in df_core.iterrows()
    ]

    return fork_segs


def main():
    parser = argparse.ArgumentParser(
        description='Call origins from predicted forks'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--left-forks', type=str,
                       help='Override left forks BED file from config')
    parser.add_argument('--right-forks', type=str,
                       help='Override right forks BED file from config')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory from config')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip benchmarking step')

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"ORIGIN CALLING: {config['experiment_name']}")
    print("=" * 70)

    # Determine fork BED files
    if args.left_forks and args.right_forks:
        left_bed = args.left_forks
        right_bed = args.right_forks
    else:
        left_bed = config['forks']['left_forks_bed']
        right_bed = config['forks']['right_forks_bed']

    # Load forks
    print("\nLoading fork predictions...")
    left_segs = load_fork_bed(left_bed, fork_type='L')
    right_segs = load_fork_bed(right_bed, fork_type='R')

    # Call origins
    print("\n" + "=" * 70)
    print("CALLING ORIGINS FROM FORK PATTERNS")
    print("=" * 70)

    min_len = config['ori_calling'].get('min_length', 0)
    print(f"\nInferring origins (min_length={min_len}bp)...")

    origins, terminations, stats = infer_events(left_segs, right_segs, min_len=min_len)

    print(f"\n✅ Origins called: {len(origins):,}")
    print(f"✅ Terminations called: {len(terminations):,}")
    print(f"\nStatistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:,}")

    # Save origins and terminations
    origins_bed = output_dir / 'predicted_origins.bed'
    terminations_bed = output_dir / 'predicted_terminations.bed'

    write_bed6(str(origins_bed), origins)
    write_bed6(str(terminations_bed), terminations)

    print(f"\n✅ Origins and terminations saved:")
    print(f"  Origins:       {origins_bed}")
    print(f"  Terminations:  {terminations_bed}")

    # Benchmark (optional)
    if not args.skip_benchmark and 'curated_ori_bed' in config.get('data', {}):
        print("\n" + "=" * 70)
        print("BENCHMARKING AGAINST CURATED DATASET")
        print("=" * 70)

        curated_bed = config['data']['curated_ori_bed']

        print(f"\nComparing predictions to: {curated_bed}")
        results = benchmark_ori_predictions(
            predicted_bed=str(origins_bed),
            curated_bed=curated_bed,
            min_overlap=config['benchmark'].get('min_overlap', 1),
            jaccard_threshold=config['benchmark'].get('jaccard_threshold', 0.0)
        )

        # Generate plots
        print("\nGenerating benchmark plots...")
        plot_benchmark_results(results, output_dir / 'benchmark_plots')

        # Save detailed report
        print("Saving benchmark report...")
        save_benchmark_report(results, output_dir / 'benchmark_report.txt')

        # Save overlap details
        if len(results['high_quality_overlaps']) > 0:
            overlaps_file = output_dir / 'origin_overlaps.tsv'
            results['high_quality_overlaps'].to_csv(overlaps_file, sep='\t', index=False)
            print(f"✅ Overlap details saved to: {overlaps_file}")

        print(f"\n✅ Benchmark results saved to: {output_dir}")

    # Final summary
    print("\n" + "=" * 70)
    print("✅ ORIGIN CALLING COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nOutput files:")
    print(f"  - predicted_origins.bed ({len(origins):,} origins)")
    print(f"  - predicted_terminations.bed ({len(terminations):,} terminations)")
    if not args.skip_benchmark:
        print(f"  - benchmark_report.txt")
        print(f"  - benchmark_plots/")
        print(f"  - origin_overlaps.tsv")

    print("\n" + "=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
