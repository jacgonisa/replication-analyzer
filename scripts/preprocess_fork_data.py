#!/usr/bin/env python
"""
Preprocessing script for Fork detection data.

This script SEPARATES data loading/encoding from training, allowing:
- One-time preprocessing (~20 min)
- Fast training retries (~2-3 min data loading vs 15-20 min)
- Data inspection before training
- Reproducible datasets

Usage:
    python scripts/preprocess_fork_data.py \
        --config configs/case_study_combined_forks.yaml \
        --output data/preprocessed/combined_forks_encoded.npz

"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replication_analyzer.data.loaders import load_all_xy_data, load_fork_data
from replication_analyzer.data.preprocessing import prepare_fork_data_hybrid, pad_sequences


def main():
    parser = argparse.ArgumentParser(description='Preprocess Fork detection data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for preprocessed data (.npz file)')
    parser.add_argument('--save-info', action='store_true',
                       help='Save detailed preprocessing info as JSON')

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*70)
    print(f"PREPROCESSING: {config['experiment_name']}")
    print("="*70)

    # ========== LOAD DATA ==========
    print("\n" + "="*70)
    print("STEP 1: LOADING RAW DATA")
    print("="*70)

    xy_data = load_all_xy_data(
        base_dir=config['data']['base_dir'],
        run_dirs=config['data'].get('run_dirs')
    )

    left_forks, right_forks = load_fork_data(
        config['data']['left_forks_bed'],
        config['data']['right_forks_bed']
    )

    # ========== PREPARE & ENCODE DATA ==========
    print("\n" + "="*70)
    print("STEP 2: ENCODING & BALANCING")
    print("="*70)

    X_sequences, y_sequences, info = prepare_fork_data_hybrid(
        xy_data,
        left_forks,
        right_forks,
        oversample_ratio=config['preprocessing'].get('oversample_ratio', 0.5),
        use_enhanced_encoding=config['preprocessing'].get('use_enhanced_encoding', True),
        random_seed=config['preprocessing'].get('random_seed', 42)
    )

    # ========== PAD SEQUENCES ==========
    print("\n" + "="*70)
    print("STEP 3: PADDING SEQUENCES")
    print("="*70)

    percentile = config['preprocessing'].get('percentile', 100)
    max_length_config = config['model'].get('max_length')

    X_padded, y_padded, max_length = pad_sequences(
        X_sequences,
        y_sequences,
        percentile=percentile,
        max_length=max_length_config
    )

    print(f"\n✅ Preprocessing complete!")
    print(f"   Final shape: {X_padded.shape}")
    print(f"   Max length: {max_length}")
    print(f"   Total sequences: {len(X_padded):,}")

    # ========== SAVE PREPROCESSED DATA ==========
    print("\n" + "="*70)
    print("STEP 4: SAVING PREPROCESSED DATA")
    print("="*70)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy archive
    print(f"\n💾 Saving to: {output_path}")
    np.savez_compressed(
        output_path,
        X=X_padded,
        y=y_padded,
        max_length=max_length,
        read_ids=info.index.values,
        has_fork=info['has_fork'].values,
        has_left=info.get('has_left', np.zeros(len(info))).values,
        has_right=info.get('has_right', np.zeros(len(info))).values
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved! File size: {file_size_mb:.1f} MB")

    # Save metadata/info
    if args.save_info or True:  # Always save info
        info_path = output_path.with_suffix('.json')
        metadata = {
            'experiment_name': config['experiment_name'],
            'preprocessing_date': datetime.now().isoformat(),
            'config_file': args.config,
            'data_shape': list(X_padded.shape),
            'max_length': int(max_length),
            'n_sequences': len(X_padded),
            'n_channels': X_padded.shape[2],
            'class_distribution': {
                'background': int(np.sum(y_padded == 0)),
                'left_fork': int(np.sum(y_padded == 1)),
                'right_fork': int(np.sum(y_padded == 2))
            },
            'reads_with_forks': int(info['has_fork'].sum()),
            'reads_without_forks': int((~info['has_fork']).sum()),
            'preprocessing_params': {
                'oversample_ratio': config['preprocessing'].get('oversample_ratio'),
                'percentile': percentile,
                'enhanced_encoding': config['preprocessing'].get('use_enhanced_encoding'),
                'random_seed': config['preprocessing'].get('random_seed', 42)
            }
        }

        with open(info_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"📋 Metadata saved to: {info_path}")

    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nPreprocessed data: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Sequences: {len(X_padded):,}")
    print(f"Shape: {X_padded.shape}")
    print("\nNext steps:")
    print(f"  1. Train model:")
    print(f"     python scripts/train_fork_model.py \\")
    print(f"       --preprocessed {output_path} \\")
    print(f"       --config {args.config}")
    print(f"\n  2. Or inspect data:")
    print(f"     python -c \"import numpy as np; data=np.load('{output_path}'); print(data['X'].shape)\"")


if __name__ == '__main__':
    main()
