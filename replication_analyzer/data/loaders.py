"""
Data loading utilities for replication fork and ORI analysis.

This module handles loading:
- XY plot data (BrdU/EdU signal)
- Curated ORI annotations (BED format)
- Fork annotations (left and right, BED format)
- Genomic region annotations (centromere, pericentromere)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


def load_curated_origins(bed_file: str) -> pd.DataFrame:
    """
    Load curated origins from BED file.

    Parameters
    ----------
    bed_file : str
        Path to BED file containing curated origins

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chr, start, end, read_id, score, strand,
        chr2, ref_start, ref_end, sample, center, length
    """
    origins = pd.read_csv(
        bed_file, sep='\t', header=None,
        names=['chr', 'start', 'end', 'read_id', 'score', 'strand',
               'chr2', 'ref_start', 'ref_end', 'sample']
    )

    origins['center'] = (origins['start'] + origins['end']) / 2
    origins['length'] = origins['end'] - origins['start']

    print(f"✓ Loaded {len(origins)} curated origins")
    print(f"✓ Unique reads: {origins['read_id'].nunique()}")
    print(f"✓ Chromosomes: {list(origins['chr'].unique())}")

    return origins


def load_xy_data_single(plot_data_file: str) -> pd.DataFrame:
    """
    Load BrdU/EdU signal data for a single read.

    Parameters
    ----------
    plot_data_file : str
        Path to plot data file (TSV format: chr, start, end, signal)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chr, start, end, signal, read_id, center, length
    """
    data = pd.read_csv(
        plot_data_file, sep='\t', header=None,
        names=['chr', 'start', 'end', 'signal']
    )

    read_id = Path(plot_data_file).stem.replace('plot_data_', '')
    data['read_id'] = read_id
    data['center'] = (data['start'] + data['end']) / 2
    data['length'] = data['end'] - data['start']

    return data


def load_all_xy_data(base_dir: str, run_dirs: Optional[list] = None) -> pd.DataFrame:
    """
    Load XY plot data from multiple run directories.

    Parameters
    ----------
    base_dir : str
        Base directory containing run subdirectories
    run_dirs : list, optional
        List of subdirectory names (e.g., ['NM30_plot_data_1strun_xy', 'NM30_plot_data_2ndrun_xy'])
        If None, searches for all directories matching '*_xy'

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all XY data, includes 'run' column
    """
    base_path = Path(base_dir)

    if run_dirs is None:
        # Auto-detect run directories
        run_dirs = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.endswith('_xy')]

    all_data = []

    for run_idx, run_dir in enumerate(run_dirs):
        dir_path = base_path / run_dir

        if not dir_path.exists():
            print(f"⚠️  Directory not found: {dir_path}")
            continue

        print(f"Loading from: {dir_path}")
        plot_files = list(dir_path.glob("plot_data_*.txt"))
        print(f"Found {len(plot_files)} files in {run_dir}")

        for i, file in enumerate(plot_files):
            if i % 100 == 0 and i > 0:
                print(f"  Loading file {i}/{len(plot_files)}...", end='\r')
            try:
                data = load_xy_data_single(file)
                data['run'] = run_dir  # Track which run this came from
                all_data.append(data)
            except Exception as e:
                print(f"\n  Error loading {file}: {e}")
        print()  # New line after progress

    if len(all_data) == 0:
        raise ValueError("No data loaded! Check your base_dir and run_dirs.")

    combined = pd.concat(all_data, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"✓ Loaded {len(combined):,} data points")
    print(f"✓ From {combined['read_id'].nunique():,} unique reads")
    for run_dir in run_dirs:
        n_reads = combined[combined['run'] == run_dir]['read_id'].nunique()
        print(f"✓ {run_dir}: {n_reads:,} reads")
    print(f"{'='*60}")

    return combined


def load_fork_data(left_forks_bed: str, right_forks_bed: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load left and right fork annotations.

    Parameters
    ----------
    left_forks_bed : str
        Path to left forks BED file
    right_forks_bed : str
        Path to right forks BED file

    Returns
    -------
    tuple of pd.DataFrame
        (left_forks, right_forks) with columns: chr, start, end, read_id,
        read_chr, read_start, read_end, direction
    """
    print("\n📂 Loading fork annotations...")

    # Left forks
    left_forks = pd.read_csv(left_forks_bed, sep='\t', header=None)
    # Handle both 7-column (no direction) and 8-column (with direction) formats
    if left_forks.shape[1] == 7:
        left_forks.columns = ['chr', 'start', 'end', 'read_id', 'read_chr',
                              'read_start', 'read_end']
    elif left_forks.shape[1] == 8:
        left_forks.columns = ['chr', 'start', 'end', 'read_id', 'read_chr',
                              'read_start', 'read_end', 'direction']
    else:
        raise ValueError(f"Expected 7 or 8 columns in left_forks_bed, got {left_forks.shape[1]}")
    print(f"  Left forks: {len(left_forks):,} regions from {left_forks['read_id'].nunique()} reads")

    # Right forks
    right_forks = pd.read_csv(right_forks_bed, sep='\t', header=None)
    # Handle both 7-column (no direction) and 8-column (with direction) formats
    if right_forks.shape[1] == 7:
        right_forks.columns = ['chr', 'start', 'end', 'read_id', 'read_chr',
                               'read_start', 'read_end']
    elif right_forks.shape[1] == 8:
        right_forks.columns = ['chr', 'start', 'end', 'read_id', 'read_chr',
                               'read_start', 'read_end', 'direction']
    else:
        raise ValueError(f"Expected 7 or 8 columns in right_forks_bed, got {right_forks.shape[1]}")
    print(f"  Right forks: {len(right_forks):,} regions from {right_forks['read_id'].nunique()} reads")

    return left_forks, right_forks


def load_genomic_regions(centromere_bed: str,
                         pericentromere_bed: str) -> Dict[str, pd.DataFrame]:
    """
    Load genomic region annotations (centromere, pericentromere).

    Parameters
    ----------
    centromere_bed : str
        Path to centromere BED file
    pericentromere_bed : str
        Path to pericentromere BED file

    Returns
    -------
    dict
        Dictionary with keys 'centromere' and 'pericentromere'
    """
    print("\n📂 Loading genomic region annotations...")

    # Centromeres
    centromeres = pd.read_csv(centromere_bed, sep='\t', header=None,
                              names=['chr', 'start', 'end', 'region_type'])
    print(f"  Loaded {len(centromeres)} centromere regions")

    # Pericentromeres
    pericentromeres = pd.read_csv(pericentromere_bed, sep='\t', header=None,
                                  names=['chr', 'start', 'end', 'region_type'])
    print(f"  Loaded {len(pericentromeres)} pericentromere regions")

    return {
        'centromere': centromeres,
        'pericentromere': pericentromeres
    }


def find_reads_with_annotations(xy_data: pd.DataFrame,
                                annotations: pd.DataFrame,
                                min_signal: float = 0.1,
                                min_max_signal: float = 0.3,
                                min_nonzero_frac: float = 0.2) -> pd.DataFrame:
    """
    Find reads that have annotations (ORIs or forks) and good signal quality.

    Parameters
    ----------
    xy_data : pd.DataFrame
        XY signal data
    annotations : pd.DataFrame
        Annotation data (ORIs or forks) with 'read_id' column
    min_signal : float
        Minimum mean signal
    min_max_signal : float
        Minimum max signal
    min_nonzero_frac : float
        Minimum fraction of non-zero signal points

    Returns
    -------
    pd.DataFrame
        Summary of good reads with statistics
    """
    reads_with_annotations = annotations['read_id'].unique()
    print(f"Reads with annotations: {len(reads_with_annotations)}")

    available_reads = xy_data['read_id'].unique()
    reads_available = [r for r in reads_with_annotations if r in available_reads]
    print(f"Available in XY data: {len(reads_available)}")

    good_reads = []
    for read_id in reads_available:
        read_data = xy_data[xy_data['read_id'] == read_id]

        signal = read_data['signal'].values
        mean_signal = np.mean(signal)
        max_signal = np.max(signal)
        nonzero_frac = (signal > 0).sum() / len(signal)

        if mean_signal > min_signal and max_signal > min_max_signal and nonzero_frac > min_nonzero_frac:
            good_reads.append({
                'read_id': read_id,
                'n_points': len(read_data),
                'mean_signal': mean_signal,
                'max_signal': max_signal,
                'std_signal': np.std(signal),
                'signal_range': max_signal - np.min(signal),
                'nonzero_frac': nonzero_frac,
                'n_annotations': len(annotations[annotations['read_id'] == read_id]),
                'chr': read_data['chr'].iloc[0],
                'run': read_data['run'].iloc[0] if 'run' in read_data.columns else 'unknown'
            })

    good_reads_df = pd.DataFrame(good_reads)

    if len(good_reads_df) > 0:
        print(f"\n✓ Good reads with signal and annotations: {len(good_reads_df)}")
    else:
        print("\n⚠️ No reads found with both annotations and good signal!")

    return good_reads_df
