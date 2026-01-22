"""
Utilities for working with BED files and comparing genomic intervals.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def read_bed_file(bed_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read a BED file into a pandas DataFrame.

    Parameters
    ----------
    bed_path : str
        Path to BED file
    columns : list, optional
        Column names to use. If None, uses standard BED column names

    Returns
    -------
    pd.DataFrame
        BED file contents
    """
    if columns is None:
        # Try to infer from first line
        with open(bed_path, 'r') as f:
            first_line = f.readline().strip()
            n_cols = len(first_line.split('\t'))

        # Standard BED column names
        bed_cols = ['chr', 'start', 'end', 'name', 'score', 'strand',
                    'thickStart', 'thickEnd', 'itemRgb', 'blockCount',
                    'blockSizes', 'blockStarts']
        columns = bed_cols[:n_cols]

    df = pd.read_csv(bed_path, sep='\t', header=None, names=columns, comment='#')

    # Convert coordinates to int
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)

    return df


def write_bed_file(df: pd.DataFrame, output_path: str, columns: Optional[List[str]] = None):
    """
    Write a DataFrame to BED format.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write
    output_path : str
        Output file path
    columns : list, optional
        Columns to write (default: all columns)
    """
    if columns is None:
        columns = df.columns.tolist()

    df[columns].to_csv(output_path, sep='\t', header=False, index=False)
    print(f"✅ Wrote {len(df)} intervals to {output_path}")


def compute_overlap(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> int:
    """
    Compute overlap length between two intervals.

    Parameters
    ----------
    interval1 : tuple
        (start, end) of first interval
    interval2 : tuple
        (start, end) of second interval

    Returns
    -------
    int
        Overlap length in bp (0 if no overlap)
    """
    start1, end1 = interval1
    start2, end2 = interval2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    return max(0, overlap_end - overlap_start)


def compute_jaccard(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> float:
    """
    Compute Jaccard index between two intervals.

    Parameters
    ----------
    interval1 : tuple
        (start, end) of first interval
    interval2 : tuple
        (start, end) of second interval

    Returns
    -------
    float
        Jaccard index (intersection / union)
    """
    start1, end1 = interval1
    start2, end2 = interval2

    overlap = compute_overlap(interval1, interval2)
    union = (end1 - start1) + (end2 - start2) - overlap

    if union == 0:
        return 0.0

    return overlap / union


def find_overlapping_intervals(query_df: pd.DataFrame,
                               reference_df: pd.DataFrame,
                               min_overlap: int = 1,
                               same_chr: bool = True) -> pd.DataFrame:
    """
    Find overlapping intervals between query and reference datasets.

    Parameters
    ----------
    query_df : pd.DataFrame
        Query intervals (must have: chr, start, end)
    reference_df : pd.DataFrame
        Reference intervals (must have: chr, start, end)
    min_overlap : int
        Minimum overlap in bp to consider a match
    same_chr : bool
        Only match intervals on same chromosome

    Returns
    -------
    pd.DataFrame
        Overlapping pairs with columns: query_idx, ref_idx, overlap_bp, jaccard
    """
    matches = []

    for query_idx, query_row in query_df.iterrows():
        query_chr = query_row['chr']
        query_start = query_row['start']
        query_end = query_row['end']

        # Filter reference by chromosome if needed
        if same_chr:
            ref_subset = reference_df[reference_df['chr'] == query_chr]
        else:
            ref_subset = reference_df

        # Find overlaps
        for ref_idx, ref_row in ref_subset.iterrows():
            ref_start = ref_row['start']
            ref_end = ref_row['end']

            overlap = compute_overlap((query_start, query_end), (ref_start, ref_end))

            if overlap >= min_overlap:
                jaccard = compute_jaccard((query_start, query_end), (ref_start, ref_end))

                matches.append({
                    'query_idx': query_idx,
                    'ref_idx': ref_idx,
                    'query_chr': query_chr,
                    'query_start': query_start,
                    'query_end': query_end,
                    'ref_start': ref_start,
                    'ref_end': ref_end,
                    'overlap_bp': overlap,
                    'jaccard': jaccard,
                    'query_length': query_end - query_start,
                    'ref_length': ref_end - ref_start
                })

    if len(matches) == 0:
        return pd.DataFrame()

    return pd.DataFrame(matches)


def merge_overlapping_intervals(df: pd.DataFrame,
                                max_gap: int = 0) -> pd.DataFrame:
    """
    Merge overlapping or nearby intervals within the same chromosome.

    Parameters
    ----------
    df : pd.DataFrame
        Intervals to merge (must have: chr, start, end)
    max_gap : int
        Maximum gap between intervals to merge (default: 0 = only overlapping)

    Returns
    -------
    pd.DataFrame
        Merged intervals
    """
    merged = []

    for chr_name, chr_df in df.groupby('chr'):
        # Sort by start position
        chr_df = chr_df.sort_values('start').reset_index(drop=True)

        if len(chr_df) == 0:
            continue

        # Initialize first interval
        current_start = chr_df.iloc[0]['start']
        current_end = chr_df.iloc[0]['end']

        for idx in range(1, len(chr_df)):
            next_start = chr_df.iloc[idx]['start']
            next_end = chr_df.iloc[idx]['end']

            # Check if intervals should be merged
            if next_start <= current_end + max_gap:
                # Merge: extend current interval
                current_end = max(current_end, next_end)
            else:
                # No merge: save current and start new
                merged.append({
                    'chr': chr_name,
                    'start': current_start,
                    'end': current_end,
                    'length': current_end - current_start
                })
                current_start = next_start
                current_end = next_end

        # Add last interval
        merged.append({
            'chr': chr_name,
            'start': current_start,
            'end': current_end,
            'length': current_end - current_start
        })

    return pd.DataFrame(merged)


def compute_coverage_stats(intervals_df: pd.DataFrame,
                           genome_size: Optional[Dict[str, int]] = None) -> Dict:
    """
    Compute coverage statistics for a set of intervals.

    Parameters
    ----------
    intervals_df : pd.DataFrame
        Intervals (must have: chr, start, end)
    genome_size : dict, optional
        Chromosome sizes {chr: size}

    Returns
    -------
    dict
        Coverage statistics
    """
    stats = {
        'n_intervals': len(intervals_df),
        'total_bp': (intervals_df['end'] - intervals_df['start']).sum(),
        'mean_length': (intervals_df['end'] - intervals_df['start']).mean(),
        'median_length': (intervals_df['end'] - intervals_df['start']).median(),
        'min_length': (intervals_df['end'] - intervals_df['start']).min(),
        'max_length': (intervals_df['end'] - intervals_df['start']).max(),
    }

    # Per-chromosome stats
    stats['per_chr'] = {}
    for chr_name, chr_df in intervals_df.groupby('chr'):
        chr_stats = {
            'n_intervals': len(chr_df),
            'total_bp': (chr_df['end'] - chr_df['start']).sum(),
        }

        if genome_size and chr_name in genome_size:
            chr_stats['coverage_fraction'] = chr_stats['total_bp'] / genome_size[chr_name]

        stats['per_chr'][chr_name] = chr_stats

    return stats


def filter_by_read_support(bed_df: pd.DataFrame,
                           min_reads: int = 2) -> pd.DataFrame:
    """
    Filter genomic regions by read support (collapse overlapping intervals from different reads).

    Parameters
    ----------
    bed_df : pd.DataFrame
        BED file with read_id/name column
    min_reads : int
        Minimum number of reads supporting a region

    Returns
    -------
    pd.DataFrame
        Filtered regions with read support counts
    """
    # Group by chromosome and find overlapping clusters
    supported_regions = []

    for chr_name, chr_df in bed_df.groupby('chr'):
        chr_df = chr_df.sort_values('start').reset_index(drop=True)

        if len(chr_df) == 0:
            continue

        # Cluster overlapping intervals
        clusters = []
        current_cluster = [chr_df.iloc[0]]

        for idx in range(1, len(chr_df)):
            current_interval = chr_df.iloc[idx]
            cluster_end = max([x['end'] for x in current_cluster])

            if current_interval['start'] <= cluster_end:
                # Overlaps with cluster
                current_cluster.append(current_interval)
            else:
                # No overlap: save cluster and start new
                clusters.append(current_cluster)
                current_cluster = [current_interval]

        # Add last cluster
        clusters.append(current_cluster)

        # Process clusters
        for cluster in clusters:
            if len(cluster) >= min_reads:
                # Get unique read IDs
                read_col = 'read_id' if 'read_id' in chr_df.columns else 'name'
                unique_reads = set([x[read_col] for x in cluster])

                if len(unique_reads) >= min_reads:
                    cluster_start = min([x['start'] for x in cluster])
                    cluster_end = max([x['end'] for x in cluster])

                    supported_regions.append({
                        'chr': chr_name,
                        'start': cluster_start,
                        'end': cluster_end,
                        'n_reads': len(unique_reads),
                        'n_intervals': len(cluster),
                        'length': cluster_end - cluster_start
                    })

    return pd.DataFrame(supported_regions)
