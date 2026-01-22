"""
Evaluation module for model assessment.
"""

from .metrics import (
    calculate_binary_metrics,
    calculate_multiclass_metrics,
    print_metrics,
    find_optimal_threshold
)

from .predictors import (
    predict_on_read,
    predict_on_all_reads,
    call_peaks_from_predictions,
    export_peaks_to_bed
)

from .regional import (
    assign_genomic_regions,
    calculate_regional_metrics,
    compare_regional_performance,
    analyze_region_specific_errors
)

from .bed_utils import (
    read_bed_file,
    write_bed_file,
    compute_overlap,
    compute_jaccard,
    find_overlapping_intervals,
    merge_overlapping_intervals,
    compute_coverage_stats,
    filter_by_read_support
)

from .benchmark import (
    benchmark_ori_predictions,
    plot_benchmark_results,
    save_benchmark_report
)

__all__ = [
    # Metrics
    'calculate_binary_metrics',
    'calculate_multiclass_metrics',
    'print_metrics',
    'find_optimal_threshold',
    # Predictors
    'predict_on_read',
    'predict_on_all_reads',
    'call_peaks_from_predictions',
    'export_peaks_to_bed',
    # Regional
    'assign_genomic_regions',
    'calculate_regional_metrics',
    'compare_regional_performance',
    'analyze_region_specific_errors',
    # BED utilities
    'read_bed_file',
    'write_bed_file',
    'compute_overlap',
    'compute_jaccard',
    'find_overlapping_intervals',
    'merge_overlapping_intervals',
    'compute_coverage_stats',
    'filter_by_read_support',
    # Benchmarking
    'benchmark_ori_predictions',
    'plot_benchmark_results',
    'save_benchmark_report',
]
