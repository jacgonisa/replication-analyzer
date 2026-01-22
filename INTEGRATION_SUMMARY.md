# Origin Calling Integration - Summary

## Overview

Successfully integrated origin calling and benchmarking functionality into the replication-analyzer repository. The system now provides a complete end-to-end pipeline: **AI Fork Prediction → Origin Calling → Benchmarking**.

## What Was Added

### 1. Core Modules

#### `replication_analyzer/evaluation/bed_utils.py`
Utilities for working with BED files:
- `read_bed_file()` - Parse BED files
- `write_bed_file()` - Write BED files
- `compute_overlap()` - Calculate overlap between intervals
- `compute_jaccard()` - Jaccard index calculation
- `find_overlapping_intervals()` - Match intervals between datasets
- `merge_overlapping_intervals()` - Collapse nearby intervals
- `compute_coverage_stats()` - Coverage statistics
- `filter_by_read_support()` - Filter by read support

#### `replication_analyzer/evaluation/benchmark.py`
Benchmarking predicted origins against curated datasets:
- `benchmark_ori_predictions()` - Complete benchmarking with metrics
- `plot_benchmark_results()` - Generate visualization plots
- `save_benchmark_report()` - Export text report
- Calculates: Precision, Recall, F1, per-chromosome metrics
- Generates: 5 comprehensive plots

#### `replication_analyzer/evaluation/ori_caller.py`
Core origin calling logic (moved from root):
- Infers origins from left/right fork patterns
- Handles overlap, gap, and containment cases
- Filters by minimum length
- Outputs BED6 format

### 2. End-to-End Pipeline Script

#### `scripts/predict_forks_and_call_oris.py`
Complete automated pipeline:
1. **Fork Prediction**: Uses trained AI model on XY data
2. **Origin Calling**: Infers origins from fork pairs
3. **Benchmarking**: Compares against curated dataset
4. **Visualization**: Generates comprehensive plots

**Features:**
- Config-based execution
- Optional steps: `--skip-prediction`, `--skip-benchmark`
- Detailed logging and progress reporting
- Exports multiple BED files and reports

### 3. Configuration

#### `configs/ori_calling_pipeline.yaml`
Complete configuration template with:
- Data paths (XY data, curated origins)
- Model parameters (path, max_length, encoding)
- Prediction thresholds (fork_threshold)
- Origin calling parameters (min_length)
- Benchmarking settings (min_overlap, jaccard_threshold)
- Output configuration

### 4. Documentation

#### `README.md` (Updated)
Added "Origin Calling Pipeline" section with:
- Quick start guide
- Output file descriptions
- Advanced options
- Parameter tuning guidance

#### `docs/ORI_CALLING_GUIDE.md` (New)
Comprehensive 200+ line guide covering:
- Prerequisites and setup
- Configuration details
- Output interpretation
- Parameter tuning strategies
- Troubleshooting common issues
- Best practices
- Code examples for direct module usage

## File Structure

```
replication-analyzer/
├── replication_analyzer/
│   └── evaluation/
│       ├── bed_utils.py          # NEW: BED file utilities
│       ├── benchmark.py          # NEW: Benchmarking module
│       ├── ori_caller.py         # MOVED: Origin calling logic
│       └── __init__.py           # UPDATED: Exports new functions
├── scripts/
│   └── predict_forks_and_call_oris.py  # NEW: End-to-end pipeline
├── configs/
│   └── ori_calling_pipeline.yaml       # NEW: Pipeline configuration
├── docs/
│   └── ORI_CALLING_GUIDE.md           # NEW: Comprehensive guide
└── README.md                           # UPDATED: Added ori calling section
```

## Usage Example

### Basic Usage
```bash
# 1. Configure
vim configs/ori_calling_pipeline.yaml

# 2. Run complete pipeline
python scripts/predict_forks_and_call_oris.py \
    --config configs/ori_calling_pipeline.yaml

# 3. Check results
ls results/ori_calling_pipeline/
```

### Python API
```python
from replication_analyzer.evaluation import (
    benchmark_ori_predictions,
    plot_benchmark_results
)

# Benchmark predicted vs curated origins
results = benchmark_ori_predictions(
    predicted_bed="predicted_origins.bed",
    curated_bed="curated_origins.bed",
    jaccard_threshold=0.1
)

# Generate plots
plot_benchmark_results(results, "output_dir")
```

## Key Features

### 1. Flexible Workflow
- Can run complete pipeline or individual steps
- Use existing fork predictions or generate new ones
- Optional benchmarking step

### 2. Comprehensive Benchmarking
**Metrics:**
- Precision, Recall, F1 Score
- Per-chromosome breakdown
- Overlap quality (Jaccard index)
- Length statistics

**Visualizations:**
- Overall metrics bar chart
- Confusion matrix
- Per-chromosome metrics
- Jaccard distribution
- Length comparison (predicted vs curated)

### 3. Configurable Parameters
- **fork_threshold**: Control fork calling strictness
- **min_length**: Filter small origins
- **jaccard_threshold**: Define matching criteria
- **min_overlap**: Minimum overlap for matches

### 4. Rich Output
- Predicted forks (left/right) as BED files
- Inferred origins and terminations
- Detailed benchmark report (text)
- 5 comprehensive plots (PNG)
- Overlap details (TSV)

## Integration Points

### With Existing Fork Training
```bash
# 1. Train fork model (existing)
python scripts/train_fork_model.py --config configs/fork_config.yaml

# 2. Call origins from trained model (NEW)
python scripts/predict_forks_and_call_oris.py \
    --config configs/ori_calling_pipeline.yaml
```

### With Existing Evaluation
```python
# Existing: Predict on reads
from replication_analyzer.evaluation import predict_on_all_reads

predictions = predict_on_all_reads(model, xy_data, max_length)

# NEW: Benchmark predictions
from replication_analyzer.evaluation import benchmark_ori_predictions

results = benchmark_ori_predictions(
    predicted_bed="preds.bed",
    curated_bed="curated.bed"
)
```

## Curated Dataset

**Location:** `/mnt/ssd-4tb/crisanto_project/data_2025Oct/DNAscent_Col0_NM30_ColCEN_ORIs_curated_final.bed`

**Statistics:**
- 10,549 curated origin annotations
- Format: 10-column BED file
- Contains: chr, start, end, read_id, strand, genomic coordinates

## Next Steps

### To Use the Pipeline:

1. **Update config** with your paths:
   ```yaml
   model:
     fork_model_path: "path/to/your/trained_fork_model.keras"
   data:
     base_dir: "path/to/your/xy_data"
     curated_ori_bed: "/mnt/ssd-4tb/crisanto_project/data_2025Oct/DNAscent_Col0_NM30_ColCEN_ORIs_curated_final.bed"
   ```

2. **Run the pipeline:**
   ```bash
   python scripts/predict_forks_and_call_oris.py \
       --config configs/ori_calling_pipeline.yaml
   ```

3. **Review results** in `results/ori_calling_pipeline/`

### For Development:

- Modules are fully integrated into package imports
- Can be used independently or as part of pipeline
- Well-documented with type hints and docstrings
- Follows existing code style and structure

## Testing

Basic import test passed:
```bash
python3 -c "from replication_analyzer.evaluation import benchmark_ori_predictions, read_bed_file"
# ✅ Success
```

For full testing, run the complete pipeline with your trained model.

## Benefits

1. **Complete Workflow**: Single command from model to benchmarked results
2. **Reproducible**: Config-based, version-controlled
3. **Modular**: Use components independently or together
4. **Well-Documented**: README, guide, inline docs
5. **Rich Output**: Multiple formats, plots, detailed reports
6. **Flexible**: Many configuration options and skip flags

## Summary

The origin calling functionality is now fully integrated into the replication-analyzer package. Users can:
- Train fork models (existing functionality)
- Predict forks and call origins (NEW)
- Benchmark against curated datasets (NEW)
- Generate comprehensive evaluation reports (NEW)

All documentation is in place, and the code follows the existing package structure and conventions.
