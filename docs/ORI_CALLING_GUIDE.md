# Origin Calling Pipeline Guide

## Overview

This guide explains how to use the integrated origin calling pipeline that:
1. Predicts forks using your trained AI model
2. Infers origins from fork patterns (Left → Right transitions)
3. Benchmarks predicted origins against curated datasets

## Prerequisites

- Trained fork detection model (`.keras` file)
- XY signal data (for fork prediction)
- Curated origin dataset (BED file) for benchmarking

## Quick Start

### 1. Configure the Pipeline

Edit `configs/ori_calling_pipeline.yaml`:

```yaml
experiment_name: "ori_calling_col0_pipeline"

data:
  # XY signal data directories
  base_dir: "/path/to/your/data"
  run_dirs:
    - "1strun_xy"
    - "2ndrun_xy"

  # Curated origins for benchmarking
  curated_ori_bed: "/path/to/curated_origins.bed"

model:
  # Your trained fork detection model
  fork_model_path: "models/fork_detector.keras"
  max_length: 200  # Must match training
  use_enhanced_encoding: true  # Must match training

prediction:
  fork_threshold: 0.5  # Higher = stricter fork calling

ori_calling:
  min_length: 100  # Minimum origin length in bp

benchmark:
  min_overlap: 1  # Minimum overlap to count as match
  jaccard_threshold: 0.0  # 0.0 = any overlap counts

output:
  results_dir: "results/ori_calling_pipeline"
```

### 2. Run the Pipeline

```bash
python scripts/predict_forks_and_call_oris.py \
    --config configs/ori_calling_pipeline.yaml
```

### 3. Review Results

The pipeline generates:

```
results/ori_calling_pipeline/
├── predicted_left_forks.bed       # AI-predicted left forks
├── predicted_right_forks.bed      # AI-predicted right forks
├── predicted_origins.bed          # ⭐ Inferred origins
├── predicted_terminations.bed     # Inferred terminations
├── benchmark_report.txt           # ⭐ Performance metrics
├── origin_overlaps.tsv            # Detailed overlap analysis
└── benchmark_plots/
    ├── overall_metrics.png        # Precision/Recall/F1
    ├── confusion_matrix.png       # TP/FP/FN
    ├── per_chromosome_metrics.png
    ├── jaccard_distribution.png
    └── length_comparison.png
```

## Understanding the Output

### BED Files

**predicted_origins.bed** format:
```
chr     start   end     read_id                 grad_left   grad_right
Chr1    25567   29859   4e6b0e79-dd9f-4357...   -0.45       0.52
```

- `grad_left`: Probability/gradient of left fork
- `grad_right`: Probability/gradient of right fork

### Benchmark Report

Example output:
```
Overall Metrics:
  Precision:  0.723 (512/708)
  Recall:     0.681 (512/752)
  F1 Score:   0.701

Confusion Matrix:
  True Positives:   512
  False Positives:  196
  False Negatives:  240

Overlap Quality:
  Mean Jaccard:     0.456
  Median Jaccard:   0.423
  Mean Overlap:     8543 bp
```

## Advanced Usage

### Option 1: Use Existing Fork Predictions

If you already have fork BED files:

1. Update config to point to existing files:
```yaml
forks:
  left_forks_bed: "path/to/left_forks.bed"
  right_forks_bed: "path/to/right_forks.bed"
```

2. Run with `--skip-prediction`:
```bash
python scripts/predict_forks_and_call_oris.py \
    --config configs/ori_calling_pipeline.yaml \
    --skip-prediction
```

### Option 2: Skip Benchmarking

If you don't have curated data:

```bash
python scripts/predict_forks_and_call_oris.py \
    --config configs/ori_calling_pipeline.yaml \
    --skip-benchmark
```

## Parameter Tuning

### Fork Threshold (`fork_threshold`)

Controls how strict fork calling is:

- **Lower (0.3-0.4)**: More forks detected, higher recall, lower precision
- **Medium (0.5)**: Balanced (default)
- **Higher (0.6-0.7)**: Fewer forks, higher precision, lower recall

### Minimum Origin Length (`min_length`)

Filters out small origins:

- **0**: Keep all origins
- **100**: Filter origins < 100bp (default, recommended)
- **500**: Only large origins
- **1000**: Very strict filtering

### Jaccard Threshold (`jaccard_threshold`)

Controls what counts as a "match" for benchmarking:

- **0.0**: Any overlap counts (default, lenient)
- **0.1**: At least 10% overlap
- **0.2**: At least 20% overlap (stricter)
- **0.5**: At least 50% overlap (very strict)

## Origin Calling Logic

### How Origins are Inferred

An origin is called when:
1. A **left fork** (negative gradient) is followed by
2. A **right fork** (positive gradient) on the **same read** and **same chromosome**

```
Read: ========L---------R========
              ^---------^
              ORIGIN
```

The origin region is defined as:
- **If no overlap**: The gap between fork end and next fork start
- **If partial overlap**: The overlapping region (intersection)
- **If total containment**: SKIPPED (nested forks not considered origins)

### Filtering Rules

Origins are filtered if:
- Length < `min_length` parameter
- One fork completely contains the other (containment)
- Empty interval (no valid region)

## Troubleshooting

### Problem: No origins called

**Possible causes:**
- Fork threshold too high → Lower `fork_threshold` to 0.3-0.4
- Min length too strict → Lower `min_length` to 0
- Fork model not detecting forks → Check fork predictions

**Debug steps:**
1. Check fork predictions: `wc -l results/*/predicted_*_forks.bed`
2. Lower thresholds and re-run
3. Visualize some reads to see fork patterns

### Problem: Low precision (many false positives)

**Solutions:**
- Increase `fork_threshold` to 0.6-0.7 (fewer, higher-confidence forks)
- Increase `min_length` to filter noise
- Check if model is overpredicting forks

### Problem: Low recall (many false negatives)

**Solutions:**
- Decrease `fork_threshold` to 0.3-0.4 (detect more forks)
- Decrease `min_length` to capture small origins
- Check if fork model has issues with certain regions

### Problem: Benchmarking shows poor overlap quality

**Indicators:**
- Low mean Jaccard index (< 0.2)
- Large difference in predicted vs curated lengths

**Possible causes:**
- Model predicting wrong boundaries
- Fork threshold affects fork extent
- Biological variability between datasets

## Integration with Existing Scripts

### Use ori_caller.py Directly

For more control, use the ori caller module directly:

```python
from replication_analyzer.evaluation.ori_caller import (
    parse_fork_bed, infer_events, write_bed6
)

# Load forks
left_segs = parse_fork_bed("left_forks.bed", kind="L")
right_segs = parse_fork_bed("right_forks.bed", kind="R")

# Call origins
origins, terminations, stats = infer_events(
    left_segs, right_segs,
    min_len=100
)

# Save
write_bed6("origins.bed", origins)
write_bed6("terminations.bed", terminations)
```

### Use Benchmarking Module

```python
from replication_analyzer.evaluation.benchmark import (
    benchmark_ori_predictions,
    plot_benchmark_results,
    save_benchmark_report
)

# Benchmark
results = benchmark_ori_predictions(
    predicted_bed="predicted_origins.bed",
    curated_bed="curated_origins.bed",
    min_overlap=1,
    jaccard_threshold=0.0
)

# Generate plots
plot_benchmark_results(results, "output_dir")

# Save report
save_benchmark_report(results, "report.txt")
```

## Best Practices

1. **Start with defaults** (threshold=0.5, min_length=100) and adjust based on results
2. **Examine benchmark plots** to understand model performance
3. **Check per-chromosome metrics** to identify problem regions
4. **Compare length distributions** (predicted vs curated) for systematic biases
5. **Iterate**: Adjust thresholds → Re-run → Evaluate → Repeat

## Citation

If you use this pipeline, please cite:
- The replication-analyzer package
- DNAscent (if using DNAscent data)
- Any relevant publications from the curated dataset

## Support

For issues or questions:
- Check documentation in `docs/`
- Review example configs in `configs/`
- Examine code in `replication_analyzer/evaluation/`
