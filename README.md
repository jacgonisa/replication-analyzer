# Replication Analyzer 🧬

Deep learning models for detecting replication origins (ORIs) and replication forks in BrdU/EdU labeled DNA sequencing data.

## Overview

This package provides modular, reproducible pipelines for:
- **ORI Detection**: Binary classification of replication origin segments
- **Fork Detection**: 3-class classification (background, left fork, right fork)
- **Origin Calling**: Infer origins from predicted forks and benchmark against curated datasets

### Key Features

- ✅ **Expert Models**: CNN + BiLSTM + Self-Attention architecture
- ✅ **Hybrid Balancing**: Combined oversampling + undersampling for class balance
- ✅ **Multi-channel Encoding**: 6 or 9-channel signal representations
- ✅ **Focal Loss**: Handles severe class imbalance
- ✅ **Regional Analysis**: Per-region evaluation (centromere, pericentromere, arms)
- ✅ **Origin Calling**: Predict forks with AI → Call origins → Benchmark performance
- ✅ **Config-based**: Easy to reproduce and modify experiments

## 📸 Gallery

### Workflow Overview

Complete pipeline from data input to annotation:

![Workflow](docs/images/workflow.png)

### Model Architecture

Expert Model: Multi-scale CNN + Bidirectional LSTM + Self-Attention

![Architecture](docs/images/architecture.png)

### Training Progress

Example training history showing convergence:

![Training History](docs/images/training_history.png)

### Prediction Example

Model prediction on a single read showing ORI detection:

![Example Prediction](docs/images/example_prediction.png)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd replication-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Organize your data as follows:
```
data/
├── raw/
│   ├── NM30_plot_data_1strun_xy/     # XY signal data (run 1)
│   ├── NM30_plot_data_2ndrun_xy/     # XY signal data (run 2)
│   ├── curated_origins.bed           # ORI annotations
│   ├── left_forks.bed                # Left fork annotations
│   └── right_forks.bed               # Right fork annotations
└── annotations/
    ├── centromere.bed
    └── pericentromere_clean.bed
```

### 2. Train ORI Detection Model

```python
from replication_analyzer import (
    load_all_xy_data,
    load_curated_origins,
    prepare_ori_data_hybrid,
    pad_sequences,
    build_ori_expert_model,
    FocalLoss
)
from sklearn.model_selection import train_test_split

# Load data
xy_data = load_all_xy_data(
    base_dir='data/raw',
    run_dirs=['NM30_plot_data_1strun_xy', 'NM30_plot_data_2ndrun_xy']
)
ori_annotations = load_curated_origins('data/raw/curated_origins.bed')

# Prepare data (hybrid balancing)
X_seq, y_seq, info = prepare_ori_data_hybrid(
    xy_data, ori_annotations,
    oversample_ratio=0.5,
    use_enhanced_encoding=True
)

# Pad sequences
X_padded, y_padded, max_length = pad_sequences(X_seq, y_seq, percentile=100)

# Reshape labels
y_binary = (y_padded > 0.3).astype('float32').reshape(-1, max_length, 1)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_padded, y_binary, test_size=0.2, random_state=42
)

# Build model
model = build_ori_expert_model(max_length=max_length, n_channels=9)

# Compile
model.compile(
    optimizer='adam',
    loss=FocalLoss(alpha=0.25, gamma=2.0),
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32
)

# Save model
model.save('models/ori_expert_model.keras')
```

### 3. Train Fork Detection Model

```python
from replication_analyzer import (
    load_fork_data,
    prepare_fork_data_hybrid,
    build_fork_detection_model,
    MultiClassFocalLoss
)

# Load fork data
left_forks, right_forks = load_fork_data(
    'data/raw/left_forks.bed',
    'data/raw/right_forks.bed'
)

# Prepare data
X_seq, y_seq, info = prepare_fork_data_hybrid(
    xy_data, left_forks, right_forks,
    oversample_ratio=0.5
)

# Pad and train (similar to ORI model)
X_padded, y_padded, max_length = pad_sequences(X_seq, y_seq)

# Build 3-class model
model = build_fork_detection_model(max_length=max_length, n_channels=9)

# Use multi-class focal loss
model.compile(
    optimizer='adam',
    loss=MultiClassFocalLoss(alpha=[1.0, 2.0, 2.0], gamma=2.0),
    metrics=['accuracy']
)

# Train...
```

## Using Config Files (Recommended)

Create a config file `configs/my_experiment.yaml`:

```yaml
experiment_name: "ori_detection_run1"

data:
  base_dir: "data/raw"
  run_dirs:
    - "NM30_plot_data_1strun_xy"
    - "NM30_plot_data_2ndrun_xy"
  ori_bed: "data/raw/curated_origins.bed"

preprocessing:
  oversample_ratio: 0.5
  percentile: 100
  use_enhanced_encoding: true

model:
  type: "ori_expert"
  max_length: 200
  n_channels: 9
  cnn_filters: 64
  lstm_units: 128
  dropout_rate: 0.3

training:
  epochs: 150
  batch_size: 32
  learning_rate: 0.0005
  loss:
    type: "focal"
    alpha: 0.25
    gamma: 2.0
  early_stopping_patience: 25

output:
  model_path: "models/ori_expert_model.keras"
  results_dir: "results/ori_run1"
```

Then run:
```bash
python scripts/train_ori_model.py --config configs/my_experiment.yaml
```

## Project Structure

```
replication-analyzer/
├── replication_analyzer/      # Main package
│   ├── data/                  # Data loading, encoding, preprocessing
│   ├── models/                # Model architectures and losses
│   ├── training/              # Training pipelines
│   ├── evaluation/            # Metrics and evaluation
│   └── visualization/         # Plotting utilities
├── scripts/                   # Executable scripts
├── configs/                   # Configuration files
├── notebooks/                 # Jupyter notebooks
├── data/                      # Data directory (gitignored)
├── models/                    # Saved models (gitignored)
└── results/                   # Results (gitignored)
```

## Key Components

### Signal Encoding
- **Basic (6 channels)**: Normalized, smooth, gradient, 2nd derivative, local mean/std
- **Enhanced (9 channels)**: + Z-score, cumulative trend, signal envelope

### Models
- **ORI Expert Model**: Multi-scale CNN + BiLSTM + Attention
- **Fork Detection Model**: Same architecture, 3-class output
- **Simple Models**: Baseline CNN architectures

### Balancing Strategy
- **Hybrid**: Combines oversampling minority + undersampling majority
- Target: ~1:1 read-level balance
- Handles segment-level imbalance with Focal Loss

## 📖 Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete step-by-step guide for training and annotation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical deep-dive: signal encoding & model architecture
- **[Example Data](data/examples/)** - Small BED files demonstrating annotation format

## Complete Pipeline: From Raw Data to Origin Calls 🎯

### Modular Workflow

The complete pipeline follows these steps:

```
1. Preprocessing → 2. Training → 3. Evaluation → 4. Prediction → 5. Origin Calling
```

Each step is independent and can be run separately, allowing you to:
- Use different fork sources (AI predictions vs DNAscent annotations)
- Iterate on origin calling parameters without re-running predictions
- Test on different datasets

---

## Step 4: Fork Prediction

Predict left and right replication forks from signal data using your trained AI model.

### Quick Start

```bash
# Run fork prediction
python scripts/predict_forks.py --config configs/fork_prediction.yaml
```

**Configuration** (`configs/fork_prediction.yaml`):
```yaml
experiment_name: "fork_prediction_col0"

data:
  base_dir: "/path/to/xy_data"
  run_dirs:
    - "NM30_plot_data_1strun_xy"
    - "NM30_plot_data_2ndrun_xy"

model:
  fork_model_path: "models/combined_fork_detector.keras"
  max_length: 411

prediction:
  fork_threshold: 0.5

output:
  results_dir: "results/fork_predictions"
```

**Output**:
- `predicted_left_forks.bed` - All predicted left forks (BED format)
- `predicted_right_forks.bed` - All predicted right forks (BED format)

---

## Step 5: Origin Calling & Benchmarking

Call origins from fork predictions (AI or DNAscent) and benchmark against curated datasets.

### Quick Start

```bash
# Call origins from AI-predicted forks
python scripts/call_origins_from_forks.py --config configs/ori_calling.yaml

# Or call origins from DNAscent forks
python scripts/call_origins_from_forks.py --config configs/ori_calling.yaml \
    --left-forks /path/to/dnascent_left_forks.bed \
    --right-forks /path/to/dnascent_right_forks.bed
```

**Configuration** (`configs/ori_calling.yaml`):
```yaml
experiment_name: "ori_calling_from_ai_forks"

forks:
  left_forks_bed: "results/fork_predictions/predicted_left_forks.bed"
  right_forks_bed: "results/fork_predictions/predicted_right_forks.bed"

ori_calling:
  min_length: 0  # Minimum origin length (bp)

data:
  curated_ori_bed: "/path/to/curated_origins.bed"

benchmark:
  min_overlap: 1
  jaccard_threshold: 0.0

output:
  results_dir: "results/ori_calling"
```

**Output Files**:
```
results/ori_calling/
├── predicted_origins.bed         # Inferred origins (L→R patterns)
├── predicted_terminations.bed    # Inferred terminations (R→L patterns)
├── benchmark_report.txt          # Performance metrics
├── origin_overlaps.tsv           # Detailed overlap analysis
└── benchmark_plots/
    ├── overall_metrics.png       # Precision/Recall/F1
    ├── confusion_matrix.png      # TP/FP/FN breakdown
    ├── per_chromosome_metrics.png
    ├── jaccard_distribution.png  # Overlap quality
    └── length_comparison.png     # Predicted vs curated
```

**What This Does**:
1. Loads fork BED files (from Step 4 or DNAscent)
2. Identifies **origins**: Left fork → Right fork (L→R) patterns on same read
3. Identifies **terminations**: Right fork → Left fork (R→L) patterns
4. Benchmarks predicted origins against curated dataset
5. Generates comprehensive evaluation plots

**Advanced Options**:

```bash
# Skip benchmarking (if no curated data available)
python scripts/call_origins_from_forks.py --config configs/ori_calling.yaml --skip-benchmark

# Override output directory
python scripts/call_origins_from_forks.py --config configs/ori_calling.yaml \
    --output-dir results/custom_output
```

---

## Tuning Parameters

- **fork_threshold**: Higher = fewer but higher-confidence forks (default: 0.5)
- **min_length**: Minimum origin length in bp (default: 0)
- **jaccard_threshold**: Minimum Jaccard overlap for true positive (default: 0.0)

## For Collaborators

### Analyzing New Fork Data

When you receive new fork annotations:

1. **Place data in `data/raw/`**
2. **Update config file** with new paths
3. **Run training**:
   ```bash
   python scripts/train_fork_model.py --config configs/new_forks.yaml
   ```
4. **Evaluate**:
   ```bash
   python scripts/evaluate_model.py --model models/fork_model.keras \
       --data-config configs/new_forks.yaml --output results/new_forks/
   ```
5. **Call origins and benchmark**:
   ```bash
   python scripts/predict_forks_and_call_oris.py --config configs/ori_calling_pipeline.yaml
   ```

### Batch Processing Multiple Datasets

See `scripts/batch_process.py` for processing multiple fork datasets in parallel.

## Citation

If you use this code, please cite:
```
[To be added when published]
```

## License

Private repository - not for public distribution.

## Contact

For questions or issues, contact: [Your contact info]
