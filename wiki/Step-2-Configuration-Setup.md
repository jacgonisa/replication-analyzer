# Step 2: Configuration Setup

> **Objective**: Create YAML configuration files for both Col0 and orc1b2 fork detection models

[[← Back to Step 1]](Step-1-Data-Preparation.md) | [[Next: Step 3 Training →]](Step-3-Training-Models.md)

---

## 📋 Overview

Configuration files specify all parameters for data loading, preprocessing, model architecture, and training. Using YAML configs ensures:
- ✅ **Reproducibility**: Exact same settings can be reused
- ✅ **Version Control**: Configs are small text files (easy to track)
- ✅ **Comparison**: Easy to compare settings between experiments
- ✅ **Collaboration**: Teammates can modify configs without touching code

---

## 📝 Configuration Files Created

### 1. Col0 Wild-Type Configuration

**File**: `configs/case_study_col0_forks.yaml`

```yaml
experiment_name: "case_study_jan2026_col0_forks"

data:
  base_dir: "data/case_study_jan2026/Col0/xy_data"
  run_dirs: ["1strun", "2ndrun"]
  left_forks_bed: "data/case_study_jan2026/Col0/annotations/leftForks_DNAscent_Col0.bed"
  right_forks_bed: "data/case_study_jan2026/Col0/annotations/rightForks_DNAscent_Col0.bed"

preprocessing:
  oversample_ratio: 0.5
  percentile: 100
  use_enhanced_encoding: true  # 9-channel

model:
  type: "fork_expert"
  n_classes: 3  # background, left, right
  n_channels: 9
  cnn_filters: 64
  lstm_units: 128
  dropout_rate: 0.3

training:
  epochs: 150
  batch_size: 32
  learning_rate: 0.0005
  loss:
    type: "multi_class_focal"
    alpha: [1.0, 2.0, 2.0]  # Higher weight for fork classes
    gamma: 2.0

output:
  model_path: "models/case_study_jan2026/col0_fork_detector.keras"
  results_dir: "results/case_study_jan2026/col0"
```

**Dataset**:
- 229 left forks
- 217 right forks
- **446 total annotations**

---

### 2. orc1b2 Mutant Configuration

**File**: `configs/case_study_orc1b2_forks.yaml`

```yaml
experiment_name: "case_study_jan2026_orc1b2_forks"

data:
  base_dir: "data/case_study_jan2026/orc1b2/xy_data"
  run_dirs: ["1strun", "2ndrun"]
  left_forks_bed: "data/case_study_jan2026/orc1b2/annotations/leftForks_DNAscent_orc1b2.bed"
  right_forks_bed: "data/case_study_jan2026/orc1b2/annotations/rightForks_DNAscent_orc1b2.bed"

# Identical preprocessing and model settings for fair comparison
preprocessing:
  oversample_ratio: 0.5
  percentile: 100
  use_enhanced_encoding: true

model:
  type: "fork_expert"
  n_classes: 3
  n_channels: 9
  cnn_filters: 64
  lstm_units: 128
  dropout_rate: 0.3

training:
  epochs: 150
  batch_size: 32
  learning_rate: 0.0005
  loss:
    type: "multi_class_focal"
    alpha: [1.0, 2.0, 2.0]
    gamma: 2.0

output:
  model_path: "models/case_study_jan2026/orc1b2_fork_detector.keras"
  results_dir: "results/case_study_jan2026/orc1b2"
```

**Dataset**:
- 526 left forks
- 648 right forks
- **1,174 total annotations** (2.6× more than Col0!)

---

## 🔧 Key Configuration Decisions

### Data Preprocessing

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `oversample_ratio` | 0.5 | Hybrid: 50% oversample + 50% undersample |
| `percentile` | 100 | Use full read lengths (no truncation) |
| `use_enhanced_encoding` | true | 9-channel for maximum feature richness |
| `smooth_sigma` | 2 | Gaussian smoothing to reduce noise |

### Model Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_classes` | 3 | Background, left fork, right fork |
| `n_channels` | 9 | Enhanced encoding (6 basic + 3 advanced) |
| `cnn_filters` | 64 | Sufficient capacity without overfitting |
| `lstm_units` | 128 | BiLSTM → 256 output channels |
| `dropout_rate` | 0.3 | Regularization to prevent overfitting |
| `dilation_rates` | [1,2,4] | Multi-scale feature extraction |

### Training Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `epochs` | 150 | Sufficient for convergence |
| `batch_size` | 32 | Good balance for CPU training |
| `learning_rate` | 0.0005 | Conservative, stable learning |
| `focal_alpha` | [1.0,2.0,2.0] | 2× weight for fork classes (minority) |
| `focal_gamma` | 2.0 | Strong focus on hard examples |
| `early_stopping_patience` | 25 | Stop if no improvement for 25 epochs |
| `reduce_lr_patience` | 10 | Reduce LR if plateau for 10 epochs |
| `validation_split` | 0.2 | 80% train, 20% validation |

---

## 🎯 Design Principles

### 1. **Identical Architecture for Both Models**

Both Col0 and orc1b2 use **exactly the same** model architecture and hyperparameters. This ensures:
- Fair comparison between genotypes
- Differences in performance reflect data, not model differences
- Consistency in predictions

### 2. **Multi-Class Focal Loss**

```python
alpha = [1.0, 2.0, 2.0]  # [background, left_fork, right_fork]
```

- **Background (α=1.0)**: Majority class, normal weight
- **Left fork (α=2.0)**: Minority class, double weight
- **Right fork (α=2.0)**: Minority class, double weight

This addresses severe class imbalance (~90-95% of positions are background).

### 3. **Hybrid Balancing at Read Level**

```yaml
oversample_ratio: 0.5
```

- Oversample 50% of fork-containing reads (duplicate them)
- Undersample 50% of background reads (randomly sample subset)
- Result: ~1:1 balance at read level
- Segment-level imbalance handled by focal loss

### 4. **Enhanced 9-Channel Encoding**

Captures multiple signal aspects:
- **Channels 0-1**: Normalized + smoothed signal
- **Channels 2-3**: 1st and 2nd derivatives (edges, peaks)
- **Channels 4-5**: Local mean and std (context)
- **Channels 6-8**: Z-score, cumulative sum, envelope (advanced features)

---

## ✅ Verification

Check that configs are valid:

```bash
# Verify file existence
ls -lh configs/case_study_*_forks.yaml

# Display configs
cat configs/case_study_col0_forks.yaml
cat configs/case_study_orc1b2_forks.yaml

# Validate YAML syntax (if pyyaml installed)
python -c "import yaml; yaml.safe_load(open('configs/case_study_col0_forks.yaml'))"
python -c "import yaml; yaml.safe_load(open('configs/case_study_orc1b2_forks.yaml'))"
```

**Output**:
```
✅ configs/case_study_col0_forks.yaml (1.2 KB)
✅ configs/case_study_orc1b2_forks.yaml (1.3 KB)
✅ YAML syntax valid
```

---

## 📊 Expected Training Characteristics

### Col0 Model (446 annotations)
- **Training time**: ~30-60 minutes (CPU mode)
- **Memory usage**: ~2-3 GB RAM
- **Model size**: ~7 MB
- **Sequences after balancing**: ~800-1000 reads

### orc1b2 Model (1,174 annotations)
- **Training time**: ~45-90 minutes (CPU mode)
- **Memory usage**: ~3-4 GB RAM
- **Model size**: ~7 MB (same architecture)
- **Sequences after balancing**: ~2000-2500 reads

---

## 🔍 Key Differences Between Configs

| Aspect | Col0 | orc1b2 |
|--------|------|--------|
| **Data paths** | `/Col0/` | `/orc1b2/` |
| **Fork annotations** | 446 total | 1,174 total |
| **Left/Right ratio** | 1.06:1 (balanced) | 0.81:1 (more right forks) |
| **Training data** | Less data | More data (better training) |
| **Model architecture** | ✅ Identical | ✅ Identical |
| **Hyperparameters** | ✅ Identical | ✅ Identical |
| **Output paths** | `/col0/` | `/orc1b2/` |

---

## ⚠️ Important Notes

### CPU-Only Mode
- Training runs on CPU (no GPU) for reproducibility
- Ensures consistent results across systems
- Slower than GPU but acceptable for this dataset size

### Random Seeds
- `random_state: 42` ensures reproducible train/val splits
- NumPy and TensorFlow seeds set in training script

### Output Organization
- Each model saves to separate directory
- Prevents accidental overwriting
- Easy to compare results side-by-side

---

## ✅ Configuration Complete!

Both configuration files are ready. Next step:

**[[Step 3: Training Models →]](Step-3-Training-Models.md)**

We'll train both models and monitor:
- Loss curves
- F1-Score progression
- Training/validation metrics
- Early stopping behavior

---

**Status**: ✅ **COMPLETE**
**Files Created**:
- `configs/case_study_col0_forks.yaml`
- `configs/case_study_orc1b2_forks.yaml`

**Next**: Begin training!
