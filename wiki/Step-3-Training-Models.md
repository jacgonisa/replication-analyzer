# Step 3: Training the Combined Fork Detection Model

> **Objective**: Train a unified fork detection model on all 1,620 fork annotations (Col0 + orc1b2 combined)

[[← Back to Step 2]](Step-2-Configuration-Setup.md) | [[Next: Step 4 Evaluation →]](Step-4-Model-Evaluation.md)

---

## 📋 Overview

Based on user feedback, we're adopting a **COMBINED APPROACH** instead of training separate models:

### ✅ **Why Combine All Data?**

1. **Maximum Training Data**: 1,620 fork annotations vs 446 (Col0 alone) or 1,174 (orc1b2 alone)
2. **Better Generalization**: Model learns from both genotypes → more robust
3. **Unified Predictions**: Single model for all future analysis
4. **Post-hoc Analysis**: Can still compare genotype-specific patterns after training

---

## 🎯 Training Strategy

### Dataset Composition

| Source | Left Forks | Right Forks | Total |
|--------|------------|-------------|-------|
| **Col0 (WT)** | 229 | 217 | 446 |
| **orc1b2 (mutant)** | 526 | 648 | 1,174 |
| **COMBINED** | **755** | **865** | **1,620** |

### XY Signal Data

- **4 sequencing runs combined**:
  - NM30_1strun (Col0 run 1)
  - NM30_2ndrun (Col0 run 2)
  - NM31_1strun (orc1b2 run 1)
  - NM31_2ndrun (orc1b2 run 2)

---

## 🔧 Configuration

**File**: `configs/case_study_combined_forks.yaml`

### Key Settings

```yaml
experiment_name: "case_study_jan2026_combined_forks"

data:
  base_dir: "data/case_study_jan2026/combined/xy_data"
  run_dirs: ["NM30_1strun", "NM30_2ndrun", "NM31_1strun", "NM31_2ndrun"]
  left_forks_bed: "data/case_study_jan2026/combined/annotations/leftForks_combined.bed"
  right_forks_bed: "data/case_study_jan2026/combined/annotations/rightForks_combined.bed"

preprocessing:
  oversample_ratio: 0.5  # Hybrid balancing
  use_enhanced_encoding: true  # 9-channel

model:
  type: "fork_expert"
  n_classes: 3  # background, left, right
  n_channels: 9
  cnn_filters: 64
  lstm_units: 128

training:
  epochs: 150
  batch_size: 32
  learning_rate: 0.0005
  loss:
    type: "multi_class_focal"
    alpha: [1.0, 2.0, 2.0]
    gamma: 2.0

output:
  model_dir: "models/case_study_jan2026"
  model_filename: "combined_fork_detector.keras"
  results_dir: "results/case_study_jan2026/combined"
```

---

## 🚀 Training Execution

### Command

```bash
# Using ONT conda environment (has TensorFlow 2.20.0)
# Running in tmux for safety (won't stop if disconnected)

tmux new-session -d -s fork_training \
  "conda run -n ONT python scripts/train_fork_model.py \
   --config configs/case_study_combined_forks.yaml \
   --plot 2>&1 | tee results/case_study_jan2026/combined/training_log.txt"
```

### Monitor Progress

```bash
# View tmux session
tmux attach -t fork_training

# Check log file
tail -f results/case_study_jan2026/combined/training_log.txt

# Check if training is running
ps aux | grep train_fork
```

---

## 📊 Training Pipeline Steps

### 1. **Data Loading** (In Progress)

```
LOADING DATA
├─ Loading XY signal from 4 runs
├─ Combining ~755 left fork annotations
├─ Combining ~865 right fork annotations
└─ Merging all data sources
```

**Status**: ✅ Running (processes active, loading data)

**Time Estimate**: 2-5 minutes (loading thousands of XY files)

### 2. **Data Preparation** (Pending)

```
PREPARING DATA
├─ Apply 9-channel encoding to each read
│   ├─ Channel 0: Normalized signal
│   ├─ Channel 1: Smoothed (Gaussian σ=2)
│   ├─ Channel 2: Gradient (1st derivative)
│   ├─ Channel 3: 2nd derivative (curvature)
│   ├─ Channel 4: Local mean (window=50)
│   ├─ Channel 5: Local std (variability)
│   ├─ Channel 6: Z-score (local normalization)
│   ├─ Channel 7: Cumulative sum (trend)
│   └─ Channel 8: Signal envelope
├─ Hybrid balancing (50% oversample + 50% undersample)
└─ Pad sequences to uniform length
```

**Time Estimate**: 10-15 minutes (encoding thousands of reads × 9 channels)

### 3. **Model Building** (Pending)

```
BUILDING MODEL
├─ Multi-scale CNN (dilations: 1, 2, 4)
├─ Encoder (2 downsampling blocks)
├─ Bidirectional LSTM (128 units)
├─ Self-Attention layer
├─ Decoder (2 upsampling blocks)
└─ Output head (3-class softmax)
```

**Parameters**: ~1.7M trainable parameters

**Time Estimate**: < 1 minute

### 4. **Training Loop** (Pending)

```
TRAINING
├─ 150 epochs (with early stopping)
├─ Batch size: 32
├─ Multi-class Focal Loss
├─ Callbacks:
│   ├─ Early stopping (patience=25)
│   ├─ LR reduction (patience=10)
│   └─ Model checkpoint (save best)
└─ Validation split: 80% train / 20% val
```

**Time Estimate**: 45-90 minutes (CPU mode, ~3000-4000 sequences)

### 5. **Plot Generation** (Pending)

```
GENERATING PLOTS
├─ Training history (loss, F1, accuracy)
├─ Validation metrics
└─ Summary statistics
```

**Time Estimate**: 1-2 minutes

---

## ⏱️ Expected Timeline

| Phase | Estimated Time | Status |
|-------|---------------|--------|
| **Data Loading** | 2-5 min | 🔄 In Progress |
| **Data Encoding** | 10-15 min | ⏳ Pending |
| **Model Build** | < 1 min | ⏳ Pending |
| **Training** | 45-90 min | ⏳ Pending |
| **Plotting** | 1-2 min | ⏳ Pending |
| **TOTAL** | **60-120 min** | 🔄 **Running** |

---

## 🖥️ Training Environment

- **System**: Linux 6.14.0-37-generic
- **Mode**: CPU only (reproducibility)
- **Conda Env**: ONT (TensorFlow 2.20.0)
- **Process**: Running in `tmux` session `fork_training`
- **Current Status**: Active (20-40% CPU usage, ~1.2 GB RAM)

---

## 📝 Real-Time Monitoring

### Check Training Status

```bash
# Process status
ps aux | grep train_fork

# CPU/Memory usage
top -p $(pgrep -f train_fork)

# Log output (when available)
tail -f results/case_study_jan2026/combined/training_log.txt
```

### Expected Log Output

Once data loading completes, you'll see:

```
======================================================================
FORK DETECTION MODEL TRAINING (3-CLASS)
======================================================================
✅ CPU cores: 64
✅ GPU disabled: True

======================================================================
LOADING DATA
======================================================================
📂 Loading XY data from 4 runs...
   ✓ NM30_1strun: XXX files
   ✓ NM30_2ndrun: XXX files
   ✓ NM31_1strun: XXX files
   ✓ NM31_2ndrun: XXX files
   Total reads: XXXX

📊 Fork annotations:
   Left forks: 755
   Right forks: 865
   Total: 1,620

======================================================================
PREPARING DATA
======================================================================
🔄 Encoding sequences with 9 channels...
⚖️  Applying hybrid balancing (ratio=0.5)...
📏 Padding sequences to max_length=XXXX...

📊 Data split:
   Train: XXXX samples
   Val: XXXX samples
   Shape: (XXXX, YYYY, 9)

⚖️  Class distribution:
   Background: XXXXXXX (XX.XX%)
   Left fork: XXXXX (X.XX%)
   Right fork: XXXXX (X.XX%)

======================================================================
TRAINING
======================================================================
Epoch 1/150
XXX/XXX [==============================] - XXs - loss: X.XXXX - f1: X.XXXX - val_loss: X.XXXX - val_f1: X.XXXX
...
```

---

## ✅ Next Steps

Once training completes (~60-120 minutes), we'll proceed to:

**[[Step 4: Model Evaluation →]](Step-4-Model-Evaluation.md)**

We'll evaluate:
1. Overall model performance
2. Per-genotype analysis (Col0 vs orc1b2)
3. Regional patterns (if genomic regions available)
4. Fork directionality accuracy

---

**Status**: 🔄 **TRAINING IN PROGRESS**

**Started**: 2026-01-04 12:26:11

**Est. Completion**: ~2026-01-04 14:00 (depending on data loading)

**Monitor Command**: `tmux attach -t fork_training`
