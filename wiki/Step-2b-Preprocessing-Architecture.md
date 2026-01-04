# Step 2b: Preprocessing Architecture (Checkpoint System)

> **Learning from experience**: Separating preprocessing from training for robust ML pipelines

[[← Back to Step 2]](Step-2-Configuration-Setup.md) | [[Next: Step 3 Training →]](Step-3-Training-Models.md)

---

## 🎯 Problem Identified

During initial training attempts, we discovered a **critical inefficiency**:

### ❌ **Original Approach Problems:**
1. **Every training attempt** reloaded 44,798 XY files (~7 min)
2. **Every attempt** re-encoded to 9 channels (~10-15 min)
3. **Config errors** meant **~20 min wasted** per retry
4. **Total**: 15-20 minutes of repeated preprocessing on every attempt

**Attempts before success**: 5 config errors × 20 min = **100+ minutes of repeated work!**

---

## ✅ **Solution: Preprocessing Checkpoint Architecture**

We implemented a **3-tier checkpoint system**:

### **Tier 1: Preprocessing Checkpoint** ⭐ (MOST IMPORTANT)
**Run ONCE, use forever**

```bash
# Step 1: Preprocess data ONCE (~20 min one-time cost)
python scripts/preprocess_fork_data.py \
    --config configs/case_study_combined_forks.yaml \
    --output data/preprocessed/combined_forks.npz \
    --save-info

# Outputs:
# - data/preprocessed/combined_forks.npz (~500MB-1GB)
# - data/preprocessed/combined_forks.json (metadata)
```

**What it does:**
- ✅ Loads 44,798 XY files from 4 runs
- ✅ Applies 9-channel encoding to all reads
- ✅ Hybrid balancing (50% oversample + 50% undersample)
- ✅ Sequence padding to uniform length
- ✅ Saves as compressed NumPy archive

**Benefits:**
- **First run**: ~20 min (same as before)
- **Every retry**: **~30 seconds** to load preprocessed data
- **Savings**: **19.5 min per retry!**

### **Tier 2: Model Training Checkpoint** (AUTO)
**Already implemented in training script**

```python
# Automatically saves best model during training
ModelCheckpoint(
    filepath='models/combined_fork_detector.keras',
    save_best_only=True,  # Only best val_loss
    monitor='val_loss'
)
```

### **Tier 3: Resume Training Checkpoint** (TODO - Future)
**For interrupted training sessions**

```python
# Save: epoch number, optimizer state, history
# Resume from last checkpoint if training interrupted
```

---

## 📂 **File: `scripts/preprocess_fork_data.py`**

### Features

```python
#!/usr/bin/env python
"""
Preprocessing script for Fork detection data.

Separates data loading/encoding from training.
"""

def main():
    # Load configuration
    config = yaml.safe_load(open(args.config))

    # STEP 1: Load XY data
    xy_data = load_all_xy_data(...)

    # STEP 2: Load fork annotations
    left_forks, right_forks = load_fork_data(...)

    # STEP 3: Encode & balance
    X_seq, y_seq, info = prepare_fork_data_hybrid(
        xy_data, left_forks, right_forks,
        oversample_ratio=0.5,
        use_enhanced_encoding=True  # 9 channels
    )

    # STEP 4: Pad sequences
    X_padded, y_padded, max_length = pad_sequences(...)

    # STEP 5: Save as compressed archive
    np.savez_compressed(
        output_path,
        X=X_padded,              # Shape: (N, L, 9)
        y=y_padded,              # Shape: (N, L)
        max_length=max_length,
        read_ids=info.index,
        has_fork=info['has_fork']
    )

    # STEP 6: Save metadata JSON
    metadata = {
        'n_sequences': len(X_padded),
        'data_shape': X_padded.shape,
        'class_distribution': {...},
        'preprocessing_params': {...}
    }
```

### Usage

```bash
# Preprocess data
python scripts/preprocess_fork_data.py \
    --config configs/case_study_combined_forks.yaml \
    --output data/preprocessed/combined_forks.npz \
    --save-info
```

**Output:**
```
======================================================================
STEP 1: LOADING RAW DATA
======================================================================
Loading from: data/case_study_jan2026/combined/xy_data/NM30_1strun
Found 15936 files...
[Progress: 100/15936, 200/15936, ...]
✓ Loaded 3,051,988 data points from 44,798 reads

======================================================================
STEP 2: ENCODING & BALANCING
======================================================================
🔄 Encoding sequences with 9 channels...
⚖️  Applying hybrid balancing (ratio=0.5)...
✅ Data prepared! Total reads: 3,151

======================================================================
STEP 3: PADDING SEQUENCES
======================================================================
📏 Padding to max_length=411...
✅ Final shape: (3151, 411, 9)

======================================================================
STEP 4: SAVING
======================================================================
💾 Saving to: data/preprocessed/combined_forks.npz
✅ Saved! File size: 512.3 MB
📋 Metadata saved to: data/preprocessed/combined_forks.json
```

---

## 📝 **Updated Training Script**

### File: `scripts/train_fork_model.py` (Modified)

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--config', required=True)
    parser.add_argument('--preprocessed', default=None,
                       help='Path to .npz file - FAST TRAINING!')
    parser.add_argument('--plot', action='store_true')

    # Load config
    config = yaml.safe_load(open(args.config))

    # Add preprocessed data path if provided
    if args.preprocessed:
        print(f"✅ Using preprocessed data: {args.preprocessed}")
        config['preprocessed_data_path'] = args.preprocessed
        print("MODE: FAST TRAINING (using cached data)")

    # Train model (handles both modes)
    model, history, max_length, info = train_fork_model(config)
```

### Usage

```bash
# METHOD 1: Traditional (slow - loads & encodes every time)
python scripts/train_fork_model.py \
    --config configs/case_study_combined_forks.yaml \
    --plot

# METHOD 2: Fast training with preprocessed data ⭐
python scripts/train_fork_model.py \
    --preprocessed data/preprocessed/combined_forks.npz \
    --config configs/case_study_combined_forks.yaml \
    --plot
```

---

## 📊 **Performance Comparison**

| Approach | Data Loading | Encoding | Training | Total | Retries Penalty |
|----------|--------------|----------|----------|-------|-----------------|
| **Traditional** | ~7 min | ~10 min | ~45-90 min | 62-107 min | +17 min per retry |
| **Preprocessed** | ~30 sec | -- | ~45-90 min | 46-91 min | +0.5 min per retry |
| **Savings** | **93%** | **100%** | -- | **16-17 min** | **34× faster retries!** |

**Real-World Impact (Case Study):**
- **Config debugging**: 5 retries
- **Traditional**: 5 × 17 min = **85 min wasted**
- **Preprocessed**: 5 × 0.5 min = **2.5 min wasted**
- **Savings**: **82.5 minutes!**

---

## 🔍 **Inspecting Preprocessed Data**

### Load and examine:

```python
import numpy as np
import json

# Load preprocessed data
data = np.load('data/preprocessed/combined_forks.npz')

print("Keys:", list(data.keys()))
print("X shape:", data['X'].shape)  # (3151, 411, 9)
print("y shape:", data['y'].shape)  # (3151, 411)
print("Max length:", data['max_length'])  # 411

# Load metadata
with open('data/preprocessed/combined_forks.json') as f:
    meta = json.load(f)
print(json.dumps(meta, indent=2))
```

### Example metadata:

```json
{
  "experiment_name": "case_study_jan2026_combined_forks",
  "preprocessing_date": "2026-01-04T16:00:00",
  "data_shape": [3151, 411, 9],
  "n_sequences": 3151,
  "n_channels": 9,
  "max_length": 411,
  "class_distribution": {
    "background": 1000569,
    "left_fork": 14912,
    "right_fork": 20239
  },
  "reads_with_forks": 1444,
  "reads_without_forks": 1707,
  "preprocessing_params": {
    "oversample_ratio": 0.5,
    "percentile": 100,
    "enhanced_encoding": true,
    "random_seed": 42
  }
}
```

---

## ⚠️ **Lessons Learned**

### 1. **Always Separate Preprocessing**
- Data loading is **I/O bound** (slow)
- Encoding is **CPU bound** (slow)
- Training is different each time (hyperparameters)
- **Never couple slow preprocessing with experimental training!**

### 2. **Save Metadata**
- Document **exactly** what preprocessing was done
- Track random seeds for reproducibility
- Record class distributions for debugging

### 3. **Version Your Preprocessed Data**
```bash
data/preprocessed/
├── combined_forks_v1.npz  # Original
├── combined_forks_v2.npz  # After fixing encoding bug
└── combined_forks_final.npz  # Production
```

### 4. **Disk Space is Cheap, Time is Not**
- **500MB-1GB** for preprocessed data
- **Saves 17 min per retry**
- **Worth it** after just 1 retry!

---

## ✅ **Best Practices**

### For Large Datasets:

1. **Always preprocess separately**
2. **Save preprocessed data with metadata**
3. **Use version control for preprocessed data**
4. **Document preprocessing steps in metadata**
5. **Test loading speed before committing**

### For Experiments:

```bash
# Preprocess ONCE
python scripts/preprocess_fork_data.py --config exp1.yaml --output data/exp1.npz

# Experiment rapidly
python scripts/train_fork_model.py --preprocessed data/exp1.npz --config exp1_v1.yaml
python scripts/train_fork_model.py --preprocessed data/exp1.npz --config exp1_v2.yaml
python scripts/train_fork_model.py --preprocessed data/exp1.npz --config exp1_v3.yaml
```

---

## 📈 **Next Steps**

Now that we have the preprocessing checkpoint system:

1. ✅ **Run preprocessing once** (Step 3a)
2. ✅ **Fast training experiments** (Step 3b)
3. ✅ **Evaluate results** (Step 4)

**[[Proceed to Step 3: Training →]](Step-3-Training-Models.md)**

---

**Status**: ✅ **Preprocessing Architecture Implemented**

**Key Achievement**: **Reduced retry penalty from 17 min to 30 sec (34× speedup!)**
