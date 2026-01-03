# Replication Analyzer - Complete Usage Guide

## 🚀 Quick Start for Collaborators

When you receive new fork data from collaborators, here's the complete workflow:

---

## 📋 Workflow for New Fork Data

### **Step 1: Organize Your Data**

Place new fork data in `data/raw/new_experiment/`:

```bash
data/raw/new_experiment/
├── plot_data_1strun_xy/      # XY signal files
│   ├── plot_data_read1.txt
│   ├── plot_data_read2.txt
│   └── ...
├── left_forks.bed             # Left fork annotations
└── right_forks.bed            # Right fork annotations
```

### **Step 2: Create a Config File**

Copy and modify the default config:

```bash
cp configs/fork_model_default.yaml configs/new_experiment.yaml
```

Edit `configs/new_experiment.yaml`:

```yaml
experiment_name: "new_experiment_forks"

data:
  base_dir: "data/raw/new_experiment"
  run_dirs:
    - "plot_data_1strun_xy"
  left_forks_bed: "data/raw/new_experiment/left_forks.bed"
  right_forks_bed: "data/raw/new_experiment/right_forks.bed"

# ... keep other settings ...

output:
  model_dir: "models"
  model_filename: "fork_new_experiment.keras"
  results_dir: "results/fork_new_experiment"
```

### **Step 3: Train the Model**

```bash
# Activate virtual environment
source venv/bin/activate

# Train with plots
python scripts/train_fork_model.py \
    --config configs/new_experiment.yaml \
    --plot
```

**Expected output:**
- Model saved: `models/fork_new_experiment.keras`
- Training history: `results/fork_new_experiment/training_history.csv`
- Training plots: `results/fork_new_experiment/plots/`
- Dataset info: `results/fork_new_experiment/dataset_info.csv`

### **Step 4: Evaluate the Model**

```bash
python scripts/evaluate_model.py \
    --model models/fork_new_experiment.keras \
    --type fork \
    --config configs/new_experiment.yaml \
    --output results/fork_new_experiment_eval
```

**Expected output:**
- Comprehensive metrics: `results/fork_new_experiment_eval/overall_metrics.csv`
- Confusion matrices, ROC curves, etc.
- Predictions on all segments

### **Step 5: Annotate NEW Data (No Ground Truth)**

When you receive completely new data without annotations:

```bash
python scripts/annotate_new_data.py \
    --model models/fork_new_experiment.keras \
    --type fork \
    --data-dir data/raw/unknown_sample \
    --output results/unknown_sample_annotations \
    --threshold 0.5 \
    --min-length 100
```

**Expected output:**
- `fork_segment_predictions.tsv` - All segment-level predictions
- `called_left_forks.bed` - Detected left forks (BED format)
- `called_right_forks.bed` - Detected right forks (BED format)
- `called_left_forks.tsv` - Detailed left fork info
- `called_right_forks.tsv` - Detailed right fork info
- `all_called_forks.tsv` - Combined results
- `example_plots/` - Visual examples

---

## 🔬 ORI Detection Workflow

### Train ORI Model

```bash
python scripts/train_ori_model.py \
    --config configs/ori_model_default.yaml \
    --plot
```

### Evaluate ORI Model

```bash
python scripts/evaluate_model.py \
    --model models/ori_expert_model.keras \
    --type ori \
    --config configs/ori_model_default.yaml
```

### Annotate New ORIs

```bash
python scripts/annotate_new_data.py \
    --model models/ori_expert_model.keras \
    --type ori \
    --data-dir data/raw/new_sample \
    --output results/new_sample_oris \
    --threshold 0.5
```

---

## 📊 Understanding the Outputs

### 1. Training Results

**`training_history.csv`**: All metrics per epoch
- loss, val_loss
- accuracy, val_accuracy
- f1, val_f1 (or f1_macro for fork models)
- precision, recall, auc

**Training plots**: Visual representation of training progress

### 2. Evaluation Results

**`overall_metrics.csv`**: Performance summary
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Kappa, MCC

**`regional_metrics.csv`** (ORI only): Per-region performance
- Centromere, Pericentromere, Arms

**`predictions.tsv`**: All segment-level predictions

### 3. Annotation Results

**BED files**: Genome browser compatible
```
chr1  1000  2000  read_id  score  .
```

**TSV files**: Detailed information
```
chr  start  end  read_id  max_prob  mean_prob  length  fork_type
```

---

## ⚙️ Configuration Options

### Key Parameters to Tune

**For Training:**
```yaml
preprocessing:
  oversample_ratio: 0.5      # How much to duplicate minority class
  percentile: 100            # Max sequence length (95 or 100)

training:
  epochs: 150
  batch_size: 32
  learning_rate: 0.0005

  loss:
    alpha: [1.0, 2.0, 2.0]  # Class weights for fork model
    gamma: 2.0               # Focal loss focusing parameter
```

**For Annotation:**
```bash
--threshold 0.5        # Lower = more sensitive (more false positives)
--min-length 100       # Minimum peak size in bp
```

---

## 🎯 Tips for Optimal Results

### 1. Choosing Threshold

- **High Precision** (few false positives): `--threshold 0.7`
- **High Recall** (catch everything): `--threshold 0.3`
- **Balanced**: `--threshold 0.5` (default)

Find optimal threshold:
```python
from replication_analyzer.evaluation.metrics import find_optimal_threshold

# After evaluation
optimal_thresh, recall, precision = find_optimal_threshold(
    y_true, y_pred_proba, target_recall=0.9
)
```

### 2. Handling Imbalance

If forks are very rare:
```yaml
training:
  loss:
    alpha: [1.0, 3.0, 3.0]  # Give more weight to fork classes
```

### 3. Model Convergence

Watch for:
- ✅ **Good**: Val loss decreases steadily
- ⚠️ **Overfitting**: Train loss ↓↓, val loss ↑
- 🔧 **Fix**: Increase dropout, reduce epochs

---

## 📁 File Structure Reference

```
replication-analyzer/
├── data/
│   ├── raw/              # Your input data (gitignored)
│   ├── processed/        # Preprocessed data (gitignored)
│   └── annotations/      # Genomic regions (centromere, etc.)
├── models/               # Trained models (gitignored)
├── results/              # All outputs (gitignored)
├── configs/              # YAML configuration files
├── scripts/              # Executable scripts ⭐
│   ├── train_ori_model.py
│   ├── train_fork_model.py
│   ├── evaluate_model.py
│   └── annotate_new_data.py  ← MOST IMPORTANT!
└── notebooks/            # Original research notebook
```

---

## 🆘 Troubleshooting

### Error: "No reads found"
- Check `data/raw/` paths in config
- Ensure directory names match config

### Error: "Model input shape mismatch"
- Models trained with different `percentile` settings
- Retrain or use same preprocessing settings

### Poor Performance
1. Check class balance in `dataset_info.csv`
2. Try different `oversample_ratio` values
3. Increase training epochs
4. Adjust `alpha` weights in focal loss

### Out of Memory
- Reduce `batch_size` to 16
- Reduce `percentile` to 95
- Force CPU mode (already enabled)

---

## 💡 Advanced Usage

### Batch Processing Multiple Datasets

```bash
# Create a loop
for exp in exp1 exp2 exp3; do
    python scripts/annotate_new_data.py \
        --model models/fork_detector.keras \
        --type fork \
        --data-dir data/raw/$exp \
        --output results/${exp}_annotations
done
```

### Programmatic Use

```python
from replication_analyzer import *
from replication_analyzer.training import train_fork_model
from replication_analyzer.evaluation import predict_on_all_reads

# Load config
import yaml
with open('configs/my_config.yaml') as f:
    config = yaml.safe_load(f)

# Train
model, history, max_len, info = train_fork_model(config)

# Predict
xy_data = load_all_xy_data('data/raw/new_data')
predictions = predict_on_all_reads(model, xy_data, max_len)
```

---

## 📧 Support

For questions or issues:
1. Check this guide first
2. Review the original notebook: `notebooks/readclassification.ipynb`
3. Contact: [Your contact info]

---

**Happy analyzing! 🧬🔬**
