#!/bin/bash
# Full v4.4 pipeline: wait for preprocessing → HP tuning → update config → train
# Run from /replication-analyzer/ root
# Usage: bash CODEX/scripts/run_v4.4_pipeline.sh

set -e
cd /mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer

PYTHON="/home/jg2070/miniforge3/envs/ONT/bin/python"
LOG_DIR="CODEX/results/forte_v4.4"
NPZ="$LOG_DIR/preprocessed_forte_v4.4.npz"
HP_DIR="$LOG_DIR/hp_tuning"
CONFIG="CODEX/configs/forte_v4.4.yaml"

echo "========================================================"
echo "FORTE v4.4 pipeline — $(date)"
echo "========================================================"

# ── Step 1: wait for preprocessing ───────────────────────────────────────────
echo ""
echo "[1/3] Waiting for preprocessing to finish..."
while [ ! -f "$NPZ" ]; do
    sleep 30
done
echo "  Preprocessing done: $NPZ"

# ── Step 2: HP tuning (max_epochs=80, n_trials=50, patience=15) ──────────────
echo ""
echo "[2/3] Starting HP tuning (n_trials=50, max_epochs=80)..."
env CUDA_VISIBLE_DEVICES="" $PYTHON -u \
    CODEX/scripts/hp_tuning_optuna.py \
    --preprocessed "$NPZ" \
    --output-dir   "$HP_DIR" \
    --n-trials     50 \
    --max-epochs   80 \
    --patience     15 \
    --study-name   forte_v4.4_hp \
    > "$LOG_DIR/hp_tuning.log" 2>&1
echo "  HP tuning done. Log: $LOG_DIR/hp_tuning.log"

# ── Step 3: update config with best HPs ──────────────────────────────────────
echo ""
echo "[3a/3] Patching config with best HPs..."
$PYTHON - << 'PYEOF'
import yaml, sys
from pathlib import Path

best_yaml = Path("CODEX/results/forte_v4.4/hp_tuning/best_hps.yaml")
config_path = Path("CODEX/configs/forte_v4.4.yaml")

with open(best_yaml) as f:
    best = yaml.safe_load(f)
with open(config_path) as f:
    cfg = yaml.safe_load(f)

cfg["training"]["learning_rate"] = best["training"]["learning_rate"]
cfg["training"]["batch_size"]    = best["training"]["batch_size"]
cfg["training"]["loss"]["alpha"] = best["training"]["loss"]["alpha"]
cfg["training"]["loss"]["gamma"] = best["training"]["loss"]["gamma"]
cfg["model"]["cnn_filters"]      = best["model"]["cnn_filters"]
cfg["model"]["dropout_rate"]     = best["model"]["dropout_rate"]

with open(config_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"  Config updated: {config_path}")
print(f"  lr={cfg['training']['learning_rate']:.6f}  "
      f"bs={cfg['training']['batch_size']}  "
      f"cnn={cfg['model']['cnn_filters']}  "
      f"dropout={cfg['model']['dropout_rate']:.3f}  "
      f"gamma={cfg['training']['loss']['gamma']:.3f}")
PYEOF

# ── Step 4: final training ────────────────────────────────────────────────────
echo ""
echo "[3b/3] Starting final training..."
env CUDA_VISIBLE_DEVICES="" $PYTHON -u \
    CODEX/scripts/train_weak5_codex.py \
    --config "$CONFIG" \
    > "$LOG_DIR/training_v4.4.log" 2>&1
echo "  Training done. Log: $LOG_DIR/training_v4.4.log"

echo ""
echo "========================================================"
echo "FORTE v4.4 pipeline COMPLETE — $(date)"
echo "========================================================"
