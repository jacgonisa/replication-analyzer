# FORTE — Fork and Origin Replication Tracking Engine

**CODEX** (CNN-biLSTM Origin Detection EXperiment) is the training and evaluation stack
for FORTE, a deep-learning model that detects DNA replication events — left forks,
right forks, and origins of replication (ORIs) — from single-molecule BrdU incorporation
signal measured by nanopore sequencing.

---

## Table of Contents

1. [Task and Data](#1-task-and-data)
2. [Signal Representation](#2-signal-representation)
3. [Model Architecture](#3-model-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Model Lineage](#5-model-lineage)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Results Summary](#7-results-summary)
8. [Key Findings](#8-key-findings)
9. [How to Run](#9-how-to-run)
10. [Repository Structure](#10-repository-structure)

---

## 1. Task and Data

### What the model does
Given a long nanopore read with BrdU signal, classify each genomic window as one of:

| Class | ID | Description |
|---|---|---|
| background | 0 | No replication event |
| left_fork | 1 | Leftward-moving replication fork (decreasing BrdU signal) |
| right_fork | 2 | Rightward-moving replication fork (increasing BrdU signal) |
| origin | 3 | Origin of replication (ORI) — peak flanked by diverging forks |

### Biology
DNA replication fires bidirectionally from origins. A replication bubble looks like:

```
←left_fork   [origin]   right_fork→
  decreasing  BrdU peak   increasing
```

Termination zones are where two opposing forks meet (not currently modelled as a separate class).

### Data
- **Organism:** *Arabidopsis thaliana* (Col-0 wild-type and orc1b/2 mutant)
- **Experiments:** NM30 (Col-0 run 1 & 2), NM31 (orc1b/2 run 1 & 2)
- **Read filter:** min length 30 kb, nascent fraction ≥ 40%
- **Ground truth annotations:** manually curated BED files
  - `data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed`
  - `data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed`
  - `data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed`
- **Annotation counts (real):** 2,133 left forks | 1,944 right forks | 21,334 origins
- **Total reads:** ~48,000 | **Val split:** ~9,643 reads (20%)

---

## 2. Signal Representation

Each read is divided into genomic bins. The raw BrdU signal per bin is expanded into
a **rectangular block** representation (uniform signal within each bin), then encoded
into a **9-channel feature vector** per window:

| Channel | Description |
|---|---|
| `norm_signal` | Normalised BrdU signal |
| `approx_upsampled` | Wavelet approximation coefficients (db4, level 2) |
| `detail1_upsampled` | Wavelet detail level 1 |
| `detail2_upsampled` | Wavelet detail level 2 |
| `local_mean` | Rolling mean over 50-window neighbourhood |
| `local_std` | Rolling std over 50-window neighbourhood |
| `z_score` | (signal − local_mean) / local_std |
| `cumsum` | Cumulative sum of normalised signal |
| `envelope` | Signal envelope (Hilbert transform magnitude) |

**Key finding:** No single feature can distinguish left_fork from right_fork (both score
identically in single-feature benchmarks). Fork direction requires spatial context across
multiple windows → the BiLSTM is essential.

---

## 3. Model Architecture

```
Input: (batch, max_length=411, 9 channels)
  ↓
Conv1D(64 filters, kernel=7) + BatchNorm + ReLU + Dropout(0.3)
Conv1D(64 filters, kernel=5) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Bidirectional LSTM(128 units) + Dropout(0.3)
  ↓
SelfAttention(64 heads)
  ↓
Dense(4) + Softmax  →  per-window class probabilities
```

- **Total parameters:** ~1.2 M
- **Loss:** Sparse Categorical Focal Loss with per-class alpha weights
  (default: bg=1.0, left_fork=2.5, right_fork=2.0, origin=2.5 in v3)
- **Why BiLSTM:** Processes the read in both directions so each window has full
  context from both upstream and downstream. Essential for directionality.
- **Why focal loss:** Addresses heavy class imbalance (~98% background windows).
  Focal term γ=2 down-weights easy background examples.
- **Why dropout=0.3:** Regularisation; causes val_loss < train_loss (expected, not overfitting).

---

## 4. Training Pipeline

### Preprocessing
```bash
python CODEX/scripts/preprocess_weak4_codex.py --config CODEX/configs/<model>.yaml
```
Encodes all reads into (x, y, weights) tensors saved as chunked pickle files.
Chunks enable low-memory streaming; FORTE v3+ loads all chunks into RAM for shuffling.

### Pseudo-label generation (FORTE only)
```bash
python CODEX/scripts/generate_pseudo_labels_forte.py \
    --config CODEX/configs/forte_v1.yaml \
    --model CODEX/models/weak5_rectangular_v4.keras \
    --source-config CODEX/configs/weak5_rectangular_v4.yaml \
    --output CODEX/results/forte_v1/pseudo_labels
```
Runs the previous model on all reads and extracts predicted events above threshold
to augment real annotations with pseudo-labels.

### Training
```bash
nohup python CODEX/scripts/train_weak5_codex.py --config CODEX/configs/<model>.yaml \
    > CODEX/results/<model>/train.log 2>&1 &
```
Logs are saved directly to `CODEX/results/<model>/train.log` (from FORTE v3 onward).

### Key training settings (v3/v4)
| Setting | Value |
|---|---|
| Optimiser | Adam, lr=2e-4 |
| Batch size | 64 |
| Max epochs | 150 |
| Early stopping | patience=40 on **val_loss** (v4.3+) |
| LR reduction | factor=0.5, patience=15 on val_loss |
| Val fraction | 20% (fixed split manifest, same across all FORTE models) |
| Metrics tracked | val_loss, val_masked_f1_macro, **val_masked_mean_iou** (v4.3+) |

Note: v1/v2 used val_f1_macro for monitoring; v4.3 onward uses val_loss (more stable).

### Flip augmentation (FORTE v3+)
For every training read that contains fork annotations, a **reversed copy** is added:
- Signal reversed along the genomic axis
- left_fork ↔ right_fork labels swapped

This symmetrises the BiLSTM's contextual learning and fixes the left_fork under-prediction
bias observed in v1/v2. Adds ~12,290 extra samples per epoch (+29%).

**Important:** `flip_augment: true` was present in v1/v2 configs but was silently ignored
(the key was read but not implemented). It is only active from **v3 onward**.

---

## 5. Model Lineage

### weak5_rectangular_v4 (predecessor)
- Trained on real annotations only, rectangular signal representation
- Used as the teacher model to generate FORTE pseudo-labels
- Window-level val F1 ~0.48
- **Known issue:** Predicts origins well but fork direction accuracy is limited

### FORTE series
All FORTE models share the same architecture and val split. Differences are in annotations only.

| Model | Fork labels | ORI labels | Flip aug | Monitor | Best epoch | Val IoU |
|---|---|---|---|---|---|---|
| **v1** | real + pseudo (thr=0.50) | real + pseudo (~42k) | ❌ broken | val_f1 | 9 | — |
| **v1_conservative** | real + pseudo (thr=0.50) | real only (21k) | ❌ broken | val_f1 | 18 | — |
| **v2** | real + pseudo (thr=0.45) | real + pseudo (~42k) | ❌ broken | val_f1 | 7 | — |
| **v2_conservative** | real + pseudo (thr=0.45) | real only (21k) | ❌ broken | val_f1 | 21 | — |
| **v3** | same as v2_cons | same as v2_cons | ✅ working | val_f1 | 4 | — |
| **v4** | same as v3 | same as v3 | ✅ working | val_f1 | 4 | — |
| **v4.2** | same as v3 | same as v3 | ✅ working | val_f1 → val_loss | 11\* | — |
| **v5** | same as v2 | same as v2 | ❌ broken | val_f1 | — | — |
| **v4.3** | 23k LF + 23k RF (thr=0.20) | 33,378 **flanked** pseudo-ORIs | ✅ | val_loss | HP tuned | ~0.71 |
| **v4.4** | same as v4.3 | same as v4.3 (flanked) | ✅ | val_masked_mean_iou | 19 | 0.709 |
| **v4.5** | same as v4.3 | **21,334 Nerea direct** ORIs | ✅ | val_masked_mean_iou | training | — |

\* v4.2 checkpoint saved epoch 1 (monitored val_f1_macro); epoch 11 was the val_loss minimum.

**Annotation counts:**

| Model | Left forks | Right forks | Origins |
|---|---|---|---|
| v1 | 3,129 | 2,681 | ~42,000 |
| v1_conservative | 3,129 | 2,681 | 21,334 |
| v2 | 5,184 | 4,386 | ~42,000 |
| v2_conservative / v3 / v4 / v4.2 | 5,184 | 4,386 | 21,334 |
| **v4.3** | **23,333** | **23,514** | **33,378 (flanked)** |

**Key design decisions:**
- `conservative` = real ORIs only → consistently outperforms pseudo-ORI variants
  (pseudo-ORIs introduce noise that hurts precision more than they help recall)
- thresh 0.45 > thresh 0.50 for forks → more pseudo-fork coverage, better recall
- v2_conservative = best annotation set before v3
- v3 adds working flip augmentation + higher left_fork focal weight (2.0→2.5)
- v4.3 = FORTE-ForkML hybrid (see below): dramatically more pseudo-forks (thr=0.20),
  flanked-only ORIs, MaskedMeanIoU metric, val_loss monitoring, Optuna HP tuning

### v4.3 — FORTE-ForkML Hybrid Design

Key improvements over v4.2:

1. **Fork pseudo-labels at thr=0.20** (vs 0.45): 23k LF + 23k RF (vs 5k+4k)
   → 55% of ORIs are now flanked by a fork on each side (vs 12% at thr=0.45)
2. **ORI labels: flanked only** (33,378 from 9,108 reads)
   → teaches the model that ORIs always appear between a left and right fork
3. **val_loss monitoring** for checkpoint, early stopping, reduce_lr
   → avoids epoch-1-best problem caused by F1 variance inflating scores at low precision
4. **MaskedMeanIoU** added as a training metric (IoU_k = TP/(TP+FP+FN), mean over classes 1–3)
   → ForkML-style metric tracked alongside F1
5. **Optuna HP tuning** (TPE sampler + MedianPruner, 40 trials, val_loss objective)
   → final lr, batch_size, cnn_filters, dropout, gamma, alpha weights chosen by search

**Pseudo-label generation workflow for v4.3:**
```bash
# Step 1: Run v2 inference once on all reads → large predictions cache
CUDA_VISIBLE_DEVICES="" python CODEX/scripts/sweep_fork_thresholds_training.py \
    --model CODEX/models/forte_v2.keras \
    --config CODEX/configs/forte_v2.yaml \
    --output CODEX/results/forte_v4.3/pseudo_labels

# Step 2: Extract BEDs at chosen thresholds (reuse cache, skip inference)
CUDA_VISIBLE_DEVICES="" python CODEX/scripts/extract_pseudo_labels_from_cache.py \
    --predictions CODEX/results/fork_threshold_sweep_predictions.tsv \
    --source-config CODEX/configs/forte_v2.yaml \
    --output CODEX/results/forte_v4.3/pseudo_labels \
    --left-fork-thresh 0.20 --right-fork-thresh 0.20 --origin-thresh 0.50

# Step 3: Curate flanked ORIs (keep only ORIs flanked ±100kb by forks)
python CODEX/scripts/curate_flanked_oris.py \
    --oris  CODEX/results/forte_v4.3/pseudo_labels/combined_origin.bed \
    --left-forks  CODEX/results/forte_v4.3/pseudo_labels/combined_left_fork.bed \
    --right-forks CODEX/results/forte_v4.3/pseudo_labels/combined_right_fork.bed \
    --flank 100000 \
    --output CODEX/results/forte_v4.3/training_labels/flanked_origin.bed

# Step 4: HP tuning (resumable — rerun same command to add more trials)
CUDA_VISIBLE_DEVICES="" python CODEX/scripts/hp_tuning_optuna.py \
    --preprocessed CODEX/results/forte_v4.3/preprocessed_forte_v4.3.npz \
    --output-dir   CODEX/results/forte_v4.3/hp_tuning \
    --n-trials 40 --max-epochs 40 --study-name forte_v4.3_hp_v2
# Results: CODEX/results/forte_v4.3/hp_tuning/best_hps.yaml
```

---

## 6. Evaluation Protocol

### IoU-based event evaluation (primary)
Events are formed by merging adjacent windows above a probability threshold (default 0.40,
max_gap=5 kb). Each predicted event is matched one-to-one with the nearest GT event by IoU.

```
IoU = intersection / union  (per event pair)
TP = IoU ≥ threshold (default 0.2)
```

**Evaluation philosophy:** Ground truth annotations are **incomplete** — not all forks and
ORIs are annotated. A "false positive" prediction may be a real unannotated event.
**Recall is the primary metric.** Precision is a lower bound.

### Val split
All FORTE models are evaluated on the **same fixed val split** defined in
`CODEX/results/forte_v2_conservative/preprocessed_forte_v2_conservative.split_manifest.tsv`.
GT is restricted to val-split reads only (to avoid train-set annotations inflating FN count).

### Window-level F1 (secondary, training monitor)
`val_masked_f1_macro` reported during training — macro F1 across 4 classes at the window level.
Useful for comparing training curves but not directly comparable to IoU-based evaluation.
Low precision at window level is expected due to 98% background class dominance.

---

## 7. Results Summary

### IoU-based evaluation @ IoU ≥ 0.2, threshold = 0.40

#### Recall (primary metric)

| Model | left_fork | right_fork | origin | Notes |
|---|---|---|---|---|
| FORTE v1 | 0.509 | **0.702** | **0.604** | right_fork boosted by pseudo-ORIs |
| FORTE v1_conservative | 0.533 | 0.602 | 0.569 | cleaner ORIs |
| FORTE v2 | ~0.51 | ~0.66 | ~0.58 | same labels as v1, lower thresh forks |
| FORTE v2_conservative | ~0.54 | ~0.64 | ~0.59 | best balance before flip aug |
| FORTE v3 | ~0.56 | ~0.65 | ~0.60 | flip aug + symmetric forks |
| FORTE v4/v4.2 | ~0.55 | ~0.64 | ~0.59 | minor variation; v4.2 checkpoint wrong epoch |

#### F1 @ IoU ≥ 0.2

| Model | left_fork | right_fork | origin |
|---|---|---|---|
| FORTE v1 | 0.295 | 0.325 | **0.628** |
| FORTE v1_conservative | 0.274 | 0.327 | 0.612 |
| FORTE v2 / v3 / v4 | ~0.28–0.32 | ~0.30–0.33 | ~0.59–0.64 |

#### Left/right fork balance (window-level recall during training)

| Model | Left recall | Right recall | Gap | Note |
|---|---|---|---|---|
| v1 | ~0.51 | ~0.70 | 0.19 | severe left_fork under-prediction |
| v2_conservative | similar | similar | ~0.19 | flip aug not working |
| v3 (ep4) | **0.70** | **0.59** | **0.11** | flip aug fixed the imbalance |
| v3 (ep5+) | ~0.73 | ~0.69 | **~0.04** | nearly symmetric |

### Mathematical methods benchmark @ IoU ≥ 0.2 (recall)

| Method | left_fork | right_fork | origin |
|---|---|---|---|
| Gaussian gradient | ~0.00 | ~0.00 | ~0.00 |
| Wavelet gradient | 0.065 | 0.083 | 0.323 |
| GradPeak sensitive (v2) | 0.370 | 0.377 | 0.059 |
| LoG multiscale (v3) | 0.143 | 0.200 | **0.231** |
| Viterbi HMM (v3) | ~0.000 | ~0.000 | ~0.004 |
| **FORTE v1 (AI)** | **0.509** | **0.702** | **0.604** |

AI is 1.4× better on forks and ~10× better on origins vs best mathematical method.

---

## 8. Key Findings

### 1. Real ORIs only > pseudo-ORIs
Conservative variants (real annotations only) consistently outperform pseudo-ORI variants
in val F1. Pseudo-ORIs (~20k synthetic labels on top of 21k real) add noise that hurts
precision more than they improve recall.

### 2. More pseudo-forks = better (thresh 0.45 > 0.50)
Lower threshold → more pseudo-fork annotations → better fork coverage without quality loss.
The v2 annotation set (5,184 left / 4,386 right forks) is strictly better than v1.

### 3. Flip augmentation was broken in v1/v2 — fixed in v3
`flip_augment: true` was set in all configs but the code in `training.py` never
implemented it. Discovered during investigation of the left_fork under-prediction problem.
Fixed in v3: each fork-containing training read gets a reversed copy with left↔right swapped.

### 4. Left_fork under-prediction root cause: BiLSTM directional bias
FORTE v1/v2 predicted 4× fewer left_fork windows than right_fork (928 vs 3,617)
despite perfectly balanced training labels (118,068 each). The BiLSTM learned a strong
sequential cue: "origin → right_fork" (since origins are 10× more abundant than forks,
this pattern dominated gradient updates). Flip augmentation breaks this asymmetry.

### 5. Fork direction requires spatial context
No single feature (signal, wavelet, envelope, etc.) can distinguish left from right fork —
all single-feature classifiers score identically for left and right. The BiLSTM is not
optional for this task.

### 6. Mathematical methods cannot detect fork direction
Best mathematical method (GradPeak sensitive) achieves ~37% fork recall vs 50–70% for AI.
Origins are detectable mathematically (LoG ~23% recall) but AI is still ~2.6× better.

### 7. In-memory training eliminates chunk-order oscillation
Chunk streaming (v1/v2 training) reads data in fixed order each epoch, causing val F1
to oscillate ±0.15 between epochs. Loading all chunks into RAM with `shuffle=True`
(v3 onward) produces much smoother training curves.

### 8. val_loss is a better training monitor than val_f1_macro (v4.2 lesson)
In v4.2, val_loss decreased steadily from epoch 1 to epoch 11, then rose.
The checkpoint (monitoring val_f1_macro) saved epoch 1 — which had the highest F1 due
to low-precision predictions inflating recall at epoch 1. The true best weights (epoch 11)
were lost. **Lesson:** always monitor val_loss (mode=min) for checkpoint and early stopping.
This is the default from v4.3 onward.

### 9. Flanked ORI labels: good in theory, disastrous in practice
At thr=0.20, 55% of Nerea ORIs are flanked (33,378 flanked ORIs introduced at v4.3).
**BUT:** `flanked_origin.bed` has median size ~2.8 kb vs Nerea's median ~1.2 kb.
59% of Nerea's 21,334 ORIs are <2 kb — and v4.3/v4.4 detect only **~2.4%** of those.
The model trained exclusively on larger flanked labels never learned small-ORI detection.
This capability existed in v3 (which used `ORIs_combined_cleaned.bed` directly) and was
silently lost at v4.3. **v4.5 restores it by returning to Nerea's direct annotations.**

### 10. ORI label priority must give ORIs precedence over forks
With fork-first label priority, ~53% of Nerea ORIs that spatially overlap pseudo-fork
labels were silently relabeled as forks during preprocessing — the model never saw them
as ORIs. Fix (v4.5+): `origin > left_fork > right_fork > termination` in `weak_labels.py`.
Biologically justified: ORIs and forks should not overlap; if they do, the ORI annotation
is more specific.

### 10. Threshold sweep + cache is more efficient than re-running inference
Running v2 inference on all 44,798 reads once and caching the window-level probabilities
(408 MB TSV) allows sweeping multiple thresholds instantly, without re-running the model.
See `sweep_fork_thresholds_training.py` + `extract_pseudo_labels_from_cache.py`.

---

## 9. How to Run

### Prerequisites
```bash
conda activate ONT
cd /mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer
```

### Train a new model
```bash
# Edit/create a config in CODEX/configs/
nohup python CODEX/scripts/train_weak5_codex.py --config CODEX/configs/forte_v3.yaml \
    > CODEX/results/forte_v3/train.log 2>&1 &
```

### Run IoU evaluation
```bash
CUDA_VISIBLE_DEVICES="" python CODEX/scripts/evaluate_iou_models.py \
    --models "v3:CODEX/models/forte_v3.keras:CODEX/configs/forte_v3.yaml" \
    --split-manifest CODEX/results/forte_v2_conservative/preprocessed_forte_v2_conservative.split_manifest.tsv \
    --output-dir CODEX/results/iou_evaluation
```

### Plot training histories
```bash
python CODEX/scripts/plot_training_history.py \
    --logs /tmp/forte_train.log /tmp/forte_v3_train.log \
    --names "v1" "v3" \
    --output CODEX/results/training_histories/comparison.png
```

### Predict on new reads
```bash
CUDA_VISIBLE_DEVICES="" python scripts/predict_forks_and_call_oris.py \
    --model CODEX/models/forte_v3.keras \
    --config CODEX/configs/forte_v3.yaml \
    --input <reads.xy.tsv> \
    --output-dir results/predictions/
```

### Visualise entropy (confident vs uncertain ORI predictions)
```bash
CUDA_VISIBLE_DEVICES="" python CODEX/scripts/plot_entropy_oris.py \
    --model CODEX/models/forte_v1.keras \
    --config CODEX/configs/forte_v1.yaml
# Output: CODEX/results/forte_v1/entropy_ori_examples/
```

### Run HP tuning with Optuna (TPE + MedianPruner)
```bash
# First: preprocess the dataset
python CODEX/scripts/preprocess_weak4_codex.py --config CODEX/configs/forte_v4.3.yaml

# Then: run tuning (resumable — rerun same command to continue)
CUDA_VISIBLE_DEVICES="" nohup /path/to/python -u CODEX/scripts/hp_tuning_optuna.py \
    --preprocessed CODEX/results/forte_v4.3/preprocessed_forte_v4.3.npz \
    --output-dir   CODEX/results/forte_v4.3/hp_tuning \
    --n-trials 40 --max-epochs 40 --study-name forte_v4.3_hp_v2 \
    > CODEX/results/forte_v4.3/hp_tuning.log 2>&1 &
# Results: CODEX/results/forte_v4.3/hp_tuning/best_hps.yaml
#          CODEX/results/forte_v4.3/hp_tuning/hp_tuning_results.tsv
```

### Run fork threshold sweep (generate pseudo-label candidates at all thresholds)
```bash
CUDA_VISIBLE_DEVICES="" python CODEX/scripts/sweep_fork_thresholds_training.py \
    --model CODEX/models/forte_v2.keras \
    --config CODEX/configs/forte_v2.yaml \
    --output CODEX/results/forte_v4.3/pseudo_labels
# Saves: CODEX/results/fork_threshold_sweep_predictions.tsv (408 MB, reusable)
#        CODEX/results/fork_threshold_sweep_training.tsv (counts per threshold)
```

### Plot read length vs N events (ForkML-style, multi-model)
```bash
CUDA_VISIBLE_DEVICES="" python CODEX/scripts/plot_readlen_vs_features.py \
    --output-dir CODEX/results/readlen_analysis
# Outputs: readlen_vs_ori_forkml.png, readlen_vs_lf_forkml.png, readlen_vs_rf_forkml.png
#          readlen_vs_events_all_models.png  (combined grid)
#          readlen_event_counts.tsv  (summary table)
```

---

## 10. Repository Structure

```
CODEX/
├── configs/
│   ├── forte_v1.yaml                  # v1: pseudo-ORIs, thresh 0.50 forks
│   ├── forte_v1_conservative.yaml     # v1_cons: real ORIs only
│   ├── forte_v2.yaml                  # v2: pseudo-ORIs, thresh 0.45 forks
│   ├── forte_v2_conservative.yaml     # v2_cons: real ORIs only (best baseline)
│   ├── forte_v3.yaml                  # v3: v2_cons + working flip augmentation
│   ├── forte_v4.yaml                  # v4: same as v3 (minor tweak attempt)
│   ├── forte_v4.2.yaml                # v4.2: val_loss monitoring introduced
│   ├── forte_v4.3.yaml                # v4.3: FORTE-ForkML hybrid (HP tuning target)
│   ├── forte_v4.4.yaml                # v4.4: HP-tuned, balanced alphas, flanked ORIs ⚠️
│   ├── forte_v4.5.yaml                # v4.5: Nerea direct ORIs, ORI-wins priority ✅
│   └── forte_v5.yaml                  # v5: experiment with pseudo-ORIs again (deprecated)
│
├── models/
│   ├── forte_v1.keras                 # Best v1 model (epoch 9)
│   ├── forte_v1_conservative.keras    # Best v1_cons model (epoch 18)
│   ├── forte_v2.keras                 # Best v2 model (epoch 7)
│   ├── forte_v2_conservative.keras    # Best v2_cons model (epoch 21)
│   ├── forte_v3.keras                 # Best v3 model (epoch 4)
│   ├── forte_v4.keras                 # v4 model
│   ├── forte_v4.2.keras               # v4.2 model (checkpoint = epoch 1, not ideal)
│   └── forte_v5.keras                 # v5 model
│
├── replication_analyzer_codex/        # Core Python package
│   ├── training.py                    # Train loop, flip augmentation, chunk loading
│   ├── evaluation.py                  # predict_reads, windows_to_events, IoU eval
│   ├── representation.py              # 9-channel feature encoding
│   ├── losses.py                      # Focal loss + MaskedMeanIoU/F1/Precision/Recall
│   ├── weak_labels.py                 # Label assignment from BED annotations
│   ├── annotations.py                 # BED file loading
│   └── splits.py                      # Train/val split manifest
│
├── scripts/
│   ├── — Training —
│   ├── train_weak5_codex.py           # Training entry point (tees log to results dir)
│   ├── preprocess_weak4_codex.py      # Offline preprocessing → .npz tensors
│   ├── build_xy_cache_codex.py        # Build xy_data.pkl cache from raw run dirs
│   │
│   ├── — Pseudo-label generation —
│   ├── generate_pseudo_labels_forte.py # Run model inference → BED pseudo-labels
│   ├── sweep_fork_thresholds_training.py # Fork threshold sweep + flanked-ORI counts
│   ├── extract_pseudo_labels_from_cache.py # Extract BEDs from cached predictions TSV
│   ├── curate_flanked_oris.py         # Keep only ORIs flanked by forks (±flank_bp)
│   ├── sweep_flanked_oris.py          # Sweep annotation combinations for flanked ORIs
│   ├── generate_ori_validated_forks.py # ORI-validated forks (legacy, used in v2_cons)
│   │
│   ├── — HP Tuning —
│   ├── hp_tuning_optuna.py            # Optuna TPE + MedianPruner, SQLite persistence
│   │
│   ├── — Evaluation —
│   ├── evaluate_iou_models.py         # IoU evaluation across multiple models
│   ├── evaluate_fork_pairing.py       # Fork pairing post-processing evaluation
│   ├── evaluate_weak5_codex.py        # Window-level evaluation
│   ├── benchmark_signal_features.py   # Single-feature baseline benchmarks
│   ├── benchmark_mathematical_methods.py # Mathematical pipeline benchmarks
│   │
│   ├── — Visualisation —
│   ├── plot_training_history.py       # Training curve plots (smoothed)
│   ├── plot_entropy_oris.py           # High/low entropy ORI examples
│   ├── plot_read_with_predictions.py  # Single read visualiser (GT + predictions)
│   ├── plot_example_reads_codex.py    # Grid of example reads
│   ├── plot_evaluation_codex.py       # Evaluation result plots
│   ├── plot_multimodel_probs.py       # Multi-model probability comparison per read
│   ├── plot_math_vs_ai.py             # Side-by-side: GT / AI / math methods
│   ├── plot_ori_validated_examples.py # ORI-validated fork examples
│   ├── plot_readlen_vs_features.py    # ForkML-style boxplots: N events vs read length
│   ├── forte_umap_comparison.py       # UMAP in signal space + probability space
│   └── umap_windows_codex.py          # Window-level UMAP
│
└── results/
    ├── cache/xy_data.pkl              # Cached XY signal data (all reads, ~6 GB)
    ├── forte_v1/                      # v1 results, pseudo-labels, train.log, UMAP, entropy
    ├── forte_v2/                      # v2 results and pseudo-labels
    ├── forte_v2_conservative/         # v2_cons results + split manifest (shared by all)
    ├── forte_v3/                      # v3 results and train.log
    ├── forte_v4/                      # v4 results
    ├── forte_v4.2/                    # v4.2 results
    ├── forte_v4.3/                    # v4.3 (in progress)
    │   ├── pseudo_labels/             # combined_left_fork.bed, combined_right_fork.bed
    │   ├── training_labels/           # flanked_origin.bed (33,378)
    │   ├── hp_tuning/                 # Optuna DB, best_hps.yaml, hp_tuning_results.tsv
    │   └── preprocessed_forte_v4.3.npz # Preprocessed tensors (train+val)
    ├── fork_threshold_sweep_predictions.tsv  # v2 window probs on all reads (408 MB)
    ├── fork_threshold_sweep_training.tsv     # Flanked-ORI counts per threshold
    ├── readlen_analysis/              # ForkML-style N-events-vs-read-length plots
    ├── iou_evaluation/                # IoU comparison plots and summary TSV
    ├── training_histories/            # Training curve plots
    ├── mathematical_benchmark_v2/     # GradPeak results
    └── mathematical_benchmark_v3/     # LoG + HMM results
```

---

## Notes for future sessions

- **Always use the same val split manifest** (`forte_v2_conservative` split) for fair comparison
- **Recall is the primary IoU metric** — annotations are incomplete, precision is a lower bound
- **`flip_augment: true` in config is not enough** — check `training.py` if starting a new
  architecture that the flip code paths are reached
- **Chunk streaming trains well but oscillates** — use in-memory loading (`_load_chunks_to_memory`)
  when flip augmentation is active so `shuffle=True` can work per epoch
- **Train logs** are now saved to `CODEX/results/<model>/train.log` (from v3 onward);
  older models logged to `/tmp/forte_*.log`
- **IoU evaluation must filter GT to val reads** — passing full GT BED files without filtering
  inflates FN count with train-set annotations (see `evaluate_fork_pairing.py` bug fix)
- **Monitor val_masked_mean_iou not val_loss** (v4.4+) — val_masked_mean_iou better reflects
  event-level quality than loss alone. Use `mode=max`.
- **ORI label source is critical** — always use `ORIs_combined_cleaned.bed` (Nerea direct,
  21,334 ORIs, median 1.2 kb). `flanked_origin.bed` (v4.3/v4.4) has median 2.8 kb and
  caused the model to miss ~98% of sub-2kb ORIs. Do not use flanked ORIs again.
- **ORI wins label priority** — in `weak_labels.py`, `origin` must come before `left_fork`
  and `right_fork` in the priority list. ~53% of Nerea ORIs overlap pseudo-fork regions;
  with fork-first priority they were silently relabeled as forks. Fixed in v4.5.
- **lstm_units=128 is fixed** — `SelfAttention(256)` in the architecture requires BiLSTM output
  = 2 × lstm_units = 256. Only lstm_units=128 is compatible.
- **HP tuning batch_size ≥ 64** — batch_size=32 with 68k training sequences takes ~9h/trial.
  The current search space is batch_size ∈ {64, 128}.
- **Optuna study names are versioned** — `forte_v4.3_hp_v2` was created after v1 (with
  batch_size=32) was abandoned. Changing the search space requires a new study name.
- **The v2 sweep cache** (`fork_threshold_sweep_predictions.tsv`, 408 MB) contains window-
  level probabilities for all 44,798 reads. Reuse it for any threshold sweep instead of
  re-running inference.
- **CUDA_VISIBLE_DEVICES=""** — always run inference/training on CPU on this machine
  (GPU driver mismatch). Use the ONT conda env directly: `/home/jg2070/miniforge3/envs/ONT/bin/python`
