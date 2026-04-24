# FORTE — Model Version History & Progress

FORTE (Fork and ORigin Tracking Engine) is the CNN-BiLSTM+SelfAttention model
for 4-class segmentation of nascent-strand BrdU sequencing reads.

**Classes:** background (0), left fork (1), right fork (2), origin (3)  
**Architecture:** 9-channel wavelet + rectangular-block encoding → CNN → BiLSTM → Self-Attention → per-window softmax  
**Labels:** human ORI annotations (Nerea) + pseudo fork labels derived from earlier 2-class model

---

## v5.x Series — Clean Labels Era (Jan 2026 →)

v5.x uses a unified training framework (`CODEX/replication_analyzer_codex/`) with
experiment-specific YAML configs (`CODEX/configs/forte_v5.*.yaml`).

### v5.0 — Baseline clean labels
- First version trained exclusively on human ORI annotations (no pseudo ORIs)
- ORI labels: Nerea's manual BED file cleaned + trust-margin filtering
- Fork labels: pseudo forks generated from v4.5 predictions
- Monitor: `val_event_f1_ori_lf_rf` (0.5×F1_ORI + 0.25×F1_LF + 0.25×F1_RF)
- Best result: ORI recall ≈ 0.77 on validation

### v5.1 — Improved fork labels + HP-tuned hyperparameters
- Fork labels regenerated with cleaner BED (non-overlapping, min-length filtered)
- Hyperparameters tuned with Optuna: lr=0.002449, cnn_filters=64, lstm_units=128,
  dropout=0.3863, batch_size=64, focal loss gamma=2.087, alpha=[1.0, 2.2, 2.2, 5.0]
- **Best saved model:** epoch 25, val_event_f1_ori_lf_rf=0.797, ORI recall=0.823
- ORI recall ceiling at prob=0.3: ~60% on test (structural, not threshold problem)
  - 84% of Nerea ORIs have some pixel overlap with AI; only 16% truly zero overlap
  - Missed ORIs are mostly small (<1 kb), model draws boundaries too broadly

### v5.2 — Small-ORI oversampling + LR warmup (Apr 2026)
- **New:** reads with ORIs ≤ 2 kb get 2× extra copies per epoch (`small_ori_oversample_ratio=2.0`)
  Diagnosis: 72% of 500–1 kb ORIs had some AI overlap but low IoU — boundary, not detection problem
- **New:** 5-epoch linear LR warmup (10% → 100% of hp-tuned LR=0.002449)
  v5.1 showed large early spikes; warmup suppresses initial instability
- Monitor: same `val_event_f1_ori_lf_rf` as v5.1
- Preprocessed tensors saved to `.npz` cache for instant reload (≈119 MB, skips 1–2 h encoding)
- **Best saved model:** epoch 25, val_event_f1_ori_lf_rf=0.797, ORI recall=0.823
  - Issue: epoch 33 had ORI recall=0.844 but was discarded by F1-based monitor

### v5.3 — Recall-weighted monitor (Apr 2026, training in progress)
- Loads preprocessed tensors from v5.2 `.npz` cache (no re-encoding)
- **New monitor:** `val_event_rec_weighted = 0.5×rec_ORI + 0.25×iou_LF + 0.25×iou_RF`
  Rationale: directly saves the epoch with best ORI recall; forks judged by IoU (boundary
  quality), not F1 — avoids precision/recall tradeoff that penalised high-recall epochs in v5.2
- Architecture and labels identical to v5.2
- Early training (epochs 1–11): best rec_weighted=0.787 at epoch 4 (ORI recall=0.808)
  Training still ongoing (patience=25, max 150 epochs)

---

## v4.x Series — Pseudo ORI Era (Nov 2025 – Jan 2026)

| Version | Key change | Best val ORI IoU |
|---------|-----------|-----------------|
| v4.0 | First 4-class model, pseudo ORIs from 2-class predictions | ~0.42 |
| v4.2 | Increased focal loss ORI weight (alpha=5.0) | ~0.46 |
| v4.3 | Bidirectional LSTM + self-attention | ~0.49 |
| v4.4 | Wavelet features (db4, level 2) | ~0.53 |
| v4.5 | Rectangular block encoding (9 channels) | ~0.55 |

v4.5 fork predictions were used to generate the pseudo fork labels for v5.x.

---

## Key Architectural Choices (fixed since v4.3)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Encoding | 9-channel rectangular + wavelet | Best boundary detection for fork tips |
| CNN | 64 filters, kernel 7 | HP-tuned |
| BiLSTM | 128 units | HP-tuned |
| Self-attention | 1 head, 64 dim | Captures long-range read context |
| Loss | Focal (gamma=2.087, alpha=[1,2.2,2.2,5]) | Corrects class imbalance; ORI upweighted |
| Optimizer | Adam, lr=0.002449, clipnorm=1.0 | HP-tuned |

## ORI Recall Diagnostics (v5.1 analysis)

- **Event-level recall ceiling:** ~60% at prob=0.3 (not improvable by threshold alone)
- **Size breakdown:**
  - ORIs > 5 kb: 95–97% detected (v5.1 solved this)
  - ORIs 1–5 kb: ~80% detected, moderate IoU
  - ORIs < 1 kb: ~55% detected, low IoU (draws events too broadly)
- **Fix in v5.2/v5.3:** oversample small-ORI reads to give the model more gradient
  updates for tight boundary discrimination around short origins

## Reannotation Thresholds (v5.1 final)

All classes use prob=0.3 (same threshold for consistency):
- Fork IoU barely changes from 0.3→0.6 but recall drops significantly
- ORI: recall-maximising threshold is 0.3
- Post-prediction: max_gap=5000 bp, min_windows=1

## AI vs Nerea Agreement (v5.1 test set)

| Class | Nerea total | AI total | Both | AI-only | Nerea-only | Recovery |
|-------|------------|----------|------|---------|-----------|---------|
| Left fork | — | — | — | — | — | 95.3% |
| Right fork | — | — | — | — | — | 97.8% |
| ORI | — | — | — | — | — | 75.4% |

ORI recovery = fraction of Nerea ORIs matched by AI at IoU≥0.1.
AI-only ORIs (potential FPs): median size 3641 bp, mean prob=0.41.
Nerea-only missed ORIs: median size 750 bp (small-ORI problem).
