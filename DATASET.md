# Training Dataset

This document describes the data used to train FORTE (v5.x series).

---

## Biological system

**Organism:** *Arabidopsis thaliana*  
**Technique:** BrdU pulse-labelling of nascent DNA + Oxford Nanopore sequencing (ONT)  
**Signal:** BrdU incorporation fraction per genomic window — proxy for replication activity

**Genotypes:**

| Run ID | Genotype | Description |
|--------|----------|-------------|
| NM30 — run 1 | Col-0 | Wild-type |
| NM30 — run 2 | Col-0 | Wild-type (replicate) |
| NM31 — run 1 | *orc1b orc2* | ORC mutant with reduced origin firing |
| NM31 — run 2 | *orc1b orc2* | ORC mutant (replicate) |

The model is trained jointly on both genotypes to learn features that generalise across origin usage levels.

---

## Reads

**QC filters (applied before any training):**
- **Minimum read length:** 30 kb — removes reads too short for meaningful segmentation
- **Minimum nascent DNA fraction:** 40% — keeps only actively replicating reads; reads below this have too little BrdU signal

The 44,798 reads are **all reads passing both filters** across the 4 sequencing runs — no subsampling was applied.

| Run | Genotype | Reads |
|-----|----------|-------|
| NM30 — run 1 | Col-0 | 15,936 |
| NM30 — run 2 | Col-0 | 8,409 |
| NM31 — run 1 | *orc1b orc2* | 12,849 |
| NM31 — run 2 | *orc1b orc2* | 7,604 |
| **Total** | | **44,798** |

| Split | Reads | Windows |
|-------|-------|---------|
| Train | 31,358 | ~2.5 M |
| Val | 8,960 | ~0.7 M |
| Test | 4,480 | ~0.3 M |
| **Total** | **44,798** | **~3.05 M** |

Splits are stratified by event content (read-level label combinations) and kept fixed across v5.x versions via `split_manifest.tsv`.

**Read length distribution:**

| Stat | Windows per read | Approx. genomic length |
|------|-----------------|----------------------|
| Min | 20 | ~20 kb |
| Median | ~55 | ~55 kb |
| Mean | 68 | ~68 kb |
| Max | 429 | ~430 kb |

Each window spans approximately **~1 kb** of genomic sequence (variable, determined by DNAscent binning).

**Reads with annotated events:**

| Reads with… | Count | % of total |
|-------------|-------|-----------|
| Any event (LF, RF, or ORI) | 22,919 | 51% |
| Left fork | 14,553 | 32% |
| Right fork | 15,692 | 35% |
| Origin | 14,637 | 33% |
| Background only | 21,879 | 49% |

---

## Labels

FORTE uses a **4-class per-window segmentation** scheme:

| Class | Label | Source |
|-------|-------|--------|
| Background | 0 | Inferred (unlabelled windows) |
| Left fork | 1 | Pseudo-annotation (see below) |
| Right fork | 2 | Pseudo-annotation (see below) |
| Origin | 3 | Human annotation (Nerea) |

### Human-annotated origins (class 3)

Manually curated by Nerea across all 5 *Arabidopsis* chromosomes plus organellar DNA.

| Stat | Value |
|------|-------|
| Total ORIs | 21,311 |
| Chromosomes | Chr1–5, ChrC, ChrM |
| Size range | 91 bp – 169 kb |
| Median size | ~1.2 kb |
| 25th percentile | 753 bp |
| 75th percentile | 6.9 kb |
| ORIs < 1 kb | 9,443 (44%) |
| ORIs < 2 kb | 12,620 (59%) |

**Chromosome distribution:**

| Chr | ORI count |
|-----|----------|
| Chr1 | 5,104 |
| Chr2 | 3,763 |
| Chr3 | 4,216 |
| Chr4 | 3,587 |
| Chr5 | 4,480 |
| ChrC | 115 |
| ChrM | 46 |

BED file: `CODEX/results/forte_v5.1/training_labels/human_origin_clean.bed`  
Cleaning: overlapping annotations merged, ambiguous boundaries removed (`trusted_negative_margin_bp=0`).

### Pseudo-annotated forks (classes 1 & 2)

Fork labels are derived from earlier FORTE model predictions (v4.5), not from manual annotation. They are called "pseudo" because they are AI-predicted, not human-curated.

**Generation pipeline:**
1. Train FORTE v4.x on fork labels from DNAscent (2-class fork detector)
2. Run v4.5 predictions on all 44,798 reads at prob threshold 0.4
3. Merge consecutive high-probability windows → fork events
4. Filter: min length 1 bp, non-overlapping per strand

| Stat | Left fork | Right fork |
|------|----------|-----------|
| Total events | 20,249 | 20,760 |
| Size range | 1 bp – 77 kb | 1 bp – 74 kb |
| Mean size | ~16 kb | ~18 kb |

BED files: `CODEX/results/forte_v5.1/training_labels/combined_left_fork_clean.bed`  
&emsp;&emsp;&emsp;&emsp;&emsp; `CODEX/results/forte_v5.1/training_labels/combined_right_fork_clean.bed`

**Caveat:** Pseudo labels inherit the errors of the v4.5 fork detector. Reads where v4.5 was confident are well-labelled; reads with ambiguous signal may have incorrect fork boundaries. Human vs pseudo GT separation is tracked throughout evaluation (see threshold heatmaps in [FORTE_PROGRESS.md](FORTE_PROGRESS.md)).

---

## Signal encoding (9 channels)

Each read is converted to a fixed-window tensor before training. The input to the model is a `(max_windows, 9)` array per read, padded to the dataset maximum.

**Signal representation:** Rectangular block expansion — raw BrdU fractions are expanded to per-base step functions, then encoded.

**Wavelet decomposition (db4, level 2) + statistical features:**

| Channel | Feature | Captures |
|---------|---------|---------|
| 0 | Normalized signal | Absolute BrdU level |
| 1 | Wavelet approximation (low-freq) | Broad replication domains |
| 2 | Wavelet detail 1 (high-freq) | Sharp transitions at fork tips |
| 3 | Wavelet detail 2 (high-freq) | Fine-scale step edges |
| 4 | Local mean (50-window) | Regional background level |
| 5 | Local std (50-window) | Local signal variability |
| 6 | Z-score (deviation from local mean) | Relative enrichment |
| 7 | Cumulative sum | Trend detection |
| 8 | Signal envelope | Peak structure |

Config keys: `signal_mode: "wavelet"`, `wavelet: "db4"`, `wavelet_level: 2`, `use_rectangular_blocks: true`

Preprocessed tensors are cached as `.npz` (~119 MB) to skip re-encoding on restarts:  
`CODEX/results/forte_v5.2/preprocessed_forte_v5.2.npz`

---

## Class imbalance and sampling

Background windows dominate by a large margin (most of each read has no annotated event). FORTE addresses this at two levels:

**1. Focal loss** — down-weights the gradient from easy background predictions:
- alpha = [1.0, 2.2, 2.2, 5.0] (background, LF, RF, ORI)
- gamma = 2.087 (HP-tuned)

**2. Read-level oversampling** — reads with annotated events are duplicated during each epoch:
- `oversample_ratio = 0.5` — event reads get 50% extra copies
- `small_ori_oversample_ratio = 2.0` — reads with ORIs ≤ 2 kb get an additional 2× (v5.2+)
- `flip_augment = true` — each read is also added in reverse complement

Oversampling is applied per training partition; val and test sets are not oversampled.

---

## Evaluation agreement with human annotator

On the test set (4,480 reads), using IoU ≥ 0.1 matching between AI predictions and Nerea's BED:

| Class | Nerea recovery | AI-only events | Nerea-only events |
|-------|--------------|--------------|-----------------|
| Left fork | 95.3% | — | — |
| Right fork | 97.8% | — | — |
| Origin | 75.4% | median 3,641 bp | median 750 bp |

The ORI gap is driven by short ORIs (<1 kb): 44% of Nerea ORIs are below 1 kb, yet these span only ~5 model windows — the primary remaining challenge. See [FORTE_PROGRESS.md](FORTE_PROGRESS.md) for the full recall diagnostics.

---

*For model architecture and training details, see [FORTE_PROGRESS.md](FORTE_PROGRESS.md) and [CODEX/README.md](CODEX/README.md).*
