# Case Study: January 2026 Fork Analysis

> **Real-world example**: Training fork detection models on new collaborator data

## 📋 Overview

**Date**: January 2026
**Dataset**: New fork annotations for Col0 (wild-type) and orc1b2 (mutant)
**Objective**: Train directional fork detection models to analyze replication patterns
**Status**: ✅ In Progress

---

## 🎯 Project Goals

1. Train separate fork detection models for **Col0** and **orc1b2** genotypes
2. Compare fork patterns between wild-type and mutant
3. Generate BED file annotations for genome browser visualization
4. Provide reproducible analysis pipeline for collaborators

---

## 📊 Dataset Summary

### **Step 1: Dataset Discovery** ✅

**Location**: `/mnt/ssd-8tb/crisanto_project/data_2025Oct/annotation_2026January/`

**Fork Annotations Found**:

| File | Genotype | Fork Type | Count | Size |
|------|----------|-----------|-------|------|
| `leftForks_DNAscent_Col0.bed` | Col0 (WT) | Left | 229 | 22 KB |
| `rightForks_DNAscent_Col0.bed` | Col0 (WT) | Right | 217 | 21 KB |
| `leftForks_DNAscent_orc1b2.bed` | orc1b2 (mutant) | Left | 526 | 45 KB |
| `rightForks_DNAscent_orc1b2.bed` | orc1b2 (mutant) | Right | 648 | 55 KB |

**Key Observations**:
- ✅ **Col0**: 446 total forks (229 left + 217 right) - balanced distribution
- ⚠️ **orc1b2**: 1,174 total forks (526 left + 648 right) - **2.6× more forks than WT!**
- 🔬 **Biological insight**: ORC1B2 mutation appears to affect fork dynamics significantly

**XY Signal Data Available**:

| Genotype | Run 1 | Run 2 |
|----------|-------|-------|
| **Col0** | `NM30_Col0/NM30_plot_data_1strun_xy/` | `NM30_Col0/NM30_plot_data_2ndrun_xy/` |
| **orc1b2** | `NM31_orc1b2/NM31_plot_data_1strun_xy/` | `NM31_orc1b2/NM31_plot_data_2ndrun_xy/` |

**Data Format Check**:

```bash
# Example fork annotation (BED format with 8 columns)
Chr1  25045266  25065917  c35251c5...  Chr1  25044912  25144163  fwd
│     │         │         │            │     │         │         │
│     │         │         │            │     │         │         └─ Strand
│     │         │         │            │     │         └─ Read end
│     │         │         │            │     └─ Read start
│     │         │         │            └─ Chromosome (read)
│     │         │         └─ Read ID
│     │         └─ Fork end position
│     └─ Fork start position
└─ Chromosome
```

---

## 🗂️ Data Organization

### Directory Structure Created:

```
replication-analyzer/
├── data/
│   └── case_study_jan2026/
│       ├── Col0/
│       │   ├── annotations/
│       │   │   ├── leftForks_DNAscent_Col0.bed
│       │   │   └── rightForks_DNAscent_Col0.bed
│       │   └── xy_data/  (symlink to original)
│       └── orc1b2/
│           ├── annotations/
│           │   ├── leftForks_DNAscent_orc1b2.bed
│           │   └── rightForks_DNAscent_orc1b2.bed
│           └── xy_data/  (symlink to original)
├── configs/
│   ├── case_study_col0_forks.yaml
│   └── case_study_orc1b2_forks.yaml
├── models/
│   └── case_study_jan2026/
│       ├── col0_fork_detector.keras  (to be created)
│       └── orc1b2_fork_detector.keras  (to be created)
└── results/
    └── case_study_jan2026/
        ├── col0/  (to be created)
        └── orc1b2/  (to be created)
```

---

## 📝 Analysis Plan

### Phase 1: Data Preparation ⏳
- [x] Discover and inspect new fork annotations
- [ ] Copy/symlink data to project structure
- [ ] Create configuration files for both genotypes
- [ ] Verify data integrity

### Phase 2: Model Training 📚
- [ ] Train Col0 fork detection model
- [ ] Train orc1b2 fork detection model
- [ ] Monitor training metrics (F1-score, loss)
- [ ] Save best models

### Phase 3: Evaluation 📊
- [ ] Evaluate Col0 model performance
- [ ] Evaluate orc1b2 model performance
- [ ] Compare metrics between genotypes
- [ ] Generate evaluation plots

### Phase 4: Comparative Analysis 🔬
- [ ] Compare fork distributions (left vs right)
- [ ] Analyze regional patterns (centromere vs arms)
- [ ] Statistical comparison (WT vs mutant)
- [ ] Generate publication-ready figures

### Phase 5: Deliverables 📦
- [ ] Export BED files for genome browsers
- [ ] Generate summary report
- [ ] Create shareable visualizations
- [ ] Document findings in Wiki

---

## 🔗 Related Wiki Pages

- [[Step 1: Data Preparation]]
- [[Step 2: Configuration Setup]]
- [[Step 3: Training Models]]
- [[Step 4: Model Evaluation]]
- [[Step 5: Results & Interpretation]]

---

## 📌 Quick Links

- **Configuration**: [`configs/case_study_col0_forks.yaml`](../configs/case_study_col0_forks.yaml)
- **Training Script**: [`scripts/train_fork_model.py`](../scripts/train_fork_model.py)
- **Evaluation Script**: [`scripts/evaluate_model.py`](../scripts/evaluate_model.py)

---

**Last Updated**: 2026-01-03
**Status**: Data discovery complete, proceeding to data preparation
