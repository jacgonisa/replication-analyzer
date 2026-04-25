#!/usr/bin/env bash
# Prepare training labels and split manifest for FORTE v5.0-onlyhuman.
#
# All three label sources are Nerea's human annotations.
# Only the 15,339 reads with at least one human label are included.
# Split: 70% train / 20% val / 10% test.
#
# Run from /replication-analyzer/:
#   bash CODEX/scripts/prepare_onlyhuman_labels.sh

set -euo pipefail

REPO=/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer
PYTHON=/home/jg2070/miniforge3/envs/ONT/bin/python

OUT_DIR="$REPO/CODEX/results/forte_v5.0_onlyhuman/training_labels"
mkdir -p "$OUT_DIR"

LF_HUMAN="$REPO/data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed"
RF_HUMAN="$REPO/data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed"
ORI_HUMAN="$REPO/data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed"

echo "=== FORTE v5.0-onlyhuman label preparation ==="
echo "LF  (Nerea human forks): $LF_HUMAN"
echo "RF  (Nerea human forks): $RF_HUMAN"
echo "ORI (Nerea human ORIs):  $ORI_HUMAN"
echo ""

echo "Cleaning all three label classes..."
$PYTHON "$REPO/CODEX/scripts/clean_annotation_beds.py" \
    --lf     "$LF_HUMAN" \
    --rf     "$RF_HUMAN" \
    --ori    "$ORI_HUMAN" \
    --out-lf  "$OUT_DIR/human_left_fork_clean.bed" \
    --out-rf  "$OUT_DIR/human_right_fork_clean.bed" \
    --out-ori "$OUT_DIR/human_origin_clean.bed"

echo ""
echo "Label counts:"
wc -l "$OUT_DIR/human_left_fork_clean.bed"
wc -l "$OUT_DIR/human_right_fork_clean.bed"
wc -l "$OUT_DIR/human_origin_clean.bed"

echo ""
echo "Generating 70/20/10 split manifest (human-annotated reads only)..."
$PYTHON "$REPO/CODEX/scripts/generate_onlyhuman_split_manifest.py" \
    --lf      "$OUT_DIR/human_left_fork_clean.bed" \
    --rf      "$OUT_DIR/human_right_fork_clean.bed" \
    --ori     "$OUT_DIR/human_origin_clean.bed" \
    --xy-cache "$REPO/CODEX/results/cache/xy_data.pkl" \
    --out     "$REPO/CODEX/results/forte_v5.0_onlyhuman/split_manifest.tsv" \
    --val-fraction  0.20 \
    --test-fraction 0.10 \
    --seed 42

echo ""
echo "Done → $OUT_DIR"
