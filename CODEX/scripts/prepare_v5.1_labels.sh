#!/usr/bin/env bash
# Prepare training labels for FORTE v5.1.
#
# Key change vs v5.0:
#   ORI labels = Nerea's human annotations (ORIs_combined_cleaned.bed)
#   Fork labels = v5.0 pseudo labels (unchanged — human forks cover only ~1,800 reads)
#
# Only the ORI BED needs cleaning (subtract fork regions that overlap with Nerea's ORIs).
# The fork BEDs are reused directly from v5.0 (already clean).
#
# Run from /replication-analyzer/:
#   bash CODEX/scripts/prepare_v5.1_labels.sh

set -euo pipefail

REPO=/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer
PYTHON=/home/jg2070/miniforge3/envs/ONT/bin/python

OUT_DIR="$REPO/CODEX/results/forte_v5.1/training_labels"
mkdir -p "$OUT_DIR"

LF_CLEAN="$REPO/CODEX/results/forte_v5.0/training_labels/combined_left_fork_clean.bed"
RF_CLEAN="$REPO/CODEX/results/forte_v5.0/training_labels/combined_right_fork_clean.bed"
NEREA_ORI="$REPO/data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed"

echo "=== FORTE v5.1 label preparation ==="
echo "LF  (reused from v5.0): $LF_CLEAN"
echo "RF  (reused from v5.0): $RF_CLEAN"
echo "ORI (Nerea human):      $NEREA_ORI"
echo ""

# Copy fork labels directly (no re-cleaning needed — forks have priority over ORI,
# so ORI changes don't affect fork labels)
cp "$LF_CLEAN" "$OUT_DIR/combined_left_fork_clean.bed"
cp "$RF_CLEAN" "$OUT_DIR/combined_right_fork_clean.bed"
echo "Copied fork labels to $OUT_DIR"

# Clean ORI labels: subtract pseudo fork regions from Nerea's human ORIs
# (fork priority rule — same as v5.0 cleaning step)
echo ""
echo "Cleaning ORI labels (subtracting fork regions from Nerea ORIs)..."
$PYTHON "$REPO/CODEX/scripts/clean_annotation_beds.py" \
    --lf  "$LF_CLEAN" \
    --rf  "$RF_CLEAN" \
    --ori "$NEREA_ORI" \
    --out-lf  /dev/null \
    --out-rf  /dev/null \
    --out-ori "$OUT_DIR/human_origin_clean.bed"

echo ""
echo "Label counts:"
wc -l "$OUT_DIR/combined_left_fork_clean.bed"
wc -l "$OUT_DIR/combined_right_fork_clean.bed"
wc -l "$OUT_DIR/human_origin_clean.bed"
echo ""
echo "Done → $OUT_DIR"
