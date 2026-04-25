#!/usr/bin/env python
"""Generate pseudo-labels from a trained CODEX model for FORTE training.

Runs the model on ALL reads (train + val) and extracts high-confidence
fork and origin events.  Saves them as BED files that can be merged
with the original annotations for fully supervised FORTE training.

Usage:
  CUDA_VISIBLE_DEVICES="" python generate_pseudo_labels_forte.py \\
      --config  CODEX/configs/forte_v1.yaml \\
      --model   CODEX/models/weak5_rectangular_v4.keras \\
      --source-config CODEX/configs/weak5_rectangular_v4.yaml \\
      --output  CODEX/results/forte_v1/pseudo_labels

Output files (BED4: chr start end read_id):
  pseudo_left_fork.bed
  pseudo_right_fork.bed
  pseudo_origin.bed
  combined_left_fork.bed   (real + pseudo, deduplicated)
  combined_right_fork.bed
  combined_origin.bed
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer.models.base import SelfAttention
from replication_analyzer_codex.annotations import load_annotations_for_codex
from replication_analyzer_codex.evaluation import (
    load_xy_for_prediction,
    predict_reads,
    windows_to_events,
)
from replication_analyzer_codex.losses import (
    MaskedClassPrecision,
    MaskedClassRecall,
    MaskedMacroF1,
    SparseCategoricalFocalLoss,
)


_PSEUDO_THRESHOLDS = {
    "left_fork":  0.40,
    "right_fork": 0.45,
    "origin":     0.50,
}

_CLASS_IDS = {
    "left_fork":  1,
    "right_fork": 2,
    "origin":     3,
}


def _load_model(model_path: str):
    custom_objects = {
        "SelfAttention": SelfAttention,
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
        "MaskedMacroF1": MaskedMacroF1,
        "MaskedClassPrecision": MaskedClassPrecision,
        "MaskedClassRecall": MaskedClassRecall,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)


def _events_to_bed(events_df: pd.DataFrame) -> pd.DataFrame:
    """Return a BED4 dataframe (chr, start, end, read_id)."""
    if events_df.empty:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return events_df[["chr", "start", "end", "read_id"]].copy()


def _merge_bed(real_df: pd.DataFrame, pseudo_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate real + pseudo annotations and drop exact duplicates."""
    combined = pd.concat([real_df, pseudo_df], ignore_index=True)
    return combined.drop_duplicates(subset=["chr", "start", "end", "read_id"]).reset_index(drop=True)


def merge_overlapping_per_read(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping/adjacent intervals within each (read_id, chr) group.

    Equivalent to: bedtools sort | bedtools merge  — but applied per read_id
    so that intervals from different reads are never merged together.
    """
    if df.empty:
        return df.copy()
    rows = []
    for (read_id, chrom), group in df.groupby(["read_id", "chr"], sort=False):
        intervals = sorted(zip(group["start"].astype(int), group["end"].astype(int)))
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                rows.append({"chr": chrom, "start": cur_start, "end": cur_end, "read_id": read_id})
                cur_start, cur_end = start, end
        rows.append({"chr": chrom, "start": cur_start, "end": cur_end, "read_id": read_id})
    return pd.DataFrame(rows, columns=["chr", "start", "end", "read_id"])


def subtract_per_read(target_df: pd.DataFrame, subtract_df: pd.DataFrame) -> pd.DataFrame:
    """Remove from target any interval that overlaps subtract, per read.

    Equivalent to: bedtools subtract -a target -b subtract — but per read_id.
    Intervals are split around the subtracted region (not just removed wholesale).
    Priority: subtract wins wherever it overlaps target.
    """
    if target_df.empty or subtract_df.empty:
        return target_df.copy()
    sub_by_read = {
        read_id: list(zip(g["start"].astype(int), g["end"].astype(int)))
        for read_id, g in subtract_df.groupby("read_id")
    }
    rows = []
    for row in target_df.itertuples(index=False):
        remaining = [(int(row.start), int(row.end))]
        for s_start, s_end in sub_by_read.get(row.read_id, []):
            new_remaining = []
            for r_start, r_end in remaining:
                if s_end <= r_start or s_start >= r_end:
                    new_remaining.append((r_start, r_end))
                else:
                    if r_start < s_start:
                        new_remaining.append((r_start, s_start))
                    if s_end < r_end:
                        new_remaining.append((s_end, r_end))
            remaining = new_remaining
            if not remaining:
                break
        for r_start, r_end in remaining:
            rows.append({"chr": row.chr, "start": r_start, "end": r_end, "read_id": row.read_id})
    return pd.DataFrame(rows, columns=["chr", "start", "end", "read_id"])


def main():
    parser = argparse.ArgumentParser(description="Generate FORTE pseudo-labels from a CODEX model")
    parser.add_argument("--model", required=True, help="Path to trained .keras model")
    parser.add_argument("--source-config", required=True,
                        help="Config YAML the model was trained with (for preprocessing settings, xy_cache_path, and real annotation paths)")
    parser.add_argument("--output", required=True, help="Output directory for pseudo-label BED files")
    parser.add_argument("--left-fork-thresh", type=float, default=_PSEUDO_THRESHOLDS["left_fork"])
    parser.add_argument("--right-fork-thresh", type=float, default=_PSEUDO_THRESHOLDS["right_fork"])
    parser.add_argument("--origin-thresh", type=float, default=_PSEUDO_THRESHOLDS["origin"])
    parser.add_argument("--min-windows", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=5000)
    args = parser.parse_args()

    thresholds = {
        "left_fork":  args.left_fork_thresh,
        "right_fork": args.right_fork_thresh,
        "origin":     args.origin_thresh,
    }

    with open(args.source_config, "r", encoding="utf-8") as fh:
        src_config = yaml.safe_load(fh)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model…")
    model = _load_model(args.model)
    max_length = model.input_shape[1]
    print(f"  max_length = {max_length}")

    print("Loading XY data (from source config xy_cache_path)…")
    xy_data = load_xy_for_prediction(src_config)
    all_read_ids = xy_data["read_id"].unique().tolist()
    print(f"  Total reads: {len(all_read_ids):,}")

    print("Running predictions on all reads…")
    predictions = predict_reads(
        model=model,
        xy_data=xy_data,
        read_ids=all_read_ids,
        max_length=max_length,
        preprocessing_config=src_config["preprocessing"],
    )
    print(f"  Prediction rows: {len(predictions):,}")
    predictions_path = output_dir / "all_predictions.tsv"
    predictions.to_csv(predictions_path, sep="\t", index=False)
    print(f"  Saved: {predictions_path}")

    print("Extracting pseudo-label events…")
    pseudo_beds = {}
    for event_type, class_id in _CLASS_IDS.items():
        thresh = thresholds[event_type]
        events = windows_to_events(
            predictions=predictions,
            class_id=class_id,
            prob_threshold=thresh,
            min_windows=args.min_windows,
            max_gap=args.max_gap,
        )
        bed = _events_to_bed(events)
        pseudo_beds[event_type] = bed
        out_path = output_dir / f"pseudo_{event_type}.bed"
        bed.to_csv(out_path, sep="\t", header=False, index=False)
        print(f"  {event_type} (thresh={thresh:.2f}): {len(bed):,} pseudo-label regions → {out_path}")

    # Load real annotations from the SOURCE config (which has the original real annotation paths).
    # The forte_config will point to the combined BED files (not yet created at this stage).
    print("Loading real annotations from source config…")
    left_forks_real, right_forks_real, origins_real, _ = load_annotations_for_codex(src_config)

    real_beds = {
        "left_fork":  left_forks_real[["chr", "start", "end", "read_id"]].copy(),
        "right_fork": right_forks_real[["chr", "start", "end", "read_id"]].copy(),
        "origin":     origins_real[["chr", "start", "end", "read_id"]].copy(),
    }

    combined_beds = {}
    for event_type in _CLASS_IDS:
        combined = _merge_bed(real_beds[event_type], pseudo_beds[event_type])
        out_path = output_dir / f"combined_{event_type}.bed"
        combined.to_csv(out_path, sep="\t", header=False, index=False)
        n_real = len(real_beds[event_type])
        n_pseudo = len(pseudo_beds[event_type])
        print(f"  {event_type}: {n_real} real + {n_pseudo} pseudo = {len(combined)} combined → {out_path}")
        combined_beds[event_type] = combined

    # ── Clean combined BEDs: merge overlaps within each class, then subtract
    # higher-priority classes from lower-priority ones (origin > forks).
    # ORIs are human-annotated ground truth; forks are AI pseudo-labels that
    # may encroach on ORI regions. ORI wins wherever they overlap.
    print("\nCleaning combined BEDs (merge overlaps + subtract ORI priority)…")

    clean_lf  = merge_overlapping_per_read(combined_beds["left_fork"])
    clean_rf  = merge_overlapping_per_read(combined_beds["right_fork"])
    clean_ori = merge_overlapping_per_read(combined_beds["origin"])

    # ORI takes priority — subtract ORI regions from both fork classes
    clean_lf = subtract_per_read(clean_lf, clean_ori)
    clean_rf = subtract_per_read(clean_rf, clean_ori)
    # Merge again in case subtraction created adjacent fragments
    clean_lf = merge_overlapping_per_read(clean_lf)
    clean_rf = merge_overlapping_per_read(clean_rf)

    for event_type, clean_df, orig_df in [
        ("left_fork",  clean_lf,  combined_beds["left_fork"]),
        ("right_fork", clean_rf,  combined_beds["right_fork"]),
        ("origin",     clean_ori, combined_beds["origin"]),
    ]:
        out_path = output_dir / f"combined_{event_type}.bed"
        clean_df.to_csv(out_path, sep="\t", header=False, index=False)
        print(f"  {event_type}: {len(orig_df)} → {len(clean_df)} after clean "
              f"({len(orig_df) - len(clean_df)} removed/merged) → {out_path}")

    print("\nDone. Combined BED files are ready for FORTE preprocessing.")
    print("Next step: run preprocess_weak4_codex.py with forte_v1.yaml")


if __name__ == "__main__":
    main()
