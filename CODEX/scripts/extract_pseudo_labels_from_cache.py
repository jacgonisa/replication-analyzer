#!/usr/bin/env python
"""Extract pseudo-label BEDs from cached predictions TSV (skip re-running inference).

Usage (from /replication-analyzer/):
  python CODEX/scripts/extract_pseudo_labels_from_cache.py \
      --predictions CODEX/results/fork_threshold_sweep_predictions.tsv \
      --source-config CODEX/configs/forte_v2.yaml \
      --output CODEX/results/forte_v4.3/pseudo_labels \
      --left-fork-thresh 0.20 \
      --right-fork-thresh 0.20 \
      --origin-thresh 0.50
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.evaluation import windows_to_events
from replication_analyzer_codex.annotations import load_annotations_for_codex

_CLASS_IDS = {"left_fork": 1, "right_fork": 2, "origin": 3}


def events_to_bed(events_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(events_df, pd.DataFrame) and not events_df.empty:
        return events_df[["chr", "start", "end", "read_id"]].copy()
    return pd.DataFrame(columns=["chr", "start", "end", "read_id"])


def merge_bed(real: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([real, pseudo], ignore_index=True)
    return combined.drop_duplicates(subset=["chr", "start", "end", "read_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",       required=True)
    parser.add_argument("--source-config",     required=True)
    parser.add_argument("--output",            required=True)
    parser.add_argument("--left-fork-thresh",  type=float, default=0.20)
    parser.add_argument("--right-fork-thresh", type=float, default=0.20)
    parser.add_argument("--origin-thresh",     type=float, default=0.50)
    parser.add_argument("--min-windows",       type=int, default=1)
    parser.add_argument("--max-gap",           type=int, default=5000)
    args = parser.parse_args()

    thresholds = {
        "left_fork":  args.left_fork_thresh,
        "right_fork": args.right_fork_thresh,
        "origin":     args.origin_thresh,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.source_config) as f:
        src_config = yaml.safe_load(f)

    print(f"Loading cached predictions: {args.predictions}")
    predictions = pd.read_csv(args.predictions, sep="\t")
    print(f"  {len(predictions):,} rows, {predictions['read_id'].nunique():,} reads")

    print("\nExtracting pseudo-label events...")
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
        bed = events_to_bed(events)
        pseudo_beds[event_type] = bed
        out_path = out_dir / f"pseudo_{event_type}.bed"
        bed.to_csv(out_path, sep="\t", header=False, index=False)
        print(f"  {event_type} (thresh={thresh:.2f}): {len(bed):,} → {out_path}")

    print("\nLoading real annotations...")
    lf_real, rf_real, ori_real, _ = load_annotations_for_codex(src_config)
    real_beds = {
        "left_fork":  lf_real[["chr", "start", "end", "read_id"]].copy(),
        "right_fork": rf_real[["chr", "start", "end", "read_id"]].copy(),
        "origin":     ori_real[["chr", "start", "end", "read_id"]].copy(),
    }

    print("\nMerging real + pseudo...")
    for event_type in _CLASS_IDS:
        combined = merge_bed(real_beds[event_type], pseudo_beds[event_type])
        out_path = out_dir / f"combined_{event_type}.bed"
        combined.to_csv(out_path, sep="\t", header=False, index=False)
        n_r = len(real_beds[event_type])
        n_p = len(pseudo_beds[event_type])
        print(f"  {event_type}: {n_r:,} real + {n_p:,} pseudo = {len(combined):,} combined → {out_path}")


if __name__ == "__main__":
    main()
