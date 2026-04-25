#!/usr/bin/env python
"""Generate a split manifest restricted to human-annotated reads.

Loads the XY cache to get read lengths (for stratification), then filters
to only reads that appear in at least one human annotation BED. Creates a
70/20/10 (or configurable) train/val/test split manifest.

Usage (from /replication-analyzer/):
  /home/jg2070/miniforge3/envs/ONT/bin/python -u \\
      CODEX/scripts/generate_onlyhuman_split_manifest.py \\
      --lf       CODEX/results/forte_v5.0_onlyhuman/training_labels/human_left_fork_clean.bed \\
      --rf       CODEX/results/forte_v5.0_onlyhuman/training_labels/human_right_fork_clean.bed \\
      --ori      CODEX/results/forte_v5.0_onlyhuman/training_labels/human_origin_clean.bed \\
      --xy-cache CODEX/results/cache/xy_data.pkl \\
      --out      CODEX/results/forte_v5.0_onlyhuman/split_manifest.tsv \\
      --val-fraction 0.20 --test-fraction 0.10 --seed 42
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.splits import (
    build_read_metadata,
    create_split_manifest,
    save_split_manifest,
)


def load_bed4(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lf",           required=True)
    parser.add_argument("--rf",           required=True)
    parser.add_argument("--ori",          required=True)
    parser.add_argument("--xy-cache",     required=True)
    parser.add_argument("--out",          required=True)
    parser.add_argument("--val-fraction",  type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    print("Loading XY cache...")
    with open(args.xy_cache, "rb") as fh:
        xy_data = pickle.load(fh)
    print(f"  Total reads in XY cache: {xy_data['read_id'].nunique():,}")

    print("Loading human annotation BEDs...")
    lf  = load_bed4(args.lf)
    rf  = load_bed4(args.rf)
    ori = load_bed4(args.ori)
    terminations = pd.DataFrame(columns=["chr", "start", "end", "read_id"])

    human_reads = set(lf["read_id"]) | set(rf["read_id"]) | set(ori["read_id"])
    print(f"  LF:  {len(lf):,} labels on {lf['read_id'].nunique():,} reads")
    print(f"  RF:  {len(rf):,} labels on {rf['read_id'].nunique():,} reads")
    print(f"  ORI: {len(ori):,} labels on {ori['read_id'].nunique():,} reads")
    print(f"  Union (any human annotation): {len(human_reads):,} reads")

    # Filter XY data to human-annotated reads only
    xy_human = xy_data[xy_data["read_id"].isin(human_reads)].copy()
    print(f"  XY data rows after filtering: {len(xy_human):,}")

    print("\nBuilding read metadata and split manifest...")
    metadata = build_read_metadata(xy_human, lf, rf, ori, terminations)
    print(f"  Reads in metadata: {len(metadata):,}")
    print(f"  Reads with any event: {metadata['has_any_event'].sum():,}")

    manifest = create_split_manifest(
        metadata=metadata,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.seed,
    )

    counts = manifest["split"].value_counts()
    print(f"\nSplit counts:")
    for split in ["train", "val", "test"]:
        n = counts.get(split, 0)
        pct = 100 * n / len(manifest)
        print(f"  {split:5s}: {n:,} reads ({pct:.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_split_manifest(manifest, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
