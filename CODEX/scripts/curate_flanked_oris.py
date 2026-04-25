#!/usr/bin/env python
"""Curate training ORI labels to only those flanked by fork annotations.

Biological constraint: keep only ORIs that have
  - a left_fork annotation ending within --flank-kb upstream of ORI start
  - a right_fork annotation starting within --flank-kb downstream of ORI end

Input BED files are training annotation BEDs (real or pseudo-labels).
No probability threshold needed — these are already called events.

Usage:
  python curate_flanked_oris.py \\
      --ori-bed     CODEX/results/forte_v2/pseudo_labels/combined_origin.bed \\
      --left-bed    CODEX/results/forte_v2/pseudo_labels/combined_left_fork.bed \\
      --right-bed   CODEX/results/forte_v2/pseudo_labels/combined_right_fork.bed \\
      --output-bed  CODEX/results/forte_v5/training_labels/combined_origin_flanked.bed \\
      --flank-kb    100 \\
      --report      CODEX/results/forte_v5/training_labels/curation_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def load_bed(path: str, min_cols: int = 4) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    cols = ["chr", "start", "end", "read_id"] + [f"col{i}" for i in range(4, len(df.columns))]
    df.columns = cols[:len(df.columns)]
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    return df


def apply_flanking_filter(
    ori_df: pd.DataFrame,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    flank_bp: int,
) -> pd.DataFrame:
    """Return ori_df with added 'flanked' boolean column."""
    lf_by_read = {rid: grp for rid, grp in left_df.groupby("read_id")} \
        if len(left_df) > 0 else {}
    rf_by_read = {rid: grp for rid, grp in right_df.groupby("read_id")} \
        if len(right_df) > 0 else {}

    flanked_flags = []
    for _, ori in ori_df.iterrows():
        rid = ori["read_id"]
        ori_start = int(ori["start"])
        ori_end   = int(ori["end"])

        # Left fork: its end should be within flank_bp of ORI start
        has_left = False
        if rid in lf_by_read:
            lf = lf_by_read[rid]
            cands = lf[(lf["end"] >= ori_start - flank_bp) &
                       (lf["end"] <= ori_start + flank_bp)]
            has_left = len(cands) > 0

        # Right fork: its start should be within flank_bp of ORI end
        has_right = False
        if rid in rf_by_read:
            rf = rf_by_read[rid]
            cands = rf[(rf["start"] >= ori_end - flank_bp) &
                       (rf["start"] <= ori_end + flank_bp)]
            has_right = len(cands) > 0

        flanked_flags.append(has_left and has_right)

    ori_df = ori_df.copy()
    ori_df["flanked"] = flanked_flags
    return ori_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori-bed",    required=True)
    parser.add_argument("--left-bed",   required=True)
    parser.add_argument("--right-bed",  required=True)
    parser.add_argument("--output-bed", required=True)
    parser.add_argument("--flank-kb",   type=float, default=100.0)
    parser.add_argument("--report",     default=None)
    args = parser.parse_args()

    flank_bp = int(args.flank_kb * 1000)

    print(f"Loading ORI BED:        {args.ori_bed}")
    ori_df   = load_bed(args.ori_bed)
    print(f"Loading left-fork BED:  {args.left_bed}")
    left_df  = load_bed(args.left_bed)
    print(f"Loading right-fork BED: {args.right_bed}")
    right_df = load_bed(args.right_bed)

    print(f"\nInput ORIs:        {len(ori_df):,}")
    print(f"Left fork annots:  {len(left_df):,}")
    print(f"Right fork annots: {len(right_df):,}")
    print(f"Flank window:      ±{args.flank_kb} kb")

    print("\nApplying flanking filter...")
    ori_labeled = apply_flanking_filter(ori_df, left_df, right_df, flank_bp)

    ori_flanked   = ori_labeled[ori_labeled["flanked"]].copy()
    ori_unflanked = ori_labeled[~ori_labeled["flanked"]].copy()

    n_total    = len(ori_df)
    n_flanked  = len(ori_flanked)
    n_unflanked = len(ori_unflanked)
    pct = 100 * n_flanked / max(n_total, 1)

    print(f"\nFlanked ORIs:   {n_flanked:,}  ({pct:.1f}%)")
    print(f"Unflanked ORIs: {n_unflanked:,}  ({100-pct:.1f}%)")

    # Per-read stats
    reads_with_oris    = ori_df["read_id"].nunique()
    reads_with_flanked = ori_flanked["read_id"].nunique() if n_flanked > 0 else 0
    print(f"\nReads with any ORI:      {reads_with_oris:,}")
    print(f"Reads with flanked ORI:  {reads_with_flanked:,}")

    # Save filtered BED (preserve all original columns)
    out_path = Path(args.output_bed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [c for c in ori_flanked.columns if c != "flanked"]
    ori_flanked[out_cols].to_csv(out_path, sep="\t", header=False, index=False)
    print(f"\nSaved flanked ORI BED → {out_path}  ({n_flanked:,} entries)")

    # Report
    report_lines = [
        "FLANKED-ORI TRAINING CURATION REPORT",
        "=" * 50,
        f"ORI BED:          {args.ori_bed}",
        f"Left-fork BED:    {args.left_bed}",
        f"Right-fork BED:   {args.right_bed}",
        f"Flank window:     ±{args.flank_kb} kb",
        "",
        f"Input ORIs:       {n_total:,}",
        f"Left fork annots: {len(left_df):,}",
        f"Right fork annots:{len(right_df):,}",
        "",
        f"Flanked ORIs:     {n_flanked:,}  ({pct:.1f}%)",
        f"Unflanked ORIs:   {n_unflanked:,}  ({100-pct:.1f}%)",
        "",
        f"Reads with any ORI:      {reads_with_oris:,}",
        f"Reads with flanked ORI:  {reads_with_flanked:,}",
        "",
        f"Output BED: {out_path}",
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    if args.report:
        rpath = Path(args.report)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        with open(rpath, "w") as f:
            f.write(report_text + "\n")
        print(f"Report saved → {rpath}")


if __name__ == "__main__":
    main()
