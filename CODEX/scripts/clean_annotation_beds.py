#!/usr/bin/env python
"""Clean per-read BED annotation files so they are mutually exclusive and non-redundant.

Two operations (equivalent to bedtools sort|merge and bedtools subtract, but applied
per read_id so intervals from different reads are never incorrectly merged):

  1. Within each class: merge overlapping/adjacent intervals on the same read
  2. Across classes:    subtract higher-priority intervals from lower-priority ones
                        Priority order: left_fork = right_fork > origin

Usage:
  python clean_annotation_beds.py \\
      --lf   CODEX/results/forte_v4.3/pseudo_labels/combined_left_fork.bed \\
      --rf   CODEX/results/forte_v4.3/pseudo_labels/combined_right_fork.bed \\
      --ori  data/.../ORIs_combined_cleaned.bed \\
      --out  data/.../ORIs_combined_cleaned_noforks.bed

  Or clean all three at once (in-place, with .bak backups):
      python clean_annotation_beds.py \\
          --lf   combined_left_fork.bed \\
          --rf   combined_right_fork.bed \\
          --ori  combined_origin.bed \\
          --out-lf  combined_left_fork.bed \\
          --out-rf  combined_right_fork.bed \\
          --out-ori combined_origin.bed \\
          --backup
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def load_bed4(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                     names=["chr", "start", "end", "read_id"])
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    return df


def merge_overlapping_per_read(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping/adjacent intervals within each (read_id, chr) group."""
    if df.empty:
        return df.copy()
    rows = []
    for (read_id, chrom), group in df.groupby(["read_id", "chr"], sort=False):
        intervals = sorted(zip(group["start"].tolist(), group["end"].tolist()))
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
    """Remove from target any bases that overlap subtract, per read."""
    if target_df.empty or subtract_df.empty:
        return target_df.copy()
    sub_by_read = {
        read_id: sorted(zip(g["start"].tolist(), g["end"].tolist()))
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


def split_lf_rf_at_midpoint_per_read(lf_df: pd.DataFrame, rf_df: pd.DataFrame):
    """Resolve LF/RF overlaps by splitting at the midpoint. No priority — equal split.

    For each overlapping (LF, RF) pair on the same read:
      overlap = [max(lf_start, rf_start), min(lf_end, rf_end)]
      midpoint = overlap centre
      → right half of overlap subtracted from LF
      → left half of overlap subtracted from RF
    """
    if lf_df.empty or rf_df.empty:
        return lf_df.copy(), rf_df.copy()

    sub_from_lf, sub_from_rf = [], []

    lf_by_read = {rid: g for rid, g in lf_df.groupby("read_id")}
    rf_by_read = {rid: g for rid, g in rf_df.groupby("read_id")}

    for read_id in set(lf_by_read) & set(rf_by_read):
        for lf_row in lf_by_read[read_id].itertuples(index=False):
            ls, le = int(lf_row.start), int(lf_row.end)
            for rf_row in rf_by_read[read_id].itertuples(index=False):
                rs, re = int(rf_row.start), int(rf_row.end)
                ov_start = max(ls, rs)
                ov_end   = min(le, re)
                if ov_start >= ov_end:
                    continue
                mid = int((ov_start + ov_end) / 2)
                # Right half of overlap → remove from LF
                sub_from_lf.append({"chr": lf_row.chr, "start": mid,      "end": ov_end, "read_id": read_id})
                # Left half of overlap  → remove from RF
                sub_from_rf.append({"chr": rf_row.chr, "start": ov_start, "end": mid,    "read_id": read_id})

    if not sub_from_lf and not sub_from_rf:
        return lf_df.copy(), rf_df.copy()

    cols = ["chr", "start", "end", "read_id"]
    clean_lf = subtract_per_read(lf_df, pd.DataFrame(sub_from_lf, columns=cols) if sub_from_lf else pd.DataFrame(columns=cols))
    clean_rf = subtract_per_read(rf_df, pd.DataFrame(sub_from_rf, columns=cols) if sub_from_rf else pd.DataFrame(columns=cols))
    clean_lf = merge_overlapping_per_read(clean_lf)
    clean_rf = merge_overlapping_per_read(clean_rf)

    return clean_lf, clean_rf


def clean_beds(lf_df, rf_df, ori_df):
    """Full cleaning pipeline. Returns (clean_lf, clean_rf, clean_ori).

    Steps (in order):
      1. Merge overlapping intervals within each class (per read)
      2. ORI takes priority over forks — subtract ORI from LF and RF
      3. LF vs RF: no priority — split overlaps at midpoint
    """
    clean_lf  = merge_overlapping_per_read(lf_df)
    clean_rf  = merge_overlapping_per_read(rf_df)
    clean_ori = merge_overlapping_per_read(ori_df)

    # Step 2: ORI priority
    clean_lf = subtract_per_read(clean_lf, clean_ori)
    clean_rf = subtract_per_read(clean_rf, clean_ori)
    clean_lf = merge_overlapping_per_read(clean_lf)
    clean_rf = merge_overlapping_per_read(clean_rf)

    # Step 3: LF vs RF — midpoint split
    clean_lf, clean_rf = split_lf_rf_at_midpoint_per_read(clean_lf, clean_rf)

    return clean_lf, clean_rf, clean_ori


def report(label, before, after):
    removed = len(before) - len(after)
    pct = removed / len(before) * 100 if len(before) else 0
    print(f"  {label}: {len(before):,} → {len(after):,}  ({removed:,} removed/merged, {pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lf",      required=True, help="Left fork BED4")
    parser.add_argument("--rf",      required=True, help="Right fork BED4")
    parser.add_argument("--ori",     required=True, help="Origin BED4 (to be cleaned)")
    parser.add_argument("--out",     help="Output path for cleaned ORI BED (default: print stats only)")
    parser.add_argument("--out-lf",  help="Output path for cleaned LF BED")
    parser.add_argument("--out-rf",  help="Output path for cleaned RF BED")
    parser.add_argument("--out-ori", help="Output path for cleaned ORI BED (alias for --out)")
    parser.add_argument("--backup",  action="store_true",
                        help="If overwriting input files, save .bak backups first")
    args = parser.parse_args()

    out_ori = args.out_ori or args.out

    print("Loading BED files...")
    lf_df  = load_bed4(args.lf)
    rf_df  = load_bed4(args.rf)
    ori_df = load_bed4(args.ori)
    print(f"  LF:  {len(lf_df):,}")
    print(f"  RF:  {len(rf_df):,}")
    print(f"  ORI: {len(ori_df):,}")

    print("Cleaning...")
    clean_lf, clean_rf, clean_ori = clean_beds(lf_df, rf_df, ori_df)

    print("Results:")
    report("LF",  lf_df,  clean_lf)
    report("RF",  rf_df,  clean_rf)
    report("ORI", ori_df, clean_ori)

    outputs = []
    if args.out_lf:
        outputs.append((args.out_lf, clean_lf, args.lf))
    if args.out_rf:
        outputs.append((args.out_rf, clean_rf, args.rf))
    if out_ori:
        outputs.append((out_ori, clean_ori, args.ori))

    if not outputs:
        print("\nNo --out* paths specified — stats only, nothing written.")
        return

    for out_path, clean_df, src_path in outputs:
        if args.backup and Path(out_path).resolve() == Path(src_path).resolve():
            bak = out_path + ".bak"
            shutil.copy2(src_path, bak)
            print(f"  Backup: {bak}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(out_path, sep="\t", header=False, index=False)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
