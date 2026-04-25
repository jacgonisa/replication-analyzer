#!/usr/bin/env python
"""Generate ORI-validated pseudo-fork labels for FORTE.

For each read that has a human-curated ORI annotation, use the v4 model's
per-window predictions to extract fork regions flanking the ORI:
  - windows LEFT of ORI_start  → candidate left_fork events
  - windows RIGHT of ORI_end   → candidate right_fork events

Because these predictions are spatially constrained by a gold-standard ORI,
they are higher-confidence pseudo-labels than unconstrained model predictions.

Combines:
  1. ORI-validated pseudo-forks (this script)
  2. General model pseudo-forks (from generate_pseudo_labels_forte.py)
  3. Real human annotations

Usage (run AFTER generate_pseudo_labels_forte.py has produced all_predictions.tsv):
  python generate_ori_validated_forks.py \\
      --predictions  CODEX/results/forte_v1/pseudo_labels/all_predictions.tsv \\
      --ori-bed      data/.../ORIs_combined_cleaned.bed \\
      --real-left    data/.../leftForks_ALL_combined.bed \\
      --real-right   data/.../rightForks_ALL_combined.bed \\
      --pseudo-left  CODEX/results/forte_v1/pseudo_labels/pseudo_left_fork.bed \\
      --pseudo-right CODEX/results/forte_v1/pseudo_labels/pseudo_right_fork.bed \\
      --output       CODEX/results/forte_v1/pseudo_labels \\
      --left-thresh  0.30 \\
      --right-thresh 0.30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_bed4(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3])
    df.columns = ["chr", "start", "end", "read_id"]
    return df


def windows_to_events(win_df: pd.DataFrame, prob_col: str, prob_threshold: float,
                      max_gap: int = 5000, min_windows: int = 1) -> pd.DataFrame:
    """Merge consecutive windows above threshold into event regions."""
    hits = win_df[win_df[prob_col] >= prob_threshold].copy()
    if len(hits) == 0:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])

    hits = hits.sort_values("start").reset_index(drop=True)
    events = []
    cur = None
    for row in hits.itertuples(index=False):
        if cur is None:
            cur = {"chr": row.chr, "start": int(row.start), "end": int(row.end),
                   "read_id": row.read_id, "n": 1}
        elif int(row.start) - cur["end"] <= max_gap:
            cur["end"] = max(cur["end"], int(row.end))
            cur["n"] += 1
        else:
            if cur["n"] >= min_windows:
                events.append(cur)
            cur = {"chr": row.chr, "start": int(row.start), "end": int(row.end),
                   "read_id": row.read_id, "n": 1}
    if cur is not None and cur["n"] >= min_windows:
        events.append(cur)

    if not events:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    df = pd.DataFrame(events)
    return df[["chr", "start", "end", "read_id"]]


def extract_ori_validated_forks(
    predictions: pd.DataFrame,
    ori_annotations: pd.DataFrame,
    left_thresh: float = 0.30,
    right_thresh: float = 0.30,
    max_gap: int = 5000,
    min_windows: int = 1,
    search_radius: int = 60000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each ORI-annotated read, extract model-predicted forks flanking the ORI.

    Only looks at windows:
      - LEFT of ORI start  (for left_fork)  — within search_radius
      - RIGHT of ORI end   (for right_fork) — within search_radius

    Returns (left_fork_bed, right_fork_bed).
    """
    # Index predictions by read_id for fast lookup
    pred_by_read = {rid: grp for rid, grp in predictions.groupby("read_id")}

    left_events = []
    right_events = []
    n_oris_used = 0
    n_reads_skipped = 0

    for _, ori in ori_annotations.iterrows():
        read_id = str(ori["read_id"])
        ori_start = int(ori["start"])
        ori_end = int(ori["end"])

        read_preds = pred_by_read.get(read_id)
        if read_preds is None:
            n_reads_skipped += 1
            continue

        # Left flank: windows that end at or before ori_start, within search_radius
        left_mask = (read_preds["end"] <= ori_start) & \
                    (read_preds["end"] >= ori_start - search_radius)
        left_wins = read_preds[left_mask].copy()

        # Right flank: windows that start at or after ori_end, within search_radius
        right_mask = (read_preds["start"] >= ori_end) & \
                     (read_preds["start"] <= ori_end + search_radius)
        right_wins = read_preds[right_mask].copy()

        if len(left_wins) > 0:
            evts = windows_to_events(left_wins, "prob_left_fork", left_thresh,
                                     max_gap=max_gap, min_windows=min_windows)
            left_events.append(evts)

        if len(right_wins) > 0:
            evts = windows_to_events(right_wins, "prob_right_fork", right_thresh,
                                     max_gap=max_gap, min_windows=min_windows)
            right_events.append(evts)

        n_oris_used += 1

    print(f"  ORIs processed: {n_oris_used:,}  (skipped {n_reads_skipped:,} reads not in predictions)")

    left_df = pd.concat(left_events, ignore_index=True) if left_events else \
              pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    right_df = pd.concat(right_events, ignore_index=True) if right_events else \
               pd.DataFrame(columns=["chr", "start", "end", "read_id"])

    return left_df, right_df


def merge_beds(*dfs: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(list(dfs), ignore_index=True)
    return combined.drop_duplicates(subset=["chr", "start", "end", "read_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="ORI-validated pseudo-fork generation for FORTE")
    parser.add_argument("--predictions", required=True,
                        help="all_predictions.tsv from generate_pseudo_labels_forte.py")
    parser.add_argument("--ori-bed", required=True, help="Human-curated ORI BED file")
    parser.add_argument("--real-left", required=True, help="Real left fork BED")
    parser.add_argument("--real-right", required=True, help="Real right fork BED")
    parser.add_argument("--pseudo-left", required=True,
                        help="General model pseudo left_fork BED (from generate_pseudo_labels_forte.py)")
    parser.add_argument("--pseudo-right", required=True,
                        help="General model pseudo right_fork BED")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--left-thresh", type=float, default=0.30,
                        help="prob_left_fork threshold for ORI-flanking windows (default 0.30 — lower because spatially constrained)")
    parser.add_argument("--right-thresh", type=float, default=0.30)
    parser.add_argument("--search-radius", type=int, default=60000,
                        help="Max distance from ORI boundary to search for forks (bp, default 60000)")
    parser.add_argument("--max-gap", type=int, default=5000)
    parser.add_argument("--min-windows", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions…")
    preds = pd.read_csv(args.predictions, sep="\t")
    print(f"  {len(preds):,} windows across {preds['read_id'].nunique():,} reads")

    print("Loading ORI annotations…")
    oris = load_bed4(args.ori_bed)
    print(f"  {len(oris):,} ORI annotations on {oris['read_id'].nunique():,} reads")

    print("Loading real fork annotations…")
    real_left = load_bed4(args.real_left)
    real_right = load_bed4(args.real_right)
    print(f"  Real left forks: {len(real_left):,}")
    print(f"  Real right forks: {len(real_right):,}")

    print("Loading general model pseudo-forks…")
    pseudo_left = load_bed4(args.pseudo_left)
    pseudo_right = load_bed4(args.pseudo_right)
    print(f"  Pseudo left forks: {len(pseudo_left):,}")
    print(f"  Pseudo right forks: {len(pseudo_right):,}")

    print(f"\nExtracting ORI-validated forks (left_thresh={args.left_thresh}, right_thresh={args.right_thresh}, radius={args.search_radius:,}bp)…")
    ori_left, ori_right = extract_ori_validated_forks(
        predictions=preds,
        ori_annotations=oris,
        left_thresh=args.left_thresh,
        right_thresh=args.right_thresh,
        max_gap=args.max_gap,
        min_windows=args.min_windows,
        search_radius=args.search_radius,
    )

    # Save ORI-validated forks separately for inspection (deduplicated)
    ori_left = ori_left.drop_duplicates(subset=["chr", "start", "end", "read_id"]).reset_index(drop=True)
    ori_right = ori_right.drop_duplicates(subset=["chr", "start", "end", "read_id"]).reset_index(drop=True)
    ori_left.to_csv(output_dir / "ori_validated_left_fork.bed", sep="\t", header=False, index=False)
    ori_right.to_csv(output_dir / "ori_validated_right_fork.bed", sep="\t", header=False, index=False)
    print(f"  ORI-validated left forks:  {len(ori_left):,}")
    print(f"  ORI-validated right forks: {len(ori_right):,}")

    print("\nBuilding final combined BED files (real + ORI-validated + general pseudo)…")
    final_left = merge_beds(real_left, ori_left, pseudo_left)
    final_right = merge_beds(real_right, ori_right, pseudo_right)

    final_left.to_csv(output_dir / "combined_left_fork.bed", sep="\t", header=False, index=False)
    final_right.to_csv(output_dir / "combined_right_fork.bed", sep="\t", header=False, index=False)

    print(f"\nFinal combined left_fork:  {len(real_left):,} real + {len(ori_left):,} ORI-val + {len(pseudo_left):,} pseudo = {len(final_left):,} total")
    print(f"Final combined right_fork: {len(real_right):,} real + {len(ori_right):,} ORI-val + {len(pseudo_right):,} pseudo = {len(final_right):,} total")
    print(f"\nSaved to {output_dir}")
    print("Run preprocess_weak4_codex.py with forte_v1.yaml next.")


if __name__ == "__main__":
    main()
