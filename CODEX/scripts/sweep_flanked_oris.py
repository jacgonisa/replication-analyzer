#!/usr/bin/env python
"""Sweep flanked-ORI counts across annotation combinations and flank window sizes.

Combinations tested (ORIs × Forks):
  A) real ORIs  ×  real forks (strict annotated)
  B) real ORIs  ×  real forks (ALL — includes borderline)
  C) real ORIs  ×  pseudo-forks (v2)
  D) real ORIs  ×  real+pseudo forks combined
  E) pseudo-ORIs (v2)  ×  pseudo-forks (v2)
  F) real+pseudo ORIs  ×  pseudo-forks (v2)
  G) real+pseudo ORIs  ×  real+pseudo forks combined

For each combination, sweep flank window: 10, 25, 50, 75, 100, 150, 200 kb.
Report: n_flanked, pct_flanked, n_reads_with_flanked.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")
ANNOT = BASE / "data/case_study_jan2026/combined/annotations"
PSEUDO = BASE / "CODEX/results/forte_v2/pseudo_labels"

REAL_ORI    = ANNOT / "ORIs_combined_cleaned.bed"
REAL_LF     = ANNOT / "leftForks_combined.bed"        # strict (curated)
REAL_LF_ALL = ANNOT / "leftForks_ALL_combined.bed"    # all (incl. borderline)
REAL_RF     = ANNOT / "rightForks_combined.bed"
REAL_RF_ALL = ANNOT / "rightForks_ALL_combined.bed"
PSEUDO_LF   = PSEUDO / "combined_left_fork.bed"
PSEUDO_RF   = PSEUDO / "combined_right_fork.bed"
PSEUDO_ORI  = PSEUDO / "combined_origin.bed"

FLANK_KBS = [10, 25, 50, 75, 100, 150, 200]


def load_bed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, low_memory=False)
    cols = ["chr", "start", "end", "read_id"] + [f"c{i}" for i in range(4, len(df.columns))]
    df.columns = cols[:len(df.columns)]
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    return df


def combine(*dfs: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(list(dfs), ignore_index=True)
    out = out.drop_duplicates(subset=["chr", "start", "end", "read_id"])
    return out


def count_flanked(ori_df: pd.DataFrame, lf_df: pd.DataFrame, rf_df: pd.DataFrame,
                  flank_bp: int) -> tuple[int, int]:
    """Returns (n_flanked_oris, n_reads_with_flanked)."""
    lf_by = {rid: grp for rid, grp in lf_df.groupby("read_id")} if len(lf_df) else {}
    rf_by = {rid: grp for rid, grp in rf_df.groupby("read_id")} if len(rf_df) else {}

    flanked = 0
    flanked_reads = set()
    for _, ori in ori_df.iterrows():
        rid = ori["read_id"]
        os, oe = int(ori["start"]), int(ori["end"])

        has_left = False
        if rid in lf_by:
            lf = lf_by[rid]
            has_left = len(lf[(lf["end"] >= os - flank_bp) & (lf["end"] <= os + flank_bp)]) > 0

        has_right = False
        if rid in rf_by:
            rf = rf_by[rid]
            has_right = len(rf[(rf["start"] >= oe - flank_bp) & (rf["start"] <= oe + flank_bp)]) > 0

        if has_left and has_right:
            flanked += 1
            flanked_reads.add(rid)

    return flanked, len(flanked_reads)


def main():
    print("Loading annotation files...")
    real_ori    = load_bed(REAL_ORI)
    real_lf     = load_bed(REAL_LF)
    real_lf_all = load_bed(REAL_LF_ALL)
    real_rf     = load_bed(REAL_RF)
    real_rf_all = load_bed(REAL_RF_ALL)
    pseudo_lf   = load_bed(PSEUDO_LF)
    pseudo_rf   = load_bed(PSEUDO_RF)
    pseudo_ori  = load_bed(PSEUDO_ORI)

    real_pseudo_ori = combine(real_ori, pseudo_ori)
    combined_lf     = combine(real_lf_all, pseudo_lf)
    combined_rf     = combine(real_rf_all, pseudo_rf)

    combinations = {
        "A_realORI_x_realFork_strict":  (real_ori,          real_lf,     real_rf),
        "B_realORI_x_realFork_ALL":     (real_ori,          real_lf_all, real_rf_all),
        "C_realORI_x_pseudoFork":       (real_ori,          pseudo_lf,   pseudo_rf),
        "D_realORI_x_real+pseudoFork":  (real_ori,          combined_lf, combined_rf),
        "E_pseudoORI_x_pseudoFork":     (pseudo_ori,        pseudo_lf,   pseudo_rf),
        "F_real+pseudoORI_x_pseudoFork":(real_pseudo_ori,   pseudo_lf,   pseudo_rf),
        "G_real+pseudoORI_x_all":       (real_pseudo_ori,   combined_lf, combined_rf),
    }

    # Header
    print(f"\n{'Combination':<38} {'Flank':>7} {'n_ORI':>8} {'flanked':>8} {'%':>6} {'reads':>7}")
    print("-" * 82)

    rows = []
    for combo_name, (ori_df, lf_df, rf_df) in combinations.items():
        n_total = len(ori_df)
        for flank_kb in FLANK_KBS:
            flank_bp = int(flank_kb * 1000)
            n_fl, n_reads = count_flanked(ori_df, lf_df, rf_df, flank_bp)
            pct = 100 * n_fl / max(n_total, 1)
            print(f"{combo_name:<38} {flank_kb:>5}kb {n_total:>8,} {n_fl:>8,} {pct:>5.1f}% {n_reads:>7,}")
            rows.append(dict(combination=combo_name, flank_kb=flank_kb,
                             n_ori=n_total, n_flanked=n_fl, pct_flanked=round(pct,1),
                             n_reads=n_reads))
        print()

    # Save TSV
    out = BASE / "CODEX/results/flanked_ori_sweep.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"\nSaved → {out}")

    # Summary: best combo per flank window (by n_flanked)
    df = pd.DataFrame(rows)
    print("\n=== BEST COMBINATION per flank window (by n_flanked) ===")
    for flank_kb in FLANK_KBS:
        sub = df[df["flank_kb"] == flank_kb].sort_values("n_flanked", ascending=False).iloc[0]
        print(f"  {flank_kb:>3}kb → {sub['combination']:<38}  {sub['n_flanked']:>6,} flanked  ({sub['pct_flanked']:.1f}%)")


if __name__ == "__main__":
    main()
