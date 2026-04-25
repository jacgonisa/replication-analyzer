#!/usr/bin/env python
"""Compare AI-predicted events vs Nerea's human annotations.

Produces a table with counts for each event class:
  - AI only   : AI predictions with no overlap with Nerea's annotations
  - Nerea only : Nerea annotations with no overlapping AI prediction
  - Both       : events present in both (overlap by >= iou_threshold)
  - Total AI   : total AI predictions
  - Total Nerea: total Nerea annotations
  - Recovery   : Both / Total Nerea  (fraction of Nerea's events found by AI)

Usage (from replication-analyzer/ root):
  python CODEX/scripts/compare_ai_vs_nerea.py \
      --events   CODEX/results/forte_v5.1/reannotation/reannotated_events.tsv \
      --output   CODEX/results/forte_v5.1/reannotation/nerea_vs_ai_agreement.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

NEREA_SOURCES = {
    "left_fork":  ROOT / "data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed",
    "right_fork": ROOT / "data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed",
    "origin":     ROOT / "data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed",
}

IOU_THRESHOLD = 0.1   # permissive: small events count if they broadly overlap


def load_bed4(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def compute_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def build_read_index(df: pd.DataFrame) -> dict:
    """Per read_id, sorted list of (start, end, row_idx)."""
    idx: dict = {}
    for i, row in df.iterrows():
        idx.setdefault(row["read_id"], []).append((int(row["start"]), int(row["end"]), i))
    return idx


def match_events(
    query: pd.DataFrame,
    ref: pd.DataFrame,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[list[int], list[int]]:
    """
    Returns:
      query_matched : indices into query that overlap ref at >= iou_threshold
      ref_matched   : indices into ref that overlap query at >= iou_threshold
    """
    ref_idx = build_read_index(ref)
    query_matched: set[int] = set()
    ref_matched:   set[int] = set()

    for qi, qrow in query.iterrows():
        candidates = ref_idx.get(qrow["read_id"], [])
        for rs, re, ri in candidates:
            if rs >= qrow["end"] or re <= qrow["start"]:
                continue  # no overlap at all
            iou = compute_iou(int(qrow["start"]), int(qrow["end"]), rs, re)
            if iou >= iou_threshold:
                query_matched.add(qi)
                ref_matched.add(ri)

    return list(query_matched), list(ref_matched)


def size_distribution(df: pd.DataFrame, col_start: str = "start", col_end: str = "end") -> str:
    lens = df[col_end] - df[col_start]
    if len(lens) == 0:
        return "n=0"
    return (f"median={np.median(lens):.0f}bp "
            f"p25={np.percentile(lens,25):.0f} "
            f"p75={np.percentile(lens,75):.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events",        required=True, help="reannotated_events.tsv from reannotate_reads_codex.py")
    parser.add_argument("--output",        required=True, help="Output TSV for agreement table")
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD,
                        help=f"IoU threshold to call a match (default {IOU_THRESHOLD})")
    args = parser.parse_args()

    ai_all = pd.read_csv(args.events, sep="\t")
    print(f"Loaded {len(ai_all):,} AI events")

    rows = []
    for event_type, nerea_path in NEREA_SOURCES.items():
        if not nerea_path.exists():
            print(f"  WARNING: {nerea_path} not found — skipping {event_type}")
            continue

        nerea = load_bed4(nerea_path)
        ai    = ai_all[ai_all["event_type"] == event_type].copy()

        print(f"\n── {event_type} ──")
        print(f"  AI predictions : {len(ai):,}")
        print(f"  Nerea events   : {len(nerea):,}")

        ai_matched_idx, nerea_matched_idx = match_events(ai, nerea, iou_threshold=args.iou_threshold)

        n_both       = len(set(nerea_matched_idx))      # Nerea events recovered by AI
        n_ai_matched = len(set(ai_matched_idx))          # AI predictions that hit a Nerea event
        n_ai_only    = len(ai) - n_ai_matched            # AI predictions with no Nerea overlap
        n_nerea_only = len(nerea) - n_both               # Nerea events not found by AI

        recovery   = n_both / len(nerea) if len(nerea) > 0 else float("nan")
        precision_ = n_ai_matched / len(ai) if len(ai) > 0 else float("nan")

        # Size distributions
        ai_only_df    = ai[~ai.index.isin(ai_matched_idx)]
        ai_both_df    = ai[ai.index.isin(ai_matched_idx)]
        nerea_only_df = nerea[~nerea.index.isin(nerea_matched_idx)]
        nerea_both_df = nerea[nerea.index.isin(nerea_matched_idx)]

        print(f"  Both (Nerea recovered)    : {n_both:,}  recovery={recovery:.1%}")
        print(f"  AI only (novel / FP)      : {n_ai_only:,}  {size_distribution(ai_only_df)}")
        print(f"  Nerea only (missed by AI) : {n_nerea_only:,}  {size_distribution(nerea_only_df)}")

        rows.append({
            "event_type":       event_type,
            "iou_threshold":    args.iou_threshold,
            "total_ai":         len(ai),
            "total_nerea":      len(nerea),
            "both":             n_both,
            "ai_only":          n_ai_only,
            "nerea_only":       n_nerea_only,
            "recovery_pct":     round(100 * recovery, 1),
            "ai_precision_pct": round(100 * precision_, 1),
            "ai_only_median_bp":    int((ai_only_df["end"] - ai_only_df["start"]).median()) if len(ai_only_df) else 0,
            "nerea_only_median_bp": int((nerea_only_df["end"] - nerea_only_df["start"]).median()) if len(nerea_only_df) else 0,
            "both_median_bp":       int((nerea_both_df["end"] - nerea_both_df["start"]).median()) if len(nerea_both_df) else 0,
        })

    summary = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, sep="\t", index=False)

    print("\n" + "="*70)
    print("AGREEMENT SUMMARY")
    print("="*70)
    display_cols = ["event_type", "total_ai", "total_nerea",
                    "both", "ai_only", "nerea_only",
                    "recovery_pct", "ai_precision_pct"]
    print(summary[display_cols].to_string(index=False))
    print(f"\nFull table saved to: {args.output}")


if __name__ == "__main__":
    main()
