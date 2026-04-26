"""
Export collaborator-ready BED files from a FORTE reannotated_events.tsv.

Produces:
  - Fork BEDs (one per type x threshold):
      forte_{MODEL}_left_fork_sensitive.bed   (mean_prob >= 0.3)
      forte_{MODEL}_left_fork_confident.bed   (mean_prob >= 0.5)
      forte_{MODEL}_right_fork_sensitive.bed
      forte_{MODEL}_right_fork_confident.bed

  - Single origin BED with all events (mean_prob >= 0.4) tiered by confidence:
      forte_{MODEL}_origin_tiered.bed
      Tiers: high (>0.6), medium (0.5-0.6), lower (0.4-0.5)

BED7 format for origins (no header):
  chr  start  end  read_id  score(mean_prob*1000)  strand(.)  confidence_tier

BED6 format for forks (no header):
  chr  start  end  read_id  score(mean_prob*1000)  strand(.)

Also writes README.txt.
"""

import argparse
import pandas as pd
from pathlib import Path

CHR_ORDER = {"Chr1": 0, "Chr2": 1, "Chr3": 2, "Chr4": 3, "Chr5": 4, "ChrC": 5, "ChrM": 6}

SENSITIVE_THRESH = 0.3
CONFIDENT_THRESH = 0.5

# Origin confidence tiers (based on threshold sweep: best recall at 0.4, best IoU at 0.6)
ORI_TIERS = [
    ("high",   0.6,  1.01),   # mean_prob > 0.6
    ("medium", 0.5,  0.6),    # mean_prob 0.5-0.6
    ("lower",  0.4,  0.5),    # mean_prob 0.4-0.5
]
ORI_MIN_PROB = 0.4  # minimum to appear in tiered BED (best-recall threshold)


def export_bed(df: pd.DataFrame, out_path: Path) -> int:
    bed = pd.DataFrame({
        "chr":    df["chr"],
        "start":  df["start"],
        "end":    df["end"],
        "name":   df["read_id"],
        "score":  (df["mean_prob"] * 1000).astype(int).clip(0, 1000),
        "strand": ".",
    })
    bed.to_csv(out_path, sep="\t", header=False, index=False)
    return len(bed)


def export_origin_tiered_bed(df: pd.DataFrame, out_path: Path) -> dict:
    """Export all ORIs >= ORI_MIN_PROB with a confidence_tier column."""
    sub = df[df["mean_prob"] >= ORI_MIN_PROB].copy()

    def assign_tier(prob):
        for tier, lo, hi in ORI_TIERS:
            if lo <= prob < hi:
                return tier
        return "lower"

    sub["confidence_tier"] = sub["mean_prob"].apply(assign_tier)
    bed = pd.DataFrame({
        "chr":             sub["chr"],
        "start":           sub["start"],
        "end":             sub["end"],
        "name":            sub["read_id"],
        "score":           (sub["mean_prob"] * 1000).astype(int).clip(0, 1000),
        "strand":          ".",
        "confidence_tier": sub["confidence_tier"],
    })
    bed.to_csv(out_path, sep="\t", header=False, index=False)
    counts = sub["confidence_tier"].value_counts().to_dict()
    return {tier: counts.get(tier, 0) for tier, _, _ in ORI_TIERS}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events",       required=True,
                        help="reannotated_events.tsv from reannotate_reads_codex.py")
    parser.add_argument("--output-dir",   required=True,
                        help="Directory to write BED files and README")
    parser.add_argument("--model-name",   required=True,
                        help="Model label used in output filenames, e.g. forte_v5.4")
    parser.add_argument("--sensitive-threshold", type=float, default=SENSITIVE_THRESH,
                        help=f"mean_prob threshold for sensitive fork BEDs (default {SENSITIVE_THRESH})")
    parser.add_argument("--confident-threshold", type=float, default=CONFIDENT_THRESH,
                        help=f"mean_prob threshold for confident fork BEDs (default {CONFIDENT_THRESH})")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.events} ...")
    df = pd.read_csv(args.events, sep="\t")
    df["chr_sort"] = df["chr"].map(CHR_ORDER).fillna(99)
    df = df.sort_values(["chr_sort", "start"])

    counts = {}

    # ── Forks: sensitive + confident BEDs ─────────────────────────────────────
    for etype in ["left_fork", "right_fork"]:
        sub = df[df["event_type"] == etype]
        for tier, thresh in [("sensitive", args.sensitive_threshold),
                              ("confident", args.confident_threshold)]:
            filtered = sub[sub["mean_prob"] >= thresh]
            fname = out_dir / f"{args.model_name}_{etype}_{tier}.bed"
            n = export_bed(filtered, fname)
            counts[(etype, tier)] = n
            print(f"  {fname.name}  ({n:,} events, mean_prob >= {thresh})")

    # ── Origins: single tiered BED ────────────────────────────────────────────
    ori_df = df[df["event_type"] == "origin"]
    ori_fname = out_dir / f"{args.model_name}_origin_tiered.bed"
    tier_counts = export_origin_tiered_bed(ori_df, ori_fname)
    total_ori = sum(tier_counts.values())
    print(f"  {ori_fname.name}  ({total_ori:,} events, mean_prob >= {ORI_MIN_PROB})")
    for tier, lo, hi in ORI_TIERS:
        print(f"    {tier:8s}  (mean_prob {lo}-{hi}): {tier_counts[tier]:>6,}")

    # README
    readme = out_dir / "README.txt"
    readme.write_text(f"""\
{args.model_name} — Collaborator BED files
{'=' * (len(args.model_name) + 28)}
Source: {args.events}

── FORK FILES ──────────────────────────────────────────────────────────────
BED6 format (tab-separated, no header):
  col1: chromosome
  col2: start (0-based, half-open)
  col3: end
  col4: read_id
  col5: score = mean_prob x 1000  (range 0-1000)
  col6: strand = '.' (events are on reads, not on reference strand)

  {args.model_name}_left_fork_sensitive.bed   {counts[('left_fork',  'sensitive')]:>7,}  mean_prob >= {args.sensitive_threshold}
  {args.model_name}_left_fork_confident.bed   {counts[('left_fork',  'confident')]:>7,}  mean_prob >= {args.confident_threshold}
  {args.model_name}_right_fork_sensitive.bed  {counts[('right_fork', 'sensitive')]:>7,}  mean_prob >= {args.sensitive_threshold}
  {args.model_name}_right_fork_confident.bed  {counts[('right_fork', 'confident')]:>7,}  mean_prob >= {args.confident_threshold}

── ORIGIN FILE ─────────────────────────────────────────────────────────────
BED7 format (tab-separated, no header):
  col1: chromosome
  col2: start (0-based, half-open)
  col3: end
  col4: read_id
  col5: score = mean_prob x 1000  (range 0-1000)
  col6: strand = '.'
  col7: confidence_tier  (high | medium | lower)

Confidence tiers (based on threshold sweep — best recall at prob=0.4,
best IoU/localisation at prob=0.6):
  high   mean_prob > 0.6   — {tier_counts['high']:>6,} events  best localisation
  medium mean_prob 0.5-0.6 — {tier_counts['medium']:>6,} events  good balance
  lower  mean_prob 0.4-0.5 — {tier_counts['lower']:>6,} events  maximum recall

  {args.model_name}_origin_tiered.bed         {total_ori:>7,}  mean_prob >= {ORI_MIN_PROB} (all tiers)

Note: each BED entry is one event on one read. Multiple reads can cover the
same genomic locus — aggregate (e.g. bedtools merge) if you want reference-
coordinate events collapsed across reads.
""")
    print(f"\nREADME written to {readme}")


if __name__ == "__main__":
    main()
