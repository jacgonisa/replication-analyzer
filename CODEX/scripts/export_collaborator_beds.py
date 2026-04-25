"""
Export collaborator-ready BED files from a FORTE reannotated_events.tsv.

Produces one BED file per event type × threshold:
  forte_{MODEL}_left_fork_sensitive.bed   (mean_prob >= SENSITIVE_THRESH)
  forte_{MODEL}_left_fork_confident.bed   (mean_prob >= CONFIDENT_THRESH)
  forte_{MODEL}_right_fork_sensitive.bed
  forte_{MODEL}_right_fork_confident.bed
  forte_{MODEL}_origin_sensitive.bed
  forte_{MODEL}_origin_confident.bed

BED6 format (no header):
  chr  start  end  read_id  score(mean_prob*1000)  strand(.)

Also writes README.txt summarising thresholds, event counts, and performance.
"""

import argparse
import pandas as pd
from pathlib import Path

CHR_ORDER = {"Chr1": 0, "Chr2": 1, "Chr3": 2, "Chr4": 3, "Chr5": 4, "ChrC": 5, "ChrM": 6}

EVENT_TYPES = ["left_fork", "right_fork", "origin"]

SENSITIVE_THRESH = 0.3
CONFIDENT_THRESH = 0.5


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events",       required=True,
                        help="reannotated_events.tsv from reannotate_reads_codex.py")
    parser.add_argument("--output-dir",   required=True,
                        help="Directory to write BED files and README")
    parser.add_argument("--model-name",   required=True,
                        help="Model label used in output filenames, e.g. forte_v5.3")
    parser.add_argument("--sensitive-threshold", type=float, default=SENSITIVE_THRESH,
                        help=f"mean_prob threshold for sensitive BEDs (default {SENSITIVE_THRESH})")
    parser.add_argument("--confident-threshold", type=float, default=CONFIDENT_THRESH,
                        help=f"mean_prob threshold for confident BEDs (default {CONFIDENT_THRESH})")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.events} ...")
    df = pd.read_csv(args.events, sep="\t")
    df["chr_sort"] = df["chr"].map(CHR_ORDER).fillna(99)
    df = df.sort_values(["chr_sort", "start"])

    thresholds = {
        "sensitive": args.sensitive_threshold,
        "confident": args.confident_threshold,
    }

    counts = {}
    for etype in EVENT_TYPES:
        sub = df[df["event_type"] == etype]
        for tier, thresh in thresholds.items():
            filtered = sub[sub["mean_prob"] >= thresh]
            fname = out_dir / f"{args.model_name}_{etype}_{tier}.bed"
            n = export_bed(filtered, fname)
            counts[(etype, tier)] = n
            print(f"  {fname.name}  ({n:,} events, mean_prob >= {thresh})")

    # README
    readme = out_dir / "README.txt"
    readme.write_text(f"""\
{args.model_name} — Collaborator BED files
{'=' * (len(args.model_name) + 28)}
Source: {args.events}

BED format (tab-separated, no header):
  col1: chromosome
  col2: start (0-based, half-open)
  col3: end
  col4: read_id (nanopore read the event was called on)
  col5: score = mean_prob x 1000  (range 0-1000)
  col6: strand = '.' (events are on reads, not on reference strand)

Files and event counts:
  {args.model_name}_left_fork_sensitive.bed   {counts[('left_fork',  'sensitive')]:>7,}  mean_prob >= {args.sensitive_threshold}
  {args.model_name}_left_fork_confident.bed   {counts[('left_fork',  'confident')]:>7,}  mean_prob >= {args.confident_threshold}
  {args.model_name}_right_fork_sensitive.bed  {counts[('right_fork', 'sensitive')]:>7,}  mean_prob >= {args.sensitive_threshold}
  {args.model_name}_right_fork_confident.bed  {counts[('right_fork', 'confident')]:>7,}  mean_prob >= {args.confident_threshold}
  {args.model_name}_origin_sensitive.bed      {counts[('origin',     'sensitive')]:>7,}  mean_prob >= {args.sensitive_threshold}
  {args.model_name}_origin_confident.bed      {counts[('origin',     'confident')]:>7,}  mean_prob >= {args.confident_threshold}

Recommended threshold for most analyses: confident (mean_prob >= {args.confident_threshold}).
Use sensitive (mean_prob >= {args.sensitive_threshold}) when maximising recall matters
more than precision (e.g. broad genomic surveys).

Note: each BED entry is one event on one read. Multiple reads can cover the
same genomic locus — aggregate (e.g. bedtools merge) if you want reference-
coordinate events collapsed across reads.
""")
    print(f"\nREADME written to {readme}")


if __name__ == "__main__":
    main()
