#!/usr/bin/env python3
"""
Infer Origins and Terminations from fork annotations per nanopore read.

INPUT (two BED-like, tab-delimited files), 5 columns:
  chrom    start    end    readID    gradient

- left_forks:  negative-gradient fork segments (Left forks)
- right_forks: positive-gradient fork segments (Right forks)

EVENT DEFINITIONS (per readID AND chromosome):
- Origin:       contiguous Left then Right  (L -> R)
- Termination:  contiguous Right then Left  (R -> L)

For each adjacent pair (after sorting by start within readID+chrom):
- If they do not overlap: output the between-region [end(first), start(second)]
- If they partially overlap: output the overlap region (intersection)
- If one fully contains the other (total containment): DO NOT output (skip)

OUTPUT (two BED-like files), 6 columns:
  chrom    start    end    readID    grad_first    grad_second

- origins:       grad_first = left_grad,  grad_second = right_grad
- terminations:  grad_first = right_grad, grad_second = left_grad


python3 call_ori_ter_from_forks.py \
  --left leftForks_DNAscent_forkSense_14.bed \
  --right rightForks_DNAscent_forkSense_14.bed \
  --origins origins_DNAscent_forkSense_14.bed \
  --terminations terminations_DNAscent_forkSense_14.bed \
  --min-len 0


"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import sys


@dataclass(frozen=True)
class ForkSeg:
    chrom: str
    start: int
    end: int
    read_id: str
    grad: float
    kind: str  # "L" or "R"


def parse_fork_bed(path: str, kind: str) -> List[ForkSeg]:
    segs: List[ForkSeg] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                raise ValueError(
                    f"{path}:{line_no}: expected >=5 tab-separated columns: chrom start end readID gradient"
                )

            chrom, s, e, read_id, g = parts[0], parts[1], parts[2], parts[3], parts[4]

            try:
                start = int(float(s))
                end = int(float(e))
            except ValueError as ex:
                raise ValueError(f"{path}:{line_no}: start/end not numeric: {s}, {e}") from ex

            try:
                grad = float(g)
            except ValueError as ex:
                raise ValueError(f"{path}:{line_no}: gradient not numeric: {g}") from ex

            if end < start:
                start, end = end, start

            segs.append(ForkSeg(chrom=chrom, start=start, end=end, read_id=read_id, grad=grad, kind=kind))

    return segs


def group_by_read_and_chrom(segs: Iterable[ForkSeg]) -> Dict[Tuple[str, str], List[ForkSeg]]:
    d: Dict[Tuple[str, str], List[ForkSeg]] = {}
    for s in segs:
        d.setdefault((s.read_id, s.chrom), []).append(s)
    return d


def compute_interval_no_containment(a: ForkSeg, b: ForkSeg) -> Optional[Tuple[int, int]]:
    """
    For adjacent segments a then b (assumed sorted by start), return:
      - overlap interval if they overlap PARTIALLY (no containment),
      - gap interval if no overlap,
      - None if total containment (one contains the other) or empty interval.
    """

    # Reject total containment (nested calls)
    a_contains_b = (a.start <= b.start) and (a.end >= b.end)
    b_contains_a = (b.start <= a.start) and (b.end >= a.end)
    if a_contains_b or b_contains_a:
        return None

    # Partial overlap -> intersection
    ov_start = max(a.start, b.start)
    ov_end = min(a.end, b.end)
    if ov_start < ov_end:
        return ov_start, ov_end

    # No overlap -> between-region
    gap_start = a.end
    gap_end = b.start
    if gap_start < gap_end:
        return gap_start, gap_end

    return None


def infer_events(
    left_segs: List[ForkSeg],
    right_segs: List[ForkSeg],
    min_len: int = 1,
) -> Tuple[
    List[Tuple[str, int, int, str, float, float]],
    List[Tuple[str, int, int, str, float, float]],
    Dict[str, int],
]:
    """
    Returns:
      origins:      (chrom, start, end, readID, left_grad, right_grad)
      terminations: (chrom, start, end, readID, right_grad, left_grad)
      stats: counts of skipped/kept events
    """
    all_segs = left_segs + right_segs
    by_key = group_by_read_and_chrom(all_segs)

    origins: List[Tuple[str, int, int, str, float, float]] = []
    terms: List[Tuple[str, int, int, str, float, float]] = []

    stats = {
        "pairs_total": 0,
        "pairs_containment_skipped": 0,
        "pairs_empty_skipped": 0,
        "pairs_minlen_skipped": 0,
        "origins_kept": 0,
        "terminations_kept": 0,
    }

    for (read_id, chrom), seglist in by_key.items():
        seglist_sorted = sorted(seglist, key=lambda x: (x.start, x.end))

        for i in range(len(seglist_sorted) - 1):
            a = seglist_sorted[i]
            b = seglist_sorted[i + 1]
            stats["pairs_total"] += 1

            # Containment rejection happens inside compute_interval_no_containment
            # We count it by checking the condition here too (for stats only)
            a_contains_b = (a.start <= b.start) and (a.end >= b.end)
            b_contains_a = (b.start <= a.start) and (b.end >= a.end)

            interval = compute_interval_no_containment(a, b)
            if interval is None:
                if a_contains_b or b_contains_a:
                    stats["pairs_containment_skipped"] += 1
                else:
                    stats["pairs_empty_skipped"] += 1
                continue

            start, end = interval
            if end - start < min_len:
                stats["pairs_minlen_skipped"] += 1
                continue

            # Origin: L -> R
            if a.kind == "L" and b.kind == "R":
                origins.append((chrom, start, end, read_id, a.grad, b.grad))
                stats["origins_kept"] += 1

            # Termination: R -> L
            elif a.kind == "R" and b.kind == "L":
                terms.append((chrom, start, end, read_id, a.grad, b.grad))
                stats["terminations_kept"] += 1

            # same-kind adjacency ignored

    return origins, terms, stats


def write_bed6(path: str, rows: List[Tuple[str, int, int, str, float, float]]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for chrom, start, end, read_id, g1, g2 in rows:
            out.write(f"{chrom}\t{start}\t{end}\t{read_id}\t{g1}\t{g2}\n")


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Infer origins and terminations from left/right fork BED files (per read)."
    )
    p.add_argument("--left", required=True, help="Left forks BED: chrom start end readID gradient")
    p.add_argument("--right", required=True, help="Right forks BED: chrom start end readID gradient")
    p.add_argument("--origins", default="origins.bed", help="Output origins BED (default: origins.bed)")
    p.add_argument("--terminations", default="terminations.bed", help="Output terminations BED (default: terminations.bed)")
    p.add_argument("--min-len", type=int, default=1, help="Minimum interval length (bp) to keep (default: 1)")
    p.add_argument("--quiet", action="store_true", help="Suppress stderr summary")
    args = p.parse_args(argv)

    left = parse_fork_bed(args.left, kind="L")
    right = parse_fork_bed(args.right, kind="R")

    origins, terms, stats = infer_events(left, right, min_len=args.min_len)

    write_bed6(args.origins, origins)
    write_bed6(args.terminations, terms)

    if not args.quiet:
        sys.stderr.write(f"Wrote {len(origins)} origins to {args.origins}\n")
        sys.stderr.write(f"Wrote {len(terms)} terminations to {args.terminations}\n")
        sys.stderr.write(
            "Stats: "
            + ", ".join([f"{k}={v}" for k, v in stats.items()])
            + "\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
