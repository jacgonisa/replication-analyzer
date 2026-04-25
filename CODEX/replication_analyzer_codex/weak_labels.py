"""Weak labels, derived terminations, and masked-background handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .constants import CLASS_NAME_TO_ID, IGNORE_INDEX


@dataclass(frozen=True)
class SegmentEvent:
    chrom: str
    start: int
    end: int
    read_id: str
    kind: str


def load_bed_events(path: str) -> pd.DataFrame:
    """Load the first four BED columns as event intervals."""
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3])
    df.columns = ["chr", "start", "end", "read_id"]
    return df


def _fork_rows_to_events(df: pd.DataFrame, kind: str) -> List[SegmentEvent]:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            SegmentEvent(
                chrom=str(row.chr),
                start=int(row.start),
                end=int(row.end),
                read_id=str(row.read_id),
                kind=kind,
            )
        )
    return rows


def derive_termination_annotations(
    left_forks: pd.DataFrame,
    right_forks: pd.DataFrame,
    min_len: int = 1,
) -> pd.DataFrame:
    """Derive termination intervals from adjacent right->left fork pairs."""
    all_events = _fork_rows_to_events(left_forks, "L") + _fork_rows_to_events(right_forks, "R")
    grouped: Dict[Tuple[str, str], List[SegmentEvent]] = {}
    for event in all_events:
        grouped.setdefault((event.read_id, event.chrom), []).append(event)

    terminations = []
    for (read_id, chrom), events in grouped.items():
        ordered = sorted(events, key=lambda x: (x.start, x.end))
        for idx in range(len(ordered) - 1):
            first = ordered[idx]
            second = ordered[idx + 1]
            if first.kind != "R" or second.kind != "L":
                continue

            first_contains_second = first.start <= second.start and first.end >= second.end
            second_contains_first = second.start <= first.start and second.end >= first.end
            if first_contains_second or second_contains_first:
                continue

            overlap_start = max(first.start, second.start)
            overlap_end = min(first.end, second.end)
            if overlap_start < overlap_end:
                start, end = overlap_start, overlap_end
            else:
                start, end = first.end, second.start

            if end - start >= min_len:
                terminations.append(
                    {
                        "chr": chrom,
                        "start": int(start),
                        "end": int(end),
                        "read_id": read_id,
                    }
                )

    if not terminations:
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return pd.DataFrame(terminations)


def build_annotation_dict(
    left_forks: pd.DataFrame,
    right_forks: pd.DataFrame,
    origins: pd.DataFrame,
    terminations: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Build a shared annotation dictionary keyed by class name."""
    return {
        "left_fork": left_forks[["chr", "start", "end", "read_id"]].copy(),
        "right_fork": right_forks[["chr", "start", "end", "read_id"]].copy(),
        "origin": origins[["chr", "start", "end", "read_id"]].copy(),
        "termination": terminations[["chr", "start", "end", "read_id"]].copy(),
    }


def _overlap_any(seg_start: int, seg_end: int, intervals: Iterable[Tuple[int, int]]) -> bool:
    for start, end in intervals:
        if max(seg_start, start) < min(seg_end, end):
            return True
    return False


def _expand_intervals(intervals: List[Tuple[int, int]], margin_bp: int) -> List[Tuple[int, int]]:
    if margin_bp <= 0:
        return intervals
    return [(start - margin_bp, end + margin_bp) for start, end in intervals]


def build_weak_labels_for_read(
    read_df: pd.DataFrame,
    annotation_dict: Dict[str, pd.DataFrame],
    background_margin_bp: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Assign weak labels for one read.

    Priority:
    1. left fork
    2. right fork
    3. origin       (forks win — ~53% of Nerea ORIs overlap pseudo-fork regions;
                     giving ORI priority caused conflicting gradients since
                     fork-like signal was labeled ORI, destabilising training)
    4. termination
    5. trusted background if far from any positive interval
    6. ignore otherwise
    """
    read_id = str(read_df["read_id"].iloc[0])
    chrom = str(read_df["chr"].iloc[0])

    labels = np.full(len(read_df), IGNORE_INDEX, dtype=np.int32)

    read_intervals_by_class: Dict[str, List[Tuple[int, int]]] = {}
    for class_name, df in annotation_dict.items():
        subset = df[(df["read_id"] == read_id) & (df["chr"] == chrom)]
        read_intervals_by_class[class_name] = [
            (int(row.start), int(row.end))
            for row in subset.itertuples(index=False)
        ]

    for class_name in ["left_fork", "right_fork", "origin", "termination"]:
        class_id = CLASS_NAME_TO_ID[class_name]
        intervals = read_intervals_by_class[class_name]
        if not intervals:
            continue
        for idx, row in enumerate(read_df.itertuples(index=False)):
            if labels[idx] != IGNORE_INDEX:
                continue
            if _overlap_any(int(row.start), int(row.end), intervals):
                labels[idx] = class_id

    all_positive_intervals = []
    for intervals in read_intervals_by_class.values():
        all_positive_intervals.extend(intervals)
    padded_positive_intervals = _expand_intervals(all_positive_intervals, background_margin_bp)

    for idx, row in enumerate(read_df.itertuples(index=False)):
        if labels[idx] != IGNORE_INDEX:
            continue
        seg_start = int(row.start)
        seg_end = int(row.end)
        if not _overlap_any(seg_start, seg_end, padded_positive_intervals):
            labels[idx] = CLASS_NAME_TO_ID["background"]

    sample_weight = (labels != IGNORE_INDEX).astype(np.float32)
    y_safe = labels.copy()
    y_safe[y_safe == IGNORE_INDEX] = CLASS_NAME_TO_ID["background"]

    stats = {
        "read_id": read_id,
        "length": len(read_df),
        "has_left": bool((labels == CLASS_NAME_TO_ID["left_fork"]).any()),
        "has_right": bool((labels == CLASS_NAME_TO_ID["right_fork"]).any()),
        "has_origin": bool((labels == CLASS_NAME_TO_ID["origin"]).any()),
        "has_termination": bool((labels == CLASS_NAME_TO_ID["termination"]).any()),
        "has_any_event": bool((labels > 0).any()),
        "n_background": int((labels == 0).sum()),
        "n_left": int((labels == 1).sum()),
        "n_right": int((labels == 2).sum()),
        "n_origin": int((labels == 3).sum()),
        "n_termination": int((labels == 4).sum()),
        "n_unknown": int((labels == IGNORE_INDEX).sum()),
    }
    return labels, sample_weight, {"y_safe": y_safe, **stats}
