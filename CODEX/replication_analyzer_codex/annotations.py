"""Annotation loading helpers with no TensorFlow dependency."""

from __future__ import annotations

import pandas as pd

from .weak_labels import load_bed_events


def load_annotations_for_codex(config: dict):
    """Load ORI/fork annotations and only use TER if explicitly provided."""
    # Use load_bed_events (4-column) for forks so both original (7-8 col) and
    # combined pseudo-label BED files (4-col) are handled transparently.
    left_forks = load_bed_events(config["data"]["left_forks_bed"])
    right_forks = load_bed_events(config["data"]["right_forks_bed"])
    print(f"  Left forks: {len(left_forks):,} regions from {left_forks['read_id'].nunique():,} reads")
    print(f"  Right forks: {len(right_forks):,} regions from {right_forks['read_id'].nunique():,} reads")
    origins = load_bed_events(config["data"]["ori_annotations_bed"])

    termination_bed = config["data"].get("termination_annotations_bed")
    if termination_bed:
        print(f"  Loading termination annotations from file: {termination_bed}")
        terminations = load_bed_events(termination_bed)
    else:
        print("  No termination annotation file configured; terminations disabled")
        terminations = pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    return left_forks, right_forks, origins, terminations
