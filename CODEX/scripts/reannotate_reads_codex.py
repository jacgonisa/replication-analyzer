#!/usr/bin/env python
"""Export fully reannotated reads with event-level confidence scores."""

import argparse
from pathlib import Path
import sys

import pandas as pd
import tensorflow as tf
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer.models.base import SelfAttention
from replication_analyzer_codex.calibration import apply_calibrators, load_calibrators
from replication_analyzer_codex.constants import CLASS_NAME_TO_ID
from replication_analyzer_codex.evaluation import load_xy_for_prediction, predict_reads, resolve_window_conflicts, windows_to_events
from replication_analyzer_codex.losses import (
    MaskedMacroF1, MaskedMeanIoU, MaskedClassRecall, MaskedClassPrecision,
    SparseCategoricalFocalLoss,
)


def _load_model(model_path: str):
    custom_objects = {
        "SelfAttention": SelfAttention,
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
        "MaskedMacroF1": MaskedMacroF1,
        "MaskedMeanIoU": MaskedMeanIoU,
        "MaskedClassRecall": MaskedClassRecall,
        "MaskedClassPrecision": MaskedClassPrecision,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)


def main():
    parser = argparse.ArgumentParser(description="Reannotate reads with CODEX model")
    parser.add_argument("--config", required=True, help="Path to CODEX YAML config")
    parser.add_argument("--model", required=True, help="Path to trained .keras model")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--calibrators", help="Optional path to event calibrator joblib")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model = _load_model(args.model)
    max_length = model.input_shape[1]
    xy_data = load_xy_for_prediction(config)
    read_ids = xy_data["read_id"].unique().tolist()

    predictions = predict_reads(
        model=model,
        xy_data=xy_data,
        read_ids=read_ids,
        max_length=max_length,
        preprocessing_config=config["preprocessing"],
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "reannotated_segments.tsv", sep="\t", index=False)

    threshold_map = config["reannotation"].get(
        "probability_thresholds",
        {
            "left_fork": 0.4,
            "right_fork": 0.4,
            "origin": 0.4,
        },
    )
    predictions = resolve_window_conflicts(
        predictions=predictions,
        threshold_map=threshold_map,
        priority_class=config["reannotation"].get("conflict_priority_class", "origin"),
    )
    default_max_gap = config["reannotation"].get("max_gap", 5000)
    max_gap_per_class = config["reannotation"].get("max_gap_per_class", {})
    all_events = []
    for event_type, prob_threshold in threshold_map.items():
        max_gap = max_gap_per_class.get(event_type, default_max_gap)
        events = windows_to_events(
            predictions=predictions,
            class_id=CLASS_NAME_TO_ID[event_type],
            prob_threshold=prob_threshold,
            min_windows=config["reannotation"].get("min_windows", 1),
            max_gap=max_gap,
        )
        if len(events) == 0:
            continue
        events["confidence_tier"] = pd.cut(
            events["confidence_score"],
            bins=[-1.0, 0.35, 0.55, 0.75, 1.01],
            labels=["very_low", "low", "medium", "high"],
        ).astype(str)
        all_events.append(events)

    if all_events:
        combined_events = pd.concat(all_events, ignore_index=True)
    else:
        combined_events = pd.DataFrame()

    # Slope filter: suppress fork events with |brdu_slope| below threshold.
    # Flat forks (slope ≈ 0) are likely null-BrdU ORI reads misclassified as forks.
    min_fork_slope = config["reannotation"].get("min_fork_slope", None)
    if min_fork_slope is not None and len(combined_events) > 0 and "brdu_slope" in combined_events.columns:
        fork_mask = combined_events["event_type"].isin(["left_fork", "right_fork"])
        flat_mask = fork_mask & (combined_events["brdu_slope"].abs() < min_fork_slope)
        n_removed = flat_mask.sum()
        if n_removed > 0:
            combined_events = combined_events[~flat_mask].reset_index(drop=True)
            print(f"Slope filter: removed {n_removed:,} flat fork events (|brdu_slope| < {min_fork_slope})")

    if args.calibrators and len(combined_events) > 0:
        calibrators = load_calibrators(args.calibrators)
        combined_events = apply_calibrators(combined_events, calibrators)
    elif len(combined_events) > 0:
        combined_events["calibrated_confidence"] = combined_events["confidence_score"]

    combined_events.to_csv(output_dir / "reannotated_events.tsv", sep="\t", index=False)
    print(f"Saved segments: {output_dir / 'reannotated_segments.tsv'}")
    print(f"Saved events: {output_dir / 'reannotated_events.tsv'}")


if __name__ == "__main__":
    main()
