#!/usr/bin/env python
"""Run event-level threshold sweep evaluation for the CODEX model."""

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
from replication_analyzer_codex.annotations import load_annotations_for_codex
from replication_analyzer_codex.calibration import apply_calibrators, fit_isotonic_calibrators, label_event_matches, save_calibrators
from replication_analyzer_codex.evaluation import (
    load_xy_for_prediction,
    predict_reads,
    resolve_active_class_ids,
    run_threshold_sweep,
    windows_to_events,
)
from replication_analyzer_codex.losses import (
    MaskedClassPrecision, MaskedClassRecall, MaskedMacroF1,
    MaskedMeanIoU, SparseCategoricalFocalLoss,
)
from replication_analyzer_codex.splits import load_split_manifest
from replication_analyzer_codex.constants import CLASS_NAME_TO_ID


def _load_model(model_path: str):
    custom_objects = {
        "SelfAttention": SelfAttention,
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
        "MaskedMacroF1": MaskedMacroF1,
        "MaskedMeanIoU": MaskedMeanIoU,
        "MaskedClassPrecision": MaskedClassPrecision,
        "MaskedClassRecall": MaskedClassRecall,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CODEX weak 5-event model")
    parser.add_argument("--config", required=True, help="Path to CODEX YAML config")
    parser.add_argument("--model", required=True, help="Path to trained .keras model")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Which split to evaluate")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    manifest_path = config["data"].get("split_manifest")
    if not manifest_path:
        manifest_path = str(Path(config["output"]["results_dir"]) / "split_manifest.tsv")
    manifest = load_split_manifest(manifest_path)
    read_ids = manifest.loc[manifest["split"] == args.split, "read_id"].tolist()

    model = _load_model(args.model)
    max_length = model.input_shape[1]

    xy_data = load_xy_for_prediction(config)
    left_forks, right_forks, origins, terminations = load_annotations_for_codex(config)
    gt_by_class = {
        CLASS_NAME_TO_ID["left_fork"]: left_forks,
        CLASS_NAME_TO_ID["right_fork"]: right_forks,
        CLASS_NAME_TO_ID["origin"]: origins,
    }
    if config["model"].get("n_classes", 4) > 4:
        gt_by_class[CLASS_NAME_TO_ID["termination"]] = terminations
    for class_id in list(gt_by_class):
        gt_by_class[class_id] = gt_by_class[class_id][gt_by_class[class_id]["read_id"].isin(read_ids)].copy()
    active_class_ids = resolve_active_class_ids(
        event_types=config.get("evaluation", {}).get("active_event_types"),
        n_classes=config["model"].get("n_classes", 4),
    )

    predictions = predict_reads(
        model=model,
        xy_data=xy_data,
        read_ids=read_ids,
        max_length=max_length,
        preprocessing_config=config["preprocessing"],
    )

    output_dir = Path(config["output"]["results_dir"]) / f"evaluation_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)

    sweep = run_threshold_sweep(
        predictions=predictions,
        ground_truth_by_class=gt_by_class,
        probability_thresholds=config["evaluation"].get("probability_thresholds", [0.3, 0.4, 0.5, 0.6]),
        iou_thresholds=config["evaluation"].get("iou_thresholds", [0.2, 0.3, 0.4, 0.5]),
        min_windows=config["evaluation"].get("min_windows", 1),
        max_gap=config["evaluation"].get("max_gap", 5000),
        active_class_ids=active_class_ids,
    )
    sweep.to_csv(output_dir / "threshold_sweep.tsv", sep="\t", index=False)

    best_rows = (
        sweep.sort_values(["event_type", "f1"], ascending=[True, False])
        .groupby("event_type", as_index=False)
        .first()
    )
    best_rows.to_csv(output_dir / "best_thresholds.tsv", sep="\t", index=False)

    calibration_tables = {}
    for row in best_rows.itertuples(index=False):
        class_id = CLASS_NAME_TO_ID[row.event_type]
        events = windows_to_events(
            predictions=predictions,
            class_id=class_id,
            prob_threshold=float(row.prob_threshold),
            min_windows=config["evaluation"].get("min_windows", 1),
            max_gap=config["evaluation"].get("max_gap", 5000),
        )
        labeled_events = label_event_matches(
            pred_events=events,
            gt_events=gt_by_class[class_id],
            iou_threshold=float(row.iou_threshold),
        )
        labeled_events.to_csv(output_dir / f"events_{row.event_type}.tsv", sep="\t", index=False)
        calibration_tables[row.event_type] = labeled_events

    calibrators = fit_isotonic_calibrators(calibration_tables)
    if calibrators:
        save_calibrators(calibrators, str(output_dir / "event_calibrators.joblib"))
        calibrated_tables = []
        for event_type, df in calibration_tables.items():
            calibrated_tables.append(apply_calibrators(df, calibrators))
        pd.concat(calibrated_tables, ignore_index=True).to_csv(
            output_dir / "events_calibrated.tsv",
            sep="\t",
            index=False,
        )

    print("Evaluation complete.")
    print(best_rows[["event_type", "prob_threshold", "iou_threshold", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
