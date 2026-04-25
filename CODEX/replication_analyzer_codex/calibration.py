"""Event-level confidence calibration for CODEX exports."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .constants import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID
from .evaluation import compute_iou


def label_event_matches(
    pred_events: pd.DataFrame,
    gt_events: pd.DataFrame,
    iou_threshold: float,
) -> pd.DataFrame:
    """Annotate predicted events as matched/unmatched using one-to-one IoU matching."""
    if len(pred_events) == 0:
        return pred_events.assign(is_match=pd.Series(dtype=int), best_iou=pd.Series(dtype=float))

    labeled = pred_events.copy()
    labeled["is_match"] = 0
    labeled["best_iou"] = 0.0

    for read_id in set(labeled["read_id"]).union(set(gt_events["read_id"])):
        pred_idx = labeled.index[labeled["read_id"] == read_id].tolist()
        gt_read = gt_events[gt_events["read_id"] == read_id]
        matched_gt = set()

        for idx in pred_idx:
            pred_row = labeled.loc[idx]
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt_row in gt_read.iterrows():
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(int(pred_row.start), int(pred_row.end), int(gt_row.start), int(gt_row.end))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            labeled.at[idx, "best_iou"] = best_iou
            if best_iou >= iou_threshold:
                labeled.at[idx, "is_match"] = 1
                matched_gt.add(best_gt_idx)
    return labeled


def fit_isotonic_calibrators(
    event_tables_by_type: Dict[str, pd.DataFrame],
    score_column: str = "confidence_score",
) -> Dict[str, IsotonicRegression]:
    """Fit one isotonic calibrator per event type."""
    calibrators = {}
    for event_type, df in event_tables_by_type.items():
        if len(df) == 0:
            continue
        if df["is_match"].nunique() < 2:
            continue
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(df[score_column].to_numpy(), df["is_match"].to_numpy())
        calibrators[event_type] = calibrator
    return calibrators


def apply_calibrators(
    events: pd.DataFrame,
    calibrators: Dict[str, IsotonicRegression],
    score_column: str = "confidence_score",
) -> pd.DataFrame:
    """Apply class-specific calibrators to event tables."""
    if len(events) == 0:
        events["calibrated_confidence"] = []
        return events

    calibrated = events.copy()
    calibrated["calibrated_confidence"] = calibrated[score_column]
    for event_type, calibrator in calibrators.items():
        mask = calibrated["event_type"] == event_type
        if not mask.any():
            continue
        calibrated.loc[mask, "calibrated_confidence"] = calibrator.predict(
            calibrated.loc[mask, score_column].to_numpy()
        )
    return calibrated


def save_calibrators(calibrators: Dict[str, IsotonicRegression], output_path: str) -> None:
    """Persist fitted calibrators."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrators, output_path)


def load_calibrators(path: str) -> Dict[str, IsotonicRegression]:
    """Load persisted calibrators."""
    return joblib.load(path)
