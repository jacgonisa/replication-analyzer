"""Event-aware evaluation for weakly supervised ORI/TER/fork models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from replication_analyzer.data.loaders import load_all_xy_data

from .constants import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, EVENT_CLASS_IDS
from .representation import encode_read_dataframe


def predict_reads(
    model,
    xy_data: pd.DataFrame,
    read_ids: Iterable[str],
    max_length: int,
    preprocessing_config: dict,
) -> pd.DataFrame:
    """Predict per-window probabilities for selected reads."""
    rows = []
    for read_id in read_ids:
        read_df = xy_data[xy_data["read_id"] == read_id].copy()
        read_df = read_df.sort_values("start").reset_index(drop=True)
        if len(read_df) == 0:
            continue

        x_encoded = encode_read_dataframe(read_df, preprocessing_config)
        x_padded = np.zeros((1, max_length, x_encoded.shape[1]), dtype=np.float32)
        use_len = min(len(x_encoded), max_length)
        x_padded[0, :use_len, :] = x_encoded[:use_len, :]

        y_pred = model.predict(x_padded, verbose=0)[0][:use_len]
        predicted_class = np.argmax(y_pred, axis=-1)
        sorted_probs = np.sort(y_pred, axis=-1)
        top_prob = sorted_probs[:, -1]
        second_prob = sorted_probs[:, -2]
        entropy = -np.sum(y_pred * np.log(np.clip(y_pred, 1e-7, 1.0)), axis=-1)

        base_cols = ["chr", "start", "end", "read_id"]
        if "signal" in read_df.columns:
            base_cols.append("signal")
        export_df = read_df.iloc[:use_len][base_cols].copy()
        export_df["predicted_class"] = predicted_class
        export_df["top_prob"] = top_prob
        export_df["margin"] = top_prob - second_prob
        export_df["entropy"] = entropy
        n_out = y_pred.shape[-1]
        for class_id, class_name in CLASS_ID_TO_NAME.items():
            if class_id < n_out:
                export_df[f"prob_{class_name}"] = y_pred[:, class_id]
        rows.append(export_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def resolve_window_conflicts(
    predictions: pd.DataFrame,
    threshold_map: dict,
    priority_class: str = "origin",
) -> pd.DataFrame:
    """Suppress ambiguous windows where multiple classes exceed their threshold.

    For each window claimed by 2+ classes, all event-class probabilities are
    zeroed out — the window is treated as background.  Ambiguous windows carry
    no reliable signal so claiming any winner would add noise to event calls.

    Args:
        predictions:    DataFrame from predict_reads() with prob_{class} columns.
        threshold_map:  {class_name: prob_threshold} — same map used by reannotation.
        priority_class: Kept for API compatibility; no longer used.

    Returns:
        Copy of predictions with all event-class prob columns zeroed on conflict
        windows.  The raw segments TSV should be saved *before* calling this
        function so the original probabilities are preserved there.
    """
    pred = predictions.copy()
    event_classes = [c for c in threshold_map if f"prob_{c}" in pred.columns]

    # Boolean mask per class: does this window exceed its threshold?
    masks = {c: pred[f"prob_{c}"] >= threshold_map[c] for c in event_classes}

    # Windows where 2+ classes are active — zero ALL event classes
    n_active = sum(m.astype(int) for m in masks.values())
    conflict_idx = n_active[n_active >= 2].index

    if len(conflict_idx) == 0:
        return pred

    for c in event_classes:
        pred.loc[conflict_idx, f"prob_{c}"] = 0.0

    print(f"resolve_window_conflicts: suppressed {len(conflict_idx):,} ambiguous windows "
          f"(all event classes zeroed)")
    return pred


def windows_to_events(
    predictions: pd.DataFrame,
    class_id: int,
    prob_threshold: float,
    min_windows: int = 1,
    max_gap: int = 5000,
) -> pd.DataFrame:
    """Convert per-window probabilities into event regions for one class."""
    class_name = CLASS_ID_TO_NAME[class_id]
    prob_col = f"prob_{class_name}"
    if prob_col not in predictions.columns:
        raise ValueError(f"Missing probability column: {prob_col}")

    class_windows = predictions[predictions[prob_col] >= prob_threshold].copy()
    if len(class_windows) == 0:
        return pd.DataFrame(columns=["read_id", "chr", "start", "end", "event_type"])

    class_windows = class_windows.sort_values(["read_id", "start"]).reset_index(drop=True)
    has_signal = "signal" in class_windows.columns

    events = []
    current = None
    for row in class_windows.itertuples(index=False):
        if current is None:
            current = {
                "read_id": row.read_id,
                "chr": row.chr,
                "start": int(row.start),
                "end": int(row.end),
                "n_windows": 1,
                "probs": [float(getattr(row, prob_col))],
                "margins": [float(row.margin)],
                "entropies": [float(row.entropy)],
                "signals": [float(row.signal)] if has_signal else [],
                "signal_starts": [int(row.start)] if has_signal else [],
            }
            continue

        same_read = row.read_id == current["read_id"]
        close_enough = int(row.start) - current["end"] <= max_gap
        if same_read and close_enough:
            current["end"] = max(current["end"], int(row.end))
            current["n_windows"] += 1
            current["probs"].append(float(getattr(row, prob_col)))
            current["margins"].append(float(row.margin))
            current["entropies"].append(float(row.entropy))
            if has_signal:
                current["signals"].append(float(row.signal))
                current["signal_starts"].append(int(row.start))
        else:
            if current["n_windows"] >= min_windows:
                events.append(current)
            current = {
                "read_id": row.read_id,
                "chr": row.chr,
                "start": int(row.start),
                "end": int(row.end),
                "n_windows": 1,
                "probs": [float(getattr(row, prob_col))],
                "margins": [float(row.margin)],
                "entropies": [float(row.entropy)],
                "signals": [float(row.signal)] if has_signal else [],
                "signal_starts": [int(row.start)] if has_signal else [],
            }

    if current is not None and current["n_windows"] >= min_windows:
        events.append(current)

    normalized = []
    for event in events:
        probs = np.array(event.pop("probs"))
        margins = np.array(event.pop("margins"))
        entropies = np.array(event.pop("entropies"))
        signals = event.pop("signals")
        signal_starts = event.pop("signal_starts")
        row_out = {
            **event,
            "event_type": class_name,
            "length": event["end"] - event["start"],
            "max_prob": float(probs.max()),
            "mean_prob": float(probs.mean()),
            "mean_margin": float(margins.mean()),
            "mean_entropy": float(entropies.mean()),
            "confidence_score": float(0.6 * probs.mean() + 0.4 * margins.mean()),
        }
        if signals:
            if class_name == "origin":
                row_out["mean_brdu_signal"] = float(np.mean(signals))
            else:
                # slope in BrdU/bp — positive = signal rises left→right (right fork pattern)
                if len(signals) >= 2:
                    row_out["brdu_slope"] = float(np.polyfit(signal_starts, signals, 1)[0])
                else:
                    row_out["brdu_slope"] = float("nan")
        normalized.append(row_out)
    return pd.DataFrame(normalized)


def evaluate_event_predictions(
    pred_events: pd.DataFrame,
    gt_events: pd.DataFrame,
    iou_threshold: float,
) -> Dict[str, float]:
    """Evaluate event regions using one-to-one IoU matching."""
    if len(pred_events) == 0 and len(gt_events) == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "num_predictions": 0,
            "num_ground_truth": 0,
        }
    if len(pred_events) == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(gt_events),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_predictions": 0,
            "num_ground_truth": len(gt_events),
        }
    if len(gt_events) == 0:
        return {
            "true_positives": 0,
            "false_positives": len(pred_events),
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_predictions": len(pred_events),
            "num_ground_truth": 0,
        }

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious = []

    for read_id in set(pred_events["read_id"]).union(set(gt_events["read_id"])):
        pred_read = pred_events[pred_events["read_id"] == read_id]
        gt_read = gt_events[gt_events["read_id"] == read_id]
        matched_gt = set()
        for pred_idx, pred_row in pred_read.iterrows():
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt_row in gt_read.iterrows():
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(int(pred_row.start), int(pred_row.end), int(gt_row.start), int(gt_row.end))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_gt_idx)
                matched_ious.append(best_iou)
            else:
                total_fp += 1
        total_fn += len(gt_read) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    event_iou = float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0
    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_iou": event_iou,
        "num_predictions": total_tp + total_fp,
        "num_ground_truth": total_tp + total_fn,
    }


def run_threshold_sweep(
    predictions: pd.DataFrame,
    ground_truth_by_class: Dict[int, pd.DataFrame],
    probability_thresholds: List[float],
    iou_thresholds: List[float],
    min_windows: int,
    max_gap: int,
    max_gap_per_class: dict | None = None,
    active_class_ids: List[int] | None = None,
) -> pd.DataFrame:
    """Grid search over class probability thresholds and event IoU thresholds."""
    if active_class_ids is None:
        active_class_ids = EVENT_CLASS_IDS
    if max_gap_per_class is None:
        max_gap_per_class = {}
    rows = []
    for class_id in active_class_ids:
        class_name = CLASS_ID_TO_NAME[class_id]
        effective_max_gap = max_gap_per_class.get(class_name, max_gap)
        for prob_threshold in probability_thresholds:
            pred_events = windows_to_events(
                predictions=predictions,
                class_id=class_id,
                prob_threshold=prob_threshold,
                min_windows=min_windows,
                max_gap=effective_max_gap,
            )
            for iou_threshold in iou_thresholds:
                metrics = evaluate_event_predictions(
                    pred_events=pred_events,
                    gt_events=ground_truth_by_class[class_id],
                    iou_threshold=iou_threshold,
                )
                rows.append(
                    {
                        "class_id": class_id,
                        "event_type": CLASS_ID_TO_NAME[class_id],
                        "prob_threshold": prob_threshold,
                        "iou_threshold": iou_threshold,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def resolve_active_class_ids(event_types: List[str] | None = None, n_classes: int | None = None) -> List[int]:
    """Resolve which non-background classes are active for a run."""
    if event_types:
        return [CLASS_NAME_TO_ID[name] for name in event_types]
    if n_classes is None:
        return EVENT_CLASS_IDS
    return list(range(1, n_classes))


def load_xy_for_prediction(config: dict) -> pd.DataFrame:
    cache_path = config["data"].get("xy_cache_path")
    if cache_path:
        from pathlib import Path
        import pickle
        p = Path(cache_path)
        if p.exists():
            print(f"📂 Loading XY data from cache: {p}", flush=True)
            with open(p, "rb") as fh:
                return pickle.load(fh)
    return load_all_xy_data(
        base_dir=config["data"]["base_dir"],
        run_dirs=config["data"].get("run_dirs"),
    )
