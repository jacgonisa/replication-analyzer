"""EventLevelIoUCallback — ForkML-style event-level IoU computed each epoch.

Instead of the window-level Jaccard (MaskedMeanIoU), this callback:
  1. Runs model.predict on the full val set each epoch
  2. Merges consecutive high-probability windows into events (same logic as
     windows_to_events, but in window-index space rather than genomic space)
  3. For each GT annotation event (contiguous run of class k in y_true),
     finds the best-matching predicted event and records its IoU  → recall-side
  4. For each predicted event, finds the best-matching GT event   → precision-side
  5. Reports mean IoU (recall-anchored), event precision, event recall, and
     event F1 per class, then mean val_event_iou across foreground classes.

Adds to Keras logs:
  val_event_iou          — mean IoU across LF, RF, ORI (primary monitor metric)
  val_event_iou_lf       — left fork IoU only
  val_event_iou_rf       — right fork IoU only
  val_event_iou_ori      — origin IoU only
  val_event_prec_lf/rf/ori  — event-level precision per class
  val_event_rec_lf/rf/ori   — event-level recall per class
  val_event_f1_lf/rf/ori    — event-level F1 per class
  val_event_f1_ori_lf_rf    — 0.5*F1_ori + 0.25*F1_lf + 0.25*F1_rf
                              (ORI-weighted composite, useful for v5.1 monitoring)
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


CLASS_NAMES = {1: "lf", 2: "rf", 3: "ori"}


def _extract_events(binary_mask: np.ndarray, max_gap: int = 5):
    """Return list of (start, end) index pairs from a boolean 1-D array.

    Consecutive True regions separated by ≤ max_gap False positions are
    merged into a single event (mimics windows_to_events max_gap logic).
    end is exclusive (half-open interval).
    """
    events = []
    n = len(binary_mask)
    in_event = False
    start = 0
    gap = 0

    for i in range(n):
        if binary_mask[i]:
            if not in_event:
                start = i
                in_event = True
            gap = 0
        else:
            if in_event:
                gap += 1
                if gap > max_gap:
                    events.append((start, i - gap + 1))
                    in_event = False
                    gap = 0

    if in_event:
        end = n
        # trim trailing gap
        while end > start and not binary_mask[end - 1]:
            end -= 1
        events.append((start, end))

    return events


def _iou(s1: int, e1: int, s2: int, e2: int) -> float:
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


class EventLevelIoUCallback(tf.keras.callbacks.Callback):
    """Computes ForkML-style event-level IoU on the validation set.

    Parameters
    ----------
    val_x : np.ndarray  shape (N, T, C)
    val_y : np.ndarray  shape (N, T)   — integer class labels
    val_w : np.ndarray  shape (N, T)   — sample weights (0 = padded)
    prob_threshold : float
        Probability threshold for calling a window as foreground.
    max_gap_windows : int
        Max number of sub-threshold windows bridged when merging events.
        At ~1 kb/window, 5 ≈ 5 kb gap (matches windows_to_events default).
    n_classes : int
    batch_size : int
        Batch size for model.predict (larger = faster on CPU).
    compute_every : int
        Compute only every N epochs (1 = every epoch).
    verbose : bool
    """

    def __init__(
        self,
        val_x: np.ndarray,
        val_y: np.ndarray,
        val_w: np.ndarray,
        prob_threshold: float = 0.4,
        max_gap_windows: int = 5,
        n_classes: int = 4,
        batch_size: int = 256,
        compute_every: int = 1,
        verbose: bool = True,
    ):
        super().__init__()
        self.val_x = val_x
        self.val_y = val_y.astype(np.int32)
        self.val_w = val_w
        self.prob_threshold = prob_threshold
        self.max_gap_windows = max_gap_windows
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.compute_every = compute_every
        self.verbose = verbose

        # Pre-identify valid (non-padded) window indices per read
        self._valid_mask = (val_w > 0)  # (N, T) bool

        # Only process reads that have ≥1 foreground GT label — speeds up
        # the Python loop without losing any signal (background-only reads
        # can never contribute to IoU)
        has_fg = np.any(
            (self.val_y > 0) & self._valid_mask, axis=1
        )
        self._fg_indices = np.where(has_fg)[0]

        if verbose:
            print(
                f"\n  EventLevelIoUCallback: {len(val_x)} val reads  "
                f"({len(self._fg_indices)} with foreground labels)  "
                f"prob_thr={prob_threshold}  max_gap={max_gap_windows} windows  "
                f"compute_every={compute_every}"
            )

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        if logs is None:
            logs = {}

        if (epoch + 1) % self.compute_every != 0:
            return

        # ── 1. Full val inference ─────────────────────────────────────────
        y_prob = self.model.predict(
            self.val_x, batch_size=self.batch_size, verbose=0
        )  # (N, T, n_classes)

        # ── 2. Event-level stats over annotated reads ─────────────────────
        # recall-side: for each GT event, best IoU with any prediction
        class_gt_ious:   dict[int, list[float]] = {k: [] for k in range(1, self.n_classes)}
        # precision-side: for each predicted event, best IoU with any GT event
        class_pred_ious: dict[int, list[float]] = {k: [] for k in range(1, self.n_classes)}

        for i in self._fg_indices:
            valid = self._valid_mask[i]
            y_true_i = self.val_y[i][valid]
            y_prob_i = y_prob[i][valid]

            for class_id in range(1, self.n_classes):
                gt_mask    = (y_true_i == class_id)
                gt_events  = _extract_events(gt_mask, self.max_gap_windows)
                pred_mask  = (y_prob_i[:, class_id] > self.prob_threshold)
                pred_events = _extract_events(pred_mask, self.max_gap_windows)

                # Recall-side: for each GT event → best match in predictions
                for gs, ge in gt_events:
                    best = max((_iou(gs, ge, ps, pe) for ps, pe in pred_events),
                               default=0.0)
                    class_gt_ious[class_id].append(best)

                # Precision-side: for each predicted event → best match in GT
                for ps, pe in pred_events:
                    best = max((_iou(ps, pe, gs, ge) for gs, ge in gt_events),
                               default=0.0)
                    class_pred_ious[class_id].append(best)

        # ── 3. Aggregate and log ──────────────────────────────────────────
        iou_thr = 0.2   # an event is a TP if its best-match IoU >= this

        per_class_iou:  dict[int, float] = {}
        per_class_prec: dict[int, float] = {}
        per_class_rec:  dict[int, float] = {}
        per_class_f1:   dict[int, float] = {}

        for class_id in range(1, self.n_classes):
            gt_ious   = class_gt_ious[class_id]
            pred_ious = class_pred_ious[class_id]

            if gt_ious:
                per_class_iou[class_id] = float(np.mean(gt_ious))
                rec = float(np.mean(np.array(gt_ious) >= iou_thr))
                per_class_rec[class_id] = rec
            else:
                rec = 0.0

            if pred_ious:
                prec = float(np.mean(np.array(pred_ious) >= iou_thr))
                per_class_prec[class_id] = prec
            else:
                prec = 0.0

            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_f1[class_id] = f1

        mean_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0

        # ORI-weighted composite: 0.5*F1_ori + 0.25*F1_lf + 0.25*F1_rf
        # Useful as monitor for v5.1 where ORI labels are most reliable
        f1_ori  = per_class_f1.get(3, 0.0)
        f1_lf   = per_class_f1.get(1, 0.0)
        f1_rf   = per_class_f1.get(2, 0.0)
        iou_lf  = per_class_iou.get(1, 0.0)
        iou_rf  = per_class_iou.get(2, 0.0)
        rec_ori = per_class_rec.get(3, 0.0)
        ori_weighted_f1 = 0.5 * f1_ori + 0.25 * f1_lf + 0.25 * f1_rf

        # Recall-weighted composite: 0.5*rec_ori + 0.25*iou_lf + 0.25*iou_rf
        # Prioritises ORI recall (user cares most about finding ORIs).
        # Fork quality measured by IoU (boundary precision), not F1.
        # Use as monitor for v5.3+.
        rec_weighted = 0.5 * rec_ori + 0.25 * iou_lf + 0.25 * iou_rf

        logs["val_event_iou"] = mean_iou
        logs["val_event_f1_ori_lf_rf"] = ori_weighted_f1
        logs["val_event_rec_weighted"]  = rec_weighted
        for class_id in range(1, self.n_classes):
            name = CLASS_NAMES[class_id]
            if class_id in per_class_iou:
                logs[f"val_event_iou_{name}"]  = per_class_iou[class_id]
            if class_id in per_class_prec:
                logs[f"val_event_prec_{name}"] = per_class_prec[class_id]
            if class_id in per_class_rec:
                logs[f"val_event_rec_{name}"]  = per_class_rec[class_id]
            logs[f"val_event_f1_{name}"] = per_class_f1[class_id]

        if self.verbose:
            detail = "  ".join(
                f"{CLASS_NAMES[k].upper()} IoU={per_class_iou.get(k,0):.3f} "
                f"P={per_class_prec.get(k,0):.3f} R={per_class_rec.get(k,0):.3f} "
                f"F1={per_class_f1.get(k,0):.3f}"
                for k in range(1, self.n_classes)
            )
            print(f"\n  ── EventStats epoch {epoch+1}: "
                  f"mean_iou={mean_iou:.4f}  ori_wtd_f1={ori_weighted_f1:.4f}"
                  f"\n     {detail}")
