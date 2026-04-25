"""CODEX loss functions and masked metrics."""

from __future__ import annotations

import tensorflow as tf


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Sparse categorical focal loss for per-window sequence labeling."""

    def __init__(self, alpha=None, gamma=2.0, name="sparse_categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        # Pre-build the constant so call() never calls tf.constant() inside a tf.function,
        # which would grow TF's tracing cache and cause monotonically increasing step times.
        self._alpha_tensor = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        n_classes = y_pred.shape[-1] or tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=n_classes, dtype=tf.float32)

        pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_factor = tf.pow(1.0 - pt, self.gamma)
        ce = -tf.math.log(pt)

        if self._alpha_tensor is not None:
            alpha_weight = tf.reduce_sum(y_true_one_hot * self._alpha_tensor, axis=-1)
        else:
            alpha_weight = 1.0

        return alpha_weight * focal_factor * ce

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


class MaskedMacroF1(tf.keras.metrics.Metric):
    """Macro F1 metric that respects temporal sample weights."""

    def __init__(self, n_classes: int, name: str = "masked_f1_macro", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.tp = self.add_weight(shape=(n_classes,), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(n_classes,), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(n_classes,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(tf.reshape(y_pred, [-1, self.n_classes]), axis=-1, output_type=tf.int32)

        if sample_weight is None:
            weights = tf.ones_like(y_true, dtype=tf.float32)
        else:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

        tp_updates = []
        fp_updates = []
        fn_updates = []
        for class_idx in range(self.n_classes):
            true_mask = tf.equal(y_true, class_idx)
            pred_mask = tf.equal(y_pred, class_idx)

            tp = tf.reduce_sum(weights * tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
            fp = tf.reduce_sum(weights * tf.cast(tf.logical_and(tf.logical_not(true_mask), pred_mask), tf.float32))
            fn = tf.reduce_sum(weights * tf.cast(tf.logical_and(true_mask, tf.logical_not(pred_mask)), tf.float32))
            tp_updates.append(tp)
            fp_updates.append(fp)
            fn_updates.append(fn)

        self.tp.assign_add(tf.stack(tp_updates))
        self.fp.assign_add(tf.stack(fp_updates))
        self.fn.assign_add(tf.stack(fn_updates))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes})
        return config


class MaskedClassPrecision(tf.keras.metrics.Metric):
    """Per-class precision that respects temporal sample weights."""

    def __init__(self, class_idx: int, n_classes: int, name: str | None = None, **kwargs):
        name = name or f"masked_precision_class{class_idx}"
        super().__init__(name=name, **kwargs)
        self.class_idx = class_idx
        self.n_classes = n_classes
        self.tp = self.add_weight(initializer="zeros", name="tp")
        self.fp = self.add_weight(initializer="zeros", name="fp")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(tf.reshape(y_pred, [-1, self.n_classes]), axis=-1, output_type=tf.int32)
        weights = (tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
                   if sample_weight is not None else tf.ones_like(y_true, dtype=tf.float32))
        true_mask = tf.equal(y_true, self.class_idx)
        pred_mask = tf.equal(y_pred, self.class_idx)
        self.tp.assign_add(tf.reduce_sum(weights * tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(weights * tf.cast(tf.logical_and(tf.logical_not(true_mask), pred_mask), tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fp + 1e-7)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"class_idx": self.class_idx, "n_classes": self.n_classes})
        return config


class MaskedMeanIoU(tf.keras.metrics.Metric):
    """Mean IoU across all classes, respecting temporal sample weights.

    IoU_k = TP_k / (TP_k + FP_k + FN_k)
    MeanIoU = mean over classes 1..n_classes-1 (excludes background class 0
              when exclude_background=True, matching ForkML convention).
    """

    def __init__(self, n_classes: int, exclude_background: bool = True,
                 name: str = "masked_mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.exclude_background = exclude_background
        self.tp = self.add_weight(shape=(n_classes,), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(n_classes,), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(n_classes,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(tf.reshape(y_pred, [-1, self.n_classes]), axis=-1, output_type=tf.int32)
        weights = (tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
                   if sample_weight is not None else tf.ones_like(y_true, dtype=tf.float32))

        tp_updates, fp_updates, fn_updates = [], [], []
        for class_idx in range(self.n_classes):
            true_mask = tf.equal(y_true, class_idx)
            pred_mask = tf.equal(y_pred, class_idx)
            tp_updates.append(tf.reduce_sum(weights * tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32)))
            fp_updates.append(tf.reduce_sum(weights * tf.cast(tf.logical_and(tf.logical_not(true_mask), pred_mask), tf.float32)))
            fn_updates.append(tf.reduce_sum(weights * tf.cast(tf.logical_and(true_mask, tf.logical_not(pred_mask)), tf.float32)))

        self.tp.assign_add(tf.stack(tp_updates))
        self.fp.assign_add(tf.stack(fp_updates))
        self.fn.assign_add(tf.stack(fn_updates))

    def result(self):
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        if self.exclude_background:
            iou = iou[1:]  # skip class 0 (background)
        return tf.reduce_mean(iou)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes, "exclude_background": self.exclude_background})
        return config


class MaskedClassRecall(tf.keras.metrics.Metric):
    """Per-class recall that respects temporal sample weights."""

    def __init__(self, class_idx: int, n_classes: int, name: str | None = None, **kwargs):
        name = name or f"masked_recall_class{class_idx}"
        super().__init__(name=name, **kwargs)
        self.class_idx = class_idx
        self.n_classes = n_classes
        self.tp = self.add_weight(initializer="zeros", name="tp")
        self.fn = self.add_weight(initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(tf.reshape(y_pred, [-1, self.n_classes]), axis=-1, output_type=tf.int32)
        weights = (tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
                   if sample_weight is not None else tf.ones_like(y_true, dtype=tf.float32))
        true_mask = tf.equal(y_true, self.class_idx)
        pred_mask = tf.equal(y_pred, self.class_idx)
        self.tp.assign_add(tf.reduce_sum(weights * tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(weights * tf.cast(tf.logical_and(true_mask, tf.logical_not(pred_mask)), tf.float32)))

    def result(self):
        return self.tp / (self.tp + self.fn + 1e-7)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"class_idx": self.class_idx, "n_classes": self.n_classes})
        return config
