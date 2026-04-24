"""
Custom callbacks and metrics for training.

This module provides custom Keras metrics and callbacks for tracking
model performance during training.
"""

import tensorflow as tf
import numpy as np


class F1Score(tf.keras.metrics.Metric):
    """
    F1-Score metric for binary classification.

    Computes F1 as the harmonic mean of precision and recall.
    """

    def __init__(self, name='f1', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall_metric = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(y_true, y_pred, sample_weight)
        self.recall_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision_metric.result()
        r = self.recall_metric.result()
        return 2 * ((p * r) / (p + r + 1e-7))

    def reset_state(self):
        self.precision_metric.reset_state()
        self.recall_metric.reset_state()


class MultiClassF1Score(tf.keras.metrics.Metric):
    """
    Macro-averaged F1 score for multi-class classification.

    Computes F1 for each class separately, then averages.
    """

    def __init__(self, n_classes=3, name='f1_macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.true_positives = self.add_weight(
            name='tp',
            shape=(n_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp',
            shape=(n_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn',
            shape=(n_classes,),
            initializer='zeros',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten and convert to proper types
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(tf.reshape(y_pred, [-1, self.n_classes]), axis=-1, output_type=tf.int32)

        # Calculate metrics for each class
        tp_list = []
        fp_list = []
        fn_list = []

        for i in range(self.n_classes):
            true_i = tf.equal(y_true, i)
            pred_i = tf.equal(y_pred, i)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(true_i, pred_i), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(true_i), pred_i), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(true_i, tf.logical_not(pred_i)), tf.float32))

            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

        # Stack and assign
        tp_update = tf.stack(tp_list)
        fp_update = tf.stack(fp_list)
        fn_update = tf.stack(fn_list)

        self.true_positives.assign_add(tp_update)
        self.false_positives.assign_add(fp_update)
        self.false_negatives.assign_add(fn_update)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))


def create_callbacks(config, model_path='best_model.keras', backup_dir=None):
    """
    Create standard callbacks for training.

    Parameters
    ----------
    config : dict
        Training configuration with callback settings
    model_path : str
        Path to save best model
    backup_dir : str or None
        Directory for BackupAndRestore mid-run checkpointing.
        When set, training automatically resumes from the last saved point
        if the process is killed and restarted.

    Returns
    -------
    list
        List of Keras callbacks
    """
    callbacks = []

    # BackupAndRestore must be first so other callbacks see the restored state.
    # This saves a full optimizer + model checkpoint after every epoch so that
    # re-running model.fit() on the same backup_dir automatically continues
    # from the last completed epoch rather than starting over.
    if backup_dir is not None:
        callbacks.append(
            tf.keras.callbacks.BackupAndRestore(
                backup_dir=backup_dir,
                save_freq="epoch",
                delete_checkpoint=False,
            )
        )

    # Early stopping
    if config.get('early_stopping', {}).get('enabled', True):
        es_config = config['early_stopping']
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                patience=es_config.get('patience', 25),
                restore_best_weights=True,
                verbose=1,
                mode=es_config.get('mode', 'min')
            )
        )

    # Reduce learning rate on plateau
    if config.get('reduce_lr', {}).get('enabled', True):
        lr_config = config['reduce_lr']
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=lr_config.get('monitor', 'val_loss'),
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 10),
                min_lr=lr_config.get('min_lr', 1e-7),
                verbose=1,
                mode=lr_config.get('mode', 'min')
            )
        )

    # Model checkpoint
    if config.get('checkpoint', {}).get('enabled', True):
        ckpt_config = config['checkpoint']
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor=ckpt_config.get('monitor', 'val_loss'),
                save_best_only=ckpt_config.get('save_best_only', True),
                verbose=1,
                mode=ckpt_config.get('mode', 'min')
            )
        )

    return callbacks


class TrainingProgressLogger(tf.keras.callbacks.Callback):
    """
    Custom callback to log training progress in a nice format.
    """

    def __init__(self, log_every=10):
        super().__init__()
        self.log_every = log_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} Summary:")
            print(f"{'='*60}")
            for key, value in logs.items():
                print(f"  {key:20s}: {value:.6f}")
            print(f"{'='*60}\n")
