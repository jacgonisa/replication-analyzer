"""
Custom loss functions for imbalanced classification.

Focal Loss is used to address class imbalance by down-weighting easy examples
and focusing training on hard negatives.
"""

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss for imbalanced classification.

    Focal Loss addresses class imbalance by reducing the weight given to
    well-classified examples, focusing learning on hard negatives.

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, name: str = 'focal_loss'):
        """
        Parameters
        ----------
        alpha : float
            Weighting factor in range [0, 1] to balance positive/negative examples
        gamma : float
            Focusing parameter for modulating loss (gamma >= 0)
        name : str
            Name of the loss
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Compute focal loss for positive class
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy

        # Handle negative class
        focal_loss += (1 - self.alpha) * tf.pow(y_pred, self.gamma) * \
                     -(1 - y_true) * tf.math.log(1 - y_pred)

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


class MultiClassFocalLoss(tf.keras.losses.Loss):
    """
    Multi-class Focal Loss for imbalanced multi-class classification.

    Extends Focal Loss to handle multiple classes with per-class weighting.
    """

    def __init__(self, alpha: list = None, gamma: float = 2.0, name: str = 'multi_class_focal_loss'):
        """
        Parameters
        ----------
        alpha : list, optional
            List of per-class weights (length = number of classes)
            If None, uses uniform weighting [1.0, 1.0, ..., 1.0]
        gamma : float
            Focusing parameter
        name : str
            Name of the loss
        """
        super().__init__(name=name)
        self.alpha = alpha if alpha is not None else [1.0, 1.0, 1.0]
        self.gamma = gamma
        self.n_classes = len(self.alpha)

    def call(self, y_true, y_pred):
        # Convert to int32 for one-hot encoding
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Convert to one-hot encoding
        y_true_one_hot = tf.one_hot(y_true, depth=self.n_classes)

        # Calculate focal loss
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = weight * cross_entropy

        # Apply per-class weighting
        alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
        alpha_weight = y_true_one_hot * alpha_tensor
        focal_loss = focal_loss * alpha_weight

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


def weighted_binary_crossentropy(pos_weight: float = 1.0):
    """
    Create a weighted binary crossentropy loss function.

    Simple alternative to Focal Loss that applies a constant weight
    to positive examples.

    Parameters
    ----------
    pos_weight : float
        Weight for positive class (> 1 gives more weight to positives)

    Returns
    -------
    function
        Loss function compatible with Keras
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Binary crossentropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Apply weight to positive class
        weight = y_true * pos_weight + (1 - y_true)
        weighted_bce = weight * bce

        return tf.reduce_mean(weighted_bce)

    return loss
