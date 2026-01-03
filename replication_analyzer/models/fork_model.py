"""
Fork detection model (3-class classification).

Detects directional replication forks:
- Class 0: Background (no fork)
- Class 1: Left fork (leftward replication)
- Class 2: Right fork (rightward replication)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from .base import (
    SelfAttention,
    build_multi_scale_cnn_encoder,
    build_encoder_block,
    build_decoder_block
)


def build_fork_detection_model(max_length: int,
                                n_channels: int = 9,
                                n_classes: int = 3,
                                cnn_filters: int = 64,
                                lstm_units: int = 128,
                                dropout_rate: float = 0.3) -> models.Model:
    """
    Build the 3-class fork detection model.

    Architecture is identical to ORI Expert Model but with softmax output
    for 3-class classification.

    Parameters
    ----------
    max_length : int
        Maximum sequence length
    n_channels : int
        Number of input channels (9 for enhanced encoding)
    n_classes : int
        Number of classes (3: background, left, right)
    cnn_filters : int
        Number of CNN filters
    lstm_units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate

    Returns
    -------
    tf.keras.Model
        Fork detection model
    """
    inputs = layers.Input(shape=(max_length, n_channels), name='input')

    # ========== MULTI-SCALE FEATURE EXTRACTION ==========
    x = build_multi_scale_cnn_encoder(
        inputs,
        filters=cnn_filters,
        kernel_size=7,
        dropout_rate=dropout_rate
    )

    # ========== ENCODER ==========
    x = build_encoder_block(
        x,
        filters=128,
        kernel_size=5,
        pool_size=2,
        dropout_rate=dropout_rate,
        name_prefix='enc1'
    )

    x = build_encoder_block(
        x,
        filters=256,
        kernel_size=3,
        pool_size=2,
        dropout_rate=dropout_rate,
        name_prefix='enc2'
    )

    # ========== BIDIRECTIONAL LSTM ==========
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name='bilstm'
    )(x)
    x = layers.Dropout(dropout_rate, name='lstm_dropout')(x)

    # ========== SELF-ATTENTION ==========
    attention_output = SelfAttention(256, name='self_attention')(x)
    x = layers.Add(name='attention_residual')([x, attention_output])
    x = layers.LayerNormalization(name='attention_norm')(x)

    # ========== DECODER ==========
    x = build_decoder_block(
        x,
        filters=256,
        kernel_size=3,
        upsample_size=2,
        name_prefix='dec1'
    )

    x = build_decoder_block(
        x,
        filters=128,
        kernel_size=3,
        upsample_size=2,
        name_prefix='dec2'
    )

    # Crop to exact length
    x = layers.Lambda(lambda t: t[:, :max_length, :], name='crop_to_length')(x)

    # ========== OUTPUT (3-CLASS SOFTMAX) ==========
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='output_conv')(x)
    outputs = layers.Conv1D(n_classes, 1, activation='softmax', padding='same', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='Fork_Detector_3Class')

    return model
