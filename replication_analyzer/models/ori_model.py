"""
ORI (Origin of Replication) detection model.

This module contains the EXPERT MODEL architecture combining:
- Multi-scale CNN with dilated convolutions
- Bidirectional LSTM for temporal context
- Self-attention for long-range dependencies
- Residual connections
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from .base import (
    SelfAttention,
    build_multi_scale_cnn_encoder,
    build_encoder_block,
    build_decoder_block
)


def build_ori_expert_model(max_length: int,
                           n_channels: int = 9,
                           cnn_filters: int = 64,
                           lstm_units: int = 128,
                           dropout_rate: float = 0.3) -> models.Model:
    """
    Build the EXPERT ORI detection model.

    This is the enhanced architecture from the notebook combining multiple
    techniques for high-accuracy ORI detection.

    Architecture:
    1. Multi-scale CNN (dilated convolutions at rates 1, 2, 4)
    2. Encoder blocks with max pooling
    3. Bidirectional LSTM
    4. Self-attention with residual connection
    5. Decoder blocks with upsampling
    6. Segment-level binary prediction

    Parameters
    ----------
    max_length : int
        Maximum sequence length (after padding)
    n_channels : int
        Number of input channels (9 for enhanced encoding)
    cnn_filters : int
        Number of CNN filters in first layer
    lstm_units : int
        Number of LSTM units
    dropout_rate : float
        Dropout rate for regularization

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training
    """
    inputs = layers.Input(shape=(max_length, n_channels), name='input')

    # ========== MULTI-SCALE FEATURE EXTRACTION ==========
    x = build_multi_scale_cnn_encoder(
        inputs,
        filters=cnn_filters,
        kernel_size=7,
        dropout_rate=dropout_rate
    )
    # Output: 3 * cnn_filters = 192 channels

    # ========== ENCODER (with residual connections) ==========
    # Block 1: 192 -> 128 channels, length /2
    x = build_encoder_block(
        x,
        filters=128,
        kernel_size=5,
        pool_size=2,
        dropout_rate=dropout_rate,
        name_prefix='enc1'
    )

    # Block 2: 128 -> 256 channels, length /4
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
    # Output: 2 * lstm_units = 256 channels
    x = layers.Dropout(dropout_rate, name='lstm_dropout')(x)

    # ========== SELF-ATTENTION ==========
    attention_output = SelfAttention(256, name='self_attention')(x)
    x = layers.Add(name='attention_residual')([x, attention_output])
    x = layers.LayerNormalization(name='attention_norm')(x)

    # ========== DECODER ==========
    # Block 1: Upsample x2
    x = build_decoder_block(
        x,
        filters=256,
        kernel_size=3,
        upsample_size=2,
        name_prefix='dec1'
    )

    # Block 2: Upsample x2 (total x4)
    x = build_decoder_block(
        x,
        filters=128,
        kernel_size=3,
        upsample_size=2,
        name_prefix='dec2'
    )

    # Crop to exact input length
    x = layers.Lambda(lambda t: t[:, :max_length, :], name='crop_to_length')(x)

    # ========== OUTPUT ==========
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='output_conv')(x)
    outputs = layers.Conv1D(1, 1, activation='sigmoid', padding='same', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ORI_Expert_Model')

    return model


def build_ori_simple_model(max_length: int,
                           n_channels: int = 6) -> models.Model:
    """
    Build a simpler ORI detection model (baseline).

    This is a more basic architecture for comparison or faster training.

    Parameters
    ----------
    max_length : int
        Maximum sequence length
    n_channels : int
        Number of input channels (6 for basic encoding)

    Returns
    -------
    tf.keras.Model
        Simple model
    """
    inputs = layers.Input(shape=(max_length, n_channels))

    # Encoder
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Decoder
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)

    # Crop to exact length
    x = layers.Lambda(lambda t: t[:, :max_length, :])(x)

    # Output
    outputs = layers.Conv1D(1, 1, activation='sigmoid', padding='same')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ORI_Simple_Model')

    return model
