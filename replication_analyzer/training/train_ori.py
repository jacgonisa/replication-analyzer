"""
Training pipeline for ORI detection model.
"""

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

from ..data.loaders import load_all_xy_data, load_curated_origins
from ..data.preprocessing import prepare_ori_data_hybrid, pad_sequences
from ..models.ori_model import build_ori_expert_model, build_ori_simple_model
from ..models.losses import FocalLoss
from .callbacks import F1Score, create_callbacks, TrainingProgressLogger


def train_ori_model(config):
    """
    Complete training pipeline for ORI detection.

    Parameters
    ----------
    config : dict
        Configuration dictionary with all training parameters

    Returns
    -------
    tuple
        (model, history, max_length, info_df)
    """
    print("\n" + "="*70)
    print("ORI DETECTION MODEL TRAINING")
    print("="*70)

    # Force CPU mode (from notebook)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['TF_NUM_INTRAOP_PARALLELISM_THREADS'] = str(os.cpu_count())
    os.environ['TF_NUM_INTEROP_PARALLELISM_THREADS'] = str(os.cpu_count())

    print(f"✅ CPU cores: {os.cpu_count()}")
    print(f"✅ GPU disabled: {len(tf.config.list_physical_devices('GPU')) == 0}")

    # ========== LOAD DATA ==========
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    xy_data = load_all_xy_data(
        base_dir=config['data']['base_dir'],
        run_dirs=config['data'].get('run_dirs')
    )

    ori_annotations = load_curated_origins(config['data']['ori_bed'])

    # ========== PREPARE DATA ==========
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)

    X_sequences, y_sequences, info = prepare_ori_data_hybrid(
        xy_data,
        ori_annotations,
        oversample_ratio=config['preprocessing'].get('oversample_ratio', 0.5),
        use_enhanced_encoding=config['preprocessing'].get('use_enhanced_encoding', True),
        random_seed=config['preprocessing'].get('random_seed', 42)
    )

    # ========== PAD SEQUENCES ==========
    percentile = config['preprocessing'].get('percentile', 100)
    max_length_config = config['model'].get('max_length')

    X_padded, y_padded, max_length = pad_sequences(
        X_sequences,
        y_sequences,
        percentile=percentile,
        max_length=max_length_config
    )

    # Convert to binary labels (threshold at 0.3)
    y_binary = (y_padded > 0.3).astype(np.float32)
    y_binary_reshaped = y_binary.reshape(y_binary.shape[0], y_binary.shape[1], 1)

    # ========== TRAIN/VAL SPLIT ==========
    test_size = config['training'].get('test_size', 0.2)

    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y_binary_reshaped,
        test_size=test_size,
        random_state=config['preprocessing'].get('random_seed', 42),
        stratify=info['has_ori']
    )

    print(f"\n📊 Data split:")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Val: {X_val.shape[0]:,} samples")
    print(f"   Shape: {X_train.shape}")

    # ========== BUILD MODEL ==========
    print(f"\n🏗️  Building model...")

    model_type = config['model'].get('type', 'ori_expert')
    n_channels = config['model'].get('n_channels', 9)

    if model_type == 'ori_expert':
        model = build_ori_expert_model(
            max_length=max_length,
            n_channels=n_channels,
            cnn_filters=config['model'].get('cnn_filters', 64),
            lstm_units=config['model'].get('lstm_units', 128),
            dropout_rate=config['model'].get('dropout_rate', 0.3)
        )
    else:
        model = build_ori_simple_model(
            max_length=max_length,
            n_channels=n_channels
        )

    print(f"\n📐 Model summary:")
    print(f"   Total params: {model.count_params():,}")

    # ========== COMPILE MODEL ==========
    print(f"\n⚙️  Compiling...")

    # Loss function
    loss_config = config['training'].get('loss', {})
    loss_type = loss_config.get('type', 'focal')

    if loss_type == 'focal':
        loss = FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0)
        )
    else:
        loss = 'binary_crossentropy'

    # Metrics
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        F1Score(name='f1'),
        tf.keras.metrics.AUC(name='auc')
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['training'].get('learning_rate', 0.0005)
        ),
        loss=loss,
        metrics=metrics
    )

    # ========== SETUP CALLBACKS ==========
    # Create output directory
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / config['output']['model_filename']

    callbacks = create_callbacks(config['training'], model_path=str(model_path))
    callbacks.append(TrainingProgressLogger(log_every=10))

    # ========== TRAIN ==========
    print(f"\n🚀 Starting training...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training'].get('epochs', 150),
        batch_size=config['training'].get('batch_size', 32),
        callbacks=callbacks,
        verbose=1
    )

    elapsed_time = time.time() - start_time

    print(f"\n✅ Training complete in {elapsed_time/60:.1f} minutes!")
    print(f"   Model saved to: {model_path}")

    # ========== SAVE TRAINING INFO ==========
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save training history
    import pandas as pd
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(results_dir / 'training_history.csv', index=False)

    # Save config
    import yaml
    with open(results_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save dataset info
    info.to_csv(results_dir / 'dataset_info.csv', index=False)

    print(f"\n📁 Results saved to: {results_dir}")

    return model, history, max_length, info


def load_trained_ori_model(model_path, custom_objects=None):
    """
    Load a trained ORI model.

    Parameters
    ----------
    model_path : str
        Path to saved model
    custom_objects : dict, optional
        Custom objects for loading (losses, metrics, etc.)

    Returns
    -------
    tf.keras.Model
        Loaded model
    """
    if custom_objects is None:
        custom_objects = {
            'FocalLoss': FocalLoss,
            'F1Score': F1Score
        }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"✅ Model loaded from: {model_path}")

    return model
