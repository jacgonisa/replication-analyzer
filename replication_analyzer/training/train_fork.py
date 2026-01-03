"""
Training pipeline for Fork detection model (3-class).
"""

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

from ..data.loaders import load_all_xy_data, load_fork_data
from ..data.preprocessing import prepare_fork_data_hybrid, pad_sequences
from ..models.fork_model import build_fork_detection_model
from ..models.losses import MultiClassFocalLoss
from .callbacks import MultiClassF1Score, create_callbacks, TrainingProgressLogger


def train_fork_model(config):
    """
    Complete training pipeline for Fork detection (3-class).

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    tuple
        (model, history, max_length, info_df)
    """
    print("\n" + "="*70)
    print("FORK DETECTION MODEL TRAINING (3-CLASS)")
    print("="*70)

    # Force CPU mode
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

    left_forks, right_forks = load_fork_data(
        config['data']['left_forks_bed'],
        config['data']['right_forks_bed']
    )

    # ========== PREPARE DATA ==========
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)

    X_sequences, y_sequences, info = prepare_fork_data_hybrid(
        xy_data,
        left_forks,
        right_forks,
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

    # ========== TRAIN/VAL SPLIT ==========
    test_size = config['training'].get('test_size', 0.2)

    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y_padded,
        test_size=test_size,
        random_state=config['preprocessing'].get('random_seed', 42),
        stratify=info['has_fork']
    )

    print(f"\n📊 Data split:")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Val: {X_val.shape[0]:,} samples")
    print(f"   Shape: {X_train.shape}")

    # Calculate class weights
    total_segs = y_train.size
    n_background = np.sum(y_train == 0)
    n_left = np.sum(y_train == 1)
    n_right = np.sum(y_train == 2)

    weight_background = total_segs / (3 * n_background) if n_background > 0 else 1.0
    weight_left = total_segs / (3 * n_left) if n_left > 0 else 1.0
    weight_right = total_segs / (3 * n_right) if n_right > 0 else 1.0

    print(f"\n⚖️  Class distribution:")
    print(f"   Background: {n_background:,} ({n_background/total_segs*100:.2f}%)")
    print(f"   Left fork:  {n_left:,} ({n_left/total_segs*100:.2f}%)")
    print(f"   Right fork: {n_right:,} ({n_right/total_segs*100:.2f}%)")

    # ========== BUILD MODEL ==========
    print(f"\n🏗️  Building 3-class fork model...")

    model = build_fork_detection_model(
        max_length=max_length,
        n_channels=config['model'].get('n_channels', 9),
        n_classes=config['model'].get('n_classes', 3),
        cnn_filters=config['model'].get('cnn_filters', 64),
        lstm_units=config['model'].get('lstm_units', 128),
        dropout_rate=config['model'].get('dropout_rate', 0.3)
    )

    print(f"\n📐 Model summary:")
    print(f"   Total params: {model.count_params():,}")

    # ========== COMPILE MODEL ==========
    print(f"\n⚙️  Compiling...")

    # Loss function
    loss_config = config['training'].get('loss', {})
    alpha = loss_config.get('alpha', [1.0, 2.0, 2.0])

    # If alpha provided but needs normalization
    if isinstance(alpha, list) and len(alpha) == 3:
        class_weights = alpha
    else:
        class_weights = [weight_background, weight_left, weight_right]

    print(f"\n⚖️  Focal loss weights:")
    print(f"   Background: {class_weights[0]:.4f}")
    print(f"   Left fork:  {class_weights[1]:.4f}")
    print(f"   Right fork: {class_weights[2]:.4f}")

    loss = MultiClassFocalLoss(
        alpha=class_weights,
        gamma=loss_config.get('gamma', 2.0)
    )

    # Metrics
    metrics = [
        'accuracy',
        MultiClassF1Score(n_classes=3, name='f1_macro')
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config['training'].get('learning_rate', 0.0005)
        ),
        loss=loss,
        metrics=metrics
    )

    # ========== SETUP CALLBACKS ==========
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


def load_trained_fork_model(model_path, custom_objects=None):
    """
    Load a trained Fork model.

    Parameters
    ----------
    model_path : str
        Path to saved model
    custom_objects : dict, optional
        Custom objects for loading

    Returns
    -------
    tf.keras.Model
        Loaded model
    """
    if custom_objects is None:
        custom_objects = {
            'MultiClassFocalLoss': MultiClassFocalLoss,
            'MultiClassF1Score': MultiClassF1Score
        }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"✅ Model loaded from: {model_path}")

    return model
