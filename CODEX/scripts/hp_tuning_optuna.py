#!/usr/bin/env python
"""Hyperparameter tuning for FORTE models using Optuna (TPE + HyperbandPruner).

Approximates ForkML's BOHB approach:
  - TPESampler  ≈ Bayesian Optimisation component of BOHB
  - HyperbandPruner ≈ Hyperband component of BOHB

Loads a preprocessed dataset once, then runs N trials where each trial:
  - Samples HPs (lr, batch_size, cnn_filters, lstm_units, dropout, gamma, alpha weights)
  - Trains for up to --max-epochs with early stopping (patience=10)
  - Reports val_masked_mean_iou to Optuna after each epoch (enables Hyperband pruning)
  - Objective: MAXIMISE val_mean_iou (excludes background — matches ForkML)

Study is persisted to a SQLite DB so it can be interrupted and resumed.

Usage (from /replication-analyzer/):
  CUDA_VISIBLE_DEVICES="" /path/to/env/python -u CODEX/scripts/hp_tuning_optuna.py \\
      --preprocessed CODEX/results/forte_v4.3/preprocessed_forte_v4.3.npz \\
      --output-dir   CODEX/results/forte_v4.3/hp_tuning \\
      --n-trials     40 \\
      --max-epochs   40

Resume an existing study (same --output-dir):
  Same command — Optuna automatically continues from the saved DB.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import yaml

# ── repo paths ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer.models.fork_model import build_4class_fork_ori_model
from replication_analyzer_codex.losses import (
    MaskedClassPrecision, MaskedClassRecall, MaskedMacroF1,
    MaskedMeanIoU, SparseCategoricalFocalLoss,
)

# ── HP search space ───────────────────────────────────────────────────────────
def sample_hps(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [64, 128]),
        "cnn_filters":   trial.suggest_categorical("cnn_filters", [32, 64, 128]),
        "lstm_units":    128,   # fixed — SelfAttention(256) requires BiLSTM output = 2*lstm_units = 256
        "dropout_rate":  trial.suggest_float("dropout_rate", 0.1, 0.5),
        "gamma":         trial.suggest_float("gamma", 1.0, 3.0),
        "alpha_bg":      1.0,  # background — fixed
        "alpha_lf":      trial.suggest_float("alpha_lf", 1.0, 5.0),
        "alpha_rf":      trial.suggest_float("alpha_rf", 1.0, 5.0),
        "alpha_ori":     trial.suggest_float("alpha_ori", 1.0, 5.0),
    }


# ── data loading ─────────────────────────────────────────────────────────────
def load_data(preprocessed_path: Path):
    """Load train/val tensors from the .npz (pre-padded tensors).
    Returns (train_x, train_y, train_w, val_x, val_y, val_w, max_length)."""
    print(f"Loading preprocessed data from {preprocessed_path}...")
    data = np.load(str(preprocessed_path))
    train_x = data["train_x"]
    train_y = data["train_y"]
    train_w = data["train_w"]
    val_x   = data["val_x"]
    val_y   = data["val_y"]
    val_w   = data["val_w"]
    max_length = int(data["max_length"])
    print(f"  Train: {train_x.shape}  Val: {val_x.shape}  max_length={max_length}")
    return train_x, train_y, train_w, val_x, val_y, val_w, max_length


# ── model builder ─────────────────────────────────────────────────────────────
def build_and_compile(hps: dict, max_length: int, n_channels: int, n_classes: int = 4):
    tf.keras.backend.clear_session()
    model = build_4class_fork_ori_model(
        max_length=max_length,
        n_channels=n_channels,
        n_classes=n_classes,
        cnn_filters=hps["cnn_filters"],
        lstm_units=hps["lstm_units"],
        dropout_rate=hps["dropout_rate"],
    )
    alpha = [hps["alpha_bg"], hps["alpha_lf"], hps["alpha_rf"], hps["alpha_ori"]]
    loss = SparseCategoricalFocalLoss(alpha=alpha, gamma=hps["gamma"])
    metrics = [
        MaskedMeanIoU(n_classes=n_classes, exclude_background=True),
        MaskedMacroF1(n_classes=n_classes),
        MaskedClassRecall(1, n_classes, name="masked_recall_left_fork"),
        MaskedClassRecall(2, n_classes, name="masked_recall_right_fork"),
        MaskedClassRecall(3, n_classes, name="masked_recall_origin"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hps["learning_rate"],
            clipnorm=1.0,
        ),
        loss=loss,
        metrics=metrics,
    )
    return model


# ── Optuna objective ──────────────────────────────────────────────────────────
class OptunaKerasPruningCallback(tf.keras.callbacks.Callback):
    """Reports val_masked_mean_iou to Optuna after each epoch and prunes if needed."""
    def __init__(self, trial: optuna.Trial, monitor: str = "val_masked_mean_iou"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned(
                f"Trial pruned at epoch {epoch} ({self.monitor}={value:.5f})"
            )


def make_objective(train_x, train_y, train_w, val_x, val_y, val_w,
                   max_length: int, max_epochs: int, patience: int,
                   flip_augment: bool):
    n_channels = train_x.shape[-1]

    def objective(trial: optuna.Trial) -> float:
        hps = sample_hps(trial)
        model = build_and_compile(hps, max_length, n_channels)

        # Flip augmentation: double fork-containing reads
        if flip_augment:
            fork_mask = np.any(train_y > 0, axis=1)  # reads with any event label
            flip_x = train_x[fork_mask, ::-1, :]
            flip_y_raw = train_y[fork_mask, ::-1]
            # swap left_fork(1) ↔ right_fork(2)
            flip_y = flip_y_raw.copy()
            flip_y[flip_y_raw == 1] = 2
            flip_y[flip_y_raw == 2] = 1
            flip_w = train_w[fork_mask, ::-1]
            aug_x = np.concatenate([train_x, flip_x])
            aug_y = np.concatenate([train_y, flip_y])
            aug_w = np.concatenate([train_w, flip_w])
        else:
            aug_x, aug_y, aug_w = train_x, train_y, train_w

        callbacks = [
            OptunaKerasPruningCallback(trial, monitor="val_masked_mean_iou"),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_masked_mean_iou", patience=patience, mode="max",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_masked_mean_iou", patience=max(3, patience // 3),
                factor=0.5, min_lr=1e-6, mode="max",
            ),
        ]

        history = model.fit(
            aug_x, aug_y, sample_weight=aug_w,
            validation_data=(val_x, val_y, val_w),
            epochs=max_epochs,
            batch_size=hps["batch_size"],
            callbacks=callbacks,
            verbose=0,
            shuffle=True,
        )

        iou_history = history.history["val_masked_mean_iou"]
        best_epoch  = int(np.argmax(iou_history))
        best_iou    = float(iou_history[best_epoch])
        best_loss   = float(history.history["val_loss"][best_epoch])
        best_f1     = float(history.history["val_masked_f1_macro"][best_epoch])
        best_recall_lf  = float(history.history["val_masked_recall_left_fork"][best_epoch])
        best_recall_rf  = float(history.history["val_masked_recall_right_fork"][best_epoch])
        best_recall_ori = float(history.history["val_masked_recall_origin"][best_epoch])

        # Store for later inspection
        trial.set_user_attr("best_epoch",       best_epoch + 1)
        trial.set_user_attr("val_mean_iou",     round(best_iou,    4))
        trial.set_user_attr("val_loss",         round(best_loss,   5))
        trial.set_user_attr("val_f1_macro",     round(best_f1,     4))
        trial.set_user_attr("recall_lf",        round(best_recall_lf,  4))
        trial.set_user_attr("recall_rf",        round(best_recall_rf,  4))
        trial.set_user_attr("recall_ori",       round(best_recall_ori, 4))
        trial.set_user_attr("n_epochs_trained", len(iou_history))

        return best_iou

    return objective


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", required=True,
                        help="Path to preprocessed .npz (or .metadata.yaml alongside chunks)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for study DB, logs, and results TSV")
    parser.add_argument("--n-trials",   type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=40,
                        help="Max epochs per trial (early stopping may terminate sooner)")
    parser.add_argument("--patience",   type=int, default=10,
                        help="EarlyStopping patience per trial")
    parser.add_argument("--flip-augment", action="store_true", default=True)
    parser.add_argument("--no-flip-augment", dest="flip_augment", action="store_false")
    parser.add_argument("--study-name", default="forte_hp_tuning")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FORTE HP TUNING — Optuna TPE + HyperbandPruner  (objective: max val_mean_iou)")
    print("=" * 70)
    print(f"  Preprocessed:  {args.preprocessed}")
    print(f"  Output dir:    {out_dir}")
    print(f"  n_trials:      {args.n_trials}")
    print(f"  max_epochs:    {args.max_epochs}")
    print(f"  patience:      {args.patience}")
    print(f"  flip_augment:  {args.flip_augment}")

    print("\nLoading data...")
    train_x, train_y, train_w, val_x, val_y, val_w, max_length = load_data(
        Path(args.preprocessed)
    )

    # Optuna study — in-memory (avoids SQLite "exceeding max length" on long histories)
    # Results saved to TSV on completion so nothing is lost
    db_path = out_dir / f"{args.study_name}.db"
    storage = None
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=args.max_epochs,
        reduction_factor=3,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",   # maximise val_mean_iou (ForkML-style)
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # resume if study already exists
    )

    n_existing = len(study.trials)
    n_remaining = args.n_trials - n_existing
    if n_remaining <= 0:
        print(f"\nStudy already has {n_existing} trials (≥ {args.n_trials} requested). Nothing to do.")
        print("To run more trials, increase --n-trials.")
    else:
        print(f"\n{'Resuming' if n_existing > 0 else 'Starting'} study "
              f"({n_existing} existing trials, running {n_remaining} more)...")

        objective = make_objective(
            train_x, train_y, train_w, val_x, val_y, val_w,
            max_length=max_length,
            max_epochs=args.max_epochs,
            patience=args.patience,
            flip_augment=args.flip_augment,
        )
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=False)

    # ── Results summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "val_mean_iou": round(t.value, 4)}
        row.update({k: v for k, v in t.params.items()})
        row.update({k: v for k, v in t.user_attrs.items()})
        rows.append(row)

    if rows:
        results_df = pd.DataFrame(rows).sort_values("val_mean_iou", ascending=False)
        tsv_path = out_dir / "hp_tuning_results.tsv"
        results_df.to_csv(tsv_path, sep="\t", index=False)
        print(f"\nTop 10 trials (sorted by val_mean_iou ↓):")
        print(results_df.head(10).to_string(index=False))
        print(f"\nFull results → {tsv_path}")

        best = study.best_trial
        print(f"\n{'='*70}")
        print(f"BEST TRIAL: #{best.number}  val_mean_iou={best.value:.4f}")
        print(f"  val_loss:     {best.user_attrs.get('val_loss', 'N/A')}")
        print(f"  recall_lf:    {best.user_attrs.get('recall_lf', 'N/A')}")
        print(f"  recall_rf:    {best.user_attrs.get('recall_rf', 'N/A')}")
        print(f"  recall_ori:   {best.user_attrs.get('recall_ori', 'N/A')}")
        print(f"  best_epoch:   {best.user_attrs.get('best_epoch', 'N/A')}")
        print(f"  HPs:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")

        # Save best HPs as YAML snippet for easy copy-paste into config
        best_yaml = out_dir / "best_hps.yaml"
        with open(best_yaml, "w") as f:
            yaml.dump({
                "training": {
                    "learning_rate": best.params["learning_rate"],
                    "batch_size": best.params["batch_size"],
                    "loss": {
                        "alpha": [1.0, best.params["alpha_lf"],
                                  best.params["alpha_rf"], best.params["alpha_ori"]],
                        "gamma": best.params["gamma"],
                    },
                },
                "model": {
                    "cnn_filters": best.params["cnn_filters"],
                    "lstm_units": 128,  # fixed — not sampled
                    "dropout_rate": best.params["dropout_rate"],
                },
            }, f, default_flow_style=False)
        print(f"Best HPs saved as YAML → {best_yaml}")


if __name__ == "__main__":
    main()
