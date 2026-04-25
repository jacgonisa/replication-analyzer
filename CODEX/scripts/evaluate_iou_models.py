#!/usr/bin/env python
"""IoU-based evaluation of FORTE models on the validation split.

Loads each model, runs inference on val-split reads that have ground truth
annotations, converts per-window probabilities to event regions, and evaluates
with IoU matching — identical protocol to the mathematical benchmark.

Usage:
  python evaluate_iou_models.py \
      --models FORTE_v1:CODEX/models/forte_v1.keras:CODEX/configs/forte_v1.yaml \
               FORTE_v1_conservative:CODEX/models/forte_v1_conservative.keras:CODEX/configs/forte_v1_conservative.yaml \
      --split-manifest CODEX/results/forte_v1_conservative/preprocessed_forte_v1_conservative.split_manifest.tsv \
      --output-dir CODEX/results/iou_evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

ROOT      = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))


# ── IoU helpers (same as mathematical benchmark) ──────────────────────────────

def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def evaluate_iou(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                 iou_threshold: float = 0.3) -> dict:
    if len(pred_df) == 0 and len(gt_df) == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0,
                    n_pred=0, n_gt=0)
    if len(pred_df) == 0:
        return dict(tp=0, fp=0, fn=len(gt_df), precision=0.0, recall=0.0, f1=0.0,
                    n_pred=0, n_gt=len(gt_df))
    if len(gt_df) == 0:
        return dict(tp=0, fp=len(pred_df), fn=0, precision=0.0, recall=0.0, f1=0.0,
                    n_pred=len(pred_df), n_gt=0)

    tp, fp = 0, 0
    gt_by_read = {rid: grp for rid, grp in gt_df.groupby("read_id")}
    gt_matched = set()

    for _, p in pred_df.iterrows():
        rid = p["read_id"]
        if rid not in gt_by_read:
            fp += 1
            continue
        gt_read = gt_by_read[rid]
        best_iou, best_idx = 0.0, None
        for idx, g in gt_read.iterrows():
            iou = compute_iou(int(p["start"]), int(p["end"]),
                              int(g["start"]), int(g["end"]))
            if iou > best_iou and idx not in gt_matched:
                best_iou, best_idx = iou, idx
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            gt_matched.add(best_idx)
        else:
            fp += 1

    fn = len(gt_df) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_pred=len(pred_df), n_gt=len(gt_df))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bed4(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0,1,2,3],
                       names=["chr","start","end","read_id"])


def load_xy_for_reads(base_dir, run_dirs, read_ids):
    wanted = set(read_ids)
    rows = []
    for run_dir in run_dirs:
        run_path = Path(base_dir) / run_dir
        if not run_path.exists():
            continue
        for f in run_path.glob("plot_data_*.txt"):
            rid = f.stem.replace("plot_data_", "")
            if rid in wanted:
                try:
                    df = pd.read_csv(f, sep="\t", header=None,
                                     names=["chr","start","end","signal"])
                    df["read_id"] = rid
                    rows.append(df)
                except Exception:
                    pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["chr","start","end","signal","read_id"])


# ── Model inference ───────────────────────────────────────────────────────────

def load_model(model_path):
    import tensorflow as tf
    from replication_analyzer_codex.losses import (
        SparseCategoricalFocalLoss, MaskedMacroF1,
        MaskedClassPrecision, MaskedClassRecall,
    )
    from replication_analyzer.models.base import SelfAttention
    from replication_analyzer.models.losses import MultiClassFocalLoss
    from replication_analyzer.training.callbacks import MultiClassF1Score

    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(model_path, custom_objects={
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
        "MaskedMacroF1": MaskedMacroF1,
        "MaskedClassPrecision": MaskedClassPrecision,
        "MaskedClassRecall": MaskedClassRecall,
        "SelfAttention": SelfAttention,
        "MultiClassFocalLoss": MultiClassFocalLoss,
        "MultiClassF1Score": MultiClassF1Score,
    })
    return model


def predict_and_call_events(model, xy_data, read_ids, preprocessing_config,
                            prob_threshold=0.4, max_gap=5000, min_windows=1):
    """Run inference + windows_to_events for all three event classes."""
    from replication_analyzer_codex.evaluation import predict_reads, windows_to_events

    max_length = model.input_shape[1]
    preds = predict_reads(model, xy_data, read_ids, max_length, preprocessing_config)

    events = {}
    for class_id, class_name in [(1,"left_fork"),(2,"right_fork"),(3,"origin")]:
        ev = windows_to_events(preds, class_id, prob_threshold,
                               min_windows=min_windows, max_gap=max_gap)
        events[class_name] = ev[["read_id","chr","start","end"]].rename(
            columns={"read_id":"read_id"})
    return events, preds


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(summary_df: pd.DataFrame, output_path: Path,
                    iou_threshold: float = 0.3):
    models  = summary_df["model"].unique()
    classes = ["left_fork","right_fork","origin"]
    sub     = summary_df[summary_df["iou_threshold"] == iou_threshold]

    colors = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#8c564b"]
    x      = np.arange(len(classes))
    n      = len(models)
    width  = 0.15
    offsets = np.linspace(-(n-1)/2,(n-1)/2,n)*width

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for metric, ax, ylabel in [("f1",       axes[0], "F1"),
                                ("precision",axes[1], "Precision"),
                                ("recall",   axes[2], "Recall")]:
        for i, mname in enumerate(models):
            row = sub[sub["model"]==mname]
            vals = [row[row["class"]==cl][metric].values[0]
                    if len(row[row["class"]==cl])>0 else 0.0
                    for cl in classes]
            bars = ax.bar(x+offsets[i], vals, width*0.88, label=mname,
                          color=colors[i%len(colors)], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                        f"{v:.2f}", ha="center", fontsize=7, va="bottom")

        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}  (IoU ≥ {iou_threshold})", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5, ncol=1)

    fig.suptitle("FORTE models — IoU-based evaluation on held-out annotated val reads",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_iou_sweep(summary_df: pd.DataFrame, output_path: Path):
    """F1 vs IoU threshold for each model × class."""
    models  = summary_df["model"].unique()
    classes = ["left_fork","right_fork","origin"]
    colors  = ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#8c564b"]
    ls_map  = {"left_fork":"solid","right_fork":"dashed","origin":"dotted"}

    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4),
                              sharey=True, gridspec_kw={"wspace":0.1})
    if len(models) == 1:
        axes = [axes]

    for ax, mname in zip(axes, models):
        sub = summary_df[summary_df["model"]==mname]
        for j, cl in enumerate(classes):
            row = sub[sub["class"]==cl].sort_values("iou_threshold")
            ax.plot(row["iou_threshold"], row["f1"],
                    color=colors[j], linestyle=ls_map[cl],
                    linewidth=2, marker="o", markersize=5, label=cl)
        ax.set_title(mname, fontsize=10, fontweight="bold")
        ax.set_xlabel("IoU threshold", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("F1", fontsize=11)
    fig.suptitle("F1 vs IoU threshold — val annotated reads", fontsize=11)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="name:model_path:config_path  (repeat for multiple models)")
    parser.add_argument("--split-manifest", required=True,
                        help="split_manifest.tsv to identify val reads")
    parser.add_argument("--gt-left",
                        default="data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed")
    parser.add_argument("--gt-right",
                        default="data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed")
    parser.add_argument("--gt-ori",
                        default="data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed")
    parser.add_argument("--output-dir", default="CODEX/results/iou_evaluation")
    parser.add_argument("--prob-threshold", type=float, default=0.4)
    parser.add_argument("--max-gap", type=int, default=5000)
    parser.add_argument("--iou-thresholds", nargs="+", type=float,
                        default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--iou-primary", type=float, default=0.3,
                        help="IoU threshold for the main bar chart")
    args = parser.parse_args()

    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")

    run_dirs = [
        "NM30_Col0/NM30_plot_data_1strun_xy",
        "NM30_Col0/NM30_plot_data_2ndrun_xy",
        "NM31_orc1b2/NM31_plot_data_1strun_xy",
        "NM31_orc1b2/NM31_plot_data_2ndrun_xy",
    ]

    # ── Load GT and val split ─────────────────────────────────────────────────
    gt_left  = load_bed4(str(base / args.gt_left))
    gt_right = load_bed4(str(base / args.gt_right))
    gt_ori   = load_bed4(str(base / args.gt_ori))

    manifest = pd.read_csv(args.split_manifest, sep="\t")
    val_ids  = set(manifest[manifest["split"] == "val"]["read_id"])

    annotated = set(gt_left.read_id) | set(gt_right.read_id) | set(gt_ori.read_id)
    val_annotated = sorted(val_ids & annotated)
    print(f"Val annotated reads: {len(val_annotated):,}")

    gt_left_val  = gt_left[gt_left.read_id.isin(val_annotated)]
    gt_right_val = gt_right[gt_right.read_id.isin(val_annotated)]
    gt_ori_val   = gt_ori[gt_ori.read_id.isin(val_annotated)]
    print(f"GT: {len(gt_left_val)} left | {len(gt_right_val)} right | {len(gt_ori_val)} origins")

    # ── Load XY once ─────────────────────────────────────────────────────────
    base_dir = "/mnt/ssd-4tb/crisanto_project/data_2025Oct/data_reads_minLen30000_nascent40"
    print(f"\nLoading XY for {len(val_annotated):,} reads…")
    xy_data = load_xy_for_reads(base_dir, run_dirs, val_annotated)
    loaded_ids = list(xy_data["read_id"].unique())
    print(f"  Loaded: {len(loaded_ids):,} reads  ({len(xy_data):,} windows)")

    # Restrict GT to loaded reads
    gt_left_val  = gt_left_val[gt_left_val.read_id.isin(loaded_ids)]
    gt_right_val = gt_right_val[gt_right_val.read_id.isin(loaded_ids)]
    gt_ori_val   = gt_ori_val[gt_ori_val.read_id.isin(loaded_ids)]
    gt_map = {"left_fork": gt_left_val, "right_fork": gt_right_val, "origin": gt_ori_val}

    # ── Evaluate each model ───────────────────────────────────────────────────
    rows = []
    for model_spec in args.models:
        parts = model_spec.split(":")
        mname, model_path, config_path = parts[0], parts[1], parts[2]
        print(f"\n{'='*60}\nModel: {mname}\n{'='*60}")

        with open(base / config_path) as f:
            config = yaml.safe_load(f)
        preprocessing_config = config["preprocessing"]

        import tensorflow as tf
        tf.keras.backend.clear_session()
        model = load_model(str(base / model_path))
        print(f"  Input shape: {model.input_shape}  max_length={model.input_shape[1]}")

        events, _ = predict_and_call_events(
            model, xy_data, loaded_ids, preprocessing_config,
            prob_threshold=args.prob_threshold,
            max_gap=args.max_gap,
        )

        for cl in ["left_fork","right_fork","origin"]:
            pred_df = events[cl]
            # filter to val annotated reads only
            pred_df = pred_df[pred_df.read_id.isin(loaded_ids)]
            print(f"  {cl}: {len(pred_df):,} predicted events")
            for iou_thr in args.iou_thresholds:
                m = evaluate_iou(pred_df, gt_map[cl], iou_thr)
                rows.append({"model": mname, "class": cl, "iou_threshold": iou_thr, **m})
                if iou_thr == args.iou_primary:
                    print(f"    IoU≥{iou_thr}: F1={m['f1']:.3f}  "
                          f"prec={m['precision']:.3f}  rec={m['recall']:.3f}  "
                          f"(pred={m['n_pred']:,} gt={m['n_gt']:,})")

        del model

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "iou_summary.tsv", sep="\t", index=False)
    print(f"\nSaved: {out / 'iou_summary.tsv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_comparison(summary_df, out / "iou_comparison.png",
                    iou_threshold=args.iou_primary)
    plot_iou_sweep(summary_df, out / "iou_sweep.png")
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
