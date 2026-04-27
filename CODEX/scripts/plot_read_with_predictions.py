#!/usr/bin/env python
"""Plot a single read with real annotations + predictions from one or more models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

import tensorflow as tf
from replication_analyzer_codex.losses import (
    SparseCategoricalFocalLoss, MaskedMacroF1,
    MaskedClassPrecision, MaskedClassRecall, MaskedMeanIoU,
)
from replication_analyzer_codex.evaluation import predict_reads, windows_to_events
from replication_analyzer.models.base import SelfAttention
from replication_analyzer.models.losses import MultiClassFocalLoss
from replication_analyzer.training.callbacks import MultiClassF1Score


CUSTOM_OBJECTS = {
    "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
    "MaskedMacroF1": MaskedMacroF1,
    "MaskedMeanIoU": MaskedMeanIoU,
    "MaskedClassPrecision": MaskedClassPrecision,
    "MaskedClassRecall": MaskedClassRecall,
    "SelfAttention": SelfAttention,
    "MultiClassFocalLoss": MultiClassFocalLoss,
    "MultiClassF1Score": MultiClassF1Score,
}

# colours
COL_REAL_ORI   = "#2ca02c"
COL_REAL_LEFT  = "#1f77b4"
COL_REAL_RIGHT = "#d62728"

MODEL_COLOURS = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]


def load_xy(base_dir, run_dirs, read_id):
    rows = []
    for run_dir in run_dirs:
        f = Path(base_dir) / run_dir / f"plot_data_{read_id}.txt"
        if f.exists():
            df = pd.read_csv(f, sep="\t", header=None, names=["chr", "start", "end", "signal"])
            df["read_id"] = read_id
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"XY data not found for {read_id}")
    return pd.concat(rows, ignore_index=True).sort_values("start").reset_index(drop=True)


def load_bed(path):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                     names=["chr", "start", "end", "read_id"])
    return df


def add_spans(ax, df, read_id, color, label, alpha=0.35):
    subset = df[df["read_id"] == read_id]
    used = False
    for row in subset.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), color=color, alpha=alpha,
                   label=label if not used else None)
        used = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base config yaml (for data paths)")
    parser.add_argument("--read-id", required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="Model keras file; can be repeated for multiple models")
    parser.add_argument("--model-name", action="append",
                        help="Display name for each model (same order as --model)")
    parser.add_argument("--model-config", action="append",
                        help="Optional per-model config yaml (overrides base --config preprocessing). "
                             "Supply one per --model, or fewer to fall back to base config.")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Probability threshold for event calling (default 0.4)")
    parser.add_argument("--max-gap", type=int, default=5000,
                        help="Max gap (bp) between windows to merge into one event (default 5000)")
    parser.add_argument("--max-gap-ori", type=int, default=None,
                        help="Override max-gap for origins only (default: same as --max-gap)")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--human-left-forks-bed",  default=None,
                        help="Human-only left fork BED — shown as solid spans; "
                             "remaining GT spans shown hatched (pseudo-labels)")
    parser.add_argument("--human-right-forks-bed", default=None,
                        help="Human-only right fork BED")
    parser.add_argument("--human-ori-bed",          default=None,
                        help="Human-only ORI BED")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_names = args.model_name or [Path(m).stem for m in args.model]
    assert len(model_names) == len(args.model), "--model-name count must match --model count"

    # Load signal
    xy = load_xy(config["data"]["base_dir"], config["data"]["run_dirs"], args.read_id)
    print(f"Loaded {len(xy)} windows for {args.read_id}")

    # Load annotations — combined (pseudo+human) for GT bar
    left_real  = load_bed(config["data"].get("left_forks_bed"))
    right_real = load_bed(config["data"].get("right_forks_bed"))
    ori_real   = load_bed(config["data"].get("ori_annotations_bed"))

    # Optional human-only annotation files for human/pseudo distinction
    left_human  = load_bed(args.human_left_forks_bed)
    right_human = load_bed(args.human_right_forks_bed)
    ori_human   = load_bed(args.human_ori_bed)
    show_human_pseudo = any([args.human_left_forks_bed, args.human_right_forks_bed, args.human_ori_bed])

    # Build per-model preprocessing configs
    model_configs = []
    for i in range(len(args.model)):
        if args.model_config and i < len(args.model_config):
            with open(args.model_config[i]) as f:
                mc = yaml.safe_load(f)
            model_configs.append(mc["preprocessing"])
        else:
            model_configs.append(config["preprocessing"])

    all_events = []   # list of (model_name, class_name, events_df)
    all_preds  = {}   # model_name -> predictions DataFrame

    for model_path, model_name, preprocessing_config in zip(args.model, model_names, model_configs):
        print(f"Loading model: {model_name} ({model_path})")
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        max_length = model.input_shape[1]
        print(f"  max_length={max_length}")

        preds = predict_reads(model, xy, [args.read_id], max_length, preprocessing_config)
        if len(preds) == 0:
            print(f"  No predictions for {args.read_id}")
            continue

        all_preds[model_name] = preds

        max_gap_ori = args.max_gap_ori if args.max_gap_ori is not None else args.max_gap
        max_gap_by_class = {1: args.max_gap, 2: args.max_gap, 3: max_gap_ori}
        for class_id, class_name in [(1, "left_fork"), (2, "right_fork"), (3, "origin")]:
            events = windows_to_events(preds, class_id, args.threshold,
                                       min_windows=1, max_gap=max_gap_by_class[class_id])
            events = events[events["read_id"] == args.read_id]
            all_events.append((model_name, class_name, events))

        tf.keras.backend.clear_session()

    # ── Per-model GT annotation sets ──────────────────────────────────────────
    # Each model uses its own training BED files (from its config).
    # Human-only BED files (--human-*-bed) are used to distinguish solid vs hatched.
    model_gt = []   # list of (lf_df, rf_df, ori_df) one per model
    for i, model_path in enumerate(args.model):
        if args.model_config and i < len(args.model_config):
            with open(args.model_config[i]) as fh:
                mc = yaml.safe_load(fh)
            cfg = mc["data"]
        else:
            cfg = config["data"]
        model_gt.append((
            load_bed(cfg.get("left_forks_bed")),
            load_bed(cfg.get("right_forks_bed")),
            load_bed(cfg.get("ori_annotations_bed")),
        ))

    # ── PLOT ──────────────────────────────────────────────────────────────────
    # Layout: signal | [GT + prob + events] x n_models
    n_models = len(args.model)
    n_rows = 1 + 3 * n_models   # signal + (GT + prob + events) per model
    height_ratios = [3] + [0.6, 2, 0.7] * n_models
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 + 3.3 * n_models + 1),
                              sharex=True, height_ratios=height_ratios,
                              gridspec_kw={"hspace": 0.08})

    x = xy["start"].to_numpy()
    y = xy["signal"].to_numpy()

    def _is_human(df_human, read_id, start, end):
        for row in df_human[df_human["read_id"] == read_id].itertuples(index=False):
            if start < int(row.end) and end > int(row.start):
                return True
        return False

    # Panel 0: BrdU signal only
    ax = axes[0]
    ax.step(x, y, where="post", color="black", linewidth=1.2, label="BrdU signal")
    ax.fill_between(x, y, step="post", alpha=0.12, color="gray")
    ax.set_ylabel("BrdU signal")
    ax.set_title(f"Read {args.read_id}  |  threshold={args.threshold}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

    # Panels 1+ : GT bar + probability track + event span bar, one triple per model
    PROB_COLS = {
        "left_fork":  COL_REAL_LEFT,
        "right_fork": COL_REAL_RIGHT,
        "origin":     COL_REAL_ORI,
    }
    for m_idx, (model_path, model_name) in enumerate(zip(args.model, model_names)):
        lf_gt, rf_gt, ori_gt = model_gt[m_idx]

        # ── GT bar for this model ──
        ax_gt = axes[1 + 3 * m_idx]
        ax_gt.set_ylabel(f"{model_name}\nGT", fontsize=8, rotation=0, labelpad=40, va="center")
        ax_gt.set_yticks([])
        ax_gt.set_ylim(0, 1)
        ax_gt.grid(alpha=0.15, axis="x")
        for df_all, df_h, color, y0, y1 in [
            (lf_gt,  left_human,  COL_REAL_LEFT,  0.68, 0.98),
            (rf_gt,  right_human, COL_REAL_RIGHT, 0.34, 0.64),
            (ori_gt, ori_human,   COL_REAL_ORI,   0.02, 0.32),
        ]:
            subset = df_all[df_all["read_id"] == args.read_id]
            for row in subset.itertuples(index=False):
                human = show_human_pseudo and _is_human(df_h, args.read_id, int(row.start), int(row.end))
                hatch = None if human else "////"
                ax_gt.axvspan(int(row.start), int(row.end), ymin=y0, ymax=y1,
                              color=color, alpha=0.7 if human else 0.4,
                              hatch=hatch, edgecolor=color if hatch else None)
        if show_human_pseudo:
            handles = [
                mpatches.Patch(color=COL_REAL_LEFT,  label="LF hum"),
                mpatches.Patch(color=COL_REAL_LEFT,  label="LF pseudo", hatch="////", alpha=0.5),
                mpatches.Patch(color=COL_REAL_RIGHT, label="RF hum"),
                mpatches.Patch(color=COL_REAL_RIGHT, label="RF pseudo", hatch="////", alpha=0.5),
                mpatches.Patch(color=COL_REAL_ORI,   label="ORI hum"),
                mpatches.Patch(color=COL_REAL_ORI,   label="ORI pseudo", hatch="////", alpha=0.5),
            ]
            ax_gt.legend(handles=handles, loc="upper right", ncol=6, fontsize=6, framealpha=0.7)
        else:
            handles = [mpatches.Patch(color=COL_REAL_LEFT,  label="L"),
                       mpatches.Patch(color=COL_REAL_RIGHT, label="R"),
                       mpatches.Patch(color=COL_REAL_ORI,   label="ORI")]
            ax_gt.legend(handles=handles, loc="upper right", ncol=3, fontsize=7, framealpha=0.7)

        preds = all_preds.get(model_name)

        # ── probability track panel ──
        ax_prob = axes[2 + 3 * m_idx]
        ax_prob.set_ylabel(f"{model_name}\nprob", fontsize=8, rotation=0, labelpad=40, va="center")
        ax_prob.set_ylim(0, 1)
        ax_prob.axhline(args.threshold, color="gray", lw=0.8, ls="--", alpha=0.7,
                        label=f"thresh={args.threshold}")
        ax_prob.grid(alpha=0.2)

        if preds is not None and len(preds) > 0:
            read_preds = preds[preds["read_id"] == args.read_id].sort_values("start")
            px = read_preds["start"].to_numpy()
            for class_name, pcol in PROB_COLS.items():
                prob_col = f"prob_{class_name}"
                if prob_col in read_preds.columns:
                    ax_prob.step(px, read_preds[prob_col].to_numpy(),
                                 where="post", color=pcol, linewidth=1.2, alpha=0.85,
                                 label=class_name.replace("_", " "))
        ax_prob.legend(loc="upper right", ncol=4, fontsize=7, framealpha=0.85)

        # ── predicted event span bar ──
        ax_ev = axes[3 + 3 * m_idx]
        ax_ev.set_ylabel(f"{model_name}\nevents", fontsize=8, rotation=0, labelpad=40, va="center")
        ax_ev.set_yticks([])
        ax_ev.set_ylim(0, 1)
        ax_ev.grid(alpha=0.15, axis="x")

        for class_name, y0, y1 in [("left_fork", 0.68, 0.98),
                                    ("right_fork", 0.34, 0.64),
                                    ("origin",     0.02, 0.32)]:
            class_col = PROB_COLS[class_name]
            match = [(mn, cn, ev) for mn, cn, ev in all_events
                     if mn == model_name and cn == class_name]
            if match:
                for row in match[0][2].itertuples(index=False):
                    ax_ev.axvspan(int(row.start), int(row.end), ymin=y0, ymax=y1,
                                  color=class_col, alpha=0.55)
                    mid = (int(row.start) + int(row.end)) / 2
                    yc  = (y0 + y1) / 2
                    if class_name == "origin" and hasattr(row, "mean_brdu_signal") and not np.isnan(row.mean_brdu_signal):
                        ax_ev.text(mid, yc, f"BrdU={row.mean_brdu_signal:.3f}",
                                   ha="center", va="center", fontsize=5.5,
                                   color="white", fontweight="bold", clip_on=True)
                    elif class_name in ("left_fork", "right_fork"):
                        # Draw the BrdU slope as a fitted line on the signal panel
                        mask = (xy["start"] >= int(row.start)) & (xy["start"] < int(row.end))
                        seg = xy[mask].sort_values("start")
                        if len(seg) >= 2:
                            xs = seg["start"].to_numpy()
                            ys = seg["signal"].to_numpy()
                            slope, intercept = np.polyfit(xs, ys, 1)
                            axes[0].plot(
                                [xs[0], xs[-1]],
                                [slope * xs[0] + intercept, slope * xs[-1] + intercept],
                                color=class_col, linewidth=2.5, linestyle="--", alpha=0.9,
                                zorder=5,
                            )

        handles = [mpatches.Patch(color=COL_REAL_LEFT,  label="pred L"),
                   mpatches.Patch(color=COL_REAL_RIGHT, label="pred R"),
                   mpatches.Patch(color=COL_REAL_ORI,   label="pred ORI")]
        ax_ev.legend(handles=handles, loc="upper right", ncol=3, fontsize=7, framealpha=0.7)

    axes[-1].set_xlabel("Genomic position (bp)")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
