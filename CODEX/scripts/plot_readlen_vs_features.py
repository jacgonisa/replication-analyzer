#!/usr/bin/env python
"""Plot N events per read vs read length — ForkML-style boxplots, multi-model.

For each event type (ORI, left fork, right fork) produces one figure with
one sub-panel per model (+ real annotations).  Each panel shows:
  - X-axis : read-length bins in 10 kb increments
  - Y-axis : number of events detected on that read
  - Box    : IQR (25th–75th percentile)
  - Whisker: 1.5×IQR (Tukey)
  - Red dot: mean
  - Number below x-axis: n reads in that bin

Useful for collaborators asking "how many ORIs/forks do we get on reads of
different lengths?" and for choosing minimum read length for adaptive sampling.

Usage (from /replication-analyzer/):
  CUDA_VISIBLE_DEVICES="" /home/jg2070/miniforge3/envs/ONT/bin/python -u \\
      CODEX/scripts/plot_readlen_vs_features.py \\
      --output-dir CODEX/results/readlen_analysis

Models are defined in MODELS dict below. Predictions are cached per model in
output-dir/predictions_{model_name}.tsv — delete to re-run inference.
The v2 fork-threshold-sweep cache is reused automatically for model "v2".
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

BASE = Path("/mnt/ssd-4tb/crisanto_project/AI_annotation/replication-analyzer")
XY_CACHE     = BASE / "CODEX/results/cache/xy_data.pkl"
REAL_ORI_BED = BASE / "data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed"
REAL_LF_BED  = BASE / "data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed"
REAL_RF_BED  = BASE / "data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed"

# ── Models to compare — {display_name: (keras_path, config_yaml)} ────────────
MODELS = {
    "v1":      (BASE / "CODEX/models/forte_v1.keras",
                BASE / "CODEX/configs/forte_v1.yaml"),
    "v2":      (BASE / "CODEX/models/forte_v2.keras",
                BASE / "CODEX/configs/forte_v2.yaml"),
    "v2_cons": (BASE / "CODEX/models/forte_v2_conservative.keras",
                BASE / "CODEX/configs/forte_v2_conservative.yaml"),
    "v3":      (BASE / "CODEX/models/forte_v3.keras",
                BASE / "CODEX/configs/forte_v3.yaml"),
    "v4":      (BASE / "CODEX/models/forte_v4.keras",
                BASE / "CODEX/configs/forte_v4.yaml"),
    "v4.2":    (BASE / "CODEX/models/forte_v4.2.keras",
                BASE / "CODEX/configs/forte_v4.2.yaml"),
    "v5":      (BASE / "CODEX/models/forte_v5.keras",
                BASE / "CODEX/configs/forte_v5.yaml"),
}

# Colours per model (+ real annotations)
MODEL_COLORS = {
    "real_annot": "#2c3e50",
    "v1":         "#e74c3c",
    "v2":         "#2980b9",
    "v2_cons":    "#8e44ad",
    "v3":         "#27ae60",
    "v4":         "#e67e22",
    "v4.2":       "#c0392b",
    "v5":         "#16a085",
}

# Prediction thresholds applied to all models
PRED_FORK_THR = 0.40
PRED_ORI_THR  = 0.40


# ── binning ───────────────────────────────────────────────────────────────────
def make_bins(min_kb=0, max_kb=300, step=10):
    """Return (edges, labels) for 10 kb bins from min_kb to max_kb then '>max_kb'.

    Edges define left-closed intervals [min_kb, min_kb+step), ..., [max_kb, ∞).
    min_kb=0 gives [0,10), [10,20), [20,30), [30,40), ...
    """
    start = max(0, min_kb)   # never negative
    edges  = list(range(start, max_kb + step, step)) + [int(1e6)]
    labels = [f"{e}–{e+step}" for e in range(start, max_kb, step)] + [f">{max_kb}"]
    return edges, labels


# ── helpers ───────────────────────────────────────────────────────────────────
def load_bed(path):
    df = pd.read_csv(path, sep="\t", header=None, low_memory=False)
    df.columns = ["chr", "start", "end", "read_id"] + [
        f"c{i}" for i in range(4, len(df.columns))
    ]
    return df


def get_read_lengths(xy_data):
    g = xy_data.groupby("read_id")
    return (g["end"].max() - g["start"].min()).rename("read_length_bp")


def count_events_per_read(bed_df, all_read_ids):
    """Return Series indexed by read_id with count of events (0 if none)."""
    counts = bed_df.groupby("read_id").size()
    return counts.reindex(all_read_ids, fill_value=0)


def load_or_run_predictions(model_name, model_path, config_path,
                             xy_data, cache_dir):
    """Return window-level predictions DataFrame (cached on disk)."""
    v2_cache  = BASE / "CODEX/results/fork_threshold_sweep_predictions.tsv"
    cache_path = cache_dir / f"predictions_{model_name}.tsv"

    # Reuse v2 sweep cache if available
    if model_name == "v2" and v2_cache.exists() and not cache_path.exists():
        import shutil
        print(f"  [{model_name}] Reusing v2 sweep cache")
        shutil.copy(v2_cache, cache_path)

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1e6
        print(f"  [{model_name}] Loading cache ({size_mb:.0f} MB)...")
        return pd.read_csv(cache_path, sep="\t")

    print(f"  [{model_name}] Running inference on "
          f"{xy_data['read_id'].nunique():,} reads...")
    import tensorflow as tf
    import yaml
    from replication_analyzer_codex.losses import (
        MaskedClassPrecision, MaskedClassRecall, MaskedMacroF1,
        MaskedMeanIoU, SparseCategoricalFocalLoss,
    )
    from replication_analyzer.models.base import SelfAttention
    from replication_analyzer_codex.evaluation import predict_reads

    with open(config_path) as f:
        src_cfg = yaml.safe_load(f)

    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
            "MaskedMacroF1": MaskedMacroF1,
            "MaskedMeanIoU": MaskedMeanIoU,
            "MaskedClassPrecision": MaskedClassPrecision,
            "MaskedClassRecall": MaskedClassRecall,
            "SelfAttention": SelfAttention,
        },
        compile=False,
        safe_mode=False,
    )
    max_length = model.input_shape[1]
    read_ids   = xy_data["read_id"].unique().tolist()
    preds = predict_reads(
        model=model, xy_data=xy_data, read_ids=read_ids,
        max_length=max_length,
        preprocessing_config=src_cfg["preprocessing"],
    )
    preds.to_csv(cache_path, sep="\t", index=False)
    print(f"  [{model_name}] Saved → {cache_path}")
    tf.keras.backend.clear_session()
    return preds


def events_from_predictions(pred_df, class_id, prob_threshold):
    """Return BED-like DataFrame with chr/start/end/read_id columns."""
    from replication_analyzer_codex.evaluation import windows_to_events
    evs = windows_to_events(
        predictions=pred_df, class_id=class_id,
        prob_threshold=prob_threshold, min_windows=1, max_gap=5000,
    )
    if isinstance(evs, pd.DataFrame) and not evs.empty:
        return evs[["chr", "start", "end", "read_id"]].copy()
    return pd.DataFrame(columns=["chr", "start", "end", "read_id"])


# ── ForkML-style boxplot ──────────────────────────────────────────────────────
def forkml_boxplot(ax, counts_per_read, read_lengths_kb, bin_edges, bin_labels,
                   color, title, ylabel="N events / read"):
    """
    Draw one ForkML-style panel:
      - one box per bin (shows distribution of event count over reads in that bin)
      - red dot = mean
      - n per bin shown below x-axis
    Only reads with ≥1 event contribute to the boxes (consistent with ForkML
    which plots fork speed only on reads where forks were detected).
    Show n_detected / n_total below the x-axis.
    """
    df = pd.DataFrame({
        "length_kb": read_lengths_kb,
        "n_events":  counts_per_read,
    })
    # bin_edges already spans [min_kb, ..., max_kb, inf]; reads below min_kb
    # are assigned NaN by pd.cut and silently dropped later via isin(bin_labels).
    full_edges = ([0] + bin_edges) if bin_edges[0] > 0 else bin_edges
    n_extra    = 1 if bin_edges[0] > 0 else 0
    full_labels = (["<" + str(bin_edges[0])] * n_extra) + list(bin_labels)
    df["bin"] = pd.cut(
        df["length_kb"],
        bins=full_edges,
        labels=full_labels,
        right=False,
        include_lowest=True,
    )
    # Keep only the user-requested bins (drops any "<min_kb" overflow label)
    df = df[df["bin"].isin(bin_labels)]

    xs       = []
    data_all = []
    means    = []
    n_labels = []   # "n_det/n_total"

    for i, lbl in enumerate(bin_labels):
        grp = df[df["bin"] == lbl]["n_events"]
        n_total = len(grp)
        grp_det = grp[grp > 0]
        n_det   = len(grp_det)
        xs.append(i)
        data_all.append(grp_det.values if n_det > 0 else np.array([0]))
        means.append(grp.mean() if n_total > 0 else 0)
        n_labels.append(f"{n_det}\n/{n_total}")

    # Draw boxes
    bp = ax.boxplot(
        data_all,
        positions=xs,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        boxprops=dict(facecolor=color, alpha=0.55),
    )
    # Red dot = mean (over all reads in bin, including those with 0 events)
    ax.scatter(xs, means, color="red", zorder=5, s=40, label="Mean (all reads)")

    # X-axis ticks
    ax.set_xticks(xs)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)

    # n per bin below x-axis
    y_min = ax.get_ylim()[0]
    for xi, lbl in zip(xs, n_labels):
        ax.text(xi, y_min - 0.12 * (ax.get_ylim()[1] - y_min),
                lbl, ha="center", va="top", fontsize=6.5, color="#333")

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.7)
    ax.legend(fontsize=7, loc="upper left")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="CODEX/results/readlen_analysis")
    parser.add_argument("--min-kb",  type=int, default=0,
                        help="Start of first 10 kb bin (0 includes 0-10, 10-20, 20-30 range)")
    parser.add_argument("--max-kb",  type=int, default=300,
                        help="Last full 10 kb bin (reads above → '>max_kb' bin)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_edges, bin_labels = make_bins(args.min_kb, args.max_kb, step=10)

    # ── Load XY data ──────────────────────────────────────────────────────
    print("Loading XY cache...")
    with open(XY_CACHE, "rb") as fh:
        xy_data = pickle.load(fh)
    read_lengths    = get_read_lengths(xy_data)
    read_lengths_kb = read_lengths / 1000
    all_read_ids    = read_lengths.index.tolist()
    print(f"  {len(all_read_ids):,} reads  median {read_lengths_kb.median():.1f} kb")

    # ── Real annotations ─────────────────────────────────────────────────
    print("Loading real annotations...")
    ori_real  = load_bed(REAL_ORI_BED)
    lf_real   = load_bed(REAL_LF_BED)
    rf_real   = load_bed(REAL_RF_BED)

    # ── Build event counts per read ───────────────────────────────────────
    # {source_name: {"ORI": Series, "LF": Series, "RF": Series}}
    event_counts = {}

    event_counts["real_annot"] = {
        "ORI": count_events_per_read(ori_real, all_read_ids),
        "LF":  count_events_per_read(lf_real,  all_read_ids),
        "RF":  count_events_per_read(rf_real,  all_read_ids),
    }
    print("  real_annot: "
          f"ORI reads={int((event_counts['real_annot']['ORI']>0).sum())}  "
          f"LF={(int((event_counts['real_annot']['LF']>0).sum()))}  "
          f"RF={(int((event_counts['real_annot']['RF']>0).sum()))}")

    print("\nLoading/running model predictions...")
    for model_name, (model_path, config_path) in MODELS.items():
        if not model_path.exists():
            print(f"  [{model_name}] Model file not found — skipping")
            continue
        pred_df = load_or_run_predictions(
            model_name, model_path, config_path, xy_data, out_dir
        )
        ori_bed = events_from_predictions(pred_df, 3, PRED_ORI_THR)
        lf_bed  = events_from_predictions(pred_df, 1, PRED_FORK_THR)
        rf_bed  = events_from_predictions(pred_df, 2, PRED_FORK_THR)
        event_counts[model_name] = {
            "ORI": count_events_per_read(ori_bed, all_read_ids),
            "LF":  count_events_per_read(lf_bed,  all_read_ids),
            "RF":  count_events_per_read(rf_bed,  all_read_ids),
        }
        print(f"  [{model_name}]  ORI reads={int((event_counts[model_name]['ORI']>0).sum())}  "
              f"LF={int((event_counts[model_name]['LF']>0).sum())}  "
              f"RF={int((event_counts[model_name]['RF']>0).sum())}")

    sources = list(event_counts.keys())   # real_annot + models that loaded

    # ── ForkML-style figures — one per event type ─────────────────────────
    for event_type in ["ORI", "LF", "RF"]:
        n_src  = len(sources)
        n_cols = min(n_src, 3)
        n_rows = (n_src + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(7 * n_cols, 5.5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Number of {event_type} events per read vs read length\n"
            f"(boxes = detected reads only; red dot = mean over all reads; "
            f"prob thr ORI={PRED_ORI_THR} fork={PRED_FORK_THR})",
            fontsize=12, fontweight="bold",
        )

        for idx, src in enumerate(sources):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            color = MODEL_COLORS.get(src, "#555555")
            forkml_boxplot(
                ax=ax,
                counts_per_read=event_counts[src][event_type],
                read_lengths_kb=read_lengths_kb,
                bin_edges=bin_edges,
                bin_labels=bin_labels,
                color=color,
                title=src,
            )

        # Hide unused axes
        for idx in range(len(sources), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = out_dir / f"readlen_vs_{event_type.lower()}_forkml.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out_path}")

    # ── Comparison figure — all sources in one panel per event type ───────
    # One row per event type, one column per source — compact version
    n_src  = len(sources)
    fig, axes = plt.subplots(
        3, n_src,
        figsize=(5 * n_src, 5 * 3),
        squeeze=False,
    )
    fig.suptitle(
        "N events per read vs read length — all models (10 kb bins)\n"
        f"(boxes = detected reads; red dot = mean; "
        f"n_det/n_total below x-axis)",
        fontsize=13, fontweight="bold",
    )
    for row, event_type in enumerate(["ORI", "LF", "RF"]):
        for col, src in enumerate(sources):
            ax = axes[row][col]
            color = MODEL_COLORS.get(src, "#555555")
            forkml_boxplot(
                ax=ax,
                counts_per_read=event_counts[src][event_type],
                read_lengths_kb=read_lengths_kb,
                bin_edges=bin_edges,
                bin_labels=bin_labels,
                color=color,
                title=f"{event_type} — {src}",
                ylabel="N events / read" if col == 0 else "",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "readlen_vs_events_all_models.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined figure → {out_path}")

    # ── Read length distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(read_lengths_kb[read_lengths_kb <= 400].values, bins=100,
            color="#555", alpha=0.7, edgecolor="none")
    for kb, ls, col, lbl in [
        (read_lengths_kb.median(), "--", "red",    f"Median {read_lengths_kb.median():.0f} kb"),
        (100,                       ":", "orange", "100 kb"),
    ]:
        ax.axvline(kb, color=col, linestyle=ls, linewidth=1.5, label=lbl)
    ax.set_xlabel("Read length (kb)")
    ax.set_ylabel("Number of reads")
    ax.set_title(f"Read length distribution (n={len(all_read_ids):,})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / "read_length_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution → {out_path}")

    # ── TSV summary ───────────────────────────────────────────────────────
    df_all = pd.DataFrame({
        "read_id":       all_read_ids,
        "read_length_kb": read_lengths_kb.values,
    })
    _full_edges2  = ([0] + bin_edges) if bin_edges[0] > 0 else bin_edges
    _full_labels2 = (["<" + str(bin_edges[0])] if bin_edges[0] > 0 else []) + list(bin_labels)
    df_all["bin"] = pd.cut(
        df_all["read_length_kb"],
        bins=_full_edges2,
        labels=_full_labels2,
        right=False, include_lowest=True,
    )
    rows = []
    for src in sources:
        for event_type in ["ORI", "LF", "RF"]:
            cnts = event_counts[src][event_type]
            df_all["n_events"] = cnts.values
            grp = df_all.groupby("bin", observed=True)
            for bn, g in grp:
                rows.append({
                    "source":     src,
                    "event_type": event_type,
                    "bin_kb":     str(bn),
                    "n_reads":    len(g),
                    "n_detected": int((g["n_events"] > 0).sum()),
                    "mean_events": round(float(g["n_events"].mean()), 3),
                    "median_events": round(float(g["n_events"].median()), 3),
                })
    pd.DataFrame(rows).to_csv(
        out_dir / "readlen_event_counts.tsv", sep="\t", index=False
    )
    print(f"Saved summary TSV → {out_dir}/readlen_event_counts.tsv")


if __name__ == "__main__":
    main()
