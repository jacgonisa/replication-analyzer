#!/usr/bin/env python
"""Run inference for multiple FORTE models on Nerea-annotated reads only,
compute Nerea&AI / onlyNerea / onlyAI overlap, and save a summary TSV + bar plot."""

from __future__ import annotations
import sys, pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

ROOT      = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.evaluation import predict_reads, windows_to_events
from replication_analyzer_codex.losses import (
    SparseCategoricalFocalLoss, MaskedMacroF1, MaskedClassPrecision,
    MaskedClassRecall, MaskedMeanIoU,
)
from replication_analyzer.models.base import SelfAttention
from replication_analyzer.models.losses import MultiClassFocalLoss
from replication_analyzer.training.callbacks import MultiClassF1Score

CUSTOM_OBJECTS = {
    "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss,
    "MaskedMacroF1": MaskedMacroF1,
    "MaskedClassPrecision": MaskedClassPrecision,
    "MaskedClassRecall": MaskedClassRecall,
    "MaskedMeanIoU": MaskedMeanIoU,
    "SelfAttention": SelfAttention,
    "MultiClassFocalLoss": MultiClassFocalLoss,
    "MultiClassF1Score": MultiClassF1Score,
}

ANNO_DIR = ROOT / "data/case_study_jan2026/combined/annotations"
OUT_DIR  = CODEX_ROOT / "results/annotation_overlap_multimodel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THR       = 0.40
MAX_GAP   = 5000

MODELS = [
    ("FORTE v1",        "CODEX/models/forte_v1.keras",         "CODEX/configs/forte_v1.yaml"),
    ("FORTE v2",        "CODEX/models/forte_v2.keras",         "CODEX/configs/forte_v2.yaml"),
    ("FORTE v3",        "CODEX/models/forte_v3.keras",         "CODEX/configs/forte_v3.yaml"),
    ("FORTE v4",        "CODEX/models/forte_v4.keras",         "CODEX/configs/forte_v4.yaml"),
    ("FORTE v4.2",      "CODEX/models/forte_v4.2.keras",       "CODEX/configs/forte_v4.2.yaml"),
    ("FORTE v4.3",      "CODEX/models/forte_v4.3_run3.keras",  "CODEX/configs/forte_v4.3.yaml"),
]


def rvs_to_uuid(s): return s.replace("rvs", "-")


def load_nerea():
    def _load(path, convert=True):
        df = pd.read_csv(path, sep="\t", header=None, usecols=[0,1,2,3],
                         names=["chr","start","end","read_id"])
        if convert:
            df["read_id"] = df["read_id"].apply(rvs_to_uuid)
        return df
    lf  = _load(ANNO_DIR / "leftForks_combined.bed")
    rf  = _load(ANNO_DIR / "rightForks_combined.bed")
    ori = _load(ANNO_DIR / "ORIs_combined_cleaned.bed", convert=False)
    return lf, rf, ori


def compute_iou(a_s, a_e, b_s, b_e):
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    union = (a_e - a_s) + (b_e - b_s) - inter
    return inter / union if union > 0 else 0.0


def overlap_counts(nerea_df, ai_df):
    """Return (n_nerea_and_ai, n_only_nerea, n_only_ai)."""
    ai_by_read = {}
    for rid, grp in ai_df.groupby("read_id"):
        ai_by_read[rid] = grp[["start","end"]].values.tolist()

    nerea_reads = set(nerea_df["read_id"])
    n_match = 0
    n_miss  = 0
    for _, row in nerea_df.iterrows():
        rid = row["read_id"]
        best = 0.0
        if rid in ai_by_read:
            for (s, e) in ai_by_read[rid]:
                iou = compute_iou(row["start"], row["end"], s, e)
                if iou > best:
                    best = iou
        if best > 0:
            n_match += 1
        else:
            n_miss += 1

    n_only_ai = sum(1 for _, row in ai_df.iterrows()
                    if row["read_id"] not in nerea_reads)
    return n_match, n_miss, n_only_ai


def main():
    print("Loading Nerea annotations...")
    lf_nerea, rf_nerea, ori_nerea = load_nerea()
    nerea_reads = list(
        set(lf_nerea.read_id) | set(rf_nerea.read_id) | set(ori_nerea.read_id)
    )
    print(f"  {len(nerea_reads)} Nerea reads")

    print("Loading XY cache...")
    with open(ROOT / "CODEX/results/cache/xy_data.pkl", "rb") as f:
        xy = pickle.load(f)
    xy_nerea = xy[xy["read_id"].isin(set(nerea_reads))].copy()
    print(f"  XY rows for Nerea reads: {len(xy_nerea):,}")

    rows = []

    for model_name, model_path, config_path in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({model_path})")

        with open(ROOT / config_path) as f:
            cfg = yaml.safe_load(f)
        preprocessing = cfg["preprocessing"]

        tf.keras.backend.clear_session()
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(
            str(ROOT / model_path), custom_objects=CUSTOM_OBJECTS
        )
        max_length = model.input_shape[1]
        print(f"  max_length={max_length}")

        preds = predict_reads(model, xy_nerea, nerea_reads, max_length, preprocessing)
        print(f"  Predictions: {len(preds):,} windows")

        tf.keras.backend.clear_session()

        for class_id, class_name, nerea_df in [
            (1, "left_fork",  lf_nerea),
            (2, "right_fork", rf_nerea),
            (3, "origin",     ori_nerea),
        ]:
            ai_events = windows_to_events(preds, class_id, THR,
                                          min_windows=1, max_gap=MAX_GAP)
            n_match, n_miss, n_only_ai = overlap_counts(nerea_df, ai_events)
            total_nerea = n_match + n_miss
            pct = 100 * n_match / total_nerea if total_nerea > 0 else 0
            print(f"  {class_name}: +AI={n_match}  onlyNerea={n_miss}  "
                  f"onlyAI={n_only_ai}  detected={pct:.1f}%")
            rows.append({
                "model": model_name,
                "event_type": class_name,
                "Nerea_and_AI": n_match,
                "only_Nerea": n_miss,
                "only_AI": n_only_ai,
                "total_Nerea": total_nerea,
                "pct_detected": round(pct, 1),
            })

    summary = pd.DataFrame(rows)
    tsv_path = OUT_DIR / "multimodel_overlap_summary.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSummary saved: {tsv_path}")

    # ── PLOT ──────────────────────────────────────────────────────────────
    event_types = ["left_fork", "right_fork", "origin"]
    event_labels = {"left_fork": "Left Fork", "right_fork": "Right Fork", "origin": "Origin"}
    model_names  = [m[0] for m in MODELS]
    n_models     = len(model_names)

    cat_colors = {
        "Nerea_and_AI": "#2ecc71",
        "only_Nerea":   "#e74c3c",
        "only_AI":      "#3498db",
    }
    cat_labels = {
        "Nerea_and_AI": "Nerea & AI",
        "only_Nerea":   "Only Nerea (missed by AI)",
        "only_AI":      "Only AI",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(f"Annotation overlap across FORTE models  (thr={THR})", fontsize=13, fontweight="bold")

    x = np.arange(n_models)
    width = 0.22

    for ax, et in zip(axes, event_types):
        sub = summary[summary["event_type"] == et].set_index("model").reindex(model_names)

        for j, (cat, color) in enumerate(cat_colors.items()):
            vals = sub[cat].fillna(0).values
            bars = ax.bar(x + (j - 1) * width, vals, width,
                          label=cat_labels[cat], color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                            str(int(v)), ha="center", va="bottom", fontsize=6.5)

        # pct detected as secondary line on twin axis
        ax2 = ax.twinx()
        pcts = sub["pct_detected"].fillna(0).values
        ax2.plot(x, pcts, "k--o", ms=5, lw=1.5, label="% detected")
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("% Nerea detected", fontsize=8)
        ax2.tick_params(labelsize=7)
        for xi, pct in zip(x, pcts):
            ax2.text(xi, pct + 3, f"{pct:.0f}%", ha="center", fontsize=6.5, color="black")

        ax.set_title(event_labels[et], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Number of annotations", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Shared legend
    handles = [mpatches.Patch(color=c, alpha=0.85, label=cat_labels[k])
               for k, c in cat_colors.items()]
    handles += [plt.Line2D([0],[0], color="black", ls="--", marker="o", ms=5, label="% Nerea detected")]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plot_path = OUT_DIR / "multimodel_overlap_barplot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
