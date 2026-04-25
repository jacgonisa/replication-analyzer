#!/usr/bin/env python
"""IoU-based benchmark using cached per-window predictions (no re-inference).

Evaluates all cached FORTE models against pure Nerea manual annotations
(leftForks_combined.bed, rightForks_combined.bed, ORIs_combined_cleaned.bed).
Computes precision / recall / F1 at multiple IoU thresholds and prob thresholds.

Usage (from /replication-analyzer/):
  python CODEX/scripts/iou_benchmark_from_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.evaluation import windows_to_events

ANNO_DIR = ROOT / "data/case_study_jan2026/combined/annotations"
OUT_DIR  = CODEX_ROOT / "results/iou_benchmark_multimodel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR  = CODEX_ROOT / "results/readlen_analysis"
PRED_V43  = CODEX_ROOT / "results/forte_v4.3/reannotation_run3/reannotated_segments.tsv"

MODELS = [
    ("FORTE v1",   PRED_DIR / "predictions_v1.tsv"),
    ("FORTE v2",   PRED_DIR / "predictions_v2.tsv"),
    ("FORTE v3",   PRED_DIR / "predictions_v3.tsv"),
    ("FORTE v4",   PRED_DIR / "predictions_v4.tsv"),
    ("FORTE v4.2", PRED_DIR / "predictions_v4.2.tsv"),
    ("FORTE v5",   PRED_DIR / "predictions_v5.tsv"),
    ("FORTE v4.3", PRED_V43),
]

PROB_THRESHOLD = 0.4
MAX_GAP        = 5000
MIN_WINDOWS    = 1
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
CLASS_MAP      = {"left_fork": 1, "right_fork": 2, "origin": 3}


# ── helpers ───────────────────────────────────────────────────────────────────

def rvs_to_uuid(s):
    return s.replace("rvs", "-")


def load_nerea():
    def _load(path, convert=True):
        df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                         names=["chr", "start", "end", "read_id"])
        if convert:
            df["read_id"] = df["read_id"].apply(rvs_to_uuid)
        return df
    lf  = _load(ANNO_DIR / "leftForks_combined.bed")
    rf  = _load(ANNO_DIR / "rightForks_combined.bed")
    ori = _load(ANNO_DIR / "ORIs_combined_cleaned.bed", convert=False)
    return {"left_fork": lf, "right_fork": rf, "origin": ori}


def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def evaluate_iou(pred_df: pd.DataFrame, gt_df: pd.DataFrame,
                 iou_threshold: float) -> dict:
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

    fn   = len(gt_df) - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_pred=len(pred_df), n_gt=len(gt_df))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading Nerea annotations...")
    nerea = load_nerea()
    for k, v in nerea.items():
        print(f"  {k}: {len(v):,}")

    # Only evaluate on reads that Nerea annotated
    nerea_reads = (set(nerea["left_fork"].read_id)
                   | set(nerea["right_fork"].read_id)
                   | set(nerea["origin"].read_id))
    print(f"  Total Nerea reads: {len(nerea_reads):,}")

    rows = []

    for model_name, pred_path in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  ({pred_path.name})")
        preds = pd.read_csv(pred_path, sep="\t")
        preds_nerea = preds[preds["read_id"].isin(nerea_reads)].copy()
        print(f"  Windows on Nerea reads: {len(preds_nerea):,}")

        for class_name, class_id in CLASS_MAP.items():
            ai_events = windows_to_events(
                predictions=preds_nerea,
                class_id=class_id,
                prob_threshold=PROB_THRESHOLD,
                min_windows=MIN_WINDOWS,
                max_gap=MAX_GAP,
            )
            gt_df = nerea[class_name]
            print(f"  {class_name}: {len(ai_events):,} AI events  vs  {len(gt_df):,} GT")

            for iou_thr in IOU_THRESHOLDS:
                m = evaluate_iou(ai_events, gt_df, iou_thr)
                rows.append({
                    "model": model_name,
                    "class": class_name,
                    "iou_threshold": iou_thr,
                    **m,
                })
                if iou_thr == 0.3:
                    print(f"    IoU≥0.3: F1={m['f1']:.3f}  "
                          f"prec={m['precision']:.3f}  rec={m['recall']:.3f}")

    summary = pd.DataFrame(rows)
    tsv_path = OUT_DIR / "iou_benchmark_summary.tsv"
    summary.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSummary saved: {tsv_path}")

    # ── Plot 1: F1 / Precision / Recall bar chart at IoU=0.3 ─────────────────
    model_names  = [m[0] for m in MODELS]
    class_names  = ["left_fork", "right_fork", "origin"]
    class_labels = {"left_fork": "Left Fork", "right_fork": "Right Fork", "origin": "Origin"}
    colors       = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22",
                    "#9b59b6", "#1abc9c", "#e91e63"]

    sub03 = summary[summary["iou_threshold"] == 0.3]
    x     = np.arange(len(class_names))
    n     = len(model_names)
    width = 0.11
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"IoU-based evaluation (prob_thr={PROB_THRESHOLD}, IoU≥0.3) — Nerea annotated reads",
                 fontsize=12, fontweight="bold")

    for metric, ax, ylabel in [("f1", axes[0], "F1"),
                                ("precision", axes[1], "Precision"),
                                ("recall", axes[2], "Recall")]:
        for i, mname in enumerate(model_names):
            sub_m = sub03[sub03["model"] == mname]
            vals  = [sub_m[sub_m["class"] == cl][metric].values[0]
                     if len(sub_m[sub_m["class"] == cl]) > 0 else 0.0
                     for cl in class_names]
            bars = ax.bar(x + offsets[i], vals, width * 0.9,
                          label=mname, color=colors[i], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.2f}", ha="center", fontsize=6.5, va="bottom")

        ax.set_xticks(x)
        ax.set_xticklabels([class_labels[c] for c in class_names],
                           fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, ncol=1)

    plt.tight_layout()
    p1 = OUT_DIR / "iou_benchmark_bar_iou03.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1}")

    # ── Plot 2: F1 vs IoU threshold sweep ─────────────────────────────────────
    ls_map = {"left_fork": "solid", "right_fork": "dashed", "origin": "dotted"}
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(4 * len(model_names), 4),
                             sharey=True, gridspec_kw={"wspace": 0.08})
    fig.suptitle(f"F1 vs IoU threshold  (prob_thr={PROB_THRESHOLD})",
                 fontsize=12, fontweight="bold")

    class_colors = {"left_fork": "#3498db", "right_fork": "#e74c3c", "origin": "#2ecc71"}

    for ax, mname in zip(axes, model_names):
        sub_m = summary[summary["model"] == mname]
        for cl in class_names:
            row = sub_m[sub_m["class"] == cl].sort_values("iou_threshold")
            ax.plot(row["iou_threshold"], row["f1"],
                    color=class_colors[cl], linestyle=ls_map[cl],
                    linewidth=2, marker="o", markersize=5, label=class_labels[cl])
        ax.set_title(mname, fontsize=9, fontweight="bold")
        ax.set_xlabel("IoU thr", fontsize=8)
        ax.set_xlim(0.05, 0.65)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("F1", fontsize=11)
    plt.tight_layout()
    p2 = OUT_DIR / "iou_benchmark_f1_sweep.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2}")

    # ── Plot 3: Heatmap — F1 at IoU=0.3 across models × classes ─────────────
    pivot_f1 = sub03.pivot(index="model", columns="class", values="f1").reindex(model_names)
    pivot_f1 = pivot_f1[class_names]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot_f1.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([class_labels[c] for c in class_names], fontsize=10)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)
    for i in range(len(model_names)):
        for j in range(len(class_names)):
            v = pivot_f1.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if 0.3 < v < 0.8 else "white",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, label="F1 (IoU≥0.3)")
    ax.set_title(f"F1 heatmap — IoU≥0.3  (prob_thr={PROB_THRESHOLD})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    p3 = OUT_DIR / "iou_benchmark_heatmap_f1.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
