#!/usr/bin/env python
"""Plot reads with novel predicted ORIs and reads where Nerea's ORIs were missed.

Novel ORI reads  : predicted ORIs with gt_source == "novel" (not in any training annotation)
Missed ORI reads : reads where Nerea's human-annotated ORIs were not recovered by the model

Two-panel style:
  Top    — BrdU signal (step) + GT annotation spans + predicted event spans (hatched)
  Bottom — per-class probability tracks (filled areas)

Usage (from /replication-analyzer/):
  /home/jg2070/miniforge3/envs/ONT/bin/python -u \\
      CODEX/scripts/plot_novel_and_missed_oris.py \\
      --config         CODEX/configs/forte_v5.0.yaml \\
      --annotations    CODEX/results/forte_v5.0/reannotation/forte_v5.0_annotations.tsv \\
      --novel-oris     CODEX/results/forte_v5.0/reannotation/novel_oris_analysis.tsv \\
      --recovery-tsv   CODEX/results/forte_v5.0/reannotation/nerea_oris_recovery_status.tsv \\
      --nerea-ori-bed  data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed \\
      --output-dir     CODEX/results/forte_v5.0/novel_and_missed_ori_plots \\
      --n-novel 5 --n-missed 5
"""

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

BASE = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(CODEX_ROOT))

# ── colours ──────────────────────────────────────────────────────────────────
COL_LF_GT      = "#1f77b4"   # blue  — training-label left fork
COL_RF_GT      = "#d62728"   # red   — training-label right fork
COL_ORI_GT     = "#2ca02c"   # green — training-label / Nerea ORI
COL_NEREA_ORI  = "#e67e22"   # orange — Nerea human ORI (highlighted separately)

COL_LF_PRED    = "#74b9ff"   # light blue
COL_RF_PRED    = "#ff7675"   # light red
COL_ORI_PRED   = "#55efc4"   # light green
COL_BG_PRED    = "#b2bec3"   # grey

PROB_THR = 0.40


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_xy(base_dir, run_dirs, read_id):
    rows = []
    for run_dir in run_dirs:
        f = Path(base_dir) / run_dir / f"plot_data_{read_id}.txt"
        if f.exists():
            df = pd.read_csv(f, sep="\t", header=None,
                             names=["chr", "start", "end", "signal"])
            rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True).sort_values("start").reset_index(drop=True)


def load_bed(path, usecols=4):
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["chr", "start", "end", "read_id"])
    cols = list(range(usecols))
    names = ["chr", "start", "end", "read_id"] + [f"_c{i}" for i in range(4, usecols)]
    df = pd.read_csv(path, sep="\t", header=None, usecols=cols, names=names)
    return df[["chr", "start", "end", "read_id"]]


def add_annotation_spans(ax, bed_df, read_id, color, label, alpha=0.30):
    sub = bed_df[bed_df["read_id"] == read_id]
    used = False
    for row in sub.itertuples(index=False):
        ax.axvspan(int(row.start), int(row.end), color=color, alpha=alpha,
                   label=label if not used else "_")
        used = True


# ── core plotting ─────────────────────────────────────────────────────────────

def plot_read(
    read_id: str,
    xy_df: pd.DataFrame,
    preds_df: pd.DataFrame,       # all per-window predictions for this run
    annotations: pd.DataFrame,    # reannotation events (all classes)
    lf_gt: pd.DataFrame,
    rf_gt: pd.DataFrame,
    ori_gt_train: pd.DataFrame,   # training-label ORIs (flanked pseudo + human)
    ori_gt_nerea: pd.DataFrame,   # Nerea's human ORIs specifically
    title: str,
    out_path: Path,
    highlight_nerea_oris: bool = False,   # for missed ORI plots
):
    """2-panel plot for one read."""
    read_preds = preds_df[preds_df["read_id"] == read_id].sort_values("start")
    if read_preds.empty or xy_df is None:
        print(f"  Skipping {read_id}: no prediction data")
        return

    positions = ((read_preds["start"] + read_preds["end"]) / 2).values

    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5]})

    # ── Panel 0: BrdU signal + annotation spans ───────────────────────────────
    ax_sig = axes[0]
    x = xy_df["start"].tolist() + [xy_df["end"].iloc[-1]]
    y = xy_df["signal"].tolist() + [xy_df["signal"].iloc[-1]]
    ax_sig.step(x, y, where="post", color="black", linewidth=1.2, zorder=3)
    ax_sig.fill_between(x, y, step="post", alpha=0.12, color="gray", zorder=2)

    # GT spans (training labels)
    add_annotation_spans(ax_sig, lf_gt,       read_id, COL_LF_GT,     "GT Left Fork",  alpha=0.28)
    add_annotation_spans(ax_sig, rf_gt,       read_id, COL_RF_GT,     "GT Right Fork", alpha=0.28)
    add_annotation_spans(ax_sig, ori_gt_train,read_id, COL_ORI_GT,    "GT Origin",     alpha=0.28)

    # Nerea's ORIs — separate orange highlight for missed-ORI plots
    if highlight_nerea_oris:
        add_annotation_spans(ax_sig, ori_gt_nerea, read_id, COL_NEREA_ORI,
                             "Nerea ORI (human)", alpha=0.45)

    # Predicted event spans (hatched on top)
    ann_read = annotations[annotations["read_id"] == read_id]
    hatch_map = {
        "left_fork":  ("//",  COL_LF_GT),
        "right_fork": ("\\\\", COL_RF_GT),
        "origin":     ("xx",  COL_ORI_PRED),
    }
    for etype, (hatch, col) in hatch_map.items():
        sub = ann_read[ann_read["event_type"] == etype]
        for row in sub.itertuples(index=False):
            ax_sig.axvspan(int(row.start), int(row.end),
                           color=col, alpha=0.20, hatch=hatch,
                           edgecolor=col, linewidth=0)

    ax_sig.set_ylabel("BrdU signal", fontsize=9)
    ax_sig.set_ylim(-0.05, 1.05)

    legend_handles = [
        mpatches.Patch(color=COL_LF_GT,   alpha=0.35, label="GT Left Fork"),
        mpatches.Patch(color=COL_RF_GT,   alpha=0.35, label="GT Right Fork"),
        mpatches.Patch(color=COL_ORI_GT,  alpha=0.35, label="GT Origin"),
    ]
    if highlight_nerea_oris:
        legend_handles.append(
            mpatches.Patch(color=COL_NEREA_ORI, alpha=0.55, label="Nerea ORI")
        )
    legend_handles += [
        mpatches.Patch(facecolor="white", edgecolor=COL_LF_GT,   hatch="//",   label="Pred LF"),
        mpatches.Patch(facecolor="white", edgecolor=COL_RF_GT,   hatch="\\\\", label="Pred RF"),
        mpatches.Patch(facecolor="white", edgecolor=COL_ORI_PRED,hatch="xx",   label="Pred ORI"),
    ]
    ax_sig.legend(handles=legend_handles, fontsize=7, loc="upper right",
                  ncol=4 if highlight_nerea_oris else 3)

    # ── Panel 1: probability tracks ───────────────────────────────────────────
    ax_prob = axes[1]
    ax_prob.fill_between(positions, read_preds["prob_left_fork"].values,
                         color=COL_LF_PRED, alpha=0.7, label="P(LF)")
    ax_prob.fill_between(positions, read_preds["prob_right_fork"].values,
                         color=COL_RF_PRED, alpha=0.7, label="P(RF)")
    ax_prob.fill_between(positions, read_preds["prob_origin"].values,
                         color=COL_ORI_PRED, alpha=0.85, label="P(ORI)")
    ax_prob.axhline(PROB_THR, color="black", linewidth=0.8, linestyle="--",
                    alpha=0.6, label=f"thr={PROB_THR}")

    ax_prob.set_ylabel("Probability", fontsize=9)
    ax_prob.set_ylim(0, 1.05)
    ax_prob.set_xlabel("Genomic position (bp)", fontsize=9)
    ax_prob.legend(fontsize=7, loc="upper right", ncol=4)

    # ── Title ─────────────────────────────────────────────────────────────────
    read_len_kb = (xy_df["end"].max() - xy_df["start"].min()) / 1000
    n_novel = len(ann_read[ann_read["event_type"] == "origin"])
    lf_pred = len(ann_read[ann_read["event_type"] == "left_fork"])
    rf_pred = len(ann_read[ann_read["event_type"] == "right_fork"])
    fig.suptitle(
        f"{title}  |  {read_id[:20]}…  ({read_len_kb:.0f} kb)\n"
        f"Predicted events — LF: {lf_pred}  RF: {rf_pred}  ORI: {n_novel}",
        fontsize=9, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── read selection ────────────────────────────────────────────────────────────

def select_novel_ori_reads(novel_oris: pd.DataFrame, n: int) -> list[str]:
    """Pick reads with the highest-BrdU novel ORIs, preferring longer ORIs."""
    # Reads with ≥1 novel ORI; rank by mean_brdu_signal desc, then length desc
    read_stats = (
        novel_oris.groupby("read_id")
        .agg(max_brdu=("mean_brdu_signal", "max"),
             max_len=("length", "max"),
             n_novel=("length", "count"))
        .reset_index()
        .sort_values(["max_brdu", "max_len"], ascending=False)
    )
    return read_stats["read_id"].head(n).tolist()


def select_missed_ori_reads(
    recovery: pd.DataFrame,
    annotations: pd.DataFrame,
    n: int,
) -> list[str]:
    """Pick reads with missed Nerea ORIs where the model *did* predict something
    (so we can inspect what went wrong), preferring larger missed ORIs."""
    missed = recovery[recovery["status"] == "missed"].copy()
    missed = missed[missed["length"] >= 3000]   # at least 3 kb — visible

    # Reads that have at least one prediction in the annotations file
    reads_with_preds = set(annotations["read_id"].unique())
    missed = missed[missed["read_id"].isin(reads_with_preds)]

    # Sort by size of missed ORI (largest = most conspicuous)
    missed = missed.sort_values("length", ascending=False)

    # One read per unique read_id (may have multiple missed ORIs)
    seen = set()
    selected = []
    for row in missed.itertuples(index=False):
        if row.read_id not in seen:
            seen.add(row.read_id)
            selected.append(row.read_id)
        if len(selected) >= n:
            break
    return selected


# ── predictions loader ────────────────────────────────────────────────────────

def load_predictions_for_read(base_dir, run_dirs, read_id):
    """Load XY data (same as predictions input to the CODEX evaluation)."""
    return load_xy(base_dir, run_dirs, read_id)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True,
                        help="forte_v5.0.yaml (data paths + preprocessing)")
    parser.add_argument("--predictions",   required=True,
                        help="TSV with per-window probabilities from evaluation "
                             "(chr/start/end/read_id/prob_* columns)")
    parser.add_argument("--annotations",   required=True,
                        help="Reannotation TSV (forte_v5.0_annotations.tsv)")
    parser.add_argument("--novel-oris",    required=True,
                        help="novel_oris_analysis.tsv")
    parser.add_argument("--recovery-tsv",  required=True,
                        help="nerea_oris_recovery_status.tsv")
    parser.add_argument("--nerea-ori-bed", required=True,
                        help="Nerea's ORIs combined cleaned BED")
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--n-novel",  type=int, default=5)
    parser.add_argument("--n-missed", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    base_dir = cfg["data"]["base_dir"]
    run_dirs = cfg["data"]["run_dirs"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load everything ───────────────────────────────────────────────────────
    print("Loading predictions (per-window)...")
    preds_all = pd.read_csv(args.predictions, sep="\t")
    print(f"  {len(preds_all):,} windows across {preds_all['read_id'].nunique():,} reads")

    print("Loading reannotation events...")
    annotations = pd.read_csv(args.annotations, sep="\t")
    print(f"  {len(annotations):,} events")

    print("Loading novel ORIs...")
    novel_oris = pd.read_csv(args.novel_oris, sep="\t")
    print(f"  {len(novel_oris):,} novel ORI events")

    print("Loading Nerea ORI recovery status...")
    recovery = pd.read_csv(args.recovery_tsv, sep="\t")
    print(f"  {len(recovery):,} Nerea ORIs  "
          f"({(recovery['status']=='recovered').sum():,} recovered, "
          f"{(recovery['status']=='missed').sum():,} missed)")

    print("Loading GT BED files...")
    lf_gt       = load_bed(cfg["data"].get("left_forks_bed"))
    rf_gt       = load_bed(cfg["data"].get("right_forks_bed"))
    ori_gt_train= load_bed(cfg["data"].get("ori_annotations_bed"))
    ori_gt_nerea= load_bed(args.nerea_ori_bed)
    print(f"  LF GT:         {len(lf_gt):,}")
    print(f"  RF GT:         {len(rf_gt):,}")
    print(f"  ORI (train):   {len(ori_gt_train):,}")
    print(f"  ORI (Nerea):   {len(ori_gt_nerea):,}")

    # ── Select reads ──────────────────────────────────────────────────────────
    novel_reads  = select_novel_ori_reads(novel_oris, args.n_novel)
    missed_reads = select_missed_ori_reads(recovery, annotations, args.n_missed)
    print(f"\nSelected {len(novel_reads)} novel-ORI reads")
    print(f"Selected {len(missed_reads)} missed-ORI reads")

    # ── Plot novel ORIs ───────────────────────────────────────────────────────
    print("\nPlotting novel ORI reads...")
    for i, rid in enumerate(novel_reads):
        xy = load_xy(base_dir, run_dirs, rid)
        if xy is None:
            print(f"  No XY data for {rid}, skipping")
            continue
        # novel ORI info for title supplement
        nov = novel_oris[novel_oris["read_id"] == rid]
        max_brdu = nov["mean_brdu_signal"].max()
        n_oris   = len(nov)
        plot_read(
            read_id=rid,
            xy_df=xy,
            preds_df=preds_all,
            annotations=annotations,
            lf_gt=lf_gt,
            rf_gt=rf_gt,
            ori_gt_train=ori_gt_train,
            ori_gt_nerea=ori_gt_nerea,
            title=f"Novel ORI #{i+1}  |  {n_oris} novel ORI(s), max BrdU={max_brdu:.3f}",
            out_path=out_dir / f"novel_ori_{i+1:02d}_{rid[:12]}.png",
            highlight_nerea_oris=False,
        )

    # ── Plot missed ORIs ──────────────────────────────────────────────────────
    print("\nPlotting missed Nerea ORI reads...")
    for i, rid in enumerate(missed_reads):
        xy = load_xy(base_dir, run_dirs, rid)
        if xy is None:
            print(f"  No XY data for {rid}, skipping")
            continue
        missed_here = recovery[(recovery["read_id"] == rid) & (recovery["status"] == "missed")]
        n_missed_here = len(missed_here)
        plot_read(
            read_id=rid,
            xy_df=xy,
            preds_df=preds_all,
            annotations=annotations,
            lf_gt=lf_gt,
            rf_gt=rf_gt,
            ori_gt_train=ori_gt_train,
            ori_gt_nerea=ori_gt_nerea,
            title=f"Missed Nerea ORI #{i+1}  |  {n_missed_here} missed ORI(s) on this read",
            out_path=out_dir / f"missed_nerea_ori_{i+1:02d}_{rid[:12]}.png",
            highlight_nerea_oris=True,
        )

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
