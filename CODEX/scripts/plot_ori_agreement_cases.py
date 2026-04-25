#!/usr/bin/env python
"""Plot example reads for the two ORI disagreement categories:

  Category A — AI-only on Nerea reads: AI predicts an ORI that Nerea didn't annotate
               on a read where Nerea DID annotate other events. Could be real ORI or FP.

  Category B — Nerea-only (missed by AI): Nerea annotated an ORI the model missed.
               Mostly small ORIs (<1kb). Shows what the prob track looks like there.

Also prints a stats summary table for both categories.

Usage (from replication-analyzer/ root):
  python CODEX/scripts/plot_ori_agreement_cases.py \
      --config   CODEX/configs/forte_v5.1.yaml \
      --segments CODEX/results/forte_v5.1/reannotation/reannotated_segments.tsv \
      --events   CODEX/results/forte_v5.1/reannotation/reannotated_events.tsv \
      --output   CODEX/results/forte_v5.1/reannotation/ori_agreement_plots \
      --n-plots  6
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NEREA_ORI_BED  = ROOT / "data/case_study_jan2026/combined/annotations/ORIs_combined_cleaned.bed"
NEREA_LF_BED   = ROOT / "data/case_study_jan2026/combined/annotations/leftForks_ALL_combined.bed"
NEREA_RF_BED   = ROOT / "data/case_study_jan2026/combined/annotations/rightForks_ALL_combined.bed"

COL_NEREA  = "#e67e22"   # orange — Nerea ORI
COL_AI     = "#55efc4"   # teal   — AI ORI
COL_LF     = "#1f77b4"   # blue
COL_RF     = "#d62728"   # red
COL_PROB_ORI = "#00b894"
COL_PROB_LF  = "#74b9ff"
COL_PROB_RF  = "#ff7675"

IOU_THR = 0.1


def load_bed4(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                       names=["chr", "start", "end", "read_id"])


def build_idx(df):
    idx = defaultdict(list)
    for i, r in df.iterrows():
        idx[r["read_id"]].append((int(r["start"]), int(r["end"]), i))
    return idx


def compute_iou(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def match_events(ai_df, nerea_df, iou_thr=IOU_THR):
    nerea_idx = build_idx(nerea_df)
    ai_matched, nerea_matched = set(), set()
    for i, r in ai_df.iterrows():
        for ns, ne, ni in nerea_idx.get(r["read_id"], []):
            if compute_iou(int(r["start"]), int(r["end"]), ns, ne) >= iou_thr:
                ai_matched.add(i)
                nerea_matched.add(ni)
    return ai_matched, nerea_matched


def load_xy(config, read_id):
    rows = []
    for run_dir in config["data"]["run_dirs"]:
        f = Path(config["data"]["base_dir"]) / run_dir / f"plot_data_{read_id}.txt"
        if f.exists():
            df = pd.read_csv(f, sep="\t", header=None,
                             names=["chr", "start", "end", "signal"])
            rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True).sort_values("start").reset_index(drop=True)


def axspan(ax, bed_df, read_id, color, label, alpha=0.30, hatch=None):
    sub = bed_df[bed_df["read_id"] == read_id] if "read_id" in bed_df.columns else pd.DataFrame()
    used = False
    for row in sub.itertuples(index=False):
        kw = dict(color=color, alpha=alpha, label=label if not used else "_")
        if hatch:
            kw.update(hatch=hatch, edgecolor=color, facecolor="none" if alpha == 0 else color,
                      linewidth=0)
        ax.axvspan(int(row.start), int(row.end), **kw)
        used = True


def plot_read(read_id, xy_df, seg_df, events_df, nerea_ori, nerea_lf, nerea_rf,
              title, out_path, highlight_ai_ori=None):
    """2-panel plot: BrdU signal + annotation spans | probability tracks."""
    read_segs = seg_df[seg_df["read_id"] == read_id].sort_values("start")
    if read_segs.empty or xy_df is None:
        print(f"  Skipping {read_id[:16]}: no segment data")
        return

    positions = ((read_segs["start"] + read_segs["end"]) / 2).values
    fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5]})

    # Panel 0: BrdU signal
    ax = axes[0]
    x = xy_df["start"].tolist() + [xy_df["end"].iloc[-1]]
    y = xy_df["signal"].tolist() + [xy_df["signal"].iloc[-1]]
    ax.step(x, y, where="post", color="black", lw=1.2, zorder=3)
    ax.fill_between(x, y, step="post", alpha=0.1, color="gray", zorder=2)

    # Nerea annotations
    axspan(ax, nerea_lf,  read_id, COL_LF,    "Nerea LF",  alpha=0.22)
    axspan(ax, nerea_rf,  read_id, COL_RF,    "Nerea RF",  alpha=0.22)
    axspan(ax, nerea_ori, read_id, COL_NEREA, "Nerea ORI", alpha=0.35)

    # AI predicted events (hatched)
    read_events = events_df[events_df["read_id"] == read_id]
    for etype, hatch, col in [("left_fork", "//", COL_LF),
                               ("right_fork", "\\\\", COL_RF),
                               ("origin", "xx", COL_AI)]:
        sub = read_events[read_events["event_type"] == etype]
        used = False
        for row in sub.itertuples(index=False):
            ax.axvspan(int(row.start), int(row.end),
                       facecolor=col, alpha=0.18, hatch=hatch,
                       edgecolor=col, linewidth=0,
                       label=f"AI {etype.replace('_',' ').title()}" if not used else "_")
            used = True

    # Highlight specific AI-only ORIs with bold border
    if highlight_ai_ori is not None:
        for _, row in highlight_ai_ori[highlight_ai_ori["read_id"] == read_id].iterrows():
            ax.axvspan(int(row["start"]), int(row["end"]),
                       facecolor="none", edgecolor="red", linewidth=2,
                       label="AI-only ORI (no Nerea match)", linestyle="--")

    ax.set_ylabel("BrdU signal", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    handles = [
        mpatches.Patch(color=COL_NEREA, alpha=0.45, label="Nerea ORI"),
        mpatches.Patch(color=COL_LF,    alpha=0.30, label="Nerea LF"),
        mpatches.Patch(color=COL_RF,    alpha=0.30, label="Nerea RF"),
        mpatches.Patch(facecolor="white", edgecolor=COL_AI, hatch="xx", label="AI ORI pred"),
        mpatches.Patch(facecolor="white", edgecolor=COL_LF, hatch="//", label="AI LF pred"),
        mpatches.Patch(facecolor="white", edgecolor=COL_RF, hatch="\\\\", label="AI RF pred"),
    ]
    if highlight_ai_ori is not None and (highlight_ai_ori["read_id"] == read_id).any():
        handles.append(mpatches.Patch(facecolor="none", edgecolor="red",
                                      linewidth=2, label="AI-only ORI (★)"))
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=4)

    # Panel 1: probability tracks
    ax2 = axes[1]
    ax2.fill_between(positions, read_segs["prob_left_fork"].values,
                     color=COL_PROB_LF, alpha=0.7, label="P(LF)")
    ax2.fill_between(positions, read_segs["prob_right_fork"].values,
                     color=COL_PROB_RF, alpha=0.7, label="P(RF)")
    ax2.fill_between(positions, read_segs["prob_origin"].values,
                     color=COL_PROB_ORI, alpha=0.85, label="P(ORI)")
    ax2.axhline(0.3, color="black", lw=0.8, ls="--", alpha=0.6, label="thr=0.3")
    ax2.set_ylabel("Probability", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Genomic position (bp)", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right", ncol=4)

    n_ai_ori = (read_events["event_type"] == "origin").sum()
    n_nerea_ori = (nerea_ori["read_id"] == read_id).sum()
    rlen_kb = (xy_df["end"].max() - xy_df["start"].min()) / 1000
    fig.suptitle(
        f"{title}  |  {read_id[:24]}…  ({rlen_kb:.0f} kb)\n"
        f"Nerea ORIs: {n_nerea_ori}   AI ORIs: {n_ai_ori}",
        fontsize=9, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def print_stats(label, ai_sub, nerea_sub=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if len(ai_sub):
        lens = ai_sub["length"]
        probs = ai_sub["mean_prob"]
        brdu = ai_sub.get("mean_brdu_signal", pd.Series(dtype=float))
        print(f"  AI events  n={len(ai_sub):,}")
        print(f"    size (bp): median={lens.median():.0f}  p25={lens.quantile(.25):.0f}  "
              f"p75={lens.quantile(.75):.0f}  p90={lens.quantile(.90):.0f}")
        print(f"    mean_prob: median={probs.median():.3f}  p25={probs.quantile(.25):.3f}  "
              f"p75={probs.quantile(.75):.3f}")
        if len(brdu.dropna()):
            print(f"    mean_brdu: median={brdu.median():.3f}  p25={brdu.quantile(.25):.3f}  "
                  f"p75={brdu.quantile(.75):.3f}")
        print(f"    size bins: <500bp={( lens<500).sum()}  500-1kb={(( lens>=500)&(lens<1000)).sum()}  "
              f"1-2kb={(( lens>=1000)&(lens<2000)).sum()}  2-5kb={(( lens>=2000)&(lens<5000)).sum()}  "
              f">5kb={( lens>=5000).sum()}")
    if nerea_sub is not None and len(nerea_sub):
        lens = nerea_sub["end"] - nerea_sub["start"]
        print(f"  Nerea events  n={len(nerea_sub):,}")
        print(f"    size (bp): median={lens.median():.0f}  p25={lens.quantile(.25):.0f}  "
              f"p75={lens.quantile(.75):.0f}  p90={lens.quantile(.90):.0f}")
        print(f"    size bins: <500bp={(lens<500).sum()}  500-1kb={((lens>=500)&(lens<1000)).sum()}  "
              f"1-2kb={((lens>=1000)&(lens<2000)).sum()}  >2kb={(lens>=2000).sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--segments", required=True,
                        help="reannotated_segments.tsv (per-window probabilities)")
    parser.add_argument("--events",   required=True,
                        help="reannotated_events.tsv (called events)")
    parser.add_argument("--output",   required=True)
    parser.add_argument("--n-plots",  type=int, default=6)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading events and Nerea BEDs…")
    events  = pd.read_csv(args.events,   sep="\t")
    ai_ori  = events[events["event_type"] == "origin"].copy()

    nerea_ori = load_bed4(NEREA_ORI_BED)
    nerea_lf  = load_bed4(NEREA_LF_BED)
    nerea_rf  = load_bed4(NEREA_RF_BED)
    nerea_reads = set(nerea_ori["read_id"])

    print("Matching AI ORIs to Nerea ORIs…")
    ai_matched_idx, nerea_matched_idx = match_events(ai_ori, nerea_ori)

    ai_only    = ai_ori[~ai_ori.index.isin(ai_matched_idx)].copy()
    nerea_only = nerea_ori[~nerea_ori.index.isin(nerea_matched_idx)].copy()

    # Split AI-only by whether the read has any Nerea annotation
    ai_only_on_nerea = ai_only[ai_only["read_id"].isin(nerea_reads)]
    ai_only_novel    = ai_only[~ai_only["read_id"].isin(nerea_reads)]

    # ── Stats ─────────────────────────────────────────────────────────────────
    print_stats("Category A — AI-only ORIs on Nerea-annotated reads (potential FPs or real missed ORIs)",
                ai_only_on_nerea)
    print_stats("Category B — AI-only ORIs on unannotated reads (expected extrapolation)",
                ai_only_novel)
    print_stats("Category C — Nerea-only ORIs (missed by AI)",
                pd.DataFrame(), nerea_only)

    print(f"\nSummary table:")
    print(f"  {'Category':<50} {'N':>7}  {'Median size (bp)':>16}")
    rows = [
        ("AI ORIs recovered by Nerea (both)",          len(ai_matched_idx),
         ai_ori[ai_ori.index.isin(ai_matched_idx)]["length"].median()),
        ("AI-only on Nerea reads (potential FP/novel)", len(ai_only_on_nerea),
         ai_only_on_nerea["length"].median()),
        ("AI-only on unannotated reads",                len(ai_only_novel),
         ai_only_novel["length"].median()),
        ("Nerea-only (missed by AI)",                   len(nerea_only),
         (nerea_only["end"]-nerea_only["start"]).median()),
    ]
    for label, n, med in rows:
        print(f"  {label:<50} {n:>7,}  {med:>16.0f}")

    # ── Load segments lazily (read-by-read to avoid 428MB in RAM) ─────────────
    print(f"\nLoading segments TSV for plotting (this may take a moment)…")
    seg_df = pd.read_csv(args.segments, sep="\t",
                         usecols=["chr", "start", "end", "read_id",
                                  "prob_origin", "prob_left_fork", "prob_right_fork"])

    # ── Category A plots: AI-only on Nerea reads ─────────────────────────────
    # Pick reads where the AI-only ORI is large (most interesting) and high confidence
    print(f"\nSelecting Category A examples…")
    cand_a = (ai_only_on_nerea
              .sort_values(["length", "mean_prob"], ascending=[False, False])
              .drop_duplicates("read_id")
              .head(args.n_plots))

    for i, (_, row) in enumerate(cand_a.iterrows()):
        read_id = row["read_id"]
        xy = load_xy(config, read_id)
        ai_only_this = ai_only_on_nerea[ai_only_on_nerea["read_id"] == read_id]
        plot_read(
            read_id=read_id,
            xy_df=xy,
            seg_df=seg_df,
            events_df=events,
            nerea_ori=nerea_ori,
            nerea_lf=nerea_lf,
            nerea_rf=nerea_rf,
            title=f"Cat A – AI-only ORI on Nerea read  (AI-only len={row['length']:.0f}bp  prob={row['mean_prob']:.2f}  brdu={row.get('mean_brdu_signal', float('nan')):.2f})",
            out_path=out_dir / f"catA_ai_only_{i+1:02d}.png",
            highlight_ai_ori=ai_only_this,
        )

    # ── Category B plots: Nerea-only (missed by AI) ───────────────────────────
    # Prefer reads with larger missed ORIs (more visible) but also include small ones
    print(f"\nSelecting Category B examples (Nerea-only / missed)…")
    nerea_only_lens = nerea_only["end"] - nerea_only["start"]
    # Mix of sizes: half large (>1kb), half small (<1kb)
    large_missed = nerea_only[nerea_only_lens > 1000].drop_duplicates("read_id").head(args.n_plots // 2)
    small_missed  = nerea_only[nerea_only_lens <= 1000].drop_duplicates("read_id").head(args.n_plots - len(large_missed))
    cand_b = pd.concat([large_missed, small_missed]).drop_duplicates("read_id")

    for i, (_, row) in enumerate(cand_b.iterrows()):
        read_id = row["read_id"]
        xy = load_xy(config, read_id)
        ori_len = int(row["end"]) - int(row["start"])
        plot_read(
            read_id=read_id,
            xy_df=xy,
            seg_df=seg_df,
            events_df=events,
            nerea_ori=nerea_ori,
            nerea_lf=nerea_lf,
            nerea_rf=nerea_rf,
            title=f"Cat B – Nerea ORI missed by AI  (Nerea ORI len={ori_len}bp)",
            out_path=out_dir / f"catB_nerea_missed_{i+1:02d}.png",
            highlight_ai_ori=None,
        )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
