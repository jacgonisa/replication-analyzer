#!/usr/bin/env python
"""Plot before/after BED cleaning to visualise fork trimming and splitting.

Layout per read (5 panels, shared x-axis):
  1. BrdU signal trace
  2. BEFORE — 3 stacked lanes (LF / RF / ORI), one per class, no overlap confusion
  3. AFTER  — same layout, with removed/trimmed regions highlighted in orange

Usage:
  python CODEX/scripts/plot_bed_cleaning_check.py \
      --lf-orig  CODEX/results/forte_v4.3/pseudo_labels/combined_left_fork.bed \
      --rf-orig  CODEX/results/forte_v4.3/pseudo_labels/combined_right_fork.bed \
      --ori-orig data/.../ORIs_combined_cleaned.bed \
      --config   CODEX/configs/forte_v4.5.yaml \
      --output   CODEX/results/forte_v4.5/bed_cleaning_check/
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pickle
import yaml

BASE = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(CODEX_ROOT))

spec = importlib.util.spec_from_file_location(
    "clean", Path(__file__).parent / "clean_annotation_beds.py")
clean_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_mod)

COL = {
    "left_fork":  "#3a86ff",
    "right_fork": "#e63946",
    "origin":     "#2dc653",
}
COL_REMOVED = "#ff9f1c"   # orange — trimmed/removed regions
LANE_LABELS = ["Left Fork", "Right Fork", "Origin"]
LANE_KEYS   = ["left_fork", "right_fork", "origin"]


def load_xy_cache(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def pick_interesting_reads(lf_orig, lf_clean, ori_orig, ori_clean, n=8, focus="lf"):
    """Pick reads where the most cleaning happened.

    focus='lf'  — reads with most LF event changes (fork trimming)
    focus='ori' — reads with most ORI event changes (ORI merging)
    """
    if focus == "ori":
        orig_set  = set(zip(ori_orig["read_id"], ori_orig["start"], ori_orig["end"]))
        clean_set = set(zip(ori_clean["read_id"], ori_clean["start"], ori_clean["end"]))
        changed   = set(r for r, s, e in orig_set - clean_set)
        change_n  = ori_orig[ori_orig["read_id"].isin(changed)].groupby("read_id").size()
        clean_n   = ori_clean[ori_clean["read_id"].isin(changed)].groupby("read_id").size()
    else:
        orig_set  = set(zip(lf_orig["read_id"], lf_orig["start"], lf_orig["end"]))
        clean_set = set(zip(lf_clean["read_id"], lf_clean["start"], lf_clean["end"]))
        changed   = set(r for r, s, e in orig_set - clean_set)
        candidates = list(changed & set(ori_orig["read_id"]))
        change_n  = lf_orig[lf_orig["read_id"].isin(candidates)].groupby("read_id").size()
        clean_n   = lf_clean[lf_clean["read_id"].isin(candidates)].groupby("read_id").size()
    diff = (change_n - clean_n.reindex(change_n.index, fill_value=0)).sort_values(ascending=False)
    return diff.index[:n].tolist()


def draw_lane_track(ax, beds_orig, beds_clean, read_id, removed_intervals):
    """Draw 3 horizontal lanes (LF / RF / ORI) with before→after comparison.

    Each lane occupies 1/3 of the axis height.
    - Cleaned intervals: solid fill
    - Fork removed/trimmed regions: orange hatching
    - Original ORI intervals: thin outlines (shows what got merged into clean blocks)
    """
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Left Fork", "Right Fork", "Origin"], fontsize=8)
    ax.yaxis.set_tick_params(length=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.axhline(1, color="#cccccc", linewidth=0.6)
    ax.axhline(2, color="#cccccc", linewidth=0.6)

    for lane_idx, key in enumerate(LANE_KEYS):
        col  = COL[key]
        y_lo = lane_idx
        y_hi = lane_idx + 1
        pad  = 0.08

        # For ORI lane: draw original intervals as thin outlines first (shows pre-merge state)
        if key == "origin":
            sub_orig = beds_orig[key][beds_orig[key]["read_id"] == read_id]
            for row in sub_orig.itertuples(index=False):
                ax.fill_betweenx([y_lo + pad, y_hi - pad],
                                  int(row.start), int(row.end),
                                  facecolor="none", edgecolor=col,
                                  linewidth=1.2, linestyle="--", alpha=0.6)

        # Draw cleaned (surviving) intervals as solid
        sub_clean = beds_clean[key][beds_clean[key]["read_id"] == read_id]
        for row in sub_clean.itertuples(index=False):
            ax.fill_betweenx([y_lo + pad, y_hi - pad],
                              int(row.start), int(row.end),
                              color=col, alpha=0.75, linewidth=0)

        # For fork lanes: draw removed/trimmed regions in orange hatching
        for s, e in removed_intervals.get(key, []):
            ax.fill_betweenx([y_lo + pad, y_hi - pad], s, e,
                              color=COL_REMOVED, alpha=0.55, linewidth=0)
            ax.fill_betweenx([y_lo + pad, y_hi - pad], s, e,
                              facecolor="none", hatch="////",
                              edgecolor=COL_REMOVED, linewidth=0.5)


def compute_removed_intervals(orig_df, clean_df, read_id):
    """Return list of (start, end) intervals that were present in orig but removed in clean."""
    orig_sub  = orig_df[orig_df["read_id"] == read_id]
    clean_sub = clean_df[clean_df["read_id"] == read_id]

    # Build set of covered bases is expensive — use interval diff approach
    # Collect all orig intervals; subtract clean intervals → remainder is removed
    removed = []
    clean_ivals = sorted(zip(clean_sub["start"].astype(int), clean_sub["end"].astype(int)))

    for row in orig_sub.itertuples(index=False):
        remaining = [(int(row.start), int(row.end))]
        for cs, ce in clean_ivals:
            new_rem = []
            for rs, re in remaining:
                if ce <= rs or cs >= re:
                    new_rem.append((rs, re))
                else:
                    if rs < cs: new_rem.append((rs, cs))
                    if ce < re: new_rem.append((ce, re))
            remaining = new_rem
        removed.extend(remaining)
    return removed


def plot_read(read_id, xy_data,
              lf_orig, rf_orig, ori_orig,
              lf_clean, rf_clean, ori_clean,
              out_path):

    xy = xy_data[xy_data["read_id"] == read_id].sort_values("start")
    if xy.empty:
        print(f"  No XY for {read_id[:16]}, skipping")
        return

    # Precompute removed intervals per class
    removed = {
        "left_fork":  compute_removed_intervals(lf_orig,  lf_clean,  read_id),
        "right_fork": compute_removed_intervals(rf_orig,  rf_clean,  read_id),
        "origin":     compute_removed_intervals(ori_orig, ori_clean, read_id),
    }
    beds_orig  = {"left_fork": lf_orig,  "right_fork": rf_orig,  "origin": ori_orig}
    beds_clean = {"left_fork": lf_clean, "right_fork": rf_clean, "origin": ori_clean}

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.8]},
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("white")

    # ── Panel 1: BrdU signal ───────────────────────────────────────────────────
    ax_sig = axes[0]
    x = xy["start"].tolist() + [xy["end"].iloc[-1]]
    y = xy["signal"].tolist() + [xy["signal"].iloc[-1]]
    ax_sig.step(x, y, where="post", color="#222222", linewidth=1.1, zorder=4)
    ax_sig.fill_between(x, y, step="post", alpha=0.08, color="#888888", zorder=2)

    # Light shading for each cleaned class (context only)
    for key in LANE_KEYS:
        sub = beds_clean[key][beds_clean[key]["read_id"] == read_id]
        for row in sub.itertuples(index=False):
            ax_sig.axvspan(int(row.start), int(row.end),
                           color=COL[key], alpha=0.10, zorder=1)

    # Highlight removed/trimmed regions on signal panel too
    for key, ivals in removed.items():
        for s, e in ivals:
            ax_sig.axvspan(s, e, color=COL_REMOVED, alpha=0.18, zorder=3)

    ax_sig.set_ylabel("BrdU signal", fontsize=10)
    ax_sig.set_ylim(-0.05, 1.12)
    ax_sig.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_sig.tick_params(labelsize=8)
    ax_sig.spines[["top", "right"]].set_visible(False)

    sig_legend = [
        mpatches.Patch(color=COL[k], alpha=0.5,
                       label=k.replace("_", " ").title()) for k in LANE_KEYS
    ] + [mpatches.Patch(color=COL_REMOVED, alpha=0.6, label="Removed / trimmed")]
    ax_sig.legend(handles=sig_legend, fontsize=8, loc="upper right",
                  framealpha=0.85, ncol=2)

    # ── Panel 2: Annotation lanes (BEFORE overlaid, AFTER solid) ──────────────
    ax_ann = axes[1]
    draw_lane_track(ax_ann, beds_orig, beds_clean, read_id, removed)

    # Axis labels
    ax_ann.set_xlabel("Genomic position (bp)", fontsize=10)
    ax_ann.tick_params(axis="x", labelsize=8)
    ax_ann.set_ylabel("Annotation", fontsize=10)

    # Legend for annotation panel
    ann_legend = [
        mpatches.Patch(color=COL[k], alpha=0.75,
                       label=f"{k.replace('_',' ').title()} (kept)") for k in LANE_KEYS
    ] + [
        mpatches.Patch(facecolor=COL_REMOVED, alpha=0.55, hatch="////",
                       edgecolor=COL_REMOVED, label="Fork removed (ORI priority)"),
        mpatches.Patch(facecolor="none", edgecolor=COL["origin"],
                       linewidth=1.2, linestyle="--", label="ORI before merge (outline)"),
    ]
    ax_ann.legend(handles=ann_legend, fontsize=8, loc="upper right",
                  framealpha=0.85, ncol=2)

    # ── Title ─────────────────────────────────────────────────────────────────
    n_lf_orig   = (lf_orig["read_id"]   == read_id).sum()
    n_lf_clean  = (lf_clean["read_id"]  == read_id).sum()
    n_rf_orig   = (rf_orig["read_id"]   == read_id).sum()
    n_rf_clean  = (rf_clean["read_id"]  == read_id).sum()
    n_ori_orig  = (ori_orig["read_id"]  == read_id).sum()
    n_ori_clean = (ori_clean["read_id"] == read_id).sum()
    read_len_kb = (xy["end"].max() - xy["start"].min()) / 1000

    n_removed_lf  = len(removed["left_fork"])
    n_removed_rf  = len(removed["right_fork"])
    n_merged_ori  = n_ori_orig - n_ori_clean

    fig.suptitle(
        f"Read: {read_id[:24]}…    ({read_len_kb:.0f} kb)\n"
        f"LF: {n_lf_orig} → {n_lf_clean} events  ({n_removed_lf} trimmed regions)    "
        f"RF: {n_rf_orig} → {n_rf_clean} events  ({n_removed_rf} trimmed regions)    "
        f"ORI: {n_ori_orig} → {n_ori_clean} events  ({n_merged_ori} merged, priority kept)",
        fontsize=9, fontweight="bold", color="#333333",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lf-orig",  required=True)
    parser.add_argument("--rf-orig",  required=True)
    parser.add_argument("--ori-orig", required=True)
    parser.add_argument("--config",   required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--n-reads",  type=int, default=8)
    parser.add_argument("--focus",    choices=["lf", "ori"], default="lf",
                        help="Pick reads with most LF changes (lf) or ORI merging (ori)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print("Loading XY cache...")
    xy_data = load_xy_cache(cfg["data"]["xy_cache_path"])

    print("Loading original BEDs...")
    lf_orig  = clean_mod.load_bed4(args.lf_orig)
    rf_orig  = clean_mod.load_bed4(args.rf_orig)
    ori_orig = clean_mod.load_bed4(args.ori_orig)

    print("Cleaning BEDs...")
    lf_clean, rf_clean, ori_clean = clean_mod.clean_beds(lf_orig, rf_orig, ori_orig)

    print(f"Picking reads with most {args.focus.upper()} changes...")
    read_ids = pick_interesting_reads(lf_orig, lf_clean, ori_orig, ori_clean,
                                      n=args.n_reads, focus=args.focus)
    print(f"  Selected {len(read_ids)} reads")

    for i, rid in enumerate(read_ids):
        out_path = out_dir / f"cleaning_{i+1:02d}_{rid[:12]}.png"
        print(f"  [{i+1}/{len(read_ids)}] {rid[:20]}...")
        plot_read(rid, xy_data,
                  lf_orig, rf_orig, ori_orig,
                  lf_clean, rf_clean, ori_clean,
                  out_path)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
