#!/usr/bin/env python
"""Plot training history for FORTE v5.1+ models.

Handles both old metrics (val_masked_*) and new event-level metrics
(val_event_iou_*, val_event_prec_*, val_event_rec_*, val_event_f1_*,
val_event_f1_ori_lf_rf).

Panels:
  [0,0] Loss (train + val, log scale)
  [0,1] val_event_iou per class (LF/RF/ORI) + mean
  [1,0] val_event_f1 per class + ori_weighted composite
  [1,1] val_event_prec per class
  [2,0] val_event_rec per class
  [2,1] Learning rate (log scale)

Best epoch marked with a vertical dashed line (per run).

Usage (from /replication-analyzer/):
  /home/jg2070/miniforge3/envs/ONT/bin/python -u \\
      CODEX/scripts/plot_training_history_v2.py \\
      --logs   CODEX/results/forte_v5.1/train.log \\
               CODEX/results/forte_v5.0_onlyhuman/train.log \\
      --names  "v5.1" "v5.0-onlyhuman" \\
      --output CODEX/results/training_history_comparison.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── log parser ────────────────────────────────────────────────────────────────

def _findall_float(pattern: str, txt: str) -> list[float]:
    return [float(x) for x in re.findall(pattern, txt)]


def parse_log(log_path: str) -> dict:
    """Extract per-epoch metrics from a train_weak5_codex log.

    Handles binary/ANSI-escaped log files by reading as bytes and decoding
    with error replacement.
    """
    raw = Path(log_path).read_bytes()
    txt = raw.decode("utf-8", errors="replace")
    # Strip ANSI escape codes
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    # Collapse duplicate lines (two processes logged simultaneously in early runs)
    lines = txt.splitlines()
    seen, deduped = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    txt = "\n".join(deduped)

    # Per-epoch regex — captures the long step-summary line
    epoch_re = re.compile(
        r"Epoch (\d+)/\d+[^\n]*\n"          # "Epoch N/M\n"
        r".*?"                                # progress bars
        r" loss: ([\d.]+) .*?"
        r"val_loss: ([\d.]+) .*?"
        r"val_masked_f1_macro: ([\d.]+).*?"
        r"learning_rate: ([\d.e+\-]+)",
        re.DOTALL,
    )

    def _epoch_float(txt_epoch: str, key: str) -> float | None:
        m = re.search(key + r": ([\d.e+\-]+)", txt_epoch)
        return float(m.group(1)) if m else None

    epochs = []
    data: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_f1_macro": [], "lrs": [],
        "event_iou_lf": [], "event_iou_rf": [], "event_iou_ori": [], "event_iou_mean": [],
        "event_f1_lf":  [], "event_f1_rf":  [], "event_f1_ori":  [], "event_f1_wtd":  [],
        "event_prec_lf":[], "event_prec_rf":[], "event_prec_ori":[],
        "event_rec_lf": [], "event_rec_rf": [], "event_rec_ori": [],
    }

    best_epoch, best_val = None, -1.0
    # Determine which metric was used as monitor
    uses_wtd = "val_event_f1_ori_lf_rf" in txt

    for m in epoch_re.finditer(txt):
        ep   = int(m.group(1))
        block = m.group(0)

        epochs.append(ep)
        data["train_loss"].append(float(m.group(2)))
        data["val_loss"].append(float(m.group(3)))
        data["val_f1_macro"].append(float(m.group(4)))
        data["lrs"].append(float(m.group(5)))

        def g(key):
            return _epoch_float(block, key)

        iou_lf  = g("val_event_iou_lf")  or 0.0
        iou_rf  = g("val_event_iou_rf")  or 0.0
        iou_ori = g("val_event_iou_ori") or 0.0
        iou_mean= g("val_event_iou")     or 0.0
        f1_lf   = g("val_event_f1_lf")   or 0.0
        f1_rf   = g("val_event_f1_rf")   or 0.0
        f1_ori  = g("val_event_f1_ori")  or 0.0
        f1_wtd  = g("val_event_f1_ori_lf_rf") or (0.5*f1_ori + 0.25*f1_lf + 0.25*f1_rf)
        prec_lf = g("val_event_prec_lf") or 0.0
        prec_rf = g("val_event_prec_rf") or 0.0
        prec_ori= g("val_event_prec_ori")or 0.0
        rec_lf  = g("val_event_rec_lf")  or 0.0
        rec_rf  = g("val_event_rec_rf")  or 0.0
        rec_ori = g("val_event_rec_ori") or 0.0

        data["event_iou_lf"].append(iou_lf);   data["event_iou_rf"].append(iou_rf)
        data["event_iou_ori"].append(iou_ori);  data["event_iou_mean"].append(iou_mean)
        data["event_f1_lf"].append(f1_lf);     data["event_f1_rf"].append(f1_rf)
        data["event_f1_ori"].append(f1_ori);   data["event_f1_wtd"].append(f1_wtd)
        data["event_prec_lf"].append(prec_lf); data["event_prec_rf"].append(prec_rf)
        data["event_prec_ori"].append(prec_ori)
        data["event_rec_lf"].append(rec_lf);   data["event_rec_rf"].append(rec_rf)
        data["event_rec_ori"].append(rec_ori)

        monitor_val = f1_wtd if uses_wtd else iou_mean
        if monitor_val > best_val:
            best_val, best_epoch = monitor_val, ep

    data["epochs"] = epochs
    data["best_epoch"] = best_epoch
    data["best_val"] = best_val
    data["uses_wtd"] = uses_wtd
    return data


# ── plotting ──────────────────────────────────────────────────────────────────

def _smooth(values: list, window: int = 3) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=float)
    arr = np.array(values, dtype=float)
    out = np.convolve(arr, np.ones(window) / window, mode="same")
    for i in range(window // 2):
        out[i]    = arr[:i + window // 2 + 1].mean()
        out[-i-1] = arr[-(i + window // 2 + 1):].mean()
    return out


COL_LF  = "#1f77b4"   # blue
COL_RF  = "#d62728"   # red
COL_ORI = "#2ca02c"   # green
COL_MEAN= "#7f7f7f"   # grey
COL_WTD = "#ff7f0e"   # orange — ORI-weighted composite

RUN_COLORS = ["#1f77b4", "#9467bd", "#8c564b", "#e377c2"]  # one per run


def _plot_class_lines(ax, ep, d, key_lf, key_rf, key_ori, key_mean=None,
                      smooth=3, alpha_raw=0.20, lw=2.0):
    """Draw LF/RF/ORI (and optionally mean) lines on one axis."""
    for key, col, label in [
        (key_lf,  COL_LF,  "LF"),
        (key_rf,  COL_RF,  "RF"),
        (key_ori, COL_ORI, "ORI"),
    ]:
        vals = d.get(key, [])
        if not vals:
            continue
        sm = _smooth(vals, smooth)
        ax.plot(ep, vals, color=col, lw=0.6, alpha=alpha_raw)
        ax.plot(ep, sm,   color=col, lw=lw,  label=label)
    if key_mean and d.get(key_mean):
        vals = d[key_mean]
        sm = _smooth(vals, smooth)
        ax.plot(ep, vals, color=COL_MEAN, lw=0.6, alpha=alpha_raw)
        ax.plot(ep, sm,   color=COL_MEAN, lw=lw, linestyle="--", label="mean")


def plot_histories(runs: list[tuple[str, dict]], output_path: Path,
                   smooth_window: int = 3):
    plt.style.use("seaborn-v0_8-whitegrid")
    n_runs = len(runs)

    fig, axes = plt.subplots(3, 2, figsize=(15, 13),
                              gridspec_kw={"hspace": 0.50, "wspace": 0.32})

    # ── [0,0] Loss ────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for i, (name, d) in enumerate(runs):
        col = RUN_COLORS[i % len(RUN_COLORS)]
        ep = d["epochs"]
        if not ep:
            continue
        tl = _smooth(d["train_loss"], smooth_window)
        vl = _smooth(d["val_loss"],   smooth_window)
        ax.plot(ep, d["train_loss"], color=col, lw=0.6, alpha=0.2)
        ax.plot(ep, tl, color=col, lw=2.0, label=f"{name} train")
        ax.plot(ep, d["val_loss"],   color=col, lw=0.6, alpha=0.2, ls="--")
        ax.plot(ep, vl, color=col, lw=2.0, ls="--", label=f"{name} val")
        if d["best_epoch"] and d["best_epoch"] in ep:
            ax.axvline(d["best_epoch"], color=col, lw=1.0, ls=":", alpha=0.7)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_title("Loss (train — val dashed)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # ── [0,1] Event IoU per class ─────────────────────────────────────────────
    ax = axes[0, 1]
    for i, (name, d) in enumerate(runs):
        ep = d["epochs"]
        if not ep:
            continue
        prefix = f"{name} " if n_runs > 1 else ""
        for key, col, label in [
            ("event_iou_lf",  COL_LF,   f"{prefix}LF"),
            ("event_iou_rf",  COL_RF,   f"{prefix}RF"),
            ("event_iou_ori", COL_ORI,  f"{prefix}ORI"),
            ("event_iou_mean",COL_MEAN, f"{prefix}mean"),
        ]:
            vals = d.get(key, [])
            if not vals or all(v == 0 for v in vals):
                continue
            sm = _smooth(vals, smooth_window)
            ls = "--" if "mean" in label else "-"
            ax.plot(ep, vals, color=col, lw=0.6, alpha=0.2)
            ax.plot(ep, sm,   color=col, lw=2.0, ls=ls, label=label)
        if d["best_epoch"] and d["best_epoch"] in ep:
            ax.axvline(d["best_epoch"], color=RUN_COLORS[i], lw=1.0, ls=":", alpha=0.7,
                       label=f"{name} best ep{d['best_epoch']}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Event-level IoU per class", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # ── [1,0] Event F1 per class + ORI-weighted composite ────────────────────
    ax = axes[1, 0]
    for i, (name, d) in enumerate(runs):
        ep = d["epochs"]
        if not ep:
            continue
        prefix = f"{name} " if n_runs > 1 else ""
        for key, col, label, ls in [
            ("event_f1_lf",  COL_LF,   f"{prefix}LF",      "-"),
            ("event_f1_rf",  COL_RF,   f"{prefix}RF",      "-"),
            ("event_f1_ori", COL_ORI,  f"{prefix}ORI",     "-"),
            ("event_f1_wtd", COL_WTD,  f"{prefix}ORI-wtd", "--"),
        ]:
            vals = d.get(key, [])
            if not vals or all(v == 0 for v in vals):
                continue
            sm = _smooth(vals, smooth_window)
            ax.plot(ep, vals, color=col, lw=0.6, alpha=0.2)
            ax.plot(ep, sm,   color=col, lw=2.0, ls=ls, label=label)
        if d["best_epoch"] and d["best_epoch"] in ep:
            ax.axvline(d["best_epoch"], color=RUN_COLORS[i], lw=1.0, ls=":", alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Event-level F1 per class  (ORI-weighted = 0.5·ORI+0.25·LF+0.25·RF)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # ── [1,1] Event Precision per class ──────────────────────────────────────
    ax = axes[1, 1]
    for i, (name, d) in enumerate(runs):
        ep = d["epochs"]
        if not ep:
            continue
        prefix = f"{name} " if n_runs > 1 else ""
        for key, col, label in [
            ("event_prec_lf",  COL_LF,  f"{prefix}LF"),
            ("event_prec_rf",  COL_RF,  f"{prefix}RF"),
            ("event_prec_ori", COL_ORI, f"{prefix}ORI"),
        ]:
            vals = d.get(key, [])
            if not vals or all(v == 0 for v in vals):
                continue
            sm = _smooth(vals, smooth_window)
            ax.plot(ep, vals, color=col, lw=0.6, alpha=0.2)
            ax.plot(ep, sm,   color=col, lw=2.0, label=label)
        if d["best_epoch"] and d["best_epoch"] in ep:
            ax.axvline(d["best_epoch"], color=RUN_COLORS[i], lw=1.0, ls=":", alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Event-level Precision per class", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # ── [2,0] Event Recall per class ─────────────────────────────────────────
    ax = axes[2, 0]
    for i, (name, d) in enumerate(runs):
        ep = d["epochs"]
        if not ep:
            continue
        prefix = f"{name} " if n_runs > 1 else ""
        for key, col, label in [
            ("event_rec_lf",  COL_LF,  f"{prefix}LF"),
            ("event_rec_rf",  COL_RF,  f"{prefix}RF"),
            ("event_rec_ori", COL_ORI, f"{prefix}ORI"),
        ]:
            vals = d.get(key, [])
            if not vals or all(v == 0 for v in vals):
                continue
            sm = _smooth(vals, smooth_window)
            ax.plot(ep, vals, color=col, lw=0.6, alpha=0.2)
            ax.plot(ep, sm,   color=col, lw=2.0, label=label)
        if d["best_epoch"] and d["best_epoch"] in ep:
            ax.axvline(d["best_epoch"], color=RUN_COLORS[i], lw=1.0, ls=":", alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Event-level Recall per class", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    # ── [2,1] Learning rate ───────────────────────────────────────────────────
    ax = axes[2, 1]
    for i, (name, d) in enumerate(runs):
        ep = d["epochs"]
        lrs = d.get("lrs", [])
        if not ep or not lrs:
            continue
        col = RUN_COLORS[i % len(RUN_COLORS)]
        ax.plot(ep, lrs[:len(ep)], color=col, lw=2.0, label=name)
        if d["best_epoch"] and d["best_epoch"] in ep:
            ax.axvline(d["best_epoch"], color=col, lw=1.0, ls=":", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("Learning rate", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # ── shared formatting ─────────────────────────────────────────────────────
    for ax in axes.flat:
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # Footer
    parts = []
    for name, d in runs:
        monitor = "ori_wtd_f1" if d["uses_wtd"] else "event_iou"
        parts.append(f"{name}: best ep {d['best_epoch']}  {monitor}={d['best_val']:.4f}")
    fig.text(0.5, 0.005, "   |   ".join(parts),
             ha="center", fontsize=8.5, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4",
                       alpha=0.85, edgecolor="#cccc00"))

    fig.suptitle("FORTE — training histories  (faint = raw · bold = 3-epoch smooth · dotted = best epoch)",
                 fontsize=11, fontweight="bold", y=1.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs",   nargs="+", required=True)
    parser.add_argument("--names",  nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth", type=int, default=3)
    args = parser.parse_args()

    names = args.names or [Path(l).parent.name for l in args.logs]
    assert len(names) == len(args.logs), "Number of --names must match --logs"

    runs = []
    for name, log in zip(names, args.logs):
        print(f"Parsing {log}…")
        d = parse_log(log)
        monitor = "ori_wtd_f1" if d["uses_wtd"] else "event_iou"
        print(f"  {len(d['epochs'])} epochs  best=ep{d['best_epoch']} "
              f"({monitor}={d['best_val']:.4f})")
        runs.append((name, d))

    plot_histories(runs, Path(args.output), smooth_window=args.smooth)


if __name__ == "__main__":
    main()
