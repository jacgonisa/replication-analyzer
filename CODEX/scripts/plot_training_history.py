#!/usr/bin/env python
"""Parse training log files and plot loss + metric curves for one or more runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path: str) -> dict:
    """Extract per-epoch metrics from a train_weak5_codex log."""
    with open(log_path) as f:
        txt = f.read()

    epochs, train_loss, val_loss = [], [], []
    val_f1, val_prec_left, val_rec_left = [], [], []
    val_prec_right, val_rec_right = [], []
    val_prec_ori, val_rec_ori = [], []
    best_epoch, best_f1 = None, 0.0

    for m in re.finditer(
        r"Epoch (\d+)/\d+.*?"
        r" loss: ([\d.]+) .*?"
        r"val_loss: ([\d.]+) .*?"
        r"val_masked_f1_macro: ([\d.]+).*?"
        r"val_masked_precision_left_fork: ([\d.]+).*?"
        r"val_masked_precision_origin: ([\d.]+).*?"
        r"val_masked_precision_right_fork: ([\d.]+).*?"
        r"val_masked_recall_left_fork: ([\d.]+).*?"
        r"val_masked_recall_origin: ([\d.]+).*?"
        r"val_masked_recall_right_fork: ([\d.]+)",
        txt, re.DOTALL,
    ):
        ep   = int(m.group(1))
        tl   = float(m.group(2))
        vl   = float(m.group(3))
        vf1  = float(m.group(4))
        vpl  = float(m.group(5))
        vpo  = float(m.group(6))
        vpr  = float(m.group(7))
        vrl  = float(m.group(8))
        vro  = float(m.group(9))
        vrr  = float(m.group(10))

        epochs.append(ep)
        train_loss.append(tl); val_loss.append(vl)
        val_f1.append(vf1)
        val_prec_left.append(vpl); val_rec_left.append(vrl)
        val_prec_right.append(vpr); val_rec_right.append(vrr)
        val_prec_ori.append(vpo); val_rec_ori.append(vro)

        if vf1 > best_f1:
            best_f1, best_epoch = vf1, ep

    # also get LR
    lrs = [float(x) for x in re.findall(r"learning_rate: ([\d.e+-]+)", txt)]
    # one LR per epoch (in step output); sample last one per epoch block
    # simpler: just collect all and downsample
    if len(lrs) > len(epochs):
        step = len(lrs) // len(epochs)
        lrs = lrs[step - 1::step][:len(epochs)]

    return dict(
        epochs=epochs, train_loss=train_loss, val_loss=val_loss,
        val_f1=val_f1,
        val_prec_left=val_prec_left, val_rec_left=val_rec_left,
        val_prec_right=val_prec_right, val_rec_right=val_rec_right,
        val_prec_ori=val_prec_ori, val_rec_ori=val_rec_ori,
        lrs=lrs[:len(epochs)],
        best_epoch=best_epoch, best_f1=best_f1,
    )


def _smooth(values: list, window: int = 3) -> np.ndarray:
    """Centered moving average, padded at edges."""
    if len(values) < window:
        return np.array(values)
    arr = np.array(values, dtype=float)
    out = np.convolve(arr, np.ones(window) / window, mode="same")
    # Fix edge bias from zero-padding
    for i in range(window // 2):
        out[i]    = arr[:i + window // 2 + 1].mean()
        out[-i-1] = arr[-(i + window // 2 + 1):].mean()
    return out


def plot_histories(runs: list[tuple[str, dict]], output_path: Path,
                   smooth_window: int = 3):
    """Plot training curves for all runs on shared axes."""
    import matplotlib.ticker as mticker

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 13),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.30})

    panels = [
        (axes[0][0], "Loss (train vs val)",   "train_loss",    "val_loss",       "train", "val"),
        (axes[0][1], "Val macro-F1",          "val_f1",        None,             None,    None),
        (axes[1][0], "Left-fork  recall / precision", "val_rec_left",  "val_prec_left",  "recall", "precision"),
        (axes[1][1], "Right-fork recall / precision", "val_rec_right", "val_prec_right", "recall", "precision"),
        (axes[2][0], "Origin     recall / precision", "val_rec_ori",   "val_prec_ori",   "recall", "precision"),
        (axes[2][1], "Learning rate",         "lrs",           None,             None,    None),
    ]

    for ax, title, key1, key2, label1, label2 in panels:
        for i, (name, d) in enumerate(runs):
            col = colors[i % len(colors)]
            ep  = d["epochs"]
            raw1 = d.get(key1, [])
            if not ep or not raw1:
                continue

            sm1 = _smooth(raw1, smooth_window)
            # raw: faint, thin; smooth: bold
            ax.plot(ep, raw1, color=col, linewidth=0.7, alpha=0.25)
            lbl = f"{name}" if label1 is None else f"{name} {label1}"
            ax.plot(ep, sm1, color=col, linewidth=2.0, label=lbl)

            if key2 and d.get(key2):
                raw2 = d[key2]
                sm2  = _smooth(raw2, smooth_window)
                ax.plot(ep, raw2, color=col, linewidth=0.7, alpha=0.25, linestyle="--")
                lbl2 = f"{name}" if label2 is None else f"{name} {label2}"
                ax.plot(ep, sm2, color=col, linewidth=2.0, linestyle="--", label=lbl2)

            # star at best epoch on the F1 panel
            if title == "Val macro-F1" and d["best_epoch"] and d["best_epoch"] in ep:
                bi = ep.index(d["best_epoch"])
                ax.scatter([d["best_epoch"]], [sm1[bi]], color=col,
                           marker="*", s=180, zorder=6,
                           label=f"{name} best ep{d['best_epoch']} ({d['best_f1']:.3f})")

        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        if "Loss" in title:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        if "Learning rate" in title:
            ax.set_yscale("log")
        if any(k in title for k in ["recall", "precision", "F1"]):
            ax.set_ylim(0, 1.05)

        leg = ax.legend(fontsize=7, ncol=1, framealpha=0.8,
                        loc="lower right" if "F1" in title or "recall" in title else "best")

    # Summary footer
    summary = "   |   ".join(
        f"{name}: best ep {d['best_epoch']}  F1={d['best_f1']:.3f}"
        for name, d in runs
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=8.5, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.85,
                       edgecolor="#cccc00"))

    fig.suptitle("FORTE — training histories  (faint = raw · bold = 3-epoch smooth)",
                 fontsize=12, fontweight="bold", y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True,
                        help="Log files to parse")
    parser.add_argument("--names", nargs="+",
                        help="Display names (same order as --logs)")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    names = args.names or [Path(l).stem for l in args.logs]
    assert len(names) == len(args.logs)

    runs = []
    for name, log in zip(names, args.logs):
        print(f"Parsing {log}…")
        d = parse_log(log)
        print(f"  {len(d['epochs'])} epochs  best={d['best_epoch']} (f1={d['best_f1']:.3f})")
        runs.append((name, d))

    plot_histories(runs, Path(args.output))


if __name__ == "__main__":
    main()
