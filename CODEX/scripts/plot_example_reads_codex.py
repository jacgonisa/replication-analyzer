#!/usr/bin/env python
"""Plot a few reads with rectangular signal and ORI/L/R/TER annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.annotations import load_annotations_for_codex


def load_all_xy_data_plain(base_dir: str, run_dirs: list[str], read_ids: list[str] | None = None) -> pd.DataFrame:
    rows = []
    wanted = set(read_ids) if read_ids else None
    for run_dir in run_dirs:
        path = Path(base_dir) / run_dir
        if wanted:
            files = [path / f"plot_data_{read_id}.txt" for read_id in wanted]
        else:
            files = list(path.glob("plot_data_*.txt"))
        for file in files:
            if not file.exists():
                continue
            df = pd.read_csv(file, sep="\t", header=None, names=["chr", "start", "end", "signal"])
            df["read_id"] = file.stem.replace("plot_data_", "")
            df["run"] = run_dir
            rows.append(df)
    if not rows:
        raise ValueError("No XY files found.")
    combined = pd.concat(rows, ignore_index=True)
    combined["center"] = (combined["start"] + combined["end"]) / 2
    return combined


def choose_example_reads(left: pd.DataFrame, right: pd.DataFrame, ori: pd.DataFrame, ter: pd.DataFrame, n_reads: int):
    summary = pd.DataFrame({"read_id": pd.unique(pd.concat([left["read_id"], right["read_id"], ori["read_id"]], ignore_index=True))})
    summary["n_left"] = summary["read_id"].map(left.groupby("read_id").size()).fillna(0).astype(int)
    summary["n_right"] = summary["read_id"].map(right.groupby("read_id").size()).fillna(0).astype(int)
    summary["n_ori"] = summary["read_id"].map(ori.groupby("read_id").size()).fillna(0).astype(int)
    summary["n_ter"] = summary["read_id"].map(ter.groupby("read_id").size()).fillna(0).astype(int)
    summary["score"] = (
        (summary["n_left"] > 0).astype(int)
        + (summary["n_right"] > 0).astype(int)
        + 2 * (summary["n_ori"] > 0).astype(int)
        + 2 * (summary["n_ter"] > 0).astype(int)
    )
    summary = summary.sort_values(
        ["score", "n_ter", "n_ori", "n_left", "n_right"],
        ascending=False,
    )
    return summary.head(n_reads)["read_id"].tolist(), summary


def add_annotation_spans(ax, df: pd.DataFrame, read_id: str, color: str, label: str, y_min: float, y_max: float):
    subset = df[df["read_id"] == read_id]
    used_label = False
    for row in subset.itertuples(index=False):
        ax.axvspan(
            int(row.start),
            int(row.end),
            ymin=y_min,
            ymax=y_max,
            color=color,
            alpha=0.35,
            label=label if not used_label else None,
        )
        used_label = True


def plot_single_read(read_df, read_id, left, right, ori, ter, output_file):
    read_df = read_df.sort_values("start").reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, height_ratios=[3, 1.2])

    x = read_df["start"].to_numpy()
    y = read_df["signal"].to_numpy()
    axes[0].step(x, y, where="post", color="black", linewidth=1.4, label="BrdU signal")
    axes[0].fill_between(x, y, step="post", alpha=0.15, color="gray")
    axes[0].set_ylabel("Signal")
    axes[0].set_title(f"Read {read_id} - rectangular signal view")
    axes[0].grid(alpha=0.25)

    add_annotation_spans(axes[0], left, read_id, "#1f77b4", "Left fork", 0.0, 1.0)
    add_annotation_spans(axes[0], right, read_id, "#d62728", "Right fork", 0.0, 1.0)
    add_annotation_spans(axes[0], ori, read_id, "#2ca02c", "Origin", 0.0, 1.0)
    add_annotation_spans(axes[0], ter, read_id, "#9467bd", "Termination", 0.0, 1.0)
    axes[0].legend(loc="upper right", ncol=5, fontsize=9)

    add_annotation_spans(axes[1], left, read_id, "#1f77b4", "Left fork", 0.72, 0.98)
    add_annotation_spans(axes[1], right, read_id, "#d62728", "Right fork", 0.48, 0.72)
    add_annotation_spans(axes[1], ori, read_id, "#2ca02c", "Origin", 0.24, 0.48)
    add_annotation_spans(axes[1], ter, read_id, "#9467bd", "Termination", 0.02, 0.24)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([0.12, 0.36, 0.60, 0.85])
    axes[1].set_yticklabels(["TER", "ORI", "R", "L"])
    axes[1].set_xlabel("Genomic position")
    axes[1].grid(alpha=0.25, axis="x")

    plt.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot example reads with ORI/L/R/TER annotations")
    parser.add_argument("--config", required=True, help="CODEX config yaml")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--read-id", action="append", help="Specific read_id to plot; can be passed multiple times")
    parser.add_argument("--n-reads", type=int, default=2, help="Auto-select this many reads if --read-id not given")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    left, right, ori, ter = load_annotations_for_codex(config)
    if args.read_id:
        read_ids = args.read_id
        summary = None
    else:
        read_ids, summary = choose_example_reads(left, right, ori, ter, args.n_reads)
        if summary is not None:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_dir / "candidate_read_summary.tsv", sep="\t", index=False)

    xy_data = load_all_xy_data_plain(config["data"]["base_dir"], config["data"]["run_dirs"], read_ids=read_ids)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    for read_id in read_ids:
        read_df = xy_data[xy_data["read_id"] == read_id].copy()
        if len(read_df) == 0:
            continue
        plot_single_read(
            read_df=read_df,
            read_id=read_id,
            left=left,
            right=right,
            ori=ori,
            ter=ter,
            output_file=output_dir / f"{read_id}.png",
        )
        selected_rows.append(
            {
                "read_id": read_id,
                "n_windows": len(read_df),
                "n_left": int((left["read_id"] == read_id).sum()),
                "n_right": int((right["read_id"] == read_id).sum()),
                "n_ori": int((ori["read_id"] == read_id).sum()),
                "n_ter": int((ter["read_id"] == read_id).sum()),
            }
        )

    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_reads.tsv", sep="\t", index=False)
    print(f"Saved plots to: {output_dir}")
    if selected_rows:
        print(pd.DataFrame(selected_rows).to_string(index=False))


if __name__ == "__main__":
    main()
