#!/usr/bin/env python
"""Diagnose raw XY loading on a single run directory with chunk checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import traceback

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))


def load_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["chr", "start", "end", "signal"])
    df["read_id"] = path.stem.replace("plot_data_", "")
    df["center"] = (df["start"] + df["end"]) / 2
    df["length"] = df["end"] - df["start"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Diagnose XY loading for one run directory")
    parser.add_argument("--config", required=True, help="Path to CODEX config")
    parser.add_argument("--run-dir", required=True, help="Run directory relative to base_dir")
    parser.add_argument("--chunk-size", type=int, default=250, help="Files per checkpoint chunk")
    parser.add_argument("--start-index", type=int, default=0, help="Start file index")
    parser.add_argument("--max-files", type=int, default=0, help="Optional max files to process")
    parser.add_argument("--output-dir", required=True, help="Output directory for diagnostics")
    args = parser.parse_args()

    print(f"[{datetime.now().isoformat(timespec='seconds')}] diagnose_xy_run_codex.py starting", flush=True)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base_dir = Path(config["data"]["base_dir"])
    run_path = base_dir / args.run_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(run_path.glob("plot_data_*.txt"))
    print(f"Run path: {run_path}")
    print(f"Files found: {len(files):,}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Start index: {args.start_index}")

    end_index = len(files) if args.max_files <= 0 else min(len(files), args.start_index + args.max_files)
    files = files[args.start_index:end_index]
    print(f"Files to process in this diagnostic pass: {len(files):,}")

    manifest_rows = []
    for chunk_start in range(0, len(files), args.chunk_size):
        chunk = files[chunk_start:chunk_start + args.chunk_size]
        chunk_abs_start = args.start_index + chunk_start
        chunk_abs_end = chunk_abs_start + len(chunk) - 1
        print(f"\nProcessing chunk {chunk_abs_start}-{chunk_abs_end}")

        rows = []
        for idx, file in enumerate(chunk, start=chunk_abs_start):
            try:
                df = load_one_file(file)
                df["run"] = args.run_dir
                rows.append(df)
            except Exception as exc:
                print(f"FAILED on file index {idx}: {file}")
                print(f"Exception: {exc}")
                print(traceback.format_exc())
                fail_path = output_dir / f"failure_{idx}.txt"
                fail_path.write_text(f"{file}\n\n{traceback.format_exc()}", encoding="utf-8")
                raise

            if (idx - chunk_abs_start + 1) % 25 == 0 or idx == chunk_abs_end:
                print(f"  Loaded file index {idx}: {file.name}")

        chunk_df = pd.concat(rows, ignore_index=True)
        chunk_path = output_dir / f"chunk_{chunk_abs_start:06d}_{chunk_abs_end:06d}.pkl"
        chunk_df.to_pickle(chunk_path)
        manifest_rows.append(
            {
                "chunk_start": chunk_abs_start,
                "chunk_end": chunk_abs_end,
                "n_rows": len(chunk_df),
                "path": str(chunk_path),
            }
        )
        pd.DataFrame(manifest_rows).to_csv(output_dir / "chunk_manifest.tsv", sep="\t", index=False)
        print(f"  Saved chunk: {chunk_path}")
        print(f"  Chunk rows: {len(chunk_df):,}")

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Diagnostic pass complete", flush=True)
    print(f"Chunks saved under: {output_dir}")


if __name__ == "__main__":
    main()
