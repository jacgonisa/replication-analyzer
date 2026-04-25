"""Cached XY loading for CODEX runs."""

from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd
import yaml


def _load_xy_data_single_plain(plot_data_file: Path) -> pd.DataFrame:
    data = pd.read_csv(
        plot_data_file,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "signal"],
    )
    data["read_id"] = plot_data_file.stem.replace("plot_data_", "")
    data["center"] = (data["start"] + data["end"]) / 2
    data["length"] = data["end"] - data["start"]
    return data


def _run_cache_path(cache_path: Path, run_dir: str) -> Path:
    safe_name = run_dir.replace("/", "__")
    return cache_path.parent / f"{cache_path.stem}.{safe_name}.pkl"


def _run_chunk_dir(cache_path: Path, run_dir: str) -> Path:
    safe_name = run_dir.replace("/", "__")
    return cache_path.parent / f"{cache_path.stem}.{safe_name}_chunks"


def build_xy_cache_table(
    base_dir: str,
    run_dirs: list[str] | None = None,
    log_every: int = 500,
    cache_path: Path | None = None,
    chunk_size: int = 250,
) -> pd.DataFrame:
    """
    Build a combined XY table with explicit newline logging.

    This avoids the carriage-return progress style used in the original loader,
    which is fragile when running long jobs through PTYs and log files.
    """
    base_path = Path(base_dir)
    if run_dirs is None:
        run_dirs = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_xy")]

    per_run_tables = []
    for run_dir in run_dirs:
        dir_path = base_path / run_dir
        if not dir_path.exists():
            print(f"  Warning: directory not found: {dir_path}")
            continue

        run_cache_path = _run_cache_path(cache_path, run_dir) if cache_path else None
        run_chunk_dir = _run_chunk_dir(cache_path, run_dir) if cache_path else None
        if run_cache_path is not None and run_cache_path.exists():
            print(f"Loading XY run from checkpoint: {run_cache_path}")
            run_df = pd.read_pickle(run_cache_path)
            per_run_tables.append(run_df)
            print(f"  Reused rows: {len(run_df):,}")
            continue

        plot_files = sorted(dir_path.glob("plot_data_*.txt"))
        print(f"Loading XY run: {dir_path}")
        print(f"  Files found: {len(plot_files):,}")
        if run_chunk_dir is not None:
            run_chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        total_rows = 0
        for chunk_start in range(0, len(plot_files), chunk_size):
            chunk_end = min(len(plot_files), chunk_start + chunk_size)
            chunk = plot_files[chunk_start:chunk_end]
            chunk_path = None
            if run_chunk_dir is not None:
                chunk_path = run_chunk_dir / f"chunk_{chunk_start:06d}_{chunk_end - 1:06d}.pkl"
            if chunk_path is not None and chunk_path.exists():
                chunk_paths.append(chunk_path)
                chunk_rows = len(pd.read_pickle(chunk_path))
                total_rows += chunk_rows
                print(f"  Reused chunk {chunk_start:,}-{chunk_end - 1:,}: {chunk_rows:,} rows")
                continue

            rows = []
            for idx, file in enumerate(chunk, start=chunk_start):
                try:
                    data = _load_xy_data_single_plain(file)
                    data["run"] = run_dir
                    rows.append(data)
                except Exception as exc:
                    print(f"  Error loading {file}: {exc}")
                    print(traceback.format_exc())
                    raise
                if (idx + 1) % 25 == 0 or (idx + 1) == chunk_end:
                    print(f"    Loaded file {idx + 1:,}/{len(plot_files):,} from {run_dir}")

            if not rows:
                continue

            chunk_df = pd.concat(rows, ignore_index=True)
            total_rows += len(chunk_df)
            if chunk_path is not None:
                chunk_df.to_pickle(chunk_path)
                chunk_paths.append(chunk_path)
                print(
                    f"  Saved chunk {chunk_start:,}-{chunk_end - 1:,} "
                    f"to {chunk_path.name} ({len(chunk_df):,} rows)"
                )
            else:
                temp_path = Path(f"/tmp/{run_dir.replace('/', '__')}_{chunk_start}_{chunk_end}.pkl")
                chunk_df.to_pickle(temp_path)
                chunk_paths.append(temp_path)
            if chunk_end % log_every == 0 or chunk_end == len(plot_files):
                print(f"  Processed {chunk_end:,}/{len(plot_files):,} files from {run_dir}")

        if not chunk_paths:
            print(f"  Warning: no data loaded for run {run_dir}")
            continue

        print(f"  Assembling run dataframe from {len(chunk_paths):,} chunk files...")
        run_df = pd.concat((pd.read_pickle(path) for path in chunk_paths), ignore_index=True)
        per_run_tables.append(run_df)
        if run_cache_path is not None:
            run_df.to_pickle(run_cache_path)
            print(f"  Saved run checkpoint: {run_cache_path}")
            print(f"  Run rows: {len(run_df):,}")
        if run_chunk_dir is not None:
            manifest = pd.DataFrame({"chunk_path": [str(path) for path in chunk_paths]})
            manifest.to_csv(run_chunk_dir / "chunk_manifest.tsv", sep="\t", index=False)

    if not per_run_tables:
        raise ValueError("No XY data loaded while building cache.")

    combined = pd.concat(per_run_tables, ignore_index=True)
    print("=" * 72)
    print(f"XY cache build complete")
    print(f"  Rows: {len(combined):,}")
    print(f"  Unique reads: {combined['read_id'].nunique():,}")
    print("=" * 72)
    return combined


def load_xy_data_cached(config: dict) -> pd.DataFrame:
    """Load XY data from cache if available, otherwise build and persist it."""
    cache_path = config["data"].get("xy_cache_path")
    if not cache_path:
        return load_all_xy_data(
            base_dir=config["data"]["base_dir"],
            run_dirs=config["data"].get("run_dirs"),
        )

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".metadata.yaml")

    if cache_path.exists():
        print(f"  Reusing cached XY table: {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"  No XY cache found. Building cache at: {cache_path}")
    try:
        xy_data = build_xy_cache_table(
            base_dir=config["data"]["base_dir"],
            run_dirs=config["data"].get("run_dirs"),
            log_every=config["data"].get("xy_cache_log_every", 500),
            cache_path=cache_path,
            chunk_size=config["data"].get("xy_cache_chunk_size", 250),
        )
        xy_data.to_pickle(cache_path)
    except Exception as exc:
        print(f"  XY cache build failed: {exc}")
        print(traceback.format_exc())
        raise

    metadata = {
        "base_dir": config["data"]["base_dir"],
        "run_dirs": config["data"].get("run_dirs"),
        "n_rows": int(len(xy_data)),
        "n_reads": int(xy_data["read_id"].nunique()),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        yaml.dump(metadata, handle, default_flow_style=False)
    print(f"  XY cache saved: {cache_path}")
    print(f"  XY cache metadata saved: {metadata_path}")
    return xy_data
