#!/usr/bin/env python
"""Build the CODEX XY cache without starting training."""

from datetime import datetime
from pathlib import Path
import sys

import argparse
import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))

from replication_analyzer_codex.data_cache import load_xy_data_cached


def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] build_xy_cache_codex.py starting", flush=True)
    parser = argparse.ArgumentParser(description="Build CODEX XY cache")
    parser.add_argument("--config", required=True, help="Path to CODEX YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Config loaded", flush=True)

    xy_data = load_xy_data_cached(config)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Cache ready", flush=True)
    print(f"Rows: {len(xy_data):,}")
    print(f"Unique reads: {xy_data['read_id'].nunique():,}")


if __name__ == "__main__":
    main()
