#!/usr/bin/env python
"""Train the CODEX weakly supervised 5-event model."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CODEX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODEX_ROOT))


def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] train_weak5_codex.py starting", flush=True)
    parser = argparse.ArgumentParser(description="Train CODEX weak 5-event model")
    parser.add_argument("--config", required=True, help="Path to CODEX YAML config")
    args = parser.parse_args()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Config path: {args.config}", flush=True)

    # Tee stdout+stderr to the results dir so the log lives alongside the model
    with open(args.config, "r", encoding="utf-8") as _fh:
        _cfg = yaml.safe_load(_fh)
    _log_dir = Path(_cfg["output"]["results_dir"])
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _log_dir / "train.log"
    _log_fh = open(_log_path, "a", encoding="utf-8", buffering=1)
    import io
    class _Tee(io.TextIOBase):
        def __init__(self, *streams): self._streams = streams
        def write(self, s):
            for st in self._streams: st.write(s)
            return len(s)
        def flush(self):
            for st in self._streams: st.flush()
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Logging to {_log_path}", flush=True)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Config loaded", flush=True)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Importing training stack...", flush=True)

    from replication_analyzer_codex.training import train_weak5_model

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Training module imported", flush=True)
    result = train_weak5_model(config)
    print("Training complete.")
    print(f"Max length: {result['max_length']}")
    if result['train_info'] is not None:
        print(f"Train reads: {len(result['train_info']):,}")
        print(f"Val reads: {len(result['val_info']):,}")
    else:
        print("(chunk-streaming mode — read counts not tracked in-memory)")


if __name__ == "__main__":
    main()
