#!/usr/bin/env bash
set -u

CONFIG="${1:-CODEX/configs/weak5_rectangular.yaml}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

CACHE_PATH="$(python - <<'PY' "$CONFIG"
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config["data"]["xy_cache_path"])
PY
)"

LOG_DIR="$ROOT_DIR/CODEX/results/cache/logs"
mkdir -p "$LOG_DIR"
SUPERVISOR_LOG="$LOG_DIR/build_xy_cache_supervisor.log"

attempt=0
while [[ ! -f "$CACHE_PATH" ]]; do
  attempt=$((attempt + 1))
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  attempt_log="$LOG_DIR/build_xy_cache_attempt_$(date '+%Y%m%d_%H%M%S').log"

  {
    echo "[$timestamp] Attempt $attempt starting"
    echo "[$timestamp] Config: $CONFIG"
    echo "[$timestamp] Cache target: $CACHE_PATH"
    echo "[$timestamp] Attempt log: $attempt_log"
  } | tee -a "$SUPERVISOR_LOG"

  /bin/bash -lc "source ~/miniforge3/etc/profile.d/conda.sh && conda activate ONT && python CODEX/scripts/build_xy_cache_codex.py --config '$CONFIG'" \
    >"$attempt_log" 2>&1
  exit_code=$?

  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ -f "$CACHE_PATH" ]]; then
    echo "[$timestamp] Cache build completed successfully on attempt $attempt" | tee -a "$SUPERVISOR_LOG"
    break
  fi

  echo "[$timestamp] Attempt $attempt exited with code $exit_code; cache not complete yet. Sleeping 5s before resume." | tee -a "$SUPERVISOR_LOG"
  sleep 5
done

timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$timestamp] Final cache present at: $CACHE_PATH" | tee -a "$SUPERVISOR_LOG"
