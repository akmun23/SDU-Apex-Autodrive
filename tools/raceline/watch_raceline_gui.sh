#!/usr/bin/env bash
# Separate dashboard: never starts/stops/changes the optimizer itself.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$repo_root/raceline_optimizer/.venv/bin/python" \
  "$repo_root/tools/raceline/optimizer_dashboard.py" "$@"
