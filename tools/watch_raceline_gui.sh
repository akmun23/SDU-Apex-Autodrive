#!/usr/bin/env bash
# A separate read-only observer: never start/stop/change the optimizer itself.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$repo_root/SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python" \
  "$repo_root/tools/optimizer_dashboard.py" "$@"
