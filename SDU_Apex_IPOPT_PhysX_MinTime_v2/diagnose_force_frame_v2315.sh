#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ ! -s native_slip_v2314/per_wheel.csv || ! -s native_slip_v2314/summary.json ]]; then
  echo 'Missing v2.3.14 saved native-slip report. Re-run diagnose_slip_v2314.sh first.' >&2
  exit 2
fi
python3 -u diagnose_force_frame_v2315.py --repo-root .. --source-dir ./native_slip_v2314 --output ./native_force_frame_v2315 --allow-history-fetch | tee native_force_frame_v2315.log
