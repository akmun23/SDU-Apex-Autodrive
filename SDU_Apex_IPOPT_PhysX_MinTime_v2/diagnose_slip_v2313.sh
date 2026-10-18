#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=5s 210s \
    python3 -u diagnose_slip_v2313.py \
    --repo-root .. --physx-binary ./native_physx_vehicle_step \
    --output ./native_slip_v2313 --allow-history-fetch \
    2>&1 | tee ./native_slip_v2313.log
if [[ ! -s native_slip_v2313/summary.json ]]; then
  echo 'ERROR: Native slip audit did not produce summary.json' >&2
  exit 2
fi
