#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ ! -s native_slip_v2313/per_wheel.csv || ! -s native_slip_v2313/trial.json ]]; then
  echo 'Missing saved v2.3.13 native slip recording. Please run diagnose_slip_v2313.sh first.' >&2
  exit 2
fi
python3 -u diagnose_slip_parity_v2314.py --repo-root .. --source-dir ./native_slip_v2313 --output ./native_slip_v2314 --allow-history-fetch | tee native_slip_v2314.log
