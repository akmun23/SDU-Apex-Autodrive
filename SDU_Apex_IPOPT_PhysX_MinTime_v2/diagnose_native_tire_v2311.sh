#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source .venv/bin/activate
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=5s 180s  python3 -u diagnose_native_tire_v2311.py  --repo-root ..  --physx-binary ./native_physx_vehicle_step  --output ./native_contact_v2311  --allow-history-fetch 2>&1 | tee ./native_contact_v2311.log
echo "Native tire contact diagnostic complete."
