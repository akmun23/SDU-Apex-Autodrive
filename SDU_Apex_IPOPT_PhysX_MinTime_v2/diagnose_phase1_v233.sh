#!/usr/bin/env bash
# Quick feasibility performance diagnostic; no native replay or track export.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -r ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  echo "Missing saved native seed: ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz" >&2
  exit 2
fi
if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing existing venv: .venv/bin/activate" >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
printf 'Phase-I v2.3.3 performance diagnostic (max 180 seconds; no native replay)\n'
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -u run_ipopt.py \
  --repo-root .. \
  --mesh-spacing 0.20 \
  --seed-backend native \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --no-seed-force-projection \
  --brake-zones 0 \
  --phase1-hessian limited-memory \
  --feasibility-iterations 12 \
  --iterations 1 \
  --native-correction-passes 0 \
  --print-level 5 \
  --skip-native \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_3_3_debug \
  --allow-history-fetch \
  2>&1 | tee ./ipopt_v233_debug.log
status=${PIPESTATUS[0]}
set -e
printf '\nExit status: %s (124 means the 180-second limit was reached).\n' "$status"
printf 'Saved diagnostic: %s/ipopt_v233_debug.log\n' "$PWD"
exit "$status"
