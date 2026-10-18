#!/usr/bin/env bash
# Bounded IPOPT diagnostic with source-corrected callback and native warm start.
# This prohibits braking only to isolate model consistency, not for a race lap.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ ! -r optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  echo 'Missing saved native PhysX warm start' >&2
  exit 2
fi
printf '%s\n' 'v2.3.15: native lateral tire force transformed into left-positive chassis frame.'
printf '%s\n' '40 Phase-I feasibility iterations, 210s timeout; NO native replay; NO racing trajectory accepted without zero defect.'
set +e
timeout --signal=TERM --kill-after=10s 210s python3 -u run_ipopt.py \
  --repo-root .. \
  --mesh-spacing 0.20 \
  --seed-backend native \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --no-seed-force-projection \
  --brake-zones 0 \
  --phase1-hessian limited-memory \
  --feasibility-iterations 40 \
  --iterations 1 \
  --native-correction-passes 0 \
  --print-level 5 \
  --skip-native \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_3_15_native_slip_debug \
  --allow-history-fetch 2>&1 | tee ipopt_v2315_debug.log
status=${PIPESTATUS[0]}
set -e
printf '\nExit status %s (2=not feasible, 124=timeout).\n' "$status"
printf 'Report: %s/optimized_practice_v2_3_15_native_slip_debug/ipopt_report_pass_00.json\n' "$PWD"
exit "$status"
