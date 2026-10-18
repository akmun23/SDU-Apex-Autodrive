#!/usr/bin/env bash
# v2.3.7: compare unprojected native seed after v2.3.6 projected-seed regression.
# This is a feasibility test, NOT a racing lap and NOT a native validation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -r ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  printf 'Missing saved native seed: optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz\n' >&2
  exit 2
fi
if [[ ! -r .venv/bin/activate ]]; then
  printf 'Missing existing .venv/bin/activate\n' >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf '%s\n' 'v2.3.7: UNPROJECTED native seed; reject checkpoints that worsen largest physical defect.'
printf '%s\n' '50 IPOPT feasibility iterations; maximum 240 seconds; no native replay.'
printf '%s\n' 'This test keeps drive-only mode to isolate the model mismatch; it does NOT optimize brake phases.'
set +e
timeout --signal=TERM --kill-after=10s 240s python3 -u run_ipopt.py \
  --repo-root .. \
  --mesh-spacing 0.20 \
  --seed-backend native \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --brake-zones 0 \
  --no-seed-force-projection \
  --phase1-hessian limited-memory \
  --feasibility-iterations 50 \
  --iterations 1 \
  --native-correction-passes 0 \
  --print-level 5 \
  --skip-native \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_3_7_unprojected_debug \
  --allow-history-fetch \
  2>&1 | tee ./ipopt_v237_debug.log
status=${PIPESTATUS[0]}
set -e
printf '\nExit status: %s (2=infeasible, 124=timeout; neither is a validated lap).\n' "$status"
printf 'Log: %s/ipopt_v237_debug.log\n' "$PWD"
printf 'Report: %s/optimized_practice_v2_3_7_unprojected_debug/ipopt_report_pass_00.json\n' "$PWD"
printf 'Projection: %s/optimized_practice_v2_3_7_unprojected_debug/warmstart_physx/seed_projection_audit.json\n' "$PWD"
exit "$status"
