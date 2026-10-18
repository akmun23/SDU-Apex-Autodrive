#!/usr/bin/env bash
# v2.3.5 short diagnostic. Reuses existing PhysX warm-start; no replay.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -r ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  printf 'Missing native seed: ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz\n' >&2
  exit 2
fi
if [[ ! -r .venv/bin/activate ]]; then
  printf 'Missing existing .venv\n' >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf 'v2.3.5 strict-dynamics checkpoint diagnostic; maximum 240 seconds, no native replay.\n'
set +e
timeout --signal=TERM --kill-after=10s 240s python3 -u run_ipopt.py \
  --repo-root .. \
  --mesh-spacing 0.20 \
  --seed-backend native \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --no-seed-force-projection \
  --brake-zones 0 \
  --phase1-hessian limited-memory \
  --feasibility-iterations 50 \
  --iterations 1 \
  --native-correction-passes 0 \
  --print-level 5 \
  --skip-native \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_3_5_debug \
  --allow-history-fetch \
  2>&1 | tee ./ipopt_v235_debug.log
status=${PIPESTATUS[0]}
set -e
printf '\nExit status: %s (2=infeasible, 124=timeout; neither is a valid optimized lap).\n' "$status"
printf 'Log: %s/ipopt_v235_debug.log\n' "$PWD"
printf 'Report: %s/optimized_practice_v2_3_5_debug/ipopt_report_pass_00.json\n' "$PWD"
exit "$status"
