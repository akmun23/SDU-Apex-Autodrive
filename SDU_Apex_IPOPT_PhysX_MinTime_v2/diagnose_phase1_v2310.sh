#!/usr/bin/env bash
# v2.3.10 source-frame corrected OCP quick experiment: 40 IPOPT Phase-I iterations.
# Does NOT solve the 4.9s objective; zero-slack strict feasibility is mandatory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -r ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  printf '%s\n' 'Missing saved native seed' >&2
  exit 2
fi
if [[ ! -r .venv/bin/activate ]]; then
  printf '%s\n' 'Missing existing .venv' >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf '%s\n' 'v2.3.10 corrected source coordinate frame: 40 IPOPT feasibility iterations; at most 210 sec.'
printf '%s\n' 'Unprojected native seed; no brake phases in this diagnostic; no native replay.'
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
  --output ./optimized_practice_v2_3_10_corrected_debug \
  --allow-history-fetch \
  2>&1 | tee ./ipopt_v2310_debug.log
status=${PIPESTATUS[0]}
set -e
printf '\nExit status: %s (2=infeasible, 124=timeout, 0=process finished; only strict report can establish a feasible lap).\n' "$status"
printf 'Log: %s/ipopt_v2310_debug.log\n' "$PWD"
printf 'Report: %s/optimized_practice_v2_3_10_corrected_debug/ipopt_report_pass_00.json\n' "$PWD"
exit "$status"
