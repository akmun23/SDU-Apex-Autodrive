#!/usr/bin/env bash
# Short, visible IPOPT diagnostics, with strict zero-slack failure report.
# Reuses the user's existing native rollout, does not replay native PhysX.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -r ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  echo 'Missing ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz' >&2
  exit 2
fi
if [[ ! -f .venv/bin/activate ]]; then
  echo 'Missing the existing .venv; install project requirements first.' >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf 'v2.3.4 IPOPT strict-defect diagnostic (180s limit, no native replay)\n'
set +e
timeout --signal=TERM --kill-after=10s 180s python3 -u run_ipopt.py \
  --repo-root .. \
  --mesh-spacing 0.20 \
  --seed-backend native \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --no-seed-force-projection \
  --brake-zones 0 \
  --phase1-hessian limited-memory \
  --feasibility-iterations 24 \
  --iterations 1 \
  --native-correction-passes 0 \
  --print-level 5 \
  --skip-native \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_3_4_debug \
  --allow-history-fetch \
  2>&1 | tee ./ipopt_v234_debug.log
status=${PIPESTATUS[0]}
set -e
printf '\nExit status: %s (2 = infeasible, 124 = timeout; neither is a lap-time result).\n' "$status"
printf 'Saved log: %s/ipopt_v234_debug.log\n' "$PWD"
printf 'Saved solver report: %s/optimized_practice_v2_3_4_debug/ipopt_report_pass_00.json\n' "$PWD"
exit "$status"
