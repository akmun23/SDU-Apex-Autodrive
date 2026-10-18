#!/usr/bin/env bash
# v2.4.0: do not repeat the same 40-iteration run; compare two seed strategies
# and CONTINUE from saved hard-feasible Phase-I incumbents (never a race lap).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
SEED=optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz
if [[ ! -s "$SEED" ]]; then echo "Missing native PhysX seed: $SEED" >&2; exit 2; fi
printf '\n== A. Contact-point tire parity, source-matched v2.3.16 ==\n'
bash diagnose_contact_v2316.sh
# Each variant can reuse two improved physical checkpoints. IPOPT may still
# fail: failure status 2 is the expected result until exact defects vanish.
for variant in unprojected projected; do
  out="v240_phase1_${variant}"
  echo "== B. Phase-I $variant: 60 iterations x up to 3 checkpointed solves =="
  extras=()
  if [[ "$variant" == unprojected ]]; then extras+=(--no-seed-force-projection); fi
  set +e
  timeout --signal=TERM --kill-after=10s 780s python3 -u run_ipopt.py \
      --repo-root .. --mesh-spacing 0.20 --seed-backend native \
      --seed-file "$SEED" "${extras[@]}" \
      --brake-zones 0 --phase1-hessian limited-memory \
      --feasibility-iterations 60 --phase1-restarts 2 \
      --phase1-min-defect-improvement 0.00001 \
      --iterations 1 --native-correction-passes 0 --print-level 5 \
      --skip-native --physx-binary ./native_physx_vehicle_step \
      --output "$out" --allow-history-fetch \
      2>&1 | tee "v240_${variant}.log"
  status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" > "v240_${variant}.exit_status"
  printf 'IPOPT diagnostic %s exit=%s (2 means infeasible, 124 timeout)\n' "$variant" "$status"
  if [[ $status -ne 0 && $status -ne 2 && $status -ne 3 ]]; then
    echo "Unusual exit for $variant; inspect log" >&2
  fi
done
# Independent Hessian experiment is opt-in. Never hide the expensive third run.
if [[ "${V240_RUN_EXACT_HESSIAN:-0}" == 1 ]]; then
  echo '== C. Exact-Hessian Phase-I comparison =='
  set +e
  timeout --signal=TERM --kill-after=10s 780s python3 -u run_ipopt.py \
    --repo-root .. --mesh-spacing 0.20 --seed-backend native \
    --seed-file "$SEED" --no-seed-force-projection \
    --brake-zones 0 --phase1-hessian exact \
    --feasibility-iterations 60 --phase1-restarts 2 \
    --iterations 1 --native-correction-passes 0 --print-level 5 \
    --skip-native --physx-binary ./native_physx_vehicle_step \
    --output v240_phase1_exact --allow-history-fetch \
    2>&1 | tee v240_exact.log
  echo "Exact-Hessian diagnostic status: ${PIPESTATUS[0]}"
  set -e
fi
python3 summarize_campaign_v240.py
# Preserve distinct directory paths; never silently overwrite same-named reports.
zip -q -r -FS v240_review_results_upload.zip \
    v240_campaign_summary.json v240_campaign_summary.md \
    v240_unprojected.log v240_projected.log \n    v240_unprojected.exit_status v240_projected.exit_status \
    v240_phase1_unprojected v240_phase1_projected \
    native_contact_v2316/summary.json \
    native_contact_v2316/per_wheel.csv \
    native_contact_v2316/per_sample.csv
if [[ -d v240_phase1_exact ]]; then
  zip -q -r -FS v240_review_results_upload.zip v240_phase1_exact v240_exact.log
fi
printf '\nOutput: %s/v240_review_results_upload.zip\n' "$PWD"
printf '%s\n' 'This package has NOT certified a strictly feasible minimum-time racing lap.'
