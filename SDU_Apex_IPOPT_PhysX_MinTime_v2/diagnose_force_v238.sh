#!/usr/bin/env bash
# v2.3.8: diagnose source-model force inconsistencies without IPOPT or replay.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 -u diagnose_force_consistency_v238.py \
  --repo-root .. \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --output ./force_consistency_v238 \
  --allow-history-fetch
