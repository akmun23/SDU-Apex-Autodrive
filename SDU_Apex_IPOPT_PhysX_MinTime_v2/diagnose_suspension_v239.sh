#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 -u diagnose_suspension_frame_v239.py \
  --repo-root .. \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --output ./suspension_frame_v239 \
  --allow-history-fetch
