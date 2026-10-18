#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f native_contact_v2311/per_wheel.csv ]]; then
  echo 'ERROR: v2.3.11 native contact CSV is missing. Run the previous successful capture first.' >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 diagnose_native_tire_v2312.py --repo-root .. \
  --input native_contact_v2311/per_wheel.csv \
  --output native_contact_v2312/summary.json
