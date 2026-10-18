#!/usr/bin/env bash
# Short source-verifiable test of full 3D angular-rate tire contact kinematics.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
if [[ ! -r native_slip_v2314/per_wheel.csv || ! -r native_slip_v2314/summary.json ]]; then
  echo 'ERROR: saved v2.3.14 safe native contact trace missing' >&2; exit 2
fi
if [[ ! -r optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  echo 'ERROR: saved 143-station native warmstart missing' >&2; exit 2
fi
python3 -u diagnose_contact_v2316.py --repo-root .. --allow-history-fetch --output native_contact_v2316
