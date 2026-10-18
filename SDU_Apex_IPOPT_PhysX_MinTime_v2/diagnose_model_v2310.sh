#!/usr/bin/env bash
# v2.3.10 source-frame correction: numeric load/force audit only, no IPOPT.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -r ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz ]]; then
  printf '%s\n' 'Missing saved native seed: optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz' >&2
  exit 2
fi
if [[ ! -r .venv/bin/activate ]]; then
  printf '%s\n' 'Missing existing .venv/bin/activate' >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf '%s\n' 'v2.3.10: comparing historical mixed-frame vs corrected source suspension geometry.'
python3 -u diagnose_suspension_frame_v239.py \
  --repo-root .. \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --output ./suspension_frame_v2310 \
  --allow-history-fetch
python3 -u diagnose_force_consistency_v238.py \
  --repo-root .. \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --output ./force_consistency_v2310 \
  --allow-history-fetch
python3 - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('suspension_frame_v2310/summary.json').read_text())
assert p['source_model_mounts_model_left_positive'], 'Current source model still uses Unity right-positive lateral mounts; OCP physics fix is not active'
old=p['candidate_results']['original_unmodified']
new=p['candidate_results']['mirrored_lateral_mount_x']
assert new['global_rms_load_error_n'] < old['global_rms_load_error_n'], 'Corrected coordinate frame failed to improve native load RMS; do not run OCP'
assert new['front_left_minus_right']['correlation'] > 0.99, 'Corrected frame does not reproduce native load-transfer sign; do not run OCP'
print('SOURCE FRAME CHECK PASSED; historical load RMS %.6f N -> corrected %.6f N' % (old['global_rms_load_error_n'],new['global_rms_load_error_n']))
print('Native-vs-symbolic lateral acceleration RMS (may still be worse after correction): %.6f m/s²' % new['symbolic_tire_ay_vs_native_motion_rms_mps2'])
print('Remaining native/symbolic total suspension-load offset: %.6f N' % new['symbolic_minus_native_total_load_mean_n'])
print('No IPOPT, no native replay, no certified racing lap.')
PY
printf '%s\n' 'Files: suspension_frame_v2310/summary.json, force_consistency_v2310/summary.json'
