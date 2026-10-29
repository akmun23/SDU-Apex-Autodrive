#!/usr/bin/env bash
# Start/resume the source-checked search independently of this terminal.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"
unit=sdu-apex-raceline-campaign.service
output="${1:-live_runs/raceline_campaign_20261029}"
target="${2:-4.7}"
[[ $# -le 2 ]] || { echo 'Usage: run_raceline_campaign.sh [OUTPUT_DIR [TARGET_SECONDS]]' >&2; exit 2; }
if systemctl --user is-active --quiet "$unit"; then
  echo "Search already running: $unit (no second optimizer launched)"
  exit 0
fi
python="$repo_root/SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python"
[[ -x "$python" ]] || { echo "Missing optimizer environment: $python" >&2; exit 2; }
output="$(realpath -m -- "$output")"
# Clear a previous failure; the transient unit can then be recreated. The
# campaign itself rejects mismatched settings and overlapping writer locks.
systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
systemd-run --user --unit="$unit" \
  --property="WorkingDirectory=$repo_root" \
  --property=Nice=5 --property=Restart=on-failure --property=RestartSec=10 \
  --property=StartLimitBurst=3 --property=StartLimitIntervalSec=300 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  "$python" -u "$repo_root/SDU_Apex_IPOPT_PhysX_MinTime_v2/command_campaign.py" \
  --seed "$repo_root/f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800_witness.npz" \
  --map "$repo_root/f1tenth_planning/maps/autodrive_practice_20260924_b.pgm" \
  --output "$output" --target-seconds "$target" --phases 8 --workers "$(nproc)" \
  --iterations 300 --stagnation-iterations 50 --refinements 6 --resume-if-exists
echo "Status: $python SDU_Apex_IPOPT_PhysX_MinTime_v2/command_campaign.py --status '$output'"
echo "Live log: journalctl --user -fu $unit"
