#!/usr/bin/env bash
# Map-only autonomous optimization; same invocation resumes its checkpoints.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
python="$repo_root/SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python"
[[ -x "$python" ]] || { echo "Missing optimizer environment: $python" >&2; exit 2; }
entry="$repo_root/SDU_Apex_IPOPT_PhysX_MinTime_v2/track_optimizer.py"
foreground=0
stop=0
args=()
for arg in "$@"; do
  case "$arg" in
    --foreground) foreground=1 ;;
    --stop) stop=1 ;;
    --help|-h) exec "$python" "$entry" --help ;;
    *) args+=("$arg") ;;
  esac
done
if [[ ${#args[@]} -eq 0 ]]; then
  echo 'Usage: tools/run_raceline_campaign.sh TRACK.pgm [TRACK.yaml] [options]' >&2
  echo 'Options: --status, --watch, --stop, --foreground, --prepare-only, --output DIR' >&2
  exit 2
fi
for arg in "${args[@]}"; do
  if [[ "$arg" == --status || "$arg" == --watch || "$arg" == --describe ]]; then
    exec "$python" "$entry" "${args[@]}"
  fi
done
description="$("$python" "$entry" "${args[@]}" --describe)"
output="$("$python" -c 'import json,sys; print(json.loads(sys.argv[1])["output"])' "$description")"
unit="$("$python" -c 'import json,sys; print(json.loads(sys.argv[1])["unit"])' "$description")"
if [[ "$stop" == 1 ]]; then
  systemctl --user stop --no-block "$unit.service"
  echo "Stop requested. Durable iterates and validated results remain in $output"
  exit 0
fi
# One campaign globally, not merely one service per output directory. Serialize
# the check+launch so two terminal invocations cannot both pass the preflight.
mkdir -p -- "$repo_root/live_runs"
exec 9>"$repo_root/live_runs/.track_optimizer_launch.lock"
flock -n 9 || { echo 'Another optimizer launch is in progress.' >&2; exit 1; }
active_units="$(systemctl --user list-units --type=service --state=active,activating \
  --plain --no-legend --no-pager 'sdu-track-optimizer-*.service')"
while read -r active_unit _; do
  if [[ -n "$active_unit" && "$active_unit" != "$unit.service" ]]; then
    echo "Another optimizer is already running: $active_unit. Stop it before starting a different campaign." >&2
    exit 1
  fi
done <<< "$active_units"
if [[ "$foreground" == 1 ]]; then
  exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$python" -u "$entry" "${args[@]}"
fi
if systemctl --user is-active --quiet "$unit.service"; then
  echo "Already running: $unit.service"
else
  mkdir -p -- "$output"
  systemctl --user reset-failed "$unit.service" >/dev/null 2>&1 || true
  systemd-run --user --unit="$unit" \
  --property="WorkingDirectory=$repo_root" \
  --property=Nice=5 --property=TimeoutStopSec=60 \
  --property="StandardOutput=append:$output/optimizer.log" --property=StandardError=inherit \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  "$python" -u "$entry" "${args[@]}"
fi
echo "Output: $output"
echo "Log: tail -F '$output/optimizer.log'"
echo 'Use the same map arguments with --watch for stage/checkpoint status.'
echo 'Use the same invocation to resume; append --stop to stop gracefully.'
