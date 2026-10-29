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
