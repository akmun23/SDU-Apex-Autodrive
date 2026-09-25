#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "Usage: $0 <controller-container> <simulator-container> <recorder-container> <target-lap|collision> [timeout-s]" >&2
  exit 2
fi

controller_container="$1"
simulator_container="$2"
recorder_container="$3"
target_lap="$4"
timeout_s="${5:-120}"
if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ||
      ("${target_lap}" != "collision" && ! "${target_lap}" =~ ^[1-9][0-9]*$) ]]; then
  echo "target-lap must be a positive integer or collision; timeout-s must be positive." >&2
  exit 2
fi

cleanup_done=0
cleanup() {
  if [[ "${cleanup_done}" == "1" ]]; then
    return
  fi
  cleanup_done=1
  docker stop --timeout 0 "${simulator_container}" >/dev/null 2>&1 || true
  docker kill --signal SIGINT "${controller_container}" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    running="$(docker inspect --format '{{.State.Running}}' "${controller_container}" 2>/dev/null || true)"
    [[ "${running}" == "true" ]] || break
    sleep 0.1
  done
  if [[ "${running:-false}" == "true" ]]; then
    docker stop --timeout 0 "${controller_container}" >/dev/null 2>&1 || true
  fi
  docker kill --signal SIGINT "${recorder_container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
if [[ "${target_lap}" == "collision" ]]; then
  docker exec "${controller_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --collision-only --timeout-s "$1"' \
    _ "${timeout_s}"
else
  docker exec "${controller_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --target-lap "$1" --timeout-s "$2"' \
    _ "${target_lap}" "${timeout_s}"
fi
watch_status=$?
set -e
exit "${watch_status}"
