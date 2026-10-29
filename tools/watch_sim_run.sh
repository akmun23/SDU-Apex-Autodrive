#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "Usage: $0 <controller-container> <simulator-container> <recorder-container> <target-lap|collision|mpc|mpc-lap-N|csv-command|csv-follow-N> [timeout-s]" >&2
  exit 2
fi

controller_container="$1"
simulator_container="$2"
recorder_container="$3"
target_lap="$4"
timeout_s="${5:-120}"
stall_timeout_s="${SDU_APEX_WATCH_STALL_TIMEOUT_S:-20}"
post_target_lap_guard_s="${SDU_APEX_WATCH_POST_TARGET_GUARD_S:-0.5}"
minimum_lap_distance_m="${SDU_APEX_WATCH_MIN_DISTANCE_PER_LAP_M:-0}"
simulator_stop_timeout_s="${SDU_APEX_WATCH_SIM_STOP_TIMEOUT_S:-2}"
watcher_container="${SDU_APEX_WATCHER_CONTAINER:-${controller_container}}"
if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ||
      ! "${stall_timeout_s}" =~ ^[1-9][0-9]*$ ||
      ! "${simulator_stop_timeout_s}" =~ ^[0-9]+$ ||
      ! "${post_target_lap_guard_s}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ||
      ! "${minimum_lap_distance_m}" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ||
      ("${target_lap}" != "collision" && "${target_lap}" != "mpc" &&
       "${target_lap}" != "csv-command" &&
       ! "${target_lap}" =~ ^mpc-lap-[1-9][0-9]*$ &&
       ! "${target_lap}" =~ ^csv-follow-[1-9][0-9]*$ &&
       ! "${target_lap}" =~ ^[1-9][0-9]*$) ]]; then
  echo "target-lap must be a positive integer, collision, mpc, mpc-lap-N, csv-command, or csv-follow-N; timeout-s and SDU_APEX_WATCH_STALL_TIMEOUT_S must be positive integers; SDU_APEX_WATCH_SIM_STOP_TIMEOUT_S must be nonnegative; SDU_APEX_WATCH_POST_TARGET_GUARD_S must be nonnegative." >&2
  exit 2
fi

cleanup_done=0
watch_status=0
cleanup() {
  if [[ "${cleanup_done}" == "1" ]]; then
    return
  fi
  cleanup_done=1
  # A rejected MPC plan or collision ends the run immediately. Do not send
  # an active-brake command and let
  # the car continue moving while a graceful Unity shutdown waits.
  if [[ "${watch_status}" == "3" || "${watch_status}" == "5" ||
        "${watch_status}" == "130" ||
        "${watch_status}" == "143" ]]; then
    docker stop --timeout 0 "${simulator_container}" >/dev/null 2>&1 || true
    docker kill --signal SIGINT "${controller_container}" >/dev/null 2>&1 || true
  elif [[ "${watch_status}" != "0" ]]; then
    docker kill --signal SIGINT "${controller_container}" >/dev/null 2>&1 || true
  fi
  # On clean completion, finalize the bag at the requested target-lap boundary.
  docker kill --signal SIGINT "${recorder_container}" >/dev/null 2>&1 || true
  recorder_state="true"
  for _ in $(seq 1 100); do
    recorder_state="$(docker inspect --format '{{.State.Running}}' \
      "${recorder_container}" 2>/dev/null || true)"
    [[ "${recorder_state}" == "true" ]] || break
    sleep 0.1
  done
  if [[ "${recorder_state}" == "true" ]]; then
    docker stop --timeout 5 "${recorder_container}" >/dev/null 2>&1 || true
  fi

  docker stop --timeout "${simulator_stop_timeout_s}" "${simulator_container}" >/dev/null 2>&1 || true
  docker kill --signal SIGINT "${controller_container}" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    running="$(docker inspect --format '{{.State.Running}}' "${controller_container}" 2>/dev/null || true)"
    [[ "${running}" == "true" ]] || break
    sleep 0.1
  done
  if [[ "${running:-false}" == "true" ]]; then
    docker stop --timeout 0 "${controller_container}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'watch_status=130; exit 130' INT
trap 'watch_status=143; exit 143' TERM

set +e
if [[ "${target_lap}" == "collision" ]]; then
  docker exec "${watcher_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --collision-only --timeout-s "$1" --stall-timeout-s "$2"' \
    _ "${timeout_s}" "${stall_timeout_s}"
elif [[ "${target_lap}" == "mpc" ]]; then
  docker exec "${watcher_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --mpc-run --timeout-s "$1" --stall-timeout-s "$2"' \
    _ "${timeout_s}" "${stall_timeout_s}"
elif [[ "${target_lap}" =~ ^mpc-lap-([1-9][0-9]*)$ ]]; then
  mpc_target_lap="${BASH_REMATCH[1]}"
  docker exec "${watcher_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --mpc-run --target-lap "$1" --timeout-s "$2" --stall-timeout-s "$3" --post-target-lap-guard-s "$4" --minimum-lap-distance-m "$5"' \
    _ "${mpc_target_lap}" "${timeout_s}" "${stall_timeout_s}" "${post_target_lap_guard_s}" "${minimum_lap_distance_m}"
elif [[ "${target_lap}" == "csv-command" ]]; then
  docker exec "${watcher_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --csv-command-test --timeout-s "$1" --stall-timeout-s "$2"' \
    _ "${timeout_s}" "${stall_timeout_s}"
elif [[ "${target_lap}" =~ ^csv-follow-([1-9][0-9]*)$ ]]; then
  csv_target_lap="${BASH_REMATCH[1]}"
  docker exec "${watcher_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --csv-path-follow --target-lap "$1" --timeout-s "$2" --stall-timeout-s "$3" --post-target-lap-guard-s "$4"' \
    _ "${csv_target_lap}" "${timeout_s}" "${stall_timeout_s}" "${post_target_lap_guard_s}"
else
  docker exec "${watcher_container}" /bin/bash -lc \
    'source /opt/ros/humble/setup.bash && python3 /workspace/src/tools/watch_sim_run.py --target-lap "$1" --timeout-s "$2" --stall-timeout-s "$3" --post-target-lap-guard-s "$4" --minimum-lap-distance-m "$5"' \
    _ "${target_lap}" "${timeout_s}" "${stall_timeout_s}" "${post_target_lap_guard_s}" "${minimum_lap_distance_m}"
fi
watch_status=$?
set -e
exit "${watch_status}"
