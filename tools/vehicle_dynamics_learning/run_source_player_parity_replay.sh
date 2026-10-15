#!/usr/bin/env bash
set -euo pipefail

if (($# < 2 || $# > 3)); then
  echo "Usage: $0 <closed-source-bag-directory> <run-id> [target-first-request-sequence]" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/tools/docker_env.sh"
input_bag="$(realpath -- "$1")"
run_id="$2"
target_first_request_sequence="${3:-8}"
run_parent="${repo_root}/live_runs/${run_id}"
api_image="${AUTODRIVE_API_IMAGE:-autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7}"
container="sdu_apex_source_replay_${run_id}"

case "${input_bag}" in
  "${repo_root}"/*) ;;
  *) echo "Input bag must be inside the workspace." >&2; exit 2 ;;
esac
if [[ ! -s "${input_bag}/run_0.db3" || ! -s "${input_bag}/metadata.yaml" ]]; then
  echo "Input must be a closed ROS 2 SQLite bag directory." >&2
  exit 2
fi
if [[ ! "${run_id}" =~ ^[[:alnum:]_-]+$ ]]; then
  echo "Run ID may contain only letters, digits, underscores, and hyphens." >&2
  exit 2
fi
if [[ -e "${run_parent}" ]]; then
  echo "Output already exists: ${run_parent}" >&2
  exit 1
fi
if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Container already exists: ${container}" >&2
  exit 1
fi
if ! docker image inspect "${api_image}" >/dev/null 2>&1; then
  echo "Required ROS 2 development image is not present locally: ${api_image}" >&2
  exit 1
fi

mkdir -p "${run_parent}"
export RUN_ID="${run_id}"
profile_host_path="${run_parent}/packet_command_profile.json"
python3 "${repo_root}/tools/vehicle_dynamics_learning/build_unity_packet_command_profile.py" \
  "${input_bag}" "${profile_host_path}" \
  --target-first-request-sequence "${target_first_request_sequence}"
export PACKET_COMMAND_PROFILE="/workspace/src/live_runs/${run_id}/packet_command_profile.json"

docker run --rm --name "${container}" \
  --network=host --ipc=host \
  --log-opt max-size=10m --log-opt max-file=2 \
  -e ROS_DOMAIN_ID=61 \
  -e RUN_ID -e PACKET_COMMAND_PROFILE \
  -v "${repo_root}:/workspace/src:rw" \
  --entrypoint /bin/bash "${api_image}" -lc '
    set -Ee
    source /opt/ros/humble/setup.bash
    source /home/autodrive_devkit/install/setup.bash
    set -u
    set -o pipefail
    export AUTODRIVE_BRIDGE_RATE_HZ=40
    run_parent="/workspace/src/live_runs/${RUN_ID}"
    bridge_log="${run_parent}/bridge.log"
    recorder_log="${run_parent}/recorder.log"
    rosbag_output="${run_parent}/run"
    bridge_pid=""
    recorder_pid=""

    stop_recorder() {
      [[ -n "${recorder_pid}" ]] || return 0
      if kill -0 "${recorder_pid}" 2>/dev/null; then
        kill -INT "${recorder_pid}" 2>/dev/null || true
        for attempt in $(seq 1 300); do
          kill -0 "${recorder_pid}" 2>/dev/null || break
          sleep 0.1
        done
        if kill -0 "${recorder_pid}" 2>/dev/null; then
          kill -TERM "${recorder_pid}" 2>/dev/null || true
          sleep 1
        fi
      fi
      wait "${recorder_pid}" 2>/dev/null || true
      recorder_pid=""
    }

    stop_bridge() {
      [[ -n "${bridge_pid}" ]] || return 0
      if kill -0 "${bridge_pid}" 2>/dev/null; then
        kill -TERM "${bridge_pid}" 2>/dev/null || true
        for attempt in $(seq 1 20); do
          kill -0 "${bridge_pid}" 2>/dev/null || break
          sleep 0.1
        done
      fi
      wait "${bridge_pid}" 2>/dev/null || true
      bridge_pid=""
    }

    cleanup() {
      set +e
      stop_recorder
      stop_bridge
    }
    trap cleanup EXIT INT TERM

    export SDU_APEX_PACKET_COMMAND_PROFILE="${PACKET_COMMAND_PROFILE}"
    python3 -u /workspace/src/sdu_apex_autodrive/sdu_apex_autodrive/bridge_40hz.py \
      >"${bridge_log}" 2>&1 &
    bridge_pid=$!
    bridge_ready=0
    for attempt in $(seq 1 300); do
      if ! kill -0 "${bridge_pid}" 2>/dev/null; then
        echo "Bridge exited before connecting." >&2
        tail -80 "${bridge_log}" >&2 || true
        exit 1
      fi
      if grep -q "Connected!" "${bridge_log}"; then
        bridge_ready=1
        break
      fi
      sleep 0.1
    done
    if [[ "${bridge_ready}" != 1 ]]; then
      echo "Timed out waiting for the source player." >&2
      tail -80 "${bridge_log}" >&2 || true
      exit 1
    fi

    record_topics=(
      /autodrive/roboracer_1/odom
      /autodrive/roboracer_1/ips
      /autodrive/roboracer_1/imu
      /autodrive/roboracer_1/left_encoder
      /autodrive/roboracer_1/right_encoder
      /autodrive/roboracer_1/steering
      /autodrive/roboracer_1/throttle
      /autodrive/roboracer_1/steering_command
      /autodrive/roboracer_1/throttle_command
      /autodrive/roboracer_1/collision_count
      /autodrive/roboracer_1/bridge_packet_timing
      /autodrive/roboracer_1/bridge_timing_fault
    )
    ros2 bag record --max-cache-size 268435456 -o "${rosbag_output}" \
      "${record_topics[@]}" >"${recorder_log}" 2>&1 &
    recorder_pid=$!
    recorder_ready=0
    for attempt in $(seq 1 100); do
      if ! kill -0 "${recorder_pid}" 2>/dev/null; then
        echo "Recorder exited during startup." >&2
        cat "${recorder_log}" >&2 || true
        exit 1
      fi
      if [[ -s "${rosbag_output}/run_0.db3" ]]; then
        recorder_ready=1
        break
      fi
      sleep 0.1
    done
    if [[ "${recorder_ready}" != 1 ]]; then
      echo "Timed out waiting for rosbag initialization." >&2
      cat "${recorder_log}" >&2 || true
      exit 1
    fi
    sleep 1

    profile_complete=0
    for attempt in $(seq 1 3600); do
      if grep -q "development packet command profile complete" "${bridge_log}"; then
        profile_complete=1
        break
      fi
      if ! kill -0 "${bridge_pid}" 2>/dev/null; then
        echo "Bridge exited before the deterministic command profile completed." >&2
        tail -80 "${bridge_log}" >&2 || true
        exit 1
      fi
      sleep 0.1
    done
    if [[ "${profile_complete}" != 1 ]]; then
      echo "Timed out waiting for deterministic packet command replay." >&2
      tail -80 "${bridge_log}" >&2 || true
      exit 1
    fi
    sleep 1
    stop_recorder
    stop_bridge
    echo "Closed replay bag: ${rosbag_output}"
  '
