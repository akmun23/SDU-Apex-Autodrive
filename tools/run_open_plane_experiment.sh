#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0" \
    "Runs the development-only open-plane experiment against the Explore simulator." \
    "Start it separately with: SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode ./tools/start_simulator.sh" \
    "Configure with SDU_APEX_EXPERIMENT_PROFILE, SDU_APEX_EXPERIMENT_RUN_ID, and related SDU_APEX_EXPERIMENT_* variables." \
    "New paired throttle-slew profile: throttle_slew_pair at 4.5 or 6.5 m/s." \
    "Use --help to display this text; positional arguments are not accepted."
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (($# != 0)); then
  usage >&2
  exit 2
fi

# Start the pinned explore simulator separately with:
#   SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode ./tools/start_simulator.sh
# Then run this script. It launches only the bridge, minimal recorder, and the
# development-only direct excitation process; no MPC/actuator stack is started.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

image="${AUTODRIVE_API_IMAGE:-autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7}"
profile="${SDU_APEX_EXPERIMENT_PROFILE:-high_angle_boundary}"
seed="${SDU_APEX_EXPERIMENT_SEED:-20260926}"
timeout_s="${SDU_APEX_EXPERIMENT_TIMEOUT_S:-180}"
speed_hold_kp="${SDU_APEX_EXPERIMENT_SPEED_HOLD_KP:-0.04}"
speed_hold_ki="${SDU_APEX_EXPERIMENT_SPEED_HOLD_KI:-0.0}"
speed_median_gate_mps="${SDU_APEX_EXPERIMENT_SPEED_MEDIAN_GATE_MPS:-0.12}"
speed_p95_gate_mps="${SDU_APEX_EXPERIMENT_SPEED_P95_GATE_MPS:-0.20}"
domain_id="${SDU_APEX_EXPERIMENT_DOMAIN_ID:-61}"
run_id="${SDU_APEX_EXPERIMENT_RUN_ID:-openplane_${profile}_$(date +%Y%m%d_%H%M%S)}"
container="${SDU_APEX_EXPERIMENT_CONTAINER:-sdu_apex_openplane_${run_id}}"
run_parent="${repo_root}/live_runs/${run_id}"

if [[ ! "${run_id}" =~ ^[[:alnum:]_-]+$ ]]; then
  echo "Run ID may contain only letters, digits, underscores, and hyphens." >&2
  exit 2
fi
if [[ "${profile}" != high_angle_boundary && "${profile}" != isolated_boundary && "${profile}" != isolated_speed_sweep && "${profile}" != isolated_force_3mps && "${profile}" != isolated_force_4mps && "${profile}" != isolated_force_5mps && "${profile}" != isolated_highspeed_surface && "${profile}" != isolated_3to5_response_surface && "${profile}" != isolated_highspeed_crossfactor && "${profile}" != isolated_highspeed_tail && "${profile}" != isolated_transition_65mps && "${profile}" != isolated_transition_45mps && "${profile}" != isolated_transition_speed_surface && "${profile}" != isolated_transition_support && "${profile}" != isolated_transition_bridge && "${profile}" != isolated_transition_low_support && "${profile}" != isolated_transition_full_surface && "${profile}" != transient_4mps && "${profile}" != transient_fullsteer_4mps && "${profile}" != transient_transition_4mps && "${profile}" != transient_transition_4mps_fixedthrottle && "${profile}" != transient_transition_dwell_4mps_fixedthrottle && "${profile}" != throttle_slew_pair && "${profile}" != full_input_excitation && "${profile}" != grid ]]; then
  echo "Invalid SDU_APEX_EXPERIMENT_PROFILE; run $0 --help for usage." >&2
  exit 2
fi
if [[ -e "${run_parent}/run" ]]; then
  echo "Bag output already exists: ${run_parent}/run" >&2
  exit 1
fi
if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Experiment container already exists: ${container}" >&2
  exit 1
fi

mkdir -p "${run_parent}"
if ! docker image inspect "${image}" >/dev/null 2>&1; then
  docker pull "${image}"
fi

echo "Run ID: ${run_id}"
echo "Profile: ${profile}; seed: ${seed}; ROS domain: ${domain_id}"
echo "Output: ${run_parent}/run (minimal dynamics topics; no LiDAR/camera)"
echo "Waiting for the explore simulator at 127.0.0.1:4567..."

docker run --rm --name "${container}" \
  --network=host --ipc=host \
  --log-opt max-size=10m --log-opt max-file=2 \
  -e "ROS_DOMAIN_ID=${domain_id}" \
  -e "RUN_ID=${run_id}" \
  -e "PROFILE=${profile}" \
  -e "EXPERIMENT_SEED=${seed}" \
  -e "EXPERIMENT_SPEED_MPS=${SDU_APEX_EXPERIMENT_SPEED_MPS:-4.5}" \
  -e "EXPERIMENT_TIMEOUT_S=${timeout_s}" \
  -e "EXPERIMENT_SPEED_HOLD_KP=${speed_hold_kp}" \
  -e "EXPERIMENT_SPEED_HOLD_KI=${speed_hold_ki}" \
  -e "EXPERIMENT_SPEED_MEDIAN_GATE_MPS=${speed_median_gate_mps}" \
  -e "EXPERIMENT_SPEED_P95_GATE_MPS=${speed_p95_gate_mps}" \
  -v "${repo_root}:/workspace/src:rw" \
  --entrypoint /bin/bash "${image}" -lc '
    set -Ee
    source /opt/ros/humble/setup.bash
    source /home/autodrive_devkit/install/setup.bash
    set -u
    set -o pipefail
    export AUTODRIVE_BRIDGE_RATE_HZ=40
    run_parent="/workspace/src/live_runs/${RUN_ID}"
    bridge_log="${run_parent}/bridge.log"
    recorder_log="${run_parent}/recorder.log"
    experiment_log="${run_parent}/experiment.log"
    rosbag_output="${run_parent}/run"
    bridge_pid=""
    recorder_pid=""

    stop_recorder() {
      [[ -n "${recorder_pid}" ]] || return 0
      if kill -0 "${recorder_pid}" 2>/dev/null; then
        kill -TERM "${recorder_pid}" 2>/dev/null || true
        for attempt in $(seq 1 300); do
          kill -0 "${recorder_pid}" 2>/dev/null || break
          sleep 0.1
        done
        if kill -0 "${recorder_pid}" 2>/dev/null; then
          echo "rosbag did not close after SIGTERM; terminating recorder." >&2
          kill -KILL "${recorder_pid}" 2>/dev/null || true
        fi
      fi
      wait "${recorder_pid}" 2>/dev/null || true
      recorder_pid=""
    }

    stop_bridge() {
      [[ -n "${bridge_pid}" ]] || return 0
      if kill -0 "${bridge_pid}" 2>/dev/null; then
        kill -TERM "${bridge_pid}" 2>/dev/null || true
        for attempt in $(seq 1 10); do
          kill -0 "${bridge_pid}" 2>/dev/null || break
          sleep 0.1
        done
        if kill -0 "${bridge_pid}" 2>/dev/null; then
          kill -KILL "${bridge_pid}" 2>/dev/null || true
        fi
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

    python3 -u /workspace/src/sdu_apex_autodrive/sdu_apex_autodrive/bridge_40hz.py \
      >"${bridge_log}" 2>&1 &
    bridge_pid=$!

    bridge_ready=0
    for attempt in $(seq 1 300); do
      if ! kill -0 "${bridge_pid}" 2>/dev/null; then
        echo "Bridge exited before simulator connection." >&2
        tail -80 "${bridge_log}" >&2 || true
        exit 1
      fi
      if grep -q "FATAL timing fault" "${bridge_log}"; then
        echo "Bridge reported a source-stream fault before excitation." >&2
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
      echo "Timed out waiting for simulator data; is the explore simulator running?" >&2
      tail -80 "${bridge_log}" >&2 || true
      exit 1
    fi

    # A large async cache reduces turnover for this run. Do not open/read the
    # SQLite bag until rosbag2 has closed it: a concurrent reader can lock the
    # database and terminate the recorder while it is writing.
    ros2 bag record --max-cache-size 268435456 -o "${rosbag_output}" \
      /autodrive/roboracer_1/odom \
      /autodrive/roboracer_1/ips \
      /autodrive/roboracer_1/imu \
      /autodrive/roboracer_1/left_encoder \
      /autodrive/roboracer_1/right_encoder \
      /autodrive/roboracer_1/steering \
      /autodrive/roboracer_1/throttle \
      /autodrive/roboracer_1/steering_command \
      /autodrive/roboracer_1/throttle_command \
      /autodrive/roboracer_1/collision_count \
      /autodrive/roboracer_1/bridge_packet_timing \
      /autodrive/roboracer_1/bridge_timing_fault \
      /open_plane_experiment/phase \
      >"${recorder_log}" 2>&1 &
    recorder_pid=$!

    db_file="${rosbag_output}/run_0.db3"
    recorder_ready=0
    for attempt in $(seq 1 100); do
      if ! kill -0 "${recorder_pid}" 2>/dev/null; then
        echo "Recorder exited during startup." >&2
        cat "${recorder_log}" >&2 || true
        exit 1
      fi
      if [[ -s "${db_file}" ]]; then
        recorder_ready=1
        break
      fi
      sleep 0.1
    done
    if [[ "${recorder_ready}" != 1 ]]; then
      echo "Timed out waiting for rosbag to initialize." >&2
      cat "${recorder_log}" >&2 || true
      exit 1
    fi
    sleep 1
    if ! kill -0 "${recorder_pid}" 2>/dev/null; then
      echo "Recorder exited before excitation; refusing to drive without a bag." >&2
      cat "${recorder_log}" >&2 || true
      exit 1
    fi

    set +e
    python3 /workspace/src/tools/open_plane_excitation.py \
      --profile "${PROFILE}" --seed "${EXPERIMENT_SEED}" \
      --transition-speed-mps "${EXPERIMENT_SPEED_MPS}" \
      --speed-hold-kp "${EXPERIMENT_SPEED_HOLD_KP}" \
      --speed-hold-ki "${EXPERIMENT_SPEED_HOLD_KI}" \
      --speed-median-gate-mps "${EXPERIMENT_SPEED_MEDIAN_GATE_MPS}" \
      --speed-p95-gate-mps "${EXPERIMENT_SPEED_P95_GATE_MPS}" \
      --timeout-s "${EXPERIMENT_TIMEOUT_S}" \
      >"${experiment_log}" 2>&1 &
    experiment_pid=$!
    while kill -0 "${experiment_pid}" 2>/dev/null; do
      if ! kill -0 "${recorder_pid}" 2>/dev/null; then
        echo "Recorder exited during excitation; requesting a safe stop." >&2
        kill -TERM "${experiment_pid}" 2>/dev/null || true
        break
      fi
      sleep 0.5
    done
    wait "${experiment_pid}"
    experiment_status=$?
    if ! kill -0 "${recorder_pid}" 2>/dev/null; then
      echo "Recorder did not remain alive for the complete excitation." >&2
      experiment_status=1
    fi
    set -e

    cat "${experiment_log}"
    stop_recorder
    stop_bridge
    echo "Bag closed at ${rosbag_output}; driver exit status=${experiment_status}"
    exit "${experiment_status}"
  '
