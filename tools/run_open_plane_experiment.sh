#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0" \
    "Runs the development-only open-plane experiment against the Explore simulator." \
    "Start it separately with: SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode ./tools/start_simulator.sh" \
    "Configure with SDU_APEX_EXPERIMENT_PROFILE, SDU_APEX_EXPERIMENT_RUN_ID, and related SDU_APEX_EXPERIMENT_* variables." \
    "For a side-by-side sensor-odometry comparison, start tools/run_explore_sensor_odometry.sh and set SDU_APEX_EXPERIMENT_CAPTURE_SENSOR_ODOM=1." \
    "New paired throttle-slew profile: throttle_slew_pair at 4.5 or 6.5 m/s." \
    "race_domain_swerve_throttle_slew_{train,validation} pairs throttle steps/ramps during reset-isolated steering reversals." \
    "race_domain_swerve_throttle_slew_frontier_validation pairs ramp/step throttle reductions at 9.5–11.1 m/s and 0.12–0.20 rad." \
    "race_domain_swerve_throttle_slew_11mps_replication repeats only the 11.1 m/s frontier (0.12/0.16/0.20 rad, both turn directions)." \
    "race_domain_swerve_throttle_slew_moderate_validation adds paired throttle up/down tests at 4.5/6.5/7.5 m/s and 0.12/0.20 rad." \
    "race_domain_swerve_throttle_slew_lowsteer_validation tests paired throttle up/down at 4.5/6.5/7.5 m/s and 0.08/0.10 rad." \
    "race_domain_swerve_throttle_rate_sweep_validation compares three throttle-rise ramps with reset-matched steps at 8 m/s and 0.30/0.42 rad." \
    "race_domain_swerve_throttle_rate_sweep_highsteer_validation repeats only the governor-clean 0.42 rad cells at 8 m/s (0.15/0.30 s ramps)." \
    "race_domain_swerve_throttle_rate_factorial_validation crosses +0.08/+0.12 throttle increments with 0.133/0.267/0.533 normalized/s ramps at 8 m/s and 0.30/0.42 rad." \
    "race_domain_swerve_throttle_rate_factorial_{4p5,6p5,7p5}_validation repeat that crossed matrix at the named initial speed." \
    "race_domain_swerve_throttle_slew_up_frontier_validation pairs positive throttle steps/ramps during swerves at 9/10 m/s and measured moderate-steering cells." \
    "race_domain_swerve_throttle_rate_race_domain_{train,validation} crosses +0.04/+0.08 throttle changes and 0.15/0.30 s ramps at 4.5/6.5/7.5 m/s and 0.08/0.14/0.20 rad." \
    "Set SDU_APEX_EXPERIMENT_THROTTLE_RATE_SWEEP_DELTA_NORM to compare a different positive throttle increment." \
    "Use throttle_reset_smoke before the reset-isolated throttle transition surface." \
    "Throttle transition surface uses randomized, reset-isolated throttle transitions." \
    "race_domain_continuous records one uninterrupted 0–12 m/s speed/steering/braking sequence." \
    "race_domain_brake_boundary records legacy low-steering paired tests at 9–11.1 m/s." \
    "race_domain_moderate_braking adds 9–11.1 m/s turn-in, throttle reduction, active braking and release." \
    "race_domain_steering_frontier maps reset-isolated steering response at 9.5, 10.5, and 11.1 m/s." \
    "isolated_highsteer_75_long repeats the observed 7.5 m/s high-steering envelope with 8 s holds." \
    "isolated_highsteer_multispeed repeats that grid at 2.5, 4.5, and 6.5 m/s." \
    "race_domain_dynamic_steering records continuous signed steering transitions at 4.5, 6.5, and 7.5 m/s." \
    "race_domain_low_speed_highsteer_transients captures reset-matched 2.5/3.0/3.5 m/s high-steer turn-in, unwind, and reversal." \
    "yaw_atlas_interpolation_validation tests randomized, reset-matched off-grid speed/steering points twice in both turn directions." \
    "yaw_atlas_offgrid_final tests a separate frozen set of low-angle speed/steering points for final generalization scoring." \
    "yaw_atlas_extratrees_final tests six new off-grid points supported by the frozen ExtraTrees atlas, both turn directions and two reset-isolated repeats." \
    "yaw_atlas_extratrees_highsteer_final tests four new high-speed, high-steering off-grid points in supported steady-response cells, both directions and two reset-isolated repeats." \
    "yaw_highsteer_speed_surface_train fills the missing >=8 m/s, 0.35–0.50 rad yaw-response region at 8.25/8.75/9.25 m/s, both directions, two reset-isolated repeats." \
    "yaw_fullband_gapfill_train adds two independent signed high-speed low-angle captures at exact atlas cells that the sealed final test found unsupported or inaccurate." \
    "yaw_unwind_throttle_slew_train pairs a 0.12 throttle cut (one-tick step vs 0.30 s ramp) with 8.75 m/s signed steering unwind through ±0.025 rad." \
    "yaw_frontier_throttle_slew_train pairs step/ramp cuts at 9.5 m/s and 0.14/0.18/0.20 rad, matching the held-out high-speed deceleration maneuvers." \
    "yaw_frontier_lowangle_unwind_train repeats the 9.5 m/s paired cut four times per steering/sign condition and aligns low-angle unwind with the 8.5–9.0 m/s band." \
    "yaw_low_angle_rate_surface pairs one-tick steering steps with 0.30 s ramps over low steering and a selected 4.25/6.25/8.25/10.25 m/s speed." \
    "race_domain_dynamic_coupled_{train,validation,final} records randomized waveforms across the supported 5–11.1 m/s envelope." \
    "subnet_highsteer_transients records reset-isolated 7.5 m/s turn-in/unwind throttle transients at 0.30/0.42 rad." \
    "Resume a partial surface with SDU_APEX_EXPERIMENT_RESUME_ANALYSIS pointing to its closed analysis JSON." \
    "Set SDU_APEX_EXPERIMENT_STEERING_ANGLES_RAD to comma-separated steering angles in radians." \
    "Set SDU_APEX_EXPERIMENT_THROTTLE_REPEAT_COUNT, TARGET_STEP_PERCENT, and BASELINE_STEP_PERCENT." \
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
default_timeout_s=180
if [[ "${profile}" == race_domain_continuous ]]; then
  default_timeout_s=300
elif [[ "${profile}" == race_domain_brake_boundary ]]; then
  default_timeout_s=270
elif [[ "${profile}" == race_domain_moderate_braking ]]; then
  default_timeout_s=430
elif [[ "${profile}" == race_domain_dynamic_coupled_train ||
        "${profile}" == race_domain_dynamic_coupled_validation ||
        "${profile}" == race_domain_dynamic_coupled_final ]]; then
  default_timeout_s=600
elif [[ "${profile}" == race_domain_steering_frontier ]]; then
  default_timeout_s=1200
elif [[ "${profile}" == isolated_highsteer_75_long ]]; then
  default_timeout_s=240
elif [[ "${profile}" == isolated_highsteer_multispeed ]]; then
  default_timeout_s=720
elif [[ "${profile}" == subnet_highsteer_transients ]]; then
  default_timeout_s=600
elif [[ "${profile}" == race_domain_low_speed_highsteer_transients ]]; then
  default_timeout_s=600
elif [[ "${profile}" == race_domain_yaw_transition_transients ]]; then
  default_timeout_s=600
elif [[ "${profile}" == yaw_atlas_interpolation_validation ||
        "${profile}" == yaw_atlas_offgrid_final ||
        "${profile}" == yaw_atlas_extratrees_final ||
        "${profile}" == yaw_atlas_extratrees_highsteer_final ||
        "${profile}" == yaw_highsteer_speed_surface_train ||
        "${profile}" == yaw_fullband_gapfill_train ]]; then
  default_timeout_s=1200
elif [[ "${profile}" == yaw_low_angle_rate_surface ]]; then
  default_timeout_s=1200
elif [[ "${profile}" == yaw_unwind_throttle_slew_train ]]; then
  default_timeout_s=600
elif [[ "${profile}" == yaw_frontier_throttle_slew_train ]]; then
  default_timeout_s=600
elif [[ "${profile}" == yaw_frontier_lowangle_unwind_train ]]; then
  default_timeout_s=1200
elif [[ "${profile}" == race_domain_swerve_throttle_slew_train ||
        "${profile}" == race_domain_swerve_throttle_slew_validation ||
        "${profile}" == race_domain_swerve_throttle_slew_frontier_validation ||
        "${profile}" == race_domain_swerve_throttle_slew_11mps_replication ||
        "${profile}" == race_domain_swerve_throttle_slew_moderate_validation ||
        "${profile}" == race_domain_swerve_throttle_slew_lowsteer_validation ||
        "${profile}" == race_domain_swerve_throttle_rate_sweep_validation ||
        "${profile}" == race_domain_swerve_throttle_rate_sweep_highsteer_validation ||
        "${profile}" == race_domain_swerve_throttle_rate_factorial_validation ||
        "${profile}" == race_domain_swerve_throttle_rate_factorial_4p5_validation ||
        "${profile}" == race_domain_swerve_throttle_rate_factorial_6p5_validation ||
        "${profile}" == race_domain_swerve_throttle_rate_factorial_7p5_validation ||
        "${profile}" == race_domain_swerve_throttle_slew_up_frontier_train ||
        "${profile}" == race_domain_swerve_throttle_slew_up_frontier_validation ||
        "${profile}" == race_domain_swerve_throttle_rate_race_domain_train ||
        "${profile}" == race_domain_swerve_throttle_rate_race_domain_validation ]]; then
  default_timeout_s=1200
fi
timeout_s="${SDU_APEX_EXPERIMENT_TIMEOUT_S:-${default_timeout_s}}"
capture_sensor_odom="${SDU_APEX_EXPERIMENT_CAPTURE_SENSOR_ODOM:-0}"
if [[ "${capture_sensor_odom}" != 0 && "${capture_sensor_odom}" != 1 ]]; then
  echo "SDU_APEX_EXPERIMENT_CAPTURE_SENSOR_ODOM must be 0 or 1." >&2
  exit 2
fi
steering_angles_rad="${SDU_APEX_EXPERIMENT_STEERING_ANGLES_RAD:-0.0}"
throttle_target_step_percent="${SDU_APEX_EXPERIMENT_THROTTLE_TARGET_STEP_PERCENT:-5}"
throttle_baseline_step_percent="${SDU_APEX_EXPERIMENT_THROTTLE_BASELINE_STEP_PERCENT:-10}"
throttle_repeat_count="${SDU_APEX_EXPERIMENT_THROTTLE_REPEAT_COUNT:-2}"
smoke_throttle="${SDU_APEX_EXPERIMENT_SMOKE_THROTTLE:-1.0}"
smoke_hold_s="${SDU_APEX_EXPERIMENT_SMOKE_HOLD_S:-5.0}"
dev_sim_reset_enabled=0
if [[ "${profile}" == throttle_reset_smoke ||
      "${profile}" == throttle_transition_surface ||
      "${profile}" == race_domain_steering_frontier ||
      "${profile}" == race_domain_dynamic_coupled_train ||
      "${profile}" == race_domain_dynamic_coupled_validation ||
      "${profile}" == race_domain_dynamic_coupled_final ||
      "${profile}" == race_domain_swerve_throttle_slew_train ||
      "${profile}" == race_domain_swerve_throttle_slew_validation ||
      "${profile}" == race_domain_swerve_throttle_slew_frontier_validation ||
      "${profile}" == race_domain_swerve_throttle_slew_11mps_replication ||
      "${profile}" == race_domain_swerve_throttle_slew_moderate_validation ||
      "${profile}" == race_domain_swerve_throttle_slew_lowsteer_validation ||
      "${profile}" == race_domain_swerve_throttle_rate_sweep_validation ||
      "${profile}" == race_domain_swerve_throttle_rate_sweep_highsteer_validation ||
      "${profile}" == race_domain_swerve_throttle_rate_factorial_validation ||
      "${profile}" == race_domain_swerve_throttle_rate_factorial_4p5_validation ||
      "${profile}" == race_domain_swerve_throttle_rate_factorial_6p5_validation ||
      "${profile}" == race_domain_swerve_throttle_rate_factorial_7p5_validation ||
      "${profile}" == race_domain_swerve_throttle_slew_up_frontier_train ||
      "${profile}" == race_domain_swerve_throttle_slew_up_frontier_validation ||
      "${profile}" == race_domain_swerve_throttle_rate_race_domain_train ||
      "${profile}" == race_domain_swerve_throttle_rate_race_domain_validation ||
      "${profile}" == race_domain_low_speed_highsteer_transients ||
      "${profile}" == race_domain_yaw_transition_transients ||
      "${profile}" == yaw_atlas_interpolation_validation ||
      "${profile}" == yaw_atlas_offgrid_final ||
      "${profile}" == yaw_atlas_extratrees_final ||
      "${profile}" == yaw_atlas_extratrees_highsteer_final ||
      "${profile}" == yaw_highsteer_speed_surface_train ||
      "${profile}" == yaw_fullband_gapfill_train ||
      "${profile}" == yaw_unwind_throttle_slew_train ||
      "${profile}" == yaw_frontier_throttle_slew_train ||
      "${profile}" == yaw_frontier_lowangle_unwind_train ||
      "${profile}" == yaw_low_angle_rate_surface ||
      "${profile}" == subnet_highsteer_transients ]]; then
  dev_sim_reset_enabled=1
fi
probe_dwell_s="${SDU_APEX_EXPERIMENT_PROBE_DWELL_S:-0}"
speed_hold_kp="${SDU_APEX_EXPERIMENT_SPEED_HOLD_KP:-0.04}"
speed_hold_ki="${SDU_APEX_EXPERIMENT_SPEED_HOLD_KI:-0.0}"
speed_median_gate_mps="${SDU_APEX_EXPERIMENT_SPEED_MEDIAN_GATE_MPS:-0.12}"
speed_p95_gate_mps="${SDU_APEX_EXPERIMENT_SPEED_P95_GATE_MPS:-0.20}"
throttle_rate_sweep_delta_norm="${SDU_APEX_EXPERIMENT_THROTTLE_RATE_SWEEP_DELTA_NORM:-0.08}"
domain_id="${SDU_APEX_EXPERIMENT_DOMAIN_ID:-61}"
resume_analysis_input="${SDU_APEX_EXPERIMENT_RESUME_ANALYSIS:-}"
resume_analysis_container=""
run_id="${SDU_APEX_EXPERIMENT_RUN_ID:-openplane_${profile}_$(date +%Y%m%d_%H%M%S)}"
container="${SDU_APEX_EXPERIMENT_CONTAINER:-sdu_apex_openplane_${run_id}}"
run_parent="${repo_root}/live_runs/${run_id}"

if [[ ! "${run_id}" =~ ^[[:alnum:]_-]+$ ]]; then
  echo "Run ID may contain only letters, digits, underscores, and hyphens." >&2
  exit 2
fi
case "${profile}" in
  high_angle_boundary|isolated_boundary|isolated_speed_sweep|isolated_force_3mps|\
  isolated_force_4mps|isolated_force_5mps|isolated_highspeed_surface|\
  isolated_highsteer_75_long|race_domain_dynamic_steering|\
  isolated_highsteer_multispeed|\
  race_domain_dynamic_coupled_train|race_domain_dynamic_coupled_validation|\
  race_domain_dynamic_coupled_final|\
  race_domain_low_speed_highsteer_transients|\
  race_domain_yaw_transition_transients|\
  yaw_atlas_interpolation_validation|\
  yaw_atlas_offgrid_final|\
  yaw_atlas_extratrees_final|\
  yaw_atlas_extratrees_highsteer_final|\
  yaw_highsteer_speed_surface_train|\
  yaw_fullband_gapfill_train|\
  yaw_unwind_throttle_slew_train|\
  yaw_frontier_throttle_slew_train|\
  yaw_frontier_lowangle_unwind_train|\
  yaw_low_angle_rate_surface|\
  race_domain_swerve_throttle_slew_train|\
  race_domain_swerve_throttle_slew_validation|\
  race_domain_swerve_throttle_slew_frontier_validation|\
  race_domain_swerve_throttle_slew_11mps_replication|\
  race_domain_swerve_throttle_slew_moderate_validation|subnet_highsteer_transients|\
  race_domain_swerve_throttle_slew_lowsteer_validation|\
  race_domain_swerve_throttle_rate_sweep_validation|\
  race_domain_swerve_throttle_rate_sweep_highsteer_validation|\
  race_domain_swerve_throttle_rate_factorial_validation|\
  race_domain_swerve_throttle_rate_factorial_4p5_validation|\
  race_domain_swerve_throttle_rate_factorial_6p5_validation|\
  race_domain_swerve_throttle_rate_factorial_7p5_validation|\
  race_domain_swerve_throttle_slew_up_frontier_train|\
  race_domain_swerve_throttle_slew_up_frontier_validation|\
  race_domain_swerve_throttle_rate_race_domain_train|\
  race_domain_swerve_throttle_rate_race_domain_validation|\
  isolated_3to5_response_surface|\
  isolated_highspeed_crossfactor|isolated_highspeed_tail|\
  isolated_transition_65mps|isolated_transition_45mps|\
  isolated_transition_speed_surface|isolated_transition_support|\
  isolated_transition_bridge|isolated_transition_low_support|\
  isolated_transition_full_surface|transient_4mps|transient_fullsteer_4mps|\
  transient_transition_4mps|transient_transition_4mps_fixedthrottle|\
  transient_transition_dwell_4mps_fixedthrottle|throttle_slew_pair|\
  throttle_reset_smoke|throttle_transition_surface|full_input_excitation|\
  race_domain_continuous|race_domain_brake_boundary|\
  race_domain_moderate_braking|race_domain_steering_frontier|grid) ;;
  *)
  echo "Invalid SDU_APEX_EXPERIMENT_PROFILE; run $0 --help for usage." >&2
  exit 2
    ;;
esac
if [[ -n "${resume_analysis_input}" ]]; then
  if [[ "${profile}" != throttle_transition_surface ]]; then
    echo "Resume analysis is supported only for throttle_transition_surface." >&2
    exit 2
  fi
  if [[ "${resume_analysis_input}" == /* ]]; then
    resume_analysis_host="${resume_analysis_input}"
  else
    resume_analysis_host="${repo_root}/${resume_analysis_input}"
  fi
  resume_analysis_host="$(realpath -- "${resume_analysis_host}")"
  case "${resume_analysis_host}" in
    "${repo_root}"/*) ;;
    *) echo "Resume analysis must be inside the workspace." >&2; exit 2 ;;
  esac
  if [[ ! -s "${resume_analysis_host}" ]]; then
    echo "Resume analysis is missing or empty: ${resume_analysis_host}" >&2
    exit 2
  fi
  resume_analysis_relative="${resume_analysis_host#"${repo_root}"/}"
  resume_analysis_container="/workspace/src/${resume_analysis_relative}"
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
  -e "CAPTURE_SENSOR_ODOM=${capture_sensor_odom}" \
  -e "EXPERIMENT_STEERING_ANGLES_RAD=${steering_angles_rad}" \
  -e "EXPERIMENT_THROTTLE_TARGET_STEP_PERCENT=${throttle_target_step_percent}" \
  -e "EXPERIMENT_THROTTLE_BASELINE_STEP_PERCENT=${throttle_baseline_step_percent}" \
  -e "EXPERIMENT_THROTTLE_REPEAT_COUNT=${throttle_repeat_count}" \
  -e "EXPERIMENT_SMOKE_THROTTLE=${smoke_throttle}" \
  -e "EXPERIMENT_SMOKE_HOLD_S=${smoke_hold_s}" \
  -e "EXPERIMENT_PROBE_DWELL_S=${probe_dwell_s}" \
  -e "SDU_APEX_DEV_SIM_RESET_ENABLED=${dev_sim_reset_enabled}" \
  -e "EXPERIMENT_SPEED_HOLD_KP=${speed_hold_kp}" \
  -e "EXPERIMENT_SPEED_HOLD_KI=${speed_hold_ki}" \
  -e "EXPERIMENT_SPEED_MEDIAN_GATE_MPS=${speed_median_gate_mps}" \
  -e "EXPERIMENT_SPEED_P95_GATE_MPS=${speed_p95_gate_mps}" \
  -e "EXPERIMENT_THROTTLE_RATE_SWEEP_DELTA_NORM=${throttle_rate_sweep_delta_norm}" \
  -e "EXPERIMENT_RESUME_ANALYSIS=${resume_analysis_container}" \
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
    experiment_pid=""

    stop_experiment() {
      [[ -n "${experiment_pid}" ]] || return 0
      if kill -0 "${experiment_pid}" 2>/dev/null; then
        kill -INT "${experiment_pid}" 2>/dev/null || true
        for attempt in $(seq 1 50); do
          kill -0 "${experiment_pid}" 2>/dev/null || break
          sleep 0.1
        done
        if kill -0 "${experiment_pid}" 2>/dev/null; then
          kill -TERM "${experiment_pid}" 2>/dev/null || true
          for attempt in $(seq 1 20); do
            kill -0 "${experiment_pid}" 2>/dev/null || break
            sleep 0.1
          done
        fi
        if kill -0 "${experiment_pid}" 2>/dev/null; then
          kill -KILL "${experiment_pid}" 2>/dev/null || true
        fi
      fi
      wait "${experiment_pid}" 2>/dev/null || true
      experiment_pid=""
    }

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
      stop_experiment
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
      /open_plane_experiment/phase
    )
    if [[ "${CAPTURE_SENSOR_ODOM}" == 1 ]]; then
      odom_topics_ready=0
      for attempt in $(seq 1 100); do
        topics="$(ros2 topic list --no-daemon 2>/dev/null || true)"
        if grep -Fxq /explore_sensor_odom <<<"${topics}" &&
           grep -Fxq /explore_sensor_odom/diagnostics <<<"${topics}"; then
          odom_topics_ready=1
          break
        fi
        sleep 0.1
      done
      if [[ "${odom_topics_ready}" != 1 ]]; then
        echo "Requested sensor-odometry capture, but its sidecar topics were not discovered." >&2
        exit 1
      fi
      record_topics+=(/explore_sensor_odom /explore_sensor_odom/diagnostics)
    fi
    if [[ "${SDU_APEX_DEV_SIM_RESET_ENABLED}" == 1 ]]; then
      record_topics+=(/autodrive/reset_command)
    fi
    ros2 bag record --max-cache-size 268435456 -o "${rosbag_output}" \
      "${record_topics[@]}" \
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
    if [[ "${PROFILE}" == throttle_reset_smoke ]]; then
      python3 /workspace/src/tools/open_plane_reset_smoke.py \
        --throttle "${EXPERIMENT_SMOKE_THROTTLE}" \
        --hold-s "${EXPERIMENT_SMOKE_HOLD_S}" \
        >"${experiment_log}" 2>&1 &
    elif [[ "${PROFILE}" == throttle_transition_surface ]]; then
      resume_args=()
      if [[ -n "${EXPERIMENT_RESUME_ANALYSIS:-}" ]]; then
        resume_args=(--resume-analysis "${EXPERIMENT_RESUME_ANALYSIS}")
      fi
      python3 /workspace/src/tools/open_plane_throttle_transition_surface.py \
        "${resume_args[@]}" \
        --seed "${EXPERIMENT_SEED}" \
        --steering-angles-rad="${EXPERIMENT_STEERING_ANGLES_RAD}" \
        --target-step-percent "${EXPERIMENT_THROTTLE_TARGET_STEP_PERCENT}" \
        --baseline-step-percent "${EXPERIMENT_THROTTLE_BASELINE_STEP_PERCENT}" \
        --repeat-count "${EXPERIMENT_THROTTLE_REPEAT_COUNT}" \
        >"${experiment_log}" 2>&1 &
    else
      python3 /workspace/src/tools/open_plane_excitation.py \
        --profile "${PROFILE}" --seed "${EXPERIMENT_SEED}" \
        --transition-speed-mps "${EXPERIMENT_SPEED_MPS}" \
        --probe-dwell-s "${EXPERIMENT_PROBE_DWELL_S}" \
        --speed-hold-kp "${EXPERIMENT_SPEED_HOLD_KP}" \
        --speed-hold-ki "${EXPERIMENT_SPEED_HOLD_KI}" \
        --speed-median-gate-mps "${EXPERIMENT_SPEED_MEDIAN_GATE_MPS}" \
        --speed-p95-gate-mps "${EXPERIMENT_SPEED_P95_GATE_MPS}" \
        --throttle-rate-sweep-delta-norm "${EXPERIMENT_THROTTLE_RATE_SWEEP_DELTA_NORM}" \
        --timeout-s "${EXPERIMENT_TIMEOUT_S}" \
        >"${experiment_log}" 2>&1 &
    fi
    experiment_pid=$!
    while kill -0 "${experiment_pid}" 2>/dev/null; do
      if ! kill -0 "${recorder_pid}" 2>/dev/null; then
        echo "Recorder exited during excitation; requesting a safe stop." >&2
        kill -INT "${experiment_pid}" 2>/dev/null || true
        break
      fi
      sleep 0.5
    done
    wait "${experiment_pid}"
    experiment_status=$?
    experiment_pid=""
    if ! kill -0 "${recorder_pid}" 2>/dev/null; then
      echo "Recorder did not remain alive for the complete excitation." >&2
      experiment_status=1
    fi
    set -e

    cat "${experiment_log}"
    stop_recorder
    stop_bridge
    if [[ "${PROFILE}" == throttle_transition_surface ]]; then
      analysis_log="${run_parent}/analysis.log"
      if ! PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}" \
          python3 /workspace/src/tools/analyze_open_plane_throttle_transitions.py \
          "${db_file}" >"${analysis_log}" 2>&1; then
        experiment_status=1
      fi
      cat "${analysis_log}"
    fi
    echo "Bag closed at ${rosbag_output}; driver exit status=${experiment_status}"
    exit "${experiment_status}"
  '
