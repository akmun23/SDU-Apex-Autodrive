#!/usr/bin/env bash
# Supervised closed-loop MPC trial against the Unity simulator.
#
#   tools/sim/run_trial.sh {practice|competition} RUN_ID
#
# The controller is built from the bind-mounted working tree inside the
# development image, so every run exercises the current source. Results are
# written to live_runs/RUN_ID (ignored by git).
#
# Optional environment:
#   SDU_APEX_TRIAL_TOTAL_LAPS         warm-up lap plus scored laps (default 6)
#   SDU_APEX_TRIAL_TRAJECTORY_FILE    raceline override for either track (inside the repository)
#   SDU_APEX_TRIAL_WALL_TIMEOUT_S     supervisor wall-clock limit (default 240)
#   SDU_APEX_MPC_PARAMETER_OVERLAY    MPC YAML overlay inside the repository
#   SDU_APEX_MPC_NATIVE_TUNING / SDU_APEX_MPC_LTO   "true" for -march=native/LTO
#   SDU_APEX_MPC_HORIZON             prediction stages of 25 ms (default 30)
#   SDU_APEX_TRIAL_CPU_ISOLATION     "false" disables the detected CPU profile
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$repo_root"
source tools/docker_env.sh
if [[ $# -ne 2 ]]; then
  echo "Usage: run_trial.sh {practice|competition} RUN_ID" >&2
  exit 2
fi
track="$1"
run_id="$2"
[[ "$run_id" =~ ^[[:alnum:]_-]+$ ]] || exit 2
case "$track" in
  practice)
    map=f1tenth_planning/maps/practice.yaml
    trajectory=${SDU_APEX_TRIAL_TRAJECTORY_FILE:-f1tenth_planning/trajectories/practice/raceline.csv}
    mpc_config=f1tenth_mpc/config/mpc_practice.yaml
    amcl_config=f1tenth_localization/config/gpu_amcl_cpp_params.yaml
    amcl_overlay=
    initial_pose='{"x":0.8,"y":3.1592,"yaw":-1.5708251687330668}'
    min_lap_distance_m=0
    ;;
  competition)
    map=f1tenth_planning/maps/competition.yaml
    trajectory=${SDU_APEX_TRIAL_TRAJECTORY_FILE:-f1tenth_planning/trajectories/competition/raceline.csv}
    mpc_config=f1tenth_mpc/config/mpc_competition.yaml
    amcl_config=f1tenth_localization/config/amcl_iros.yaml
    amcl_overlay=f1tenth_localization/config/amcl_iros.yaml
    initial_pose='{"x":-0.4476,"y":-3.7823,"yaw":-1.5471}'
    min_lap_distance_m=25
    ;;
  *) echo "Unknown track: $track" >&2; exit 2 ;;
esac
# This profile belongs to the trial supervisor, which owns all three containers.
# Reserve complete physical cores; never assume this numbering on another host.
trial_cpu_topology_supported() {
  [[ -r /sys/devices/system/cpu/online &&
     "$(</sys/devices/system/cpu/online)" == 0-15 ]] || return 1
  local cpu siblings
  for cpu in 0 2 4 6 8 10 12 13 14 15; do
    siblings="/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list"
    [[ -r "$siblings" ]] || return 1
    if (( cpu < 12 )); then
      [[ "$(<"$siblings")" == "$cpu-$((cpu + 1))" ]] || return 1
    else
      [[ "$(<"$siblings")" == "$cpu" ]] || return 1
    fi
  done
}
case "${SDU_APEX_TRIAL_CPU_ISOLATION:-true}" in
  true)
    if trial_cpu_topology_supported; then
      export SDU_APEX_CPUSET_CPUS="${SDU_APEX_CPUSET_CPUS-0,2,4,6,8-11}"
      export SDU_APEX_MPC_RUNTIME_AFFINITY_CPUS="${SDU_APEX_MPC_RUNTIME_AFFINITY_CPUS-9,11}"
      export SDU_APEX_MPC_CPUSET_CPUS="${SDU_APEX_MPC_CPUSET_CPUS-0,2,4,6,8,10}"
      export SDU_APEX_MPC_JACOBIAN_THREADS="${SDU_APEX_MPC_JACOBIAN_THREADS-6}"
      export SDU_APEX_SIM_CPUSET_CPUS="${SDU_APEX_SIM_CPUSET_CPUS-12-15}"
      export SDU_APEX_SIM_RUNTIME_AFFINITY_CPUS="${SDU_APEX_SIM_RUNTIME_AFFINITY_CPUS-$SDU_APEX_SIM_CPUSET_CPUS}"
      export SDU_APEX_RECORDER_CPUSET_CPUS="${SDU_APEX_RECORDER_CPUSET_CPUS-12-15}"
      export OMP_NUM_THREADS="${OMP_NUM_THREADS-6}"
      export OMP_PROC_BIND="${OMP_PROC_BIND-spread}"
      if [[ ! -v OMP_PLACES ]]; then
        export OMP_PLACES='{0},{2},{4},{6},{8},{10}'
      fi
      export OMP_WAIT_POLICY="${OMP_WAIT_POLICY-active}"
      export GOMP_SPINCOUNT="${GOMP_SPINCOUNT-1000000}"
    fi
    ;;
  false) ;;
  *) echo 'SDU_APEX_TRIAL_CPU_ISOLATION must be true or false' >&2; exit 2 ;;
esac
controller_cpuset="${SDU_APEX_TRIAL_CONTROLLER_CPUSET_CPUS:-${SDU_APEX_CPUSET_CPUS:-0-11}}"
runtime_affinity="${SDU_APEX_TRIAL_RUNTIME_AFFINITY_CPUS:-${SDU_APEX_MPC_RUNTIME_AFFINITY_CPUS:-}}"
# The Unity bridge, ROS topics and 40 Hz controller are one shared simulation
# domain; concurrent trials would corrupt each other's command epochs.
exec {trial_lock_fd}>"${TMPDIR:-/tmp}/sdu_apex_trial.lock"
if ! flock -n "$trial_lock_fd"; then
  echo "Another trial already owns the simulator/ROS domain" >&2
  exit 73
fi
mkdir -p live_runs && touch live_runs/COLCON_IGNORE
run_dir="live_runs/$run_id"
[[ ! -e "$run_dir" ]] || { echo "Run already exists: $run_dir" >&2; exit 2; }
total_laps="${SDU_APEX_TRIAL_TOTAL_LAPS:-6}"
[[ "$total_laps" =~ ^[0-9]+$ && "$total_laps" -ge 2 ]] || {
  echo 'SDU_APEX_TRIAL_TOTAL_LAPS must include warm-up plus >=1 scored lap' >&2
  exit 2
}
controller="${run_id}_controller"
simulator="${run_id}_sim"
recorder="rec_${run_id}"
mkdir -p "$run_dir"
set -- --config "mpc=$mpc_config" --config "amcl=$amcl_config"
cp -- "$mpc_config" "$run_dir/mpc_parameters.yaml"
cp -- "$amcl_config" "$run_dir/amcl_parameters.yaml"
mpc_overlay="${SDU_APEX_MPC_PARAMETER_OVERLAY:-}"
if [[ -n "$mpc_overlay" ]]; then
  [[ -f "$mpc_overlay" ]] || { echo 'Missing MPC overlay' >&2; exit 2; }
  cp -- "$mpc_overlay" "$run_dir/mpc_overlay.yaml"
  set -- "$@" --config "overlay=$mpc_overlay"
fi
# Record the actual worker parameter, including the runtime default and overlay.
jacobian_workers="$(python3 - "$mpc_config" "$mpc_overlay" <<'PY'
import os, sys, yaml
from pathlib import Path
from tools.analysis.write_run_manifest import validate_mpc_overlay
parameters = yaml.safe_load(Path(sys.argv[1]).read_text())["/**"]["ros__parameters"]
if os.environ.get("SDU_APEX_MPC_JACOBIAN_THREADS"):
    parameters["jacobian_threads"] = int(os.environ["SDU_APEX_MPC_JACOBIAN_THREADS"])
if sys.argv[2]:
    validate_mpc_overlay(Path(sys.argv[2]))
    for entry in yaml.safe_load(Path(sys.argv[2]).read_text()).values():
        parameters.update(entry["ros__parameters"])
print(parameters["jacobian_threads"])
PY
)"
{
  printf 'controller_container=%s\nros_background=%s\nmpc=%s\nsimulator=%s\nrecorder=%s\njacobian_workers=%s\n' \
    "$controller_cpuset" "$runtime_affinity" "${SDU_APEX_MPC_CPUSET_CPUS:-}" \
    "${SDU_APEX_SIM_RUNTIME_AFFINITY_CPUS:-}" "${SDU_APEX_RECORDER_CPUSET_CPUS:-}" "$jacobian_workers"
  env | sort | rg '^(OMP_|GOMP_)' || true
} > "$run_dir/cpu_isolation.txt"
python3 tools/analysis/write_run_manifest.py --run-id "$run_id" \
  --output "$run_dir/run_manifest.json" --acknowledge-not-started \
  --track "$track" --unity-mode batchmode \
  --simulator-image autodriveecosystem/autodrive_roboracer_sim@sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76 \
  --controller-image "${SDU_APEX_IMAGE:-sdu-apex-autodrive:dev}" \
  --simulator-container "$simulator" --controller-container "$controller" \
  --map "$map" --trajectory "$trajectory" "$@" \
  --initial-pose "$initial_pose" \
  --supervisor "tools/sim/watch_sim_run.sh $controller $simulator $recorder mpc-lap-$total_laps 180" \
  --diagnostics on --ros-executor-threads 4 \
  --openmp-threads "${OMP_NUM_THREADS:-8}" \
  --jacobian-workers "$jacobian_workers"

cleanup() {
  docker stop --timeout 0 "$simulator" >/dev/null 2>&1 || true
  docker kill --signal SIGINT "$recorder" "$controller" >/dev/null 2>&1 || true
}
trap cleanup EXIT
env AUTODRIVE_REBUILD=0 SDU_APEX_MPC_PARAMETER_OVERLAY="$mpc_overlay" \
  SDU_APEX_AMCL_PARAMETER_OVERLAY="$amcl_overlay" \
  SDU_APEX_CONTAINER="$controller" \
  SDU_APEX_CPUSET_CPUS="$controller_cpuset" \
  SDU_APEX_MPC_RUNTIME_AFFINITY_CPUS="$runtime_affinity" \
  SDU_APEX_MPC_PUBLISH_DIAGNOSTICS=true \
  SDU_APEX_MPC_START_DELAY_SEC=2 SDU_APEX_DEV_SIM_RESET_ENABLED=true \
  SDU_APEX_MAP_YAML="$map" SDU_APEX_TRAJECTORY_FILE="$trajectory" \
  tools/sim/start_dev.sh >"$run_dir/controller.log" 2>&1 &
controller_job=$!
env SDU_APEX_RUN_ID="$run_id" SDU_APEX_RECORD_PROFILE=mpc_performance \
  tools/sim/record_debug_bag.sh >"$run_dir/recorder.log" 2>&1 &
recorder_job=$!
wait_log() {
  local log_file="$1" pattern="$2"
  for _ in $(seq 1 240); do
    if [[ -f "$log_file" ]] && rg -q "$pattern" "$log_file"; then return; fi
    kill -0 "$controller_job" && kill -0 "$recorder_job" || return 1
    sleep 1
  done
  echo "Timed out waiting for $pattern in $log_file" >&2
  return 1
}
wait_log "$run_dir/controller.log" 'Loaded MPC trajectory'
wait_log "$run_dir/recorder.log" Recording
env SDU_APEX_WATCH_STALL_TIMEOUT_S=30 \
  SDU_APEX_MPC_RUNTIME_AFFINITY_CPUS="$runtime_affinity" \
  SDU_APEX_WATCH_MIN_DISTANCE_PER_LAP_M="$min_lap_distance_m" \
  tools/sim/watch_sim_run.sh "$controller" "$simulator" "$recorder" \
  "mpc-lap-$total_laps" "${SDU_APEX_TRIAL_WALL_TIMEOUT_S:-240}" >"$run_dir/watcher.log" 2>&1 &
watcher_job=$!
wait_log "$run_dir/watcher.log" WATCHER_READY
# A reset pulse makes the live modules start from a clean spawn state.
reset_prefix=()
if [[ -n "$runtime_affinity" ]]; then
  reset_prefix=(taskset --cpu-list "$runtime_affinity")
fi
docker exec "$controller" "${reset_prefix[@]}" bash -lc \
  'source /opt/ros/humble/setup.bash; timeout 10s ros2 topic pub -r 5 /autodrive/reset_command std_msgs/msg/Bool "{data: true}" >/dev/null; ros2 topic pub --once /autodrive/reset_command std_msgs/msg/Bool "{data: false}" >/dev/null' \
  >"$run_dir/reset.log" 2>&1 &
reset_job=$!
wait_log "$run_dir/controller.log" 'reset ROS callback: True'
sim_track=practice
[[ "$track" == competition ]] && sim_track=iros
env SDU_APEX_SIM_TRACK="$sim_track" SDU_APEX_SIM_MODE=batchmode \
  SDU_APEX_SIM_CONTAINER="$simulator" SDU_APEX_SIM_CPUSET_CPUS="${SDU_APEX_SIM_CPUSET_CPUS:-12-15}" \
  tools/sim/start_simulator.sh >"$run_dir/simulator.log" 2>&1 &
simulator_job=$!
set +e
wait "$watcher_job"
trial_status=$?
wait "$controller_job" "$recorder_job" "$simulator_job" "$reset_job" 2>/dev/null
set -e
echo "Trial $run_id ($track): supervisor exit $trial_status; $run_dir"
exit "$trial_status"
