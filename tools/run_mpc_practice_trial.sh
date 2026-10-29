#!/usr/bin/env bash
# Supervised full-speed practice trial: warmup plus scored laps (five default).
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"
source tools/docker_env.sh
if [[ $# -ne 1 ]]; then
  echo "Usage: run_mpc_practice_trial.sh RUN_ID" >&2
  exit 2
fi
run_id="$1"
[[ "$run_id" =~ ^[[:alnum:]_-]+$ ]] || exit 2
run_dir="live_runs/$run_id"
[[ ! -e "$run_dir" ]] || { echo "Run already exists: $run_dir" >&2; exit 2; }
trajectory=f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv
total_laps="${SDU_APEX_TRIAL_TOTAL_LAPS:-6}"
[[ "$total_laps" =~ ^[0-9]+$ && "$total_laps" -ge 2 ]] || {
  echo 'SDU_APEX_TRIAL_TOTAL_LAPS must include warmup plus >=1 scored lap' >&2
  exit 2
}
map=f1tenth_planning/maps/autodrive_practice_20260924_b.yaml
controller="$run_id"_controller
simulator="$run_id"_sim
recorder="rec_$run_id"
mkdir -p "$run_dir"
set -- --config mpc=f1tenth_mpc/config/mpc_practice_fastline.yaml
set -- "$@" --config amcl=f1tenth_localization/config/gpu_amcl_cpp_params.yaml
cp -- f1tenth_mpc/config/mpc_practice_fastline.yaml "$run_dir/mpc_parameters.yaml"
cp -- f1tenth_localization/config/gpu_amcl_cpp_params.yaml "$run_dir/amcl_parameters.yaml"
amcl_overlay="${SDU_APEX_AMCL_PARAMETER_OVERLAY:-}"
if [[ -n "$amcl_overlay" ]]; then
  [[ -f "$amcl_overlay" ]] || { echo 'Missing AMCL overlay' >&2; exit 2; }
  cp -- "$amcl_overlay" "$run_dir/amcl_overlay.yaml"
  set -- "$@" --config "amcl_overlay=$amcl_overlay"
fi
python3 tools/seal_mpc_run_manifest.py --run-id "$run_id" \
  --output "$run_dir/run_manifest.json" --acknowledge-not-started \
  --track practice --unity-mode batchmode \
  --simulator-image autodriveecosystem/autodrive_roboracer_sim@sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76 \
  --controller-image sdu-apex-autodrive:dev \
  --simulator-container "$simulator" --controller-container "$controller" \
  --map "$map" --trajectory "$trajectory" "$@" \
  --initial-pose '{"x":0.8,"y":3.1592,"yaw":-1.5708251687330668}' \
  --supervisor "tools/watch_sim_run.sh $controller $simulator $recorder mpc-lap-$total_laps 180" \
  --diagnostics on --ros-executor-threads 4 --openmp-threads 8 --jacobian-workers 8

cleanup() {
  docker stop --timeout 0 "$simulator" >/dev/null 2>&1 || true
  docker kill --signal SIGINT "$recorder" "$controller" >/dev/null 2>&1 || true
}
trap cleanup EXIT
env -u SDU_APEX_MPC_PARAMETER_OVERLAY AUTODRIVE_REBUILD=0 \
  SDU_APEX_CONTAINER="$controller" \
  SDU_APEX_CPUSET_CPUS=0-11 SDU_APEX_MPC_PUBLISH_DIAGNOSTICS=true \
  SDU_APEX_MPC_START_DELAY_SEC=2 SDU_APEX_DEV_SIM_RESET_ENABLED=true \
  SDU_APEX_MAP_YAML="$map" SDU_APEX_TRAJECTORY_FILE="$trajectory" \
  tools/start_dev.sh >"$run_dir/controller.log" 2>&1 &
controller_job=$!
env SDU_APEX_RUN_ID="$run_id" SDU_APEX_RECORD_PROFILE=mpc_performance \
  tools/record_debug_bag.sh >"$run_dir/recorder.log" 2>&1 &
recorder_job=$!
wait_log() {
  local log_file="$1" pattern="$2"
  for _ in $(seq 1 180); do
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
  tools/watch_sim_run.sh "$controller" "$simulator" "$recorder" \
  "mpc-lap-$total_laps" 180 >"$run_dir/watcher.log" 2>&1 &
watcher_job=$!
wait_log "$run_dir/watcher.log" WATCHER_READY
# Reset suppresses sensor forwarding, so live modules receive a clean spawn
# state when the pulse ends rather than a moving pre-reset sample.
docker exec "$controller" bash -lc \
  'source /opt/ros/humble/setup.bash; timeout 10s ros2 topic pub -r 5 /autodrive/reset_command std_msgs/msg/Bool "{data: true}" >/dev/null; ros2 topic pub --once /autodrive/reset_command std_msgs/msg/Bool "{data: false}" >/dev/null' \
  >"$run_dir/reset.log" 2>&1 &
reset_job=$!
wait_log "$run_dir/controller.log" 'reset ROS callback: True'
env SDU_APEX_SIM_TRACK=practice SDU_APEX_SIM_MODE=batchmode \
  SDU_APEX_SIM_CONTAINER="$simulator" SDU_APEX_SIM_CPUSET_CPUS=12-13 \
  tools/start_simulator.sh >"$run_dir/simulator.log" 2>&1 &
simulator_job=$!
set +e
wait "$watcher_job"
trial_status=$?
wait "$controller_job" "$recorder_job" "$simulator_job" "$reset_job" 2>/dev/null
set -e
echo "Trial $run_id: supervisor exit $trial_status; $run_dir"
exit "$trial_status"
