#!/usr/bin/env bash
# Validate the IROS map/raceline with the production sensor/localization/MPC stack.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source tools/docker_env.sh
run_id="${1:-iros_mpc_$(date +%Y%m%d_%H%M%S)}"
[[ "$run_id" =~ ^[[:alnum:]_-]+$ ]] || exit 2
run_dir="live_runs/$run_id"
[[ ! -e "$run_dir" ]] || { echo "Run already exists: $run_dir" >&2; exit 2; }
map=f1tenth_planning/maps/autodrive_iros_2026_compete.yaml
trajectory=${SDU_APEX_IROS_TRAJECTORY_FILE:-f1tenth_planning/trajectories/autodrive_iros_2026_compete/initial_raceline.csv}
trajectory="$(realpath -- "$trajectory")"
[[ "$trajectory" == "$repo_root/"* ]] || {
  echo "Trajectory must resolve inside the repository" >&2; exit 2;
}
trajectory_container="/workspace/src/${trajectory#"$repo_root/"}"
mpc_profile="${SDU_APEX_IROS_MPC_PARAMETER_FILE:-}"
mpc_profile_container=""
if [[ -n "$mpc_profile" ]]; then
  mpc_profile="$(realpath -e -- "$mpc_profile")"
  [[ "$mpc_profile" == "$repo_root/"* && -f "$mpc_profile" ]] || {
    echo 'MPC profile must resolve to a repository file' >&2; exit 2;
  }
  mpc_profile_container="/workspace/src/${mpc_profile#"$repo_root/"}"
fi
total_laps="${SDU_APEX_IROS_TOTAL_LAPS:-2}"
timeout_s="${SDU_APEX_IROS_WALL_TIMEOUT_S:-240}"
[[ "$total_laps" =~ ^[0-9]+$ && "$total_laps" -ge 2 &&
   "$timeout_s" =~ ^[0-9]+$ && "$timeout_s" -ge 60 ]] || {
   echo "Total laps must be >=2 and timeout >=60 seconds" >&2; exit 2;
}
[[ -f "$map" && -f "$trajectory" ]] || { echo 'Build the IROS map/raceline first.' >&2; exit 2; }
image="${SDU_APEX_MAPPING_IMAGE:-sdu-apex-autodrive:iros-mapping}"
controller="${run_id}_controller"
simulator="${run_id}_sim"
recorder="rec_${run_id}"
mkdir -p "$run_dir"
if [[ "${SDU_APEX_IROS_SKIP_BUILD:-0}" == 1 ]]; then
  docker image inspect "$image" >/dev/null 2>&1 || {
    echo "The requested prebuilt IROS controller image is missing" >&2; exit 1;
  }
  printf "Using existing controller image %s\n" "$image" >"$run_dir/build.log"
else
  SDU_APEX_IMAGE="$image" tools/build_competition.sh >"$run_dir/build.log" 2>&1
fi
docker image inspect --format '{{.Id}}' "$image" >"$run_dir/controller_image_id.txt"
cp -- "$map" "$run_dir/map.yaml"
cp -- "$trajectory" "$run_dir/raceline.csv"
cp -- f1tenth_localization/config/amcl_iros.yaml "$run_dir/amcl_overlay.yaml"
cp -- f1tenth_mpc/config/mpc_competition.yaml "$run_dir/mpc_base.yaml"
if [[ -n "$mpc_profile" ]]; then cp -- "$mpc_profile" "$run_dir/mpc_overlay.yaml"; fi
cleanup() {
  docker stop --timeout 0 "$simulator" >/dev/null 2>&1 || true
  docker kill --signal SIGINT "$recorder" "$controller" >/dev/null 2>&1 || true
  docker stop --timeout 3 "$recorder" "$controller" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker run --rm --name "$controller" --network=host --ipc=host --privileged --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -v "$repo_root:/workspace/src:rw" -e "SDU_APEX_TRAJECTORY_FILE=$trajectory_container" \
  -e "SDU_APEX_MPC_PARAMETER_FILE=$mpc_profile_container" --entrypoint /bin/bash "$image" -lc '
    source /opt/ros/humble/setup.bash &&
    source /home/autodrive_devkit/install/setup.bash &&
    source /workspace/install/setup.bash &&
    exec ros2 launch sdu_apex_autodrive controller.launch.py \
      map_yaml:=/workspace/src/f1tenth_planning/maps/autodrive_iros_2026_compete.yaml \
      trajectory_file:="$SDU_APEX_TRAJECTORY_FILE" \
      mpc_parameter_overlay:="$SDU_APEX_MPC_PARAMETER_FILE" \
      amcl_parameter_overlay:=/workspace/src/f1tenth_localization/config/amcl_iros.yaml \
      mpc_publish_diagnostics:=true with_rviz:=false
  ' >"$run_dir/controller.log" 2>&1 &
controller_job=$!
SDU_APEX_IMAGE="$image" SDU_APEX_RUN_ID="$run_id" SDU_APEX_RECORD_PROFILE=mpc_performance \
  tools/record_debug_bag.sh >"$run_dir/recorder.log" 2>&1 &
recorder_job=$!
wait_ready() {
  local filename="$1" pattern="$2"
  for _ in $(seq 1 90); do
    if [[ -f "$filename" ]] && rg -q "$pattern" "$filename"; then return; fi
    kill -0 "$controller_job" && kill -0 "$recorder_job" || return 1
    sleep 1
  done
  echo "Timed out waiting for $pattern; inspect $run_dir" >&2
  return 1
}
wait_ready "$run_dir/controller.log" 'Loaded MPC trajectory'
wait_ready "$run_dir/recorder.log" 'Recording'
SDU_APEX_WATCH_STALL_TIMEOUT_S=30 SDU_APEX_WATCH_MIN_DISTANCE_PER_LAP_M=25 \
  tools/watch_sim_run.sh "$controller" "$simulator" "$recorder" "mpc-lap-$total_laps" "$timeout_s" \
  >"$run_dir/watcher.log" 2>&1 &
watcher_job=$!
wait_ready "$run_dir/watcher.log" 'WATCHER_READY'
SDU_APEX_SIM_TRACK=iros SDU_APEX_SIM_MODE=batchmode SDU_APEX_SIM_LOG_FILE=/dev/null \
  SDU_APEX_SIM_CONTAINER="$simulator" tools/start_simulator.sh >"$run_dir/simulator.log" 2>&1 &
simulator_job=$!
echo "IROS production trial: $run_dir, target $total_laps laps, trajectory $trajectory"
set +e
wait "$watcher_job"
result=$?
wait "$controller_job" "$recorder_job" "$simulator_job" 2>/dev/null
set -e
cat "$run_dir/watcher.log"
exit "$result"
