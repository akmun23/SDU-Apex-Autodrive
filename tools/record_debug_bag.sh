#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"
image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:dev}"
run_id="${SDU_APEX_RUN_ID:-debug_$(date +%Y%m%d_%H%M%S)}"
container="${SDU_APEX_RECORDER_CONTAINER:-rec_${run_id}}"
run_parent="${repo_root}/live_runs/${run_id}"
ros_domain_args=()
if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
  ros_domain_args+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi

if [[ ! "${run_id}" =~ ^[[:alnum:]_-]+$ ]]; then
  echo "SDU_APEX_RUN_ID may contain only letters, digits, underscores, and hyphens." >&2
  exit 2
fi

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Image ${image} is not built locally; build or start the development stack first." >&2
  exit 1
fi

if [[ -e "${run_parent}/run" ]]; then
  echo "Bag output already exists: ${run_parent}/run" >&2
  exit 1
fi

mkdir -p "${run_parent}"
exec docker run --rm --name "${container}" \
  --network=host --ipc=host \
  --log-opt max-size=10m --log-opt max-file=3 \
  "${ros_domain_args[@]}" \
  -e ROSBAG_OUTPUT="/workspace/src/live_runs/${run_id}/run" \
  -v "${repo_root}:/workspace/src:rw" \
  --entrypoint /bin/bash \
  "${image}" -lc '
    set -e
    source /opt/ros/humble/setup.bash
    source /home/autodrive_devkit/install/setup.bash
    source /workspace/install/setup.bash
    exec ros2 bag record --max-cache-size 10485760 -o "${ROSBAG_OUTPUT}" \
      /autodrive/roboracer_1/lidar \
      /autodrive/roboracer_1/imu \
      /autodrive/roboracer_1/left_encoder \
      /autodrive/roboracer_1/right_encoder \
      /autodrive/roboracer_1/odom \
      /autodrive/roboracer_1/ips \
      /autodrive/roboracer_1/steering \
      /autodrive/roboracer_1/throttle \
      /autodrive/roboracer_1/steering_command \
      /autodrive/roboracer_1/throttle_command \
      /autodrive/roboracer_1/collision_count \
      /autodrive/roboracer_1/lap_count \
      /autodrive/roboracer_1/lap_time \
      /autodrive/roboracer_1/last_lap_time \
      /autodrive/roboracer_1/best_lap_time \
      /autodrive/roboracer_1/bridge_packet_timing \
      /autodrive/roboracer_1/bridge_timing_fault \
      /autodrive/roboracer_1/bridge_timing_fault_detail \
      /odom \
      /odom/diagnostics \
      /ekf_odom \
      /ekf_pose \
      /amcl_pose \
      /amcl_kld_diagnostics \
      /amcl_localization_health \
      /amcl_scan_alignment \
      /amcl_particle_count \
      /amcl_gpu_timing \
      /amcl_timing \
      /current_map_pose \
      /cmd/speed \
      /mpc/diagnostics \
      /sdu/tf \
      /sdu/tf_static \
      /tf \
      /tf_static \
      /map
  '
