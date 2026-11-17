#!/usr/bin/env bash
# Records the competition topics to an MCAP bag from a separate container on the
# host network; the autonomous stack is untouched. Foxglove opens the bag as it
# is: MCAP carries the ROS 2 message definitions of every topic, including the
# team's sdu_apex_msgs types, and chunks are zstd-compressed.
#
#   tools/record_bag.sh                       # bags/competition_<timestamp>/
#   SDU_APEX_BAG_NAME=lap_test tools/record_bag.sh
#   SDU_APEX_RECORDER_CPUSET_CPUS=12-15 tools/record_bag.sh   # keep it off the MPC cores
#
# Stop with Ctrl-C; the bag is finalized on SIGINT.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:competition}"
bag_dir="${SDU_APEX_BAG_DIR:-${repo_root}/bags}"
name="${SDU_APEX_BAG_NAME:-competition_$(date +%Y%m%d_%H%M%S)}"
container="${SDU_APEX_RECORDER_CONTAINER:-sdu_apex_recorder}"

[[ "${name}" =~ ^[[:alnum:]_-]+$ ]] || {
  echo "SDU_APEX_BAG_NAME may contain only letters, digits, underscores and hyphens." >&2
  exit 2
}
docker image inspect "${image}" >/dev/null 2>&1 || {
  echo "Image ${image} is not built locally; run ./tools/build_competition.sh first." >&2
  exit 1
}
mkdir -p "${bag_dir}"
[[ ! -e "${bag_dir}/${name}" ]] || {
  echo "Bag already exists: ${bag_dir}/${name}" >&2
  exit 1
}

cpuset_args=()
if [[ -n "${SDU_APEX_RECORDER_CPUSET_CPUS:-}" ]]; then
  [[ "${SDU_APEX_RECORDER_CPUSET_CPUS}" =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || {
    echo "Invalid SDU_APEX_RECORDER_CPUSET_CPUS" >&2
    exit 2
  }
  cpuset_args=(--cpuset-cpus "${SDU_APEX_RECORDER_CPUSET_CPUS}")
fi

exec docker run --rm --log-driver=none --name "${container}" \
  "${cpuset_args[@]}" \
  --network=host --ipc=host \
  --tmpfs /run/ros:rw,nosuid,nodev,noexec,size=16m \
  -e ROS_HOME=/run/ros \
  -v "${bag_dir}:/bags:rw" \
  --entrypoint /bin/bash \
  "${image}" -c '
    set -e
    source /opt/ros/humble/setup.bash
    source /home/autodrive_devkit/install/setup.bash
    source /workspace/install/setup.bash
    exec ros2 bag record \
      --storage mcap --storage-preset-profile zstd_fast \
      --max-cache-size 268435456 \
      -o "/bags/'"${name}"'" \
      /autodrive/roboracer_1/lidar \
      /autodrive/roboracer_1/imu \
      /autodrive/roboracer_1/left_encoder \
      /autodrive/roboracer_1/right_encoder \
      /autodrive/roboracer_1/steering \
      /autodrive/roboracer_1/throttle \
      /autodrive/roboracer_1/steering_command \
      /autodrive/roboracer_1/throttle_command \
      /odom \
      /ekf_odom \
      /current_map_pose \
      /odom/unity_plant_state \
      /odom/bridge_request_command \
      /odom/bridge_actuator_feedback \
      /mpc/drive_command \
      /actuators/drive_command \
      /map >/dev/null 2>&1
  '
