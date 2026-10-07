#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:practice-lowfade060-20261006}"
container="${SDU_APEX_ODOM_SIDECAR_CONTAINER:-sdu_apex_explore_sensor_odometry}"
domain_id="${SDU_APEX_EXPERIMENT_DOMAIN_ID:-${ROS_DOMAIN_ID:-61}}"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Odometry sidecar image is not available locally: ${image}" >&2
  exit 1
fi
if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Odometry sidecar container already exists: ${container}" >&2
  exit 1
fi

echo "Starting sensor-only odometry sidecar on ROS domain ${domain_id}."
echo "Inputs: simulator encoders and IMU only. Outputs: /explore_sensor_odom and diagnostics."
echo "The sidecar does not start a bridge, MPC, actuator, AMCL, or simulator."

exec docker run --rm --name "${container}" \
  --network=host --ipc=host \
  --log-opt max-size=10m --log-opt max-file=2 \
  -e "ROS_DOMAIN_ID=${domain_id}" \
  -e SDU_APEX_BUILD_LOCALIZATION=1 \
  -v "${repo_root}:/workspace/src:rw" \
  --entrypoint /bin/bash \
  "${image}" /workspace/src/docker/development_entrypoint.sh \
  ros2 run f1tenth_localization sensor_odometry_node --ros-args \
    --params-file /workspace/src/f1tenth_localization/config/sensor_odometry.yaml \
    -p odom_topic:=/explore_sensor_odom \
    -p diagnostics_topic:=/explore_sensor_odom/diagnostics \
    -p publish_tf:=false
