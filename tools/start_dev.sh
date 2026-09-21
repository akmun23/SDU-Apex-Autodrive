#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

controller="${SDU_APEX_CONTROLLER:-mpc}"
if [[ "${controller}" != "mpc" && "${controller}" != "pure_pursuit" && "${controller}" != "ftg" ]]; then
  echo "SDU_APEX_CONTROLLER must be mpc, pure_pursuit, or ftg." >&2
  exit 2
fi

image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:dev}"
container="${SDU_APEX_CONTAINER:-sdu_apex_autodrive_dev}"

echo "Starting the ROS 2 development/controller stack..."
echo "Controller: ${controller}"
echo "The bridge, odometry, localization, controller, and actuator are one launch."
echo "The simulator may already be connected or may connect while this starts."

python3 tools/verify_runtime_topic_policy.py

if [[ "${AUTODRIVE_REBUILD:-1}" == "1" ]]; then
  docker build -t "${image}" .
fi

docker rm -f "${container}" >/dev/null 2>&1 || true
exec docker run --rm --name "${container}" \
  --network=host --ipc=host --privileged --gpus all \
  -e SDU_APEX_AUTOSTART=1 \
  -e SDU_APEX_CONTROLLER="${controller}" \
  -e SDU_APEX_WITH_RVIZ="${SDU_APEX_WITH_RVIZ:-false}" \
  -e SDU_APEX_MPC_PUBLISH_DIAGNOSTICS="${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}" \
  -v "${repo_root}:/workspace/src:rw" \
  --entrypoint /workspace/src/docker/entrypoint.sh \
  "${image}"
