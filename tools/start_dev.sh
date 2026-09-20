#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

controller="${SDU_APEX_CONTROLLER:-mpc}"
if [[ "${controller}" != "mpc" && "${controller}" != "pure_pursuit" && "${controller}" != "ftg" ]]; then
  echo "SDU_APEX_CONTROLLER must be mpc, pure_pursuit, or ftg." >&2
  exit 2
fi

compose_build=()
# The image contains the installed ROS/Python workspace.  Rebuild by default
# so source/config changes cannot silently leave the container on an old
# controller; Docker reuses all unchanged layers.
if [[ "${AUTODRIVE_REBUILD:-1}" == "1" ]]; then
  compose_build+=(--build)
fi

echo "Starting the ROS 2 development/controller stack in Terminal 2..."
echo "Controller: ${controller}"
echo "The bridge, odometry, localization, controller, and actuator are one launch."
echo "The simulator may already be connected or may connect while this starts."

python3 tools/verify_runtime_topic_policy.py

SDU_APEX_AUTOSTART=1 \
SDU_APEX_CONTROLLER="${controller}" \
SDU_APEX_WITH_RVIZ="${SDU_APEX_WITH_RVIZ:-false}" \
SDU_APEX_MPC_OVERRIDE_PARAMS="${SDU_APEX_MPC_OVERRIDE_PARAMS:-}" \
SDU_APEX_MPC_PUBLISH_DIAGNOSTICS="${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}" \
docker compose up "${compose_build[@]}" workspace
