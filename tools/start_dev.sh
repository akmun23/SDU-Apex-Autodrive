#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

controller="${SDU_APEX_CONTROLLER:-pure_pursuit}"
if [[ "${controller}" != "pure_pursuit" && "${controller}" != "ftg" ]]; then
  echo "SDU_APEX_CONTROLLER must be pure_pursuit or ftg." >&2
  exit 2
fi

compose_build=()
if [[ "${AUTODRIVE_REBUILD:-0}" == "1" ]]; then
  compose_build+=(--build)
fi

echo "Starting the ROS 2 development/controller stack in Terminal 2..."
echo "Controller: ${controller}"
echo "The bridge, odometry, localization, controller, and actuator are one launch."
echo "The simulator may already be connected or may connect while this starts."

SDU_APEX_AUTOSTART=1 \
SDU_APEX_CONTROLLER="${controller}" \
SDU_APEX_WITH_RVIZ="${SDU_APEX_WITH_RVIZ:-false}" \
docker compose up "${compose_build[@]}" workspace
