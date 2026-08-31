#!/usr/bin/env bash
set -e

# Installed as /home/autodrive_devkit.sh in the derived team image. The
# competition startup contract must not depend on ~/.bashrc.
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
source /workspace/install/setup.bash

# Explicit commands and maintenance shells remain available.
if [[ "${SDU_APEX_AUTOSTART:-0}" != "1" ]]; then
  if [[ "$#" -eq 0 ]]; then
    exec bash
  fi
  exec "$@"
fi

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

# Start exactly one official bridge and one selected team controller.
exec ros2 launch sdu_apex_autodrive controller.launch.py \
  "controller:=${SDU_APEX_CONTROLLER:-ftg}" \
  "with_rviz:=${SDU_APEX_WITH_RVIZ:-false}"
