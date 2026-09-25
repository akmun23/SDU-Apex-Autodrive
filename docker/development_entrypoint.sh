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

python3 /workspace/src/tools/verify_runtime_topic_policy.py

# Start exactly one official bridge and one selected team controller. MPC is
# the normal local racing controller; explicitly enable its command publisher
# here so a GUI simulator only needs its Connect button after this container is
# running. Diagnostics are limited to the live MPC publisher and remain off by
# default so the normal 40 Hz path does not serialize JSON.
controller="${SDU_APEX_CONTROLLER:-mpc}"
launch_args=(
  "controller:=${controller}"
  "with_rviz:=${SDU_APEX_WITH_RVIZ:-false}"
)
if [[ "${controller}" == "mpc" ]]; then
  launch_args+=(
    "mpc_publish_diagnostics:=${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}"
  )
fi
exec ros2 launch sdu_apex_autodrive controller.launch.py "${launch_args[@]}"
