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

# Start exactly one official bridge and one selected team controller. MPC is
# the normal local racing controller; explicitly enable its command publisher
# here so a GUI simulator only needs its Connect button after this container is
# running. Diagnostics recorders remain opt-in to avoid adding CSV I/O to the
# timing-sensitive default path.
controller="${SDU_APEX_CONTROLLER:-mpc}"
launch_args=(
  "controller:=${controller}"
  "with_rviz:=${SDU_APEX_WITH_RVIZ:-false}"
)
if [[ "${controller}" == "mpc" ]]; then
  launch_args+=(
    "mpc_enabled:=true"
    "mpc_publish_diagnostics:=${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}"
  )
  if [[ -n "${SDU_APEX_MPC_OVERRIDE_PARAMS:-}" ]]; then
    launch_args+=("mpc_override_params:=${SDU_APEX_MPC_OVERRIDE_PARAMS}")
  fi
fi
exec ros2 launch sdu_apex_autodrive controller.launch.py "${launch_args[@]}"
