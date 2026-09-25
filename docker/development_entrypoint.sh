#!/usr/bin/env bash
set -e

# Development-only launcher. The competition image uses its separate fixed
# entrypoint and does not mount this source file.
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
source /workspace/install/setup.bash

if [[ "${SDU_APEX_BUILD_MPC:-0}" == "1" ]]; then
  echo "Building f1tenth_mpc from the mounted workspace source..."
  cd /workspace
  CMAKE_BUILD_PARALLEL_LEVEL=1 colcon build \
    --packages-select f1tenth_mpc \
    --parallel-workers 1 \
    --merge-install --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
  source /workspace/install/setup.bash
fi

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

launch_mode="${SDU_APEX_LAUNCH:-controller}"
if [[ "${launch_mode}" == "mapping" ]]; then
  map_name="${SDU_APEX_MAP_NAME:-track_map}"
  if [[ ! "${map_name}" =~ ^[[:alnum:]_-]+$ ]]; then
    echo "SDU_APEX_MAP_NAME may contain only letters, digits, underscores, and hyphens." >&2
    exit 2
  fi
  exec ros2 launch sdu_apex_autodrive mapping.launch.py \
    "map_output_directory:=/workspace/src/live_runs/maps" \
    "map_name:=${map_name}"
fi
if [[ "${launch_mode}" != "controller" ]]; then
  echo "SDU_APEX_LAUNCH must be controller or mapping." >&2
  exit 2
fi

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
if [[ -n "${SDU_APEX_MAP_YAML:-}" ]]; then
  launch_args+=("map_yaml:=${SDU_APEX_MAP_YAML}")
fi
if [[ -n "${SDU_APEX_TRAJECTORY_FILE:-}" ]]; then
  launch_args+=("trajectory_file:=${SDU_APEX_TRAJECTORY_FILE}")
fi
if [[ "${controller}" == "mpc" ]]; then
  launch_args+=(
    "mpc_publish_diagnostics:=${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}"
  )
  if [[ -n "${SDU_APEX_MPC_PARAMETER_OVERLAY:-}" ]]; then
    launch_args+=(
      "mpc_parameter_overlay:=${SDU_APEX_MPC_PARAMETER_OVERLAY}"
    )
  fi
fi
exec ros2 launch sdu_apex_autodrive controller.launch.py "${launch_args[@]}"
