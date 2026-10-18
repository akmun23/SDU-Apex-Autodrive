#!/usr/bin/env bash
set -e

# Development-only launcher. The competition image uses its separate fixed
# entrypoint and does not mount this source file.
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash

if [[ "${SDU_APEX_BUILD_MOUNTED_SOURCE:-0}" == "1" ]]; then
  echo "Building the mounted runtime and its package dependencies"
  # The reusable dev image may carry generated artifacts from an older
  # sdu_apex_autodrive source tree. In particular, setuptools can retain
  # deleted config/launch files in SOURCES.txt and then fail while copying
  # them from the stale build directory. Remove only this package's generated
  # build/install outputs; the mounted source and other package caches remain.
  rm -rf \
    /workspace/build/sdu_apex_autodrive \
    /workspace/install/lib/sdu_apex_autodrive \
    /workspace/install/share/ament_index/resource_index/packages/sdu_apex_autodrive \
    /workspace/install/share/colcon-core/packages/sdu_apex_autodrive \
    /workspace/install/share/sdu_apex_autodrive
  cd /workspace
  CMAKE_BUILD_PARALLEL_LEVEL=1 colcon build \
    --packages-up-to sdu_apex_autodrive \
    --parallel-workers 1 \
    --merge-install --symlink-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
  source /workspace/install/setup.bash
else
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

# Start the bridge and the single MPC/localization runtime. Explicitly enable
# its command publisher
# here so a GUI simulator only needs its Connect button after this container is
# running. Diagnostics are limited to the live MPC publisher and remain off by
# default so the normal 40 Hz path does not serialize JSON.
launch_args=(
  "with_rviz:=${SDU_APEX_WITH_RVIZ:-false}"
)
if [[ -n "${SDU_APEX_MAP_YAML:-}" ]]; then
  launch_args+=("map_yaml:=${SDU_APEX_MAP_YAML}")
fi
if [[ -n "${SDU_APEX_TRAJECTORY_FILE:-}" ]]; then
  launch_args+=("trajectory_file:=${SDU_APEX_TRAJECTORY_FILE}")
fi
launch_args+=(
  "mpc_publish_diagnostics:=${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}"
  "mpc_start_delay_sec:=${SDU_APEX_MPC_START_DELAY_SEC:-2.0}"
)
if [[ -n "${SDU_APEX_MPC_PARAMETER_OVERLAY:-}" ]]; then
  launch_args+=("mpc_parameter_overlay:=${SDU_APEX_MPC_PARAMETER_OVERLAY}")
fi
if [[ -n "${SDU_APEX_ACTUATOR_PARAMETER_OVERLAY:-}" ]]; then
  launch_args+=(
    "actuator_parameter_overlay:=${SDU_APEX_ACTUATOR_PARAMETER_OVERLAY}"
  )
fi
if [[ -n "${SDU_APEX_ODOM_PARAMETER_OVERLAY:-}" ]]; then
  launch_args+=(
    "odom_parameter_overlay:=${SDU_APEX_ODOM_PARAMETER_OVERLAY}"
  )
fi
exec ros2 launch sdu_apex_autodrive controller.launch.py "${launch_args[@]}"
