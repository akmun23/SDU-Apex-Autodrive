#!/usr/bin/env bash
# ROS setup files intentionally reference variables that may be unset in a
# clean container. Source them before enabling nounset so the fixed
# competition entrypoint is safe on a fresh process environment.
set -eo pipefail

# Competition startup is deliberately unconditional and independent of
# ~/.bashrc, Compose, host source mounts, or environment overrides.
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
source /workspace/install/setup.bash

set -u
exec ros2 launch sdu_apex_autodrive competition.launch.py >/dev/null 2>&1
