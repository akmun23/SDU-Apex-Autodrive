#!/usr/bin/env bash
set -euo pipefail

# Competition startup is deliberately unconditional and independent of
# ~/.bashrc, Compose, host source mounts, or environment overrides.
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
source /workspace/install/setup.bash

exec ros2 launch sdu_apex_autodrive competition.launch.py
