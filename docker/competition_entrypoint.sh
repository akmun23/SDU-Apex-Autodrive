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
# Pin the background ROS processes to the same CPU pool as development;
# MPC itself is separately pinned by the launch file. On unsupported hosts,
# run without affinity.
runtime_cpus="${SDU_APEX_COMPETITION_RUNTIME_CPUS:-}"
if [[ -n "$runtime_cpus" ]]; then
  exec taskset --cpu-list "$runtime_cpus" python3 -c '
import logging
import launch.logging
from ros2cli.cli import main
launch.logging.launch_config.log_handler_factory = lambda *a, **k: logging.NullHandler()
raise SystemExit(main(argv=["launch", "sdu_apex_autodrive", "competition.launch.py"]))
' >/dev/null 2>&1
fi
exec python3 -c '
import logging
import launch.logging
from ros2cli.cli import main
launch.logging.launch_config.log_handler_factory = lambda *a, **k: logging.NullHandler()
raise SystemExit(main(argv=["launch", "sdu_apex_autodrive", "competition.launch.py"]))
' >/dev/null 2>&1
