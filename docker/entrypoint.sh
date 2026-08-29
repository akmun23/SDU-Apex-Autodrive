#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
source /workspace/install/setup.bash

exec "$@"
