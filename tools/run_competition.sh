#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:competition}"
container="${SDU_APEX_CONTAINER:-sdu_apex_autodrive_competition}"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Image ${image} is not built locally; run ./tools/build_competition.sh first." >&2
  exit 1
fi

if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Container ${container} already exists; stop it before starting another competition run." >&2
  exit 1
fi

exec docker run --rm --log-driver=none --name "${container}" \
  --network=host --ipc=host --privileged --gpus all \
  --tmpfs /run/ros:rw,nosuid,nodev,noexec,size=32m \
  -e ROS_HOME=/run/ros \
  "${image}"
