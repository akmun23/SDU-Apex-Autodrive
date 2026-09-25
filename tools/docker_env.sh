#!/usr/bin/env bash

# Prefer the user's rootless daemon on this machine, but respect an explicit
# DOCKER_HOST when another daemon/context is intentionally selected.
if [[ -z "${DOCKER_HOST:-}" ]]; then
  docker_runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if [[ -S "${docker_runtime_dir}/docker.sock" ]]; then
    export DOCKER_HOST="unix://${docker_runtime_dir}/docker.sock"
  elif [[ -S /tmp/apex-rootless/docker.sock ]]; then
    export DOCKER_HOST="unix:///tmp/apex-rootless/docker.sock"
  fi
fi
