#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"
cd "${repo_root}"

container="${SDU_APEX_CONTAINER:-sdu_apex_autodrive_dev}"
if docker container inspect "${container}" >/dev/null 2>&1; then
  docker stop "${container}"
else
  echo "Container ${container} is not running."
fi
