#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

docker rm -f "${SDU_APEX_CONTAINER:-sdu_apex_autodrive_dev}" >/dev/null 2>&1 || true
