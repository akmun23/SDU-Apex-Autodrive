#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"
image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:competition}"
api_image="${AUTODRIVE_API_IMAGE:-autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7}"
map_rel="${AUTODRIVE_MAP_REL:-maps/autodrive_track_ftg_commit_20260909_025m.yaml}"
trajectory_rel="${AUTODRIVE_TRAJECTORY_REL:-trajectories/autodrive_mintime_sim_5p0_dense/autodrive_mintime_raceline.csv}"

for track_asset in "${map_rel}" "${trajectory_rel}"; do
  if [[ "${track_asset}" == /* || "${track_asset}" == *".."* ||
        ! -f "${repo_root}/f1tenth_planning/${track_asset}" ]]; then
    echo "Track asset must be a repository-relative file under f1tenth_planning: ${track_asset}" >&2
    exit 2
  fi
done

cd "${repo_root}"
python3 tools/verify_runtime_topic_policy.py
exec docker build \
  --network=host \
  --build-arg "AUTODRIVE_API_IMAGE=${api_image}" \
  --build-arg "AUTODRIVE_MAP_REL=${map_rel}" \
  --build-arg "AUTODRIVE_TRAJECTORY_REL=${trajectory_rel}" \
  -t "${image}" "${repo_root}"
