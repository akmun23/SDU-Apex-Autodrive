#!/usr/bin/env bash
# Supervised, recorded FTG/SLAM mapping of the IROS competition scene.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"
image="${SDU_APEX_MAPPING_IMAGE:-sdu-apex-autodrive:iros-mapping}"
container="${SDU_APEX_MAPPING_CONTAINER:-sdu_apex_autodrive_mapping}"
simulator="${SDU_APEX_SIM_CONTAINER:-sdu_apex_sim_iros}"
output="${SDU_APEX_MAPPING_OUTPUT:-${repo_root}/live_runs/iros_mapping_$(date +%Y%m%d_%H%M%S)}"
mkdir -p -- "$output"
output="$(realpath -- "$output")"
cp -- "${repo_root}/f1tenth_control/config/ftg_mapping.yaml" "$output/ftg_parameters.yaml"
cp -- "${repo_root}/sdu_apex_autodrive/config/slam_mapping.yaml" "$output/slam_parameters.yaml"
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Mapping container already exists: $container" >&2
  exit 2
fi
# Docker cache makes unchanged launches cheap, while always checking current
# sources avoids accidentally running an old controller binary.
SDU_APEX_IMAGE="$image" "${repo_root}/tools/build_competition.sh"
cleanup() {
  docker stop --timeout 0 "$simulator" >/dev/null 2>&1 || true
  docker stop --timeout 3 "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "Mapping output: $output"
docker run --rm --name "$container" \
  --network=host --ipc=host --privileged --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -e "SDU_APEX_MAPPING_MAP_PREFIX=${SDU_APEX_MAPPING_MAP_PREFIX:-/workspace/src/f1tenth_planning/maps/autodrive_iros_2026_compete}" \
  -v "${repo_root}/f1tenth_planning/maps:/workspace/src/f1tenth_planning/maps:rw" \
  -v "$output:/mapping:rw" \
  --entrypoint /bin/bash "$image" -lc '
    source /opt/ros/humble/setup.bash &&
    source /home/autodrive_devkit/install/setup.bash &&
    source /workspace/install/setup.bash &&
    exec ros2 launch sdu_apex_autodrive mapping.launch.py "map_prefix:=${SDU_APEX_MAPPING_MAP_PREFIX}"
  ' > >(tee "$output/runtime.log") 2>&1 &
runner_pid=$!
while kill -0 "$runner_pid" 2>/dev/null; do
  if [[ -s "$output/report.json" ]]; then
    docker stop --timeout 0 "$simulator" >/dev/null 2>&1 || true
    break
  fi
  sleep 0.1
done
set +e
wait "$runner_pid"
runtime_status=$?
set -e
if [[ -f "$output/report.json" ]]; then
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r,indent=2)); sys.exit(0 if r["result"]=="completed_laps" else 3)' "$output/report.json"
else
  echo "Mapping runtime ended without a report; see $output/runtime.log" >&2
  exit "${runtime_status:-2}"
fi
