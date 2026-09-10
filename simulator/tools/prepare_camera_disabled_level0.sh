#!/usr/bin/env bash
set -euo pipefail

# Generate a scene-file override for the supplied prebuilt simulator. This
# removes the Socket component's serialized front-camera assignment without
# modifying the Docker image or the original Unity asset.

image="${1:-autodriveecosystem/autodrive_roboracer_sim:2026-icra-compete}"
output="${2:-$PWD/simulator/generated/level0-no-camera}"
workdir="$(mktemp -d /tmp/autodrive-camera-off.XXXXXX)"
container="autodrive_asset_extract_$$"

cleanup() {
  docker rm -f "${container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker create --name "${container}" "${image}" -c 'sleep infinity' >/dev/null
docker cp "${container}:/home/autodrive_simulator/Data/level0" "${workdir}/level0"

python3 -m venv "${workdir}/venv"
"${workdir}/venv/bin/pip" -q install UnityPy
"${workdir}/venv/bin/python" \
  "$(dirname "$0")/disable_packaged_socket_cameras.py" \
  "${workdir}/level0" "${output}"

echo "Camera-disabled scene override: ${output}"
