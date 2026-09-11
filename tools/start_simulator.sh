#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

sim_ip="${AUTODRIVE_SIM_IP:-127.0.0.1}"
sim_port="${AUTODRIVE_SIM_PORT:-4567}"
if [[ "${AUTODRIVE_MODEL_ID_MODE:-0}" == "1" ]]; then
  default_player="/home/akselmo/Documents/GitHub/AutoDRIVE/Builds/AutoDRIVE-ModelIdentification.x86_64"
  mode_label="open-plane model-identification"
else
  default_player="/home/akselmo/Documents/GitHub/AutoDRIVE/Builds/AutoDRIVE-Simulator.x86_64"
  mode_label="competition track"
fi
player="${AUTODRIVE_PLAYER:-${default_player}}"

if [[ ! -x "${player}" ]]; then
  echo "The verified 40 Hz Unity player was not found or is not executable:" >&2
  echo "  ${player}" >&2
  echo >&2
  echo "Build it from the local AutoDRIVE source, then retry:" >&2
  echo "  /home/akselmo/Unity/Hub/Editor/2022.3.52f1/Editor/Unity \\" >&2
  echo "    -batchmode -quit \\" >&2
  echo "    -projectPath /home/akselmo/Documents/GitHub/AutoDRIVE \\" >&2
  if [[ "${AUTODRIVE_MODEL_ID_MODE:-0}" == "1" ]]; then
    echo "    -executeMethod BuildLinuxPlayer.BuildModelIdentification \\" >&2
  else
    echo "    -executeMethod BuildLinuxPlayer.BuildCompete \\" >&2
  fi
  echo "    -logFile /tmp/autodrive-compete-build.log" >&2
  exit 1
fi

echo "Starting the ${mode_label} simulator in Terminal 1..."
echo "Endpoint: ${sim_ip}:${sim_port}"
echo "Mode: Unity graphical HDRP (no -batchmode, no -no-graphics)"
echo "Player: ${player}"
player_args=(
  -ip "${sim_ip}"
  -port "${sim_port}"
  -logFile "${AUTODRIVE_UNITY_LOG_FILE:-/tmp/autodrive-source-40hz.log}"
)

if [[ "${AUTODRIVE_BATCHMODE:-0}" == "1" ]]; then
  echo "Diagnostic override: adding -batchmode"
  player_args=(-batchmode "${player_args[@]}")
fi

"${player}" "${player_args[@]}"
