#!/usr/bin/env bash
set -euo pipefail

display_number="${AUTODRIVE_DISPLAY_NUM:-99}"
screen_geometry="${AUTODRIVE_XVFB_SCREEN:-1920x1080x24}"
unity_screen_width="${AUTODRIVE_UNITY_SCREEN_WIDTH:-}"
unity_screen_height="${AUTODRIVE_UNITY_SCREEN_HEIGHT:-}"
unity_fullscreen="${AUTODRIVE_UNITY_FULLSCREEN:-0}"
unity_log_file="${AUTODRIVE_UNITY_LOG_FILE:-/dev/null}"

if [[ -n "${unity_screen_width}" || -n "${unity_screen_height}" ]]; then
  if [[ ! "${unity_screen_width}" =~ ^[1-9][0-9]*$ || ! "${unity_screen_height}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Set AUTODRIVE_UNITY_SCREEN_WIDTH and AUTODRIVE_UNITY_SCREEN_HEIGHT to positive integers." >&2
    exit 2
  fi

  if [[ "${unity_fullscreen}" != "0" && "${unity_fullscreen}" != "1" ]]; then
    echo "AUTODRIVE_UNITY_FULLSCREEN must be 0 or 1." >&2
    exit 2
  fi
fi

Xvfb ":${display_number}" -screen 0 "${screen_geometry}" -ac \
  >/tmp/autodrive-xvfb.log 2>&1 &
xvfb_pid=$!

cleanup() {
  kill "${xvfb_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export DISPLAY=":${display_number}"
for _ in {1..50}; do
  if xdpyinfo >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! xdpyinfo >/dev/null 2>&1; then
  echo "Xvfb did not become ready on DISPLAY=${DISPLAY}" >&2
  exit 1
fi

cd /home/autodrive_simulator

if [[ "$#" -eq 0 ]]; then
  simulator_args=("./AutoDRIVE Simulator.x86_64")

  # Xvfb size alone does not constrain a standalone Unity window. These
  # standard Unity arguments make a local resolution benchmark meaningful.
  if [[ -n "${unity_screen_width}" ]]; then
    simulator_args+=(
      -screen-width "${unity_screen_width}"
      -screen-height "${unity_screen_height}"
      -screen-fullscreen "${unity_fullscreen}"
    )
  fi

  simulator_args+=(
    -ip "${AUTODRIVE_SIM_IP:-127.0.0.1}"
    -port "${AUTODRIVE_SIM_PORT:-4567}"
    -logFile "${unity_log_file}"
  )
  set -- "${simulator_args[@]}"
fi

"$@"
