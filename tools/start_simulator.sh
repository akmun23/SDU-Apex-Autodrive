#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

track="${SDU_APEX_SIM_TRACK:-compete}"
mode="${SDU_APEX_SIM_MODE:-batchmode}"
case "${track}" in
  practice)
    image="autodriveecosystem/autodrive_roboracer_sim@sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76"
    container="${SDU_APEX_SIM_CONTAINER:-sdu_apex_sim_practice}"
    ;;
  explore)
    image="autodriveecosystem/autodrive_roboracer_sim@sha256:1a26bcc8b91fe845f2caafb95533bf3606092502430efca493709e167a5adb40"
    container="${SDU_APEX_SIM_CONTAINER:-sdu_apex_sim_explore}"
    ;;
  compete)
    image="autodriveecosystem/autodrive_roboracer_sim:2026-icra-compete"
    container="${SDU_APEX_SIM_CONTAINER:-sdu_apex_sim_compete}"
    ;;
  *)
    echo "SDU_APEX_SIM_TRACK must be practice, explore, or compete." >&2
    exit 2
    ;;
esac

# The Unity client retries noisily when the bridge is not listening yet.
bridge_wait_command='
echo "Waiting for the ROS bridge on 127.0.0.1:4567"
bridge_ready=0
for attempt in $(seq 1 1800); do
  if (exec 3<>/dev/tcp/127.0.0.1/4567) >/dev/null 2>&1; then
    bridge_ready=1
    break
  fi
  sleep 0.1
done
if [[ "${bridge_ready}" != "1" ]]; then
  echo "ROS bridge did not start listening on TCP port 4567 within 180 seconds" >&2
  exit 1
fi
'

container_player_binary="./AutoDRIVE Simulator.x86_64"
simulator_log_file="${SDU_APEX_SIM_LOG_FILE:--}"
player_mount_args=()
if [[ -n "${SDU_APEX_SIM_PLAYER_DIR:-}" ]]; then
  player_dir="$(realpath -- "${SDU_APEX_SIM_PLAYER_DIR}")"
  if [[ ! -x "${player_dir}/AutoDRIVE Simulator.x86_64" || \
        ! -d "${player_dir}/AutoDRIVE Simulator_Data" ]]; then
    echo "SDU_APEX_SIM_PLAYER_DIR must contain the Linux player and its _Data directory." >&2
    exit 2
  fi
  container_player_binary="/tmp/sdu_apex_dev_player/AutoDRIVE Simulator.x86_64"
  player_mount_args=(-v "${player_dir}:/tmp/sdu_apex_dev_player:ro")
fi

dynamics_mount_args=()
if [[ -n "${SDU_APEX_SIM_DYNAMICS_DIR:-}" ]]; then
  dynamics_dir="$(realpath -m -- "${SDU_APEX_SIM_DYNAMICS_DIR}")"
  mkdir -p -- "${dynamics_dir}"
  dynamics_run_id="${SDU_APEX_SIM_DYNAMICS_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
  dynamics_mount_args=(
    -v "${dynamics_dir}:/tmp/sdu_apex_dynamics:rw"
    -e SDU_APEX_DYNAMICS_DIR=/tmp/sdu_apex_dynamics
    -e "SDU_APEX_DYNAMICS_RUN_ID=${dynamics_run_id}"
  )
fi

case "${mode}" in
  gui)
    simulator_command="${bridge_wait_command}exec \"${container_player_binary}\" -ip 127.0.0.1 -port 4567 -logFile \"${simulator_log_file}\""
    tty_args=(-it)
    ;;
  batchmode)
    simulator_command="${bridge_wait_command}Xvfb :123 -screen 0 1920x1080x24 -ac >/dev/null 2>&1 & xvfb_pid=\$!; export DISPLAY=:123; for attempt in \$(seq 1 50); do [[ -S /tmp/.X11-unix/X123 ]] && break; sleep 0.1; done; kill -0 \"\$xvfb_pid\"; exec \"${container_player_binary}\" -batchmode -ip 127.0.0.1 -port 4567 -logFile \"${simulator_log_file}\""
    tty_args=()
    ;;
  *)
    echo "SDU_APEX_SIM_MODE must be gui or batchmode." >&2
    exit 2
    ;;
esac

if [[ "${mode}" == "gui" && -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is unset; start this from the graphical desktop session." >&2
  exit 1
fi

xauthority_args=()
if [[ "${mode}" == "gui" ]]; then
  if [[ -z "${XAUTHORITY:-}" || ! -f "${XAUTHORITY}" ]]; then
    echo "XAUTHORITY does not name a readable desktop authorization file." >&2
    exit 1
  fi
  xauthority_path="$(realpath -- "${XAUTHORITY}")"
  xauthority_args=(-e "XAUTHORITY=${xauthority_path}" \
    -v "${xauthority_path}:${xauthority_path}:ro")
fi

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  docker pull "${image}"
fi
if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Container ${container} already exists; stop it before starting this simulator." >&2
  exit 1
fi

echo "Starting official ${track} simulator image in ${mode} mode: ${image}"
echo "Unity log destination: ${simulator_log_file}"
if [[ -n "${SDU_APEX_SIM_PLAYER_DIR:-}" ]]; then
  echo "Development player overlay: ${player_dir}"
fi
exec docker run --rm --name "${container}" "${tty_args[@]}" \
  --log-opt max-size=10m --log-opt max-file=3 \
  --ulimit core=0 \
  --network=host --ipc=host --privileged --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  ${DISPLAY:+-e DISPLAY} \
  "${xauthority_args[@]}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  "${player_mount_args[@]}" \
  "${dynamics_mount_args[@]}" \
  --entrypoint /bin/bash \
  "${image}" -lc "${simulator_command}"
