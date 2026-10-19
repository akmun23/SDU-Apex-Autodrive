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

if [[ -n "${SDU_APEX_SIM_IMAGE_OVERRIDE:-}" ]]; then
  image="${SDU_APEX_SIM_IMAGE_OVERRIDE}"
fi

container_player_binary="./AutoDRIVE Simulator.x86_64"
simulator_log_file="${SDU_APEX_SIM_LOG_FILE:--}"
case "${mode}" in
  gui)
    simulator_command="exec \"${container_player_binary}\" -ip 127.0.0.1 -port 4567 -logFile \"${simulator_log_file}\""
    tty_args=(-it)
    ;;
  batchmode)
    simulator_command="Xvfb :123 -screen 0 1920x1080x24 -ac >/dev/null 2>&1 & xvfb_pid=\$!; export DISPLAY=:123; for attempt in \$(seq 1 50); do [[ -S /tmp/.X11-unix/X123 ]] && break; sleep 0.1; done; kill -0 \"\$xvfb_pid\"; exec \"${container_player_binary}\" -batchmode -ip 127.0.0.1 -port 4567 -logFile \"${simulator_log_file}\""
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

capture_args=()
capture_profile="${SDU_APEX_SIM_CAPTURE_PROFILE:-}"
if [[ -n "${capture_profile}" ]]; then
  if [[ "${track}" != "explore" ]]; then
    echo "Development wheel-capture profiles are restricted to the Explore simulator." >&2
    exit 2
  fi
  if [[ -z "${SDU_APEX_SIM_CAPTURE_DIR:-}" ]]; then
    echo "SDU_APEX_SIM_CAPTURE_PROFILE requires SDU_APEX_SIM_CAPTURE_DIR." >&2
    exit 2
  fi
  if [[ -z "${SDU_APEX_SIM_PLAYER_DIR:-}" ]]; then
    echo "SDU_APEX_SIM_CAPTURE_PROFILE requires a custom development player." >&2
    exit 2
  fi
  case "${capture_profile}" in
    brake_pulse|yaw_transition) ;;
    *) echo "Unsupported capture profile: ${capture_profile}" >&2; exit 2 ;;
  esac
fi
if [[ -n "${SDU_APEX_SIM_CAPTURE_DIR:-}" ]]; then
  requested_capture_directory="${SDU_APEX_SIM_CAPTURE_DIR}"
  if [[ "${requested_capture_directory}" != /* ]]; then
    requested_capture_directory="${repo_root}/${requested_capture_directory}"
  fi
  capture_directory="$(realpath -m -- "${requested_capture_directory}")"
  case "${capture_directory}" in
    "${repo_root}"/live_runs/*) ;;
    *) echo "SDU_APEX_SIM_CAPTURE_DIR must be inside ${repo_root}/live_runs." >&2; exit 2 ;;
  esac
  mkdir -p -- "${capture_directory}"
  capture_args=(-e SDU_APEX_WHEEL_CAPTURE_PATH=/sdu_apex_capture \
    -v "${capture_directory}:/sdu_apex_capture:rw")
  if [[ -n "${capture_profile}" ]]; then
    capture_args+=(-e "SDU_APEX_WHEEL_CAPTURE_PROFILE=${capture_profile}")
    if [[ "${capture_profile}" == "yaw_transition" ]]; then
      capture_args+=(-e "SDU_APEX_WHEEL_CAPTURE_SEED=${SDU_APEX_WHEEL_CAPTURE_SEED:-20261018}")
      echo "No ROS controller is launched; the development player owns only the diagnostic inputs."
    fi
    echo "Unity diagnostic command profile: ${capture_profile}"
  fi
  echo "Development wheel capture output: ${capture_directory}"
  echo "The selected simulator image must include the development capture source for this to produce a file."
fi

player_args=()
if [[ -n "${SDU_APEX_SIM_PLAYER_DIR:-}" ]]; then
  requested_player_directory="${SDU_APEX_SIM_PLAYER_DIR}"
  if [[ "${requested_player_directory}" != /* ]]; then
    requested_player_directory="${repo_root}/${requested_player_directory}"
  fi
  player_directory="$(realpath -- "${requested_player_directory}")"
  case "${player_directory}" in
    "${repo_root}"/*) ;;
    *) echo "SDU_APEX_SIM_PLAYER_DIR must be inside ${repo_root}." >&2; exit 2 ;;
  esac
  if [[ ! -x "${player_directory}/AutoDRIVE Simulator.x86_64" ||
        ! -d "${player_directory}/AutoDRIVE Simulator_Data" ]]; then
    echo "SDU_APEX_SIM_PLAYER_DIR must contain the Linux player and its _Data directory." >&2
    exit 2
  fi
  player_args=(-v "${player_directory}:/home/autodrive_simulator:ro")
  echo "Using workspace development player: ${player_directory}"
fi

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  docker pull "${image}"
fi
if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Container ${container} already exists; stop it before starting this simulator." >&2
  exit 1
fi

echo "Starting ${track} simulator image in ${mode} mode: ${image}"
echo "Unity log destination: ${simulator_log_file}"
exec docker run --rm --name "${container}" "${tty_args[@]}" \
  --log-opt max-size=10m --log-opt max-file=3 \
  --ulimit core=0 \
  --network=host --ipc=host --privileged --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  ${DISPLAY:+-e DISPLAY} \
  "${xauthority_args[@]}" \
  "${capture_args[@]}" \
  "${player_args[@]}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --entrypoint /bin/bash \
  "${image}" -lc "${simulator_command}"
