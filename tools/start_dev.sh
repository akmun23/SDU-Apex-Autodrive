#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"
cd "${repo_root}"

image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:dev}"
container="${SDU_APEX_CONTAINER:-sdu_apex_autodrive_dev}"
api_image="${AUTODRIVE_API_IMAGE:-autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7}"

if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Container ${container} already exists; stop it before starting another development stack." >&2
  exit 1
fi

echo "Starting the ROS 2 development stack..."
if [[ "${SDU_APEX_MPC_ENABLED:-true}" == "false" ]]; then
  echo "MPC disabled; the explicitly selected development test controller will run."
else
  echo "The bridge, PhysX odometry, localization, MPC, and direct actuator are one launch."
fi
echo "The simulator may already be connected or may connect while this starts."

python3 tools/verify_runtime_topic_policy.py

mpc_parameter_overlay=""
if [[ -n "${SDU_APEX_MPC_PARAMETER_OVERLAY:-}" ]]; then
  if [[ ! -f "${SDU_APEX_MPC_PARAMETER_OVERLAY}" ]]; then
    echo "MPC parameter overlay does not exist: ${SDU_APEX_MPC_PARAMETER_OVERLAY}" >&2
    exit 1
  fi
  overlay_host_path="$(realpath -- "${SDU_APEX_MPC_PARAMETER_OVERLAY}")"
  case "${overlay_host_path}" in
    "${repo_root}"/*)
      mpc_parameter_overlay="/workspace/src/${overlay_host_path#"${repo_root}"/}"
      ;;
    *)
      echo "MPC parameter overlay must be inside the repository bind mount." >&2
      exit 1
      ;;
  esac
fi

actuator_parameter_overlay=""
if [[ -n "${SDU_APEX_ACTUATOR_PARAMETER_OVERLAY:-}" ]]; then
  if [[ ! -f "${SDU_APEX_ACTUATOR_PARAMETER_OVERLAY}" ]]; then
    echo "Actuator parameter overlay does not exist: ${SDU_APEX_ACTUATOR_PARAMETER_OVERLAY}" >&2
    exit 1
  fi
  overlay_host_path="$(realpath -- "${SDU_APEX_ACTUATOR_PARAMETER_OVERLAY}")"
  case "${overlay_host_path}" in
    "${repo_root}"/*)
      actuator_parameter_overlay="/workspace/src/${overlay_host_path#"${repo_root}"/}"
      ;;
    *)
      echo "Actuator parameter overlay must be inside the repository bind mount." >&2
      exit 1
      ;;
  esac
fi

odom_parameter_overlay=""
if [[ -n "${SDU_APEX_ODOM_PARAMETER_OVERLAY:-}" ]]; then
  if [[ ! -f "${SDU_APEX_ODOM_PARAMETER_OVERLAY}" ]]; then
    echo "SDU_APEX_ODOM_PARAMETER_OVERLAY does not exist: ${SDU_APEX_ODOM_PARAMETER_OVERLAY}" >&2
    exit 1
  fi
  overlay_host_path="$(realpath -- "${SDU_APEX_ODOM_PARAMETER_OVERLAY}")"
  case "${overlay_host_path}" in
    "${repo_root}"/*)
      odom_parameter_overlay="/workspace/src/${overlay_host_path#"${repo_root}"/}"
      ;;
    *)
      echo "SDU_APEX_ODOM_PARAMETER_OVERLAY must be inside the repository bind mount." >&2
      exit 1
      ;;
  esac
fi

ros_domain_args=()
if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
  ros_domain_args+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi

map_yaml_override=""
trajectory_file_override=""
csv_command_file_override=""
csv_path_repeats="${SDU_APEX_CSV_PATH_REPEATS:-3}"
if [[ ! "${csv_path_repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SDU_APEX_CSV_PATH_REPEATS must be a positive integer." >&2
  exit 2
fi
for path_setting in SDU_APEX_MAP_YAML SDU_APEX_TRAJECTORY_FILE SDU_APEX_CSV_COMMAND_FILE; do
  path_value="${!path_setting:-}"
  [[ -n "${path_value}" ]] || continue
  if [[ ! -f "${path_value}" ]]; then
    echo "${path_setting} does not name an existing file: ${path_value}" >&2
    exit 1
  fi
  resolved_path="$(realpath -- "${path_value}")"
  case "${resolved_path}" in
    "${repo_root}"/*)
      container_path="/workspace/src/${resolved_path#"${repo_root}"/}"
      if [[ "${path_setting}" == SDU_APEX_TRAJECTORY_FILE ]]; then
        trajectory_file_override="${container_path}"
      elif [[ "${path_setting}" == SDU_APEX_MAP_YAML ]]; then
        map_yaml_override="${container_path}"
      else
        csv_command_file_override="${container_path}"
      fi
      ;;
    *)
      echo "${path_setting} must be inside the repository bind mount." >&2
      exit 1
      ;;
  esac
done

build_mounted_source=1
if [[ "${AUTODRIVE_REBUILD:-1}" == "1" ]]; then
  docker build --network=host \
    --build-arg "AUTODRIVE_API_IMAGE=${api_image}" -t "${image}" .
  build_mounted_source=0
elif ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Image ${image} is not available locally; rebuild or choose an existing SDU_APEX_IMAGE." >&2
  exit 1
fi

exec docker run --rm --name "${container}" \
  --network=host --ipc=host --privileged --gpus all \
  --log-opt max-size=10m --log-opt max-file=3 \
  "${ros_domain_args[@]}" \
  -e SDU_APEX_AUTOSTART=1 \
  -e SDU_APEX_WITH_RVIZ="${SDU_APEX_WITH_RVIZ:-false}" \
  -e SDU_APEX_MPC_NATIVE_TUNING="${SDU_APEX_MPC_NATIVE_TUNING:-false}" \
  -e SDU_APEX_MPC_PUBLISH_DIAGNOSTICS="${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}" \
  -e "SDU_APEX_MPC_START_DELAY_SEC=${SDU_APEX_MPC_START_DELAY_SEC:-2.0}" \
  -e "SDU_APEX_CONTROLLER_MAX_SPEED=${SDU_APEX_CONTROLLER_MAX_SPEED:-}" \
  -e "SDU_APEX_BUILD_MOUNTED_SOURCE=${build_mounted_source}" \
  -e "SDU_APEX_MPC_PARAMETER_OVERLAY=${mpc_parameter_overlay}" \
  -e "SDU_APEX_ACTUATOR_PARAMETER_OVERLAY=${actuator_parameter_overlay}" \
  -e "SDU_APEX_ODOM_PARAMETER_OVERLAY=${odom_parameter_overlay}" \
  -e "SDU_APEX_MAP_YAML=${map_yaml_override}" \
  -e "SDU_APEX_TRAJECTORY_FILE=${trajectory_file_override}" \
  -e "SDU_APEX_CSV_COMMAND_FILE=${csv_command_file_override}" \
  -e "SDU_APEX_CSV_PATH_REPEATS=${csv_path_repeats}" \
  -e "SDU_APEX_CSV_LOOKAHEAD_SPEED_GAIN_S=${SDU_APEX_CSV_LOOKAHEAD_SPEED_GAIN_S:-0.12}" \
  -e "SDU_APEX_CSV_NATIVE_CONTROL_DELAY_S=${SDU_APEX_CSV_NATIVE_CONTROL_DELAY_S:-0.05}" \
  -e "SDU_APEX_DEV_SIM_RESET_ENABLED=${SDU_APEX_DEV_SIM_RESET_ENABLED:-false}" \
  -e "SDU_APEX_MPC_ENABLED=${SDU_APEX_MPC_ENABLED:-true}" \
  -v "${repo_root}:/workspace/src:rw" \
  --entrypoint /bin/bash \
  "${image}" /workspace/src/docker/development_entrypoint.sh
