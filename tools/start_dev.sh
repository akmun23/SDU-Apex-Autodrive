#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"
cd "${repo_root}"

controller="${SDU_APEX_CONTROLLER:-mpc}"
launch_mode="${SDU_APEX_LAUNCH:-controller}"
if [[ "${launch_mode}" != "controller" && "${launch_mode}" != "mapping" ]]; then
  echo "SDU_APEX_LAUNCH must be controller or mapping." >&2
  exit 2
fi
if [[ "${controller}" != "mpc" && "${controller}" != "pure_pursuit" && "${controller}" != "ftg" ]]; then
  echo "SDU_APEX_CONTROLLER must be mpc, pure_pursuit, or ftg." >&2
  exit 2
fi

image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:dev}"
container="${SDU_APEX_CONTAINER:-sdu_apex_autodrive_dev}"
api_image="${AUTODRIVE_API_IMAGE:-autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7}"

if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Container ${container} already exists; stop it before starting another development stack." >&2
  exit 1
fi

echo "Starting the ROS 2 development/controller stack..."
if [[ "${launch_mode}" == "mapping" ]]; then
  map_name="${SDU_APEX_MAP_NAME:-track_map}"
  if [[ ! "${map_name}" =~ ^[[:alnum:]_-]+$ ]]; then
    echo "SDU_APEX_MAP_NAME may contain only letters, digits, underscores, and hyphens." >&2
    exit 2
  fi
  echo "Launch: mapping (${map_name})"
  echo "The bridge, odometry, SLAM mapper, mapping FTG, and actuator are one launch."
else
  echo "Controller: ${controller}"
  echo "The bridge, odometry, localization, controller, and actuator are one launch."
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

ros_domain_args=()
if [[ -n "${ROS_DOMAIN_ID:-}" ]]; then
  ros_domain_args+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi

map_yaml_override=""
trajectory_file_override=""
for path_setting in SDU_APEX_MAP_YAML SDU_APEX_TRAJECTORY_FILE; do
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
      else
        map_yaml_override="${container_path}"
      fi
      ;;
    *)
      echo "${path_setting} must be inside the repository bind mount." >&2
      exit 1
      ;;
  esac
done

if [[ "${AUTODRIVE_REBUILD:-1}" == "1" ]]; then
  docker build --network=host \
    --build-arg "AUTODRIVE_API_IMAGE=${api_image}" -t "${image}" .
elif ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Image ${image} is not available locally; rebuild or choose an existing SDU_APEX_IMAGE." >&2
  exit 1
fi

exec docker run --rm --name "${container}" \
  --network=host --ipc=host --privileged --gpus all \
  --log-opt max-size=10m --log-opt max-file=3 \
  "${ros_domain_args[@]}" \
  -e SDU_APEX_AUTOSTART=1 \
  -e "SDU_APEX_LAUNCH=${launch_mode}" \
  -e "SDU_APEX_MAP_NAME=${SDU_APEX_MAP_NAME:-track_map}" \
  -e SDU_APEX_CONTROLLER="${controller}" \
  -e SDU_APEX_WITH_RVIZ="${SDU_APEX_WITH_RVIZ:-false}" \
  -e SDU_APEX_MPC_PUBLISH_DIAGNOSTICS="${SDU_APEX_MPC_PUBLISH_DIAGNOSTICS:-false}" \
  -e SDU_APEX_BUILD_MPC="${SDU_APEX_BUILD_MPC:-0}" \
  -e "SDU_APEX_MPC_PARAMETER_OVERLAY=${mpc_parameter_overlay}" \
  -e "SDU_APEX_MAP_YAML=${map_yaml_override}" \
  -e "SDU_APEX_TRAJECTORY_FILE=${trajectory_file_override}" \
  -v "${repo_root}:/workspace/src:rw" \
  --entrypoint /bin/bash \
  "${image}" /workspace/src/docker/development_entrypoint.sh
