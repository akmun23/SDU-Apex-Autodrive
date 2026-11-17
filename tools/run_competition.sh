#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/tools/docker_env.sh"

image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:competition}"
container="${SDU_APEX_CONTAINER:-sdu_apex_autodrive_competition}"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  echo "Image ${image} is not built locally; run ./tools/build_competition.sh first." >&2
  exit 1
fi

if docker container inspect "${container}" >/dev/null 2>&1; then
  echo "Container ${container} already exists; stop it before starting another competition run." >&2
  exit 1
fi


# Match the proven development run's CPU isolation on this known host topology.
# Docker cpuset alone is silently discarded by the host's rootless runtime.
# On other machines use the portable unrestricted profile.
competition_runtime_cpus=""
competition_mpc_cpus=""
competition_openmp_args=()
if [[ "${SDU_APEX_COMPETITION_CPU_ISOLATION:-auto}" == auto ]]; then
  topology_ok=true
  for cpu in 0 2 4 6 8 10 12 13 14 15; do
    siblings="/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list"
    if [[ ! -r "$siblings" ]]; then topology_ok=false; break; fi
    if (( cpu < 12 )); then expected="${cpu}-$((cpu + 1))"; else
      if (( cpu < 14 )); then expected=12-13; else expected=14-15; fi
    fi
    if [[ "$(<"$siblings")" != "$expected" ]]; then
      topology_ok=false; break
    fi
  done
  if [[ "$topology_ok" == true ]]; then
    competition_runtime_cpus=9,11
    competition_mpc_cpus=0,2,4,6,8,10
    competition_openmp_args=(-e OMP_NUM_THREADS=6 -e OMP_PROC_BIND=spread
      -e 'OMP_PLACES={0},{2},{4},{6},{8},{10}'
      -e OMP_WAIT_POLICY=active -e GOMP_SPINCOUNT=1000000)
  fi
elif [[ "${SDU_APEX_COMPETITION_CPU_ISOLATION}" != off ]]; then
  echo 'SDU_APEX_COMPETITION_CPU_ISOLATION must be auto or off' >&2
  exit 2
fi


# Optional host-only tuning overlay. The production image is not rebuilt or
# modified; the mounted YAML is read at component startup. Never enable this
# override from the official competition runner unless explicitly requested.
tuner_args=()
if [[ -n "${SDU_APEX_MPC_CONFIG_OVERRIDE:-}" ]]; then
  config_override="$(realpath -- "${SDU_APEX_MPC_CONFIG_OVERRIDE}")"
  if [[ ! -f "$config_override" || ! -r "$config_override" ]]; then
    echo "MPC override must be a readable regular file: $config_override" >&2
    exit 2
  fi
  tuner_args+=(--mount "type=bind,source=$config_override,target=/workspace/install/share/f1tenth_mpc/config/mpc_competition.yaml,readonly")
fi
if [[ -n "${SDU_APEX_TUNER_SESSION:-}" ]]; then
  if [[ ! "${SDU_APEX_TUNER_SESSION}" =~ ^[a-zA-Z0-9_-]{8,80}$ ]]; then
    echo 'Invalid tuner session identifier' >&2
    exit 2
  fi
  tuner_args+=(--label "sdu.apex.tuner.session=${SDU_APEX_TUNER_SESSION}")
fi

exec docker run --rm --log-driver=none --name "${container}" \
  --network=host --ipc=host --privileged --gpus all \
  --tmpfs /run/ros:rw,nosuid,nodev,noexec,size=32m \
  "${tuner_args[@]}" \
  -e ROS_HOME=/run/ros \
  -e "SDU_APEX_COMPETITION_RUNTIME_CPUS=${competition_runtime_cpus}" \
  -e "SDU_APEX_MPC_CPUSET_CPUS=${competition_mpc_cpus}" \
  "${competition_openmp_args[@]}" \
  "${image}"
