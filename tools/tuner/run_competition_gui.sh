#!/usr/bin/env bash
# Host-side single-command graphical IROS test of the cleaned competition image.
# Unity's Connect button is pressed by the operator. AMCL/MPC wait for fresh
# ROS samples in competition.launch.py; never use a fixed startup sleep.
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/tools/docker_env.sh"
image="${SDU_APEX_IMAGE:-sdu-apex-autodrive:competition}"
controller="${SDU_APEX_CONTAINER:-sdu_apex_autodrive_competition}"
sim="${SDU_APEX_SIM_CONTAINER:-sdu_apex_sim_iros_gui}"
simulator_image='autodriveecosystem/autodrive_roboracer_sim@sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d'
# Avoid tying the Unity graphics workload to a full-desktop resolution.
sim_width="${SDU_APEX_SIM_WINDOW_WIDTH:-1280}"
sim_height="${SDU_APEX_SIM_WINDOW_HEIGHT:-720}"
if [[ ! "$sim_width" =~ ^[0-9]+$ || ! "$sim_height" =~ ^[0-9]+$ ]] ||
        (( sim_width < 640 || sim_width > 1920 || sim_height < 480 || sim_height > 1080 )); then
    echo 'Unity GUI dimensions must be between 640x480 and 1920x1080.' >&2
    exit 2
fi

if command -v powerprofilesctl >/dev/null &&
        [[ "$(powerprofilesctl get)" != performance ]]; then
    echo 'GUI test requires the host Performance power profile.' >&2
    exit 2
fi
if [[ -z "${DISPLAY:-}" || -z "${XAUTHORITY:-}" || ! -r "$XAUTHORITY" ]]; then
    echo 'An active graphical X11 desktop session and XAUTHORITY are required.' >&2
    exit 2
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Competition image missing: $image" >&2
    exit 2
fi
for name in "$controller" "$sim"; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "Container $name exists; stop the prior test first." >&2
        exit 2
    fi
done
if ss -lnt | grep -Eq ':4567[[:space:]]'; then
    echo 'Bridge port 4567 is already occupied.' >&2
    exit 2
fi

controller_started=false
sim_started=false
cleanup() {
    trap - EXIT INT TERM
    if "$sim_started"; then docker stop --timeout 3 "$sim" >/dev/null 2>&1 || true; fi
    if "$controller_started"; then docker stop --timeout 3 "$controller" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

cd "$repo_root"
SDU_APEX_IMAGE="$image" SDU_APEX_CONTAINER="$controller" ./tools/run_competition.sh &
controller_pid=$!
controller_started=true
ready=false
for _ in $(seq 1 90); do
    if ! kill -0 "$controller_pid" >/dev/null 2>&1; then
        echo 'Competition controller exited before bridge startup.' >&2
        exit 1
    fi
    if docker ps --format '{{.Names}}' | grep -Fxq "$controller" &&
            ss -lnt | grep -Eq ':4567[[:space:]]'; then
        ready=true
        break
    fi
    sleep 1
done
if ! "$ready"; then
    echo 'Competition bridge failed to start.' >&2
    exit 1
fi

# Docker -d is required: GUI mode must not allocate a terminal. This is the
# official IROS player with -batchmode omitted, not a modified simulator.
authority_file="$(realpath "$XAUTHORITY")"
docker run -d --rm --log-driver=none --name "$sim" \
    --network=host --ipc=host --privileged --gpus all --ulimit core=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    -e "DISPLAY=$DISPLAY" -e "XAUTHORITY=$authority_file" \
    -v "$authority_file:$authority_file:ro" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -w /home/autodrive_simulator --entrypoint /bin/bash \
    "$simulator_image" -lc \
    "if command -v taskset >/dev/null; then taskset --pid --cpu-list 12-15 \$\$ >/dev/null || exit 1; fi; exec './AutoDRIVE Simulator.x86_64' -ip 127.0.0.1 -port 4567 -logFile /dev/null -screen-width ${sim_width} -screen-height ${sim_height} -screen-fullscreen 0" >/dev/null
sim_started=true

echo 'Clean competition controller and graphical IROS Unity are running.'
echo 'Press Connect in Unity. AMCL waits for fresh IMU/odometry; MPC waits for AMCL.'
echo 'Close Unity or press Ctrl+C here to stop this test.'
while docker ps --format '{{.Names}}' | grep -Fxq "$sim"; do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "$controller"; then
        echo 'Competition controller stopped unexpectedly.' >&2
        exit 1
    fi
    sleep 2
done
echo 'Unity exited; stopping the competition controller.'
