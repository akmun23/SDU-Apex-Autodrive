#!/usr/bin/env python3
"""Static guard for the rules-compliant autonomous launch path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# These files constitute the normal autonomous launch and its runtime nodes.
RUNTIME_FILES = (
    "sdu_apex_autodrive/launch/controller.launch.py",
    "sdu_apex_autodrive/launch/localization.launch.py",
    "sdu_apex_autodrive/launch/mapping.launch.py",
    "sdu_apex_autodrive/sdu_apex_autodrive/bridge_40hz.py",
    "sdu_apex_autodrive/sdu_apex_autodrive/actuator_interface.py",
    "sdu_apex_autodrive/sdu_apex_autodrive/lap_map_saver.py",
    "f1tenth_localization/src/sensor_odometry_node.cpp",
    "f1tenth_localization/gpu_amcl_cpp/src/core/ekf_node.cpp",
    "f1tenth_localization/gpu_amcl_cpp/include/gpu_amcl_cpp/core/ekf_node.hpp",
    "f1tenth_localization/config/sensor_odometry.yaml",
    "f1tenth_localization/config/ekf.yaml",
)

COMPETITION_FILES = (
    "sdu_apex_autodrive/launch/competition.launch.py",
    "docker/competition_entrypoint.sh",
)

# These are prohibited subscription endpoints, not merely strings that may
# occur in historical analysis documentation.
RESTRICTED = (
    "/autodrive/reset_command",
    "/autodrive/roboracer_1/collision_count",
    "/autodrive/roboracer_1/ips",
    "/autodrive/roboracer_1/odom",
    "/autodrive/roboracer_1/lap_count",
    "/autodrive/roboracer_1/lap_time",
    "/autodrive/roboracer_1/last_lap_time",
    "/autodrive/roboracer_1/best_lap_time",
)


def main() -> int:
    errors = []
    for relative in RUNTIME_FILES:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        if relative.endswith("bridge_40hz.py"):
            if "_disable_restricted_bridge_subscriptions" not in text:
                errors.append(f"{relative}: bridge reset guard is missing")
            continue
        for topic in RESTRICTED:
            if topic in text:
                errors.append(f"{relative}: restricted topic remains: {topic}")

    competition_launch = (ROOT / COMPETITION_FILES[0]).read_text(encoding="utf-8")
    for token in (
        "autodrive_bridge_40hz",
        "mpc_shadow_node",
        "with_mpc_shadow",
        "mpc_override_params",
        "model_id_timing_recorder",
        "ground_truth_",
        "rviz",
    ):
        if token in competition_launch:
            errors.append(
                f"{COMPETITION_FILES[0]}: development-only token remains: {token}")
    if "bringup_headless.launch.py" not in competition_launch:
        errors.append(
            f"{COMPETITION_FILES[0]}: official headless API launch is not included")

    competition_entrypoint = (ROOT / COMPETITION_FILES[1]).read_text(encoding="utf-8")
    if "exec ros2 launch sdu_apex_autodrive competition.launch.py" not in competition_entrypoint:
        errors.append(
            f"{COMPETITION_FILES[1]}: entrypoint is not the fixed competition launch")

    if errors:
        for error in errors:
            print(error)
        return 1
    print("runtime topic policy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
