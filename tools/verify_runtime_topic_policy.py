#!/usr/bin/env python3
"""Static guard for the rules-compliant autonomous launch path."""

import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The maintained autonomous launch is subject to this runtime input policy.
RUNTIME_FILES = (
    "sdu_apex_autodrive/launch/competition.launch.py",
    "sdu_apex_autodrive/sdu_apex_autodrive/bridge_40hz.py",
    "sdu_apex_autodrive/sdu_apex_autodrive/actuator_interface.py",
    "f1tenth_localization/src/sensor_odometry_node.cpp",
    "f1tenth_localization/gpu_amcl_cpp/src/core/amcl_node.cpp",
    "f1tenth_localization/gpu_amcl_cpp/src/core/ekf_node.cpp",
    "f1tenth_mpc/src/mpc_controller_node.cpp",
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
    "/tf",
)

TF_LISTENER_TOKENS = (
    "tf2_ros::TransformListener",
    "create_subscription<tf2_msgs::msg::TFMessage>",
)

AUTONOMY_NODES = frozenset({
    'sensor_odometry', 'ekf_localization', 'gpu_amcl_cpp',
    'mpc_controller_node', 'autodrive_actuator_interface',
    'autodrive_bridge_command_input',
})
ALLOWED_AUTODRIVE_INPUTS = frozenset({
    '/autodrive/roboracer_1/lidar', '/autodrive/roboracer_1/front_camera',
    '/autodrive/roboracer_1/imu', '/autodrive/roboracer_1/left_encoder',
    '/autodrive/roboracer_1/right_encoder', '/autodrive/roboracer_1/steering',
    '/autodrive/roboracer_1/throttle',
    '/autodrive/roboracer_1/steering_command',
    '/autodrive/roboracer_1/throttle_command',
})


def subscription_errors(subscriptions: dict[str, list[str]]) -> list[str]:
    """Audit actual resolved endpoints, including disallowed debug aliases."""
    errors = []
    for node, topics in subscriptions.items():
        for topic in topics:
            if (topic in RESTRICTED or topic in ('/sdu/tf', '/sdu/tf_static') or
                    (topic.startswith('/autodrive/') and
                     topic not in ALLOWED_AUTODRIVE_INPUTS)):
                errors.append(f'{node}: forbidden runtime subscription {topic}')
    return errors


def verify_live_graph() -> int:
    """Inspect the running competition graph; do not subscribe to sensor data."""
    import time
    import rclpy

    rclpy.init()
    probe = rclpy.create_node('sdu_runtime_policy_audit')
    try:
        deadline = time.monotonic() + 8.0
        graph = []
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=.1)
            graph = probe.get_node_names_and_namespaces()
            if AUTONOMY_NODES.issubset(name for name, _ in graph):
                # Let endpoint discovery catch up after the node announcement.
                for _ in range(5):
                    rclpy.spin_once(probe, timeout_sec=.1)
                break
        counts = Counter(name for name, _ in graph)
        errors = [f'{name}: expected exactly one runtime node, found {counts[name]}'
                  for name in sorted(AUTONOMY_NODES) if counts[name] != 1]
        endpoints = {}
        for name, namespace in graph:
            if name not in AUTONOMY_NODES:
                continue
            endpoints[name] = [topic for topic, _ in
                               probe.get_subscriber_names_and_types_by_node(
                                   name, namespace)]
        errors.extend(subscription_errors(endpoints))
        for name, topics in sorted(endpoints.items()):
            print(f'{name}: {", ".join(sorted(topics))}')
        for error in errors:
            print(error)
        print('runtime subscription graph: ' + ('FAIL' if errors else 'PASS'))
        print('Topic checks do not certify organizer approval of the bridge wrapper.')
        return int(bool(errors))
    finally:
        probe.destroy_node()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true',
                        help='audit resolved subscriptions of the running competition stack')
    if parser.parse_args().live:
        return verify_live_graph()
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

    for relative in (
        "f1tenth_localization/src/sensor_odometry_node.cpp",
        "f1tenth_localization/gpu_amcl_cpp/src/core/amcl_node.cpp",
        "f1tenth_localization/gpu_amcl_cpp/src/core/ekf_node.cpp",
        "f1tenth_mpc/src/mpc_controller_node.cpp",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for token in TF_LISTENER_TOKENS:
            if token in text:
                errors.append(f"{relative}: restricted TF listener remains")

    competition_launch = (ROOT / COMPETITION_FILES[0]).read_text(encoding="utf-8")
    if '"SDU_APEX_DEV_SIM_RESET_ENABLED": "0"' not in competition_launch:
        errors.append(
            f"{COMPETITION_FILES[0]}: development simulator reset input must be disabled")
    if '"SDU_APEX_DEV_PACKET_DIAGNOSTICS_ENABLED": "0"' not in competition_launch:
        errors.append(
            f"{COMPETITION_FILES[0]}: development packet diagnostics must be disabled")
    if any(topic in competition_launch for topic in (
        '"/tf"', '"/tf_static"', '"/sdu/tf"', '"/sdu/tf_static"',
    )):
        errors.append(
            f"{COMPETITION_FILES[0]}: transform topic use or alias is forbidden")
    for required in ('"publish_tf": False',):
        if required not in competition_launch:
            errors.append(
                f"{COMPETITION_FILES[0]}: team transform output must be disabled")
    for token in (
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
    if "autodrive_bridge_40hz" not in competition_launch:
        errors.append(
            f"{COMPETITION_FILES[0]}: rate-controlled official bridge wrapper is missing")

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
