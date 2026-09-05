"""Calibration launcher with a mandatory live AutoDRIVE timing gate."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


RAW = {
    "throttle_sweep", "throttle_steps", "zero_throttle_decel", "steering_steps",
    "steering_response", "throttle_speed_grid", "identification_grid",
}
SPEED = {"speed_steps", "speed_ramp"}
ACCELERATION = {"acceleration_steps"}
ALL = RAW | SPEED | ACCELERATION | {"sensor_record", "full_suite", "timing_only"}


def _setup(context):
    mode = LaunchConfiguration("test").perform(context)
    if mode not in ALL:
        raise RuntimeError(f"unknown test mode: {mode}")

    bridge_node = Node(
        package="sdu_apex_autodrive",
        executable="autodrive_bridge_40hz",
        name="autodrive_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_bridge")),
    )
    timing_validator = Node(
        package="sdu_apex_autodrive",
        executable="timing_validator",
        name="autodrive_timing_validator",
        output="screen",
        parameters=[{
            "duration_sec": LaunchConfiguration("timing_gate_duration_sec"),
            "startup_timeout_sec": LaunchConfiguration("timing_gate_startup_timeout_sec"),
            "report_path": LaunchConfiguration("timing_gate_report_path"),
        }],
    )

    def after_timing_gate(event, _context):
        if event.returncode != 0:
            return [Shutdown(
                reason=f"AutoDRIVE timing gate failed with exit code {event.returncode}")]
        if mode == "timing_only":
            return [Shutdown(reason="AutoDRIVE timing gate passed")]

        runtime_actions = [
            Node(
                package="f1tenth_localization",
                executable="sensor_odometry_node",
                name="sensor_odometry",
                output="screen",
                parameters=[
                    LaunchConfiguration("sensor_odom_params"),
                    {"reset_enabled": mode in {
                        "identification_grid", "speed_steps", "speed_ramp",
                        "acceleration_steps", "full_suite"},
                     "reset_topic": "/autodrive/reset_command"},
                ],
                remappings=[
                    ("/tf", "/sdu/tf"),
                    ("/tf_static", "/sdu/tf_static"),
                ],
            ),
        ]

        if mode == "identification_grid":
            runtime_actions.append(Node(
                package="f1tenth_localization",
                executable="ekf_localization_node",
                name="ekf_localization",
                output="screen",
                parameters=[
                    LaunchConfiguration("ekf_params"),
                    {"reset_enabled": True, "reset_topic": "/autodrive/reset_command"},
                ],
            ))

        if mode in SPEED or mode in ACCELERATION or mode == "full_suite":
            runtime_actions.append(Node(
                package="sdu_apex_autodrive",
                executable="actuator_interface",
                name="autodrive_actuator_interface",
                output="screen",
                parameters=[LaunchConfiguration("actuator_params"), {
                    "input_topic": "/cmd/acceleration"
                    if mode in ACCELERATION else "/cmd/speed",
                    "command_mode": "acceleration"
                    if mode in ACCELERATION else "speed",
                    "allow_raw_throttle_override": True,
                }],
            ))

        calibration_node = Node(
            package="sdu_apex_autodrive",
            executable="calibration",
            name="calibration",
            output="screen",
            parameters=[
                LaunchConfiguration("calibration_params"),
                {
                    "mode": mode,
                    "output_dir": LaunchConfiguration("output_dir"),
                },
            ],
        )
        runtime_actions.append(calibration_node)
        runtime_actions.append(RegisterEventHandler(OnProcessExit(
            target_action=calibration_node,
            on_exit=[Shutdown(reason="calibration completed")],
        )))
        return runtime_actions

    return [
        bridge_node,
        timing_validator,
        RegisterEventHandler(OnProcessExit(
            target_action=timing_validator,
            on_exit=after_timing_gate,
        )),
    ]


def generate_launch_description():
    share = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    return LaunchDescription([
        DeclareLaunchArgument("test", default_value="timing_only"),
        DeclareLaunchArgument("start_bridge", default_value="true"),
        DeclareLaunchArgument("timing_gate_duration_sec", default_value="15.0"),
        DeclareLaunchArgument("timing_gate_startup_timeout_sec", default_value="30.0"),
        DeclareLaunchArgument(
            "timing_gate_report_path",
            default_value="/tmp/sdu_autodrive_timing_report.json",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/workspace/src/sdu_apex_autodrive/artifacts/calibration/raw",
        ),
        DeclareLaunchArgument(
            "calibration_params",
            default_value=os.path.join(share, "config", "calibration.yaml"),
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(share, "config", "actuator_interface.yaml"),
        ),
        DeclareLaunchArgument(
            "sensor_odom_params",
            default_value=os.path.join(localization, "config", "sensor_odometry.yaml"),
        ),
        DeclareLaunchArgument(
            "ekf_params",
            default_value=os.path.join(localization, "config", "ekf.yaml"),
        ),
        OpaqueFunction(function=_setup),
    ])
