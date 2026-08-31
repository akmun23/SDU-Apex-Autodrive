"""Calibration launcher. No arming and no guarded-run wrapper."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


RAW = {
    "throttle_sweep", "throttle_steps", "zero_throttle_decel", "steering_steps",
    "steering_response", "throttle_speed_grid",
}
SPEED = {"speed_steps", "speed_ramp"}
ALL = RAW | SPEED | {"sensor_record", "full_suite"}


def _setup(context):
    mode = LaunchConfiguration("test").perform(context)
    if mode not in ALL:
        raise RuntimeError(f"unknown test mode: {mode}")

    actions = [
        Node(
            package="autodrive_roboracer",
            executable="autodrive_bridge",
            name="autodrive_bridge",
            output="screen",
        ),
        Node(
            package="f1tenth_localization",
            executable="sensor_odometry_node",
            name="sensor_odometry",
            output="screen",
            parameters=[LaunchConfiguration("sensor_odom_params")],
            remappings=[
                ("/tf", "/sdu/tf"),
                ("/tf_static", "/sdu/tf_static"),
            ],
        ),
    ]

    if mode in SPEED or mode == "full_suite":
        actions.append(Node(
            package="sdu_apex_autodrive",
            executable="actuator_interface",
            name="autodrive_actuator_interface",
            output="screen",
            parameters=[LaunchConfiguration("actuator_params")],
        ))

    actions.append(Node(
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
    ))
    return actions


def generate_launch_description():
    share = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    return LaunchDescription([
        DeclareLaunchArgument("test", default_value="sensor_record"),
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
        OpaqueFunction(function=_setup),
    ])
