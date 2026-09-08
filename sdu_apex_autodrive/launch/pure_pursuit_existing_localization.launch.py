"""Run Pure Pursuit against an already-running simulator/localization stack.

This launch deliberately does not start the bridge, odometry, map server, or
AMCL.  It is for controlled live experiments where those nodes are already
running and must not be duplicated.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


DEFAULT_TRAJECTORY = (
    "/workspace/src/f1tenth_planning/trajectories/"
    "autodrive_track_ftg_commit_20260908_lap01_mintime_raceline.csv"
)


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    control = get_package_share_directory("f1tenth_control")

    pure_pursuit = ComposableNode(
        package="f1tenth_control",
        plugin="f1tenth_control::PurePursuitNode",
        name="pure_pursuit_node",
        parameters=[
            LaunchConfiguration("path_tracking_params"),
            {
                "trajectory_file": LaunchConfiguration("trajectory"),
                "max_speed": LaunchConfiguration("controller_max_speed"),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("trajectory", default_value=DEFAULT_TRAJECTORY),
        DeclareLaunchArgument(
            "controller_max_speed",
            default_value="1.5",
            description=(
                "Safe simulator startup cap in m/s. Raise explicitly only "
                "after the baseline follows the planned raceline."
            ),
        ),
        DeclareLaunchArgument(
            "path_tracking_params",
            default_value=os.path.join(
                control, "config", "path_tracking_autodrive.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(
                integration, "config", "actuator_interface.yaml"
            ),
        ),
        DeclareLaunchArgument(
            "with_collision_safety",
            default_value="false",
            description="Enable simulator-only collision reset for diagnostics",
        ),
        ComposableNodeContainer(
            name="controller_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[pure_pursuit],
            output="screen",
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="actuator_interface",
            name="autodrive_actuator_interface",
            output="screen",
            parameters=[
                LaunchConfiguration("actuator_params"),
                {
                    "input_topic": "/cmd/speed",
                    "command_mode": "speed",
                    "collision_reset_enabled": LaunchConfiguration(
                        "with_collision_safety"
                    ),
                },
            ],
        ),
    ])
