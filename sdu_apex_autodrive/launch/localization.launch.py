"""Run the map server, encoder/IMU odometry, and custom AMCL only.

This launch is deliberately separate from the racing controller.  It allows
map-frame localization to be brought up against an already-running official
bridge without starting a second bridge or sending drive commands.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


DEFAULT_MAP = "/workspace/src/f1tenth_planning/maps/autodrive_compete_2026.yaml"
DEFAULT_TRAJECTORY = (
    "/workspace/src/f1tenth_planning/trajectories/icra_2025_raceline.csv"
)


def _setup(context):
    localization = get_package_share_directory("f1tenth_localization")
    map_path = LaunchConfiguration("map").perform(context)

    return [
        Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
            name="autodrive_bridge",
            output="screen",
            condition=IfCondition(LaunchConfiguration("start_bridge")),
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
        LifecycleNode(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            namespace="",
            output="screen",
            parameters=[{"yaml_filename": map_path}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_map",
            output="screen",
            parameters=[{
                "autostart": True,
                "node_names": ["map_server"],
            }],
        ),
        Node(
            package="f1tenth_localization",
            executable="gpu_amcl_cpp_node",
            name="gpu_amcl_cpp",
            output="screen",
            parameters=[
                LaunchConfiguration("amcl_params"),
                {
                    "global_heading_trajectory_file": LaunchConfiguration(
                        "trajectory"
                    ),
                    "global_initialization": LaunchConfiguration(
                        "amcl_global_initialization"
                    ),
                    "global_pose_max_track_distance_m": LaunchConfiguration(
                        "amcl_max_track_distance"
                    ),
                    "initial_pose_heading_offset_rad": LaunchConfiguration(
                        "amcl_initial_heading_offset"
                    ),
                },
            ],
        ),
    ]


def generate_launch_description():
    localization = get_package_share_directory("f1tenth_localization")

    return LaunchDescription([
        DeclareLaunchArgument("map", default_value=DEFAULT_MAP),
        DeclareLaunchArgument("trajectory", default_value=DEFAULT_TRAJECTORY),
        DeclareLaunchArgument("start_bridge", default_value="false"),
        DeclareLaunchArgument("amcl_global_initialization", default_value="true"),
        DeclareLaunchArgument("amcl_max_track_distance", default_value="0.65"),
        DeclareLaunchArgument("amcl_initial_heading_offset", default_value="-0.09"),
        DeclareLaunchArgument(
            "sensor_odom_params",
            default_value=os.path.join(localization, "config", "sensor_odometry.yaml"),
        ),
        DeclareLaunchArgument(
            "amcl_params",
            default_value=os.path.join(localization, "config", "gpu_amcl_cpp_params.yaml"),
        ),
        OpaqueFunction(function=_setup),
    ])
