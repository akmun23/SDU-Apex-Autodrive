"""SLAM using allowed encoder/IMU odometry plus LiDAR."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node, SetRemap
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    control = get_package_share_directory("f1tenth_control")
    slam = get_package_share_directory("slam_toolbox")

    return LaunchDescription([
        DeclareLaunchArgument(
            "sensor_odom_params",
            default_value=os.path.join(localization, "config", "sensor_odometry.yaml"),
        ),
        DeclareLaunchArgument(
            "slam_params",
            default_value=os.path.join(integration, "config", "slam_params.yaml"),
        ),
        DeclareLaunchArgument(
            "ftg_params",
            default_value=os.path.join(control, "config", "ftg_autodrive.yaml"),
        ),
        DeclareLaunchArgument(
            "mapping_ftg_params",
            default_value=os.path.join(integration, "config", "mapping_ftg.yaml"),
        ),
        DeclareLaunchArgument(
            "mapping_params",
            default_value=os.path.join(integration, "config", "five_lap_mapping.yaml"),
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(integration, "config", "actuator_interface.yaml"),
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
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
        GroupAction([
            SetRemap(src="/tf", dst="/sdu/tf"),
            SetRemap(src="/tf_static", dst="/sdu/tf_static"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam, "launch", "online_async_launch.py")
                ),
                launch_arguments={
                    "slam_params_file": LaunchConfiguration("slam_params"),
                    "use_sim_time": "false",
                }.items(),
            ),
        ]),
        ComposableNodeContainer(
            name="mapping_ftg_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="f1tenth_control",
                    plugin="f1tenth_control::FTGNode",
                    name="ftg_node",
                    parameters=[
                        LaunchConfiguration("ftg_params"),
                        LaunchConfiguration("mapping_ftg_params"),
                    ],
                ),
            ],
            output="screen",
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="actuator_interface",
            name="autodrive_actuator_interface",
            output="screen",
            parameters=[
                LaunchConfiguration("actuator_params"),
                # Mapping is the only workflow that uses the completion
                # latch; racing launches leave this optional hook disabled.
                {"external_stop_topic": "/sdu/mapping_complete"},
            ],
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="lap_map_saver",
            name="five_lap_map_saver",
            output="screen",
            parameters=[LaunchConfiguration("mapping_params")],
        ),
    ])
