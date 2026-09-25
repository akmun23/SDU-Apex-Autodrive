"""Development-only SLAM mapping from simulator pose and LiDAR."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    control = get_package_share_directory("f1tenth_control")
    slam = get_package_share_directory("slam_toolbox")
    launch_bridge = IfCondition(LaunchConfiguration("launch_bridge"))

    map_saver = Node(
        package="sdu_apex_autodrive",
        executable="lap_map_saver",
        name="five_lap_map_saver",
        output="screen",
        parameters=[
            LaunchConfiguration("mapping_params"),
            {
                "output_directory": ParameterValue(
                    LaunchConfiguration("map_output_directory"), value_type=str),
                "map_name": ParameterValue(
                    LaunchConfiguration("map_name"), value_type=str),
            },
        ],
    )

    return LaunchDescription([
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
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
            default_value=os.path.join(control, "config", "ftg_params.yaml"),
        ),
        DeclareLaunchArgument(
            "mapping_params",
            default_value=os.path.join(integration, "config", "five_lap_mapping.yaml"),
        ),
        DeclareLaunchArgument(
            "map_output_directory",
            default_value="/workspace/src/live_runs/maps",
            description="Output directory for this mapped track; never the canonical map folder.",
        ),
        DeclareLaunchArgument(
            "map_name",
            default_value="track_map",
            description="Map filename stem for this mapping run.",
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(integration, "config", "actuator_interface.yaml"),
        ),
        DeclareLaunchArgument(
            "launch_bridge",
            default_value="true",
            description="Start the simulator bridge in this launch instance",
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
            name="autodrive_bridge",
            condition=launch_bridge,
            output="screen",
        ),
        # Keep team odometry live for comparison and bag analysis, but the
        # development-only map is registered against simulator ground truth.
        Node(
            package="f1tenth_localization",
            executable="sensor_odometry_node",
            name="sensor_odometry",
            output="screen",
            parameters=[LaunchConfiguration("sensor_odom_params")],
            remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
        ),
        # Keep team sensor transforms private to odometry. SLAM uses the
        # simulator's native world->vehicle->lidar transforms on /tf.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="mapping_base_to_lidar_tf",
            arguments=[
                "0.2733", "0.0", "0.096", "0.0", "0.0", "0.0",
                "base_link", "lidar",
            ],
            remappings=[("/tf_static", "/sdu/tf_static")],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="mapping_base_to_imu_tf",
            arguments=[
                "0.08", "0.0", "0.055", "0.0", "0.0", "0.0",
                "base_link", "imu",
            ],
            remappings=[("/tf_static", "/sdu/tf_static")],
            output="screen",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam, "launch", "online_async_launch.py")
            ),
            launch_arguments={
                "slam_params_file": LaunchConfiguration("slam_params"),
                "use_sim_time": "false",
            }.items(),
        ),
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
                    ],
                    remappings=[
                        ("scan", "/autodrive/roboracer_1/lidar"),
                        ("drive", "/cmd/speed"),
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
                # latch. Collision counters and reset commands are restricted
                # simulator topics and are never consumed here.
                {"external_stop_topic": "/sdu/mapping_complete",
                 "odom_topic": "/odom"},
            ],
        ),
        map_saver,
        # A collision makes the map run invalid. The map saver exits with a
        # failure code and this handler shuts down the remaining ROS nodes.
        RegisterEventHandler(
            OnProcessExit(
                target_action=map_saver,
                on_exit=[
                    EmitEvent(event=Shutdown(reason="mapping process exited"))
                ],
            )
        ),
    ])
