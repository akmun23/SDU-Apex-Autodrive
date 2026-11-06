"""Build a new map using only live sensors and sensor-derived odometry."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable, RegisterEventHandler, EmitEvent,
                            TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    control = get_package_share_directory("f1tenth_control")
    slam = get_package_share_directory("slam_toolbox")
    supervisor = Node(
        package="sdu_apex_autodrive", executable="mapping_supervisor",
        name="mapping_supervisor", output="screen", parameters=[{
            "output_directory": LaunchConfiguration("output_directory"),
            "map_prefix": LaunchConfiguration("map_prefix"),
        }])

    return LaunchDescription([
        DeclareLaunchArgument("output_directory", default_value="/mapping"),
        DeclareLaunchArgument("map_prefix", default_value=(
            "/workspace/src/f1tenth_planning/maps/autodrive_iros_2026_compete")),
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
        RegisterEventHandler(OnProcessExit(target_action=supervisor,
            on_exit=[EmitEvent(event=Shutdown(reason="Mapping supervisor finished"))])),
        supervisor,
        Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
            name="autodrive_bridge",
            output="screen",
            # The official bridge also emits Unity's world-pose TF. Keep it
            # available for diagnostics but out of SLAM's sensor-derived tree.
            remappings=[
                ("/tf", "/autodrive_ground_truth/tf"),
                ("/tf_static", "/autodrive_ground_truth/tf_static"),
            ],
        ),
        Node(
            package="f1tenth_localization",
            executable="sensor_odometry_node",
            name="sensor_odometry",
            output="screen",
            parameters=[os.path.join(
                localization, "config", "sensor_odometry.yaml")],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                slam, "launch", "online_async_launch.py")),
            launch_arguments={
                "slam_params_file": os.path.join(
                    integration, "config", "slam_mapping.yaml"),
                "use_sim_time": "false",
            }.items(),
        ),
        TimerAction(period=3.0, actions=[ComposableNodeContainer(
            name="mapping_ftg_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[ComposableNode(
                package="f1tenth_control",
                plugin="f1tenth_control::FTGNode",
                name="ftg_node",
                parameters=[
                    os.path.join(control, "config", "ftg_mapping.yaml"),
                ],
            )],
            output="screen",
        )]),
        Node(
            package="sdu_apex_autodrive",
            executable="actuator_interface",
            name="autodrive_actuator_interface",
            output="screen",
            parameters=[
                os.path.join(integration, "config", "actuator_interface.yaml"),
                # FTG's emergency result must be able to request the
                # simulator's explicit hard-brake mode while mapping.
                {"minimum_throttle_normalized": 0.0},
            ],
        ),
    ])
