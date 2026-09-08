"""SLAM using allowed encoder/IMU odometry plus LiDAR."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    GroupAction,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node, SetRemap
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    control = get_package_share_directory("f1tenth_control")
    slam = get_package_share_directory("slam_toolbox")
    use_ftg = IfCondition(
        PythonExpression(["'", LaunchConfiguration("mapping_controller"), "' == 'ftg'"]))
    use_ground_truth_pp = IfCondition(
        PythonExpression(["'", LaunchConfiguration("mapping_controller"), "' == 'gt_pp'"]))
    launch_bridge = IfCondition(LaunchConfiguration("launch_bridge"))

    map_saver = Node(
        package="sdu_apex_autodrive",
        executable="lap_map_saver",
        name="five_lap_map_saver",
        output="screen",
        parameters=[LaunchConfiguration("mapping_params")],
    )

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
            default_value=os.path.join(control, "config", "ftg_params.yaml"),
        ),
        DeclareLaunchArgument(
            "mapping_params",
            default_value=os.path.join(integration, "config", "five_lap_mapping.yaml"),
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(integration, "config", "actuator_interface.yaml"),
        ),
        DeclareLaunchArgument("mapping_controller", default_value="ftg"),
        DeclareLaunchArgument(
            "launch_bridge",
            default_value="true",
            description="Start the simulator bridge in this launch instance",
        ),
        DeclareLaunchArgument(
            "trajectory_file",
            default_value=(
                "/workspace/src/f1tenth_planning/trajectories/"
                "autodrive_compete_2026_autodrive_sim_raceline.csv"
            ),
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
            name="autodrive_bridge",
            condition=launch_bridge,
            output="screen",
        ),
        # Mapping is deliberately isolated from the racing odometry TF tree.
        # The actuator and lap gate consume simulator odometry below, while
        # these are the only base-to-sensor transforms needed by SLAM.
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
        Node(
            package="sdu_apex_autodrive",
            executable="ground_truth_mapping_tf",
            name="ground_truth_mapping_tf",
            output="screen",
            parameters=[{
                "ground_truth_topic": "/autodrive/roboracer_1/odom",
                "odom_frame": "gt_odom",
                "base_frame": "base_link",
            }],
            remappings=[("/tf", "/sdu/tf")],
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
            condition=use_ftg,
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
                        ("odom", "/autodrive/roboracer_1/odom"),
                        ("drive", "/cmd/speed"),
                    ],
                ),
            ],
            output="screen",
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="ground_truth_path_pose",
            name="ground_truth_path_pose",
            condition=use_ground_truth_pp,
            output="screen",
            parameters=[{
                "ground_truth_topic": "/autodrive/roboracer_1/odom",
                "pose_topic": "/current_map_pose",
                "trajectory_file": LaunchConfiguration("trajectory_file"),
                "path_frame": "map",
            }],
        ),
        ComposableNodeContainer(
            name="mapping_pure_pursuit_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            condition=use_ground_truth_pp,
            composable_node_descriptions=[
                ComposableNode(
                    package="f1tenth_control",
                    plugin="f1tenth_control::PurePursuitNode",
                    name="mapping_ground_truth_pure_pursuit",
                    parameters=[
                        os.path.join(control, "config", "path_tracking_autodrive.yaml"),
                        {
                            "trajectory_file": LaunchConfiguration("trajectory_file"),
                            "odom_topic": "/autodrive/roboracer_1/odom",
                            "pose_topic": "/current_map_pose",
                            "command_topic": "/cmd/speed",
                            # Conservative mapping-only values.  The supplied
                            # raceline already supplies the geometric turn;
                            # extra curvature feed-forward double-counts it
                            # in the first tight bend.
                            "max_speed": 0.45,
                            "min_regulated_speed": 0.30,
                            "max_lateral_accel": 1.0,
                            "min_lookahead": 0.35,
                            "max_lookahead": 0.50,
                            "lookahead_gain": 0.03,
                            "curvature_feedforward_gain": 0.0,
                            "heading_error_gain": 0.0,
                            "steering_feedback_lead_gain": 0.0,
                            "speed_preview_distance": 0.0,
                            "speed_profile_braking_decel": 0.0,
                            "max_accel_cmd": 1.0,
                            "startup_path_max_distance_m": 1.0,
                            "startup_path_heading_tolerance_rad": 0.80,
                        },
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
                # A collision is a terminal mapping failure. Explicitly
                # disable reset publication so a crash cannot be hidden by
                # continuing from a simulator reset.
                {"external_stop_topic": "/sdu/mapping_complete",
                 "odom_topic": "/autodrive/roboracer_1/odom",
                 "collision_reset_enabled": False,
                 "collision_terminal_stop": True},
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
