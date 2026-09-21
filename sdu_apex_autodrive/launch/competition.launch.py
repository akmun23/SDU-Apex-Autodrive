"""Immutable autonomous launch used by the IROS 2026 competition image.

This launch has no development switches, shadow controller, recorder, RViz,
simulator telemetry input, or custom bridge.  The official API bridge is
included from the installed competition package; the team side consumes only
the allowed sensor topics and its own derived state.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node
from launch_ros.descriptions import ComposableNode


STARTUP_DELAY_AFTER_AMCL_S = 2.0


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    planning = get_package_share_directory("f1tenth_planning")
    mpc = get_package_share_directory("f1tenth_mpc")
    official_api = get_package_share_directory("autodrive_roboracer")

    map_path = os.path.join(
        planning, "maps", "autodrive_track_ftg_commit_20260909_025m.yaml")
    trajectory_path = os.path.join(
        planning, "trajectories", "autodrive_mintime_sim_5p0_dense",
        "autodrive_mintime_raceline.csv")
    mpc_params = os.path.join(mpc, "config", "mpc_iros_2026_competition.yaml")
    sensor_odom_params = os.path.join(
        localization, "config", "sensor_odometry.yaml")
    ekf_params = os.path.join(localization, "config", "ekf.yaml")
    amcl_params = os.path.join(
        localization, "config", "gpu_amcl_cpp_params.yaml")
    actuator_params = os.path.join(
        integration, "config", "actuator_interface.yaml")
    official_launch = os.path.join(
        official_api, "launch", "bringup_headless.launch.py")

    required_files = (
        map_path, trajectory_path, mpc_params, sensor_odom_params,
        ekf_params, amcl_params, actuator_params, official_launch,
    )
    missing = [path for path in required_files if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "competition launch is missing installed runtime file(s): "
            + ", ".join(missing))

    # This is the unchanged official bridge launch.  Do not replace it with
    # the development bridge or patch the installed official package.
    official_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(official_launch))

    sensor_odom = Node(
        package="f1tenth_localization",
        executable="sensor_odometry_node",
        name="sensor_odometry",
        output="screen",
        parameters=[sensor_odom_params],
        remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
    )
    ekf = Node(
        package="f1tenth_localization",
        executable="ekf_localization_node",
        name="ekf_localization",
        output="screen",
        parameters=[ekf_params],
    )
    map_server = LifecycleNode(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        namespace="",
        output="screen",
        parameters=[{"yaml_filename": map_path}],
    )
    map_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[{"autostart": True, "node_names": ["map_server"]}],
    )
    amcl = Node(
        package="f1tenth_localization",
        executable="gpu_amcl_cpp_node",
        name="gpu_amcl_cpp",
        output="screen",
        parameters=[
            amcl_params,
            {
                "global_heading_trajectory_file": trajectory_path,
                "odom_topic": "/ekf_odom",
                "global_initialization": True,
                "global_pose_max_track_distance_m": 0.65,
                "initial_pose_heading_offset_rad": 0.0,
            },
        ],
        remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
    )
    actuator = Node(
        package="sdu_apex_autodrive",
        executable="actuator_interface",
        name="autodrive_actuator_interface",
        output="screen",
        parameters=[
            actuator_params,
            {"input_topic": "/cmd/speed", "external_stop_topic": ""},
        ],
    )
    mpc_component = ComposableNode(
        package="f1tenth_mpc",
        plugin="f1tenth_mpc::MpcControllerNode",
        name="mpc_controller_node",
        parameters=[
            mpc_params,
            {
                "trajectory_file": trajectory_path,
                "max_speed_mps": 16.0,
                "publish_diagnostics": False,
                "odom_topic": "/odom",
                "pose_topic": "/current_map_pose",
                "command_topic": "/cmd/speed",
                "diagnostics_topic": "/mpc/diagnostics",
            },
        ],
        remappings=[
            ("odom", "/odom"),
            ("pose", "/current_map_pose"),
            ("drive", "/cmd/speed"),
        ],
    )
    mpc_container = ComposableNodeContainer(
        name="controller_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[mpc_component],
        output="screen",
    )

    # AMCL is allowed to establish a valid global hypothesis before MPC is
    # composed.  MPC also retains its own valid-state/startup-path gate.
    delayed_mpc = RegisterEventHandler(
        OnProcessStart(
            target_action=amcl,
            on_start=[
                TimerAction(
                    period=STARTUP_DELAY_AFTER_AMCL_S,
                    actions=[
                        LogInfo(msg=(
                            "Competition MPC starting after AMCL warm-up "
                            f"({STARTUP_DELAY_AFTER_AMCL_S:.1f} s)")),
                        mpc_container,
                    ],
                )
            ],
        )
    )

    return LaunchDescription([
        official_bridge,
        sensor_odom,
        ekf,
        map_server,
        map_lifecycle,
        amcl,
        actuator,
        delayed_mpc,
    ])
