"""Immutable autonomous launch used by the competition container.

This launch has no development switches, shadow controller, recorder, RViz,
or simulator telemetry input.  The repository bridge wrapper runs the
official API bridge with the fixed numeric telemetry pacing required by the
competition simulator; the team side consumes only allowed sensor topics and
its own derived state.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo, RegisterEventHandler, SetEnvironmentVariable, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node
from launch_ros.descriptions import ComposableNode


STARTUP_DELAY_AFTER_AMCL_S = 2.0


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    planning = get_package_share_directory("f1tenth_planning")
    mpc = get_package_share_directory("f1tenth_mpc")

    map_relative_path = os.environ.get(
        "SDU_APEX_MAP_REL",
        "maps/autodrive_track_ftg_commit_20260909_025m.yaml")
    trajectory_relative_path = os.environ.get(
        "SDU_APEX_TRAJECTORY_REL",
        "trajectories/autodrive_mintime_sim_5p0_dense/autodrive_mintime_raceline.csv")
    if (os.path.isabs(map_relative_path) or ".." in map_relative_path.split(os.sep) or
            os.path.isabs(trajectory_relative_path) or
            ".." in trajectory_relative_path.split(os.sep)):
        raise RuntimeError("competition track assets must be relative to f1tenth_planning")
    map_path = os.path.normpath(os.path.join(planning, map_relative_path))
    trajectory_path = os.path.normpath(
        os.path.join(planning, trajectory_relative_path))
    mpc_params = os.path.join(mpc, "config", "mpc_competition.yaml")
    sensor_odom_params = os.path.join(
        localization, "config", "sensor_odometry.yaml")
    ekf_params = os.path.join(localization, "config", "ekf.yaml")
    amcl_params = os.path.join(
        localization, "config", "gpu_amcl_cpp_params.yaml")
    actuator_params = os.path.join(
        integration, "config", "actuator_interface.yaml")

    required_files = (
        map_path, trajectory_path, mpc_params, sensor_odom_params,
        ekf_params, amcl_params, actuator_params,
    )
    missing = [path for path in required_files if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "competition launch is missing installed runtime file(s): "
            + ", ".join(missing))

    # The wrapper imports the installed official bridge, but paces numeric
    # packets independently of decoder/publication latency.  It also removes
    # the stock restricted reset subscription before creating the bridge node.
    bridge = Node(
        package="sdu_apex_autodrive",
        executable="autodrive_bridge_40hz",
        name="autodrive_bridge",
        output="screen",
        emulate_tty=True,
    )

    sensor_odom = Node(
        package="f1tenth_localization",
        executable="sensor_odometry_node",
        name="sensor_odometry",
        output="screen",
        parameters=[sensor_odom_params, {"publish_tf": False}],
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
                "publish_tf": False,
            },
        ],
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
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
        bridge,
        sensor_odom,
        ekf,
        map_server,
        map_lifecycle,
        amcl,
        actuator,
        delayed_mpc,
    ])
