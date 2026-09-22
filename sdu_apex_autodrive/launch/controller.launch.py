"""Local development launch for the legal AutoDRIVE controller stack.

The launch exposes the three maintained driving modes: FTG for mapping,
Pure Pursuit as a baseline, and the production MPC.  There is one installed
raceline and no shadow controller, simulator-truth recorder, or parameter
overlay.
"""

import math
import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


MAP_NAME = "autodrive_track_ftg_commit_20260909_025m.yaml"
RACELINE_PATH = os.path.join(
    "autodrive_mintime_sim_5p0_dense", "autodrive_mintime_raceline.csv")


def _bool(value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"expected boolean, got {value!r}")


def _reject_existing_runtime_nodes(expected_names: set[str]) -> None:
    """Reject duplicate local publishers before starting a second stack."""
    try:
        result = subprocess.run(
            ["ros2", "node", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("unable to verify that no previous stack is running") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "ros2 node list failed during controller preflight: "
            f"{result.stderr.strip()}")
    existing_names = {
        line.strip().rsplit("/", 1)[-1]
        for line in result.stdout.splitlines()
        if line.strip()
    }
    duplicates = sorted(expected_names.intersection(existing_names))
    if duplicates:
        raise RuntimeError(
            "controller preflight found existing runtime node(s): "
            f"{', '.join(duplicates)}. Stop the previous stack first.")


def _setup(context):
    controller = LaunchConfiguration("controller").perform(context).lower()
    if controller not in ("ftg", "pure_pursuit", "mpc"):
        raise RuntimeError("controller must be ftg, pure_pursuit, or mpc")

    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    planning = get_package_share_directory("f1tenth_planning")
    control = get_package_share_directory("f1tenth_control")
    mpc = get_package_share_directory("f1tenth_mpc")

    map_path = os.path.join(planning, "maps", MAP_NAME)
    trajectory = os.path.join(planning, "trajectories", RACELINE_PATH)
    with_rviz = _bool(LaunchConfiguration("with_rviz").perform(context))
    use_localization = _bool(
        LaunchConfiguration("use_localization").perform(context))
    force_localization = _bool(
        LaunchConfiguration("force_localization").perform(context))
    needs_localization = use_localization and (
        controller != "ftg" or force_localization)
    mpc_start_delay_sec = float(
        LaunchConfiguration("mpc_start_delay_sec").perform(context))
    if not math.isfinite(mpc_start_delay_sec) or mpc_start_delay_sec < 0.0:
        raise RuntimeError("mpc_start_delay_sec must be finite and non-negative")

    if needs_localization:
        for path in (map_path, trajectory):
            if not os.path.isfile(path):
                raise RuntimeError(f"required installed file does not exist: {path}")

    expected_names = {
        "autodrive_bridge",
        "sensor_odometry",
        "controller_container",
        "autodrive_actuator_interface",
    }
    if needs_localization:
        expected_names.update({
            "ekf_localization", "map_server", "lifecycle_manager_map",
            "gpu_amcl_cpp",
        })
    _reject_existing_runtime_nodes(expected_names)

    bridge = Node(
        package="sdu_apex_autodrive",
        executable="autodrive_bridge_40hz",
        name="autodrive_bridge",
        output="screen",
        emulate_tty=True,
    )
    odometry = Node(
        package="f1tenth_localization",
        executable="sensor_odometry_node",
        name="sensor_odometry",
        output="screen",
        parameters=[os.path.join(localization, "config", "sensor_odometry.yaml")],
        remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
    )
    actions = [SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
               bridge, odometry]
    amcl_node = None

    if needs_localization:
        map_server = LifecycleNode(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            namespace="",
            output="screen",
            parameters=[{"yaml_filename": map_path}],
        )
        actions.extend([
            Node(
                package="f1tenth_localization",
                executable="ekf_localization_node",
                name="ekf_localization",
                output="screen",
                parameters=[os.path.join(localization, "config", "ekf.yaml")],
            ),
            map_server,
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_map",
                output="screen",
                parameters=[{"autostart": True, "node_names": ["map_server"]}],
            ),
        ])
        amcl_node = Node(
            package="f1tenth_localization",
            executable="gpu_amcl_cpp_node",
            name="gpu_amcl_cpp",
            output="screen",
            parameters=[
                os.path.join(localization, "config", "gpu_amcl_cpp_params.yaml"),
                {
                    "global_heading_trajectory_file": trajectory,
                    "odom_topic": "/ekf_odom",
                    "global_initialization": True,
                    "global_pose_max_track_distance_m": 0.65,
                    "initial_pose_heading_offset_rad": 0.0,
                },
            ],
            remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
        )
        actions.append(amcl_node)

    if controller == "ftg":
        controller_component = ComposableNode(
            package="f1tenth_control",
            plugin="f1tenth_control::FTGNode",
            name="ftg_node",
            parameters=[
                os.path.join(control, "config", "ftg_params.yaml"),
                {"max_speed": 0.40},
            ],
            remappings=[
                ("scan", "/autodrive/roboracer_1/lidar"),
                ("drive", "/cmd/speed"),
            ],
        )
    elif controller == "pure_pursuit":
        controller_component = ComposableNode(
            package="f1tenth_control",
            plugin="f1tenth_control::PurePursuitNode",
            name="pure_pursuit_node",
            parameters=[
                os.path.join(control, "config", "path_tracking_autodrive.yaml"),
                {"trajectory_file": trajectory, "max_speed": 16.0},
            ],
        )
    else:
        controller_component = ComposableNode(
            package="f1tenth_mpc",
            plugin="f1tenth_mpc::MpcControllerNode",
            name="mpc_controller_node",
            parameters=[
                os.path.join(mpc, "config", "mpc_iros_2026_competition.yaml"),
                {
                    "trajectory_file": trajectory,
                    "max_speed_mps": ParameterValue(
                        LaunchConfiguration("controller_max_speed"),
                        value_type=float),
                    "diagnostics_topic": "/mpc/diagnostics",
                    "publish_diagnostics": ParameterValue(
                        LaunchConfiguration("mpc_publish_diagnostics"),
                        value_type=bool),
                },
            ],
            remappings=[
                ("odom", "/odom"),
                ("pose", "/current_map_pose"),
                ("drive", "/cmd/speed"),
            ],
        )

    controller_container = ComposableNodeContainer(
        name="controller_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[controller_component],
        output="screen",
    )
    if controller == "mpc" and needs_localization and mpc_start_delay_sec > 0.0:
        actions.insert(0, RegisterEventHandler(
            OnProcessStart(
                target_action=amcl_node,
                on_start=[TimerAction(
                    period=mpc_start_delay_sec,
                    actions=[
                        LogInfo(msg=(
                            "AMCL warm-up complete; launching MPC after "
                            f"{mpc_start_delay_sec:.2f} s")),
                        controller_container,
                    ],
                )],
            )))
    else:
        actions.append(controller_container)

    actions.append(Node(
        package="sdu_apex_autodrive",
        executable="actuator_interface",
        name="autodrive_actuator_interface",
        output="screen",
        parameters=[
            os.path.join(integration, "config", "actuator_interface.yaml"),
            {"input_topic": "/cmd/speed", "external_stop_topic": ""},
        ],
    ))
    if with_rviz:
        actions.append(Node(
            package="rviz2",
            executable="rviz2",
            name="rviz",
            output="screen",
            remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
        ))
    actions.append(LogInfo(msg=(
        f"controller={controller} localization={needs_localization} "
        f"rviz={with_rviz}")))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "controller", default_value="mpc",
            description="Controller: mpc, pure_pursuit, or ftg."),
        DeclareLaunchArgument(
            "with_rviz", default_value="false",
            description="Start RViz on the team TF tree."),
        DeclareLaunchArgument(
            "use_localization", default_value="true",
            description="Start the map, EKF, and AMCL chain."),
        DeclareLaunchArgument(
            "force_localization", default_value="false",
            description="Also start localization when running FTG mapping diagnostics."),
        DeclareLaunchArgument(
            "controller_max_speed", default_value="16.0",
            description="MPC/PP target-speed ceiling in m/s."),
        DeclareLaunchArgument(
            "mpc_publish_diagnostics", default_value="false",
            description="Publish live MPC diagnostics; disabled on the normal 40 Hz path."),
        DeclareLaunchArgument(
            "mpc_start_delay_sec", default_value="2.0",
            description="Delay MPC startup after AMCL process start."),
        OpaqueFunction(function=_setup),
    ])
