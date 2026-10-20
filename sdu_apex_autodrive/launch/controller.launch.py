"""Development launch for the single production MPC/localization stack."""

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


MAP_NAME = "autodrive_practice_20260924_b.yaml"
RACELINE_PATH = os.path.join(
    "autodrive_practice_20260924_b",
    "autodrive_practice_20260924_b_mintime_raceline.csv")


def _bool(value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"expected boolean, got {value!r}")


def _reject_existing_runtime_nodes(expected_names: set[str]) -> None:
    try:
        result = subprocess.run(
            ["ros2", "node", "list"], check=False, capture_output=True,
            text=True, timeout=3.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("unable to verify that no previous stack is running") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "ros2 node list failed during preflight: " + result.stderr.strip())
    names = {
        line.strip().rsplit("/", 1)[-1]
        for line in result.stdout.splitlines() if line.strip()
    }
    duplicates = sorted(expected_names.intersection(names))
    if duplicates:
        raise RuntimeError(
            "runtime node(s) already exist: " + ", ".join(duplicates))


def _setup(context):
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    planning = get_package_share_directory("f1tenth_planning")
    mpc = get_package_share_directory("f1tenth_mpc")

    map_override = LaunchConfiguration("map_yaml").perform(context).strip()
    trajectory_override = LaunchConfiguration(
        "trajectory_file").perform(context).strip()
    if bool(map_override) != bool(trajectory_override):
        raise RuntimeError("map_yaml and trajectory_file must be overridden together")
    map_path = map_override or os.path.join(planning, "maps", MAP_NAME)
    trajectory = trajectory_override or os.path.join(
        planning, "trajectories", RACELINE_PATH)
    if not os.path.isfile(map_path) or not os.path.isfile(trajectory):
        raise RuntimeError("the selected map/raceline pair is incomplete")

    overlays = {
        "mpc_parameter_overlay": LaunchConfiguration(
            "mpc_parameter_overlay").perform(context).strip(),
        "odom_parameter_overlay": LaunchConfiguration(
            "odom_parameter_overlay").perform(context).strip(),
        "actuator_parameter_overlay": LaunchConfiguration(
            "actuator_parameter_overlay").perform(context).strip(),
    }
    for name, path in overlays.items():
        if path and not os.path.isfile(path):
            raise RuntimeError(f"{name} does not exist: {path}")

    start_delay = float(
        LaunchConfiguration("mpc_start_delay_sec").perform(context))
    if not math.isfinite(start_delay) or start_delay < 0.0:
        raise RuntimeError("mpc_start_delay_sec must be finite and nonnegative")
    with_rviz = _bool(LaunchConfiguration("with_rviz").perform(context))
    expected = {
        "autodrive_bridge", "sensor_odometry", "ekf_localization",
        "map_server", "lifecycle_manager_map", "gpu_amcl_cpp",
        "controller_container", "autodrive_actuator_interface",
    }
    _reject_existing_runtime_nodes(expected)

    bridge = Node(
        package="sdu_apex_autodrive", executable="autodrive_bridge_40hz",
        name="autodrive_bridge", output="screen", emulate_tty=True)
    odometry = Node(
        package="f1tenth_localization", executable="sensor_odometry_node",
        name="sensor_odometry", output="screen",
        parameters=[
            os.path.join(localization, "config", "sensor_odometry.yaml"),
            *([overlays["odom_parameter_overlay"]]
              if overlays["odom_parameter_overlay"] else []),
        ], remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")])
    ekf = Node(
        package="f1tenth_localization", executable="ekf_localization_node",
        name="ekf_localization", output="screen",
        parameters=[os.path.join(localization, "config", "ekf.yaml")])
    map_server = LifecycleNode(
        package="nav2_map_server", executable="map_server",
        name="map_server", namespace="", output="screen",
        parameters=[{"yaml_filename": map_path}])
    map_lifecycle = Node(
        package="nav2_lifecycle_manager", executable="lifecycle_manager",
        name="lifecycle_manager_map", output="screen",
        parameters=[{"autostart": True, "node_names": ["map_server"]}])
    amcl = Node(
        package="f1tenth_localization", executable="gpu_amcl_cpp_node",
        name="gpu_amcl_cpp", output="screen",
        parameters=[
            os.path.join(localization, "config", "gpu_amcl_cpp_params.yaml"),
            {
                "global_heading_trajectory_file": trajectory,
                "odom_topic": "/ekf_odom",
                "global_initialization": True,
                "global_pose_max_track_distance_m": 0.65,
                "initial_pose_heading_offset_rad": 0.0,
            },
        ], remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")])

    mpc_parameters = [os.path.join(mpc, "config", "mpc_competition.yaml")]
    if overlays["mpc_parameter_overlay"]:
        mpc_parameters.append(overlays["mpc_parameter_overlay"])
    diagnostics = _bool(
        LaunchConfiguration("mpc_publish_diagnostics").perform(context))
    mpc_node = ComposableNode(
        package="f1tenth_mpc", plugin="f1tenth_mpc::MpcControllerNode",
        name="mpc_controller_node",
        parameters=mpc_parameters + [{
            "trajectory_file": trajectory,
            "max_speed_mps": ParameterValue(
                LaunchConfiguration("controller_max_speed"), value_type=float),
            "publish_diagnostics": diagnostics,
            "diagnostics_topic": "/mpc/diagnostics",
        }], remappings=[
            ("odom", "/odom"), ("pose", "/current_map_pose")])
    controller = ComposableNodeContainer(
        name="controller_container", namespace="",
        package="rclcpp_components", executable="component_container_mt",
        arguments=["--ros-args", "-p", "thread_num:=4"],
        composable_node_descriptions=[mpc_node], output="screen")
    actuator_parameters = [
        os.path.join(integration, "config", "actuator_interface.yaml")]
    if overlays["actuator_parameter_overlay"]:
        actuator_parameters.append(overlays["actuator_parameter_overlay"])
    actuator = Node(
        package="sdu_apex_autodrive", executable="actuator_interface",
        name="autodrive_actuator_interface", output="screen",
        parameters=actuator_parameters)

    actions = [
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
        bridge, odometry, ekf, map_server, map_lifecycle, amcl, actuator,
        RegisterEventHandler(OnProcessStart(
            target_action=amcl,
            on_start=[TimerAction(period=start_delay, actions=[
                LogInfo(msg=f"Starting MPC after {start_delay:.2f}s AMCL warm-up"),
                controller,
            ])])),
    ]
    if with_rviz:
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz",
            output="screen", remappings=[("/tf", "/sdu/tf"),
                                         ("/tf_static", "/sdu/tf_static")]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", default_value=""),
        DeclareLaunchArgument("trajectory_file", default_value=""),
        DeclareLaunchArgument("with_rviz", default_value="false"),
        DeclareLaunchArgument("controller_max_speed", default_value="16.0"),
        DeclareLaunchArgument("mpc_publish_diagnostics", default_value="false"),
        DeclareLaunchArgument("mpc_start_delay_sec", default_value="2.0"),
        DeclareLaunchArgument("mpc_parameter_overlay", default_value=""),
        DeclareLaunchArgument("actuator_parameter_overlay", default_value=""),
        DeclareLaunchArgument("odom_parameter_overlay", default_value=""),
        OpaqueFunction(function=_setup),
    ])
