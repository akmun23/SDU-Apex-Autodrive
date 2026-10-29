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
    "ipopt_v343_tracking_reserve_raceline.csv")


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
        "amcl_parameter_overlay": LaunchConfiguration(
            "amcl_parameter_overlay").perform(context).strip(),
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
    enable_mpc = _bool(LaunchConfiguration("enable_mpc").perform(context))
    csv_command_file = LaunchConfiguration(
        "csv_command_file").perform(context).strip()
    if csv_command_file and enable_mpc:
        raise RuntimeError(
            "CSV command replay requires enable_mpc:=false to avoid competing publishers")
    if csv_command_file and not os.path.isfile(csv_command_file):
        raise RuntimeError(f"CSV command file does not exist: {csv_command_file}")
    expected = {
        "autodrive_bridge", "sensor_odometry", "ekf_localization",
        "map_server", "lifecycle_manager_map", "gpu_amcl_cpp",
        "autodrive_actuator_interface",
    }
    if enable_mpc:
        expected.add("controller_container")
    if csv_command_file:
        expected.add("csv_command_replay")
    _reject_existing_runtime_nodes(expected)

    bridge = Node(
        package="sdu_apex_autodrive", executable="autodrive_bridge_40hz",
        name="autodrive_bridge", output="screen", emulate_tty=True,
        additional_env={"SDU_APEX_DEV_PACKET_DIAGNOSTICS_ENABLED": "1"})
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
            *([overlays["amcl_parameter_overlay"]]
              if overlays["amcl_parameter_overlay"] else []),
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
    actuator_parameters.append({
        "development_timing_topic":
            "/autodrive/roboracer_1/development_actuator_timing"})
    if csv_command_file and not enable_mpc:
        # The development path follower uses Unity's q=0 full-brake mode.
        # MPC remains bounded to positive drive throttle and competition uses
        # the default actuator minimum of 0.001.
        actuator_parameters.append({"minimum_throttle_normalized": 0.0})
    if overlays["actuator_parameter_overlay"]:
        actuator_parameters.append(overlays["actuator_parameter_overlay"])
    actuator = Node(
        package="sdu_apex_autodrive", executable="actuator_interface",
        name="autodrive_actuator_interface", output="screen",
        parameters=actuator_parameters)

    actions = [
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
        bridge, odometry, ekf, map_server, map_lifecycle,
    ]
    if enable_mpc:
        # Register before adding AMCL to the launch graph; otherwise AMCL can
        # start before this handler is active and the delayed MPC never starts.
        actions.append(RegisterEventHandler(OnProcessStart(
            target_action=amcl,
            on_start=[TimerAction(period=start_delay, actions=[
                LogInfo(msg=f"Starting MPC after {start_delay:.2f}s AMCL warm-up"),
                controller,
            ])])))
    actions.extend([amcl, actuator])
    if csv_command_file:
        actions.append(Node(
            package="sdu_apex_autodrive", executable="csv_command_replay",
            name="csv_command_replay", output="screen",
            parameters=[{
                "csv_file": csv_command_file,
                "path_repeats": int(LaunchConfiguration(
                    "csv_path_repeats").perform(context)),
                "lookahead_speed_gain_s": float(LaunchConfiguration(
                    "csv_lookahead_speed_gain_s").perform(context)),
                "native_control_delay_s": float(LaunchConfiguration(
                    "csv_native_control_delay_s").perform(context)),
            }],
        ))
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
        DeclareLaunchArgument("enable_mpc", default_value="true"),
        DeclareLaunchArgument("csv_command_file", default_value=""),
        DeclareLaunchArgument("csv_path_repeats", default_value="3"),
        DeclareLaunchArgument("csv_lookahead_speed_gain_s", default_value="0.12"),
        DeclareLaunchArgument("csv_native_control_delay_s", default_value="0.05"),
        DeclareLaunchArgument("controller_max_speed", default_value="16.0"),
        DeclareLaunchArgument("mpc_publish_diagnostics", default_value="false"),
        DeclareLaunchArgument("mpc_start_delay_sec", default_value="2.0"),
        DeclareLaunchArgument("mpc_parameter_overlay", default_value=""),
        DeclareLaunchArgument("actuator_parameter_overlay", default_value=""),
        DeclareLaunchArgument("odom_parameter_overlay", default_value=""),
        DeclareLaunchArgument("amcl_parameter_overlay", default_value=""),
        OpaqueFunction(function=_setup),
    ])
