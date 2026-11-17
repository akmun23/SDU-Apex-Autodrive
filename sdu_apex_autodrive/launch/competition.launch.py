"""Autonomous launch of the competition container: bridge, odometry, localization,
actuator interface and MPC. The development switches of the bridge are forced off; the team side consumes only permitted sensor
topics and its own derived state.
"""

import os
import re
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node
from launch_ros.descriptions import ComposableNode



def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    planning = get_package_share_directory("f1tenth_planning")
    mpc = get_package_share_directory("f1tenth_mpc")

    map_relative_path = os.environ.get(
        "SDU_APEX_MAP_REL",
        "maps/autodrive_iros_2026_compete.yaml")
    trajectory_relative_path = os.environ.get(
        "SDU_APEX_TRAJECTORY_REL",
        "trajectories/autodrive_iros_2026_compete/ipopt_best_6p925_source_witness.csv")
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
    amcl_iros_params = os.path.join(localization, "config", "amcl_iros.yaml")
    actuator_params = os.path.join(
        integration, "config", "actuator_interface.yaml")

    required_files = (
        map_path, trajectory_path, mpc_params, sensor_odom_params,
        ekf_params, amcl_params, amcl_iros_params, actuator_params,
    )
    missing = [path for path in required_files if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "competition launch is missing installed runtime file(s): "
            + ", ".join(missing))

    # The wrapper imports the installed official bridge, but paces numeric
    # packets independently of decoder/publication latency.  It also removes
    # the stock restricted reset subscription before creating the bridge node.
    # Suppress non-bag ROS logging for all competition processes.
    quiet_node = {
        "ros_arguments": [
            "--disable-rosout-logs", "--disable-stdout-logs",
            "--disable-external-lib-logs",
        ],
        "output": "screen",
    }

    bridge = Node(
        package="sdu_apex_autodrive",
        executable="autodrive_bridge_40hz",
        name="autodrive_bridge",
        **quiet_node,
        additional_env={
            "SDU_APEX_DEV_SIM_RESET_ENABLED": "0",
            "SDU_APEX_DEV_PACKET_DIAGNOSTICS_ENABLED": "0",
        },
    )

    sensor_odom = Node(
        package="f1tenth_localization",
        executable="sensor_odometry_node",
        name="sensor_odometry",
        **quiet_node,
        parameters=[sensor_odom_params, {"publish_tf": False}],
    )
    ekf = Node(
        package="f1tenth_localization",
        executable="ekf_localization_node",
        name="ekf_localization",
        **quiet_node,
        parameters=[ekf_params],
    )
    map_server = LifecycleNode(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        **quiet_node,
        namespace="",
        parameters=[{"yaml_filename": map_path}],
    )
    map_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        **quiet_node,
        parameters=[{"autostart": True, "node_names": ["map_server"]}],
    )
    amcl = Node(
        package="f1tenth_localization",
        executable="gpu_amcl_cpp_node",
        name="gpu_amcl_cpp",
        **quiet_node,
        parameters=[
            amcl_params,
            amcl_iros_params,
            {
                "global_heading_trajectory_file": trajectory_path,
                "odom_topic": "/ekf_odom",
                "global_initialization": True,
                "global_pose_max_track_distance_m": 0.65,
                # Preserve the IROS spawn prior from amcl_iros.yaml.
                # The first optimized raceline waypoint is NOT the spawn pose;
                # forcing it here relocated AMCL by several metres and drove
                # MPC into the first hairpin rather than starting from spawn.
                "initial_pose_heading_offset_rad": 0.0,
                "publish_tf": False,
            },
        ],
    )
    actuator = Node(
        package="sdu_apex_autodrive",
        executable="actuator_interface",
        name="autodrive_actuator_interface",
        **quiet_node,
        parameters=[
            actuator_params,
            {
                "drive_command_topic": "/mpc/drive_command",
            },
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
                "odom_topic": "/odom",
                "pose_topic": "/current_map_pose",
                "publish_diagnostics": False,
            },
        ],
        remappings=[
            ("odom", "/odom"),
            ("pose", "/current_map_pose"),
        ],
    )
    # Multi-threaded executor: ROS callbacks run separately from the solver, and
    # the container is pinned to the designated MPC CPU pool when one is given.
    mpc_cpus = os.environ.get("SDU_APEX_MPC_CPUSET_CPUS", "")
    if mpc_cpus and not re.fullmatch(
            r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", mpc_cpus):
        raise RuntimeError("invalid competition MPC CPU affinity")
    mpc_container = ComposableNodeContainer(
        name="controller_container",
        **quiet_node,
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        arguments=["--ros-args", "-p", "thread_num:=4"],
        composable_node_descriptions=[mpc_component],
        prefix=[f"taskset --cpu-list {mpc_cpus}"] if mpc_cpus else [],
    )

    # Unity GUI requires an explicit operator Connect. Port 4567 listening is
    # not evidence that it connected or that simulator timestamps are fresh.
    # Launching AMCL/MPC against missing/stale input can poison startup state.
    # These tiny one-shot subscribers exit after four consecutive fresh samples.
    sensor_gate = ExecuteProcess(
        cmd=[sys.executable, "-m", "sdu_apex_autodrive.competition_readiness",
             "sensors"],
        output="screen",
    )
    pose_gate = ExecuteProcess(
        cmd=[sys.executable, "-m", "sdu_apex_autodrive.competition_readiness",
             "pose"],
        output="screen",
    )

    def on_sensor_ready(event, _context):
        if event.returncode != 0:
            return [EmitEvent(event=Shutdown(reason="sensor readiness gate failed"))]
        return [amcl, pose_gate]

    def on_pose_ready(event, _context):
        if event.returncode != 0:
            return [EmitEvent(event=Shutdown(reason="AMCL readiness gate failed"))]
        return [mpc_container]

    return LaunchDescription([
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
        # Install handlers before the corresponding gate process starts.
        RegisterEventHandler(OnProcessExit(
            target_action=sensor_gate, on_exit=on_sensor_ready)),
        RegisterEventHandler(OnProcessExit(
            target_action=pose_gate, on_exit=on_pose_ready)),
        bridge,
        sensor_odom,
        ekf,
        map_server,
        map_lifecycle,
        actuator,
        sensor_gate,
    ])
