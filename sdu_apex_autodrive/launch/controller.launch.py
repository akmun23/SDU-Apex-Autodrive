"""Single user-facing launch for one AutoDRIVE racing controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node
from launch_ros.descriptions import ComposableNode


DEFAULT_MAP = (
    "/workspace/src/f1tenth_planning/maps/"
    "autodrive_track_ftg_commit_20260909_025m.yaml"
)
DEFAULT_TRAJECTORY = (
    "/workspace/src/f1tenth_planning/trajectories/"
    "autodrive_track_ftg_commit_20260908_lap01_mintime_raceline.csv"
)


def _bool(value: str) -> bool:
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"expected boolean, got {value!r}")


def _setup(context):
    controller = LaunchConfiguration("controller").perform(context).lower()
    if controller not in ("ftg", "pure_pursuit", "stanley", "mpc"):
        raise RuntimeError("controller must be ftg, pure_pursuit, stanley, or mpc")

    map_path = LaunchConfiguration("map").perform(context)
    trajectory = LaunchConfiguration("trajectory").perform(context)
    with_rviz = _bool(LaunchConfiguration("with_rviz").perform(context))
    with_ground_truth_monitor = _bool(
        LaunchConfiguration("with_ground_truth_monitor").perform(context))
    with_collision_safety = _bool(
        LaunchConfiguration("with_collision_safety").perform(context))
    with_telemetry_recorder = _bool(
        LaunchConfiguration("with_telemetry_recorder").perform(context))
    force_localization = _bool(
        LaunchConfiguration("force_localization").perform(context))
    lateral_value = LaunchConfiguration("with_lateral_planner").perform(context).lower()
    with_lateral = controller == "mpc" if lateral_value == "auto" else _bool(lateral_value)
    avoidance = _bool(LaunchConfiguration("avoidance_enabled").perform(context))
    # FTG normally runs without localization for mapping. This explicit
    # diagnostic mode exercises the full localization stack alongside the
    # same LiDAR-only FTG command path on a saved map.
    needs_localization = controller != "ftg" or force_localization

    if needs_localization and not os.path.isfile(map_path):
        raise RuntimeError(f"map does not exist: {map_path}")
    if needs_localization and not os.path.isfile(trajectory):
        raise RuntimeError(f"trajectory does not exist: {trajectory}")
    if controller == "mpc" and not with_lateral:
        raise RuntimeError("mpc requires with_lateral_planner:=true")

    actuator_input_topic = "/cmd/acceleration" if controller == "mpc" else "/cmd/speed"
    actuator_command_mode = "acceleration" if controller == "mpc" else "speed"

    actions = [
        # Pace commands independently of the official telemetry decoder so
        # the simulator is not throttled by camera/LIDAR ROS publication.
        Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
            name="autodrive_bridge",
            output="screen",
            emulate_tty=True,
        ),
        # Team odometry uses only allowed encoders and IMU.
        # Its TF is isolated from the restricted simulator /tf.
        Node(
            package="f1tenth_localization",
            executable="sensor_odometry_node",
            name="sensor_odometry",
            output="screen",
            parameters=[
                LaunchConfiguration("sensor_odom_params"),
                # Keep simulator reset epochs from carrying old speed/pose
                # into the next controller run.
                {"reset_enabled": True, "reset_topic": "/autodrive/reset_command"},
            ],
            remappings=[
                ("/tf", "/sdu/tf"),
                ("/tf_static", "/sdu/tf_static"),
            ],
        ),
    ]

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
            # AMCL consumes this causal covariance-bearing local odometry
            # stream. Raw /odom stays available for diagnostics and is never
            # treated as a process covariance by the trust filter.
            Node(
                package="f1tenth_localization",
                executable="ekf_localization_node",
                name="ekf_localization",
                output="screen",
                parameters=[
                    LaunchConfiguration("ekf_params"),
                    {"reset_enabled": True, "reset_topic": "/autodrive/reset_command"},
                ],
            ),
            map_server,
            # Use a launch-owned lifecycle manager.  Emitting configure and
            # activate events directly can race map_server discovery and leave
            # AMCL permanently waiting for a map.
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_map",
                output="screen",
                parameters=[{
                    "autostart": True,
                    "node_names": ["map_server"],
                }],
            ),
            # This is the user's CUDA AMCL, not Nav2 AMCL.
            Node(
                package="f1tenth_localization",
                executable="gpu_amcl_cpp_node",
                name="gpu_amcl_cpp",
                output="screen",
                parameters=[
                    LaunchConfiguration("amcl_params"),
                    {"global_heading_trajectory_file": trajectory},
                    {
                        "global_initialization": LaunchConfiguration(
                            "amcl_global_initialization"
                        ),
                        "global_pose_max_track_distance_m": LaunchConfiguration(
                            "amcl_max_track_distance"
                        ),
                        "initial_pose_heading_offset_rad": LaunchConfiguration(
                            "amcl_initial_heading_offset"
                        ),
                    },
                ],
                # AMCL and sensor odometry must publish into the same
                # team-isolated TF tree so map->base_link is available.
                remappings=[
                    ("/tf", "/sdu/tf"),
                    ("/tf_static", "/sdu/tf_static"),
                ],
            ),
        ])

    if with_lateral:
        actions.extend([
            Node(
                package="f1tenth_lidar",
                executable="scan_splitter_node",
                name="scan_splitter_node",
                output="screen",
                parameters=[LaunchConfiguration("scan_splitter_params")],
                remappings=[
                    ("/tf", "/sdu/tf"),
                    ("/tf_static", "/sdu/tf_static"),
                ],
            ),
            Node(
                package="f1tenth_lateral_planner",
                executable="lateral_planner_node",
                name="lateral_planner_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("lateral_planner_params"),
                    {
                        "trajectory_file": trajectory,
                        "avoidance_enabled": avoidance,
                    },
                ],
                remappings=[
                    ("/tf", "/sdu/tf"),
                    ("/tf_static", "/sdu/tf_static"),
                ],
            ),
        ])

    if with_ground_truth_monitor:
        # Development diagnostics only. Ground truth is not consumed by AMCL
        # or any controller and must be disabled for a rules-only run.
        actions.append(Node(
            package="sdu_apex_autodrive",
            executable="ground_truth_amcl_monitor",
            name="ground_truth_amcl_monitor",
            output="screen",
            parameters=[{
                "output_csv": LaunchConfiguration("ground_truth_output_csv"),
                "map_provenance_file": LaunchConfiguration("map_provenance_file"),
                "require_absolute_map_scoring": LaunchConfiguration(
                    "require_absolute_map_scoring"),
            }],
        ))

    if with_telemetry_recorder:
        # Allowed-sensor recorder only.  It does not subscribe to simulator
        # pose, collision, lap, or debug-state topics and does not publish a
        # command in sensor_record mode.
        actions.append(Node(
            package="sdu_apex_autodrive",
            executable="calibration",
            name="allowed_telemetry_recorder",
            output="screen",
            parameters=[
                LaunchConfiguration("calibration_params"),
                {
                    "mode": "sensor_record",
                    "output_dir": LaunchConfiguration("telemetry_output_dir"),
                    "duration_sec": LaunchConfiguration("telemetry_duration_sec"),
                    # Preserve callback-level snapshots as well as the 50 Hz
                    # timer rows. This is required to reconstruct the exact
                    # first-turn ordering at native 20 Hz simulator cadence.
                    "capture_source_events": True,
                },
            ],
        ))

    if controller == "ftg":
        ftg_max_speed = float(LaunchConfiguration("ftg_max_speed").perform(context))
        component = ComposableNode(
            package="f1tenth_control",
            plugin="f1tenth_control::FTGNode",
            name="ftg_node",
            parameters=[
                LaunchConfiguration("ftg_params"),
                {"max_speed": ftg_max_speed},
            ],
            remappings=[
                ("scan", "/autodrive/roboracer_1/lidar"),
                ("odom", "/autodrive/roboracer_1/odom"),
                ("drive", "/cmd/speed"),
            ],
        )
        actions.append(ComposableNodeContainer(
            name="controller_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[component],
            output="screen",
        ))
    elif controller in ("pure_pursuit", "stanley"):
        component = ComposableNode(
            package="f1tenth_control",
            plugin=(
                "f1tenth_control::PurePursuitNode"
                if controller == "pure_pursuit"
                else "f1tenth_control::StanleyNode"
            ),
            name=(
                "pure_pursuit_node"
                if controller == "pure_pursuit"
                else "stanley_node"
            ),
            parameters=[
                LaunchConfiguration("path_tracking_params"),
                {
                    "trajectory_file": trajectory,
                    "max_speed": LaunchConfiguration("controller_max_speed"),
                },
            ],
        )
        actions.append(ComposableNodeContainer(
            name="controller_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[component],
            output="screen",
        ))
    else:
        actions.append(Node(
            package="mpc_riccati",
            executable="mpc_autodrive_node",
            name="mpc_autodrive_node",
            output="screen",
            emulate_tty=True,
            additional_env={
                "MPC_ODOM_TOPIC": "/odom",
                # The MPC input name is historical. Feed it the current
                # map-frame pose propagated from AMCL's map->odom correction,
                # not the scan-time /amcl_pose measurement.
                "MPC_EKF_TOPIC": "/current_map_pose",
                "MPC_LOCAL_RACELINE_TOPIC": "/local_raceline",
                "MPC_DRIVE_TOPIC": "/cmd/acceleration",
                "MPC_POSE_FRAME": "map",
                "MPC_PATH_FRAME": "map",
                "MPC_COMMAND_FRAME": "base_link",
            },
        ))

    actions.append(Node(
        package="sdu_apex_autodrive",
        executable="actuator_interface",
        name="autodrive_actuator_interface",
        output="screen",
        parameters=[
            LaunchConfiguration("actuator_params"),
            # Keep the simulator-only collision reset outside the default
            # competition interface. Enable it explicitly for test runs.
            {
                "input_topic": actuator_input_topic,
                "command_mode": actuator_command_mode,
                "collision_reset_enabled": with_collision_safety,
            },
        ],
    ))

    if with_rviz:
        # RViz sees only the team's allowed-sensor TF tree.
        actions.append(Node(
            package="rviz2",
            executable="rviz2",
            name="rviz",
            output="screen",
            remappings=[
                ("/tf", "/sdu/tf"),
                ("/tf_static", "/sdu/tf_static"),
            ],
        ))

    actions.append(LogInfo(msg=(
        f"controller={controller} custom_amcl={needs_localization} "
        f"lateral_planner={with_lateral} rviz={with_rviz}"
    )))
    return actions


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    control = get_package_share_directory("f1tenth_control")
    lidar = get_package_share_directory("f1tenth_lidar")
    planner = get_package_share_directory("f1tenth_lateral_planner")

    return LaunchDescription([
        DeclareLaunchArgument(
            "controller",
            default_value="pure_pursuit",
            description="Controller to run; Pure Pursuit is the validated simulator default",
        ),
        DeclareLaunchArgument("map", default_value=DEFAULT_MAP),
        DeclareLaunchArgument("trajectory", default_value=DEFAULT_TRAJECTORY),
        DeclareLaunchArgument(
            "with_rviz",
            default_value="false",
            description="Development visualization; disabled for race launch by default",
        ),
        DeclareLaunchArgument(
            "with_ground_truth_monitor",
            default_value="false",
            description="Development-only aligned simulator ground-truth diagnostics",
        ),
        DeclareLaunchArgument(
            "with_telemetry_recorder",
            default_value="false",
            description="Record allowed sensor/controller telemetry for offline tuning",
        ),
        DeclareLaunchArgument(
            "force_localization",
            default_value="false",
            description=(
                "Start map localization alongside FTG for diagnostics; FTG still "
                "uses only LiDAR and official odometry for its command"
            ),
        ),
        DeclareLaunchArgument(
            "telemetry_output_dir",
            default_value="/workspace/src/sdu_apex_autodrive/artifacts/calibration/raw",
        ),
        DeclareLaunchArgument(
            "telemetry_duration_sec",
            default_value="0.0",
            description=(
                "Finite diagnostics recorder duration; zero keeps recording until "
                "the launch is stopped"
            ),
        ),
        DeclareLaunchArgument(
            "ground_truth_output_csv",
            default_value=(
                "/workspace/src/sdu_apex_autodrive/artifacts/calibration/"
                "ground_truth_amcl.csv"
            ),
            description="Diagnostics-only AMCL/EKF/odom versus simulator ground-truth CSV",
        ),
        DeclareLaunchArgument(
            "map_provenance_file",
            default_value="",
            description=(
                "Diagnostics-only fixed map-to-simulator transform produced when the "
                "map was saved"
            ),
        ),
        DeclareLaunchArgument(
            "require_absolute_map_scoring",
            default_value="false",
            description=(
                "Fail the diagnostics monitor instead of silently using relative "
                "first-pair scoring"
            ),
        ),
        DeclareLaunchArgument(
            "with_collision_safety",
            default_value="false",
            description=(
                "Simulator-only collision latch/reset for diagnostics; false is the "
                "rules-compliant default"
            ),
        ),
        DeclareLaunchArgument("with_lateral_planner", default_value="auto"),
        DeclareLaunchArgument("avoidance_enabled", default_value="false"),
        DeclareLaunchArgument(
            "controller_max_speed",
            default_value="1.5",
            description=(
                "Safe simulator startup cap for Pure Pursuit or Stanley [m/s]. "
                "Raise explicitly only after the baseline follows the track."
            ),
        ),
        DeclareLaunchArgument(
            "ftg_max_speed",
            default_value="0.40",
            description="FTG diagnostic speed cap [m/s]",
        ),
        DeclareLaunchArgument(
            "amcl_global_initialization",
            default_value="false",
            description=(
                "Use known simulator reset/raceline startup; enable global "
                "recovery only for arbitrary track-position tests."
            ),
        ),
        DeclareLaunchArgument(
            "amcl_max_track_distance",
            default_value="0.65",
            description=(
                "Maximum global AMCL candidate distance from the raceline [m]. "
                "Use a temporary override only for measured acceptance tests."
            ),
        ),
        DeclareLaunchArgument(
            "amcl_initial_heading_offset",
            default_value="0.0",
            description=(
                "Optional local AMCL startup yaw offset in map radians. The "
                "default is zero so the known-start pose uses the raceline "
                "heading exactly; only override for a measured frame offset."
            ),
        ),

        DeclareLaunchArgument(
            "sensor_odom_params",
            default_value=os.path.join(localization, "config", "sensor_odometry.yaml"),
        ),
        DeclareLaunchArgument(
            "amcl_params",
            default_value=os.path.join(localization, "config", "gpu_amcl_cpp_params.yaml"),
        ),
        DeclareLaunchArgument(
            "ekf_params",
            default_value=os.path.join(localization, "config", "ekf.yaml"),
        ),
        DeclareLaunchArgument(
            "ftg_params",
            default_value=os.path.join(control, "config", "ftg_params.yaml"),
        ),
        DeclareLaunchArgument(
            "path_tracking_params",
            default_value=os.path.join(control, "config", "path_tracking_autodrive.yaml"),
        ),
        DeclareLaunchArgument(
            "scan_splitter_params",
            default_value=os.path.join(lidar, "config", "scan_splitter.yaml"),
        ),
        DeclareLaunchArgument(
            "lateral_planner_params",
            default_value=os.path.join(planner, "config", "lateral_planner.yaml"),
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(integration, "config", "actuator_interface.yaml"),
        ),
        DeclareLaunchArgument(
            "calibration_params",
            default_value=os.path.join(integration, "config", "calibration.yaml"),
        ),
        OpaqueFunction(function=_setup),
    ])
