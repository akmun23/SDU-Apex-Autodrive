"""Single user-facing launch for one AutoDRIVE racing controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LifecycleNode, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


DEFAULT_MAP = (
    "/workspace/src/f1tenth_planning/maps/"
    "autodrive_track_ftg_commit_20260909_025m.yaml"
)
DEFAULT_TRAJECTORY = (
    "/workspace/src/f1tenth_planning/trajectories/"
    "autodrive_track_ftg_commit_20260909_025m_mintime_raceline.csv"
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
    if controller not in ("ftg", "pure_pursuit", "mpc_shadow"):
        raise RuntimeError("controller must be ftg, pure_pursuit, or mpc_shadow")

    map_path = LaunchConfiguration("map").perform(context)
    trajectory = LaunchConfiguration("trajectory").perform(context)
    amcl_override_path = LaunchConfiguration("amcl_override_params").perform(context)
    with_rviz = _bool(LaunchConfiguration("with_rviz").perform(context))
    with_ground_truth_monitor = _bool(
        LaunchConfiguration("with_ground_truth_monitor").perform(context))
    with_collision_safety = _bool(
        LaunchConfiguration("with_collision_safety").perform(context))
    with_telemetry_recorder = _bool(
        LaunchConfiguration("with_telemetry_recorder").perform(context))
    with_model_id_recorder = _bool(
        LaunchConfiguration("with_model_id_recorder").perform(context))
    force_localization = _bool(
        LaunchConfiguration("force_localization").perform(context))
    use_localization = _bool(
        LaunchConfiguration("use_localization").perform(context))
    # FTG normally runs without localization for mapping. This explicit
    # diagnostic mode exercises the full localization stack alongside the
    # same LiDAR-only FTG command path on a saved map.
    needs_localization = use_localization and (
        controller != "ftg" or force_localization)

    if needs_localization and not os.path.isfile(map_path):
        raise RuntimeError(f"map does not exist: {map_path}")
    if needs_localization and not os.path.isfile(trajectory):
        raise RuntimeError(f"trajectory does not exist: {trajectory}")
    amcl_parameter_sources = [LaunchConfiguration("amcl_params")]
    if amcl_override_path.strip():
        if not os.path.isfile(amcl_override_path):
            raise RuntimeError(
                f"amcl_override_params does not exist: {amcl_override_path}"
            )
        amcl_parameter_sources.append(amcl_override_path)
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
                    *amcl_parameter_sources,
                    {
                        "global_heading_trajectory_file": trajectory,
                        "odom_topic": LaunchConfiguration("amcl_odom_topic"),
                    },
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
                "map_start_x_m": LaunchConfiguration("map_start_x_m"),
                "map_start_y_m": LaunchConfiguration("map_start_y_m"),
                "map_start_yaw_rad": LaunchConfiguration("map_start_yaw_rad"),
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
                    # Launch arguments are strings and ROS 2 otherwise infers
                    # an integer for values such as ``100``.  The recorder
                    # declares this parameter as a double; force the type so
                    # a valid test command cannot silently kill the recorder
                    # before it captures source-time data.
                    "duration_sec": ParameterValue(
                        LaunchConfiguration("telemetry_duration_sec"),
                        value_type=float,
                    ),
                    # Preserve callback-level snapshots as well as the 50 Hz
                    # timer rows. This is required to reconstruct the exact
                    # first-turn ordering at native 20 Hz simulator cadence.
                    "capture_source_events": True,
                },
            ],
        ))

    if with_model_id_recorder:
        # Causal diagnostics only. This recorder embeds the packet-side
        # simulator fields offline and never publishes or feeds ground truth
        # into localization/control.
        actions.append(Node(
            package="sdu_apex_autodrive",
            executable="model_id_timing_recorder",
            name="model_id_timing_recorder",
            output="screen",
            parameters=[{
                "output_dir": LaunchConfiguration("model_id_output_dir"),
                "run_name": LaunchConfiguration("model_id_run_name"),
                "experiment_mode": "track_validation",
                "duration_sec": ParameterValue(
                    LaunchConfiguration("model_id_duration_sec"), value_type=float),
            }],
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
    else:
        component = ComposableNode(
            package="f1tenth_control",
            plugin="f1tenth_control::PurePursuitNode",
            name="pure_pursuit_node",
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
        if controller == "mpc_shadow":
            # Pure Pursuit remains the sole /cmd/speed publisher.  The new
            # MPC only consumes the authoritative legal control state and
            # publishes diagnostics plus /mpc/shadow_command.
            actions.extend([
                Node(
                    package="f1tenth_mpc",
                    executable="control_state_node",
                    name="mpc_control_state",
                    output="screen",
                    parameters=[LaunchConfiguration("mpc_params")],
                ),
                Node(
                    package="f1tenth_mpc",
                    executable="mpc_shadow_node",
                    name="mpc_shadow",
                    output="screen",
                    parameters=[
                        LaunchConfiguration("mpc_params"),
                        {"trajectory_file": trajectory},
                    ],
                ),
            ])
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
                "input_topic": "/cmd/speed",
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
        f"rviz={with_rviz}"
    )))
    return actions


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    control = get_package_share_directory("f1tenth_control")

    return LaunchDescription([
        DeclareLaunchArgument(
            "controller",
            default_value="pure_pursuit",
            description=(
                "Controller to run: FTG, Pure Pursuit, or mpc_shadow. "
                "mpc_shadow never publishes actuator commands."
            ),
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
            "with_model_id_recorder",
            default_value="false",
            description=(
                "Record immutable causal bridge/packet/sensor tables for offline "
                "model validation; no simulator truth enters runtime control"
            ),
        ),
        DeclareLaunchArgument(
            "model_id_output_dir",
            default_value="/workspace/src/sdu_apex_autodrive/artifacts/model_id",
        ),
        DeclareLaunchArgument(
            "model_id_run_name",
            default_value="track_validation",
        ),
        DeclareLaunchArgument(
            "model_id_duration_sec",
            default_value="0.0",
            description="Optional finite duration for the causal model-ID recorder",
        ),
        DeclareLaunchArgument(
            "force_localization",
            default_value="false",
            description=(
                "Start map localization alongside FTG for diagnostics; FTG uses "
                "only LiDAR for its command"
            ),
        ),
        DeclareLaunchArgument(
            "use_localization",
            default_value="true",
            description=(
                "Start the map/AMCL stack. Disable only for diagnostics that "
                "provide an explicit external pose on /current_map_pose."
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
            "map_start_x_m",
            default_value="nan",
            description=(
                "Diagnostics-only map-frame X of the vehicle at map recording start"
            ),
        ),
        DeclareLaunchArgument(
            "map_start_y_m",
            default_value="nan",
            description=(
                "Diagnostics-only map-frame Y of the vehicle at map recording start"
            ),
        ),
        DeclareLaunchArgument(
            "map_start_yaw_rad",
            default_value="nan",
            description=(
                "Diagnostics-only map-frame yaw of the vehicle at map recording start"
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
        DeclareLaunchArgument(
            "controller_max_speed",
            default_value="22.88",
            description=(
                "Normal Pure Pursuit maximum speed [m/s]. The first-lap "
                "startup ramp is configured in path_tracking_autodrive.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "ftg_max_speed",
            default_value="0.40",
            description="FTG diagnostic speed cap [m/s]",
        ),
        DeclareLaunchArgument(
            "amcl_global_initialization",
            default_value="true",
            description=(
                "Localize from LiDAR over the complete track at startup; "
                "disable only for a deliberate known-pose unit test."
            ),
        ),
        DeclareLaunchArgument(
            "amcl_max_track_distance",
            default_value="0.65",
            description=(
                "Maximum global AMCL candidate distance from the raceline [m]. "
                "This rejects visually plausible closed-track aliases during "
                "startup."
            ),
        ),
        DeclareLaunchArgument(
            "amcl_odom_topic",
            default_value="/ekf_odom",
            description="Timestamped odometry input used by AMCL",
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
            "amcl_override_params",
            default_value="",
            description=(
                "Optional YAML layered after the production AMCL parameters "
                "for one-factor offline/live experiments"
            ),
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
            "actuator_params",
            default_value=os.path.join(integration, "config", "actuator_interface.yaml"),
        ),
        DeclareLaunchArgument(
            "mpc_params",
            default_value=os.path.join(
                get_package_share_directory("f1tenth_mpc"), "config", "mpc_autodrive.yaml"),
            description="Candidate model and shadow MPC parameters",
        ),
        DeclareLaunchArgument(
            "calibration_params",
            default_value=os.path.join(integration, "config", "calibration.yaml"),
        ),
        OpaqueFunction(function=_setup),
    ])
