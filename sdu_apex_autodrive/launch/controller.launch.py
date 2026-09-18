"""Single user-facing launch for one AutoDRIVE racing controller."""

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


def _reject_existing_runtime_nodes(expected_names: set[str]) -> None:
    """Fail before startup if a previous controller stack is still alive.

    ROS 2 permits two processes with the same node name.  That is unsafe for
    this stack because both processes can publish the absolute ``/odom`` and
    ``/cmd/speed`` topics, producing alternating states that look like an
    odometry fault.  The check is deliberately launch-local and does not
    touch simulator state.
    """
    try:
        result = subprocess.run(
            ["ros2", "node", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "unable to verify that no previous controller stack is running"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            "ros2 node list failed during controller-stack preflight: "
            f"{result.stderr.strip()}"
        )
    existing_names = {
        line.strip().rsplit("/", 1)[-1]
        for line in result.stdout.splitlines()
        if line.strip()
    }
    duplicates = sorted(expected_names.intersection(existing_names))
    if duplicates:
        raise RuntimeError(
            "controller-stack preflight found existing runtime node(s): "
            f"{', '.join(duplicates)}. Stop the previous launch/container "
            "before starting another one; duplicate publishers corrupt "
            "/odom and /cmd/speed."
        )


def _setup(context):
    controller = LaunchConfiguration("controller").perform(context).lower()
    if controller not in ("ftg", "pure_pursuit", "mpc"):
        raise RuntimeError(
            "controller must be ftg, pure_pursuit, or mpc")
    with_mpc_shadow = _bool(
        LaunchConfiguration("with_mpc_shadow").perform(context))
    if with_mpc_shadow and controller != "pure_pursuit":
        raise RuntimeError(
            "with_mpc_shadow:=true is supported only alongside Pure Pursuit")

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
    model_id_record_lidar_ranges = _bool(
        LaunchConfiguration("model_id_record_lidar_ranges").perform(context))
    if model_id_record_lidar_ranges and not with_model_id_recorder:
        raise RuntimeError(
            "model_id_record_lidar_ranges requires with_model_id_recorder:=true")
    force_localization = _bool(
        LaunchConfiguration("force_localization").perform(context))
    use_localization = _bool(
        LaunchConfiguration("use_localization").perform(context))
    mpc_start_delay_sec = float(
        LaunchConfiguration("mpc_start_delay_sec").perform(context))
    if not math.isfinite(mpc_start_delay_sec) or mpc_start_delay_sec < 0.0:
        raise RuntimeError("mpc_start_delay_sec must be finite and non-negative")
    # FTG normally runs without localization for mapping. This explicit
    # diagnostic mode exercises the full localization stack alongside the
    # same LiDAR-only FTG command path on a saved map.
    needs_localization = use_localization and (
        controller != "ftg" or force_localization)

    expected_runtime_nodes = {
        "autodrive_bridge",
        "sensor_odometry",
        "controller_container",
        "autodrive_actuator_interface",
    }
    amcl_node = None
    if needs_localization:
        expected_runtime_nodes.update({
            "ekf_localization", "map_server", "lifecycle_manager_map",
            "gpu_amcl_cpp",
        })
    if with_ground_truth_monitor:
        expected_runtime_nodes.add("ground_truth_amcl_monitor")
    if with_model_id_recorder:
        expected_runtime_nodes.add("model_id_timing_recorder")
    if with_mpc_shadow:
        expected_runtime_nodes.add("mpc_shadow_node")
    _reject_existing_runtime_nodes(expected_runtime_nodes)

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
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
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
            # AMCL and sensor odometry must publish into the same
            # team-isolated TF tree so map->base_link is available.
        ])
        # This is the user's CUDA AMCL, not Nav2 AMCL. Keep the action handle
        # so an MPC-specific warm-up timer can start from its process-start event.
        amcl_node = Node(
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
            remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
        )
        actions.append(amcl_node)

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
                "record_lidar_ranges": model_id_record_lidar_ranges,
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
    elif controller == "pure_pursuit":
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
        components = [component]
        if with_mpc_shadow:
            if not os.path.isfile(trajectory):
                raise RuntimeError(
                    f"MPC shadow requires a valid raceline: {trajectory}")
            shadow_component = ComposableNode(
                package="f1tenth_mpc",
                plugin="f1tenth_mpc::MpcControllerNode",
                name="mpc_shadow_node",
                parameters=[
                    LaunchConfiguration("mpc_params"),
                    {
                        "enabled": False,
                        "shadow_mode": True,
                        "trajectory_file": trajectory,
                        "max_speed_mps": ParameterValue(
                            LaunchConfiguration("controller_max_speed"),
                            value_type=float,
                        ),
                        "command_topic": "/cmd/speed",
                        "diagnostics_topic": "/mpc_shadow/diagnostics",
                    },
                ],
            )
            components.append(shadow_component)
        actions.append(ComposableNodeContainer(
            name="controller_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=components,
            output="screen",
        ))
    else:
        # The MPC adapter remains command-inhibited unless the explicit launch
        # argument is set. This keeps a source-command baseline from silently
        # becoming actuator authority before its legal-state N30 acceptance.
        component = ComposableNode(
            package="f1tenth_mpc",
            plugin="f1tenth_mpc::MpcControllerNode",
            name="mpc_controller_node",
            parameters=[
                LaunchConfiguration("mpc_params"),
                {
                    "trajectory_file": trajectory,
                    "max_speed_mps": ParameterValue(
                        LaunchConfiguration("controller_max_speed"),
                        value_type=float,
                    ),
                    "enabled": ParameterValue(
                        LaunchConfiguration("mpc_enabled"), value_type=bool),
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
            composable_node_descriptions=[component],
            output="screen",
        )
        if needs_localization and mpc_start_delay_sec > 0.0:
            # Register ahead of AMCL process startup. AMCL continues to receive
            # scans during this timer; only MPC composition is delayed.
            actions.insert(0, RegisterEventHandler(
                OnProcessStart(
                    target_action=amcl_node,
                    on_start=[TimerAction(
                        period=mpc_start_delay_sec,
                        actions=[
                            LogInfo(msg=(
                                "AMCL warm-up complete; launching MPC after "
                                f"{mpc_start_delay_sec:.2f} s"
                            )),
                            mpc_container,
                        ],
                    )],
                )
            ))
            actions.append(LogInfo(msg=(
                f"MPC launch delayed {mpc_start_delay_sec:.2f} s after AMCL starts"
            )))
        else:
            actions.append(mpc_container)
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
                "collision_terminal_stop": with_collision_safety,
                # Bridge timing diagnostics are non-fatal on the constrained
                # competition hardware; they must never stop actuator output.
                "external_stop_topic": "",
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
        f"mpc_shadow={with_mpc_shadow} rviz={with_rviz}"
    )))
    return actions


def generate_launch_description():
    integration = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    control = get_package_share_directory("f1tenth_control")
    mpc = get_package_share_directory("f1tenth_mpc")

    return LaunchDescription([
        DeclareLaunchArgument(
            "controller",
            default_value="pure_pursuit",
            description=(
                "Controller to run: FTG, Pure Pursuit, or MPC."
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
            "model_id_record_lidar_ranges",
            default_value="false",
            description=(
                "Optional full LiDAR range sidecar for offline AMCL scan-"
                "observability analysis; requires the model-ID recorder"
            ),
        ),
        DeclareLaunchArgument(
            "model_id_output_dir",
            default_value="/workspace/src/sdu_apex_autodrive/artifacts/simulator_trace",
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
            default_value="16.0",
            description=(
                "Maximum controller target speed [m/s]. The project "
                "operating ceiling is 16 m/s; this does not alter simulator physics."
            ),
        ),
        DeclareLaunchArgument(
            "mpc_params",
            default_value=os.path.join(mpc, "config", "mpc_autodrive.yaml"),
            description="Source-command MPC adapter configuration",
        ),
        DeclareLaunchArgument(
            "mpc_enabled",
            default_value="false",
            description=(
                "Allow MPC commands. Keep false until the source-command "
                "stage map passes legal-state N30 validation."
            ),
        ),
        DeclareLaunchArgument(
            "mpc_start_delay_sec",
            default_value="2.0",
            description=(
                "Wait this many seconds after the AMCL process starts before "
                "launching the MPC controller; set to 0 to disable"
            ),
        ),
        DeclareLaunchArgument(
            "with_mpc_shadow",
            default_value="false",
            description=(
                "Run the 9-state MPC beside Pure Pursuit without /cmd/speed "
                "authority; record /mpc_shadow/diagnostics for Phase 10."
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
            "calibration_params",
            default_value=os.path.join(integration, "config", "calibration.yaml"),
        ),
        OpaqueFunction(function=_setup),
    ])
