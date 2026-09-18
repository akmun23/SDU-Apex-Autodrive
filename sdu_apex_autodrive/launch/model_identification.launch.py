"""Development-only causal model-identification launch.

This launch keeps the racing controller out of the experiment. The existing
calibration node supplies bounded diagnostic excitation, the normal actuator
interface applies it, and model_id_timing_recorder captures timing plus
allowed sensor/feedback events. Simulator ground truth is not subscribed to
by the recorder or used for runtime control.
"""

import math
import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler,
    SetEnvironmentVariable, Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _setup(context):
    output_dir = LaunchConfiguration("output_dir").perform(context)
    run_name = LaunchConfiguration("run_name").perform(context)
    if not run_name.strip():
        run_name = time.strftime("model_id_timing_test_%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(output_dir, run_name)
    mode = LaunchConfiguration("mode").perform(context)
    duration = float(LaunchConfiguration("duration_sec").perform(context))
    speed_sequence_text = LaunchConfiguration("speed_sequence_mps").perform(
        context).strip()
    reset_setting = LaunchConfiguration("reset_between_steps").perform(
        context).strip().lower()
    if reset_setting == "auto":
        reset_between_steps = mode == "identification_grid"
    elif reset_setting in {"true", "false"}:
        reset_between_steps = reset_setting == "true"
    else:
        raise ValueError("reset_between_steps must be auto, true, or false")

    calibration_parameters = {
        "mode": mode,
        "output_dir": artifact_dir,
        "duration_sec": ParameterValue(duration, value_type=float),
        # The timing recorder below is the canonical 40 Hz source stream.
        # Keep this wide phase/state CSV supplemental and bounded.
        "capture_source_events": False,
        "calibration_record_rate_hz": 10.0,
        "capture_source_event_names": [
            "gt_odom", "imu", "left_encoder", "right_encoder",
            "odom_diagnostics",
        ],
        # Grid profiles reset by default. Speed-step profiles can be run as
        # short fresh-player trials; opt into repeated teleports explicitly.
        "reset_between_steps": reset_between_steps,
        "ground_truth_boundary_distance_m": ParameterValue(
            LaunchConfiguration("ground_truth_boundary_distance_m"),
            value_type=float,
        ),
    }
    if speed_sequence_text:
        speed_sequence_parts = [
            value.strip() for value in speed_sequence_text.split(",")]
        if any(not value for value in speed_sequence_parts):
            raise ValueError("speed_sequence_mps must be comma-separated numbers")
        speed_sequence = [float(value) for value in speed_sequence_parts]
        if not all(math.isfinite(value) for value in speed_sequence):
            raise ValueError("speed_sequence_mps values must be finite")
        calibration_parameters["speed_sequence_mps"] = speed_sequence

    calibration_node = Node(
        package="sdu_apex_autodrive",
        executable="calibration",
        # calibration.yaml is scoped to the ROS node name `calibration`.
        # Keep this name aligned so model-ID sequences are loaded instead of
        # Calibration's low-speed defaults. High-speed runs can override the
        # YAML sequence explicitly through the launch argument below.
        name="calibration",
        output="screen",
        parameters=[
            LaunchConfiguration("calibration_params"),
            calibration_parameters,
        ],
    )
    bridge_node = Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
            name="autodrive_bridge",
            output="screen",
    )
    actions = [
        SetEnvironmentVariable("AUTODRIVE_BRIDGE_RATE_HZ", "40"),
        SetEnvironmentVariable("AUTODRIVE_REQUIRE_SOURCE_TIMING", "1"),
        SetEnvironmentVariable("AUTODRIVE_ALLOW_MISSING_LIDAR", "1"),
        bridge_node,
        Node(
            package="f1tenth_localization",
            executable="sensor_odometry_node",
            name="sensor_odometry",
            output="screen",
            parameters=[
                LaunchConfiguration("sensor_odom_params"),
                {
                    "reset_enabled": True,
                    "reset_topic": "/autodrive/reset_command",
                },
            ],
            remappings=[("/tf", "/sdu/tf"), ("/tf_static", "/sdu/tf_static")],
        ),
        Node(
            package="sdu_apex_autodrive",
            executable="actuator_interface",
            name="autodrive_actuator_interface",
            output="screen",
            parameters=[
                LaunchConfiguration("actuator_params"),
                {
                    "allow_raw_throttle_override": True,
                    "allow_raw_steering_override": True,
                    "collision_reset_enabled": False,
                    "collision_terminal_stop": False,
                    "external_stop_topic": "/autodrive/roboracer_1/bridge_timing_fault",
                },
            ],
        ),
        calibration_node,
        Node(
            package="sdu_apex_autodrive",
            executable="model_id_timing_recorder",
            name="model_id_timing_recorder",
            output="screen",
            parameters=[
                {
                    "output_dir": output_dir,
                    "run_name": run_name,
                    "experiment_mode": mode,
                    "duration_sec": ParameterValue(duration, value_type=float),
                },
            ],
        ),
    ]
    # Calibration is finite for every model-ID phase list, including the
    # duration==0 command-line mode.  A normal completion or a timing fault
    # must terminate the bridge/recorder too; otherwise a stopped calibration
    # process can leave a live command publisher behind.
    actions.append(RegisterEventHandler(OnProcessExit(
        target_action=calibration_node,
        on_exit=[Shutdown(reason="model-identification process exited")],
    )))
    return actions


def generate_launch_description():
    localization_share = get_package_share_directory("f1tenth_localization")
    integration_share = get_package_share_directory("sdu_apex_autodrive")
    return LaunchDescription([
        DeclareLaunchArgument(
            "output_dir",
            default_value="/workspace/src/sdu_apex_autodrive/artifacts/simulator_trace",
        ),
        DeclareLaunchArgument(
            "run_name",
            default_value="",
            description="Directory name for the causal timing artifact",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="identification_grid",
            description="Existing calibration excitation mode",
        ),
        DeclareLaunchArgument("duration_sec", default_value="0.0"),
        DeclareLaunchArgument(
            "speed_sequence_mps",
            default_value="",
            description=(
                "Optional comma-separated speed-steps override; empty uses "
                "calibration.yaml"),
        ),
        DeclareLaunchArgument(
            "reset_between_steps",
            default_value="auto",
            description=(
                "auto resets only identification_grid; otherwise true or false"),
        ),
        DeclareLaunchArgument(
            "ground_truth_boundary_distance_m",
            default_value="450.0",
            description=(
                "Diagnostic world-origin boundary guard; lower for finite "
                "open-ground model-ID runs"),
        ),
        DeclareLaunchArgument(
            "sensor_odom_params",
            default_value=os.path.join(
                localization_share, "config", "sensor_odometry.yaml"),
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(
                integration_share, "config", "actuator_interface.yaml"),
        ),
        DeclareLaunchArgument(
            "calibration_params",
            default_value=os.path.join(
                integration_share, "config", "calibration.yaml"),
        ),
        OpaqueFunction(function=_setup),
    ])
