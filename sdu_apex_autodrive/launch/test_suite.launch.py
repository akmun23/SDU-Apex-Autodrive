"""Calibration launcher with a mandatory live AutoDRIVE timing gate."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


RAW = {
    "throttle_sweep", "throttle_steps", "zero_throttle_decel", "steering_steps",
    "steering_response", "throttle_speed_grid", "identification_grid",
}
SPEED = {"speed_steps", "speed_ramp"}
ACCELERATION = {"acceleration_steps"}
ALL = RAW | SPEED | ACCELERATION | {"sensor_record", "full_suite", "timing_only"}


def _setup(context):
    mode = LaunchConfiguration("test").perform(context)
    if mode not in ALL:
        raise RuntimeError(f"unknown test mode: {mode}")

    bridge_node = Node(
        package="sdu_apex_autodrive",
        executable="autodrive_bridge_40hz",
        name="autodrive_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_bridge")),
    )

    # Sensor odometry is allowed to run during the preflight only so the gate
    # can validate the complete downstream stream. No actuator or calibration
    # process starts until the gate passes.
    sensor_odom_node = Node(
        package="f1tenth_localization",
        executable="sensor_odometry_node",
        name="sensor_odometry",
        output="screen",
        parameters=[
            LaunchConfiguration("sensor_odom_params"),
            {"reset_enabled": mode in {
                "identification_grid", "speed_steps", "speed_ramp",
                "acceleration_steps", "full_suite"},
             "reset_topic": "/autodrive/reset_command"},
        ],
        remappings=[
            ("/tf", "/sdu/tf"),
            ("/tf_static", "/sdu/tf_static"),
        ],
    )

    timing_validator = Node(
        package="sdu_apex_autodrive",
        executable="timing_validator",
        name="autodrive_timing_validator",
        output="screen",
        parameters=[{
            "duration_sec": LaunchConfiguration("timing_gate_duration_sec"),
            "startup_timeout_sec": LaunchConfiguration("timing_gate_startup_timeout_sec"),
            "report_path": LaunchConfiguration("timing_gate_report_path"),
            "require_simulation_metadata": LaunchConfiguration(
                "require_simulation_metadata"),
            "initial_grace_sec": LaunchConfiguration(
                "timing_gate_grace_sec"),
            "fail_fast": True,
        }],
    )

    def after_timing_gate(event, _context):
        if event.returncode != 0:
            return [Shutdown(
                reason=f"AutoDRIVE timing gate failed with exit code {event.returncode}")]
        if mode == "timing_only":
            return [Shutdown(reason="AutoDRIVE timing gate passed")]

        runtime_actions = []

        if mode == "identification_grid":
            # The identification run needs the local odom-frame estimator output
            # for comparison, but must not start AMCL: the open ground scene has
            # no matching map. AMCL is evaluated later on the track scene.
            runtime_actions.append(Node(
                package="f1tenth_localization",
                executable="ekf_localization_node",
                name="ekf_localization",
                output="screen",
                parameters=[
                    LaunchConfiguration("ekf_params"),
                    {"reset_enabled": True, "reset_topic": "/autodrive/reset_command"},
                ],
            ))

        if mode in SPEED or mode in ACCELERATION or mode == "full_suite":
            runtime_actions.append(Node(
                package="sdu_apex_autodrive",
                executable="actuator_interface",
                name="autodrive_actuator_interface",
                output="screen",
                parameters=[LaunchConfiguration("actuator_params"), {
                    "input_topic": "/cmd/acceleration"
                    if mode in ACCELERATION else "/cmd/speed",
                    "command_mode": "acceleration"
                    if mode in ACCELERATION else "speed",
                    "allow_raw_throttle_override": True,
                }],
            ))

        # Continue observing while the finite run is active. A timing fault
        # during the run shuts the run down instead of silently contaminating
        # the saved model data after a clean preflight.
        timing_monitor = Node(
            package="sdu_apex_autodrive",
            executable="timing_validator",
            name="autodrive_timing_monitor",
            output="screen",
            parameters=[{
                "duration_sec": LaunchConfiguration("timing_monitor_duration_sec"),
                "startup_timeout_sec": LaunchConfiguration("timing_gate_startup_timeout_sec"),
                "report_path": LaunchConfiguration("timing_monitor_report_path"),
                "require_simulation_metadata": LaunchConfiguration(
                "require_simulation_metadata"),
                "initial_grace_sec": LaunchConfiguration(
                    "timing_monitor_grace_sec"),
                # The long-run observer is one Python ROS subscriber among
                # the recorder and odometry nodes. A single callback wake-up
                # can miss a derived /odom sample even when the recorder's
                # exact source-event CSV is continuous. Keep accumulating the
                # report and let the CSV source audit decide data acceptance.
                "fail_fast": False,
                "max_source_gap_fraction": LaunchConfiguration(
                    "timing_monitor_max_source_gap_fraction"),
                "reject_encoder_outliers": False,
                # The exact recorder source-event CSV is authoritative for
                # acceptance. A single downstream ROS executor wake-up can
                # bunch a derived callback without changing native cadence.
                "reject_arrival_timing": False,
            }],
        )
        runtime_actions.append(timing_monitor)

        calibration_node = Node(
            package="sdu_apex_autodrive",
            executable="calibration",
            name="calibration",
            output="screen",
            parameters=[
                LaunchConfiguration("calibration_params"),
                {
                    "mode": mode,
                    "output_dir": LaunchConfiguration("output_dir"),
                },
            ],
        )
        runtime_actions.append(TimerAction(
            period=LaunchConfiguration("timing_monitor_grace_sec"),
            actions=[calibration_node],
        ))
        runtime_actions.append(RegisterEventHandler(OnProcessExit(
            target_action=timing_monitor,
            on_exit=[Shutdown(
                reason="AutoDRIVE timing monitor failed or reached its limit")],
        )))
        runtime_actions.append(RegisterEventHandler(OnProcessExit(
            target_action=calibration_node,
            on_exit=[Shutdown(reason="calibration completed")],
        )))
        return runtime_actions

    return [
        bridge_node,
        sensor_odom_node,
        timing_validator,
        RegisterEventHandler(OnProcessExit(
            target_action=timing_validator,
            on_exit=after_timing_gate,
        )),
    ]


def generate_launch_description():
    share = get_package_share_directory("sdu_apex_autodrive")
    localization = get_package_share_directory("f1tenth_localization")
    return LaunchDescription([
        DeclareLaunchArgument("test", default_value="timing_only"),
        DeclareLaunchArgument("start_bridge", default_value="true"),
        DeclareLaunchArgument("publish_camera", default_value="false"),
        DeclareLaunchArgument("timing_gate_duration_sec", default_value="15.0"),
        DeclareLaunchArgument("timing_gate_startup_timeout_sec", default_value="30.0"),
        DeclareLaunchArgument("timing_gate_grace_sec", default_value="2.0"),
        DeclareLaunchArgument("require_simulation_metadata", default_value="true"),
        DeclareLaunchArgument("timing_monitor_duration_sec", default_value="3600.0"),
        DeclareLaunchArgument("timing_monitor_grace_sec", default_value="2.0"),
        DeclareLaunchArgument(
            "timing_monitor_max_source_gap_fraction", default_value="0.001"),
        DeclareLaunchArgument(
            "timing_gate_report_path",
            default_value="/tmp/sdu_autodrive_timing_gate.json",
        ),
        DeclareLaunchArgument(
            "timing_monitor_report_path",
            default_value="/tmp/sdu_autodrive_timing_monitor.json",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/workspace/src/sdu_apex_autodrive/artifacts/calibration/raw",
        ),
        DeclareLaunchArgument(
            "calibration_params",
            default_value=os.path.join(share, "config", "calibration.yaml"),
        ),
        DeclareLaunchArgument(
            "actuator_params",
            default_value=os.path.join(share, "config", "actuator_interface.yaml"),
        ),
        DeclareLaunchArgument(
            "sensor_odom_params",
            default_value=os.path.join(localization, "config", "sensor_odometry.yaml"),
        ),
        DeclareLaunchArgument(
            "ekf_params",
            default_value=os.path.join(localization, "config", "ekf.yaml"),
        ),
        SetEnvironmentVariable(
            name="AUTODRIVE_BRIDGE_PUBLISH_CAMERA",
            value=LaunchConfiguration("publish_camera"),
        ),
        OpaqueFunction(function=_setup),
    ])
