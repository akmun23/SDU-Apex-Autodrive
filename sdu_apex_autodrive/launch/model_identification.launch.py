"""Development-only causal model-identification launch.

This launch keeps the racing controller out of the experiment. The existing
calibration node supplies bounded diagnostic excitation, the normal actuator
interface applies it, and model_id_timing_recorder captures timing plus
allowed sensor/feedback events. Simulator ground truth is not subscribed to
by the recorder or used for runtime control.
"""

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
    calibration_node = Node(
        package="sdu_apex_autodrive",
        executable="calibration",
        name="model_id_excitation",
        output="screen",
        parameters=[
            LaunchConfiguration("calibration_params"),
            {
                "mode": mode,
                "output_dir": artifact_dir,
                "duration_sec": ParameterValue(duration, value_type=float),
                "capture_source_events": True,
                # The identification grid has explicit teleport epochs. The
                # continuous throttle/speed sweeps deliberately keep one
                # plant trajectory so their recursive transitions remain
                # causal; reset is enabled only for the grid profile.
                "reset_between_steps": mode == "identification_grid",
            },
        ],
    )
    actions = [
        SetEnvironmentVariable("AUTODRIVE_ALLOW_MISSING_LIDAR", "1"),
        Node(
            package="sdu_apex_autodrive",
            executable="autodrive_bridge_40hz",
            name="autodrive_bridge",
            output="screen",
        ),
        Node(
            package="f1tenth_localization",
            executable="sensor_odometry_node",
            name="sensor_odometry",
            output="screen",
            parameters=[
                LaunchConfiguration("sensor_odom_params"),
                {"reset_enabled": False},
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
                    "collision_reset_enabled": False,
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
    if duration > 0.0:
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=calibration_node,
            on_exit=[Shutdown(reason="model-identification duration reached")],
        )))
    return actions


def generate_launch_description():
    localization_share = get_package_share_directory("f1tenth_localization")
    integration_share = get_package_share_directory("sdu_apex_autodrive")
    return LaunchDescription([
        DeclareLaunchArgument(
            "output_dir",
            default_value="/workspace/src/sdu_apex_autodrive/artifacts/model_id",
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
