"""Launch command arbitration and single safe AutoDRIVE actuator adapter."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('sdu_apex_autodrive')
    default_mux = os.path.join(share, 'config', 'mux.yaml')
    default_adapter = os.path.join(share, 'config', 'adapter.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('mux_params', default_value=default_mux),
        DeclareLaunchArgument('adapter_params', default_value=default_adapter),
        Node(
            package='ackermann_mux',
            executable='ackermann_mux',
            name='ackermann_mux',
            output='screen',
            parameters=[LaunchConfiguration('mux_params')],
            remappings=[('ackermann_cmd', '/cmd/selected')],
        ),
        Node(
            package='sdu_apex_autodrive',
            executable='command_adapter',
            name='autodrive_command_adapter',
            output='screen',
            parameters=[LaunchConfiguration('adapter_params')],
        ),
    ])
