"""Launch map-aware obstacle extraction for native AutoDRIVE LiDAR."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('f1tenth_lidar'),
        'config',
        'scan_splitter.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'scan_splitter_params',
            default_value=default_config,
            description='Scan splitter ROS parameter file',
        ),
        Node(
            package='f1tenth_lidar',
            executable='scan_splitter_node',
            name='scan_splitter_node',
            output='screen',
            parameters=[LaunchConfiguration('scan_splitter_params')],
        ),
    ])
