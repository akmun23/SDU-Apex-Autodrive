"""Launch map-aware obstacle extraction and lateral planning."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    lidar_share = get_package_share_directory('f1tenth_lidar')
    planner_share = get_package_share_directory('f1tenth_lateral_planner')
    default_scan_params = os.path.join(
        lidar_share, 'config', 'scan_splitter.yaml')
    default_planner_params = os.path.join(
        planner_share, 'config', 'lateral_planner.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'scan_splitter_params',
            default_value=default_scan_params,
            description='Scan splitter ROS parameter file',
        ),
        DeclareLaunchArgument(
            'lateral_planner_params',
            default_value=default_planner_params,
            description='Lateral planner ROS parameter file',
        ),
        DeclareLaunchArgument(
            'trajectory_file',
            default_value='',
            description='AutoDRIVE map-frame raceline CSV',
        ),
        DeclareLaunchArgument('avoidance_enabled', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(lidar_share, 'launch', 'scan_splitter.launch.py')),
            launch_arguments={
                'scan_splitter_params': LaunchConfiguration(
                    'scan_splitter_params'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(planner_share, 'launch', 'lateral_planner.launch.py')),
            launch_arguments={
                'lateral_planner_params': LaunchConfiguration(
                    'lateral_planner_params'),
                'trajectory_file': LaunchConfiguration('trajectory_file'),
                'avoidance_enabled': LaunchConfiguration('avoidance_enabled'),
            }.items(),
        ),
    ])
