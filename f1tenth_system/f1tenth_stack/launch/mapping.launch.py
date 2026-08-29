"""Launch SLAM Toolbox using AutoDRIVE world/base/LiDAR frames."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    stack_share = get_package_share_directory('f1tenth_stack')
    slam_share = get_package_share_directory('slam_toolbox')
    default_params = os.path.join(stack_share, 'config', 'slam_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'slam_params_file',
            default_value=default_params,
            description='SLAM Toolbox parameter file',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Enable only when AutoDRIVE publishes a usable /clock',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_share, 'launch', 'online_async_launch.py')),
            launch_arguments={
                'slam_params_file': LaunchConfiguration('slam_params_file'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }.items(),
        ),
    ])
