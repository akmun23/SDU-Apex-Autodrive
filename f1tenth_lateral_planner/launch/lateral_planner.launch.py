"""Launch AutoDRIVE lateral planner with runtime-configurable interfaces."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('f1tenth_lateral_planner'),
        'config',
        'lateral_planner.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'lateral_planner_params',
            default_value=default_config,
            description='Lateral planner ROS parameter file',
        ),
        DeclareLaunchArgument(
            'avoidance_enabled',
            default_value='true',
            description='Enable obstacle avoidance',
        ),
        DeclareLaunchArgument(
            'trajectory_file',
            default_value='',
            description='Absolute raceline CSV; empty uses package-relative setting',
        ),
        Node(
            package='f1tenth_lateral_planner',
            executable='lateral_planner_node',
            name='lateral_planner_node',
            output='screen',
            parameters=[
                LaunchConfiguration('lateral_planner_params'),
                {
                    'avoidance_enabled': ParameterValue(
                        LaunchConfiguration('avoidance_enabled'), value_type=bool),
                    'trajectory_file': LaunchConfiguration('trajectory_file'),
                },
            ],
        ),
    ])
