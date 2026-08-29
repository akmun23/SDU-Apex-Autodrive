"""Launch Stanley using map-frame AMCL pose and native AutoDRIVE odom."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('f1tenth_control'),
        'config',
        'path_tracking_autodrive.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument(
            'trajectory_file',
            default_value='',
            description='AutoDRIVE map-frame raceline CSV; local raceline may replace it',
        ),
        ComposableNodeContainer(
            name='stanley_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='f1tenth_control',
                    plugin='f1tenth_control::StanleyNode',
                    name='stanley_node',
                    parameters=[
                        LaunchConfiguration('params_file'),
                        {'trajectory_file': LaunchConfiguration('trajectory_file')},
                    ],
                ),
            ],
            output='screen',
        ),
    ])
