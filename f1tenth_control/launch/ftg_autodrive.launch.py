"""Launch Follow The Gap on native AutoDRIVE topics."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('f1tenth_control'),
        'config',
        'ftg_autodrive.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_config,
            description='FTG ROS parameter file',
        ),
        ComposableNodeContainer(
            name='ftg_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='f1tenth_control',
                    plugin='f1tenth_control::FTGNode',
                    name='ftg_node',
                    parameters=[LaunchConfiguration('params_file')],
                    extra_arguments=[{'use_intra_process_comms': True}],
                ),
            ],
            output='screen',
        ),
    ])
