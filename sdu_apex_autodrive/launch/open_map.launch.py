"""Launch an active map server for a featureless ROS software-test map."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', description='Absolute blank-map YAML path'),
        LifecycleNode(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            namespace='',
            output='screen',
            parameters=[{'yaml_filename': LaunchConfiguration('map')}],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_open_map',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'],
                'bond_timeout': 0.0,
            }],
        ),
    ])
