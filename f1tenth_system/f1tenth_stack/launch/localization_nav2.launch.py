"""Launch map server + Nav2 AMCL with map -> world TF ownership."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    localization_share = get_package_share_directory('f1tenth_localization')
    default_params = os.path.join(
        localization_share, 'config', 'nav2_amcl_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            description='Absolute YAML path for map created from AutoDRIVE track',
        ),
        DeclareLaunchArgument(
            'amcl_params_file',
            default_value=default_params,
            description='Nav2 AMCL parameter file',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Enable only when AutoDRIVE publishes a usable /clock',
        ),
        LifecycleNode(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            namespace='',
            output='screen',
            parameters=[{
                'yaml_filename': LaunchConfiguration('map'),
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool),
            }],
        ),
        LifecycleNode(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            namespace='',
            output='screen',
            parameters=[
                LaunchConfiguration('amcl_params_file'),
                {
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'), value_type=bool),
                },
            ],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'autostart': True,
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool),
                'node_names': ['map_server', 'amcl'],
                'bond_timeout': 0.0,
            }],
        ),
    ])
