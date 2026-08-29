"""Launch independent AutoDRIVE LiDAR -> FTG -> mux -> adapter pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    integration_share = get_package_share_directory('sdu_apex_autodrive')
    control_share = get_package_share_directory('f1tenth_control')
    default_adapter = os.path.join(integration_share, 'config', 'adapter.yaml')
    default_ftg = os.path.join(control_share, 'config', 'ftg_autodrive.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('adapter_params', default_value=default_adapter),
        DeclareLaunchArgument('ftg_params', default_value=default_ftg),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(integration_share, 'launch', 'autodrive_base.launch.py')),
            launch_arguments={
                'adapter_params': LaunchConfiguration('adapter_params'),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(control_share, 'launch', 'ftg_autodrive.launch.py')),
            launch_arguments={
                'params_file': LaunchConfiguration('ftg_params'),
            }.items(),
        ),
    ])
