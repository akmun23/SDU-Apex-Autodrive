"""Single user-facing launch for one AutoDRIVE racing controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


DEFAULT_MAP = '/workspace/src/autodrive_artifacts/maps/autodrive_track_multi_lap.yaml'
DEFAULT_TRAJECTORY = (
    '/workspace/src/autodrive_artifacts/trajectories/autodrive_raceline_safe.csv')


def _bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise RuntimeError('%s must be true or false, got %r' % (name, value))


def _auto_bool(value: str, automatic: bool, name: str) -> bool:
    if value.strip().lower() == 'auto':
        return automatic
    return _bool(value, name)


def _include(package: str, launch_name: str, arguments=None):
    path = os.path.join(
        get_package_share_directory(package), 'launch', launch_name)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments=(arguments or {}).items(),
    )


def _launch_setup(context):
    controller = LaunchConfiguration('controller').perform(context).strip().lower()
    allowed = ('ftg', 'pure_pursuit', 'stanley', 'mpc')
    if controller not in allowed:
        raise RuntimeError(
            'controller must be one of %s, got %r' % (', '.join(allowed), controller))

    map_path = LaunchConfiguration('map').perform(context)
    trajectory = LaunchConfiguration('trajectory').perform(context)
    with_rviz = _bool(
        LaunchConfiguration('with_rviz').perform(context), 'with_rviz')
    needs_map = controller in ('pure_pursuit', 'stanley', 'mpc')
    with_localization = _auto_bool(
        LaunchConfiguration('with_localization').perform(context),
        needs_map,
        'with_localization',
    )
    with_lateral = _auto_bool(
        LaunchConfiguration('with_lateral_planner').perform(context),
        controller == 'mpc',
        'with_lateral_planner',
    )
    auto_global = _bool(
        LaunchConfiguration('auto_global_localization').perform(context),
        'auto_global_localization',
    )
    avoidance = _bool(
        LaunchConfiguration('avoidance_enabled').perform(context),
        'avoidance_enabled',
    )

    if needs_map and not with_localization:
        raise RuntimeError('%s requires with_localization:=true' % controller)
    if controller == 'mpc' and not with_lateral:
        raise RuntimeError('mpc requires with_lateral_planner:=true')
    if with_localization and not os.path.isfile(map_path):
        raise RuntimeError('map file does not exist: %s' % map_path)
    if (needs_map or with_lateral) and not os.path.isfile(trajectory):
        raise RuntimeError('trajectory file does not exist: %s' % trajectory)

    actions = []
    bridge_launch = (
        'bringup_graphics.launch.py' if with_rviz
        else 'bringup_headless.launch.py')
    actions.append(_include('autodrive_roboracer', bridge_launch))

    if with_localization:
        actions.append(_include(
            'f1tenth_stack',
            'localization_nav2.launch.py',
            {
                'map': map_path,
                'amcl_params_file': LaunchConfiguration('amcl_params'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            },
        ))
        if auto_global:
            actions.append(Node(
                package='sdu_apex_autodrive',
                executable='auto_global_localizer',
                name='auto_global_localizer',
                output='screen',
                parameters=[
                    LaunchConfiguration('localization_bootstrap_params'),
                    {
                        'max_position_variance_m2': ParameterValue(
                            LaunchConfiguration('localization_covariance_xy_max'),
                            value_type=float),
                        'max_yaw_variance_rad2': ParameterValue(
                            LaunchConfiguration('localization_covariance_yaw_max'),
                            value_type=float),
                    },
                ],
            ))

    if with_lateral:
        actions.append(_include(
            'f1tenth_stack',
            'perception.launch.py',
            {
                'scan_splitter_params': LaunchConfiguration(
                    'scan_splitter_params'),
                'lateral_planner_params': LaunchConfiguration(
                    'lateral_planner_params'),
                'trajectory_file': trajectory,
                'avoidance_enabled': 'true' if avoidance else 'false',
            },
        ))

    if controller == 'ftg':
        actions.append(_include(
            'f1tenth_control',
            'ftg_autodrive.launch.py',
            {'params_file': LaunchConfiguration('ftg_params')},
        ))
    elif controller == 'pure_pursuit':
        actions.append(_include(
            'f1tenth_control',
            'pure_pursuit_autodrive.launch.py',
            {
                'params_file': LaunchConfiguration('path_tracking_params'),
                'trajectory_file': trajectory,
            },
        ))
    elif controller == 'stanley':
        actions.append(_include(
            'f1tenth_control',
            'stanley_autodrive.launch.py',
            {
                'params_file': LaunchConfiguration('path_tracking_params'),
                'trajectory_file': trajectory,
            },
        ))
    else:
        actions.append(_include(
            'mpc_riccati',
            'mpc_autodrive.launch.py',
            {
                'command_topic': '/cmd/controller',
                'pose_covariance_xy_max': LaunchConfiguration(
                    'localization_covariance_xy_max'),
                'pose_covariance_yaw_max': LaunchConfiguration(
                    'localization_covariance_yaw_max'),
            },
        ))

    actions.append(Node(
        package='sdu_apex_autodrive',
        executable='actuator_interface',
        name='autodrive_actuator_interface',
        output='screen',
        parameters=[LaunchConfiguration('actuator_params')],
    ))
    actions.append(LogInfo(msg=(
        'AutoDRIVE controller=%s map=%s trajectory=%s localization=%s '
        'lateral_planner=%s avoidance=%s RViz=%s actuator=direct-watchdog'
    ) % (
        controller,
        map_path if with_localization else 'unused',
        trajectory if (needs_map or with_lateral) else 'unused',
        with_localization,
        with_lateral,
        avoidance if with_lateral else False,
        with_rviz,
    )))
    return actions


def generate_launch_description():
    integration_share = get_package_share_directory('sdu_apex_autodrive')
    localization_share = get_package_share_directory('f1tenth_localization')
    control_share = get_package_share_directory('f1tenth_control')
    lidar_share = get_package_share_directory('f1tenth_lidar')
    planner_share = get_package_share_directory('f1tenth_lateral_planner')

    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='ftg'),
        DeclareLaunchArgument('map', default_value=DEFAULT_MAP),
        DeclareLaunchArgument('trajectory', default_value=DEFAULT_TRAJECTORY),
        DeclareLaunchArgument('with_rviz', default_value='true'),
        DeclareLaunchArgument('with_localization', default_value='auto'),
        DeclareLaunchArgument('auto_global_localization', default_value='true'),
        DeclareLaunchArgument('with_lateral_planner', default_value='auto'),
        DeclareLaunchArgument('avoidance_enabled', default_value='false'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'actuator_params',
            default_value=os.path.join(
                integration_share, 'config', 'actuator_interface.yaml')),
        DeclareLaunchArgument(
            'localization_bootstrap_params',
            default_value=os.path.join(
                integration_share, 'config', 'localization_bootstrap.yaml')),
        DeclareLaunchArgument(
            'amcl_params',
            default_value=os.path.join(
                localization_share, 'config', 'nav2_amcl_params.yaml')),
        DeclareLaunchArgument(
            'ftg_params',
            default_value=os.path.join(
                control_share, 'config', 'ftg_autodrive.yaml')),
        DeclareLaunchArgument(
            'path_tracking_params',
            default_value=os.path.join(
                control_share, 'config', 'path_tracking_autodrive.yaml')),
        DeclareLaunchArgument(
            'scan_splitter_params',
            default_value=os.path.join(
                lidar_share, 'config', 'scan_splitter.yaml')),
        DeclareLaunchArgument(
            'lateral_planner_params',
            default_value=os.path.join(
                planner_share, 'config', 'lateral_planner.yaml')),
        DeclareLaunchArgument(
            'localization_covariance_xy_max', default_value='0.25'),
        DeclareLaunchArgument(
            'localization_covariance_yaw_max', default_value='0.12'),
        OpaqueFunction(function=_launch_setup),
    ])
