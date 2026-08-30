"""Launch one deterministic AutoDRIVE calibration or recording test."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise RuntimeError('boolean argument expected, got %r' % value)


def _launch_setup(context):
    test = LaunchConfiguration('test').perform(context).strip().lower()
    allowed = (
        'throttle_sweep', 'throttle_steps', 'zero_throttle_decel',
        'speed_steps', 'speed_ramp', 'steering_steps', 'sensor_record')
    if test not in allowed:
        raise RuntimeError('test must be one of: %s' % ', '.join(allowed))
    with_rviz = _bool(LaunchConfiguration('with_rviz').perform(context))
    bridge_launch = (
        'bringup_graphics.launch.py' if with_rviz
        else 'bringup_headless.launch.py')
    bridge_path = os.path.join(
        get_package_share_directory('autodrive_roboracer'),
        'launch',
        bridge_launch,
    )
    actions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(bridge_path))]
    common = [LaunchConfiguration('calibration_params')]
    output_dir = LaunchConfiguration('output_dir')
    max_speed = ParameterValue(LaunchConfiguration('max_test_speed'), value_type=float)
    max_throttle = ParameterValue(LaunchConfiguration('max_throttle'), value_type=float)

    if test in ('throttle_sweep', 'throttle_steps', 'zero_throttle_decel'):
        actions.append(Node(
            package='sdu_apex_autodrive',
            executable='throttle_characterization',
            name='throttle_characterization',
            output='screen',
            parameters=common + [{
                'mode': test,
                'output_dir': output_dir,
                'maximum_test_speed_mps': max_speed,
                'maximum_throttle': max_throttle,
            }],
        ))
    elif test in ('speed_steps', 'speed_ramp'):
        actions.append(Node(
            package='sdu_apex_autodrive',
            executable='actuator_interface',
            name='autodrive_actuator_interface',
            output='screen',
            parameters=[LaunchConfiguration('actuator_params')],
        ))
        actions.append(Node(
            package='sdu_apex_autodrive',
            executable='speed_tracking_test',
            name='speed_tracking_test',
            output='screen',
            parameters=common + [{
                'mode': test,
                'output_dir': output_dir,
                'maximum_test_speed_mps': max_speed,
            }],
        ))
    elif test == 'steering_steps':
        actions.append(Node(
            package='sdu_apex_autodrive',
            executable='steering_characterization',
            name='steering_characterization',
            output='screen',
            parameters=common + [{'output_dir': output_dir}],
        ))
    else:
        actions.append(Node(
            package='sdu_apex_autodrive',
            executable='data_recorder',
            name='autodrive_data_recorder',
            output='screen',
            parameters=common + [{'output_dir': output_dir}],
        ))
    actions.append(LogInfo(msg='AutoDRIVE test=%s; native publisher ownership enforced' % test))
    return actions


def generate_launch_description():
    share = get_package_share_directory('sdu_apex_autodrive')
    return LaunchDescription([
        DeclareLaunchArgument('test', default_value='sensor_record'),
        DeclareLaunchArgument('with_rviz', default_value='true'),
        DeclareLaunchArgument(
            'output_dir',
            default_value='/workspace/src/autodrive_artifacts/calibration/raw'),
        DeclareLaunchArgument('max_test_speed', default_value='3.0'),
        DeclareLaunchArgument('max_throttle', default_value='0.10'),
        DeclareLaunchArgument(
            'calibration_params',
            default_value=os.path.join(share, 'config', 'calibration.yaml')),
        DeclareLaunchArgument(
            'actuator_params',
            default_value=os.path.join(
                share, 'config', 'actuator_interface.yaml')),
        OpaqueFunction(function=_launch_setup),
    ])
