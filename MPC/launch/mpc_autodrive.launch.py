"""Launch the CPU Riccati-ADMM controller against AutoDRIVE topics."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _argument(name, default, description):
    return DeclareLaunchArgument(name, default_value=default, description=description)


def _environment(variable, argument):
    return SetEnvironmentVariable(variable, LaunchConfiguration(argument))


def generate_launch_description():
    arguments = [
        _argument(
            'odom_topic',
            '/autodrive/roboracer_1/odom',
            'AutoDRIVE odometry topic',
        ),
        _argument('pose_topic', '/amcl_pose', 'Localized map-frame pose topic'),
        _argument('local_path_topic', '/local_raceline', 'Local planner path topic'),
        _argument('command_topic', '/cmd/controller', 'Common Ackermann controller output'),
        _argument(
            'steering_feedback_topic',
            '/autodrive/roboracer_1/steering',
            'Raw AutoDRIVE steering feedback topic',
        ),
        _argument('pose_frame', 'map', 'Required localization pose frame'),
        _argument('path_frame', 'map', 'Required local path frame'),
        _argument('command_frame', 'roboracer_1', 'Ackermann command frame'),
        _argument(
            'steering_feedback_radians',
            '0',
            'Accept feedback as radians only after validating its units (0/1)',
        ),
        _argument('watchdog_timeout', '0.3', 'Odometry/output timeout in seconds'),
        _argument('local_path_timeout', '0.5', 'Local path timeout in seconds'),
        _argument('pose_covariance_xy_max', '0.25', 'Maximum AMCL x/y variance'),
        _argument('pose_covariance_yaw_max', '0.12', 'Maximum AMCL yaw variance'),
        _argument('pose_required_good_updates', '5', 'Qualified AMCL poses before drive'),
        _argument('drive_republish_period_ms', '20', 'Command republish period'),
        _argument('raceline_speed_margin', '0.3', 'Speed above path reference allowed'),
        _argument('verbose', '0', 'Verbose controller logging (0/1)'),
        _argument('solver_csv_log', '0', 'Per-cycle solver CSV logging (0/1)'),
    ]

    environment = [
        _environment('MPC_ODOM_TOPIC', 'odom_topic'),
        _environment('MPC_EKF_TOPIC', 'pose_topic'),
        _environment('MPC_LOCAL_RACELINE_TOPIC', 'local_path_topic'),
        _environment('MPC_DRIVE_TOPIC', 'command_topic'),
        _environment('MPC_STEERING_FEEDBACK_TOPIC', 'steering_feedback_topic'),
        _environment('MPC_POSE_FRAME', 'pose_frame'),
        _environment('MPC_PATH_FRAME', 'path_frame'),
        _environment('MPC_COMMAND_FRAME', 'command_frame'),
        _environment('MPC_STEERING_FEEDBACK_RADIANS', 'steering_feedback_radians'),
        _environment('MPC_WATCHDOG_TIMEOUT', 'watchdog_timeout'),
        _environment('MPC_RACELINE_TIMEOUT', 'local_path_timeout'),
        _environment('MPC_POSE_COVARIANCE_XY_MAX', 'pose_covariance_xy_max'),
        _environment('MPC_POSE_COVARIANCE_YAW_MAX', 'pose_covariance_yaw_max'),
        _environment('MPC_POSE_REQUIRED_GOOD_UPDATES', 'pose_required_good_updates'),
        _environment('MPC_DRIVE_REPUBLISH_PERIOD_MS', 'drive_republish_period_ms'),
        _environment('MPC_RACELINE_SPEED_MARGIN', 'raceline_speed_margin'),
        _environment('MPC_VERBOSE', 'verbose'),
        _environment('MPC_SOLVER_CSV_LOG', 'solver_csv_log'),
    ]

    node = Node(
        package='mpc_riccati',
        executable='mpc_autodrive_node',
        name='mpc_autodrive_node',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(arguments + environment + [node])
