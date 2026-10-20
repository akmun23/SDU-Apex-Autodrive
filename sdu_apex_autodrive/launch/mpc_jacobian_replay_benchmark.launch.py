"""Temporary isolated MPC component launch for recorded-input timing replay."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    mpc_share = get_package_share_directory("f1tenth_mpc")
    trajectory = os.environ.get(
        "MPC_TRAJECTORY_FILE",
        os.path.join(
            repo_root, "f1tenth_planning", "trajectories",
            "autodrive_practice_20260924_b",
            "autodrive_practice_20260924_b_mintime_raceline.csv"))
    node = ComposableNode(
        package="f1tenth_mpc",
        plugin="f1tenth_mpc::MpcControllerNode",
        name="mpc_controller_node",
        parameters=[
            os.path.join(mpc_share, "config", "mpc_competition.yaml"),
            {
                "use_sim_time": True,
                "trajectory_file": trajectory,
                "max_speed_mps": float(os.environ.get(
                    "MPC_MAX_SPEED_MPS", "16.0")),
                "startup_speed_mps": float(os.environ.get(
                    "MPC_STARTUP_SPEED_MPS", "1.5")),
                "startup_ramp_laps": float(os.environ.get(
                    "MPC_STARTUP_RAMP_LAPS", "1.0")),
                "publish_diagnostics": True,
                "diagnostics_topic": "/mpc/candidate_diagnostics",
                "drive_command_topic": "/mpc/candidate_drive_command",
                "jacobian_threads": max(0, min(4, int(os.environ.get(
                    "MPC_JACOBIAN_THREADS", "4")))),
                "max_solver_iterations": int(os.environ.get(
                    "MPC_MAX_ITERATIONS", "50")),
                "solver_tolerance": float(os.environ.get(
                    "MPC_SOLVER_TOLERANCE", "0.02")),
                "admm_rho": float(os.environ.get("MPC_RHO", "7.0")),
                "admm_rho_input": float(os.environ.get(
                    "MPC_RHO_INPUT", "7.0")),
                "admm_rho_ey": float(os.environ.get(
                    "MPC_RHO_X_EY", "0.0")),
                "admm_rho_forward_speed": float(os.environ.get(
                    "MPC_RHO_X_U", "0.0")),
                "admm_rho_actual_steering": float(os.environ.get(
                    "MPC_RHO_X_STEERING", "0.0")),
                "admm_rho_steering": float(os.environ.get(
                    "MPC_RHO_STEERING", "0.0")),
                "admm_rho_throttle": float(os.environ.get(
                    "MPC_RHO_THROTTLE", "0.0")),
                "admm_rho_max": float(os.environ.get(
                    "MPC_RHO_MAX", "32.0")),
                "admm_over_relaxation": float(os.environ.get(
                    "MPC_OVER_RELAXATION", "1.6")),
                "adaptive_rho": os.environ.get(
                    "MPC_ADAPTIVE_RHO", "true").lower() in ("1", "true", "yes"),
                "solver_cold_start_each_solve": os.environ.get(
                    "MPC_COLD_START_EACH_SOLVE", "false").lower()
                    in ("1", "true", "yes"),
                "adaptive_rho_update_period": int(os.environ.get(
                    "MPC_RHO_UPDATE_PERIOD", "10")),
                "weight_u": float(os.environ.get("MPC_WEIGHT_U", "500")),
                "weight_u_overspeed": float(os.environ.get(
                    "MPC_WEIGHT_U_OVERSPEED", "300")),
                "weight_e_y": float(os.environ.get("MPC_WEIGHT_EY", "125")),
                "weight_e_psi": float(os.environ.get("MPC_WEIGHT_EPSI", "31.25")),
                "weight_v": float(os.environ.get("MPC_WEIGHT_V", "5.0")),
                "weight_r": float(os.environ.get("MPC_WEIGHT_R", "7.8125")),
                "weight_longitudinal_slip": float(os.environ.get(
                    "MPC_WEIGHT_SLIP", "5.0")),
                "weight_steering_command": float(os.environ.get(
                    "MPC_WEIGHT_STEERING_COMMAND", "3.0")),
                "weight_steering_target": float(os.environ.get(
                    "MPC_WEIGHT_STEERING_TARGET", "5.0")),
                "weight_throttle_command": float(os.environ.get(
                    "MPC_WEIGHT_THROTTLE", "0.1")),
                "weight_steering_slew": float(os.environ.get(
                    "MPC_WEIGHT_STEERING_SLEW", "25.0")),
                "weight_throttle_slew": float(os.environ.get(
                    "MPC_WEIGHT_THROTTLE_SLEW", "2.0")),
                "startup_nominal_throttle_seed": float(os.environ.get(
                    "MPC_STARTUP_THROTTLE_SEED", "0.0")),
                "rti_cycle_budget_ms": 0.0,
            },
        ],
    )
    container = ComposableNodeContainer(
        name="mpc_replay_benchmark_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[node],
        output="screen",
    )
    return LaunchDescription([container])
