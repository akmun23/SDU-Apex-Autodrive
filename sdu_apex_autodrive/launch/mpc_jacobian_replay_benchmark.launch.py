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
    node_overrides = {
        "use_sim_time": True,
        "trajectory_file": trajectory,
        "max_speed_mps": float(os.environ.get("MPC_MAX_SPEED_MPS", "16.0")),
        "publish_diagnostics": True,
        "diagnostics_topic": "/mpc/candidate_diagnostics",
        "drive_command_topic": "/mpc/candidate_drive_command",
    }

    # The production YAML is the sole source of defaults. This diagnostic
    # launch used to silently override it with stale race weights, a 50-solve
    # cap, a zero RTI budget and a nonzero startup ramp. Apply an experiment
    # override only when the caller explicitly supplies its environment key.
    float_overrides = {
        "MPC_STARTUP_SPEED_MPS": "startup_speed_mps",
        "MPC_STARTUP_RAMP_LAPS": "startup_ramp_laps",
        "MPC_SOLVER_TOLERANCE": "solver_tolerance",
        "MPC_RHO": "admm_rho",
        "MPC_RHO_INPUT": "admm_rho_input",
        "MPC_RHO_X_EY": "admm_rho_ey",
        "MPC_RHO_X_U": "admm_rho_forward_speed",
        "MPC_RHO_X_STEERING": "admm_rho_actual_steering",
        "MPC_RHO_STEERING": "admm_rho_steering",
        "MPC_RHO_THROTTLE": "admm_rho_throttle",
        "MPC_RHO_MAX": "admm_rho_max",
        "MPC_OVER_RELAXATION": "admm_over_relaxation",
        "MPC_WEIGHT_U": "weight_u",
        "MPC_WEIGHT_U_OVERSPEED": "weight_u_overspeed",
        "MPC_WEIGHT_EY": "weight_e_y",
        "MPC_WEIGHT_EPSI": "weight_e_psi",
        "MPC_WEIGHT_V": "weight_v",
        "MPC_WEIGHT_R": "weight_r",
        "MPC_WEIGHT_SLIP": "weight_longitudinal_slip",
        "MPC_WEIGHT_STEERING_COMMAND": "weight_steering_command",
        "MPC_WEIGHT_STEERING_TARGET": "weight_steering_target",
        "MPC_WEIGHT_THROTTLE": "weight_throttle_command",
        "MPC_WEIGHT_STEERING_SLEW": "weight_steering_slew",
        "MPC_WEIGHT_THROTTLE_SLEW": "weight_throttle_slew",
        "MPC_STARTUP_THROTTLE_SEED": "startup_nominal_throttle_seed",
        "MPC_RTI_CYCLE_BUDGET_MS": "rti_cycle_budget_ms",
    }
    for env_name, parameter_name in float_overrides.items():
        if env_name in os.environ:
            node_overrides[parameter_name] = float(os.environ[env_name])

    int_overrides = {
        "MPC_MAX_ITERATIONS": "max_solver_iterations",
        "MPC_RHO_UPDATE_PERIOD": "adaptive_rho_update_period",
    }
    for env_name, parameter_name in int_overrides.items():
        if env_name in os.environ:
            node_overrides[parameter_name] = int(os.environ[env_name])
    if "MPC_JACOBIAN_THREADS" in os.environ:
        node_overrides["jacobian_threads"] = max(
            0, min(8, int(os.environ["MPC_JACOBIAN_THREADS"])))

    for env_name, parameter_name in (
            ("MPC_ADAPTIVE_RHO", "adaptive_rho"),
            ("MPC_COLD_START_EACH_SOLVE", "solver_cold_start_each_solve")):
        if env_name in os.environ:
            node_overrides[parameter_name] = os.environ[env_name].lower() in (
                "1", "true", "yes")

    node = ComposableNode(
        package="f1tenth_mpc",
        plugin="f1tenth_mpc::MpcControllerNode",
        name="mpc_controller_node",
        parameters=[
            os.path.join(mpc_share, "config", "mpc_competition.yaml"),
            node_overrides,
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
