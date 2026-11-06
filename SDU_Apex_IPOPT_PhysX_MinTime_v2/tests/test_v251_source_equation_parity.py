"""Targeted equation checks for IPOPT v2.5.1; no solver/simulator is run."""
from __future__ import annotations

import math
import csv
import inspect
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "SDU_Apex_IPOPT_PhysX_MinTime_v2"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from recover import LEGACY, load_modules  # noqa: E402
from native_validate import _menger_curvature, export_mpc  # noqa: E402
from seed_projection import project_algebraics  # noqa: E402
from run_ipopt import write_midpoint_witness  # noqa: E402
from source_vehicle import (  # noqa: E402
    _cpp_array, _cpp_number, _unity_scene_vehicle_parameters,
    _validate_reset_wheel_geometry, bind_current_wheel_moment_arms,
    current_model)


@pytest.fixture(scope="module")
def candidate():
    optimizer, wheel, _, root, provenance = load_modules(
        ROOT, PACKAGE, allow_fetch=False)
    model, metadata = current_model(ROOT, wheel)
    bind_current_wheel_moment_arms(ROOT, optimizer, wheel)
    return optimizer, wheel, model, root, provenance, metadata


def test_candidate_is_v251_and_uses_source_geometry(candidate):
    optimizer, wheel, model, root, provenance, metadata = candidate
    assert provenance["candidate"] == "v2.5.6-positive-drive-algebraic-source-state"
    assert provenance["algebraic_load_and_acceleration_states_eliminated"] is True
    assert provenance["algebraic_elimination_is_source_equation_substitution"] is True
    minimum = re.search(r'#define\s+MPC_THROTTLE_MIN_DRIVE\s+([0-9.eE+-]+)',
        (ROOT/'f1tenth_mpc/include/mpc_types.h').read_text())
    assert minimum is not None
    assert provenance["mpc_throttle_min_drive"] == pytest.approx(float(minimum[1]))
    assert provenance["optimizer_input_is_positive_throttle_only"] is True
    assert provenance["optimizer_minimum_throttle_matches_mpc"] is True
    assert provenance["ipopt_symbolic_graph_expansion"] is False
    assert provenance["ipopt_progress_markers"] is True
    assert "_working_solver_v251_limited_memory" in str(root)
    raw = (ROOT / "f1tenth_mpc/src/unity_source_plant.cpp").read_text()
    assert wheel.WHEEL_FORCE_APPLICATION_Z_M == pytest.approx(
        tuple(value - _cpp_number(raw, "kWheelRadiusM") for value in
              _cpp_array(raw, "kWheelColliderTransformZCom", 4)), abs=1e-12)
    assert optimizer.WHEEL_X_M == pytest.approx(_cpp_array(raw, "kWheelX", 4))
    assert optimizer.WHEEL_Y_M == pytest.approx(_cpp_array(raw, "kWheelY", 4))
    assert model.sprung_mass_kg == pytest.approx(
        _cpp_array(raw, "kSprungMassKg", 4), abs=1e-12)
    assert metadata["model_metadata"]["reset_json_sprung_mass_kg"] != pytest.approx(
        model.sprung_mass_kg, abs=1e-8)
    generated = (root / LEGACY /
                 "f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py")
    source = generated.read_text()
    wheel_source = generated.with_name("unity_wheel_model.py").read_text()
    assert source.count("opti.solver('ipopt', {'expand': False") == 2
    assert "opti.solver('ipopt', {'expand': True" not in source
    assert source.count("IPOPT stage solver-interface setup begins") == 1
    assert source.count("IPOPT time-continuation solver setup (one-time)") == 1
    assert "Phase-I elastic seed evaluation begins" in source
    assert "Phase-I hard-bound preparation begins" in source
    assert "Phase-I hard-bound preparation complete" in source
    assert "Phase-I entering CasADi/IPOPT solve call" in source
    assert "Phase-I returned from CasADi/IPOPT solve call" in source
    assert "Phase-I IPOPT iteration={int(iteration)}" in source
    assert "elastic_residual_vector = ca.vertcat" in source
    assert "phase1_bounds_cache['low']" in source
    assert "phase1_bounds_cache['high']" in source
    assert "_, low, high, _, _ = numeric_evaluation(primal_vector(initial_snapshot))" not in source
    assert "Phase-I initial numeric constraint evaluation complete" in source
    assert "brake_mode" not in source
    assert "brake_torque" not in source
    assert "_brake_phases" not in source
    assert not re.search(r"\bbrakes?\b|\bbraking\b|zero.throttle|CAWB",
                         source, re.IGNORECASE)
    assert "phase_modes" not in source
    assert "physx_no_drive_wheel_speed_interval_casadi" not in wheel_source
    assert not re.search(r"\bbrakes?\b|\bbraking\b|zero.throttle|CAWB",
                         wheel_source, re.IGNORECASE)
    assert "brake_mode_seed" not in str(inspect.signature(
        optimizer.solve_bottom_up))
    assert "opti.bounded(UNITY_MPC_THROTTLE_MIN_DRIVE, throttle[k], 1.0)" in source
    # The similarly named `_working_solver_limited_memory` directory is a
    # stale historical cache. Lock acceleration semantics to the exact module
    # selected by load_modules/run_ipopt: ax/ay are du/dt,dv/dt here.
    assert "u_dot = fx / model.mass_kg + gravity_body[0]" in source
    assert "v_dot = fy / model.mass_kg + gravity_body[1]" in source
    assert "ax, ay = u_dot, v_dot" in source
    assert "accel_xy_nodes.append(ca.vertcat(ax, ay))" in source
    assert "add_constraint(ax - u_dot == 0" not in source
    assert "add_constraint(ay - v_dot == 0" not in source
    assert "normal_load_nodes.append(ca.vertcat(*predicted_load))" in source
    assert "normal_load = ca.horzcat(*normal_load_nodes)" in source
    assert "accel_xy = ca.horzcat(*accel_xy_nodes)" in source
    assert "normal_load = opti.variable(4, n)" not in source
    assert "accel_xy = opti.variable(2, n)" not in source
    assert "physical_vars = (x, qdelta, throttle, *elastic_vars)" in source
    assert "phase_lengths = opti.parameter(phase_count)" in source
    assert "Initial-only NLP dimensions: variables={int(opti.nx)}, constraints={int(opti.ng)}, dynamic_residuals={dynamic_residuals}" in source
    assert "longitudinal_dot = ax\n        lateral_dot = ay" in source
    assert "longitudinal_dot = ax + yaw_rate[k] * v[k]" not in source
    assert "_unity_euler_rates_casadi" in source
    assert "world_vertical_force_acceleration" in source
    assert "yaw_euler_rate / sdot - curvature" in source
    assert "yaw_acceleration / sdot" in source
    assert "yaw_dot / sdot" not in source
    assert "path_lateral_speed_nodes[k]" in source
    assert "path_lateral_speed_nodes[kp]" in source
    assert "lateral_path_rate = path_lateral_speed / sdot" in source
    assert "_unity_body_acceleration_numpy" in source
    assert "force_x_values / model.mass_kg - model.drag_per_s" not in source
    assert "unity_wheel_target_raceline.csv" in source
    assert "candidate['steering_rad'][k]" not in source.split(
        "raceline = output_dir / 'unity_wheel_target_raceline.csv'", 1)[1].split(
            'state_path =', 1)[0]
    assert "'witness_controls_are_runtime_commands': False" in source
    assert "_physx_filtered_tire_load_casadi" in wheel.__dict__
    assert "midpoint_force_nodes" in source
    assert "longitudinal_force_balance_midpoint_k" not in source
    assert "lateral_force_balance_midpoint_k" not in source
    assert "yaw_moment_balance_midpoint_k" not in source
    assert "heave_force_balance_midpoint_k" not in source
    assert "roll_moment_balance_midpoint_k" not in source
    assert "pitch_moment_balance_midpoint_k" not in source
    assert "midpoint_acceleration_nodes" in source
    assert "mid_world_vertical_force_acceleration" in source
    assert "midpoint_load = list(mid_suspension_load)" in source
    assert "midpoint_forward_wheel_rotation_w" in source
    assert "midpoint_normal_load_w" in source
    assert "midpoint_unity_suspension_travel_w" in source
    assert "midpoint_steering_angle_k" not in source
    assert "midpoint_steering_rate_k" not in source
    assert "midpoint_unity_positive_cawd_drive_k" not in source
    assert "unity_spring_damper_load_midpoint_w" not in source
    assert "periodic_collocation_state_midpoint_ey_k" not in source
    assert "periodic_collocation_state_midpoint_epsi_k" not in source
    assert "six body force/moment balance equalities at OCP nodes" in source
    assert "not extra midpoint collocation equalities" in source
    assert "'yaw_moment_balance_'" in source
    assert "'heave_force_balance_'" in source
    assert "'roll_moment_balance_'" in source
    assert "'pitch_moment_balance_'))]" in source
    assert "0.5 * (accel_xy[0, k] + accel_xy[0, kp])\n                       - (mid_mz" not in source
    assert "def _periodic_suspension_seed(" not in source
    assert "source force evaluation at seeded suspension state" in source
    assert "source_curve_speed = np.sqrt(" in source
    assert "wall_clearance_m: float=0.05" in source
    assert "seed_v = np.zeros(n)" in source
    assert "0.15532 * kappa_seed" not in source
    assert "seed_lateral_accel * 0.72" not in source
    assert "min(5.5, max_speed_mps - 0.2)" not in source
    assert "worst_constraint_label" in source
    assert "initialization_consistency_iterations = 1" in source
    assert "UNITY_MPC_THROTTLE_MIN_DRIVE = 0.001" in source
    assert "opti.bounded(UNITY_MPC_THROTTLE_MIN_DRIVE," in source
    assert "opti.bounded(0.015, throttle[k], 1.0)" not in source
    # Acceleration is an exact source-equation expression after algebraic
    # state elimination, not an Opti decision variable. Calling set_initial on
    # it raises CasADi's "arbitrary expression" error before Phase-I starts.
    assert "opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))" not in source
    assert "set_initial(throttle, seed_throttle)" in source
    assert "seed_ax_model = seed_fx / model.mass_kg" in source
    assert "seed_ay_model = seed_fy / model.mass_kg" in source
    assert "*seed_fx_w, seed_ax, seed_ay)]" not in source
    run_source = (PACKAGE / "run_ipopt.py").read_text()
    assert "--brake-zones" not in run_source
    assert "brake_mode_seed" not in run_source
    assert "brake_mask" not in run_source
    assert "default=.06" in run_source
    assert "exact native PhysX 1ms validation" not in run_source
    assert "source_feasibility_sample_gap_max_m" in run_source
    assert "source_feasibility_sampling_is_continuous_certification" in run_source
    assert "validate_initial_diagnostic_seed_mode" in run_source
    assert "initial_guess_source" in run_source


def test_reset_wheel_force_point_is_checked_in_source_frame():
    wheel = {
        "wheel_center_offset_com_body_m": (-0.12, -0.02, 0.18),
        "force_application_offset_com_body_m": (-0.12, -0.079, 0.18),
    }
    _validate_reset_wheel_geometry(
        0, wheel, wheel_x=0.18, wheel_y=0.12,
        transform_z=-0.02, radius=0.059)

    wrong_force_point = dict(wheel)
    wrong_force_point["force_application_offset_com_body_m"] = (
        -0.12, -0.070, 0.18)
    with pytest.raises(RuntimeError, match="force point disagrees"):
        _validate_reset_wheel_geometry(
            0, wrong_force_point, wheel_x=0.18, wheel_y=0.12,
            transform_z=-0.02, radius=0.059)


def test_optimizer_export_separates_target_from_control_witness(candidate, tmp_path):
    optimizer = candidate[0]
    n = 4
    scalar = lambda value: np.full(n, value, dtype=np.float64)
    witness = {
        "x_m": np.array([0.0, 1.0, 1.0, 0.0]),
        "y_m": np.array([0.0, 0.0, 1.0, 1.0]),
        "s_ref_m": np.arange(n, dtype=np.float64),
        "route_arc_m": np.array([0.0, 1.0, 2.0, 3.0]),
        "speed_mps": scalar(6.0),
        "ey_m": scalar(0.0), "epsi_rad": scalar(0.0),
        "lateral_speed_mps": scalar(0.1), "yaw_rate_radps": scalar(1.2),
        "steering_rad": scalar(0.25), "steering_rate_radps": scalar(0.0),
        "body_height_m": scalar(0.08), "body_height_rate_mps": scalar(0.0),
        "roll_rad": scalar(0.0), "roll_rate_radps": scalar(0.0),
        "pitch_rad": scalar(0.0), "pitch_rate_radps": scalar(0.0),
        "throttle_norm": scalar(0.5), "ax_mps2": scalar(0.2),
        "ay_mps2": scalar(1.0), "normal_load_n": np.ones((4, n)) * 8.0,
        "slip_x": np.zeros((4, n)), "slip_y": np.zeros((4, n)),
    }
    witness["path_curvature_per_m"] = _menger_curvature(np.column_stack(
        (witness["x_m"], witness["y_m"])))
    centerline = tmp_path / "centerline.csv"
    centerline.write_text("reference\n")
    report = {}
    output = tmp_path / "candidate"
    optimizer.write_candidate(
        witness, report, SimpleNamespace(count=n), centerline, output)

    target_path = output / "unity_wheel_target_raceline.csv"
    target_lines = target_path.read_text().splitlines()
    assert target_lines[0] == (
        "# s_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps")
    target_rows = list(csv.reader(target_lines[1:]))
    assert len(target_rows) == n
    assert all(len(row) == 6 for row in target_rows)
    assert [float(row[5]) for row in target_rows] == pytest.approx([6.0] * n)
    assert "steering" not in target_lines[0] and "throttle" not in target_lines[0]

    witness_path = output / "unity_wheel_feasibility_witness_states.csv"
    witness_header = witness_path.read_text().splitlines()[0]
    assert "witness_steering_rad" in witness_header
    assert "witness_throttle_norm" in witness_header
    assert report["trajectory_semantics"].startswith("target geometry")
    assert report["witness_controls_are_runtime_commands"] is False


def test_runtime_target_export_sweeps_footprint_between_knots(tmp_path):
    class Grid:
        resolution = 0.05

        def __init__(self, blocked_midpoint=False):
            self.blocked_midpoint = blocked_midpoint

        def distance(self, xy):
            xy = np.asarray(xy)
            return np.full(xy.shape[:-1], 10.0)

        def footprint_clearance(self, x, y, psi, *, margin):
            del psi, margin
            if self.blocked_midpoint and math.hypot(x - 0.5, y) < 0.03:
                return -0.01
            return 0.5

    candidate = {
        "x_m": np.array([0.0, 1.0, 1.0, 0.0]),
        "y_m": np.array([0.0, 0.0, 1.0, 1.0]),
        "speed_mps": np.full(4, 6.0),
        "ax_mps2": np.full(4, 0.2),
    }
    candidate["path_curvature_per_m"] = _menger_curvature(np.column_stack(
        (candidate["x_m"], candidate["y_m"])))
    target = export_mpc(candidate, Grid(), tmp_path / "target.csv", 0.05)
    text = target.read_text()
    assert text.splitlines()[0] == (
        "# s_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps,"
        "acceleration_mps2,d_left_m,d_right_m")
    assert "steering" not in text.splitlines()[0]
    assert "throttle" not in text.splitlines()[0]
    exported_kappa = np.asarray(
        [float(row.split(",")[4]) for row in text.splitlines()[1:]])
    assert exported_kappa == pytest.approx(
        candidate["path_curvature_per_m"], abs=1e-9)

    with pytest.raises(ValueError, match="swept original-map footprint"):
        export_mpc(candidate, Grid(blocked_midpoint=True),
                   tmp_path / "blocked.csv", 0.05)

    mismatched = dict(candidate)
    mismatched["path_curvature_per_m"] = np.zeros(4)
    with pytest.raises(ValueError, match="curvature target does not match"):
        export_mpc(mismatched, Grid(), tmp_path / "mismatched.csv", 0.05)


def test_midpoint_witness_is_persisted_separately_from_target(tmp_path):
    n = 3
    candidate = {
        "midpoint_state": np.zeros((16, n)),
        "midpoint_s_ref_m": np.arange(n, dtype=np.float64) + 0.5,
        "midpoint_wheel_omega_radps": np.ones((4, n)),
        "midpoint_steering_rad": np.zeros(n),
        "midpoint_steering_rate_radps": np.zeros(n),
        "midpoint_throttle_norm": np.full(n, 0.5),
        "midpoint_normal_load_n": np.ones((4, n)) * 8.0,
        "midpoint_slip_x": np.ones((4, n)) * 0.01,
        "midpoint_slip_y": np.ones((4, n)) * -0.02,
        "midpoint_force_summary": np.ones((6, n)),
        "midpoint_acceleration_summary": np.ones((6, n)) * 0.1,
    }
    path = write_midpoint_witness(candidate, tmp_path / "midpoints.csv")
    lines = path.read_text().splitlines()
    assert len(lines) == n + 1
    assert lines[0].startswith("s_ref_m,speed_u_mps,lateral_v_mps")
    assert "slipx_fl" in lines[0] and "slipy_rr" in lines[0]
    assert "witness_throttle_norm" in lines[0]
    assert "force_z_body_n" in lines[0]
    assert "moment_y_body_nm" in lines[0]
    assert "world_vertical_acceleration_mps2" in lines[0]
    assert len(lines[0].split(",")) == len(lines[1].split(","))


def test_practice_scene_tire_overrides_are_applied_to_all_wheels():
    config = _unity_scene_vehicle_parameters(ROOT)
    assert len(config["scene_tire_curve_overrides"]) == 8
    assert config["scene_driving_mode"] == 1
    assert config["scene_hmi_type"] == 0
    assert config["prefab_driving_mode"] == 0
    assert config["drive_type"] == "CAWD"
    assert config["brake_type"] == "CAWB"
    assert config["steer_type"] == "FrontWheelSteer"
    assert config["steering_limit_deg"] == pytest.approx(30.0)
    assert config["steering_rate_degps"] == pytest.approx(183.346)
    assert config["wheelbase_mm"] == pytest.approx(324.0)
    assert config["track_width_mm"] == pytest.approx(236.0)
    assert config["controller_wheel_radius_m_inactive_for_cawd"] == pytest.approx(
        0.0325)
    assert config["rigidbody_drag_per_s"] == pytest.approx(0.273)
    assert config["rigidbody_angular_drag_per_s"] == pytest.approx(0.1)
    assert config["rigidbody_use_gravity"] is True
    assert config["rigidbody_is_kinematic"] is False
    assert config["rigidbody_constraints"] == 0
    for wheel in config["wheel_physics_parameters"]:
        assert wheel["forward_curve"][1] == pytest.approx(0.9)
        assert wheel["forward_curve"][4] == pytest.approx(0.8)


def test_optimizer_step_matches_unity_and_production_fixed_step(candidate):
    optimizer, _, _, _, _, _ = candidate
    source = (ROOT / "f1tenth_mpc/src/unity_source_plant.cpp").read_text()
    time_manager = (ROOT / "external/autodrive-unity-source/ProjectSettings/TimeManager.asset").read_text()
    assert optimizer.UNITY_SOURCE_FIXED_STEP_S == pytest.approx(0.001, abs=1e-15)
    assert _cpp_number(source, "kDt") == pytest.approx(0.001, abs=1e-15)
    assert len(re.findall(
        r"(?m)^\s*Fixed Timestep:\s*([0-9.eE+-]+)\s*$", time_manager)) == 1
    assert float(re.search(
        r"(?m)^\s*Fixed Timestep:\s*([0-9.eE+-]+)\s*$",
        time_manager).group(1)) == pytest.approx(0.001, abs=1e-15)
    assert _cpp_number(source, "kAngularDampingPerS") == pytest.approx(
        optimizer.UNITY_SOURCE_ANGULAR_DAMPING_PER_S, abs=1e-15)


def test_ipopt_wheel_ray_mounts_match_production_geometry(candidate):
    _, wheel, model, *_ = candidate
    source = (ROOT / "f1tenth_mpc/src/unity_source_plant.cpp").read_text()
    wheel_x = _cpp_array(source, "kWheelX", 4)
    wheel_y = _cpp_array(source, "kWheelY", 4)
    transform_z = _cpp_array(source, "kWheelColliderTransformZCom", 4)
    com_forward, com_left, com_up = _cpp_array(
        source, "kComOffsetFromActorBody", 3)
    assert wheel.RIGIDBODY_CENTER_OF_MASS_BODY_M == pytest.approx(
        (-com_left, com_up, com_forward), abs=1e-12)
    offset = _cpp_number(source, "kWheelColliderCenterOffsetM")
    front_cos = _cpp_number(source, "kFrontCamberCos")
    target_offset = _cpp_number(source, "kWheelCentreTargetOffsetM")
    for index, (base_left, base_up, base_forward) in enumerate(
            model.suspension_base_xyz_body_m):
        model_target = model.suspension_target_travel_m[index]
        model_ray = (
            base_forward - com_forward,
            base_left - com_left,
            base_up - model_target + model.wheel_radius_m
            + model_target - com_up)
        source_ray = (
            wheel_x[index],
            wheel_y[index],
            transform_z[index] + offset * (front_cos if index < 2 else 1.0)
            - target_offset + model.wheel_radius_m
            + model_target)
        assert model_ray == pytest.approx(source_ray, abs=1e-12)
    angle = math.atan2(_cpp_number(source, "kFrontCamberSin"), front_cos)
    assert model.suspension_camber_rad == pytest.approx(
        (-angle, angle, 0.0, 0.0), abs=1e-12)


def test_yaw_rate_state_uses_source_angular_acceleration(candidate):
    optimizer, _, model, *_ = candidate
    source = (ROOT / "f1tenth_mpc/src/unity_source_plant.cpp").read_text()
    inverse_inertia = _cpp_array(source, "kInverseInertiaVehicleBodyKgM2", 3)
    assert model.yaw_inertia_kgm2 == pytest.approx(1.0 / inverse_inertia[2],
                                                    rel=1e-12)
    source_text = (Path(optimizer.__file__).read_text(encoding="utf-8"))
    assert "yaw_acceleration = mz / model.yaw_inertia_kgm2" in source_text
    assert "mz / model.yaw_inertia_kgm2" in source_text
    assert "yaw_euler_rate / sdot - curvature" in source_text
    assert "yaw_acceleration / sdot," in source_text


def test_seed_projection_uses_same_full_contact_and_body_equations_as_ocp(candidate):
    optimizer, wheel, model, *_ = candidate
    n = 5
    seed = {
        "ey_m": np.zeros(n),
        "epsi_rad": np.zeros(n),
        "u_mps": np.array([3.0, 3.2, 3.4, 3.1, 2.9]),
        "v_mps": np.array([0.12, -0.08, 0.16, -0.11, 0.04]),
        "yaw_rate_radps": np.array([0.04, -0.03, 0.055, -0.045, 0.02]),
        "wheel_omega_radps": np.tile(
            np.array([3.0, 3.2, 3.4, 3.1, 2.9]) / model.wheel_radius_m,
            (4, 1)),
        "steering_rad": np.array([0.12, -0.10, 0.18, -0.16, 0.06]),
        "steering_rate_radps": np.zeros(n),
        "throttle_norm": np.full(n, 0.12),
        "ax_mps2": np.zeros(n),
        "ay_mps2": np.zeros(n),
        "normal_load_n": np.tile(np.asarray(model.static_normal_load_n)[:, None],
                                 (1, n)),
        "body_height_m": np.full(n, model.suspension_reference_body_height_m),
        "body_height_rate_mps": np.array([0.0, 0.002, -0.001, 0.0015, 0.0]),
        "roll_rad": np.array([0.02, -0.03, 0.04, -0.02, 0.01]),
        "roll_rate_radps": np.array([0.03, -0.02, 0.04, -0.035, 0.01]),
        "pitch_rad": np.array([-0.02, 0.03, -0.01, 0.025, 0.0]),
        "pitch_rate_radps": np.array([-0.015, 0.02, -0.01, 0.012, 0.0]),
    }
    projected, audit = project_algebraics(
        seed, model, wheel,
        suspension_loads_fn=optimizer._unity_suspension_loads_numpy,
        body_acceleration_fn=optimizer._unity_body_acceleration_numpy)
    assert audit["tire_contact_state_matches_ocp_full_18_state"] is True
    assert audit["suspension_uses_body_to_euler_rate_conversion"] is True
    assert audit["scalar_vs_vectorized_tire_force_max_error_n"] < 1.0e-8
    for k in range(n):
        expected_load = optimizer._unity_suspension_loads_numpy(
            model, projected["body_height_m"][k],
            projected["body_height_rate_mps"][k],
            projected["roll_rad"][k], projected["roll_rate_radps"][k],
            projected["pitch_rad"][k], projected["pitch_rate_radps"][k],
            projected["yaw_rate_radps"][k])
        np.testing.assert_allclose(projected["normal_load_n"][:, k],
                                   expected_load, atol=1.0e-12)
        forces = wheel.tire_forces_casadi(
            model, projected["u_mps"][k], projected["v_mps"][k],
            projected["yaw_rate_radps"][k], projected["steering_rad"][k],
            projected["wheel_omega_radps"][:, k],
            projected["normal_load_n"][:, k],
            projected["body_height_m"][k],
            projected["body_height_rate_mps"][k],
            projected["roll_rad"][k], projected["roll_rate_radps"][k],
            projected["pitch_rad"][k], projected["pitch_rate_radps"][k])
        expected_accel = optimizer._unity_body_acceleration_numpy(
            model, float(forces[0]), float(forces[1]),
            projected["u_mps"][k], projected["v_mps"][k],
            projected["yaw_rate_radps"][k],
            projected["body_height_rate_mps"][k],
            projected["roll_rad"][k], projected["roll_rate_radps"][k],
            projected["pitch_rad"][k], projected["pitch_rate_radps"][k])
        np.testing.assert_allclose(
            [projected["ax_mps2"][k], projected["ay_mps2"][k]],
            expected_accel, atol=1.0e-12)


def test_tire_load_filter_matches_physx_piecewise_map(candidate):
    _, wheel, model, *_ = candidate
    raw_load = ca.MX.sym("raw_suspension_force")
    expression = wheel._physx_filtered_tire_load_casadi(
        raw_load, model.sprung_mass_kg[0], wheel.WHEEL_MASS_KG,
        wheel.UNITY_PHYSICS_GRAVITY_MPS2)
    evaluate = ca.Function("v251_load_filter_test", [raw_load], [expression])
    rest_load = ((model.sprung_mass_kg[0] + wheel.WHEEL_MASS_KG)
                 * wheel.UNITY_PHYSICS_GRAVITY_MPS2)
    for normalized in (-0.2, 0.0, 0.5, 3.0, 4.0):
        expected_fraction = min(
            3.0, max(0.2308, 0.2308 + normalized * (3.0 - 0.2308) / 3.0))
        assert float(evaluate(normalized * rest_load)) == pytest.approx(
            expected_fraction * rest_load, rel=1e-12, abs=1e-12)


def test_full_wrench_resists_leftward_slip_in_source_frame(candidate):
    _, wheel, model, *_ = candidate
    outputs = wheel.tire_forces_casadi(
        model, 5.0, 1.0, 0.0, 0.0,
        [5.0 / model.wheel_radius_m] * 4, [100.0] * 4,
        model.suspension_reference_body_height_m, 0.0,
        0.0, 0.0, 0.0, 0.0)
    evaluate = ca.Function("v251_flat_wrench_test", [],
                           [ca.vertcat(*outputs[:6])])
    wrench = np.asarray(evaluate.call([])[0], dtype=float).reshape(-1)
    assert wrench.shape == (6,)
    assert np.isfinite(wrench).all()
    assert wrench[1] < 0.0  # positive leftward contact speed -> rightward tire force
    assert wrench[2] > 0.0  # positive spring/contact load points along ground normal


def test_body_rates_and_horizontal_velocity_match_zyx_source_equations(candidate):
    optimizer = candidate[0]
    x = ca.MX.sym("body_state", 6)
    roll, pitch, p, q, r, height_rate = [x[i] for i in range(6)]
    rates = optimizer._unity_euler_rates_casadi(roll, pitch, p, q, r)
    body_vertical = (height_rate + ca.sin(pitch) * 4.2
                     - ca.cos(pitch) * ca.sin(roll) * 0.35) / (
                         ca.cos(pitch) * ca.cos(roll))
    horizontal = optimizer._unity_horizontal_velocity_casadi(
        4.2, 0.35, body_vertical, roll, pitch)
    evaluate = ca.Function("v251_attitude_equations", [x],
                           [ca.vertcat(*rates, *horizontal)])
    values = np.asarray(evaluate([0.22, -0.17, 0.31, -0.28, 0.74, 0.08]),
                        dtype=float).reshape(-1)
    phi, theta, p_value, q_value, r_value, hdot = (
        0.22, -0.17, 0.31, -0.28, 0.74, 0.08)
    sr, cr, sp, cp = math.sin(phi), math.cos(phi), math.sin(theta), math.cos(theta)
    expected_rates = (
        p_value + math.tan(theta) * (sr * q_value + cr * r_value),
        cr * q_value - sr * r_value,
        (sr * q_value + cr * r_value) / cp,
    )
    w = (hdot + sp * 4.2 - cp * sr * 0.35) / (cp * cr)
    expected_horizontal = (
        cp * 4.2 + sp * (sr * 0.35 + cr * w),
        cr * 0.35 - sr * w,
    )
    np.testing.assert_allclose(values, (*expected_rates, *expected_horizontal),
                               rtol=1e-12, atol=1e-12)
