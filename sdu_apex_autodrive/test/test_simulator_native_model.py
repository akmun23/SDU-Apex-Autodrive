import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

# Keep direct source-checkout test invocation independent of whether the
# repository root or only the ROS package directory is on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_id.simulator_native_model import (
    GUIDE_LATERAL_CURVE,
    NativeModelParameters,
    UnityWheelDynamicsParameters,
    ackermann_angles,
    body_axis_inertia,
    body_frame_inertias,
    contact_kinematics,
    f1tenth_prefab_parameters,
    _moment_basis_coordinates,
    parameters_from_dump,
    position_point_body_velocity,
    _split_moment_basis_coordinates,
    step,
    static_normal_loads,
    step_unity_wheel_collider,
    step_unity_wheel_collider_with_suspension,
    unity_suspension_loads,
    unity_wheel_travel_loads,
    unity_contact_kinematics,
    unity_normal_loads,
    unity_normal_loads_from_acceleration,
    _unity_wheel_torques,
    _unity_vehicle_controller_source_step,
    _unity_vehicle_controller_steering_segments,
    _unity_wheel_friction_value,
    unity_wheel_collider_ackermann_angles,
)
from tools.model_id.analyze_simulator_diagnostics import analyze as analyze_diagnostics
from tools.model_id.wheel_friction_curve import friction_derivative, friction_value


def test_documented_curve_is_odd_bounded_and_c1_at_breakpoints():
    curve = GUIDE_LATERAL_CURVE
    assert friction_value(curve.extremum_slip, curve) == pytest.approx(1.0)
    assert friction_value(curve.asymptote_slip, curve) == pytest.approx(0.5)
    assert friction_value(-0.03, curve) == pytest.approx(-friction_value(0.03, curve))
    assert friction_derivative(curve.extremum_slip, curve) == pytest.approx(0.0, abs=1e-8)
    assert friction_derivative(curve.asymptote_slip, curve) == pytest.approx(0.0, abs=1e-8)
    assert friction_value(100.0, curve) == pytest.approx(0.5)


def test_ackermann_has_inner_wheel_larger_for_positive_turn():
    left, right, rear_left, rear_right = ackermann_angles(0.2, 0.324, 0.236)
    assert left > right > 0.0
    assert rear_left == pytest.approx(0.0)
    assert rear_right == pytest.approx(0.0)


def test_prefab_profile_preserves_contact_and_command_geometry():
    parameters = f1tenth_prefab_parameters()
    assert parameters.mass_kg == pytest.approx(3.47)
    assert parameters.wheel_mass_total_kg == pytest.approx(0.436)
    assert parameters.additive_system_mass_kg == pytest.approx(3.906)
    assert parameters.wheel_radius_m == pytest.approx(0.059)
    assert parameters.controller_wheel_radius_m == pytest.approx(0.0325)
    assert parameters.drive_type == "CAWD"
    assert parameters.wheelbase_m == pytest.approx(0.330)
    assert parameters.steering_geometry_wheelbase_m == pytest.approx(0.324)
    assert parameters.track_m == pytest.approx(0.236)
    assert parameters.position_offset_from_velocity_point_x_m == pytest.approx(
        -0.15532)
    assert parameters.longitudinal_curve.source == "unity_prefab:F1TENTH.prefab"
    assert parameters.lateral_curve.source == "unity_prefab:F1TENTH.prefab"
    assert parameters.steering_rate_radps == pytest.approx(
        np.deg2rad(183.346))


def test_default_candidate_uses_measured_unity_structural_anchors():
    parameters = NativeModelParameters()
    assert parameters.mass_kg == pytest.approx(3.47)
    assert parameters.com_x_from_rear_axle_m == pytest.approx(0.155320086)
    assert parameters.wheelbase_m == pytest.approx(0.33000004)
    assert parameters.lf_m == pytest.approx(0.174679914)
    assert parameters.lr_m == pytest.approx(0.155320086)
    assert parameters.steering_geometry_wheelbase_m == pytest.approx(0.324)


def test_prefab_profile_uses_controller_wheelbase_for_ackermann_angles():
    parameters = f1tenth_prefab_parameters()
    command = 0.2
    contacts = contact_kinematics(
        np.asarray([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0]),
        command, parameters)
    expected = ackermann_angles(command, 0.324, 0.236)
    assert [contact["steering_rad"] for contact in contacts] == pytest.approx(
        expected)


def test_raw_unity_wheel_angle_ordering_is_kept_separate_from_api_frame():
    left, right, rear_left, rear_right = unity_wheel_collider_ackermann_angles(
        0.2, 0.324, 0.236)
    assert right > left > 0.0
    assert rear_left == pytest.approx(0.0)
    assert rear_right == pytest.approx(0.0)


def test_prefab_keeps_api_frame_steering_sign_explicit():
    parameters = f1tenth_prefab_parameters()
    assert parameters.steering_to_wheel_angle_sign == pytest.approx(1.0)


def test_exact_dump_profile_maps_applied_steering_to_model_frame():
    parameters = parameters_from_dump(Path(
        "sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/"
        "unity_exact_open_raceline_relevant_holdout_v1_20260916/"
        "simulator_parameters.json"))
    # Unity writes the opposite sign internally, and the lateral frame
    # conversion reverses it once more.  The effective model angle is +1.
    assert parameters.steering_to_wheel_angle_sign == pytest.approx(1.0)


def test_contact_slips_use_four_wheel_geometry_and_dimensionless_ratio():
    parameters = NativeModelParameters(yaw_inertia_kgm2=0.04)
    state = np.asarray([0.0, 0.0, 0.0, 2.0, 0.1, 0.2, 0.0, 2.0])
    contacts = contact_kinematics(state, 0.1, parameters)
    assert len(contacts) == 4
    assert contacts[0]["x_m"] == pytest.approx(parameters.lf_m)
    assert contacts[2]["x_m"] == pytest.approx(-parameters.lr_m)
    assert contacts[0]["y_m"] == pytest.approx(-contacts[1]["y_m"])
    assert all(np.isfinite([contact["sx"], contact["sy"]]).all() for contact in contacts)


def test_wheel_state_is_already_surface_speed_not_angular_rate():
    parameters = NativeModelParameters(yaw_inertia_kgm2=0.04)
    state = np.asarray([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])
    contacts = contact_kinematics(state, 0.0, parameters)
    assert [contact["sx"] for contact in contacts] == pytest.approx([0.0] * 4)


def test_four_wheel_contacts_use_causal_left_and_right_wheel_states():
    parameters = NativeModelParameters(yaw_inertia_kgm2=0.04)
    state = np.asarray([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.2, 1.8])
    contacts = contact_kinematics(state, 0.0, parameters)
    assert contacts[0]["wheel_surface_speed_mps"] == pytest.approx(2.2)
    assert contacts[1]["wheel_surface_speed_mps"] == pytest.approx(1.8)
    assert contacts[2]["wheel_surface_speed_mps"] == pytest.approx(2.2)
    assert contacts[3]["wheel_surface_speed_mps"] == pytest.approx(1.8)


def test_unity_wheel_contacts_apply_radius_once_to_angular_state():
    parameters = replace(
        NativeModelParameters(yaw_inertia_kgm2=0.096),
        wheel_radii_m=(0.059,) * 4,
        unity_wheel_dynamics=UnityWheelDynamicsParameters())
    state = np.asarray([
        0.0, 0.0, 0.0, 5.9, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0,
        100.0,
    ])
    contacts = unity_contact_kinematics(state, 0.0, parameters)
    assert [contact["wheel_surface_speed_mps"] for contact in contacts] == \
        pytest.approx([5.9] * 4)
    assert [contact["sx"] for contact in contacts] == pytest.approx([0.0] * 4)


def test_unity_friction_curve_is_piecewise_linear_at_serialized_knots():
    curve = GUIDE_LATERAL_CURVE
    assert _unity_wheel_friction_value(0.0, curve) == pytest.approx(0.0)
    assert _unity_wheel_friction_value(
        curve.extremum_slip, curve) == pytest.approx(curve.extremum_value)
    assert _unity_wheel_friction_value(
        curve.asymptote_slip, curve) == pytest.approx(curve.asymptote_value)
    midpoint = (curve.extremum_slip + curve.asymptote_slip) / 2.0
    assert _unity_wheel_friction_value(midpoint, curve) == pytest.approx(0.75)
    assert _unity_wheel_friction_value(2.0, curve) == pytest.approx(
        curve.asymptote_value)
    assert _unity_wheel_friction_value(-midpoint, curve) == pytest.approx(-0.75)


def test_unity_contact_slip_uses_source_denominators_and_signs():
    parameters = replace(
        NativeModelParameters(yaw_inertia_kgm2=0.096),
        wheel_radii_m=(0.059,) * 4)
    state = np.asarray([
        0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0,
        2.0 / 0.059, 2.0 / 0.059, 2.0 / 0.059, 2.0 / 0.059,
    ])
    contacts = unity_contact_kinematics(state, 0.0, parameters)
    # With zero steering, surface speed is 2 m/s and ground-forward speed is
    # 4 m/s: the exact forward denominator is 4 m/s, not the generic floor.
    assert [contact["sx"] for contact in contacts] == pytest.approx([-0.5] * 4)
    # Source WheelHit.sidewaysSlip has the negative wheel-frame lateral sign;
    # at 4 m/s its denominator is the wheel-forward velocity.
    assert [contact["sy"] for contact in contacts] == pytest.approx([-0.25] * 4)
    low_speed_state = state.copy()
    low_speed_state[3] = 0.2
    low_speed_state[7:] = 0.2 / 0.059
    low_speed_contacts = unity_contact_kinematics(
        low_speed_state, 0.0, parameters)
    # Below 0.5 m/s the source floor is active: -1/0.5 = -2.
    assert [contact["sy"] for contact in low_speed_contacts] == pytest.approx(
        [-2.0] * 4)


def test_unity_active_cawd_cawb_torque_branch_is_explicit():
    dynamics = UnityWheelDynamicsParameters()
    parameters = replace(
        NativeModelParameters(), unity_wheel_dynamics=dynamics)
    motor, brake = _unity_wheel_torques(1.0, parameters)
    assert motor == pytest.approx((107.0,) * 4)
    assert brake == pytest.approx((0.0,) * 4)
    motor, brake = _unity_wheel_torques(0.0, parameters)
    assert motor == pytest.approx((0.0,) * 4)
    assert brake == pytest.approx((428.0,) * 4)


def test_unity_native_steering_replays_source_cadence_and_clamp():
    parameters = replace(
        NativeModelParameters(), steering_source_fixed_dt_s=0.001)
    current = 0.1 * parameters.steering_limit_rad
    segments = _unity_vehicle_controller_steering_segments(
        current, -0.1, 0.025, parameters)
    assert len(segments) == 25
    assert segments[-1][2] == pytest.approx(-0.02764, abs=1e-12)
    # A source update from the exact effective state reverses only one 1 ms
    # rate step at a time; the source clamp is not a 25 ms MoveTowards.
    assert _unity_vehicle_controller_source_step(
        current, -0.1, 0.001, parameters) > 0.0


def test_unity_wheel_plant_uses_dumped_yaw_inertia_and_is_finite():
    parameters = replace(
        NativeModelParameters(yaw_inertia_kgm2=0.096),
        wheel_radii_m=(0.059,) * 4,
        wheel_sprung_masses_kg=(0.817, 0.816, 0.919, 0.918),
        unity_wheel_dynamics=UnityWheelDynamicsParameters())
    state = np.asarray([
        0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 84.0, 84.0, 84.0, 84.0,
    ])
    assert sum(unity_normal_loads(parameters)) == pytest.approx(
        parameters.mass_kg * 9.81, rel=2.0e-3)
    next_state = step_unity_wheel_collider(
        state, 0.15, 0.5, 0.025, parameters)
    assert next_state.shape == (11,)
    assert np.isfinite(next_state).all()
    assert next_state[7] != pytest.approx(state[7])


def test_unity_cawb_brake_holds_a_locked_wheel_without_sign_flip():
    parameters = replace(
        NativeModelParameters(yaw_inertia_kgm2=0.096),
        wheel_radii_m=(0.059,) * 4,
        unity_wheel_dynamics=UnityWheelDynamicsParameters())
    state = np.asarray([
        0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    next_state = step_unity_wheel_collider(
        state, 0.0, 0.0, 0.025, parameters)
    assert next_state[7:] == pytest.approx((0.0,) * 4, abs=1.0e-8)


def test_mechanical_longitudinal_load_screen_is_explicit_and_conservative():
    parameters = replace(
        NativeModelParameters(
            wheel_sprung_masses_kg=(0.817, 0.816, 0.919, 0.918),
            unity_com_height_m=0.06434),
        unity_normal_load_mode="mechanical_longitudinal_cg_transfer")
    base = np.asarray(unity_normal_loads(parameters), dtype=float)
    loads = np.asarray(unity_normal_loads_from_acceleration(
        parameters, longitudinal_accel_mps2=-5.0, lateral_accel_mps2=4.0))
    assert loads[0] > base[0]
    assert loads[1] > base[1]
    assert loads[2] < base[2]
    assert loads[3] < base[3]
    assert loads[0] - loads[1] == pytest.approx(base[0] - base[1])
    assert loads[2] - loads[3] == pytest.approx(base[2] - base[3])
    assert loads.sum() == pytest.approx(base.sum())


def test_explicit_suspension_loads_use_serialized_spring_and_damper():
    parameters = parameters_from_dump(Path(
        "sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/"
        "unity_exact_open_raceline_relevant_holdout_v1_20260916/"
        "simulator_parameters.json"))
    base = np.asarray(unity_normal_loads(parameters), dtype=float)
    heave = np.asarray(unity_suspension_loads(
        parameters, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert heave == pytest.approx(base - 0.5)
    assert heave.sum() < base.sum()


def test_wheel_travel_load_law_uses_serialized_distance_without_hidden_gain():
    parameters = parameters_from_dump(Path(
        "sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/"
        "unity_exact_open_raceline_relevant_holdout_v1_20260916/"
        "simulator_parameters.json"))
    loads = np.asarray(unity_wheel_travel_loads(
        parameters, (0.1, 0.1, -0.1, -0.1), (0.0,) * 4))
    base = np.asarray(unity_normal_loads(parameters))
    expected_delta = np.asarray((2.5, 2.5, -2.5, -2.5))
    assert loads == pytest.approx(base - expected_delta)


def test_explicit_suspension_plant_is_finite_and_keeps_state_separate():
    parameters = parameters_from_dump(Path(
        "sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/"
        "unity_exact_open_raceline_relevant_holdout_v1_20260916/"
        "simulator_parameters.json"))
    state = np.asarray([
        0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0,
        84.0, 84.0, 84.0, 84.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    next_state = step_unity_wheel_collider_with_suspension(
        state, 0.15, 0.5, 0.025, parameters)
    assert next_state.shape == (17,)
    assert np.isfinite(next_state).all()
    assert next_state[11:] != pytest.approx(state[11:])


def test_moment_basis_rotates_all_four_contacts_and_builds_yaw_moment():
    parameters = NativeModelParameters(contact_model="moment_basis")
    state = np.asarray([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0])
    qx, qy, qmoment = _moment_basis_coordinates(state, 0.2, parameters)
    assert np.isfinite((qx, qy, qmoment)).all()
    assert qx > 0.0
    assert qy > 0.0
    assert qmoment > 0.0


def test_moment_basis_preserves_left_right_wheel_state_effect():
    parameters = NativeModelParameters(contact_model="moment_basis")
    equal = np.asarray([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0])
    split = equal.copy()
    split[7:] = (3.2, 2.8)
    assert _moment_basis_coordinates(equal, 0.2, parameters) != pytest.approx(
        _moment_basis_coordinates(split, 0.2, parameters))


def test_split_moment_basis_retains_front_rear_and_longitudinal_moments():
    parameters = NativeModelParameters(contact_model="moment_basis_split")
    state = np.asarray([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0])
    coordinates = _split_moment_basis_coordinates(state, 0.2, parameters)
    assert len(coordinates) == 6
    assert np.isfinite(coordinates).all()
    assert coordinates[1] != pytest.approx(coordinates[2])


def test_moment_basis_step_uses_effective_force_and_moment_gains():
    parameters = NativeModelParameters(
        contact_model="moment_basis",
        effective_force_x_gain_mps2=3.0,
        effective_drag_linear_per_s=0.2,
        effective_drag_quadratic_per_m=0.0,
        effective_force_y_gain_mps2=4.0,
        effective_yaw_moment_gain_per_s2=2.0)
    state = np.asarray([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0])
    next_state = step(state, 0.3, 0.5, 0.025, parameters)
    assert next_state.shape == (9,)
    assert np.isfinite(next_state).all()


def test_split_moment_basis_step_uses_all_split_gains():
    parameters = NativeModelParameters(
        contact_model="moment_basis_split",
        effective_force_x_gain_mps2=3.0,
        effective_drag_linear_per_s=0.2,
        effective_drag_quadratic_per_m=0.0,
        effective_force_y_front_gain_mps2=4.0,
        effective_force_y_rear_gain_mps2=3.0,
        effective_yaw_lateral_front_gain_per_s2=2.0,
        effective_yaw_lateral_rear_gain_per_s2=1.5,
        effective_yaw_longitudinal_gain_per_s2=0.5)
    state = np.asarray([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 3.0])
    next_state = step(state, 0.3, 0.5, 0.025, parameters)
    assert next_state.shape == (9,)
    assert np.isfinite(next_state).all()


def test_load_transfer_screen_is_opt_in_and_changes_turn_basis():
    state = np.asarray([0.0, 0.0, 0.0, 6.0, 0.0, 0.5, 0.0, 6.0, 6.0])
    baseline = NativeModelParameters(contact_model="moment_basis")
    load_transfer = replace(
        baseline, effective_load_transfer_height_m=0.08)
    base_coordinates = _moment_basis_coordinates(state, 0.2, baseline)
    transferred_coordinates = _moment_basis_coordinates(
        state, 0.2, load_transfer)
    assert transferred_coordinates != pytest.approx(base_coordinates)
    assert NativeModelParameters(
        contact_model="moment_basis").effective_load_transfer_height_m == 0.0


def test_pose_velocity_uses_recorded_rear_reference_point():
    parameters = NativeModelParameters(
        yaw_inertia_kgm2=0.04,
        position_offset_from_velocity_point_x_m=-0.15532)
    state = np.asarray([0.0, 0.0, 0.0, 2.0, 0.4, 0.5, 0.0, 2.0])
    pose_u, pose_v = position_point_body_velocity(state, parameters)
    assert pose_u == pytest.approx(2.0)
    assert pose_v == pytest.approx(
        0.4 - parameters.com_x_from_rear_axle_m * 0.5)


def test_static_loads_sum_to_model_weight():
    parameters = NativeModelParameters(yaw_inertia_kgm2=0.04)
    loads = static_normal_loads(parameters)
    assert sum(loads) == pytest.approx(parameters.mass_kg * 9.81)
    assert loads[0] == pytest.approx(loads[1])
    assert loads[2] == pytest.approx(loads[3])


def test_physical_candidate_requires_dumped_yaw_inertia():
    state = np.zeros(8, dtype=float)
    with pytest.raises(ValueError, match="dumped yaw inertia"):
        step(state, 0.0, 0.0, 0.025, NativeModelParameters())


def test_parameters_from_dump_preserves_dump_provenance(tmp_path: Path):
    payload = {
        "diagnosticOnly": True,
        "runtimeControlInput": False,
        "simulatorBuildTag": "test-build",
        "unityVersion": "test-unity",
        "fixedDeltaTime": 0.02,
        "maximumAllowedTimestep": 0.1,
        "timeScale": 1.0,
        "gravity": {"x": 0.0, "y": -9.81, "z": 0.0},
        "defaultSolverIterations": 6,
        "defaultSolverVelocityIterations": 1,
        "defaultContactOffset": 0.01,
        "defaultMaxDepenetrationVelocity": 10.0,
        "defaultMaxAngularSpeed": 50.0,
        "vSyncCount": 0,
        "targetFrameRate": -1,
        "vehicle": {
            "wheelbaseM": 0.324,
            "trackWidthM": 0.236,
            "wheelRadiusControllerM": 0.059,
            "throttleLimit": 1.0,
            "motorTorqueNm": 428.0,
            "driveType": "CAWD",
            "brakeType": "CAWB",
            "steeringLimitRad": 0.5236,
            "steeringRateRadPerSecond": 3.2,
            "rigidBody": {
                "mass": 3.47,
                "centerOfMass": {"x": 0.0, "y": 0.06434, "z": 0.0},
                "bodyInertiaX": 0.01,
                "bodyInertiaY": 0.02,
                "bodyInertiaZ": 0.04,
                "yawAxis": "body_y",
                "yawInertiaBodyFrame": 0.02,
                "inertiaTensor": {"x": 0.01, "y": 0.02, "z": 0.04},
                "inertiaTensorRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "drag": 0.27,
                "angularDrag": 0.10,
                "maxAngularVelocity": 7.0,
            },
            "wheels": [],
        },
    }
    wheel = {
        "name": "wheel",
        "mass": 0.109,
        "radius": 0.059,
        "sprungMass": 0.8675,
        "suspensionDistance": 0.03,
        "wheelDampingRate": 0.25,
        "forceAppPointDistance": 0.0,
        "localPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
        "localRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "suspensionSpring": {
            "spring": 100.0, "damper": 1.0, "targetPosition": 0.5,
        },
        "positionVehicleFrame": {"x": 0.118, "y": 0.06, "z": 0.17},
        "forwardFriction": {
            "extremumSlip": 0.15, "extremumValue": 0.72,
            "asymptoteSlip": 0.25, "asymptoteValue": 0.464, "stiffness": 0.8,
        },
        "sidewaysFriction": {
            "extremumSlip": 0.01, "extremumValue": 1.0,
            "asymptoteSlip": 0.1, "asymptoteValue": 0.5, "stiffness": 1.0,
        },
    }
    payload["vehicle"]["wheels"] = [
        dict(wheel, positionVehicleFrame={"x": -0.118, "y": 0.06, "z": 0.16868}),
        dict(wheel, positionVehicleFrame={"x": 0.118, "y": 0.06, "z": 0.16868}),
        dict(wheel, positionVehicleFrame={"x": -0.118, "y": 0.05, "z": -0.15532}),
        dict(wheel, positionVehicleFrame={"x": 0.118, "y": 0.05, "z": -0.15532}),
    ]
    path = tmp_path / "simulator_parameters.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    parameters = parameters_from_dump(path)
    assert parameters.mass_kg == pytest.approx(3.47)
    assert parameters.wheel_radius_m == pytest.approx(0.059)
    assert parameters.position_offset_from_velocity_point_x_m == pytest.approx(-0.15532)
    assert parameters.parameter_provenance.endswith(str(path))
    assert parameters.lateral_curve.source == "unity_diagnostic_dump"
    assert parameters.wheel_lateral_curves is not None
    assert len(parameters.wheel_lateral_curves) == 4
    assert parameters.wheel_radii_m == pytest.approx((0.059,) * 4)
    assert parameters.wheel_masses_kg == pytest.approx((0.109,) * 4)
    assert parameters.wheel_sprung_masses_kg == pytest.approx((0.8675,) * 4)
    assert parameters.suspension_distances_m == pytest.approx((0.03,) * 4)
    assert parameters.suspension_spring_rates_n_per_m == pytest.approx((100.0,) * 4)
    assert parameters.suspension_damper_rates_ns_per_m == pytest.approx((1.0,) * 4)
    assert parameters.suspension_target_positions == pytest.approx((0.5,) * 4)
    assert parameters.wheel_damping_rates_per_s == pytest.approx((0.25,) * 4)
    assert parameters.rigid_body_drag_per_s == pytest.approx(0.27)
    assert parameters.rigid_body_angular_drag_per_s == pytest.approx(0.10)
    assert parameters.rigid_body_max_angular_velocity_radps == pytest.approx(7.0)
    assert parameters.unity_wheel_dynamics is not None
    assert parameters.unity_wheel_dynamics.motor_torque_nm == pytest.approx(428.0)
    assert parameters.unity_wheel_dynamics.drive_torque_fractions == pytest.approx(
        (0.25,) * 4)
    assert parameters.unity_wheel_dynamics.brake_torque_fractions == pytest.approx(
        (1.0,) * 4)
    diagnostic_report = analyze_diagnostics(path)
    assert diagnostic_report["status"] == "diagnostic_dump_analyzed"
    assert len(diagnostic_report["friction_curve_guide_crosscheck"]) == 4
    assert diagnostic_report["derived"]["derived_contact_geometry"][
        "wheelbase_m"] == pytest.approx(0.324)
    assert diagnostic_report["derived"]["rigid_body_damping"][
        "angular_drag"] == pytest.approx(0.10)
    assert diagnostic_report["derived"]["diagnostic_field_completeness"][
        "required_dynamic_fields_present"] is True
    assert diagnostic_report["derived"]["rigid_body_damping"][
        "max_angular_velocity_radps"] == pytest.approx(7.0)
    assert diagnostic_report["derived"]["controller_parameters"][
        "physical_wheel_radius_m"] == pytest.approx(0.059)
    assert diagnostic_report["derived"]["controller_parameters"][
        "controller_wheel_radius_role"].startswith(
            "VehicleController.WheelRadius")
    assert diagnostic_report["derived"]["mass_accounting"][
        "status"] == "explicit_body_and_wheel_accounting"
    assert diagnostic_report["derived"]["wheel_dynamics"][0][
        "sprung_mass_kg"] == pytest.approx(0.8675)
    assert diagnostic_report["derived"]["wheel_dynamics"][0][
        "suspension_spring"]["damper_ns_per_m"] == pytest.approx(1.0)


def test_body_inertia_axes_are_projected_explicitly():
    # A 90-degree rotation about Unity's y axis maps principal x to body -z
    # and principal z to body +x.  Physical yaw remains Unity body y.
    half_sqrt_two = 2.0 ** -0.5
    moments = {"x": 0.01, "y": 0.02, "z": 0.04}
    rotation = {
        "x": 0.0, "y": half_sqrt_two,
        "z": 0.0, "w": half_sqrt_two,
    }
    assert body_axis_inertia(moments, rotation, "x") == pytest.approx(
        0.04, abs=1.0e-10)
    assert body_axis_inertia(moments, rotation, "y") == pytest.approx(
        0.02, abs=1.0e-10)
    assert body_axis_inertia(moments, rotation, "z") == pytest.approx(
        0.01, abs=1.0e-10)
    assert body_frame_inertias(moments, rotation) == pytest.approx({
        "x": 0.04, "y": 0.02, "z": 0.01}, abs=1.0e-10)


def test_body_inertia_accepts_a_normalized_arbitrary_body_axis():
    half_sqrt_two = 2.0 ** -0.5
    inertia = body_axis_inertia(
        {"x": 0.01, "y": 0.02, "z": 0.04},
        {"x": 0.0, "y": half_sqrt_two,
         "z": 0.0, "w": half_sqrt_two}, (0.0, 2.0, 0.0))
    assert inertia == pytest.approx(0.02, abs=1.0e-10)


def test_parameters_from_dump_rejects_legacy_implicit_yaw_axis(tmp_path: Path):
    payload = {
        "diagnosticOnly": True,
        "runtimeControlInput": False,
        "vehicle": {
            "rigidBody": {
                "mass": 3.47,
                "inertiaTensor": {"x": 0.01, "y": 0.02, "z": 0.04},
                "inertiaTensorRotation": {
                    "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "yawInertiaBodyFrame": 0.04,
            },
            "wheels": [],
        },
    }
    path = tmp_path / "legacy_simulator_parameters.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="body Y as yawAxis"):
        parameters_from_dump(path)
