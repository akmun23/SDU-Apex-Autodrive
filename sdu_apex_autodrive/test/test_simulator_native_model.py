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
    ackermann_angles,
    body_frame_yaw_inertia,
    contact_kinematics,
    f1tenth_prefab_parameters,
    _moment_basis_coordinates,
    parameters_from_dump,
    position_point_body_velocity,
    _split_moment_basis_coordinates,
    step,
    static_normal_loads,
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
    assert parameters.wheelbase_m == pytest.approx(0.330)
    assert parameters.steering_geometry_wheelbase_m == pytest.approx(0.324)
    assert parameters.track_m == pytest.approx(0.236)
    assert parameters.position_offset_from_velocity_point_x_m == pytest.approx(
        -0.15532)
    assert parameters.longitudinal_curve.source == "unity_prefab:F1TENTH.prefab"
    assert parameters.lateral_curve.source == "unity_prefab:F1TENTH.prefab"
    assert parameters.steering_rate_radps == pytest.approx(
        np.deg2rad(183.346))


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
            "steeringLimitRad": 0.5236,
            "steeringRateRadPerSecond": 3.2,
            "rigidBody": {
                "mass": 3.47,
                "centerOfMass": {"x": 0.0, "y": 0.06434, "z": 0.0},
                "yawInertiaBodyFrame": 0.04,
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
    assert diagnostic_report["derived"]["wheel_dynamics"][0][
        "sprung_mass_kg"] == pytest.approx(0.8675)
    assert diagnostic_report["derived"]["wheel_dynamics"][0][
        "suspension_spring"]["damper_ns_per_m"] == pytest.approx(1.0)


def test_yaw_inertia_is_projected_from_rotated_principal_tensor():
    # A 90-degree rotation about Unity's y axis maps the principal x axis to
    # the body z axis.  The body-frame yaw inertia is therefore I_x, not the
    # scalar stored in any convenience field in a diagnostic dump.
    half_sqrt_two = 2.0 ** -0.5
    inertia = body_frame_yaw_inertia(
        {"x": 0.01, "y": 0.02, "z": 0.04},
        {"x": 0.0, "y": half_sqrt_two,
         "z": 0.0, "w": half_sqrt_two})
    assert inertia == pytest.approx(0.01, abs=1.0e-10)
