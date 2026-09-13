import json
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
    parameters_from_dump,
    position_point_body_velocity,
    step,
    static_normal_loads,
)
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
        "vehicle": {
            "steeringLimitRad": 0.5236,
            "steeringRateRadPerSecond": 3.2,
            "rigidBody": {
                "mass": 3.47,
                "centerOfMass": {"x": 0.0, "y": 0.06434, "z": 0.0},
                "yawInertiaBodyFrame": 0.04,
                "inertiaTensor": {"x": 0.01, "y": 0.02, "z": 0.04},
                "inertiaTensorRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "wheels": [],
        },
    }
    wheel = {
        "mass": 0.109,
        "radius": 0.059,
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
    assert parameters.mass_kg == pytest.approx(3.906)
    assert parameters.wheel_radius_m == pytest.approx(0.059)
    assert parameters.position_offset_from_velocity_point_x_m == pytest.approx(-0.15532)
    assert parameters.parameter_provenance.endswith(str(path))
    assert parameters.lateral_curve.source == "unity_diagnostic_dump"
    assert parameters.wheel_lateral_curves is not None
    assert len(parameters.wheel_lateral_curves) == 4


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
