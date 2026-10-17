import math

import casadi as ca
import pytest

from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.unity_wheel_model import (
    UnityWheelModel,
    _unity_wheel_inertia_kgm2,
    _physx_compute_sprung_masses,
    _unity_friction_curve_response,
    _unity_tire_force_shader_casadi,
    _physx_applied_torque_slips_casadi,
    _physx_filtered_tire_load_casadi,
    _wheel_surface_speed_from_autodrive_guide_slip,
    _wheel_surface_speed_from_physx_long_slip,
    suspension_loads_casadi,
    suspension_travel_casadi,
)


FORWARD = (0.15, 0.9, 0.25, 0.58, 0.8)
LATERAL = (0.01, 1.0, 0.1, 0.5, 1.0)


def test_wheel_inertia_uses_native_wrapper_formula_and_prefab_values():
    # Native wrapper multiplies 0.5 * WheelCollider.mass * effectiveRadius^2.
    assert _unity_wheel_inertia_kgm2(0.109, 0.059) == pytest.approx(
        0.0001897145)


def test_sprung_masses_match_physx_four_wheel_setup_solve():
    wheel_centres = (
        (-0.11582111, 0.08490487, 0.17),
        (0.11582111, 0.08490487, 0.17),
        (-0.118, 0.075, -0.16),
        (0.118, 0.075, -0.16),
    )
    masses = _physx_compute_sprung_masses(
        wheel_centres, (-0.00008, 0.06434, -0.00468), 3.47)

    assert masses == pytest.approx(
        (0.8171947, 0.8160186, 0.9189924, 0.9177942), abs=1e-6)
    assert sum(masses) == pytest.approx(3.47)


def test_optimizer_model_uses_source_values_not_fit_artifacts():
    model = UnityWheelModel.from_source_capture()

    assert model.wheel_inertia_kgm2 == pytest.approx(0.0001897145)
    assert model.sprung_mass_kg == pytest.approx(
        (0.8171947, 0.8160186, 0.9189924, 0.9177942), abs=1e-6)
    assert model.suspension_target_travel_m == (0.025,) * 4
    assert model.suspension_spring_n_per_m == (500.0,) * 4
    assert model.suspension_damper_ns_per_m == (100.0,) * 4
    assert "training_only_fits" not in model.metadata()


def _curve_value(slip, curve):
    variable = ca.SX.sym("slip")
    function = ca.Function(
        "curve_value", [variable],
        [_unity_friction_curve_response(variable, curve)])
    return float(function(slip))


def _tire_force(long_slip, lateral_slip, load=100.0, radius=0.059):
    long_var = ca.SX.sym("long_slip")
    lateral_var = ca.SX.sym("lateral_slip")
    outputs = _unity_tire_force_shader_casadi(
        long_var, lateral_var, load, radius, FORWARD, LATERAL)
    function = ca.Function("tire_force", [long_var, lateral_var], list(outputs))
    return tuple(float(value) for value in function(long_slip, lateral_slip))


def _applied_torque_slips(long_speed, lateral_speed, wheel_surface_speed):
    long_var = ca.SX.sym("long_speed")
    lateral_var = ca.SX.sym("lateral_speed")
    wheel_var = ca.SX.sym("wheel_surface_speed")
    function = ca.Function(
        "applied_torque_slips", [long_var, lateral_var, wheel_var],
        list(_physx_applied_torque_slips_casadi(
            long_var, lateral_var, wheel_var)))
    return tuple(float(value) for value in function(
        long_speed, lateral_speed, wheel_surface_speed))


def test_native_curve_matches_piecewise_formula_at_knots_and_midpoint():
    assert _curve_value(0.0, FORWARD) == pytest.approx(0.0)
    assert _curve_value(0.075, FORWARD) == pytest.approx(0.9 * 0.8 * 0.625)
    assert _curve_value(0.15, FORWARD) == pytest.approx(0.9 * 0.8)
    assert _curve_value(0.25, FORWARD) == pytest.approx(0.58 * 0.8)
    assert _curve_value(0.5, FORWARD) == pytest.approx(0.58 * 0.8)


def test_native_curve_is_odd_for_both_turn_directions():
    for curve in (FORWARD, LATERAL):
        for slip in (0.001, curve[0] * 0.5, curve[0],
                     (curve[0] + curve[2]) * 0.5, curve[2], 0.5):
            assert _curve_value(-slip, curve) == pytest.approx(
                -_curve_value(slip, curve), abs=1e-12)


def test_combined_slip_ellipse_and_native_output_signs():
    torque, force_long, force_lateral, aligning_moment = _tire_force(
        FORWARD[0], LATERAL[0])

    assert (force_long / (100.0 * FORWARD[1]))**2 + (
        force_lateral / (100.0 * LATERAL[1]))**2 == pytest.approx(1.0)
    assert force_long > 0.0
    assert force_lateral < 0.0
    assert torque == pytest.approx(-force_long * 0.059)
    assert aligning_moment == 0.0


def test_below_combined_slip_limit_is_not_rescaled():
    torque, force_long, force_lateral, aligning_moment = _tire_force(
        0.01, 0.001)
    assert force_long == pytest.approx(100.0 * _curve_value(0.01, FORWARD))
    assert force_lateral == pytest.approx(-100.0 * _curve_value(0.001, LATERAL))
    assert torque == pytest.approx(-force_long * 0.059)
    assert aligning_moment == 0.0


def test_physx_applied_torque_slips_use_wheel_speed_and_atan():
    long_slip, lateral_slip = _applied_torque_slips(5.0, 0.5, 5.5)
    assert long_slip == pytest.approx(0.5 / 5.6)
    assert lateral_slip == pytest.approx(math.atan(0.5 / 6.0))


def test_physx_applied_torque_slips_are_well_defined_at_rest():
    assert _applied_torque_slips(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0))


def test_source_suspension_ray_uses_body_down_axis_not_wheel_camber():
    model = UnityWheelModel.from_source_capture()
    body_height = (model.suspension_target_travel_m[0]
                   - model.suspension_base_xyz_body_m[0][1]
                   + model.wheel_radius_m)
    state = ca.SX.sym("suspension_state", 6)
    travels, _ = suspension_travel_casadi(
        model, *(state[index] for index in range(6)))
    travel_fn = ca.Function("source_ray_travel", [state], [travels[0]])

    assert float(travel_fn([body_height, 0.0, 0.0, 0.0, 0.0, 0.0])) == pytest.approx(
        model.suspension_target_travel_m[0])


def test_source_suspension_force_at_zero_jounce_is_sprung_weight():
    model = UnityWheelModel.from_source_capture()
    body_height = (model.suspension_target_travel_m[0]
                   - model.suspension_base_xyz_body_m[0][1]
                   + model.wheel_radius_m)
    state = ca.SX.sym("suspension_force_state", 6)
    loads = suspension_loads_casadi(
        model, *(state[index] for index in range(6)))
    load_fn = ca.Function("source_suspension_force", [state], [loads[0]])

    force = float(load_fn([body_height, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert force == pytest.approx(
        model.sprung_mass_kg[0] * 9.81, rel=1e-12)


def test_physx_default_tire_load_filter_uses_wheel_plus_sprung_rest_mass():
    rest_load_mass = 0.8171947 + 0.109
    rest_load = 9.81 * rest_load_mass
    suspension_force = 9.81 * 0.8171947
    variable = ca.SX.sym("suspension_force")
    tire_load = _physx_filtered_tire_load_casadi(
        variable, 0.8171947, 0.109, 9.81)
    function = ca.Function("source_tire_load_filter", [variable], [tire_load])

    normalized_load = 0.8171947 / rest_load_mass
    expected_filter = 0.2308 + normalized_load * (3.0 - 0.2308) / 3.0
    assert float(function(suspension_force)) == pytest.approx(
        rest_load * expected_filter)


@pytest.mark.parametrize("wheel_surface_speed", (4.0, 5.0, 6.0))
def test_physx_long_slip_inverse_reconstructs_forward_wheel_speed(
        wheel_surface_speed):
    long_speed = 5.0
    long_slip, _ = _applied_torque_slips(
        long_speed, 0.0, wheel_surface_speed)
    reconstructed = _wheel_surface_speed_from_physx_long_slip(
        long_speed, long_slip)
    assert reconstructed == pytest.approx(wheel_surface_speed)


@pytest.mark.parametrize(
    ("long_speed", "guide_slip", "expected_surface_speed"),
    ((5.0, 1.2, 11.0), (5.0, -1.0, 0.0), (0.2, -0.4, 0.0)),
)
def test_legacy_guide_slip_inverse_matches_recorded_convention(
        long_speed, guide_slip, expected_surface_speed):
    reconstructed = _wheel_surface_speed_from_autodrive_guide_slip(
        long_speed, guide_slip)
    assert reconstructed == pytest.approx(expected_surface_speed)
