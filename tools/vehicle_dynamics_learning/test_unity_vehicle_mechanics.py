import math

import pytest

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    RIGIDBODY_INERTIA_DIAGONAL_KGM2,
    RIGIDBODY_INERTIA_ROTATION_XYZW,
    RIGIDBODY_YAW_INERTIA_KGM2,
    MOTOR_TORQUE_NM,
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
    ackermann_angles,
    ackermann_center_from_wheel_angles,
    inertia_about_body_axis,
    guide_lateral_slip,
    guide_longitudinal_slip,
    inertia_tensor_body,
    planar_body_velocity_derivative,
    step_steering,
    wheel_actuation,
    wheel_contact_velocity,
    wheel_contact_velocity_3d,
    wheel_friction_curve_value,
    wheel_hit_slip_candidate,
    wheel_hit_longitudinal_slip_from_state,
    wheel_angular_acceleration,
    wheel_angular_velocity_step,
    wheel_surface_speed,
    wheel_suspension_compression_m,
    wheel_suspension_load_n,
    wheel_suspension_travel_from_body_pose,
    rigid_body_angular_torque,
)


def test_zero_throttle_is_the_exact_all_wheel_brake_branch():
    result = wheel_actuation(0.0)
    assert result.motor_torque_nm == (0.0, 0.0, 0.0, 0.0)
    assert result.brake_torque_nm == (MOTOR_TORQUE_NM,) * 4


@pytest.mark.parametrize("command, expected", [(0.2, 21.4), (-0.2, -21.4)])
def test_nonzero_throttle_is_equal_signed_motor_torque(command, expected):
    result = wheel_actuation(command)
    assert result.motor_torque_nm == pytest.approx((expected,) * 4)
    assert result.brake_torque_nm == (0.0, 0.0, 0.0, 0.0)


def test_steering_matches_source_increment_and_target_branch():
    dt = 0.025
    result = step_steering(0.0, 0.5 * math.radians(30.0), dt)
    assert result == pytest.approx(STEERING_RATE_RADPS * dt)
    assert step_steering(
        result, 0.5 * math.radians(30.0), 0.2) == pytest.approx(
            0.5 * STEERING_LIMIT_RAD)


def test_source_steering_unwind_can_snap_to_new_target():
    feedback = math.radians(15.0)
    target = math.radians(12.0)
    assert step_steering(feedback, target, 0.001) == pytest.approx(target)


def test_ackermann_has_correct_inner_wheel_for_both_turn_directions():
    left_turn = ackermann_angles(0.4)
    right_turn = ackermann_angles(-0.4)
    assert left_turn[0] > left_turn[1] > 0.0
    assert right_turn[1] < right_turn[0] < 0.0
    assert left_turn[2:] == right_turn[2:] == (0.0, 0.0)


@pytest.mark.parametrize("center", [-0.5, -0.3, 0.0, 0.3, 0.5])
def test_ackermann_inverse_recovers_center_without_average_angle_bias(center):
    left, right, _, _ = ackermann_angles(center)
    recovered = ackermann_center_from_wheel_angles(left, right)
    assert recovered == pytest.approx(center, abs=1e-12)


def test_wheel_point_velocity_includes_yaw_and_actual_steer():
    # Rear axle center longitudinal velocity is u; left/right wheel speeds
    # differ by the expected yaw-rate times half-track.
    left = wheel_contact_velocity(4.0, 0.0, 2.0, 2)
    right = wheel_contact_velocity(4.0, 0.0, 2.0, 3)
    assert left[0] == pytest.approx(4.0 - 2.0 * 0.118)
    assert right[0] == pytest.approx(4.0 + 2.0 * 0.118)
    assert wheel_contact_velocity(4.0, 0.0, 0.0, 0, 0.2)[1] < 0.0


def test_rpm_to_surface_speed_keeps_sign():
    assert wheel_surface_speed(60.0) == pytest.approx(2.0 * math.pi * 0.059)
    assert wheel_surface_speed(-60.0) == pytest.approx(-2.0 * math.pi * 0.059)


def test_unity_suspension_load_uses_compression_and_damper_rate():
    sprung_mass = 0.817168832
    static_load = sprung_mass * 9.80665
    assert wheel_suspension_load_n(sprung_mass, 0.01, 0.02) == pytest.approx(
        static_load + 7.0)
    assert wheel_suspension_load_n(sprung_mass, 0.0, -0.2) == 0.0
    assert wheel_suspension_load_n(
        sprung_mass, 0.1, 1.0, grounded=False) == 0.0


def test_unity_wheel_travel_is_intersection_of_ray_and_tire_radius_plane():
    camber_half_angle = math.radians(2.5)
    travel = wheel_suspension_travel_from_body_pose(
        body_position_world_m=(0.0, 0.0, 0.0),
        body_rotation_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        wheel_transform_position_body_m=(0.0, 0.06, 0.17),
        wheel_transform_rotation_body_xyzw=(
            0.0, 0.0, -math.sin(camber_half_angle),
            math.cos(camber_half_angle)),
        wheel_collider_center_local_m=(0.0, 0.025, 0.0),
        ground_point_world_m=(0.0, 0.0, 0.0),
        ground_normal_world=(0.0, 1.0, 0.0),
        wheel_radius_m=0.059,
    )
    assert travel == pytest.approx(0.026, abs=5e-6)
    assert wheel_suspension_compression_m(travel) == pytest.approx(
        -0.001, abs=5e-6)


def test_contact_point_velocity_uses_rigidbody_com_and_full_angular_velocity():
    longitudinal, lateral = wheel_contact_velocity_3d(
        body_velocity=(1.0, 0.0, 0.0),
        body_angular_velocity=(0.0, 0.0, 2.0),
        contact_position_from_body_origin=(0.0, 2.0, 0.0),
        center_of_mass_from_body_origin=(0.0, 1.0, 0.0),
        contact_forward=(1.0, 0.0, 0.0),
        contact_sideways=(0.0, 1.0, 0.0),
    )
    assert longitudinal == pytest.approx(-1.0)
    assert lateral == pytest.approx(0.0)


def test_empirical_slip_candidate_uses_wheel_surface_speed_and_slip_angle():
    radius = 0.059
    rpm_for_5mps = 5.0 / (2.0 * math.pi * radius) * 60.0
    longitudinal, lateral = wheel_hit_slip_candidate(
        5.0, 1.0, rpm_for_5mps, radius_m=radius)
    assert longitudinal == pytest.approx(0.0, abs=1e-12)
    assert lateral == pytest.approx(math.atan2(1.0, 6.0))


def test_unity_longitudinal_slip_candidate_uses_the_faster_surface_speed():
    radius = 0.059
    rpm_for_4mps = 4.0 / (2.0 * math.pi * radius) * 60.0
    assert wheel_hit_longitudinal_slip_from_state(
        5.0, rpm_for_4mps, radius) == pytest.approx(-1.0 / 5.1)
    assert wheel_hit_longitudinal_slip_from_state(
        0.3, 0.0, radius) == pytest.approx(-0.75)


def test_documented_slip_equations_are_distinct_from_regularized_wheelhit_fit():
    radius = 0.059
    rpm_for_5_5mps = 5.5 / (2.0 * math.pi * radius) * 60.0
    assert guide_longitudinal_slip(5.0, rpm_for_5_5mps, radius) == pytest.approx(0.1)
    assert guide_lateral_slip(-5.0, 1.0) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="singular"):
        guide_lateral_slip(0.0, 1.0)
    with pytest.raises(ValueError, match="singular"):
        guide_longitudinal_slip(0.0, rpm_for_5_5mps, radius)


def test_contact_velocity_rejects_malformed_geometry():
    with pytest.raises(ValueError):
        wheel_contact_velocity_3d(
            (0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def test_piecewise_tire_curve_hits_serialized_landmarks_and_is_odd():
    curve = lambda slip: wheel_friction_curve_value(
        slip, 0.15, 0.9, 0.25, 0.58, 0.8, 5.0)
    assert curve(0.0) == 0.0
    assert curve(0.15) == pytest.approx(0.72)
    assert curve(0.25) == pytest.approx(0.464)
    assert curve(-0.15) == pytest.approx(-0.72)
    assert curve(0.40) == pytest.approx(0.464)


def test_newton_euler_and_wheel_torque_balance_signs():
    u_dot, v_dot, yaw_accel = planar_body_velocity_derivative(
        forward_speed_mps=4.0,
        lateral_speed_mps=0.5,
        yaw_rate_radps=2.0,
        force_forward_n=0.0,
        force_lateral_n=0.0,
        yaw_moment_nm=0.1,
        mass_kg=2.0,
        yaw_inertia_kgm2=0.05,
    )
    assert (u_dot, v_dot, yaw_accel) == pytest.approx((1.0, -8.0, 2.0))
    spin_accel = wheel_angular_acceleration(
        motor_torque_nm=2.0,
        brake_torque_nm=1.0,
        wheel_angular_speed_radps=3.0,
        tire_force_forward_n=4.0,
        radius_m=0.5,
        wheel_inertia_kgm2=0.25,
    )
    assert spin_accel == pytest.approx(-4.0)


def test_wheel_brake_impulse_stops_without_reversing_spin():
    inertia = 4.366961924e-4
    assert wheel_angular_velocity_step(
        wheel_angular_speed_radps=150.0,
        motor_torque_nm=0.0,
        brake_torque_nm=428.0,
        tire_force_forward_n=0.0,
        radius_m=0.059,
        wheel_inertia_kgm2=inertia,
        damping_nms=0.25,
        dt_s=0.001,
    ) == 0.0
    assert wheel_angular_velocity_step(
        wheel_angular_speed_radps=-10.0,
        motor_torque_nm=0.0,
        brake_torque_nm=1.0,
        tire_force_forward_n=0.0,
        radius_m=0.059,
        wheel_inertia_kgm2=0.01,
        damping_nms=0.0,
        dt_s=0.01,
    ) == pytest.approx(-9.0)


def test_wheel_spin_step_uses_torque_balance_and_signed_tire_force():
    result = wheel_angular_velocity_step(
        wheel_angular_speed_radps=10.0,
        motor_torque_nm=2.0,
        brake_torque_nm=0.0,
        tire_force_forward_n=1.0,
        radius_m=0.5,
        wheel_inertia_kgm2=0.25,
        damping_nms=0.0,
        dt_s=0.01,
    )
    assert result == pytest.approx(10.06)


def test_unity_rotated_inertia_tensor_yaw_projection_matches_capture():
    yaw_inertia = inertia_about_body_axis(
        RIGIDBODY_INERTIA_DIAGONAL_KGM2,
        RIGIDBODY_INERTIA_ROTATION_XYZW,
        (0.0, 1.0, 0.0),
    )
    assert yaw_inertia == pytest.approx(RIGIDBODY_YAW_INERTIA_KGM2, abs=1e-7)


def test_unity_rotated_inertia_matrix_and_full_angular_momentum_equation():
    inertia = inertia_tensor_body(
        RIGIDBODY_INERTIA_DIAGONAL_KGM2,
        RIGIDBODY_INERTIA_ROTATION_XYZW,
    )
    assert inertia[1][1] == pytest.approx(RIGIDBODY_YAW_INERTIA_KGM2, abs=1e-7)
    diagonal_torque = rigid_body_angular_torque(
        (1.0, 2.0, 3.0), (4.0, 5.0, 6.0),
        ((0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.3)),
    )
    assert diagonal_torque == pytest.approx((1.0, 0.4, 2.0))
