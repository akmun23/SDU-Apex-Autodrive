from __future__ import annotations

import math

import pytest

from tools.vehicle_dynamics_learning.signal_semantics import (
    DT_S,
    ENCODER_SOURCE_DT_GATE_S,
    REAR_AXLE_TO_COM_X_M,
    REAR_TRACK_WIDTH_M,
    SIGNAL_DICTIONARY,
    WHEEL_RADIUS_M,
    com_to_rear_axle_velocity,
    elapsed_window_surface_rate,
    fixed_n_period_surface_rate,
    fixed_period_surface_rate,
    rear_contact_speeds,
    variable_stamp_surface_rate,
    wrapped_angle_difference,
)


def test_fixed_25ms_rate_uses_physical_cadence_not_source_stamp_jitter() -> None:
    delta = 2.0 * DT_S / WHEEL_RADIUS_M
    expected = 2.0
    assert fixed_period_surface_rate(0.0, delta) == pytest.approx(expected)
    assert variable_stamp_surface_rate(
        0.0, delta, ENCODER_SOURCE_DT_GATE_S[1]) == pytest.approx(
            expected * DT_S / ENCODER_SOURCE_DT_GATE_S[1])


@pytest.mark.parametrize("periods", (1, 2, 4))
def test_fixed_n_period_encoder_increment_reconstructs_rate(periods: int) -> None:
    rate = -3.25
    delta = rate * periods * DT_S / WHEEL_RADIUS_M
    assert fixed_n_period_surface_rate(0.0, delta, periods) == pytest.approx(rate)


def test_stored_100ms_proxy_uses_elapsed_window_and_validity_gate() -> None:
    elapsed = 0.103
    expected = 2.7
    delta = expected * elapsed / WHEEL_RADIUS_M
    assert elapsed_window_surface_rate(0.0, delta, elapsed) == pytest.approx(expected)
    with pytest.raises(ValueError):
        elapsed_window_surface_rate(0.0, delta, 0.13)


def test_variable_stamp_gate_is_enforced() -> None:
    low, high = ENCODER_SOURCE_DT_GATE_S
    assert variable_stamp_surface_rate(0.0, 1.0, low) == pytest.approx(
        WHEEL_RADIUS_M / low)
    assert variable_stamp_surface_rate(0.0, 1.0, high) == pytest.approx(
        WHEEL_RADIUS_M / high)
    with pytest.raises(ValueError):
        variable_stamp_surface_rate(0.0, 1.0, high + 1e-6)


def test_com_to_rear_axle_velocity_uses_signed_rigid_body_shift() -> None:
    u_rear, v_rear = com_to_rear_axle_velocity(
        4.0, 0.8, 2.0, REAR_AXLE_TO_COM_X_M)
    assert u_rear == pytest.approx(4.0)
    assert v_rear == pytest.approx(0.8 - 2.0 * REAR_AXLE_TO_COM_X_M)


def test_rear_contact_speed_left_right_sign_convention() -> None:
    left, right = rear_contact_speeds(5.0, 2.0, REAR_TRACK_WIDTH_M)
    assert left == pytest.approx(5.0 - 2.0 * REAR_TRACK_WIDTH_M / 2.0)
    assert right == pytest.approx(5.0 + 2.0 * REAR_TRACK_WIDTH_M / 2.0)
    assert rear_contact_speeds(5.0, -2.0) == pytest.approx((right, left))


def test_angle_difference_wraps_across_pi() -> None:
    assert wrapped_angle_difference(-math.pi + 0.1, math.pi - 0.1) == pytest.approx(0.2)


def test_signal_dictionary_covers_required_semantics_and_causality() -> None:
    required = {
        "body_u_rear", "body_v_rear", "yaw_rate", "rear_axle_world_pose",
        "com_world_pose", "steering_command", "throttle_command",
        "steering_feedback", "throttle_feedback", "encoder_left_angle",
        "encoder_right_angle", "encoder_rate_25ms_fixed",
        "encoder_rate_variable_stamp_diagnostic", "encoder_rate_100ms_stored",
        "production_wheel_quantities", "imu_accel_yaw", "imu_roll_roll_rate",
        "linear_acceleration_labels",
    }
    assert required == set(SIGNAL_DICTIONARY)
    for record in SIGNAL_DICTIONARY.values():
        for key in ("name", "units", "source", "frame_reference", "observation",
                    "formula", "causal_window", "nominal_delay", "validity",
                    "training_use", "rollout_use", "role"):
            assert record[key]
    assert "future" in SIGNAL_DICTIONARY["encoder_rate_25ms_fixed"]["rollout_use"]
