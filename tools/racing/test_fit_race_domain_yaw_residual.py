from __future__ import annotations

import numpy as np

from tools.racing.fit_race_domain_yaw_residual import (
    ATLAS_LABEL,
    _nearest_state_for_source_time,
    candidate_overlay_parameters,
)


def test_source_time_join_tolerates_only_float_rounding() -> None:
    stamp = 1_791_270_862_373_482_500
    state = np.asarray((8.0, 0.2, -0.7))
    states = {stamp: state, stamp + 25_000_000: state + 1.0}

    matched, delta = _nearest_state_for_source_time(
        states, sorted(states), (stamp - 128) / 1.0e9)

    assert matched is state
    assert delta is not None and delta < 500


def test_source_time_join_rejects_a_neighboring_sensor_packet() -> None:
    stamp = 1_791_270_862_373_482_500
    state = np.asarray((8.0, 0.2, -0.7))
    states = {stamp: state, stamp + 25_000_000: state + 1.0}

    matched, delta = _nearest_state_for_source_time(
        states, sorted(states), (stamp + 1_000_000) / 1.0e9)

    assert matched is None
    assert delta is not None and 999_000 <= delta <= 1_001_000


def test_highsteer_atlas_labels_retain_speed_angle_sign_and_repeat() -> None:
    match = ATLAS_LABEL.fullmatch("atlas_r02_v8.75_a0.5000_turn-1")

    assert match is not None
    assert match["replicate"] == "02"
    assert float(match["speed"]) == 8.75
    assert float(match["steering"]) == 0.5
    assert int(match["direction"]) == -1


def test_candidate_overlay_uses_runtime_parameter_names_and_support_gates() -> None:
    model = {
        "gain": 0.8,
        "correction_clip_radps2": 12.0,
        "feature_mean": [0.0] * 13,
        "feature_scale": [1.0] * 13,
        "coefficients_with_intercept": [0.0] * 14,
        "support_gate": {
            "kind": "smoothstep_target_tracking_steering_v1",
            "target_speed_zero_mps": 7.5,
            "target_speed_full_mps": 8.0,
            "speed_deficit_full_mps": 0.5,
            "speed_deficit_zero_mps": 1.0,
            "abs_steering_zero_rad": 0.3,
            "abs_steering_full_rad": 0.35,
            "actual_speed_zero_mps": 7.75,
            "actual_speed_full_mps": 8.25,
            "actual_speed_upper_full_mps": 10.0,
            "actual_speed_upper_zero_mps": 10.5,
        },
    }

    overlay = candidate_overlay_parameters(model)

    assert overlay["yaw_rate_residual_enabled"] is True
    assert overlay["yaw_rate_residual_gain"] == 0.8
    assert overlay["yaw_rate_residual_target_speed_full_mps"] == 8.0
    assert overlay["yaw_rate_residual_actual_speed_upper_zero_mps"] == 10.5
