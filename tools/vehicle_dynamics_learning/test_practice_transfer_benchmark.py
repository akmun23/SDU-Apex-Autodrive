from __future__ import annotations

import numpy as np

from tools.vehicle_dynamics_learning.build_practice_transfer_benchmark import (
    _category_tags,
    _select_spread,
)


def test_category_labels_use_frozen_race_speed_bins_and_active_brake_semantics():
    speed = np.full(240, 4.0)
    steering = np.zeros(240)
    throttle = np.full(240, 0.2)
    mismatch = np.zeros(240)
    tags = _category_tags(100, speed, steering, throttle, mismatch, 0.5, 1.0)
    assert "speed_0_5_mps" in tags
    assert "straight" in tags
    assert "brake" not in tags

    throttle[100] = 0.0
    tags = _category_tags(100, speed, steering, throttle, mismatch, 0.5, 1.0)
    assert "brake" in tags


def test_turn_in_apex_exit_and_high_steering_use_feedback_history():
    speed = np.full(240, 5.5)
    throttle = np.full(240, 0.2)
    mismatch = np.zeros(240)

    turn_steering = np.zeros(240)
    turn_steering[102:115] = np.linspace(0.0, 0.25, 13)
    tags = _category_tags(100, speed, turn_steering, throttle, mismatch,
                          0.5, 1.0)
    assert "turn_in" in tags

    apex_steering = np.full(240, 0.1)
    apex_steering[90:111] = 0.2
    apex_steering[100] = 0.28
    tags = _category_tags(100, speed, apex_steering, throttle, mismatch,
                          0.5, 1.0)
    assert "apex" in tags

    exit_steering = np.full(240, 0.1)
    exit_steering[100] = 0.32
    exit_steering[120] = 0.08
    tags = _category_tags(100, speed, exit_steering, throttle, mismatch,
                          0.5, 1.0)
    assert "exit" in tags

    high_steering = np.zeros(240)
    high_steering[100] = 0.42
    tags = _category_tags(100, speed, high_steering, throttle, mismatch,
                          0.5, 1.0)
    assert "high_steering" in tags


def test_wheel_body_mismatch_is_labeled_as_a_proxy_in_training_quantiles():
    speed = np.full(240, 5.5)
    steering = np.zeros(240)
    throttle = np.full(240, 0.2)
    mismatch = np.zeros(240)
    mismatch[101:120] = 0.75
    tags = _category_tags(100, speed, steering, throttle, mismatch,
                          0.5, 1.0)
    assert "wheel_body_mismatch_moderate" in tags
    assert "wheel_body_mismatch_high" not in tags

    mismatch[110:120] = 1.1
    tags = _category_tags(100, speed, steering, throttle, mismatch,
                          0.5, 1.0)
    assert "wheel_body_mismatch_high" in tags


def test_spread_selection_is_deterministic_and_preserves_extremes():
    candidates = list(range(100))
    selected = _select_spread(candidates, 8)
    assert selected == _select_spread(candidates, 8)
    assert len(selected) == 8
    assert selected[0] == 0
    assert selected[-1] == 99
    assert selected == sorted(set(selected))
