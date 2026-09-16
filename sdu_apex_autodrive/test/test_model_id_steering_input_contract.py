"""Regression tests for the causal steering-command timing contract."""

import math
import sys


sys.path.insert(0, "tools/model_id")
from fit_speed_regime_vehicle_model import _steering_command  # noqa: E402


def test_transition_uses_previous_packet_command_when_available():
    row = {
        "commanded_steering_norm_k1": 0.8,
        "steering_target_norm_k": -0.2,
    }
    assert _steering_command(row) == -0.2


def test_missing_transition_command_does_not_leak_current_packet_command():
    row = {
        "commanded_steering_norm_k1": 0.8,
        "steering_target_norm_k": math.nan,
    }
    assert math.isnan(_steering_command(row))
