import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "tools/model_id/analyze_encoder_sides.py"
_SPEC = importlib.util.spec_from_file_location("analyze_encoder_sides", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_paired_encoder_average_beats_either_side_on_a_straight_holdout():
    radius = 0.05
    rows = []
    for sample in range(8):
        time_s = sample * 0.025
        rows.append({
            "simulation_time_s": str(time_s),
            "simulator_position_x": str(2.0 * time_s),
            "simulator_position_y": "0",
            "simulator_encoder_angles_left": str(1.5 * time_s / radius),
            "simulator_encoder_angles_right": str(2.5 * time_s / radius),
            "simulator_orientation_quaternion_x": "0",
            "simulator_orientation_quaternion_y": "0",
            "simulator_orientation_quaternion_z": "0",
            "simulator_orientation_quaternion_w": "1",
            "simulator_angular_velocity_z": "0",
            "simulator_feedback_throttle_norm": "0",
        })
    parameters = {
        "wheel_radius_m": radius,
        "wheel_speed_window_s": 0.1,
        "wheel_speed_scale_speeds_mps": [0.0, 10.0],
        "wheel_speed_scale_values": [1.0, 1.0],
    }

    report = _MODULE.score_packets(rows, parameters)
    scores = report["error_by_regime"]["full_brake_straight"]

    assert scores["paired_mean"]["samples"] == 4
    assert scores["paired_mean"]["abs_p95_mps"] == pytest.approx(0.0)
    assert scores["left_only"]["abs_p95_mps"] == pytest.approx(0.5)
    assert scores["right_only"]["abs_p95_mps"] == pytest.approx(0.5)


def test_turn_encoder_ablation_is_split_by_signed_yaw_direction():
    radius = 0.05
    rows = []
    left_angle = 0.0
    right_angle = 0.0
    for sample in range(20):
        time_s = sample * 0.025
        if sample > 0:
            positive_yaw = sample < 10
            left_angle += (2.0 if positive_yaw else 3.0) * 0.025 / radius
            right_angle += (3.0 if positive_yaw else 2.0) * 0.025 / radius
        rows.append({
            "simulation_time_s": str(time_s),
            "simulator_position_x": str(2.0 * time_s),
            "simulator_position_y": "0",
            "simulator_encoder_angles_left": str(left_angle),
            "simulator_encoder_angles_right": str(right_angle),
            "simulator_orientation_quaternion_x": "0",
            "simulator_orientation_quaternion_y": "0",
            "simulator_orientation_quaternion_z": "0",
            "simulator_orientation_quaternion_w": "1",
            "simulator_angular_velocity_z": "0.2" if sample < 10 else "-0.2",
            "simulator_feedback_throttle_norm": "0.5",
        })
    parameters = {
        "wheel_radius_m": radius,
        "wheel_speed_window_s": 0.1,
        "wheel_speed_scale_speeds_mps": [0.0, 10.0],
        "wheel_speed_scale_values": [1.0, 1.0],
    }

    report = _MODULE.score_packets(rows, parameters)
    regimes = report["error_by_regime"]

    assert regimes["powered_turn_positive_yaw"]["paired_mean"]["samples"] == 6
    assert regimes["powered_turn_negative_yaw"]["paired_mean"]["samples"] == 10
    assert (regimes["powered_turn_positive_yaw"]["yaw_direction_selected"][
        "mae_mps"] == pytest.approx(
            regimes["powered_turn_positive_yaw"]["left_only"]["mae_mps"]))
    assert (regimes["powered_turn_negative_yaw"]["yaw_direction_selected"][
        "mae_mps"] == pytest.approx(
            regimes["powered_turn_negative_yaw"]["right_only"]["mae_mps"]))
    assert (regimes["powered_turn_positive_yaw"]["paired_mean"]["samples"] +
            regimes["powered_turn_negative_yaw"]["paired_mean"]["samples"] ==
            regimes["powered_turn"]["paired_mean"]["samples"])
