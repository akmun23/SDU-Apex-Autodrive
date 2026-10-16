import csv
import pytest

from tools.racing.build_unity_open_loop_profile_from_capture import build_profile


def _capture(path, count=100, gap_index=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sim_time_s", "steering_command", "throttle_command",
                         "pos_world_x_m", "pos_world_z_m"])
        for index in range(count):
            elapsed = index + (1 if gap_index is not None and index >= gap_index else 0)
            writer.writerow([elapsed * 0.001, index / 1000.0, -index / 1000.0,
                             index * 0.01, index * -0.02])


def _source():
    return {
        "raceline_feedback_control": True,
        "warmup_command_count": 2,
        "physics_ticks_per_command": 25,
        "fixed_delta_time_s": 0.001,
        "commands": [
            {"steering_norm": 0.0, "throttle_norm": 0.0},
            {"steering_norm": 0.1, "throttle_norm": 0.2},
            {"steering_norm": 0.2, "throttle_norm": 0.3},
        ],
    }


def test_profile_keeps_warmup_then_samples_captured_commands_at_25ms(tmp_path):
    capture = tmp_path / "capture.csv"
    _capture(capture, count=100)

    profile = build_profile(_source(), {"scored_control_updates": 2}, capture)

    assert profile["spatial_command_lookup"] is False
    assert profile["commands"] == [
        {"steering_norm": 0.0, "throttle_norm": 0.0},
        {"steering_norm": 0.1, "throttle_norm": 0.2},
        {"steering_norm": 0.05, "throttle_norm": -0.05},
        {"steering_norm": 0.075, "throttle_norm": -0.075},
    ]


def test_profile_rejects_a_capture_with_a_fixed_step_gap(tmp_path):
    capture = tmp_path / "capture.csv"
    _capture(capture, count=100, gap_index=51)

    with pytest.raises(ValueError, match="continuous 1 kHz"):
        build_profile(_source(), {"scored_control_updates": 2}, capture)


def test_spatial_profile_preserves_reference_positions_without_feedback(tmp_path):
    capture = tmp_path / "capture.csv"
    _capture(capture, count=100)

    profile = build_profile(
        _source(), {"scored_control_updates": 2}, capture, spatial=True)

    assert profile["spatial_command_lookup"] is True
    assert profile["warmup_command_count"] == 2
    assert profile["scored_horizon_commands"] == 2
    assert profile["commands"][1]["has_reference_position"] is False
    assert profile["commands"][2] == {
        "steering_norm": 0.05,
        "throttle_norm": -0.05,
        "has_reference_position": True,
        "reference_x_m": 0.5,
        "reference_z_m": -1.0,
    }


def test_spatial_seed_profile_can_be_reindexed_to_captured_unity_path(tmp_path):
    capture = tmp_path / "capture.csv"
    _capture(capture, count=100)
    source = {
        "raceline_feedback_control": False,
        "spatial_command_lookup": True,
        "warmup_command_count": 2,
        "scored_horizon_commands": 2,
        "physics_ticks_per_command": 25,
        "fixed_delta_time_s": 0.001,
        "commands": [
            {"steering_norm": 0.0, "throttle_norm": 0.0},
            {"steering_norm": 0.1, "throttle_norm": 0.2},
            {"steering_norm": -0.2, "throttle_norm": 0.3,
             "has_reference_position": True,
             "reference_x_m": 1.0, "reference_z_m": 2.0},
            {"steering_norm": -0.3, "throttle_norm": 0.4,
             "has_reference_position": True,
             "reference_x_m": 3.0, "reference_z_m": 4.0},
        ],
    }

    profile = build_profile(source, {}, capture, spatial=True)

    assert profile["warmup_command_count"] == 2
    assert profile["scored_horizon_commands"] == 2
    assert profile["commands"][2] == {
        "steering_norm": 0.05,
        "throttle_norm": -0.05,
        "has_reference_position": True,
        "reference_x_m": 0.5,
        "reference_z_m": -1.0,
    }
    assert profile["commands"][3]["reference_x_m"] == pytest.approx(0.75)


def test_spatial_route_can_be_shorter_than_scored_simulation_horizon(tmp_path):
    capture = tmp_path / "capture.csv"
    _capture(capture, count=100)
    source = _source()
    source["raceline_feedback_control"] = False
    source["spatial_command_lookup"] = True
    source["scored_horizon_commands"] = 2

    profile = build_profile(
        source, {}, capture, spatial=True, captured_control_updates=1,
        scored_horizon_commands=2)

    assert profile["scored_horizon_commands"] == 2
    assert len(profile["commands"]) == 3
    assert profile["commands"][-1]["reference_x_m"] == pytest.approx(0.5)


def test_spatial_route_cannot_use_more_points_than_recorded(tmp_path):
    capture = tmp_path / "capture.csv"
    _capture(capture, count=100)
    source = _source()
    source["raceline_feedback_control"] = False
    source["spatial_command_lookup"] = True
    source["scored_horizon_commands"] = 2

    with pytest.raises(ValueError, match="captured control count"):
        build_profile(
            source, {}, capture, spatial=True, captured_control_updates=3)
