import csv

import pytest

from tools.racing.validate_unity_wheel_raceline import (
    _prediction_tracking_metrics,
    _validate_time_parameterized_profile,
)


def _profile(predicted_s=5.7, scored_count=228):
    return {
        "warmup_command_count": 311,
        "scored_horizon_commands": scored_count,
        "bottom_up_optimizer_predicted_lap_time_s": predicted_s,
        "bottom_up_optimizer_time_parameterized": True,
        "bottom_up_optimizer_time_sample_interval_s": 0.025,
        "commands": [{} for _ in range(311 + scored_count)],
    }


def test_time_parameterized_horizon_matches_predicted_lap_time():
    assert _validate_time_parameterized_profile(_profile()) == (311, 228)


def test_rejects_stale_full_horizon_instead_of_one_lap():
    with pytest.raises(ValueError, match="horizon length disagrees"):
        _validate_time_parameterized_profile(_profile(scored_count=1421))


def test_rejects_profile_without_explicit_40hz_time_sampling():
    profile = _profile()
    profile["bottom_up_optimizer_time_parameterized"] = False
    with pytest.raises(ValueError, match="not sampled on the 40 Hz"):
        _validate_time_parameterized_profile(profile)


def test_prediction_tracking_metrics_separates_spatial_and_state_errors(tmp_path):
    states_path = tmp_path / "states.csv"
    with states_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x_m", "y_m", "speed_mps", "lateral_speed_mps",
                         "yaw_rate_radps", "steering_rad"))
        for x in (0.0, 0.1, 0.2):
            writer.writerow((x, 0.0, 1.0, 0.0, 0.0, 0.0))

    capture_path = tmp_path / "capture.csv"
    with capture_path.open("w", newline="") as stream:
        stream.write("# unity metadata\n")
        writer = csv.writer(stream)
        writer.writerow(("sim_time_s", "pos_world_x_m", "pos_world_z_m",
                         "vel_body_x_mps", "vel_body_z_mps", "omega_body_y_rps",
                         "fl_steer_deg", "fr_steer_deg"))
        for step in range(101):
            writer.writerow((step * 0.001, 0.03, -0.22, 0.0, 1.1, 0.0, 0.0, 0.0))

    result = _prediction_tracking_metrics(capture_path, states_path, 0.0, 0.075)

    assert result["sample_count"] == 3
    assert "not equal-time free rollout" in result["method"]
    assert result["errors"]["path_distance_m"]["rmse"] == pytest.approx(0.02)
    assert result["signed_path_error_m"]["median"] == pytest.approx(0.02)
    assert result["errors"]["forward_speed_mps"]["rmse"] == pytest.approx(0.1)
    assert result["errors"]["road_wheel_steering_rad"]["rmse"] == pytest.approx(0.0)
