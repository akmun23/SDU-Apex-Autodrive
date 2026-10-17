import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.racing.optimize_unity_wheel_raceline import (
    UNITY_STEERING_LIMIT_RAD,
    _reference_lap_interval,
    path_reference_offsets_for_track,
    tracking_bias_values_for_track,
    unity_steering_norm_from_map_angle,
    time_parameterize_candidate,
    write_open_loop_profile,
)
from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.optimizer_unity_wheels import (
    _normalize_gate_state_anchor,
    DEFAULT_TIME_TRUST_REGION_FACTORS,
    should_retain_diagnostic_iterate,
    validate_time_trust_region_factors,
    validate_minimum_wheel_normal_load,
)


def test_map_steering_sign_matches_unity_autonomous_command():
    for map_steering_rad in (-0.2, -0.05, 0.05, 0.2):
        unity_command = unity_steering_norm_from_map_angle(map_steering_rad)
        unity_wheel_angle = -unity_command * UNITY_STEERING_LIMIT_RAD
        map_wheel_angle = -unity_wheel_angle

        assert np.isclose(map_wheel_angle, map_steering_rad)


def test_reference_lap_interval_accepts_explicit_capture_window(tmp_path):
    # Explicit times let a stable middle lap seed the periodic OCP without
    # pretending the whole multi-lap capture is one scored lap.
    assert _reference_lap_interval(
        tmp_path / "unused.json", {}, (24.575, 30.275)) == (24.575, 30.275)


def test_reference_lap_interval_rejects_nonincreasing_window(tmp_path):
    with pytest.raises(ValueError, match="finite and increasing"):
        _reference_lap_interval(tmp_path / "unused.json", {}, (2.0, 2.0))


def test_tracking_bias_interpolates_periodically_over_track_nodes():
    track = SimpleNamespace(
        s=np.arange(12, dtype=np.float64) * 0.1,
        length=1.2,
    )
    metrics = {
        "path_bias_by_route_fraction_12": [
            {"route_fraction": (index + 0.5) / 12.0,
             "median_signed_path_error_m": 0.12,
             "sample_count": 6}
            for index in range(12)
        ]
    }

    bias, summary = tracking_bias_values_for_track(track, metrics)

    assert np.allclose(bias, 0.12)
    assert summary["max_abs_bias_m"] == pytest.approx(0.12)


def test_tracking_bias_requires_repeated_samples_in_every_sector():
    track = SimpleNamespace(s=np.arange(12), length=12.0)
    metrics = {
        "path_bias_by_route_fraction_12": [
            {"route_fraction": (index + 0.5) / 12.0,
             "median_signed_path_error_m": 0.0,
             "sample_count": 4 if index == 3 else 6}
            for index in range(12)
        ]
    }

    with pytest.raises(ValueError, match="at least 5 samples per sector"):
        tracking_bias_values_for_track(track, metrics)


def test_path_reference_tube_maps_signed_offset_and_rejects_wrong_route(
        tmp_path):
    count = 80
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    reference_radius = 4.8
    center_radius = 5.0
    reference_path = tmp_path / "reference.csv"
    with reference_path.open("w", encoding="utf-8") as stream:
        stream.write("# s_m,x_m,y_m,psi_rad\n")
        for index, angle in enumerate(angles):
            stream.write(
                f"{index * 2*np.pi*reference_radius/count:.9f},"
                f"{reference_radius*np.cos(angle):.9f},"
                f"{reference_radius*np.sin(angle):.9f},"
                f"{angle + np.pi/2:.9f}\n")
    center_track = SimpleNamespace(
        count=count,
        x=center_radius * np.cos(angles),
        y=center_radius * np.sin(angles),
        psi=angles + np.pi / 2.0,
    )

    offsets, summary = path_reference_offsets_for_track(
        center_track, reference_path, maximum_distance_m=0.5)

    assert np.allclose(offsets, 0.2, atol=0.01)
    assert summary["max_centerline_to_reference_distance_m"] < 0.21
    with pytest.raises(ValueError, match="not a close, unambiguous route match"):
        path_reference_offsets_for_track(
            center_track, reference_path, maximum_distance_m=0.1)


def _small_candidate():
    return {
        "s_ref_m": np.asarray([0.0, 1.0, 2.0]),
        "ey_m": np.zeros(3),
        "epsi_rad": np.zeros(3),
        "speed_mps": np.full(3, 2.0),
        "yaw_rate_radps": np.full(3, -1.0),
        "lateral_speed_mps": np.zeros(3),
        "steering_rad": np.asarray([0.0, 0.1, 0.2]),
        "throttle_norm": np.asarray([0.1, 0.2, 0.3]),
        "x_m": np.asarray([0.0, 1.0, 2.0]),
        "y_m": np.asarray([0.0, 1.0, 0.0]),
    }


def _small_track():
    return SimpleNamespace(
        s=np.asarray([0.0, 1.0, 2.0]),
        kappa=np.zeros(3),
        length=3.0,
    )


def test_candidate_is_resampled_by_predicted_time_not_template_length():
    sampled, lap_time = time_parameterize_candidate(
        _small_candidate(), _small_track())

    assert np.isclose(lap_time, 1.5)
    assert len(sampled["sample_time_s"]) == 60
    assert np.isclose(sampled["sample_time_s"][20], 0.5)
    assert np.isclose(sampled["x_m"][20], 1.0)
    assert np.isclose(sampled["y_m"][20], 1.0)
    assert np.isclose(sampled["yaw_rate_radps"][20], -1.0)
    assert np.isclose(sampled["route_arc_m"][20], np.sqrt(2.0))


def test_open_loop_profile_uses_40hz_time_samples_and_keeps_warmup(
        tmp_path):
    warmup = [
        {"steering_norm": -0.2, "throttle_norm": 0.3},
        {"steering_norm": -0.1, "throttle_norm": 0.4},
    ]
    template = {
        "schema_version": 1,
        "fixed_delta_time_s": 0.001,
        "physics_ticks_per_command": 25,
        "warmup_command_count": len(warmup),
        "scored_horizon_commands": 4,
        "raceline_feedback_control": True,
        "commands": warmup + [dict(warmup[0]) for _ in range(4)],
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")

    output = write_open_loop_profile(
        _small_candidate(), {"predicted_lap_time_s": 1.5}, _small_track(),
        template_path, tmp_path)
    profile = json.loads(output.read_text(encoding="utf-8"))
    scored = profile["commands"][len(warmup):]

    assert profile["commands"][:len(warmup)] == warmup
    assert len(scored) == profile["scored_horizon_commands"] == 60
    assert profile["bottom_up_optimizer_time_parameterized"] is True
    assert profile["bottom_up_optimizer_time_sample_interval_s"] == 0.025
    assert np.isclose(scored[20]["target_speed_mps"], 2.0)
    assert np.isclose(scored[20]["reference_x_m"], -0.95)
    assert np.isclose(scored[20]["reference_z_m"], 0.76)
    assert np.isclose(scored[20]["throttle_norm"], 0.2)


def test_gate_state_anchor_requires_complete_finite_four_wheel_state():
    anchor = {
        "ey_m": 0.1,
        "epsi_rad": -0.2,
        "u_mps": 4.0,
        "v_mps": 0.3,
        "yaw_rate_radps": -1.2,
        "steering_rad": 0.15,
        "wheel_omega_radps": [10.0, 11.0, 12.0, 13.0],
    }

    normalized = _normalize_gate_state_anchor(anchor)
    assert normalized["wheel_omega_radps"] == [10.0, 11.0, 12.0, 13.0]
    assert normalized["yaw_rate_radps"] == -1.2
    with pytest.raises(ValueError, match="four finite values"):
        _normalize_gate_state_anchor({
            **anchor, "wheel_omega_radps": [10.0, 11.0, 12.0]})


def test_time_trust_region_schedule_is_strictly_increasing():
    assert validate_time_trust_region_factors([1.2, 1.5, 2.0]) == (
        1.2, 1.5, 2.0)
    assert validate_time_trust_region_factors(
        DEFAULT_TIME_TRUST_REGION_FACTORS) == DEFAULT_TIME_TRUST_REGION_FACTORS

    for schedule in ([], [1.0], [1.5, 1.4], [1.2, float("inf")]):
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_time_trust_region_factors(schedule)


def test_minimum_wheel_normal_load_is_between_old_floor_and_static_load():
    static_loads = (8.0, 8.0, 9.0, 9.0)
    assert validate_minimum_wheel_normal_load(0.2, static_loads) == 0.2
    for invalid in (0.049, float("nan"), 8.0):
        with pytest.raises(ValueError, match="below every static wheel load"):
            validate_minimum_wheel_normal_load(invalid, static_loads)


def test_failed_ipopt_iterate_is_retained_only_for_bounded_diagnostics():
    assert should_retain_diagnostic_iterate(False, False, 0.06) is False
    assert should_retain_diagnostic_iterate(True, True, 0.0) is False
    assert should_retain_diagnostic_iterate(True, False, 0.0637) is True
    assert should_retain_diagnostic_iterate(True, False, 0.1001) is False
    assert should_retain_diagnostic_iterate(True, False, float("nan")) is False
