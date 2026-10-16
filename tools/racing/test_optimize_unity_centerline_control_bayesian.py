import json
from pathlib import Path

import numpy as np

from tools.racing.optimize_unity_centerline_control_bayesian import (
    DEFAULT_CENTERLINE,
    DEFAULT_REFERENCE_CAPTURE,
    DEFAULT_SEED_PROFILE,
    KNOTS,
    LOWER,
    PARAMETER_NAMES,
    UPPER,
    _closed_centerline_for_gate,
    _candidate_objective,
    _initial_design,
    _map_from_unity,
    _resample_seed_controls,
    _seed_route,
    _smooth_window_weights,
    _seed_parameters,
    _unity_from_map,
    _validate_seed_trial,
    make_candidate_profile,
)


def _real_seed_and_centerline():
    seed = json.loads(DEFAULT_SEED_PROFILE.read_text(encoding="utf-8"))
    route_count = seed["scored_horizon_commands"]
    route, progress, metadata = _closed_centerline_for_gate(
        DEFAULT_CENTERLINE, DEFAULT_REFERENCE_CAPTURE,
        seed["warmup_command_count"], route_count)
    return seed, route, progress, metadata


def test_map_unity_coordinate_transform_is_inverse():
    points = np.asarray([[-3.7, 0.8], [0.0, -2.4], [1.2, 5.6]])

    assert np.allclose(_unity_from_map(_map_from_unity(points)), points)


def test_seed_trial_validation_accepts_only_verified_clear_lap(tmp_path):
    seed_path = tmp_path / "command_profile.json"
    trial_path = tmp_path / "trial.json"
    seed_path.write_text("{}", encoding="utf-8")
    trial_path.write_text(json.dumps({
        "lap_valid": True,
        "track_valid": True,
        "collisions_observed": 0,
        "unity_lap_time_s": 5.7,
        "track_clearance": {"track_clearance_min_m": 0.21},
    }), encoding="utf-8")

    summary = _validate_seed_trial(seed_path)

    assert summary["lap_time_s"] == 5.7
    assert summary["minimum_map_clearance_m"] == 0.21


def test_seed_trial_validation_rejects_known_off_track_seed(tmp_path):
    seed_path = tmp_path / "command_profile.json"
    (tmp_path / "trial.json").write_text(json.dumps({
        "lap_valid": False,
        "track_valid": False,
        "collisions_observed": 0,
        "unity_lap_time_s": 7.2,
        "track_clearance_min_m": 0.0,
    }), encoding="utf-8")

    with np.testing.assert_raises_regex(ValueError, "complete, collision-free"):
        _validate_seed_trial(seed_path)


def test_invalid_full_lap_objective_preserves_time_and_clearance_signal():
    score = _candidate_objective({
        "lap_valid": False,
        "unity_lap_time_s": 5.5,
        "track_clearance_min_m": 0.0,
        "track_samples_with_vehicle_margin_fraction": 0.6,
    })

    assert np.isclose(score, 23.95)
    assert score > 5.5
    assert _candidate_objective({
        "lap_valid": False,
        "reference_lap_progress_fraction": 0.5,
    }) == 37.0


def test_focused_control_design_changes_only_selected_sector_pair():
    seed = np.zeros(2 * KNOTS)
    focus_knot = KNOTS - 1

    design = _initial_design(seed, focus_knot=focus_knot)
    active = {focus_knot, KNOTS + focus_knot}

    assert len(design) == 9
    assert all(
        set(np.flatnonzero(point)) <= active
        for point in design)
    assert any(point[focus_knot] == 0.10 for point in design)
    assert any(point[KNOTS + focus_knot] == 0.09 for point in design)


def test_local_route_window_has_compact_smooth_support():
    progress = np.linspace(0.0, 1.0, 101)

    weights = _smooth_window_weights(progress, 0.4, 0.6)

    assert weights[0] == 0.0
    assert weights[39] == 0.0
    assert weights[40] == 0.0
    assert weights[50] == 1.0
    assert weights[60] == 0.0
    assert weights[-1] == 0.0


def test_seed_route_keeps_its_spatial_path_and_control_samples():
    warmup = [
        {"steering_norm": 0.0, "throttle_norm": 0.0,
         "has_reference_position": False,
         "reference_x_m": 0.0, "reference_z_m": 0.0}
        for _ in range(2)
    ]
    route = []
    for index, angle in enumerate(np.linspace(
            0.0, 2.0 * np.pi, 100, endpoint=False)):
        route.append({
            "steering_norm": float(0.2 * np.sin(angle)),
            "throttle_norm": 0.3,
            "has_reference_position": True,
            "reference_x_m": float(2.0 * np.cos(angle)),
            "reference_z_m": float(2.0 * np.sin(angle)),
        })
    seed = {
        "warmup_command_count": 2,
        "scored_horizon_commands": 100,
        "commands": warmup + route,
    }

    route_points, _ = _seed_route(seed, Path(__file__))
    steering, throttle, transfer = _resample_seed_controls(seed, route_points)
    base_controls = np.column_stack((steering, throttle))
    profile = make_candidate_profile(
        seed, route_points, base_controls, np.zeros(2 * KNOTS))
    scored = profile["commands"][2:]

    assert transfer["transfer"] == "exact seed route and control samples"
    assert np.allclose(
        route_points,
        [[row["reference_x_m"], row["reference_z_m"]] for row in route])
    assert np.allclose(steering, [row["steering_norm"] for row in route])
    assert np.allclose(throttle, [row["throttle_norm"] for row in route])
    assert profile["commands"][:2] == warmup
    assert profile["scored_horizon_commands"] == 100
    assert np.allclose(
        [[row["reference_x_m"], row["reference_z_m"]] for row in scored],
        route_points)
    assert np.allclose(
        [[row["steering_norm"], row["throttle_norm"]] for row in scored],
        base_controls)


def test_zero_throttle_floor_changes_only_exact_zero_scored_commands():
    warmup = [{
        "steering_norm": 0.0,
        "throttle_norm": 0.0,
        "has_reference_position": False,
        "reference_x_m": 0.0,
        "reference_z_m": 0.0,
    }]
    route = np.asarray([[np.cos(t), np.sin(t)] for t in
                        np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)])
    seed = {
        "warmup_command_count": 1,
        "scored_horizon_commands": 100,
        "commands": warmup + [{
            "steering_norm": 0.0,
            "throttle_norm": 0.0,
            "has_reference_position": True,
            "reference_x_m": float(point[0]),
            "reference_z_m": float(point[1]),
        } for point in route],
    }
    controls = np.column_stack((np.zeros(100), np.r_[0.0, np.full(99, 0.2)]))

    profile = make_candidate_profile(
        seed, route, controls, np.zeros(2 * KNOTS), zero_throttle_floor=0.05)
    scored = profile["commands"][1:]

    assert profile["commands"][0] == warmup[0]
    assert scored[0]["throttle_norm"] == 0.05
    assert all(row["throttle_norm"] == 0.2 for row in scored[1:])


def test_centerline_reference_starts_near_scored_gate_and_is_closed():
    seed, route, progress, metadata = _real_seed_and_centerline()

    assert route.shape == (seed["scored_horizon_commands"], 2)
    assert np.isfinite(route).all()
    assert progress[0] == 0.0
    assert progress[-1] < 1.0
    assert metadata["gate_to_centerline_distance_m"] < 0.75
    assert np.linalg.norm(route[0] - route[-1]) < 0.5


def test_seed_control_knots_and_centerline_profile_are_finite_and_bounded():
    seed, route, _, _ = _real_seed_and_centerline()
    parameters = _seed_parameters(seed, route)
    steering, throttle, transfer = _resample_seed_controls(seed, route)
    base_controls = np.column_stack((steering, throttle))
    profile = make_candidate_profile(seed, route, base_controls, parameters)
    warmup = int(seed["warmup_command_count"])
    route_commands = profile["commands"][warmup:]

    assert len(parameters) == len(PARAMETER_NAMES) == 2 * KNOTS
    assert np.isfinite(parameters).all()
    assert np.all(parameters >= LOWER)
    assert np.all(parameters <= UPPER)
    assert transfer["source_commands"] == (
        len(seed["commands"]) - seed["warmup_command_count"])
    assert base_controls.shape == (len(route), 2)
    assert profile["commands"][:warmup] == seed["commands"][:warmup]
    assert len(route_commands) == len(route)
    assert all(row["has_reference_position"] for row in route_commands)
    assert np.allclose(
        [[row["reference_x_m"], row["reference_z_m"]]
         for row in route_commands], route)
    assert np.isfinite([row["steering_norm"] for row in route_commands]).all()
    assert np.isfinite([row["throttle_norm"] for row in route_commands]).all()
    assert all(-1.0 <= row["steering_norm"] <= 1.0 for row in route_commands)
    assert all(0.0 <= row["throttle_norm"] <= 1.0 for row in route_commands)


def test_zero_adjustment_preserves_seed_control_shape_on_centerline_index():
    seed, route, _, _ = _real_seed_and_centerline()
    parameters = _seed_parameters(seed, route)
    steering, throttle, _ = _resample_seed_controls(seed, route)
    base_controls = np.column_stack((steering, throttle))
    profile = make_candidate_profile(seed, route, base_controls, parameters)
    warmup = int(seed["warmup_command_count"])

    assert profile["commands"][:warmup] == seed["commands"][:warmup]
    assert np.allclose(
        [[row["steering_norm"], row["throttle_norm"]]
         for row in profile["commands"][warmup:]], base_controls)
    assert len(profile["commands"]) == warmup + seed["scored_horizon_commands"]
    assert len(profile["commands"][warmup:]) != len(seed["commands"][warmup:])
    assert np.linalg.norm(
        np.asarray([profile["commands"][warmup]["reference_x_m"],
                    profile["commands"][warmup]["reference_z_m"]]) -
        np.asarray([seed["commands"][warmup]["reference_x_m"],
                    seed["commands"][warmup]["reference_z_m"]])) > 0.05


def test_control_knot_changes_change_unity_inputs_not_route_seed():
    seed, route, _, _ = _real_seed_and_centerline()
    baseline = _seed_parameters(seed, route)
    steering, throttle, _ = _resample_seed_controls(seed, route)
    base_controls = np.column_stack((steering, throttle))
    modified = baseline.copy()
    modified[0] = np.clip(modified[0] + 0.08, LOWER[0], UPPER[0])
    modified[KNOTS + 2] = np.clip(
        modified[KNOTS + 2] + 0.08, LOWER[KNOTS + 2], UPPER[KNOTS + 2])

    base_profile = make_candidate_profile(seed, route, base_controls, baseline)
    candidate_profile = make_candidate_profile(seed, route, base_controls, modified)
    warmup = int(seed["warmup_command_count"])
    base_controls = np.asarray([
        [row["steering_norm"], row["throttle_norm"]]
        for row in base_profile["commands"][warmup:]])
    candidate_controls = np.asarray([
        [row["steering_norm"], row["throttle_norm"]]
        for row in candidate_profile["commands"][warmup:]])

    assert np.allclose(
        [row["reference_x_m"] for row in base_profile["commands"][warmup:]],
        [row["reference_x_m"] for row in candidate_profile["commands"][warmup:]])
    assert np.max(np.abs(base_controls - candidate_controls)) > 1.0e-3
