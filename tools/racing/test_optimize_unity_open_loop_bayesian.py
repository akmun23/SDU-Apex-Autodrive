import numpy as np

from tools.racing.optimize_unity_open_loop_bayesian import (
    PARAMETER_NAMES,
    SEED_PARAMETERS,
    _horizon_check,
    make_candidate_profile,
    propose,
)


def _seed_profile():
    commands = [
        {"steering_norm": 0.0, "throttle_norm": 0.0,
         "has_reference_position": False,
         "reference_x_m": 0.0, "reference_z_m": 0.0}
        for _ in range(2)
    ]
    for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        commands.append({
            "steering_norm": float(0.2 * np.sin(angle)),
            "throttle_norm": 0.3,
            "has_reference_position": True,
            "reference_x_m": float(2.0 * np.cos(angle)),
            "reference_z_m": float(2.0 * np.sin(angle)),
        })
    return {
        "schema_version": 1,
        "fixed_delta_time_s": 0.001,
        "physics_ticks_per_command": 25,
        "command_phase_fixed_steps": 0,
        "spatial_command_lookup": True,
        "spatial_progress_distance_scale": 3.0,
        "warmup_command_count": 2,
        "scored_horizon_commands": 32,
        "commands": commands,
    }


def test_unit_knots_reproduce_seed_open_loop_commands_exactly():
    seed = _seed_profile()

    result = make_candidate_profile(seed, SEED_PARAMETERS)

    assert result == seed


def test_knotted_controls_remain_finite_bounded_and_preserve_path_tags():
    seed = _seed_profile()
    parameters = np.asarray([1.3, 1.0, 0.9, 1.1,
                             1.05, 0.95, 1.1, 0.9])

    result = make_candidate_profile(seed, parameters)

    assert result["commands"][:2] == seed["commands"][:2]
    assert all(row["has_reference_position"] for row in result["commands"][2:])
    for before, after in zip(seed["commands"][2:], result["commands"][2:]):
        assert after["reference_x_m"] == before["reference_x_m"]
        assert after["reference_z_m"] == before["reference_z_m"]
        assert -1.0 <= after["steering_norm"] <= 1.0
        assert -1.0 <= after["throttle_norm"] <= 1.0
        assert np.isfinite(after["steering_norm"])
        assert np.isfinite(after["throttle_norm"])


def test_initial_joint_design_stays_inside_configured_search_bounds():
    parameters, proposal = propose([], seed=20261014, index=1)

    assert proposal["method"] == "targeted_joint_throttle_steering_design"
    assert parameters.shape == (len(PARAMETER_NAMES),)
    assert np.all(parameters >= 0.75)
    assert np.all(parameters <= 1.60)


def test_constrained_acquisition_uses_map_margin_and_lap_observations():
    rows = []
    for index in range(17):
        parameters = np.asarray([
            0.78 + 0.035 * (index % 9),
            0.80 + 0.025 * ((index * 3) % 9),
            0.82 + 0.030 * ((index * 5) % 8),
            0.85 + 0.025 * ((index * 7) % 7),
            0.82 + 0.020 * ((index * 2) % 9),
            0.84 + 0.018 * ((index * 4) % 8),
            0.86 + 0.017 * ((index * 6) % 8),
            0.88 + 0.015 * ((index * 8) % 7),
        ])
        valid = index in (0, 4, 9, 13)
        rows.append({
            "parameters": dict(zip(PARAMETER_NAMES, parameters.tolist())),
            "lap_valid": valid,
            "unity_lap_time_s": 5.70 + 0.01 * index,
            "track_samples_with_vehicle_margin_fraction": 1.0 if valid else 0.55 + 0.02 * (index % 5),
            "track_clearance_p05_m": 0.25 if valid else 0.10 + 0.01 * (index % 4),
        })

    candidate, proposal = propose(
        rows, seed=20261014, index=17, acquisition="constrained_ei")

    assert candidate.shape == (len(PARAMETER_NAMES),)
    assert np.all(candidate >= np.asarray([0.75] * 4 + [0.80] * 4))
    assert np.all(candidate <= np.asarray([1.60] * 4 + [1.20] * 4))
    assert proposal["method"] == (
        "constrained_expected_improvement_from_lap_time_and_map_margin")
    assert 0.0 <= proposal["probability_feasible"] <= 1.0


def test_completed_unity_run_must_match_requested_profile_horizon():
    profile = _seed_profile()
    expected = (profile["warmup_command_count"] +
                profile["scored_horizon_commands"]) * 25

    assert _horizon_check(profile, {
        "unity_status": "complete",
        "fixed_steps_applied": expected,
    }) == (expected, True)
    assert _horizon_check(profile, {
        "unity_status": "complete",
        "fixed_steps_applied": expected - 1,
    }) == (expected, False)
    assert _horizon_check(profile, {
        "unity_status": "collision",
        "fixed_steps_applied": 13,
    }) == (expected, None)
