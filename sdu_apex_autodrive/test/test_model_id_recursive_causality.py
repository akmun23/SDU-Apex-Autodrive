"""Regression tests for the model-identification anti-leak contract."""

import importlib.util
import sys

import numpy as np
import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_LONG = _load("fit_longitudinal_models", "tools/model_id/fit_longitudinal_models.py")
_VEHICLE = _load("fit_vehicle_model", "tools/model_id/fit_vehicle_model.py")
_PLANT = _load("structured_vehicle_plant", "tools/model_id/structured_vehicle_plant.py")


def _vehicle_models():
    # Non-zero coefficients make this test exercise the dynamic basis rather
    # than passing only because every prediction is zero.
    return {
        channel: {"coefficients": [0.01] * len(
            _VEHICLE._basis_values(4.0, 0.2, 0.4, 0.12, 0.6, channel)[1])}
        for channel in ("u", "v", "r")}


def test_vehicle_recursive_prediction_ignores_future_measured_dynamics():
    state = np.asarray([1.0, -2.0, 0.3, 4.0, 0.2, 0.4], dtype=float)
    row = {
        "dt_sim_s": 0.025,
        "applied_steering_rad_k1": 0.12,
        "applied_throttle_norm_k1": 0.6,
        # These fields represent a future recorded transition and must not be
        # consulted by _predict after state initialization.
        "u_k_mps": 4.1,
        "v_k_mps": 0.3,
        "r_k_radps": 0.5,
    }
    corrupted = dict(row, u_k_mps=91.0, v_k_mps=-73.0, r_k_radps=44.0)

    predicted = _VEHICLE._predict(state, row, _vehicle_models())
    corrupted_prediction = _VEHICLE._predict(
        state, corrupted, _vehicle_models())
    np.testing.assert_array_equal(predicted, corrupted_prediction)


def test_longitudinal_wheel_state_uses_predicted_body_speed():
    row = {
        "applied_throttle_norm_k1": 0.7,
        "u_k_mps": 4.0,
        "v_k_mps": 0.2,
        "r_k_radps": 0.3,
    }
    corrupted = dict(row, u_k_mps=91.0, v_k_mps=-73.0, r_k_radps=44.0)
    model = {
        "wheel_dynamics": {"coefficients": [0.0, 0.8, 0.2, 0.0, 0.0, 0.1]},
        "force_max_n": 17.7,
        "slip_gain_per_mps": 2.2,
        "coast_speed_drag_n_per_mps": 0.9,
    }

    first = _LONG._wheel_acceleration_dynamic(
        model, row, u=4.0, wheel=4.5, lateral_coupling=0.0)
    second = _LONG._wheel_acceleration_dynamic(
        model, corrupted, u=4.0, wheel=4.5, lateral_coupling=0.0)
    assert first == second


def test_continuous_wheel_state_uses_measured_transition_dt():
    row = {
        "dt_sim_s": 0.025,
        "applied_throttle_norm_k1": 0.5,
        "u_k_mps": 2.0,
    }
    model = {
        "wheel_dynamics": {"coefficients": [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]},
        "force_max_n": 17.7,
        "slip_gain_per_mps": 2.2,
        "coast_speed_drag_n_per_mps": 0.9,
    }

    _, short_wheel = _LONG._wheel_acceleration_continuous(
        model, row, u=2.0, wheel=2.0, lateral_coupling=0.0)
    longer = dict(row, dt_sim_s=0.050)
    _, long_wheel = _LONG._wheel_acceleration_continuous(
        model, longer, u=2.0, wheel=2.0, lateral_coupling=0.0)

    assert long_wheel > short_wheel
    assert long_wheel == pytest.approx(2.0 * short_wheel - 2.0)


def test_longitudinal_parameters_select_continuous_wheel_dynamics():
    parameters = _PLANT.PlantParameters.from_longitudinal_parameters({
        "wheel_dynamics": {
            "kind": "identified_continuous_wheel_speed_derivative",
            "coefficients": [0.0] * 6,
        },
    })
    assert parameters.wheel_dynamics_kind == "continuous"


def test_required_recursive_horizons_include_source_step_and_750ms():
    assert 0.025 in _LONG.HORIZONS_S
    assert 0.75 in _LONG.HORIZONS_S
    assert 0.025 in _VEHICLE.HORIZONS_S
    assert 0.75 in _VEHICLE.HORIZONS_S


def test_recursive_scorer_reports_divergence_instead_of_raising():
    rows = []
    for index in range(4):
        rows.append({
            "x_k_m": 0.0,
            "y_k_m": 0.0,
            "yaw_k_rad": 0.0,
            "u_k_mps": 4.0,
            "v_k_mps": 0.0,
            "r_k_radps": 0.0,
            "dt_sim_s": 0.025,
            "applied_steering_rad_k1": 0.0,
            "applied_throttle_norm_k1": 1.0,
            "segment_id": 0.0,
            "x_k1_m": 0.1,
            "y_k1_m": 0.0,
            "yaw_k1_rad": 0.0,
            "u_k1_mps": 4.1,
            "v_k1_mps": 0.0,
            "r_k1_radps": 0.0,
        })
    # The cubic longitudinal term grows beyond the scorer envelope during the
    # recursive rollout. The report must retain the failure as evidence.
    models = _vehicle_models()
    models["u"] = {
        "coefficients": [0.0] * 13,
    }
    models["u"]["coefficients"][3] = 1.0e3
    result = _VEHICLE._recursive_scores({"unstable": rows}, models)
    short = result["0.05s"]
    assert short["divergent_origin_count"] > 0
    assert short["divergence_examples"]
    assert short["valid_origin_count"] < short["attempted_origin_count"]


def test_one_step_scorer_reports_divergence_instead_of_raising():
    row = {
        "x_k_m": 0.0,
        "y_k_m": 0.0,
        "yaw_k_rad": 0.0,
        "u_k_mps": 4.0,
        "v_k_mps": 0.0,
        "r_k_radps": 0.0,
        "dt_sim_s": 0.025,
        "applied_steering_rad_k1": 0.0,
        "applied_throttle_norm_k1": 1.0,
        "x_k1_m": 0.1,
        "y_k1_m": 0.0,
        "yaw_k1_rad": 0.0,
        "u_k1_mps": 4.1,
        "v_k1_mps": 0.0,
        "r_k1_radps": 0.0,
    }
    models = _vehicle_models()
    models["u"] = {"coefficients": [0.0] * 13}
    models["u"]["coefficients"][3] = 1.0e3
    result = _VEHICLE._one_step_scores({"unstable": [row]}, models)
    assert result["divergent_transition_count"] == 1
    assert result["valid_transition_count"] == 0
    assert result["divergence_examples"]


def test_longitudinal_recursive_scorer_reports_divergence_without_overflow():
    row = {
        "x_k_m": 0.0,
        "y_k_m": 0.0,
        "yaw_k_rad": 0.0,
        "u_k_mps": 4.0,
        "v_k_mps": 0.0,
        "r_k_radps": 0.0,
        "dt_sim_s": 0.025,
        "applied_steering_rad_k1": 0.0,
        "applied_throttle_norm_k1": 1.0,
        "segment_id": 0.0,
        "u_k1_mps": 4.1,
    }
    model = {"coefficients": [0.0] * 7}
    model["coefficients"][2] = 1.0e3
    result = _LONG._score_recursive({"unstable": [row] * 100}, "direct", model)
    short = result["0.05s"]
    assert short["divergent_origin_count"] > 0
    assert short["divergence_examples"]
    assert short["valid_origin_count"] < short["attempted_origin_count"]
