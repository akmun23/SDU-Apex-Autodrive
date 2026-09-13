"""Regression tests for the model-identification anti-leak contract."""

import importlib.util

import numpy as np


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LONG = _load("fit_longitudinal_models", "tools/model_id/fit_longitudinal_models.py")
_VEHICLE = _load("fit_vehicle_model", "tools/model_id/fit_vehicle_model.py")


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


def test_required_recursive_horizons_include_source_step_and_750ms():
    assert 0.025 in _LONG.HORIZONS_S
    assert 0.75 in _LONG.HORIZONS_S
    assert 0.025 in _VEHICLE.HORIZONS_S
    assert 0.75 in _VEHICLE.HORIZONS_S
