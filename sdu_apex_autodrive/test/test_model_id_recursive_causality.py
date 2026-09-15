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


def test_longitudinal_parameters_select_discrete_wheel_dynamics():
    parameters = _PLANT.PlantParameters.from_longitudinal_parameters({
        "wheel_dynamics": {
            "kind": "identified_discrete_wheel_speed_state",
            "coefficients": [0.0] * 6,
        },
    })
    assert parameters.wheel_dynamics_kind == "discrete"


def test_unknown_wheel_dynamics_kind_is_rejected():
    with pytest.raises(ValueError, match="unsupported identified wheel dynamics"):
        _PLANT.PlantParameters.from_longitudinal_parameters({
            "wheel_dynamics": {"kind": "accidentally_ambiguous"},
        })


def test_first_order_effective_steering_is_causal_and_bounded():
    parameters = _PLANT.PlantParameters(
        steering_dynamics_kind="first_order",
        steering_lag_time_constant_s=0.050,
    )
    start = _PLANT._steering_next(0.0, 0.4, 0.025, parameters)
    end = _PLANT._steering_next(start, 0.4, 0.025, parameters)
    assert 0.0 < start < 0.4
    assert start < end < 0.4


def test_regime_profile_is_continuous_at_speed_boundaries():
    values = (1.0, 2.0, 3.0)
    left = _PLANT._speed_regime_value(values, 7.999, 2.0)
    right = _PLANT._speed_regime_value(values, 8.001, 2.0)
    assert left == pytest.approx(right, abs=0.002)
    assert _PLANT._speed_regime_value(values, 2.0, 2.0) == pytest.approx(1.0)
    assert _PLANT._speed_regime_value(values, 16.0, 2.0) == pytest.approx(3.0)


def test_regime_parameters_are_used_by_lateral_and_longitudinal_force():
    parameters = _PLANT.PlantParameters(
        force_max_n=1.0,
        slip_gain_per_mps=1.0,
        force_max_regimes_n=(10.0, 20.0, 30.0),
        slip_gain_regimes_per_mps=(1.0, 1.0, 1.0),
        cf_n_per_rad=1.0,
        cr_n_per_rad=1.0,
        df_n=2.0,
        dr_n=2.0,
        lateral_cf_regimes_n_per_rad=(10.0, 20.0, 30.0),
        lateral_cr_regimes_n_per_rad=(10.0, 20.0, 30.0),
        lateral_df_regimes_n=(2.0, 3.0, 4.0),
        lateral_dr_regimes_n=(2.0, 3.0, 4.0),
        tire_model="regime_speed_combined_tanh",
    )
    low = _PLANT.lateral_forces(5.0, 0.0, 0.0, 0.1, parameters)
    high = _PLANT.lateral_forces(14.0, 0.0, 0.0, 0.1, parameters)
    assert abs(high[0]) > abs(low[0])
    low_body = _PLANT._body_derivative(
        5.0, 0.0, 0.0, 0.0, 6.0, parameters, throttle=1.0)
    high_body = _PLANT._body_derivative(
        14.0, 0.0, 0.0, 0.0, 15.0, parameters, throttle=1.0)
    assert high_body[0] > low_body[0]


def test_steering_transition_residual_uses_only_causal_steering_rate():
    base = _PLANT.PlantParameters()
    corrected = _PLANT.PlantParameters(
        steering_rate_force_gain_n_per_radps=2.0,
        steering_rate_moment_gain_nm_per_radps=0.5,
    )
    base_body = _PLANT._body_derivative(
        8.0, 0.1, 0.2, 0.05, 8.2, base, throttle=0.4,
        steering_rate_radps=0.0)
    corrected_body = _PLANT._body_derivative(
        8.0, 0.1, 0.2, 0.05, 8.2, corrected, throttle=0.4,
        steering_rate_radps=0.0)
    np.testing.assert_allclose(corrected_body, base_body)
    moving_body = _PLANT._body_derivative(
        8.0, 0.1, 0.2, 0.05, 8.2, corrected, throttle=0.4,
        steering_rate_radps=1.0)
    assert moving_body[1] > base_body[1]
    assert moving_body[2] > base_body[2]


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
