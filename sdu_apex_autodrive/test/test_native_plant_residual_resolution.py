"""Keep offline residual attribution tied to the reports named by the caller."""

import importlib.util
import json
import math
from pathlib import Path
import sys
from dataclasses import replace

import pytest


ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "analyze_native_plant_residuals",
    ROOT / "tools/model_id/analyze_native_plant_residuals.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_residual_resolution_uses_selected_longitudinal_report(tmp_path):
    lateral = {
        "candidate_comparison": {
            "Y2_tanh_fixed_iz": {
                "parameters": {"parameters": {
                    "iz_kgm2": 0.0961908,
                    "cf_n_per_rad": 2674.0,
                    "cr_n_per_rad": 4853.0,
                    "df_n": 12.6,
                    "dr_n": 14.1,
                    "tire_model": "tanh",
                }}
            }
        }
    }
    longitudinal = {
        "models": {"wheel_continuous": {"parameters": {
            "force_max_n": 19.035,
            "hard_brake_force_n": 19.084,
            "slip_gain_per_mps": 0.918,
            "coast_speed_drag_n_per_mps": 0.882,
            "wheel_dynamics": {
                "kind": "identified_continuous_wheel_speed_derivative",
                "coefficients": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        }}}
    }
    lateral_path = tmp_path / "lateral.json"
    longitudinal_path = tmp_path / "longitudinal.json"
    lateral_path.write_text(json.dumps(lateral), encoding="utf-8")
    longitudinal_path.write_text(json.dumps(longitudinal), encoding="utf-8")

    parameters = _MODULE._resolve_parameters(
        lateral_path, longitudinal_path, "Y2_tanh_fixed_iz",
        "wheel_continuous", "unity_measured")

    assert parameters.force_max_n == 19.035
    assert parameters.slip_gain_per_mps == 0.918
    assert parameters.wheel_coefficients == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert parameters.coast_speed_drag_n_per_mps == 0.0
    assert parameters.linear_damping_per_s == 0.273


def test_residual_attribution_uses_discrete_steering_ramp():
    parameters = _MODULE.PlantParameters()
    previous = {
        "segment_id": 0,
        "wheel_speed_mps_k1": 6.29,
        "simulator_feedback_steering_rad_k1": 0.0785,
    }
    current = {
        "segment_id": 0,
        "dt_sim_s": 0.025,
        "x_k_m": 0.0,
        "y_k_m": 0.0,
        "yaw_k_rad": 0.0,
        "u_k_mps": 6.05,
        "v_k_mps": 0.1825,
        "r_k_radps": 1.4017,
        "applied_throttle_norm_k1": 0.25,
        "applied_steering_rad_k1": 0.0,
        "simulator_feedback_steering_rad_k1": 0.0,
        "wheel_speed_mps_k1": 6.29,
        "u_k1_mps": 6.05,
        "v_k1_mps": 0.1825,
        "r_k1_radps": 1.4017,
        "simulation_time_k_s": 0.025,
    }
    result = _MODULE._residual_rows("reversal", [previous, current], parameters)
    assert len(result) == 1

    expected = _MODULE.step(
        _MODULE.np.asarray([
            current["x_k_m"], current["y_k_m"], current["yaw_k_rad"],
            current["u_k_mps"], current["v_k_mps"], current["r_k_radps"],
            previous["simulator_feedback_steering_rad_k1"],
            previous["wheel_speed_mps_k1"],
        ], dtype=float),
        0.0, current["applied_throttle_norm_k1"], current["dt_sim_s"],
        parameters)
    expected_r_dot = (expected[5] - current["r_k_radps"]) / current["dt_sim_s"]
    assert result[0]["model_dr_radps2"] == pytest.approx(expected_r_dot)
    assert math.isfinite(result[0]["residual_dr_radps2"])


def test_instantaneous_steering_uses_reported_applied_angle():
    parameters = replace(
        _MODULE.PlantParameters(), steering_dynamics_kind="instantaneous")
    state = _MODULE.np.asarray(
        [0.0, 0.0, 0.0, 8.0, 0.0, 0.0, -0.1047, 8.0], dtype=float)
    next_state = _MODULE.step(state, 0.0, 1.0, 0.025, parameters)
    assert next_state[6] == pytest.approx(0.0)


def test_residual_attribution_can_score_source_steering_transition():
    parameters = replace(
        _MODULE.PlantParameters(), steering_dynamics_kind="instantaneous")
    previous = {
        "segment_id": 0,
        "wheel_speed_mps_k1": 19.0,
        "simulator_feedback_steering_rad_k1": -0.1047,
    }
    current = {
        "segment_id": 0,
        "dt_sim_s": 0.025,
        "x_k_m": 0.0,
        "y_k_m": 0.0,
        "yaw_k_rad": 0.0,
        "u_k_mps": 18.0,
        "v_k_mps": 0.0,
        "r_k_radps": -0.3,
        "applied_throttle_norm_k1": 0.8,
        "applied_steering_rad_k1": 0.0,
        "simulator_feedback_steering_rad_k1": 0.0,
        "wheel_speed_mps_k1": 19.0,
        "u_k1_mps": 18.0,
        "v_k1_mps": 0.0,
        "r_k1_radps": -0.3,
        "simulation_time_k_s": 0.025,
    }
    result = _MODULE._residual_rows("instantaneous", [previous, current], parameters)
    assert len(result) == 1
    assert result[0]["model_steering_rate_radps"] == pytest.approx(
        0.1047 / 0.025)
