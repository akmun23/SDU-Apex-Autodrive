#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import casadi as ca
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OPTIMIZER_ROOT = ROOT / "SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3"
sys.path.insert(0, str(OPTIMIZER_ROOT))

from f1tenth_planning.autodrive_mintime.model import VehicleModel, load_yaml  # noqa: E402
from f1tenth_planning.autodrive_mintime.optimizer import (  # noqa: E402
    _steering_robustness_config,
    _steering_robustness_node_cost,
    _yaw_output_casadi,
    _yaw_output_numeric,
    _yaw_transition_terms_casadi,
    _yaw_transition_terms_numeric,
)


def main() -> int:
    config = load_yaml(ROOT / "config/racing/practice_y1_phase_response_optimizer_candidate.yaml")
    model = VehicleModel.from_repo(ROOT, config)
    speed, steering, curvature = 3.0, -0.35, -0.7
    expected_tau = {
        "turn-in": ( -0.3, 0.0711231428),
        "hold": (0.0, 0.1293144252),
        "unwind": (0.3, 0.4011282334),
    }
    steady_values = []
    for phase, (steering_rate, tau_expected) in expected_tau.items():
        numeric = _yaw_transition_terms_numeric(
            model, config, speed, steering, curvature, steering_rate)
        u, delta, kappa, rate = ca.SX.sym("u"), ca.SX.sym("d"), ca.SX.sym("k"), ca.SX.sym("rate")
        steady, tau = _yaw_transition_terms_casadi(
            model, config, u, delta, kappa, rate)
        evaluate = ca.Function("phase_response", [u, delta, kappa, rate], [steady, tau])
        symbolic = np.asarray(evaluate(speed, steering, curvature, steering_rate)).reshape(-1)
        assert abs(numeric[1] - tau_expected) < 1.0e-10, (phase, numeric[1])
        assert np.allclose(numeric, symbolic, rtol=1.0e-10, atol=1.0e-10), (phase, numeric, symbolic)
        steady_values.append(numeric[0])

    assert max(steady_values) - min(steady_values) < 1.0e-10

    disabled = {**config, "yaw_phase_response": {"enabled": False}}
    default = _yaw_transition_terms_numeric(
        model, disabled, speed, steering, curvature, -0.3)
    expected_default = model.yaw_transition_terms_numeric(
        speed, steering, curvature, -0.3)
    assert np.allclose(default, expected_default, rtol=0.0, atol=1.0e-12)

    robustness = _steering_robustness_config({
        "steering_robustness": {
            "enabled": True,
            "perturbation_rad": 0.04,
            "yaw_response_weight_s_per_radps2": 0.05,
        },
    }, model.max_steering_rad)
    assert robustness == {
        "perturbation_rad": 0.04,
        "yaw_response_weight_s_per_radps2": 0.05,
    }
    assert _steering_robustness_config({}, model.max_steering_rad) is None
    assert _steering_robustness_config({
        "steering_robustness": {"enabled": False},
    }, model.max_steering_rad) is None
    penalty = _steering_robustness_node_cost(-1.0, -1.2, -0.8, 0.05)
    assert abs(penalty - 0.002) < 1.0e-12

    u, r, delta, qdelta, physical_rate, kappa, ax, qv, target = [
        ca.MX.sym(name) for name in
        ("u", "r", "delta", "qdelta", "physical_rate", "kappa",
         "ax", "qv", "target")]
    yaw_output = _yaw_output_casadi(
        model, config, u, r, delta, qdelta, physical_rate, kappa, ax, qv,
        target)
    evaluate_output = ca.Function(
        "yaw_output_parity", [u, r, delta, qdelta, physical_rate, kappa,
                              ax, qv, target], [yaw_output])
    output_values = (speed, -0.8, steering, -0.3, -0.2, curvature,
                     0.7, 0.1, 3.0)
    numeric_output = _yaw_output_numeric(model, config, *output_values)
    symbolic_output = float(evaluate_output(*output_values))
    assert abs(numeric_output - symbolic_output) < 1.0e-10
    print("PASS: phase-specific optimizer yaw response matches numeric/CasADi and preserves default behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
