#!/usr/bin/env python3
"""Run the handoff's A0--A6 model ablation.

This is an offline attribution experiment.  It does not fit new parameters,
modify the simulator, or change the production MPC.  Every variant is scored
with the same run split, raceline CORE/GUARD envelope, causal rollout, and
0.10/0.25/0.50/0.75 s horizons.  The 0.75 s result is the MPC-relevant gate:
30 commands at 40 Hz.

The sequence is deliberately additive so that an improvement can be assigned
to one model component:

* A0: scalar canonical baseline;
* A1: A0 plus fitted longitudinal speed profiles;
* A2: A1 with steering semantics varied independently;
* A3: A2-instantaneous plus the fitted steering-transition residual;
* A4: A2 with fitted lateral regimes/combined slip;
* A5: A4 plus the steering-transition residual;
* A6: A2 with one fitted common lateral speed-scale coefficient.

A6 is fitted only when A4 shows a repeatable held-out benefit over the
first-order A2 steering-semantic variant.  The one-parameter structure is
deliberately constrained: it scales both lateral stiffness and peak capacity
by the same smooth speed factor, so it cannot independently retune front,
rear, stiffness, peak, and combined-slip terms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structured_vehicle_plant as plant  # noqa: E402
from fit_speed_regime_vehicle_model import (  # noqa: E402
    HORIZONS_S,
    MAX_SPEED_MPS,
    _load_runs,
    _raceline_context,
    _score,
    _steering_command,
    replace_for_baseline,
)


def _candidate_parameters(
        report: dict[str, Any], variant: str,
        a6_speed_scale_gain: float | None = None) -> plant.PlantParameters:
    """Construct one ablation plant without silently mixing components."""
    base = plant.PlantParameters.from_manifest()
    baseline = replace_for_baseline(base)
    values = report["candidate_parameters"]
    fitted_longitudinal = {
        "force_max_regimes_n": tuple(
            float(value) for value in values["force_max_regimes_n"]),
        "slip_gain_regimes_per_mps": tuple(
            float(value) for value in values["slip_gain_regimes_per_mps"]),
        "coast_speed_drag_n_per_mps": 0.0,
    }
    fitted_lateral = {
        "tire_model": "regime_speed_combined_tanh",
        "lateral_cf_regimes_n_per_rad": tuple(
            float(value) for value in values[
                "lateral_cf_regimes_n_per_rad"]),
        "lateral_cr_regimes_n_per_rad": tuple(
            float(value) for value in values[
                "lateral_cr_regimes_n_per_rad"]),
        "lateral_df_regimes_n": tuple(
            float(value) for value in values["lateral_df_regimes_n"]),
        "lateral_dr_regimes_n": tuple(
            float(value) for value in values["lateral_dr_regimes_n"]),
        "combined_slip_gain": float(values["combined_slip_gain"]),
    }
    fitted_steering = {
        "steering_dynamics_kind": str(values["steering_dynamics_kind"]),
        "steering_lag_time_constant_s": float(
            values.get("steering_lag_time_constant_s", 0.0)),
    }
    fitted_residual = {
        "steering_rate_force_gain_n_per_radps": float(
            values.get("steering_rate_force_gain_n_per_radps", 0.0)),
        "steering_rate_moment_gain_nm_per_radps": float(
            values.get("steering_rate_moment_gain_nm_per_radps", 0.0)),
    }

    if variant == "A0_scalar_baseline":
        return baseline

    # A1 and all A2 variants intentionally retain the canonical scalar
    # lateral law.  This makes the steering-semantics comparison independent
    # of the lateral-profile comparison.
    if variant == "A1_longitudinal_only":
        return plant.PlantParameters(**{
            **baseline.__dict__,
            **fitted_longitudinal,
        })
    if variant.startswith("A2_steering_"):
        steering_kind = variant.removeprefix("A2_steering_")
        if steering_kind not in {"rate_limited", "first_order", "instantaneous"}:
            raise ValueError(f"unsupported A2 steering variant: {variant}")
        steering = {
            "steering_dynamics_kind": steering_kind,
            "steering_lag_time_constant_s": (
                fitted_steering["steering_lag_time_constant_s"]
                if steering_kind == "first_order" else 0.0),
        }
        return plant.PlantParameters(**{
            **baseline.__dict__,
            **fitted_longitudinal,
            **steering,
        })
    if variant == "A3_steering_transition_residual":
        return plant.PlantParameters(**{
            **baseline.__dict__,
            **fitted_longitudinal,
            **fitted_steering,
            **fitted_residual,
        })
    if variant == "A4_fitted_lateral":
        return plant.PlantParameters(**{
            **baseline.__dict__,
            **fitted_longitudinal,
            **fitted_lateral,
            # The lateral fit must be evaluated with the steering semantics
            # under which its coefficients were identified. For the current
            # candidate this is first-order steering; for the earlier
            # instantaneous candidate it remains instantaneous.
            **fitted_steering,
        })
    if variant == "A5_full_current":
        return plant.PlantParameters(**{
            **baseline.__dict__,
            **fitted_longitudinal,
            **fitted_lateral,
            **fitted_steering,
            **fitted_residual,
        })
    if variant == "A6_minimal_lateral_speed_scale":
        if a6_speed_scale_gain is None:
            raise ValueError("A6 requires the fitted speed-scale gain")
        return plant.PlantParameters(**{
            **baseline.__dict__,
            **fitted_longitudinal,
            **fitted_steering,
            "tire_model": "speed_combined_tanh",
            "lateral_speed_stiffness_gain": float(a6_speed_scale_gain),
            "lateral_speed_peak_gain": float(a6_speed_scale_gain),
            "combined_slip_gain": 0.0,
        })
    raise ValueError(f"unknown ablation variant: {variant}")


VARIANTS = (
    "A0_scalar_baseline",
    "A1_longitudinal_only",
    "A2_steering_rate_limited",
    "A2_steering_first_order",
    "A2_steering_instantaneous",
    "A3_steering_transition_residual",
    "A4_fitted_lateral",
    "A5_full_current",
)


def _a4_has_repeatable_benefit(scores: dict[str, dict[str, Any]]) -> bool:
    """Require A4 to improve the relevant held-out runs before fitting A6."""
    a2 = scores["A2_steering_first_order"]["0.75s"]
    a4 = scores["A4_fitted_lateral"]["0.75s"]
    improved_metrics = 0
    for operating_class in ("core", "guard"):
        for field in ("e_cross_m", "e_heading_rad", "v_mps", "r_radps"):
            a2_value = a2[operating_class][field]["p95"]
            a4_value = a4[operating_class][field]["p95"]
            if (a2_value is not None and a4_value is not None and
                    a4_value < a2_value):
                improved_metrics += 1
    if improved_metrics < 4:
        return False

    # The benefit must not be confined to a single validation run.  Require
    # the lateral/yaw errors to improve in each substantial validation run.
    for run_name, a2_values in a2["per_run"].items():
        a4_values = a4["per_run"].get(run_name)
        if (a4_values is None or a2_values["r_radps"]["p95"] is None or
                a2_values["r_radps"]["count"] < 100):
            continue
        if (a4_values["r_radps"]["p95"] >=
                a2_values["r_radps"]["p95"]):
            return False
    return True


def _fit_a6_speed_scale(
        train_runs: dict[str, list[dict[str, float]]],
        baseline: plant.PlantParameters,
        fitted_longitudinal: dict[str, Any],
        fitted_steering: dict[str, Any]) -> dict[str, Any]:
    """Fit one common lateral force scale over causal 0.75 s train windows."""
    parameters_base = plant.PlantParameters(**{
        **baseline.__dict__,
        **fitted_longitudinal,
        **fitted_steering,
        "tire_model": "speed_combined_tanh",
        "combined_slip_gain": 0.0,
    })
    windows: list[list[dict[str, float]]] = []
    for rows in train_runs.values():
        for origin in range(0, len(rows), 8):
            first = rows[origin]
            segment = int(first["segment_id"])
            if (not math.isfinite(first["wheel_k_mps"]) or
                    not math.isfinite(first["delta_k_rad"]) or
                    not (2.0 <= first["u_k_mps"] <= MAX_SPEED_MPS)):
                continue
            window: list[dict[str, float]] = []
            elapsed = 0.0
            index = origin
            while index < len(rows) and elapsed < 0.75:
                row = rows[index]
                if (int(row["segment_id"]) != segment or
                        not (2.0 <= row["u_k_mps"] <= MAX_SPEED_MPS) or
                        not math.isfinite(row["wheel_k_mps"]) or
                        not math.isfinite(row["delta_k_rad"])):
                    window = []
                    break
                window.append(row)
                elapsed += row["dt_sim_s"]
                index += 1
            if window and elapsed >= 0.75 - 1.0e-10:
                windows.append(window)
                if len(windows) >= 240:
                    break
        if len(windows) >= 240:
            break
    if len(windows) < 8:
        raise ValueError(f"insufficient A6 fitting windows: {len(windows)}")

    def make_parameters(gain: float) -> plant.PlantParameters:
        return plant.PlantParameters(**{
            **parameters_base.__dict__,
            "lateral_speed_stiffness_gain": gain,
            "lateral_speed_peak_gain": gain,
        })

    def residual(values: np.ndarray) -> np.ndarray:
        parameters = make_parameters(float(values[0]))
        output: list[float] = []
        for window in windows:
            current_v = window[0]["v_k_mps"]
            current_r = window[0]["r_k_radps"]
            current_yaw = window[0]["yaw_k_rad"]
            elapsed = 0.0
            horizon_index = 0
            for row in window:
                previous_r = current_r
                current_v, current_r = plant.lateral_body_step(
                    row["u_k_mps"], current_v, current_r,
                    row["delta_k_rad"],
                    _steering_command(row) * plant.MAX_STEERING_RAD,
                    row["dt_sim_s"], parameters,
                    wheel=row["wheel_k_mps"],
                    throttle=row["applied_throttle_norm_k1"])
                current_yaw += 0.5 * (previous_r + current_r) * row["dt_sim_s"]
                elapsed += row["dt_sim_s"]
                while (horizon_index < len((0.25, 0.50, 0.75)) and
                       elapsed + 1.0e-9 >= (0.25, 0.50, 0.75)[horizon_index]):
                    heading_error = ((current_yaw - row["yaw_k1_rad"] +
                                      math.pi) % (2.0 * math.pi) - math.pi)
                    output.extend([
                        (current_v - row["v_k1_mps"]) / 0.10,
                        (current_r - row["r_k1_radps"]) / 0.20,
                        heading_error / 0.10,
                    ])
                    horizon_index += 1
                if horizon_index == 3:
                    break
        output.append(0.05 * float(values[0]))
        return np.asarray(output, dtype=float)

    result = least_squares(
        residual, np.asarray([0.0]), bounds=(np.asarray([-0.9]),
                                             np.asarray([2.0])),
        loss="soft_l1", f_scale=1.0, max_nfev=80)
    gain = float(result.x[0])
    return {
        "status": "fitted_on_training_windows",
        "parameter_count": 1,
        "speed_scale_gain": gain,
        "speed_scale_definition": "1 + gain * abs(u) / 16 m/s",
        "training_windows": len(windows),
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
        },
    }

CORE_075S_GATES = {
    "e_cross_m": 0.05,
    "e_heading_rad": 0.05,
    "u_mps": 0.20,
    "v_mps": 0.08,
    "r_radps": 0.12,
}


def _metric_vector(score: dict[str, Any]) -> tuple[float, ...]:
    """Return the fixed 0.75 s CORE/GUARD gate vector for Pareto checks."""
    horizon = score["0.75s"]
    values: list[float] = []
    for operating_class in ("core", "guard"):
        bucket = horizon[operating_class]
        for field in ("e_cross_m", "e_heading_rad", "u_mps", "v_mps",
                      "r_radps"):
            value = bucket[field]["p95"]
            values.append(float(value) if value is not None else float("inf"))
    return tuple(values)


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right))


def _pareto_front(scores: dict[str, dict[str, Any]]) -> list[str]:
    vectors = {name: _metric_vector(score) for name, score in scores.items()}
    return [
        name for name, vector in vectors.items()
        if not any(
            other != name and _dominates(vectors[other], vector)
            for other in vectors)
    ]


def _summary(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, score in scores.items():
        horizon = score["0.75s"]
        output[name] = {
            operating_class: {
                field: horizon[operating_class][field]["p95"]
                for field in ("e_cross_m", "e_heading_rad", "u_mps",
                              "v_mps", "r_radps")
            }
            for operating_class in ("core", "guard")
        }
    return output


def _passes_core_gate(score: dict[str, Any]) -> bool:
    bucket = score["0.75s"]["core"]
    return all(bucket[field]["p95"] is not None and
               bucket[field]["p95"] <= limit
               for field, limit in CORE_075S_GATES.items())


def run(candidate_json: Path, output: Path, include_train: bool = False,
        origin_stride: int = 1) -> dict[str, Any]:
    report = json.loads(candidate_json.read_text(encoding="utf-8"))
    if report.get("primary_horizon_commands") != 30 or abs(
            float(report.get("primary_horizon_command_dt_s", 0.0)) - 0.025) > 1e-12:
        raise ValueError("candidate report is not on the active N30/40 Hz contract")
    validation_specs = report["validation_specs"]
    train_specs = report["train_specs"]
    context_path = Path(report["operating_envelope"]["raceline_file"])
    context = _raceline_context(context_path)
    validation_runs = _load_runs(validation_specs)
    train_runs = _load_runs(train_specs) if include_train else None

    scores: dict[str, dict[str, Any]] = {}
    train_scores: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        parameters = _candidate_parameters(report, variant)
        scores[variant] = _score(
            validation_runs, parameters, origin_stride, context)
        if train_runs is not None:
            train_scores[variant] = _score(
                train_runs, parameters, origin_stride, context)

    # A6 is intentionally trained only after the fixed ablation demonstrates
    # a repeatable A4 benefit.  It is not allowed to use validation data for
    # coefficient selection.  The report's normal train split is therefore
    # always loaded for this conditional fit, even when train scoring was not
    # requested in the output.
    a6_fit: dict[str, Any] | None = None
    if _a4_has_repeatable_benefit(scores):
        if train_runs is None:
            train_runs = _load_runs(train_specs)
        base = plant.PlantParameters.from_manifest()
        baseline = replace_for_baseline(base)
        values = report["candidate_parameters"]
        fitted_longitudinal = {
            "force_max_regimes_n": tuple(
                float(value) for value in values["force_max_regimes_n"]),
            "slip_gain_regimes_per_mps": tuple(
                float(value) for value in values["slip_gain_regimes_per_mps"]),
            "coast_speed_drag_n_per_mps": 0.0,
        }
        fitted_steering = {
            "steering_dynamics_kind": str(values["steering_dynamics_kind"]),
            "steering_lag_time_constant_s": float(
                values.get("steering_lag_time_constant_s", 0.0)),
        }
        a6_fit = _fit_a6_speed_scale(
            train_runs, baseline, fitted_longitudinal,
            fitted_steering)
        a6_gain = float(a6_fit["speed_scale_gain"])
        a6_parameters = _candidate_parameters(
            report, "A6_minimal_lateral_speed_scale",
            a6_speed_scale_gain=a6_gain)
        scores["A6_minimal_lateral_speed_scale"] = _score(
            validation_runs, a6_parameters, origin_stride, context)
        if train_runs is not None and include_train:
            train_scores["A6_minimal_lateral_speed_scale"] = _score(
                train_runs, a6_parameters, origin_stride, context)

    variant_names = list(VARIANTS)
    if a6_fit is not None:
        variant_names.append("A6_minimal_lateral_speed_scale")

    pareto = _pareto_front(scores)
    gate_eligible = [name for name, score in scores.items()
                     if _passes_core_gate(score)]
    eligible_scores = {name: scores[name] for name in gate_eligible}
    eligible_pareto = _pareto_front(eligible_scores) if eligible_scores else []
    complexity = {name: index for index, name in enumerate(VARIANTS)}
    simplest = (min(eligible_pareto, key=lambda name: complexity[name])
                if eligible_pareto else None)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_fixed_parameter_ablation_not_runtime_validated",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "candidate_source": str(candidate_json),
        "raceline_operating_envelope": report["operating_envelope"],
        "validation_specs": list(validation_specs),
        "train_specs": list(train_specs),
        "train_scored": include_train,
        "origin_stride": origin_stride,
        "horizons_s": list(HORIZONS_S),
        "primary_horizon": {
            "commands": 30,
            "dt_s": 0.025,
            "physical_horizon_s": 0.75,
        },
        "variants": {
            name: {
                "parameters": (
                    _candidate_parameters(
                        report, name,
                        a6_speed_scale_gain=(
                            float(a6_fit["speed_scale_gain"])
                            if a6_fit is not None else None))
                    .__dict__),
                "validation": scores[name],
                **({"train": train_scores[name]} if include_train else {}),
            }
            for name in variant_names
        },
        "validation_p95_summary_0.75s": _summary(scores),
        "pareto": {
            "metric": "CORE and GUARD p95 of e_cross, e_heading, u, v, r at 0.75 s",
            "front": pareto,
            "provisional_core_gate_0.75s": CORE_075S_GATES,
            "gate_eligible": gate_eligible,
            "gate_eligible_front": eligible_pareto,
            "simplest_front_candidate": simplest,
            "freeze_status": ("candidate_ready_for_next_gate" if simplest else
                               "no_candidate_meets_provisional_core_gate"),
            "runtime_migration": False,
        },
        "A6": {
            **(a6_fit or {
                "status": "reserved",
                "reason": (
                    "A4 did not show a repeatable benefit over the first-order "
                    "A2 steering variant; no A6 fit was permitted"),
            }),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-train", action="store_true",
                        help="also score the full training split")
    parser.add_argument("--origin-stride", type=int, default=1)
    args = parser.parse_args()
    result = run(args.candidate, args.output, args.include_train,
                 args.origin_stride)
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "pareto": result["pareto"],
        "validation_p95_summary_0.75s": result[
            "validation_p95_summary_0.75s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
