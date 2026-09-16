#!/usr/bin/env python3
"""Score an offline Unity-wheel candidate on simulator-trace transitions.

This benchmark is not a live-input-compatible replay: each rollout origin
uses simulator-trace pose/twist/steering, and the latent wheel rates are
initialized from encoder-derived transition fields. Those values are useful
for diagnosing the plant, but they are not the production MPC input contract.
Future measured states are endpoint targets only within each rollout. No
simulator, ROS node, or production MPC file is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_simulator_native_model import (  # noqa: E402
    HORIZONS_S,
    _horizon_key,
    _read_runs,
    _score_origins,
    _stats,
)
from simulator_native_model import (  # noqa: E402
    NativeModelParameters,
    parameters_from_dump,
    step_unity_wheel_collider,
    step_unity_wheel_collider_with_suspension,
)

IDENTIFIED_SPEED_CEILING_MPS = 16.0
DRAG_INTERPRETATIONS = (
    "serialized_force_coefficient",
    "unity_velocity_decay_rate",
)
ORIGIN_SPEED_BANDS_MPS = (
    ("0_4mps", 0.0, 4.0),
    ("4_8mps", 4.0, 8.0),
    ("8_12mps", 8.0, 12.0),
    ("12_16mps", 12.0, IDENTIFIED_SPEED_CEILING_MPS),
)


def _state_from_row(row: dict[str, float], parameters: NativeModelParameters,
                    suspension_state: bool) -> np.ndarray:
    """Build an offline trace-state origin, including wheel-rate telemetry."""
    radius = (parameters.wheel_radius_m if parameters.wheel_radius_m > 0.0
              else 0.059)
    left = float(row["wheel_left_k_mps"]) / radius
    right = float(row["wheel_right_k_mps"]) / radius
    base = [
        row["x_k_m"], row["y_k_m"], row["yaw_k_rad"], row["u_k_mps"],
        row["v_k_mps"], row["r_k_radps"], row["delta_k_rad"], left, right,
        left, right,
    ]
    if suspension_state:
        base.extend((0.0,) * 6)
    return np.asarray(base, dtype=float)


def _heading_error(predicted: float, target: float) -> float:
    return (predicted - target + math.pi) % (2.0 * math.pi) - math.pi


def _parameters_with_drag_interpretation(
        parameters: NativeModelParameters, interpretation: str
        ) -> NativeModelParameters:
    """Map Unity's serialized drag value onto the plant's force coefficient.

    ``step_unity_wheel_collider`` subtracts ``coefficient * velocity`` from
    force and subsequently divides by mass.  The current serialized-value
    path therefore damps at ``drag / mass``.  The alternative treats Unity's
    Rigidbody drag as a velocity-decay rate and multiplies by mass so the
    resulting acceleration is ``-drag * velocity``.  This is an explicit
    offline interpretation candidate, not a production-model change.
    """
    if interpretation not in DRAG_INTERPRETATIONS:
        raise ValueError(f"unsupported Rigidbody drag interpretation: {interpretation}")
    if interpretation == "serialized_force_coefficient":
        return parameters
    if parameters.rigid_body_drag_per_s is None:
        raise ValueError("Unity drag interpretation requires a dumped Rigidbody drag")
    return replace(
        parameters,
        rigid_body_drag_per_s=(parameters.rigid_body_drag_per_s *
                               parameters.mass_kg),
    )


def _error_vector(state: np.ndarray, row: dict[str, float]) -> dict[str, float]:
    return {
        "x_m": float(state[0] - row["x_k1_m"]),
        "y_m": float(state[1] - row["y_k1_m"]),
        "position_m": float(math.hypot(
            state[0] - row["x_k1_m"], state[1] - row["y_k1_m"])),
        "heading_rad": _heading_error(float(state[2]), row["yaw_k1_rad"]),
        "u_mps": float(state[3] - row["u_k1_mps"]),
        "v_mps": float(state[4] - row["v_k1_mps"]),
        "r_radps": float(state[5] - row["r_k1_radps"]),
    }


def _score_runs(runs: dict[str, list[dict[str, float]]],
                parameters: NativeModelParameters,
                max_origins_per_run: int,
                suspension_state: bool) -> dict[str, Any]:
    fields = (
        "x_m", "y_m", "position_m", "heading_rad", "u_mps", "v_mps",
        "r_radps")

    def empty_errors() -> dict[float, dict[str, list[float]]]:
        return {
            horizon: {field: [] for field in fields}
            for horizon in HORIZONS_S
        }

    errors_by_horizon = empty_errors()
    errors_by_run: dict[str, dict[float, dict[str, list[float]]]] = {}
    errors_within_envelope = empty_errors()
    errors_within_envelope_by_run: dict[
        str, dict[float, dict[str, list[float]]]] = {}
    errors_by_speed_band = {
        name: empty_errors() for name, _, _ in ORIGIN_SPEED_BANDS_MPS
    }
    errors_by_speed_band_and_run: dict[
        str, dict[str, dict[float, dict[str, list[float]]]]] = {
            name: {} for name, _, _ in ORIGIN_SPEED_BANDS_MPS
        }
    origins_used: dict[str, int] = {}
    for run_name, rows in runs.items():
        run_errors = empty_errors()
        errors_by_run[run_name] = run_errors
        run_envelope_errors = empty_errors()
        errors_within_envelope_by_run[run_name] = run_envelope_errors
        for band_name, _, _ in ORIGIN_SPEED_BANDS_MPS:
            errors_by_speed_band_and_run[band_name][run_name] = empty_errors()
        origins = [
            index for index in _score_origins(rows, max_origins_per_run)
            if math.isfinite(rows[index]["wheel_left_k_mps"])
            and math.isfinite(rows[index]["wheel_right_k_mps"])
            and math.isfinite(rows[index]["delta_k_rad"])
        ]
        origins_used[run_name] = len(origins)
        for origin in origins:
            first = rows[origin]
            segment = int(first["segment_id"])
            origin_speed = abs(float(first["u_k_mps"]))
            within_speed_envelope = origin_speed <= (
                IDENTIFIED_SPEED_CEILING_MPS + 1.0e-9)
            origin_band = next((
                name for name, lower, upper in ORIGIN_SPEED_BANDS_MPS
                if lower <= origin_speed < upper or
                (upper == IDENTIFIED_SPEED_CEILING_MPS and
                 origin_speed <= upper + 1.0e-9)
            ), None)
            state = _state_from_row(first, parameters, suspension_state)
            elapsed = 0.0
            index = origin
            horizon_index = 0
            while (index < len(rows) and horizon_index < len(HORIZONS_S)
                   and elapsed < HORIZONS_S[-1] - 1.0e-10):
                row = rows[index]
                if int(row["segment_id"]) != segment:
                    break
                within_speed_envelope = (
                    within_speed_envelope and
                    abs(float(row["u_k_mps"])) <=
                    IDENTIFIED_SPEED_CEILING_MPS + 1.0e-9 and
                    abs(float(row["u_k1_mps"])) <=
                    IDENTIFIED_SPEED_CEILING_MPS + 1.0e-9)
                if suspension_state:
                    state = step_unity_wheel_collider_with_suspension(
                        state,
                        row["transition_steering_norm_k"],
                        row["transition_throttle_norm_k"], row["dt_sim_s"],
                        parameters)
                else:
                    state = step_unity_wheel_collider(
                        state,
                        row["transition_steering_norm_k"],
                        row["transition_throttle_norm_k"], row["dt_sim_s"],
                        parameters)
                elapsed += row["dt_sim_s"]
                index += 1
                while (horizon_index < len(HORIZONS_S) and
                       elapsed >= HORIZONS_S[horizon_index] - 1.0e-10):
                    target = rows[index - 1]
                    errors = errors_by_horizon[HORIZONS_S[horizon_index]]
                    run_horizon_errors = run_errors[
                        HORIZONS_S[horizon_index]]
                    error_values = _error_vector(state, target)
                    envelope_horizon_errors = errors_within_envelope[
                        HORIZONS_S[horizon_index]]
                    run_envelope_horizon_errors = run_envelope_errors[
                        HORIZONS_S[horizon_index]]
                    band_horizon_errors = (
                        errors_by_speed_band[origin_band][
                            HORIZONS_S[horizon_index]]
                        if origin_band is not None else None)
                    run_band_horizon_errors = (
                        errors_by_speed_band_and_run[origin_band][run_name][
                            HORIZONS_S[horizon_index]]
                        if origin_band is not None else None)
                    for field, value in error_values.items():
                        errors[field].append(value)
                        run_horizon_errors[field].append(value)
                        # Retain only source trajectories whose measured speed
                        # stays inside the declared identification envelope
                        # through this horizon. Candidate predictions are not
                        # filtered, so overshoot remains visible as error.
                        if within_speed_envelope:
                            envelope_horizon_errors[field].append(value)
                            run_envelope_horizon_errors[field].append(value)
                            if band_horizon_errors is not None:
                                band_horizon_errors[field].append(value)
                            if run_band_horizon_errors is not None:
                                run_band_horizon_errors[field].append(value)
                    horizon_index += 1

    def summarize(errors: dict[float, dict[str, list[float]]]) -> dict[str, Any]:
        return {
            _horizon_key(horizon): {
                field: _stats(values) for field, values in fields_at_horizon.items()
            }
            for horizon, fields_at_horizon in errors.items()
        }

    return summarize(errors_by_horizon) | {
        "per_run": {
            run_name: summarize(run_errors)
            for run_name, run_errors in errors_by_run.items()
        },
        "within_16mps": summarize(errors_within_envelope),
        "within_16mps_per_run": {
            run_name: summarize(run_errors)
            for run_name, run_errors in errors_within_envelope_by_run.items()
        },
        "within_16mps_by_origin_speed_band": {
            band_name: {
                "aggregate": summarize(band_errors),
                "per_run": {
                    run_name: summarize(run_errors)
                    for run_name, run_errors in
                    errors_by_speed_band_and_run[band_name].items()
                },
            }
            for band_name, band_errors in errors_by_speed_band.items()
        },
        "evaluation": {
            "max_origins_per_run": max_origins_per_run,
            "origins_used_per_run": origins_used,
            "identified_speed_ceiling_mps": IDENTIFIED_SPEED_CEILING_MPS,
            "within_envelope_semantics": (
                "origin and every recorded u_k/u_k1 through the scored "
                "horizon remain at or below 16 m/s; candidate state is not "
                "filtered"),
            "target_horizon_semantics": (
                "first source transition sum >= requested horizon"),
        },
    }


def benchmark(root: Path, run_names: Iterable[str], dump: Path, output: Path,
              max_origins_per_run: int,
              normal_load_mode: str = "static_sprung_mass",
              suspension_state: bool = False,
              drag_interpretation: str = "serialized_force_coefficient"
              ) -> dict[str, Any]:
    if max_origins_per_run < 1:
        raise ValueError("max_origins_per_run must be positive")
    names = tuple(run_names)
    if not names:
        raise ValueError("at least one assembled run is required")
    parameters = parameters_from_dump(dump)
    serialized_rigidbody_drag = parameters.rigid_body_drag_per_s
    if normal_load_mode not in {
            "static_sprung_mass", "mechanical_longitudinal_cg_transfer",
            "mechanical_cg_transfer"}:
        raise ValueError(f"unsupported normal-load mode: {normal_load_mode}")
    if suspension_state and normal_load_mode != "static_sprung_mass":
        raise ValueError(
            "explicit suspension state cannot be combined with algebraic "
            "load transfer")
    # ``model_transition_v4.csv`` already contains the established
    # API-positive effective steering angle.  The raw Unity diagnostic dump,
    # by contrast, stores ``AppliedSteering`` before
    # ``VehicleController.SteeringAngle = -AppliedSteering``.  Keep this
    # conversion at the data-interface boundary instead of allowing the two
    # meanings to share one silent sign.
    parameters = replace(
        parameters,
        unity_normal_load_mode=normal_load_mode,
        steering_to_wheel_angle_sign=1.0,
    )
    parameters = _parameters_with_drag_interpretation(
        parameters, drag_interpretation)
    if parameters.unity_wheel_dynamics is None:
        raise ValueError(
            "the diagnostic dump lacks the active Unity torque branch")
    runs = _read_runs(root, names)
    report = {
        "schema_version": 1,
        "status": "offline_explicit_unity_wheel_model_benchmark",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "offline_origin_initialization_and_endpoint_scoring",
        "runtime_input_compatible": False,
        "runtime_input_contract": {
            "permitted_live_sources": [
                "/odom", "/ekf_odom", "/current_map_pose", "/cmd/speed",
            ],
            "this_benchmark_uses": [
                "simulator-trace pose, twist, yaw rate, and steering at each origin",
                "encoder-derived wheel rates to initialize four latent wheel states",
                "recorded Unity transition commands during open-loop rollout",
            ],
            "reason_not_live_compatible": (
                "the origin state and four wheel rates are not reconstructed "
                "from the permitted live topics"
            ),
        },
        "source": {
            "assembled_root": str(root),
            "runs": list(names),
            "unity_parameter_dump": str(dump),
            "transition_table": "model_transition_v4.csv",
        },
        "model": {
            "state": [
                "x_m", "y_m", "yaw_rad", "u_mps", "v_mps", "r_radps",
                "steering_rad", "omega_front_left_radps",
                "omega_front_right_radps", "omega_rear_left_radps",
                "omega_rear_right_radps",
            ] + ([
                "heave_m", "heave_rate_mps", "pitch_rad",
                "pitch_rate_radps", "roll_rad", "roll_rate_radps",
            ] if suspension_state else []),
            "inputs": ["transition_steering_norm_k",
                       "transition_throttle_norm_k"],
            "initial_state_source": (
                "simulator-trace pose/twist/steering plus encoder-derived "
                "wheel rates; offline diagnostic initialization only"),
            "steering_input_semantics": (
                "the transition table's previous-packet command drives the "
                "transition; steering_state_rad_k is the API-positive "
                "effective state and raw Unity AppliedSteering uses the "
                "exact-dump sign in parameters_from_dump"),
            "contact_law": {
                "forward_slip": (
                    "clip((wheel_surface_speed - wheel_forward_speed) / "
                    "max(abs(wheel_surface_speed), abs(wheel_forward_speed), "
                    "0.25 m/s), -1, 1)"),
                "sideways_slip": (
                    "-wheel_sideways_speed / "
                    "max(abs(wheel_forward_speed), 0.5 m/s)"),
                "friction_curve": (
                    "F0 linear interpolation of serialized WheelFrictionCurve "
                    "knots is the current implementation; Unity documents a "
                    "two-piece spline, so F0 is an unverified baseline pending "
                    "the Phase 3 force-level comparison"),
                "force": "normal_load * signed_friction_curve(slip)",
            },
            "normal_load_law": (
                "max(0, sprungMass*g - spring*suspensionDistance*travel - "
                "damper*suspensionDistance*travel_rate)"
                if suspension_state else
                "captured static sprungMass*g"),
            "normal_load_source": (
                "per-wheel sprungMass from exact diagnostic snapshot"
                if normal_load_mode == "static_sprung_mass" else
                "exact sprungMass plus structural CG transfer from predicted tire acceleration"),
            "body_inertia_source": "exact dumped Rigidbody body-Y inertia",
            "body_drag_law": (
                "the plant subtracts the interpreted drag coefficient times "
                "body velocity from force, then divides by Rigidbody mass"),
            "body_drag_interpretation": {
                "mode": drag_interpretation,
                "serialized_drag_per_s": serialized_rigidbody_drag,
                "effective_force_coefficient_n_per_mps": (
                    parameters.rigid_body_drag_per_s),
                "candidate_only": (
                    drag_interpretation != "serialized_force_coefficient"),
                "runtime_or_production_use": False,
            },
            "wheel_torque_source": (
                "VehicleController CAWD motor torque and CAWB brake branch"),
            "wheel_rotational_response": {
                "effective_inertia_regimes_kgm2": list(
                    parameters.unity_wheel_dynamics
                    .effective_inertia_regimes_kgm2),
                "breakpoints_mps": list(
                    parameters.unity_wheel_dynamics.inertia_breakpoints_mps),
                "damping_nms": list(
                    parameters.unity_wheel_dynamics.wheel_damping_nms),
                "provenance": parameters.unity_wheel_dynamics.provenance,
            },
            "dynamic_suspension_transfer": suspension_state,
            "normal_load_mode": normal_load_mode,
            "mechanical_cg_transfer_is_suspension_model": False,
            "suspension_state_enabled": suspension_state,
            "suspension_state_source": (
                "captured WheelCollider spring/damper, body pitch/roll inertias, "
                "and dumped COM geometry"
                if suspension_state else None),
        },
        "scores": _score_runs(
            runs, parameters, max_origins_per_run, suspension_state),
        "promotion": {
            "runtime_use": False,
            "production_mpc_use": False,
            "requires": [
                "blind full-horizon comparison against clean open and track runs",
                "separate dynamic suspension/load-transfer identification",
                "legal sensor-only observer validation before odometry tuning",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", dest="runs", action="append", required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-origins-per-run", type=int, default=100)
    parser.add_argument(
        "--normal-load-mode",
        choices=("static_sprung_mass", "mechanical_longitudinal_cg_transfer",
                 "mechanical_cg_transfer"),
        default="static_sprung_mass",
        help="offline normal-load screen; never changes production behavior",
    )
    parser.add_argument(
        "--suspension-state", action="store_true",
        help="use the offline explicit heave/pitch/roll plant path",
    )
    parser.add_argument(
        "--drag-interpretation", choices=DRAG_INTERPRETATIONS,
        default="serialized_force_coefficient",
        help=("offline interpretation of Unity Rigidbody drag; "
              "unity_velocity_decay_rate is an unpromoted candidate"),
    )
    args = parser.parse_args()
    report = benchmark(args.root, args.runs, args.dump, args.output,
                       args.max_origins_per_run, args.normal_load_mode,
                       args.suspension_state, args.drag_interpretation)
    print(json.dumps({
        "output": str(args.output),
        "scores": report["scores"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
