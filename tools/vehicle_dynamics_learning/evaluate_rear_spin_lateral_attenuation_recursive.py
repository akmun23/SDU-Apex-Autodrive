#!/usr/bin/env python3
"""Fit a rear combined-slip term on r07; recursively test command-only on r11.

The candidate scales rear lateral tire force by
exp(-k * max(abs(longitudinal_slip) - peak_slip, 0)). The transition begins
at the serialized rear longitudinal friction-curve peak (0.15 slip). The
coefficient is fitted only to r07 force/moment balance. r11 truth and
WheelHit fields are scoring-only; the free rollout computes its own slip
from predicted state and uses only recorded steering/throttle commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from tools.vehicle_dynamics_learning.analyze_unity_3d_moment_balance import (
    DEFAULT_SAMPLE_STRIDE,
    PREFIX,
    WHEELS,
    _balance_data,
    _curve,
)
from tools.vehicle_dynamics_learning.evaluate_unity_3d_yaw_residual import (
    _evaluation_starts,
)
from tools.vehicle_dynamics_learning.replay_unity_source_3d_plant import (
    DEFAULT_SLIP_FIT,
    DEFAULT_SUSPENSION_FIT,
    DEFAULT_TIRE_FIT,
    DEFAULT_WHEEL_FIT,
    HORIZON_MILLISECONDS,
    ROOT,
    _advance,
    _initial_state,
    _metadata,
    _prepare_constants,
    _state_error,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    DT_S,
    _load_capture,
)


SPIN_THRESHOLD = 0.15
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_rear_spin_lateral_recursive_20261013.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tire_parameters(path: Path) -> np.ndarray:
    report = json.loads(path.read_text(encoding="utf-8"))
    fit = report["curve_model"]["training_fit"]
    return np.asarray((fit["longitudinal_scale"],
                       fit["longitudinal_zero_slip_tangent"],
                       fit["lateral_scale"],
                       fit["lateral_zero_slip_tangent"]), dtype=np.float64)


def _force_torque(data: dict[str, Any], parameters: np.ndarray,
                  attenuation: float) -> tuple[np.ndarray, np.ndarray]:
    scale_x, tangent_x, scale_y, tangent_y = parameters
    force = np.zeros_like(data["force_target_total"])
    torque = np.zeros_like(data["torque_target_total"])
    for index, wheel_name in enumerate(WHEELS):
        wheel = data["wheels"][index]
        prefix = PREFIX[wheel_name]
        curve_x = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_forward_curve"].split(",")])
        curve_y = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_sideways_curve"].split(",")])
        fx = (scale_x * wheel["load"]
              * _curve(wheel["slip_x"], curve_x, tangent_x))
        fy = (-scale_y * wheel["load"]
              * _curve(wheel["slip_y"], curve_y, tangent_y))
        if index >= 2 and attenuation > 0.0:
            fy *= np.exp(-attenuation * np.maximum(
                np.abs(wheel["slip_x"]) - SPIN_THRESHOLD, 0.0))
        tire_force = (fx[:, None] * wheel["forward"]
                      + fy[:, None] * wheel["sideways"])
        normal_force = wheel["load"][:, None] * wheel["normal"]
        total_force = tire_force + normal_force
        force += total_force
        torque += np.cross(wheel["lever"], total_force)
    return force, torque


def _fit_attenuation(train_path: Path, tire_fit_path: Path
                     ) -> tuple[float, dict[str, Any]]:
    training = _balance_data(
        train_path, derivative_window=31, sample_stride=1)
    parameters = _load_tire_parameters(tire_fit_path)
    fit_rows = training["rows"][::4]
    rear_slip = np.maximum(
        np.abs(training["wheels"][2]["slip_x"][fit_rows]),
        np.abs(training["wheels"][3]["slip_x"][fit_rows]))
    selected = fit_rows[rear_slip >= SPIN_THRESHOLD]
    if len(selected) < 100:
        raise ValueError(f"insufficient r07 high-spin fit rows: {len(selected)}")

    def residual(candidate: np.ndarray) -> np.ndarray:
        _, torque = _force_torque(training, parameters, float(candidate[0]))
        return torque[selected, 1] - training["torque_target_total"][selected, 1]

    result = least_squares(
        residual, np.asarray([0.3]), bounds=(np.asarray([0.0]),
                                             np.asarray([3.0])),
        loss="linear", max_nfev=100)
    fitted = float(result.x[0])
    baseline_force, baseline_torque = _force_torque(training, parameters, 0.0)
    fitted_force, fitted_torque = _force_torque(training, parameters, fitted)

    def score(force: np.ndarray, torque: np.ndarray,
              rows: np.ndarray) -> dict[str, Any]:
        f = force[rows] - training["force_target_total"][rows]
        t = torque[rows] - training["torque_target_total"][rows]
        return {
            "samples": int(len(rows)),
            "force_vector_rmse_n": float(np.sqrt(np.mean(
                np.sum(f * f, axis=1)))),
            "yaw_moment_rmse_nm": float(np.sqrt(np.mean(t[:, 1] ** 2))),
            "yaw_moment_bias_nm": float(np.mean(t[:, 1])),
        }

    return fitted, {
        "training_capture": str(train_path.relative_to(ROOT)),
        "fit_method": "least squares on yaw moment balance, high rear longitudinal slip only",
        "fit_sample_stride_from_physics_valid_rows": 4,
        "force_balance_base_sample_stride": 1,
        "serialized_rear_longitudinal_peak_slip": SPIN_THRESHOLD,
        "selected_fit_samples": int(len(selected)),
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "fitted_attenuation_per_excess_slip": fitted,
        "r07_high_spin_force_balance": {
            "unmodified": score(baseline_force, baseline_torque, selected),
            "attenuated": score(fitted_force, fitted_torque, selected),
        },
        "r07_all_valid_force_balance": {
            "unmodified": score(baseline_force, baseline_torque,
                                 training["rows"]),
            "attenuated": score(fitted_force, fitted_torque,
                                training["rows"]),
        },
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    position = np.asarray([row["position_horizontal_m"] for row in rows])
    heading = np.asarray([row["heading_rad"] for row in rows])
    orientation = np.asarray([row["orientation_rad"] for row in rows])
    velocity = np.asarray([row["body_velocity_xyz_mps"] for row in rows])
    omega = np.asarray([row["body_omega_xyz_radps"] for row in rows])
    wheels = np.asarray([row["wheel_omega_radps"] for row in rows])
    return {
        "windows": int(len(rows)),
        "position_horizontal_rmse_m": float(np.sqrt(np.mean(position**2))),
        "position_horizontal_p95_m": float(np.quantile(position, 0.95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading**2))),
        "heading_bias_rad": float(np.mean(heading)),
        "orientation_rmse_rad": float(np.sqrt(np.mean(orientation**2))),
        "body_velocity_xyz_rmse_mps": np.sqrt(
            np.mean(velocity**2, axis=0)).tolist(),
        "body_omega_xyz_rmse_radps": np.sqrt(
            np.mean(omega**2, axis=0)).tolist(),
        "wheel_omega_each_rmse_radps": np.sqrt(
            np.mean(wheels**2, axis=0)).tolist(),
        "steering_rmse_rad": float(np.sqrt(np.mean(
            np.asarray([row["steering_rad"] for row in rows]) ** 2))),
    }


def _rollout_compare(path: Path, baseline_constants: Any,
                     candidate_constants: Any) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    if not math.isclose(float(metadata["fixed_delta_time_s"]), DT_S,
                        rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("held-out capture is not sampled at 1 kHz")
    horizon_steps = round(5.0 / DT_S)
    stride_steps = round(5.0 / DT_S)
    starts = _evaluation_starts(data, horizon_steps, stride_steps)
    if not starts:
        raise ValueError(f"no fully grounded rollout windows in {path}")
    checkpoints = {
        round(ms / 1000.0 / DT_S): f"{ms / 1000.0:.3f}s"
        for ms in HORIZON_MILLISECONDS
        if ms / 1000.0 <= 5.0
    }
    modes = {"source_mechanics": baseline_constants,
             "rear_spin_lateral_attenuation": candidate_constants}
    errors = {name: {label: [] for label in checkpoints.values()}
              for name in modes}
    windows = []
    for start in starts:
        states = {name: _initial_state(data, start) for name in modes}
        per_window: dict[str, Any] = {
            "start_sample": int(start),
            "initial_speed_mps": float(data["speed_mps"][start]),
            "initial_abs_steering_rad": float(abs(states["source_mechanics"][17])),
            "checkpoints": {name: {} for name in modes},
        }
        for offset in range(horizon_steps):
            index = start + offset
            steering = float(data["steering_command"][index])
            throttle = float(data["throttle_command"][index])
            for name, constants in modes.items():
                states[name] = _advance(
                    states[name], steering, throttle, constants, DT_S)
                if not np.isfinite(states[name]).all():
                    raise FloatingPointError(
                        f"non-finite {name} state at {path}:{index}")
            elapsed = offset + 1
            if elapsed not in checkpoints:
                continue
            label = checkpoints[elapsed]
            truth = _initial_state(data, index + 1)
            for name in modes:
                error = _state_error(states[name], truth)
                errors[name][label].append(error)
                per_window["checkpoints"][name][label] = error
        windows.append(per_window)

    aggregate = {
        name: {label: _metrics(rows)
               for label, rows in horizon_errors.items()}
        for name, horizon_errors in errors.items()
    }
    paired = {}
    for label in checkpoints.values():
        base = [window["checkpoints"]["source_mechanics"][label]
                for window in windows]
        candidate = [window["checkpoints"][
            "rear_spin_lateral_attenuation"][label] for window in windows]
        for field in ("position_horizontal_m", "heading_rad"):
            delta = np.asarray([c[field] - b[field]
                                for b, c in zip(base, candidate)])
            paired.setdefault(label, {})[field] = {
                "candidate_better_windows": int(np.count_nonzero(delta < 0.0)),
                "candidate_worse_windows": int(np.count_nonzero(delta > 0.0)),
                "median_candidate_minus_baseline": float(np.median(delta)),
                "mean_candidate_minus_baseline": float(np.mean(delta)),
            }
    return {
        "capture": str(path.relative_to(ROOT)),
        "capture_sha256": _sha256(path),
        "window_count": len(starts),
        "window_stride_s": 5.0,
        "horizon_s": 5.0,
        "aggregate_errors": aggregate,
        "paired_window_changes": paired,
        "per_window": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--slip-fit", type=Path, default=DEFAULT_SLIP_FIT)
    parser.add_argument("--suspension-fit", type=Path,
                        default=DEFAULT_SUSPENSION_FIT)
    parser.add_argument("--wheel-fit", type=Path, default=DEFAULT_WHEEL_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    for path in (args.train, args.heldout, args.tire_fit, args.slip_fit,
                 args.suspension_fit, args.wheel_fit):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.time()
    attenuation, fit_diagnostics = _fit_attenuation(args.train, args.tire_fit)
    metadata = _metadata(args.train)
    baseline_constants, baseline_description = _prepare_constants(
        metadata, args.tire_fit, args.slip_fit, args.suspension_fit,
        args.wheel_fit)
    candidate_constants, candidate_description = _prepare_constants(
        metadata, args.tire_fit, args.slip_fit, args.suspension_fit,
        args.wheel_fit, rear_lateral_spin_attenuation=attenuation,
        rear_lateral_spin_threshold=SPIN_THRESHOLD)
    # Verify the optional term is a numerical no-op at zero and is active only
    # for a rear tire whose predicted longitudinal slip exceeds the curve peak.
    zero_constants, _ = _prepare_constants(
        metadata, args.tire_fit, args.slip_fit, args.suspension_fit,
        args.wheel_fit, rear_lateral_spin_attenuation=0.0,
        rear_lateral_spin_threshold=SPIN_THRESHOLD)
    initial = _initial_state(_load_capture(args.train)[0], 100)
    base_step = _advance(initial, 0.0, 0.0, baseline_constants, DT_S)
    zero_step = _advance(initial, 0.0, 0.0, zero_constants, DT_S)
    if not np.array_equal(base_step, zero_step):
        raise AssertionError("zero attenuation changed baseline plant behavior")

    # These are offline force-balance labels and held-out comparisons only.
    force_fit_path = args.output.with_name(
        args.output.stem + "_force_balance.json")
    training_balance = _balance_data(
        args.train, derivative_window=31,
        sample_stride=DEFAULT_SAMPLE_STRIDE)
    heldout_balance = _balance_data(
        args.heldout, derivative_window=31,
        sample_stride=DEFAULT_SAMPLE_STRIDE)
    tire_parameters = _load_tire_parameters(args.tire_fit)

    def balance_summary(balance: dict[str, Any]) -> dict[str, Any]:
        base_force, base_torque = _force_torque(balance, tire_parameters, 0.0)
        adjusted_force, adjusted_torque = _force_torque(
            balance, tire_parameters, attenuation)
        all_rows = balance["rows"]
        rear_slip = np.maximum(
            np.abs(balance["wheels"][2]["slip_x"][all_rows]),
            np.abs(balance["wheels"][3]["slip_x"][all_rows]))
        high_rows = all_rows[rear_slip >= SPIN_THRESHOLD]

        def score(force: np.ndarray, torque: np.ndarray,
                  rows: np.ndarray) -> dict[str, Any]:
            force_error = force[rows] - balance["force_target_total"][rows]
            torque_error = torque[rows] - balance["torque_target_total"][rows]
            return {
                "samples": int(len(rows)),
                "force_vector_rmse_n": float(np.sqrt(np.mean(
                    np.sum(force_error * force_error, axis=1)))),
                "yaw_moment_rmse_nm": float(np.sqrt(np.mean(
                    torque_error[:, 1] ** 2))),
                "yaw_moment_bias_nm": float(np.mean(torque_error[:, 1])),
            }

        return {
            "all_physics_valid_samples": {
                "unmodified": score(base_force, base_torque, all_rows),
                "attenuated": score(adjusted_force, adjusted_torque, all_rows),
            },
            "rear_longitudinal_slip_at_least_peak": {
                "unmodified": score(base_force, base_torque, high_rows),
                "attenuated": score(adjusted_force, adjusted_torque, high_rows),
            },
        }

    report = {
        "schema_version": 1,
        "status": "research-only candidate; recursive transfer decides; not production odometry",
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
        "heldout_capture": str(args.heldout.relative_to(ROOT)),
        "heldout_capture_sha256": _sha256(args.heldout),
        "serialized_tire_fit": str(args.tire_fit.relative_to(ROOT)),
        "fit": fit_diagnostics,
        "plant_baseline": baseline_description,
        "plant_candidate": candidate_description,
        "zero_gain_is_exact_plant_noop": True,
        "heldout_recursive_command_only": _rollout_compare(
            args.heldout, baseline_constants, candidate_constants),
        "force_balance_diagnostics": {
            "training_r07": balance_summary(training_balance),
            "heldout_r11": balance_summary(heldout_balance),
            "scoring_note": "WheelHit labels are used only for offline force-balance diagnostics; they are not used by the recursive state rollout",
        },
        "truth_or_future_sensors_used_during_rollout": False,
        "r11_used_for_fit_or_checkpoint_selection": False,
        "runtime_odometry_mpc_unity_or_trajectory_changed": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    heldout = report["heldout_recursive_command_only"]
    print(json.dumps({
        "report": str(args.output),
        "fitted_attenuation": attenuation,
        "fit_samples": fit_diagnostics["selected_fit_samples"],
        "windows": heldout["window_count"],
        "r11_5s_baseline": heldout["aggregate_errors"][
            "source_mechanics"]["5.000s"],
        "r11_5s_candidate": heldout["aggregate_errors"][
            "rear_spin_lateral_attenuation"]["5.000s"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
