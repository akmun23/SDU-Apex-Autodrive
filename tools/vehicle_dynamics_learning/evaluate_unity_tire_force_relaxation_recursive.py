#!/usr/bin/env python3
"""Test r07-fitted tire-force relaxation in command-only 3-D r11 rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.evaluate_unity_3d_yaw_residual import (
    _evaluation_starts,
    _features,
)
from tools.vehicle_dynamics_learning.evaluate_unity_tire_force_relaxation import (
    DEFAULT_OUTPUT as BALANCE_REPORT,
)
from tools.vehicle_dynamics_learning.replay_unity_source_3d_plant import (
    DEFAULT_SLIP_FIT,
    DEFAULT_SUSPENSION_FIT,
    DEFAULT_TIRE_FIT,
    DEFAULT_WHEEL_FIT,
    HORIZON_MILLISECONDS,
    ROOT,
    _advance,
    _curve,
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


DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_tire_force_relaxation_recursive_20261013.json")
HORIZON_S = 5.0
WINDOW_STRIDE_S = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    position = np.asarray([row["position_horizontal_m"] for row in rows])
    heading = np.asarray([row["heading_rad"] for row in rows])
    orientation = np.asarray([row["orientation_rad"] for row in rows])
    velocity = np.asarray([row["body_velocity_xyz_mps"] for row in rows])
    omega = np.asarray([row["body_omega_xyz_radps"] for row in rows])
    wheels = np.asarray([row["wheel_omega_radps"] for row in rows])
    return {
        "windows": len(rows),
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
            np.asarray([row["steering_rad"] for row in rows])**2))),
    }


def _steady_force_targets(feature: np.ndarray, constants: Any
                          ) -> np.ndarray:
    tire = constants[10]
    curves_x = constants[8]
    curves_y = constants[9]
    force = np.empty((4, 2), dtype=np.float64)
    for wheel in range(4):
        base = 17 + wheel * 5
        slip_x, slip_y, load = feature[base + 2:base + 5]
        fx = (math.copysign(1.0, slip_x) * load * tire[0]
              * float(_curve(abs(slip_x), curves_x[wheel], tire[1])))
        fy = (-math.copysign(1.0, slip_y) * load * tire[2]
              * float(_curve(abs(slip_y), curves_y[wheel], tire[3])))
        force[wheel] = (fx, fy)
    return force


def _rollout(path: Path, constants: Any,
             tau_x_s: float, tau_y_s: float) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    if not math.isclose(float(metadata["fixed_delta_time_s"]), DT_S,
                        rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("held-out capture is not sampled at 1 kHz")
    steps = round(HORIZON_S / DT_S)
    stride = round(WINDOW_STRIDE_S / DT_S)
    starts = _evaluation_starts(data, steps, stride)
    if not starts:
        raise ValueError(f"no grounded windows in {path}")
    checkpoints = {
        round(ms / 1000.0 / DT_S): f"{ms / 1000.0:.3f}s"
        for ms in HORIZON_MILLISECONDS
        if ms / 1000.0 <= HORIZON_S
    }
    model_names = ("source_controller_static_tires",
                   "source_controller_force_relaxation")
    errors = {name: {label: [] for label in checkpoints.values()}
              for name in model_names}
    windows = []
    alpha_x = -math.expm1(-DT_S / tau_x_s) if tau_x_s > 0.0 else 1.0
    alpha_y = -math.expm1(-DT_S / tau_y_s) if tau_y_s > 0.0 else 1.0

    for start in starts:
        base = _initial_state(data, start)
        relaxed = base.copy()
        start_steering = float(data["steering_command"][start])
        start_throttle = float(data["throttle_command"][start])
        initial_feature = _features(
            relaxed, start_steering, start_throttle,
            float(data["steering_command"][max(0, start - 1)]),
            float(data["throttle_command"][max(0, start - 1)]), constants)
        force_state = _steady_force_targets(initial_feature, constants)
        row: dict[str, Any] = {
            "start_sample": int(start),
            "initial_speed_mps": float(data["speed_mps"][start]),
            "initial_abs_steering_rad": float(abs(base[17])),
            "checkpoints": {name: {} for name in model_names},
        }
        for offset in range(steps):
            index = start + offset
            # FixedUpdate captures state and current torque at row i; its
            # steering update for row i+1 is made from the next command.
            steering_next = float(data["steering_command"][index + 1])
            throttle_current = float(data["throttle_command"][index])
            base = _advance(
                base, steering_next, throttle_current, constants, DT_S,
                unity_autonomous_steer=True)

            feature = _features(
                relaxed, steering_next, throttle_current,
                float(data["steering_command"][index]),
                float(data["throttle_command"][max(0, index - 1)]), constants)
            steady = _steady_force_targets(feature, constants)
            force_state[:, 0] += alpha_x * (steady[:, 0] - force_state[:, 0])
            force_state[:, 1] += alpha_y * (steady[:, 1] - force_state[:, 1])
            relaxed = _advance(
                relaxed, steering_next, throttle_current, constants, DT_S,
                unity_autonomous_steer=True,
                relaxed_tire_forces=force_state)
            if not np.isfinite(base).all() or not np.isfinite(relaxed).all():
                raise FloatingPointError(
                    f"non-finite state at {path}:{index}")
            elapsed = offset + 1
            if elapsed not in checkpoints:
                continue
            label = checkpoints[elapsed]
            truth = _initial_state(data, index + 1)
            base_error = _state_error(base, truth)
            relaxed_error = _state_error(relaxed, truth)
            errors[model_names[0]][label].append(base_error)
            errors[model_names[1]][label].append(relaxed_error)
            row["checkpoints"][model_names[0]][label] = base_error
            row["checkpoints"][model_names[1]][label] = relaxed_error
        windows.append(row)

    aggregate = {
        name: {label: _metrics(rows)
               for label, rows in horizon_errors.items()}
        for name, horizon_errors in errors.items()
    }
    paired = {}
    for label in checkpoints.values():
        base_rows = [window["checkpoints"][model_names[0]][label]
                     for window in windows]
        candidate_rows = [window["checkpoints"][model_names[1]][label]
                          for window in windows]
        for field in ("position_horizontal_m", "heading_rad"):
            delta = np.asarray([candidate[field] - baseline[field]
                                for baseline, candidate
                                in zip(base_rows, candidate_rows)])
            paired.setdefault(label, {})[field] = {
                "candidate_better_windows": int(np.count_nonzero(delta < 0.0)),
                "candidate_worse_windows": int(np.count_nonzero(delta > 0.0)),
                "mean_candidate_minus_baseline": float(np.mean(delta)),
                "median_candidate_minus_baseline": float(np.median(delta)),
            }
    return {
        "capture": str(path.relative_to(ROOT)),
        "capture_sha256": _sha256(path),
        "window_count": len(starts),
        "window_stride_s": WINDOW_STRIDE_S,
        "horizon_s": HORIZON_S,
        "aggregate_errors": aggregate,
        "paired_window_changes": paired,
        "per_window": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--balance-report", type=Path, default=BALANCE_REPORT)
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--slip-fit", type=Path, default=DEFAULT_SLIP_FIT)
    parser.add_argument("--suspension-fit", type=Path,
                        default=DEFAULT_SUSPENSION_FIT)
    parser.add_argument("--wheel-fit", type=Path, default=DEFAULT_WHEEL_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    balance = json.loads(args.balance_report.read_text(encoding="utf-8"))
    tau_x = float(balance["fit"]["tau_longitudinal_s"])
    tau_y = float(balance["fit"]["tau_lateral_s"])
    constants, description = _prepare_constants(
        _metadata(args.train), args.tire_fit, args.slip_fit,
        args.suspension_fit, args.wheel_fit)
    started = time.time()
    heldout = _rollout(args.heldout, constants, tau_x, tau_y)
    report = {
        "schema_version": 1,
        "status": "research-only recursive candidate; no runtime integration",
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
        "heldout_capture": str(args.heldout.relative_to(ROOT)),
        "heldout_capture_sha256": _sha256(args.heldout),
        "force_balance_fit_report": str(args.balance_report.relative_to(ROOT)),
        "force_balance_fit_report_sha256": _sha256(args.balance_report),
        "fitted_tau_longitudinal_s": tau_x,
        "fitted_tau_lateral_s": tau_y,
        "plant_parameters": description,
        "heldout_recursive_command_only": heldout,
        "future_truth_or_wheelhit_used_during_rollout": False,
        "r11_used_for_fit_or_parameter_selection": False,
        "runtime_odometry_mpc_unity_or_trajectory_changed": False,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "tau_longitudinal_s": tau_x,
        "tau_lateral_s": tau_y,
        "windows": heldout["window_count"],
        "r11_5s_baseline": heldout["aggregate_errors"][
            "source_controller_static_tires"]["5.000s"],
        "r11_5s_candidate": heldout["aggregate_errors"][
            "source_controller_force_relaxation"]["5.000s"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
