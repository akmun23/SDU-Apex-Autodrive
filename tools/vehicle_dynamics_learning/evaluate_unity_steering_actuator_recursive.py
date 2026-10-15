#!/usr/bin/env python3
"""Compare the source steering actuator and 1 ms input alignment on held-out r11.

VehicleController.Steer() has a direction-dependent update path. This test
compares its float32 arithmetic with a symmetric rate limiter and checks both
same-row and next-row command alignment. Each 5 s rollout is command-only;
truth initializes and scores windows but never refreshes their states.
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


DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_unity_steering_actuator_recursive_20261013.json")
HORIZON_S = 5.0
STRIDE_S = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _evaluate(path: Path, constants: Any) -> dict[str, Any]:
    data, capture_metadata = _load_capture(path)
    if not math.isclose(float(capture_metadata["fixed_delta_time_s"]), DT_S,
                        rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("expected the 1 kHz source capture")
    horizon_steps = round(HORIZON_S / DT_S)
    stride_steps = round(STRIDE_S / DT_S)
    starts = _evaluation_starts(data, horizon_steps, stride_steps)
    if not starts:
        raise ValueError(f"no fully grounded windows in {path}")
    checkpoints = {
        round(milliseconds / 1000.0 / DT_S): f"{milliseconds / 1000.0:.3f}s"
        for milliseconds in HORIZON_MILLISECONDS
        if milliseconds / 1000.0 <= HORIZON_S
    }
    modes = {
        "symmetric_same_row": (False, 0, 0),
        "symmetric_next_steer": (False, 1, 0),
        "symmetric_next_steer_and_throttle": (False, 1, 1),
        "unity_source_same_row": (True, 0, 0),
        "unity_source_next_steer": (True, 1, 0),
        "unity_source_next_steer_and_throttle": (True, 1, 1),
    }
    errors = {name: {label: [] for label in checkpoints.values()}
              for name in modes}
    windows = []
    for start in starts:
        states = {name: _initial_state(data, start) for name in modes}
        row: dict[str, Any] = {
            "start_sample": int(start),
            "initial_speed_mps": float(data["speed_mps"][start]),
            "initial_abs_steering_rad": float(abs(states["symmetric_same_row"][17])),
            "checkpoints": {name: {} for name in modes},
        }
        for offset in range(horizon_steps):
            index = start + offset
            for name, (source_steering, steering_shift,
                       throttle_shift) in modes.items():
                steer_index = index + steering_shift
                throttle_index = index + throttle_shift
                states[name] = _advance(
                    states[name],
                    float(data["steering_command"][steer_index]),
                    float(data["throttle_command"][throttle_index]),
                    constants, DT_S,
                    unity_autonomous_steer=source_steering,
                )
                if not np.isfinite(states[name]).all():
                    raise FloatingPointError(
                        f"non-finite state ({name}) at {path}:{index}")
            elapsed = offset + 1
            if elapsed not in checkpoints:
                continue
            label = checkpoints[elapsed]
            truth = _initial_state(data, index + 1)
            for name in modes:
                error = _state_error(states[name], truth)
                errors[name][label].append(error)
                row["checkpoints"][name][label] = error
        windows.append(row)

    aggregate = {
        name: {label: _metric(values)
               for label, values in horizon_rows.items()}
        for name, horizon_rows in errors.items()
    }
    paired = {}
    reference = "symmetric_same_row"
    for label in checkpoints.values():
        ref_rows = [window["checkpoints"][reference][label]
                    for window in windows]
        for name in modes:
            if name == reference:
                continue
            candidate_rows = [window["checkpoints"][name][label]
                              for window in windows]
            for field in ("position_horizontal_m", "heading_rad"):
                delta = np.asarray([c[field] - b[field]
                                    for b, c in zip(ref_rows, candidate_rows)])
                paired.setdefault(label, {}).setdefault(name, {})[field] = {
                    "candidate_better_windows": int(np.count_nonzero(delta < 0.0)),
                    "candidate_worse_windows": int(np.count_nonzero(delta > 0.0)),
                    "mean_candidate_minus_reference": float(np.mean(delta)),
                    "median_candidate_minus_reference": float(np.median(delta)),
                }
    return {
        "capture": str(path.relative_to(ROOT)),
        "capture_sha256": _sha256(path),
        "window_count": len(starts),
        "window_stride_s": STRIDE_S,
        "horizon_s": HORIZON_S,
        "model_modes": {
            name: {
                "source_vehiclecontroller_steer_logic": values[0],
                "steering_command_row_shift": values[1],
                "throttle_command_row_shift": values[2],
            }
            for name, values in modes.items()
        },
        "aggregate_errors": aggregate,
        "paired_window_changes_vs_symmetric_same_row": paired,
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
    constants, description = _prepare_constants(
        _metadata(args.train), args.tire_fit, args.slip_fit,
        args.suspension_fit, args.wheel_fit)
    heldout = _evaluate(args.heldout, constants)
    report = {
        "schema_version": 1,
        "status": "offline source-actuator experiment; no runtime integration",
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
        "heldout_capture": str(args.heldout.relative_to(ROOT)),
        "plant_parameters": description,
        "unity_source": {
            "file": "external/autodrive-unity-source/Assets/Scripts/VehicleController.cs",
            "method": "VehicleController.Steer()",
            "scene_steering_rate_deg_per_s": 183.346,
            "scene_steering_limit_deg": 30.0,
            "observed_rule": "AutonomousSteering is compared to last physical SteeringAngle / SteeringLimit; a steering unwind may reach its target immediately, unlike a symmetric rate limiter",
            "command_alignment_probe": "the FixedUpdate-captured command at sample i+1 actuates the i→i+1 state transition",
        },
        "heldout_recursive_command_only": heldout,
        "truth_used_only_for_initial_window_state_and_scoring": True,
        "future_truth_or_future_sensor_used_during_rollout": False,
        "no_simulator_runtime_physics_odometry_mpc_or_raceline_changed": True,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "window_count": heldout["window_count"],
        "5s_errors": {
            name: values["5.000s"]
            for name, values in heldout["aggregate_errors"].items()
        },
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
