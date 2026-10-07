#!/usr/bin/env python3
"""Measure local and recursive active-optimizer vs production-MPC defects.

This is a mathematical contract audit on a saved optimizer solution. It does
not read a bag, estimate plant accuracy, solve the OCP, or start a simulator.
It first measures one production 25 ms stage from each nominal optimizer state,
then separately replays the optimizer's spatial rate schedule recursively
through production C for up to one lap. The delayed-command values are
reconstructed from the command schedule; this is model-to-model validation,
not simulator-truth validation.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OPTIMIZER_ROOT = ROOT / "SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3"
if str(OPTIMIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_ROOT))

from f1tenth_planning.autodrive_mintime.model import VehicleModel  # noqa: E402
from tools.racing.offline.controller.production_mpc import ProductionMpc  # noqa: E402


class ModelState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "e_y", "e_psi", "u", "v", "r", "target_speed",
        "steering_command", "delayed_steering_command_1",
        "delayed_steering_command_2", "actual_steering_angle")]


class ModelControl(ctypes.Structure):
    _fields_ = [("steering_rate", ctypes.c_float),
                ("target_speed_rate", ctypes.c_float)]


class StageResult(ctypes.Structure):
    _fields_ = [("next", ModelState), ("delta_s_m", ctypes.c_float),
                ("body_accel_mps2", ctypes.c_float),
                ("branch_flags", ctypes.c_uint), ("valid", ctypes.c_int)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_nodes(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = ("s_ref_m", "kappa_ref", "ey_m", "epsi_rad", "u_mps",
                "r_radps", "target_mps", "delta_rad", "qdelta_radps",
                "qv_mps2")
    if len(rows) < 8 or not rows or any(key not in rows[0] for key in required):
        raise ValueError(f"optimizer nodes lack required fields: {path}")
    values = {key: np.asarray([float(row[key]) for row in rows], dtype=float)
              for key in required}
    if "steering_command_rad" in rows[0]:
        values["steering_command_rad"] = np.asarray(
            [float(row["steering_command_rad"]) for row in rows], dtype=float)
    s = values["s_ref_m"]
    if not np.isfinite(np.column_stack(list(values.values()))).all():
        raise ValueError("optimizer node data contains non-finite values")
    if s[0] < -1e-8 or np.any(np.diff(s) <= 0.0):
        raise ValueError("optimizer nodes must have strictly increasing s from zero")
    return values


def periodic_interp(query: float, nodes: dict[str, np.ndarray],
                    key: str, lap_length: float) -> float:
    s = nodes["s_ref_m"]
    y = nodes[key]
    q = query % lap_length
    hi = int(np.searchsorted(s, q, side="right"))
    if hi == 0:
        lo, hi = len(s) - 1, 0
        x0, x1 = s[lo] - lap_length, s[hi]
        y0, y1 = y[lo], y[hi]
        if q > x1:
            q -= lap_length
    elif hi == len(s):
        lo = len(s) - 1
        x0, x1 = s[lo], s[0] + lap_length
        y0, y1 = y[lo], y[0]
    else:
        lo = hi - 1
        x0, x1 = s[lo], s[hi]
        y0, y1 = y[lo], y[hi]
    return float(y0 + (q - x0) * (y1 - y0) / (x1 - x0))


def optimizer_state(s: float, nodes: dict[str, np.ndarray], lap_length: float,
                    model: VehicleModel, use_observer_v: bool) -> dict[str, float]:
    result = {key: periodic_interp(s, nodes, key, lap_length)
              for key in ("ey_m", "epsi_rad", "u_mps", "r_radps",
                          "target_mps", "delta_rad", "qdelta_radps",
                          "qv_mps2", "kappa_ref")}
    command_key = ("steering_command_rad" if
                   "steering_command_rad" in nodes else "delta_rad")
    result["steering_command_rad"] = periodic_interp(
        s, nodes, command_key, lap_length)
    result["v_mps"] = (float(model.lateral_velocity_numeric(
        result["u_mps"], result["r_radps"])) if use_observer_v else 0.0)
    den = 1.0 - result["kappa_ref"] * result["ey_m"]
    result["sdot_mps"] = (
        result["u_mps"] * math.cos(result["epsi_rad"])
        - result["v_mps"] * math.sin(result["epsi_rad"])) / den
    if result["sdot_mps"] <= 0.0:
        raise ValueError(f"nonpositive optimizer progress rate at s={s:.6f}")
    return result


def production_step(library: ctypes.CDLL, state_values: dict[str, float],
                    delta_delay1: float, delta_delay2: float,
                    qdelta: float, qv: float, dt: float,
                    curvature: float) -> tuple[dict[str, float], float, int]:
    state = ModelState(
        state_values["ey_m"], state_values["epsi_rad"],
        state_values["u_mps"], state_values["v_mps"],
        state_values["r_radps"], state_values["target_mps"],
        state_values["steering_command_rad"], delta_delay1, delta_delay2,
        state_values["delta_rad"])
    control = ModelControl(qdelta, qv)
    stage = library.mpc_vehicle_model_step(
        ctypes.byref(state), ctypes.byref(control), ctypes.c_float(dt),
        ctypes.c_float(curvature))
    if not stage.valid:
        raise ArithmeticError(f"production model rejected stage at s={state_values['s_m']:.6f}")
    next_values = {name: float(getattr(stage.next, name)) for name, _ in ModelState._fields_}
    return next_values, float(stage.delta_s_m), int(stage.branch_flags)


def error_summary(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def recursive_schedule_rollout(library: ctypes.CDLL,
                               nodes: dict[str, np.ndarray],
                               lap_length: float, model: VehicleModel,
                               use_observer_v: bool, dt: float,
                               optimizer_lap_time_s: float | None) -> dict[str, Any]:
    """Replay one candidate lap through production C using its spatial rate schedule."""
    command_key = ("steering_command_rad" if
                   "steering_command_rad" in nodes else "delta_rad")
    initial = optimizer_state(0.0, nodes, lap_length, model, use_observer_v)
    state: dict[str, float] = {
        "s_m": 0.0,
        "ey_m": initial["ey_m"],
        "epsi_rad": initial["epsi_rad"],
        "u_mps": initial["u_mps"],
        "v_mps": initial["v_mps"],
        "r_radps": initial["r_radps"],
        "target_mps": initial["target_mps"],
        "steering_command_rad": initial["steering_command_rad"],
        "delta_rad": initial["delta_rad"],
    }
    state["delayed_steering_command_1"] = periodic_interp(
        -initial["sdot_mps"] * dt, nodes, command_key, lap_length)
    state["delayed_steering_command_2"] = periodic_interp(
        -initial["sdot_mps"] * 2.0 * dt, nodes, command_key, lap_length)
    fields = ("e_y_m", "e_psi_rad", "u_mps", "v_mps", "r_radps",
              "target_speed_mps", "steering_command_rad",
              "actual_steering_angle_rad")
    errors: dict[str, list[float]] = {name: [] for name in fields}
    divergence_thresholds = {
        "e_y_m": 0.10,
        "e_psi_rad": 0.05,
        "u_mps": 0.10,
        "r_radps": 0.25,
        "steering_command_rad": 0.05,
        "actual_steering_angle_rad": 0.05,
    }
    first_divergence: dict[str, dict[str, float]] = {}
    rollout_trace: list[dict[str, float]] = []
    current_s = 0.0
    steps = 0
    max_steps = max(100, int(math.ceil(lap_length / (dt * 0.5))) + 20)
    crossing_fraction: float | None = None
    invalid_reason: str | None = None

    while current_s < lap_length and steps < max_steps:
        schedule = optimizer_state(current_s, nodes, lap_length, model,
                                   use_observer_v)
        state["s_m"] = current_s
        previous_s = current_s
        try:
            next_values, delta_s, _ = production_step(
                library, state, state["delayed_steering_command_1"],
                state["delayed_steering_command_2"],
                schedule["qdelta_radps"], schedule["qv_mps2"], dt,
                schedule["kappa_ref"])
        except (ArithmeticError, KeyError, ValueError) as exc:
            invalid_reason = f"{type(exc).__name__}: {exc}"
            break
        if not math.isfinite(delta_s) or delta_s <= 0.0:
            invalid_reason = f"nonpositive progress increment: {delta_s}"
            break
        current_s += delta_s
        if current_s >= lap_length:
            crossing_fraction = min(1.0, max(
                0.0, (lap_length - previous_s) / delta_s))
        expected = optimizer_state(current_s, nodes, lap_length, model,
                                   use_observer_v)
        step_errors = {
            "e_y_m": next_values["e_y"] - expected["ey_m"],
            "e_psi_rad": math.atan2(
                math.sin(next_values["e_psi"] - expected["epsi_rad"]),
                math.cos(next_values["e_psi"] - expected["epsi_rad"])),
            "u_mps": next_values["u"] - expected["u_mps"],
            "v_mps": next_values["v"] - expected["v_mps"],
            "r_radps": next_values["r"] - expected["r_radps"],
            "target_speed_mps": next_values["target_speed"] - expected["target_mps"],
            "steering_command_rad": (
                next_values["steering_command"] - expected["steering_command_rad"]),
            "actual_steering_angle_rad": (
                next_values["actual_steering_angle"] - expected["delta_rad"]),
        }
        for name, value in step_errors.items():
            errors[name].append(value)
            threshold = divergence_thresholds.get(name)
            if threshold is not None and name not in first_divergence and abs(value) > threshold:
                first_divergence[name] = {
                    "step": float(steps + 1),
                    "reference_progress_m": float(current_s),
                    "error": float(value),
                    "threshold": threshold,
                }
        if steps < 16:
            rollout_trace.append({
                "step": float(steps + 1),
                "reference_progress_m": float(current_s),
                "speed_mps": float(state["u_mps"]),
                "curvature_inv_m": float(schedule["kappa_ref"]),
                "steering_rate_radps": float(schedule["qdelta_radps"]),
                "target_speed_rate_mps2": float(schedule["qv_mps2"]),
                "lateral_error_m": float(step_errors["e_y_m"]),
                "heading_error_rad": float(step_errors["e_psi_rad"]),
                "forward_speed_error_mps": float(step_errors["u_mps"]),
                "yaw_rate_error_radps": float(step_errors["r_radps"]),
                "steering_command_error_rad": float(
                    step_errors["steering_command_rad"]),
                "actual_steering_error_rad": float(
                    step_errors["actual_steering_angle_rad"]),
                "production_yaw_rate_radps": float(next_values["r"]),
                "optimizer_yaw_rate_radps": float(expected["r_radps"]),
                "production_actual_steering_rad": float(
                    next_values["actual_steering_angle"]),
                "optimizer_actual_steering_rad": float(expected["delta_rad"]),
            })
        state = {
            "s_m": current_s,
            "ey_m": next_values["e_y"],
            "epsi_rad": next_values["e_psi"],
            "u_mps": next_values["u"],
            "v_mps": next_values["v"],
            "r_radps": next_values["r"],
            "target_mps": next_values["target_speed"],
            "steering_command_rad": next_values["steering_command"],
            "delta_rad": next_values["actual_steering_angle"],
            "delayed_steering_command_1": next_values[
                "delayed_steering_command_1"],
            "delayed_steering_command_2": next_values[
                "delayed_steering_command_2"],
        }
        steps += 1

    completed = current_s >= lap_length
    summaries = {name: error_summary(np.asarray(values, dtype=float))
                 for name, values in errors.items() if values}
    model_lap_time = ((steps - 1 + crossing_fraction) * dt
                      if completed and crossing_fraction is not None else None)
    return {
        "status": "completed_one_reference_lap" if completed else "incomplete",
        "is_simulator_truth_validation": False,
        "steps": steps,
        "fixed_step_s": dt,
        "reference_lap_length_m": lap_length,
        "production_progress_m": current_s,
        "progress_overrun_m": max(0.0, current_s - lap_length),
        "model_lap_time_s": model_lap_time,
        "optimizer_lap_time_s": optimizer_lap_time_s,
        "model_minus_optimizer_lap_time_s": (
            model_lap_time - optimizer_lap_time_s
            if model_lap_time is not None and optimizer_lap_time_s is not None
            else None),
        "invalid_reason": invalid_reason,
        "first_error_threshold_crossing": first_divergence,
        "initial_step_trace": rollout_trace,
        "recursive_errors_vs_optimizer_schedule": summaries,
        "terminal_state": {
            "e_y_m": state["ey_m"],
            "e_psi_rad": state["epsi_rad"],
            "u_mps": state["u_mps"],
            "v_mps": state["v_mps"],
            "r_radps": state["r_radps"],
            "target_speed_mps": state["target_mps"],
            "steering_command_rad": state["steering_command_rad"],
            "actual_steering_angle_rad": state["delta_rad"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--mpc-library", type=Path, required=True)
    parser.add_argument("--mpc-config", type=Path,
                        default=ROOT / "f1tenth_mpc/config/mpc_competition.yaml")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--sample-spacing-m", type=float, default=0.04)
    parser.add_argument("--optimizer-lap-time-s", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {name: (path if path.is_absolute() else ROOT / path).resolve()
             for name, path in vars(args).items() if isinstance(path, Path)}
    for name, path in paths.items():
        if name != "output" and not path.is_file():
            parser.error(f"{name} does not exist: {path}")
    if args.sample_spacing_m <= 0.0:
        parser.error("--sample-spacing-m must be positive")

    nodes = read_nodes(paths["nodes"])
    s = nodes["s_ref_m"]
    lap_length = float(s[-1] + np.median(np.diff(s)))
    optimizer_doc = yaml.safe_load(paths["optimizer_config"].read_text(encoding="utf-8")) or {}
    model = VehicleModel.from_repo(ROOT, optimizer_doc)
    use_observer_v = bool(optimizer_doc.get("lateral_dynamics", {}).get(
        "use_odometry_lateral_velocity", False))
    dt = 0.025
    sample_s = np.arange(0.0, lap_length, args.sample_spacing_m)
    library = ctypes.CDLL(str(paths["mpc_library"]))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl),
        ctypes.c_float, ctypes.c_float]
    library.mpc_vehicle_model_step.restype = StageResult

    metric_names = ("e_y_m", "e_psi_rad", "u_mps", "v_mps", "r_radps",
                    "target_speed_mps", "steering_command_rad",
                    "actual_steering_angle_rad")
    queued_errors = {name: [] for name in metric_names}
    immediate_errors = {name: [] for name in metric_names}
    queued_by_regime: dict[str, dict[str, list[float]]] = {}
    regime_counts: dict[str, int] = {}
    sample_context: list[dict[str, float]] = []
    max_progress_step = 0.0
    sample_count = 0
    branch_counts: dict[str, int] = {}

    # Creating the production wrapper applies the exact frozen competition
    # YAML configuration to the C vehicle model before its stage function is
    # called. The P0 trajectory is passed only for that initialization.
    with ProductionMpc(paths["mpc_library"], paths["mpc_config"],
                       paths["trajectory"]):
        for s0 in sample_s:
            current = optimizer_state(float(s0), nodes, lap_length, model,
                                      use_observer_v)
            back1 = current["sdot_mps"] * dt
            back2 = current["sdot_mps"] * 2.0 * dt
            command_key = ("steering_command_rad" if
                           "steering_command_rad" in nodes else "delta_rad")
            delay1 = periodic_interp(float(s0 - back1), nodes, command_key, lap_length)
            delay2 = periodic_interp(float(s0 - back2), nodes, command_key, lap_length)
            current["s_m"] = float(s0)
            next_queued, delta_s, branch_flags = production_step(
                library, current, delay1, delay2,
                current["qdelta_radps"], current["qv_mps2"], dt,
                current["kappa_ref"])
            if delta_s <= 0.0:
                raise ArithmeticError(f"production model made no progress at s={s0:.6f}")
            s1 = float(s0 + delta_s)
            expected = optimizer_state(s1, nodes, lap_length, model,
                                       use_observer_v)

            # Counterfactual: preserve the same production equations but feed
            # the next planned angle into the delayed-command slot, removing
            # the queued-sample lag while retaining the physical rate limit.
            next_immediate, _, _ = production_step(
                library, current, expected["steering_command_rad"], delay1,
                current["qdelta_radps"], current["qv_mps2"], dt,
                current["kappa_ref"])
            observed = {
                "e_y_m": next_queued["e_y"],
                "e_psi_rad": math.atan2(math.sin(next_queued["e_psi"] - expected["epsi_rad"]),
                                         math.cos(next_queued["e_psi"] - expected["epsi_rad"])),
                "u_mps": next_queued["u"],
                "v_mps": next_queued["v"],
                "r_radps": next_queued["r"],
                "target_speed_mps": next_queued["target_speed"],
                "steering_command_rad": next_queued["steering_command"],
                "actual_steering_angle_rad": next_queued["actual_steering_angle"],
            }
            observed_immediate = {
                "e_y_m": next_immediate["e_y"],
                "e_psi_rad": math.atan2(math.sin(next_immediate["e_psi"] - expected["epsi_rad"]),
                                         math.cos(next_immediate["e_psi"] - expected["epsi_rad"])),
                "u_mps": next_immediate["u"],
                "v_mps": next_immediate["v"],
                "r_radps": next_immediate["r"],
                "target_speed_mps": next_immediate["target_speed"],
                "steering_command_rad": next_immediate["steering_command"],
                "actual_steering_angle_rad": next_immediate["actual_steering_angle"],
            }
            expected_values = {
                "e_y_m": expected["ey_m"],
                "e_psi_rad": 0.0,
                "u_mps": expected["u_mps"],
                "v_mps": expected["v_mps"],
                "r_radps": expected["r_radps"],
                "target_speed_mps": expected["target_mps"],
                "steering_command_rad": expected["steering_command_rad"],
                "actual_steering_angle_rad": expected["delta_rad"],
            }
            for name in metric_names:
                queued_errors[name].append(observed[name] - expected_values[name])
                immediate_errors[name].append(
                    observed_immediate[name] - expected_values[name])
            regime = ("high_speed_high_steer" if current["u_mps"] >= 6.0
                      and abs(current["delta_rad"]) >= 0.2 else
                      "high_steer" if abs(current["delta_rad"]) >= 0.2 else
                      "other")
            group = queued_by_regime.setdefault(regime, {name: [] for name in metric_names})
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            for name in metric_names:
                group[name].append(queued_errors[name][-1])
            sample_context.append({
                "s_m": float(s0),
                "speed_mps": current["u_mps"],
                "steering_command_rad": current["steering_command_rad"],
                "steering_rate_radps": current["qdelta_radps"],
                "yaw_rate_radps": current["r_radps"],
                "curvature_inv_m": current["kappa_ref"],
                "progress_rate_mps": current["sdot_mps"],
                "production_progress_step_m": delta_s,
                "planned_next_steering_rad": expected["delta_rad"],
                "production_next_command_rad": next_queued["steering_command"],
                "production_next_actual_steering_rad": next_queued["actual_steering_angle"],
                "planned_next_yaw_rate_radps": expected["r_radps"],
                "production_next_yaw_rate_radps": next_queued["r"],
                "delay1_command_rad": delay1,
            })
            branch_counts[str(branch_flags)] = branch_counts.get(str(branch_flags), 0) + 1
            max_progress_step = max(max_progress_step, delta_s)
            sample_count += 1

        recursive = recursive_schedule_rollout(
            library, nodes, lap_length, model, use_observer_v, dt,
            args.optimizer_lap_time_s)

    report = {
        "schema_version": 1,
        "purpose": "one-stage and recursive optimizer-to-production model-contract audit; not simulator truth validation",
        "source": {
            "optimizer_nodes": str(paths["nodes"].relative_to(ROOT)),
            "optimizer_nodes_sha256": sha256(paths["nodes"]),
            "optimizer_config": str(paths["optimizer_config"].relative_to(ROOT)),
            "optimizer_config_sha256": sha256(paths["optimizer_config"]),
            "production_mpc_config": str(paths["mpc_config"].relative_to(ROOT)),
            "production_mpc_config_sha256": sha256(paths["mpc_config"]),
            "p0_trajectory": str(paths["trajectory"].relative_to(ROOT)),
            "p0_trajectory_sha256": sha256(paths["trajectory"]),
        },
        "contract": {
            "sample_period_s": dt,
            "lap_length_from_spatial_nodes_m": lap_length,
            "sample_spacing_m": args.sample_spacing_m,
            "samples": sample_count,
            "optimizer_uses_odometry_lateral_velocity": use_observer_v,
            "optimizer_model": (
                "continuous spatial collocation; actual steering angle drives yaw; "
                "steering command queue and rate-limited physical-angle state are "
                f"{'modeled' if optimizer_doc.get('steering_actuator', {}).get('model_command_queue', False) else 'not modeled'}; "
                "lateral v is recomputed algebraically"),
            "production_model": "discrete source-command transition; one-sample steering queue and rate-limited actual steering; lateral v held per step",
            "comparison": "each C stage starts from its matching nominal optimizer state; next-state errors are measured at C-predicted progress",
            "delay_free_case": "counterfactual supplies next planned command to delayed-command input; physical rate limit and remaining production equations unchanged",
            "recursive_schedule_rollout": recursive,
            "max_production_progress_step_m": max_progress_step,
            "branch_flag_counts": branch_counts,
            "speed_steering_sample_counts": {
                "speed_ge_6mps_and_abs_steer_ge_0p2rad": sum(
                    count for name, count in regime_counts.items()
                    if name == "high_speed_high_steer"),
                "abs_steer_ge_0p2rad": sum(
                    count for name, count in regime_counts.items()
                    if name in ("high_steer", "high_speed_high_steer")),
                "other": regime_counts.get("other", 0),
            },
            "error_worst_context": {
                name: {
                    **sample_context[int(np.argmax(np.abs(values)))],
                    "signed_error": float(values[int(np.argmax(np.abs(values)))]),
                }
                for name, values in queued_errors.items()
            },
        },
        "queued_command_one_step_errors": {
            name: error_summary(np.asarray(values, dtype=float))
            for name, values in queued_errors.items()
        },
        "delay_free_counterfactual_errors": {
            name: error_summary(np.asarray(values, dtype=float))
            for name, values in immediate_errors.items()
        },
        "queued_errors_by_regime": {
            group_name: {
                name: error_summary(np.asarray(values, dtype=float))
                for name, values in group.items()
            }
            for group_name, group in queued_by_regime.items()
        },
        "interpretation_limits": [
            "One-stage metrics initialize C from matching nominal optimizer states; recursive metrics separately accumulate schedule mismatch.",
            "The recursive schedule rollout accumulates production-model versus OCP differences for one lap; it still is not simulator-truth validation.",
            "Queue-aware nodes export command and physical steering separately; legacy nodes use delta_rad for both.",
            "Agreement or disagreement here does not establish truth accuracy, closed-loop convergence, collision risk, or lap-time improvement.",
        ],
    }
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
