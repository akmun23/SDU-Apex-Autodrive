#!/usr/bin/env python3
"""Test a causal tire-force relaxation state against r07/r11 force balance.

Static WheelCollider curves and all prior force-fit coefficients are frozen.
Only two physically interpretable first-order time constants (longitudinal and
lateral force response) are fitted to r07, using reconstructed rigid-body
force/moment labels. r11 is held out. This is a necessary force-level screen,
not a recursive plant acceptance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from tools.vehicle_dynamics_learning.analyze_unity_3d_moment_balance import (
    DEFAULT_SAMPLE_STRIDE,
    PREFIX,
    WHEELS,
    _balance_data,
    _curve,
)
from tools.vehicle_dynamics_learning.replay_unity_source_3d_plant import (
    DEFAULT_TIRE_FIT,
    ROOT,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    DT_S,
)


DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_force_relaxation_balance_20261013.json")
FIT_ROW_STRIDE = 4
MAX_RELAXATION_S = 0.15
SPIN_PEAK = 0.15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tire_parameters(path: Path) -> np.ndarray:
    report = json.loads(path.read_text(encoding="utf-8"))
    fit = report["curve_model"]["training_fit"]
    wheel = json.loads((ROOT / (
        "live_runs/unity_exact_binary_audit_20261012/"
        "wheel_spin_state_inputs_v10_20261013.json")).read_text(
            encoding="utf-8"))
    lateral = wheel["source_force_application_point_lateral_fit"][
        "fit"]["parameters"]
    return np.asarray((fit["longitudinal_scale"],
                       fit["longitudinal_zero_slip_tangent"],
                       lateral[0], lateral[1]), dtype=np.float64)


def _steady_tire_magnitudes(data: dict[str, Any],
                            parameters: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    sx = np.empty((len(data["speed"]), 4), dtype=np.float64)
    sy = np.empty_like(sx)
    scale_x, tangent_x, scale_y, tangent_y = parameters
    for index, wheel_name in enumerate(WHEELS):
        wheel = data["wheels"][index]
        prefix = PREFIX[wheel_name]
        curve_x = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_forward_curve"].split(",")])
        curve_y = np.asarray([float(value) for value in data["metadata"][
            f"{prefix}_sideways_curve"].split(",")])
        sx[:, index] = (scale_x * wheel["load"]
                        * _curve(wheel["slip_x"], curve_x, tangent_x))
        sy[:, index] = (-scale_y * wheel["load"]
                        * _curve(wheel["slip_y"], curve_y, tangent_y))
    return sx, sy


def _causal_filter(values: np.ndarray, tau_s: float, dt_s: float
                   ) -> np.ndarray:
    if tau_s <= 0.0:
        return values.copy()
    alpha = -np.expm1(-dt_s / tau_s)
    result = np.empty_like(values)
    for index in range(values.shape[1]):
        column = values[:, index]
        initial_state = np.asarray([(1.0 - alpha) * column[0]])
        result[:, index] = lfilter(
            [alpha], [1.0, -(1.0 - alpha)], column, zi=initial_state)[0]
    return result


def _force_torque(data: dict[str, Any], parameters: np.ndarray,
                  tau_x_s: float, tau_y_s: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = _steady_tire_magnitudes(data, parameters)
    fx = _causal_filter(fx, tau_x_s, DT_S)
    fy = _causal_filter(fy, tau_y_s, DT_S)
    force = np.zeros_like(data["force_target_total"])
    torque = np.zeros_like(data["torque_target_total"])
    for index, wheel in enumerate(data["wheels"]):
        tire = (fx[:, index, None] * wheel["forward"]
                + fy[:, index, None] * wheel["sideways"])
        normal = wheel["load"][:, None] * wheel["normal"]
        total = tire + normal
        force += total
        torque += np.cross(wheel["lever"], total)
    return force, torque


def _axis_scales(data: dict[str, Any], rows: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    force_scale = np.maximum(np.sqrt(np.mean(
        data["force_target_total"][rows] ** 2, axis=0)), 1.0)
    moment_scale = np.maximum(np.sqrt(np.mean(
        data["torque_target_total"][rows] ** 2, axis=0)), 0.02)
    return force_scale, moment_scale


def _fit(data: dict[str, Any], parameters: np.ndarray
         ) -> dict[str, Any]:
    rows = data["rows"][::FIT_ROW_STRIDE]
    force_scale, moment_scale = _axis_scales(data, rows)

    def residual(candidate: np.ndarray) -> np.ndarray:
        force, torque = _force_torque(
            data, parameters, float(candidate[0]), float(candidate[1]))
        return np.concatenate((
            ((force[rows] - data["force_target_total"][rows])
             / force_scale).ravel(),
            ((torque[rows] - data["torque_target_total"][rows])
             / moment_scale).ravel(),
        ))

    result = least_squares(
        residual, np.asarray([0.005, 0.005]),
        bounds=(np.asarray([0.0, 0.0]),
                np.asarray([MAX_RELAXATION_S, MAX_RELAXATION_S])),
        loss="soft_l1", f_scale=1.0, max_nfev=40,
    )
    return {
        "tau_longitudinal_s": float(result.x[0]),
        "tau_lateral_s": float(result.x[1]),
        "training_rows": int(len(rows)),
        "source_valid_rows": int(len(data["rows"])),
        "fit_row_stride": FIT_ROW_STRIDE,
        "max_tau_s": MAX_RELAXATION_S,
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "normalized_robust_cost": float(result.cost),
        "force_axis_scales_n": force_scale.tolist(),
        "moment_axis_scales_nm": moment_scale.tolist(),
    }


def _score(data: dict[str, Any], parameters: np.ndarray,
           tau_x_s: float, tau_y_s: float,
           split_name: str) -> dict[str, Any]:
    force, torque = _force_torque(data, parameters, tau_x_s, tau_y_s)
    rows = data["rows"]
    rear_slip = np.maximum(
        np.abs(data["wheels"][2]["slip_x"][rows]),
        np.abs(data["wheels"][3]["slip_x"][rows]))
    speed = data["speed"][rows]
    steering = np.max(np.abs(np.column_stack([
        wheel["steering_rad"][rows] for wheel in data["wheels"]])), axis=1)
    phase = np.full(len(rows), "other", dtype=object)
    sample_ids = rows
    phase[(sample_ids >= 167000) & (sample_ids < 167600)] = "event_pre"
    phase[(sample_ids >= 167600) & (sample_ids < 167800)] = "event_spin_onset"
    phase[(sample_ids >= 167800) & (sample_ids < 168100)] = "event_slip_unwind"
    phase[(sample_ids >= 168100) & (sample_ids < 168400)] = "event_lateral_build"
    if split_name != "r11":
        phase[:] = "other"

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        selected_rows = rows[mask]
        if len(selected_rows) == 0:
            return {"samples": 0}
        f_error = force[selected_rows] - data["force_target_total"][selected_rows]
        m_error = torque[selected_rows] - data["torque_target_total"][selected_rows]
        return {
            "samples": int(len(selected_rows)),
            "force_axis_rmse_n": np.sqrt(np.mean(f_error**2, axis=0)).tolist(),
            "force_vector_rmse_n": float(np.sqrt(np.mean(
                np.sum(f_error**2, axis=1)))),
            "moment_axis_rmse_nm": np.sqrt(np.mean(m_error**2, axis=0)).tolist(),
            "yaw_moment_rmse_nm": float(np.sqrt(np.mean(m_error[:, 1]**2))),
            "yaw_moment_bias_nm": float(np.mean(m_error[:, 1])),
        }

    strata: dict[str, Any] = {"all_valid": metrics(np.ones(len(rows), bool))}
    for name, mask in (
        ("rear_longitudinal_slip_ge_0p15", rear_slip >= SPIN_PEAK),
        ("speed_1_3", (speed >= 1) & (speed < 3)),
        ("speed_3_6", (speed >= 3) & (speed < 6)),
        ("speed_6_9", (speed >= 6) & (speed < 9)),
        ("speed_9_12", (speed >= 9) & (speed <= 12.1)),
        ("steer_lt_0p15", steering < 0.15),
        ("steer_0p15_0p30", (steering >= 0.15) & (steering < 0.30)),
        ("steer_ge_0p30", steering >= 0.30),
    ):
        strata[name] = metrics(mask)
    for name in ("event_pre", "event_spin_onset", "event_slip_unwind",
                 "event_lateral_build"):
        strata[name] = metrics(phase == name)
    return {
        "capture": split_name,
        "tau_longitudinal_s": tau_x_s,
        "tau_lateral_s": tau_y_s,
        "strata": strata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    for path in (args.train, args.heldout, args.tire_fit):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    train = _balance_data(args.train, derivative_window=31, sample_stride=1)
    heldout = _balance_data(args.heldout, derivative_window=31, sample_stride=1)
    parameters = _tire_parameters(args.tire_fit)
    fit = _fit(train, parameters)
    base = {"tau_longitudinal_s": 0.0, "tau_lateral_s": 0.0}
    candidate = {"tau_longitudinal_s": fit["tau_longitudinal_s"],
                 "tau_lateral_s": fit["tau_lateral_s"]}
    report = {
        "schema_version": 1,
        "status": "force-balance identification only; must pass command-only recursive rollout before use",
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
        "heldout_capture": str(args.heldout.relative_to(ROOT)),
        "heldout_capture_sha256": _sha256(args.heldout),
        "tire_curve_fit": str(args.tire_fit.relative_to(ROOT)),
        "fit": fit,
        "train_force_balance": {
            "unfiltered_source_curves": _score(train, parameters, 0.0, 0.0,
                                                "r07"),
            "causal_relaxation_candidate": _score(
                train, parameters, candidate["tau_longitudinal_s"],
                candidate["tau_lateral_s"], "r07"),
        },
        "heldout_force_balance": {
            "unfiltered_source_curves": _score(heldout, parameters, 0.0, 0.0,
                                                "r11"),
            "causal_relaxation_candidate": _score(
                heldout, parameters, candidate["tau_longitudinal_s"],
                candidate["tau_lateral_s"], "r11"),
        },
        "wheelhit_slip_load_and_contact_frames_used_for_fit": True,
        "heldout_truth_used_only_for_scoring": True,
        "recursive_rollout_not_yet_tested": True,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "fit": fit,
        "r11_baseline": report["heldout_force_balance"][
            "unfiltered_source_curves"]["strata"]["all_valid"],
        "r11_candidate": report["heldout_force_balance"][
            "causal_relaxation_candidate"]["strata"]["all_valid"],
        "r11_high_spin_baseline": report["heldout_force_balance"][
            "unfiltered_source_curves"]["strata"][
                "rear_longitudinal_slip_ge_0p15"],
        "r11_high_spin_candidate": report["heldout_force_balance"][
            "causal_relaxation_candidate"]["strata"][
                "rear_longitudinal_slip_ge_0p15"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
