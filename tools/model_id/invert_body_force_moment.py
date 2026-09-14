#!/usr/bin/env python3
"""Offline effective body force/moment inversion from a Unity trace.

The trace exposes wheel slips, contact directions and a scalar WheelHit.force,
but not tire-frame Fx/Fy. This tool therefore does not claim to recover the
actual per-tire forces. It compares the configured Unity curve-shaped force
bases against the measured body acceleration and yaw acceleration, then fits
only bounded effective scale factors for an offline structural screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import lsq_linear

try:
    from wheel_friction_curve import TireCurveParameters, friction_value
except ModuleNotFoundError:
    from tools.model_id.wheel_friction_curve import (  # type: ignore
        TireCurveParameters,
        friction_value,
    )


GRAVITY_MPS2 = 9.81
MIN_SPEED_MPS = 1.0
BRAKE_THRESHOLD_NM = 0.01


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _stats(values: Iterable[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=float)
    if finite.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "p50": None,
                "p95": None, "max": None, "bias": None}
    return {
        "count": int(finite.size),
        "mae": float(np.mean(np.abs(finite))),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "p50": float(np.percentile(np.abs(finite), 50)),
        "p95": float(np.percentile(np.abs(finite), 95)),
        "max": float(np.max(np.abs(finite))),
        "bias": float(np.mean(finite)),
    }


def _quat_inverse_rotate(quaternion: tuple[float, float, float, float],
                         vector: np.ndarray) -> np.ndarray:
    """Rotate a world Unity vector into the body frame."""
    x, y, z, w = quaternion
    q = np.asarray((-x, -y, -z), dtype=float)
    v = np.asarray(vector, dtype=float)
    t = 2.0 * np.cross(q, v)
    return v + w * t + np.cross(q, t)


def _body_inertia_tensor(principal: dict[str, float],
                         rotation: dict[str, float]) -> np.ndarray:
    """Project Unity principal moments into the Rigidbody body frame."""
    x = float(rotation["x"])
    y = float(rotation["y"])
    z = float(rotation["z"])
    w = float(rotation["w"])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("inertia tensor rotation has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation_matrix = np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])
    return rotation_matrix @ np.diag([
        float(principal["x"]), float(principal["y"]), float(principal["z"]),
    ]) @ rotation_matrix.T


def _unity_to_api(vector: np.ndarray) -> tuple[float, float]:
    """Convert Unity body x-right/z-forward to API forward/lateral."""
    return float(vector[2]), float(-vector[0])


def _curve(payload: dict[str, Any], key: str) -> TireCurveParameters:
    source = payload[key]
    return TireCurveParameters(
        extremum_slip=float(source["extremumSlip"]),
        extremum_value=float(source["extremumValue"]),
        asymptote_slip=float(source["asymptoteSlip"]),
        asymptote_value=float(source["asymptoteValue"]),
        stiffness=float(source["stiffness"]),
        source="unity_runtime_diagnostic_dump",
    )


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.gradient(values, times)
    dts = np.diff(times)
    positive = dts[dts > 0.0]
    median_dt = float(np.median(positive)) if positive.size else 0.0
    if (len(values) >= 21 and median_dt > 0.0 and
            float(np.max(dts)) <= 3.0 * median_dt):
        from scipy.signal import savgol_filter
        return savgol_filter(values, 21, 2, deriv=1,
                             delta=median_dt, mode="interp")
    return result


def _load_rows(trace: Path) -> list[dict[str, str]]:
    with trace.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("trace contains no rows")
    return rows


def _causal_load_prediction(times: np.ndarray, ax: np.ndarray, ay: np.ndarray,
                            dump_payload: dict[str, Any],
                            load_payload: dict[str, Any]) -> np.ndarray:
    """Propagate the fitted load states and conserve total supported load."""
    vehicle = dump_payload["vehicle"]
    base = np.asarray([
        float(wheel.get("sprungMass", 0.0)) * 9.81
        for wheel in vehicle["wheels"]
    ], dtype=float)
    left_right = load_payload["causal_states"]["left_right"]["parameters"]
    front_rear = load_payload["causal_states"]["front_rear"]["parameters"]
    lr_bias = float(left_right["steady_bias_N"])
    lr_gain = float(left_right["transfer_gain_N_per_mps2"])
    lr_tau = float(left_right["time_constant_s"])
    fr_bias = float(front_rear["steady_bias_N"])
    fr_gain = float(front_rear["transfer_gain_N_per_mps2"])
    fr_tau = float(front_rear["time_constant_s"])
    prediction = np.zeros((len(times), 4), dtype=float)
    z_lr = lr_bias
    z_fr = fr_bias
    for index, time_s in enumerate(times):
        if index:
            dt = max(0.0, min(0.02, time_s - times[index - 1]))
            z_lr += dt / max(lr_tau, 1.0e-6) * (
                lr_bias + lr_gain * ay[index - 1] - z_lr)
            z_fr += dt / max(fr_tau, 1.0e-6) * (
                fr_bias + fr_gain * ax[index - 1] - z_fr)
        current = base.copy()
        current[0] += z_lr / 4.0 + z_fr / 4.0
        current[1] -= z_lr / 4.0 + z_fr / 4.0
        current[2] += z_lr / 4.0 - z_fr / 4.0
        current[3] -= z_lr / 4.0 - z_fr / 4.0
        prediction[index] = np.maximum(0.0, current)
    return prediction


def _fit_scale(basis: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    denominator = float(np.dot(basis, basis))
    signed_scale = (float(np.dot(basis, target)) / denominator
                    if denominator > 1.0e-12 else 0.0)
    scale = max(0.0, signed_scale)
    raw_residual = basis - target
    fitted_residual = scale * basis - target
    return {
        "basis_rms": float(np.sqrt(np.mean(basis * basis))),
        "target_rms": float(np.sqrt(np.mean(target * target))),
        "raw_scale": 1.0,
        "signed_least_squares_scale": signed_scale,
        "nonnegative_scale": scale,
        "raw_residual": _stats(raw_residual),
        "scaled_residual": _stats(fitted_residual),
    }


def _fit_axle_scales(axle_long: np.ndarray, axle_lat: np.ndarray,
                     axle_moment_long: np.ndarray,
                     axle_moment_lat: np.ndarray,
                     target_long: np.ndarray,
                     target_lat: np.ndarray, target_moment: np.ndarray) -> dict[str, Any]:
    """Fit shared front/rear force scales against force and yaw moment."""
    samples = len(target_long)
    design = np.zeros((3 * samples, 4), dtype=float)
    target = np.zeros(3 * samples, dtype=float)
    for index in range(samples):
        force_row = 3 * index
        lateral_row = force_row + 1
        moment_row = force_row + 2
        design[force_row, 0:2] = axle_long[index]
        design[lateral_row, 2:4] = axle_lat[index]
        design[moment_row, 0:2] = axle_moment_long[index]
        design[moment_row, 2:4] = axle_moment_lat[index]
        target[force_row] = target_long[index]
        target[lateral_row] = target_lat[index]
        target[moment_row] = target_moment[index]
    force_norm = max(float(np.sqrt(np.mean(target[0::3] ** 2))), 1.0)
    lateral_norm = max(float(np.sqrt(np.mean(target[1::3] ** 2))), 1.0)
    moment_norm = max(float(np.sqrt(np.mean(target[2::3] ** 2))), 0.01)
    weights = np.tile((1.0 / force_norm, 1.0 / lateral_norm,
                       1.0 / moment_norm), samples)
    weighted_design = design * weights[:, None]
    weighted_target = target * weights
    result = lsq_linear(weighted_design, weighted_target,
                        bounds=(0.0, np.inf), lsmr_tol="auto")
    residual = design @ result.x - target
    return {
        "parameters": {
            "front_longitudinal_scale": float(result.x[0]),
            "rear_longitudinal_scale": float(result.x[1]),
            "front_lateral_scale": float(result.x[2]),
            "rear_lateral_scale": float(result.x[3]),
        },
        "residual_all_equations": _stats(residual),
        "residual_longitudinal_force": _stats(residual[0::3]),
        "residual_lateral_force": _stats(residual[1::3]),
        "residual_yaw_moment": _stats(residual[2::3]),
        "equation_normalization": {
            "longitudinal_force_rms_N": force_norm,
            "lateral_force_rms_N": lateral_norm,
            "yaw_moment_rms_Nm": moment_norm,
            "purpose": "balance force and moment residuals during the joint screen",
        },
        "optimizer": {
            "status": int(result.status),
            "message": result.message,
            "optimality": float(result.optimality),
            "active_mask": result.active_mask.tolist(),
        },
    }


def _fit_axle_force_only_and_check_moment(
        axle_long: np.ndarray, axle_lat: np.ndarray,
        axle_moment_long: np.ndarray, axle_moment_lat: np.ndarray,
        target_long: np.ndarray, target_lat: np.ndarray,
        target_moment: np.ndarray,
        alternate_target_moment: np.ndarray | None = None) -> dict[str, Any]:
    """Fit force balance only, then test the implied yaw moment."""
    samples = len(target_long)
    design = np.zeros((2 * samples, 4), dtype=float)
    target = np.zeros(2 * samples, dtype=float)
    for index in range(samples):
        design[2 * index, 0:2] = axle_long[index]
        design[2 * index + 1, 2:4] = axle_lat[index]
        target[2 * index] = target_long[index]
        target[2 * index + 1] = target_lat[index]
    result = lsq_linear(design, target, bounds=(0.0, np.inf), lsmr_tol="auto")
    force_residual = design @ result.x - target
    predicted_moment = (
        axle_moment_long @ result.x[0:2] +
        axle_moment_lat @ result.x[2:4])
    moment_residual = predicted_moment - target_moment
    report = {
        "parameters": {
            "front_longitudinal_scale": float(result.x[0]),
            "rear_longitudinal_scale": float(result.x[1]),
            "front_lateral_scale": float(result.x[2]),
            "rear_lateral_scale": float(result.x[3]),
        },
        "force_fit": {
            "longitudinal_residual": _stats(force_residual[0::2]),
            "lateral_residual": _stats(force_residual[1::2]),
        },
        "implied_yaw_moment_check": {
            "predicted_moment": _stats(predicted_moment),
            "target_moment": _stats(target_moment),
            "residual": _stats(moment_residual),
        },
        "optimizer": {
            "status": int(result.status),
            "message": result.message,
            "optimality": float(result.optimality),
            "active_mask": result.active_mask.tolist(),
        },
    }
    if alternate_target_moment is not None:
        alternate_residual = predicted_moment - alternate_target_moment
        report["implied_alternate_yaw_moment_check"] = {
            "target_moment": _stats(alternate_target_moment),
            "residual": _stats(alternate_residual),
        }
    return report


def _regime_name(sx: float, sy: float) -> str:
    abs_sx = abs(sx)
    abs_sy = abs(sy)
    if abs_sx < 0.02 and abs_sy < 0.01:
        return "low_slip"
    if abs_sx >= 0.10 and abs_sx < 0.25:
        return "moderate_longitudinal_slip"
    if abs_sy >= 0.15:
        return "high_lateral_slip"
    if abs_sx >= 0.25:
        return "high_longitudinal_slip"
    if abs_sy >= 0.05:
        return "moderate_lateral_slip"
    return "ordinary_combined_or_transition"


def analyze(trace_path: Path, dump_path: Path,
            load_model_path: Path | None = None) -> dict[str, Any]:
    rows = _load_rows(trace_path)
    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    vehicle = payload["vehicle"]
    rigid_body = vehicle["rigidBody"]
    mass_kg = float(rigid_body["mass"])
    if rigid_body.get("yawAxis") != "body_y":
        raise ValueError(
            "diagnostic dump must explicitly declare Unity body Y as yaw axis")
    inertia_tensor = _body_inertia_tensor(
        rigid_body["inertiaTensor"], rigid_body["inertiaTensorRotation"])
    unity_yaw_inertia = float(inertia_tensor[1, 1])
    reported_yaw_inertia = float(rigid_body["yawInertiaBodyFrame"])
    if not math.isclose(
            unity_yaw_inertia, reported_yaw_inertia,
            rel_tol=1.0e-4, abs_tol=2.0e-6):
        raise ValueError(
            "diagnostic yawInertiaBodyFrame does not match corrected body-Y "
            f"inertia: tensor={unity_yaw_inertia} reported={reported_yaw_inertia}")
    for axis_index, axis in enumerate(("X", "Y", "Z")):
        field = f"bodyInertia{axis}"
        if field not in rigid_body:
            raise ValueError(f"diagnostic dump is missing corrected {field} field")
        if not math.isclose(
                float(np.diag(inertia_tensor)[axis_index]),
                float(rigid_body[field]), rel_tol=1.0e-4, abs_tol=2.0e-6):
            raise ValueError(f"diagnostic {field} disagrees with inertia tensor")
    yaw_inertia = unity_yaw_inertia
    com = rigid_body["centerOfMass"]
    wheel_geometry: list[tuple[float, float]] = []
    wheel_curves: list[tuple[TireCurveParameters, TireCurveParameters]] = []
    for wheel in vehicle["wheels"]:
        position = wheel["positionVehicleFrame"]
        # Wheel position and COM are Unity body x-right/z-forward. Convert to
        # API x-forward/y-left relative to the Rigidbody COM.
        wheel_geometry.append((
            float(position["z"]) - float(com["z"]),
            -float(position["x"]) + float(com["x"]),
        ))
        wheel_curves.append((
            _curve(wheel, "forwardFriction"),
            _curve(wheel, "sidewaysFriction"),
        ))

    times = np.asarray([_float(row, "fixed_time_s") or 0.0 for row in rows],
                       dtype=float)
    u = np.asarray([_float(row, "body_velocity_z_mps") or 0.0 for row in rows],
                   dtype=float)
    v = np.asarray([_float(row, "body_velocity_x_mps") or 0.0 for row in rows],
                   dtype=float)
    yaw_rate = np.asarray([
        _float(row, "body_angular_velocity_y_radps") or 0.0 for row in rows
    ], dtype=float)
    u_dot = _derivative(u, times)
    v_dot = _derivative(v, times)
    yaw_dot = _derivative(yaw_rate, times)
    measured_long = mass_kg * (u_dot - yaw_rate * v)
    measured_lat = mass_kg * (v_dot + yaw_rate * u)
    measured_moment = yaw_inertia * yaw_dot
    measured_moment_unity_yaw = unity_yaw_inertia * yaw_dot
    load_payload = (json.loads(load_model_path.read_text(encoding="utf-8"))
                    if load_model_path else None)
    predicted_loads = (_causal_load_prediction(
        times, u_dot - yaw_rate * v, v_dot + yaw_rate * u, payload,
        load_payload) if load_payload else None)

    basis_long: list[float] = []
    basis_lat: list[float] = []
    basis_moment: list[float] = []
    axle_long: list[list[float]] = []
    axle_lat: list[list[float]] = []
    axle_moment_long: list[list[float]] = []
    axle_moment_lat: list[list[float]] = []
    target_long: list[float] = []
    target_lat: list[float] = []
    target_moment: list[float] = []
    target_moment_unity_yaw: list[float] = []
    slips: list[tuple[float, float]] = []
    regime_rows: dict[str, dict[str, list[float]]] = {}
    accepted_rows = 0
    rejected_rows = 0

    for index, row in enumerate(rows):
        if abs(u[index]) < MIN_SPEED_MPS:
            rejected_rows += 1
            continue
        if any((_float(row, f"wheel{wheel}_grounded") or 0.0) < 0.5
               for wheel in range(4)):
            rejected_rows += 1
            continue
        if any((_float(row, f"wheel{wheel}_brake_torque_nm") or 0.0) >
               BRAKE_THRESHOLD_NM for wheel in range(4)):
            rejected_rows += 1
            continue

        total_force = np.zeros(2, dtype=float)
        total_moment = 0.0
        force_by_axle = np.zeros((2, 2), dtype=float)
        moment_long_by_axle = np.zeros(2, dtype=float)
        moment_lat_by_axle = np.zeros(2, dtype=float)
        all_slips: list[tuple[float, float]] = []
        row_quaternion = np.asarray([
            _float(row, "root_rotation_x") or 0.0,
            _float(row, "root_rotation_y") or 0.0,
            _float(row, "root_rotation_z") or 0.0,
            _float(row, "root_rotation_w") or 1.0,
        ])
        valid = True
        for wheel in range(4):
            prefix = f"wheel{wheel}_"
            force = _float(row, prefix + "contact_force_n")
            normal_y = _float(row, prefix + "contact_normal_y")
            sx = _float(row, prefix + "forward_slip")
            sy = _float(row, prefix + "sideways_slip")
            if None in (force, normal_y, sx, sy):
                valid = False
                break
            measured_fz = max(0.0, float(force) * float(normal_y))
            fz = (float(predicted_loads[index, wheel])
                  if predicted_loads is not None else measured_fz)
            forward = np.asarray([
                _float(row, prefix + "forward_dir_x") or 0.0,
                _float(row, prefix + "forward_dir_y") or 0.0,
                _float(row, prefix + "forward_dir_z") or 0.0,
            ])
            sideways = np.asarray([
                _float(row, prefix + "sideways_dir_x") or 0.0,
                _float(row, prefix + "sideways_dir_y") or 0.0,
                _float(row, prefix + "sideways_dir_z") or 0.0,
            ])
            forward_api = _unity_to_api(
                _quat_inverse_rotate(tuple(row_quaternion), forward))
            sideways_api = _unity_to_api(
                _quat_inverse_rotate(tuple(row_quaternion), sideways))
            forward_curve, lateral_curve = wheel_curves[wheel]
            # The WheelHit slip signs in this trace are aligned with the
            # measured body-force response: positive longitudinal slip is
            # retained as positive forward force. This is an effective
            # diagnostic basis, not a claim about the private PhysX sign.
            fx = fz * friction_value(float(sx), forward_curve)
            fy = fz * friction_value(float(sy), lateral_curve)
            force_api = np.asarray(forward_api) * fx + np.asarray(sideways_api) * fy
            total_force += force_api
            x_m, y_m = wheel_geometry[wheel]
            total_moment += x_m * force_api[1] - y_m * force_api[0]
            axle = 0 if wheel < 2 else 1
            force_by_axle[axle] += force_api
            moment_long_by_axle[axle] += -y_m * force_api[0]
            moment_lat_by_axle[axle] += x_m * force_api[1]
            all_slips.append((float(sx), float(sy)))
        if not valid:
            rejected_rows += 1
            continue
        accepted_rows += 1
        basis_long.append(float(total_force[0]))
        basis_lat.append(float(total_force[1]))
        basis_moment.append(float(total_moment))
        axle_long.append([float(force_by_axle[0, 0]),
                          float(force_by_axle[1, 0])])
        axle_lat.append([float(force_by_axle[0, 1]),
                         float(force_by_axle[1, 1])])
        axle_moment_long.append([float(moment_long_by_axle[0]),
                                 float(moment_long_by_axle[1])])
        axle_moment_lat.append([float(moment_lat_by_axle[0]),
                                float(moment_lat_by_axle[1])])
        target_long.append(float(measured_long[index]))
        target_lat.append(float(measured_lat[index]))
        target_moment.append(float(measured_moment[index]))
        target_moment_unity_yaw.append(float(measured_moment_unity_yaw[index]))
        mean_sx = sum(sx for sx, _ in all_slips) / len(all_slips)
        mean_sy = sum(sy for _, sy in all_slips) / len(all_slips)
        regime = _regime_name(mean_sx, mean_sy)
        bucket = regime_rows.setdefault(regime, {
            "basis_long": [], "basis_lat": [], "basis_moment": [],
            "target_long": [], "target_lat": [], "target_moment": [],
        })
        bucket["basis_long"].append(float(total_force[0]))
        bucket["basis_lat"].append(float(total_force[1]))
        bucket["basis_moment"].append(float(total_moment))
        bucket["target_long"].append(float(measured_long[index]))
        bucket["target_lat"].append(float(measured_lat[index]))
        bucket["target_moment"].append(float(measured_moment[index]))
        slips.extend(all_slips)

    long_basis = np.asarray(basis_long, dtype=float)
    lat_basis = np.asarray(basis_lat, dtype=float)
    moment_basis = np.asarray(basis_moment, dtype=float)
    long_target = np.asarray(target_long, dtype=float)
    lat_target = np.asarray(target_lat, dtype=float)
    moment_target = np.asarray(target_moment, dtype=float)
    moment_target_unity_yaw = np.asarray(target_moment_unity_yaw, dtype=float)
    axle_long_array = np.asarray(axle_long, dtype=float)
    axle_lat_array = np.asarray(axle_lat, dtype=float)
    axle_moment_long_array = np.asarray(axle_moment_long, dtype=float)
    axle_moment_lat_array = np.asarray(axle_moment_lat, dtype=float)

    inversion = {
        "longitudinal_force_N": _fit_scale(long_basis, long_target),
        "lateral_force_N": _fit_scale(lat_basis, lat_target),
        "yaw_moment_Nm": _fit_scale(moment_basis, moment_target),
        "yaw_moment_unity_y_axis_Nm": _fit_scale(
            moment_basis, moment_target_unity_yaw),
    }
    axle_inversion = _fit_axle_scales(
        axle_long_array, axle_lat_array, axle_moment_long_array,
        axle_moment_lat_array,
        long_target, lat_target, moment_target)
    axle_force_fit = _fit_axle_force_only_and_check_moment(
        axle_long_array, axle_lat_array, axle_moment_long_array,
        axle_moment_lat_array, long_target, lat_target, moment_target,
        moment_target_unity_yaw)
    per_regime: dict[str, Any] = {}
    for regime, bucket in sorted(regime_rows.items()):
        per_regime[regime] = {
            "samples": len(bucket["target_long"]),
            "longitudinal_force": _fit_scale(
                np.asarray(bucket["basis_long"]), np.asarray(bucket["target_long"])),
            "lateral_force": _fit_scale(
                np.asarray(bucket["basis_lat"]), np.asarray(bucket["target_lat"])),
            "yaw_moment": _fit_scale(
                np.asarray(bucket["basis_moment"]), np.asarray(bucket["target_moment"])),
        }

    return {
        "schema_version": 1,
        "status": "body_force_moment_inversion_offline_only",
        "acceptance": "effective_structural_screen_only",
        "offline_only": True,
        "future_ground_truth_used": False,
        "input": {"trace": str(trace_path), "static_dump": str(dump_path),
                  "causal_load_model": (str(load_model_path)
                                         if load_model_path else None)},
        "contact_load_source": (
            "causal_load_state_propagated_from_source_acceleration"
            if load_model_path else "measured_WheelHit_force_times_normal_y"),
        "structural_parameters": {
            "rigidbody_mass_kg": mass_kg,
            "yaw_inertia_kgm2": yaw_inertia,
            "raw_body_inertia_tensor_kgm2": inertia_tensor.tolist(),
            "body_frame_inertia_kgm2": {
                "x": float(inertia_tensor[0, 0]),
                "y": float(inertia_tensor[1, 1]),
                "z": float(inertia_tensor[2, 2]),
            },
            "unity_physical_yaw_axis": "body_y",
            "exported_yawInertiaBodyFrame_projection_axis": "body_y",
            "exported_vs_physical_yaw_axis_mismatch": False,
            "wheel_geometry_api_body_frame": [
                {"x_m": x, "y_m": y} for x, y in wheel_geometry],
            "force_basis": "dumped WheelCollider curves and WheelHit.force*normal_y",
        },
        "source_time": {
            "derivative_time_field": "fixed_time_s",
            "callback_time_used": False,
            "derivative_smoothing": "Savitzky-Golay 21 samples, offline only",
        },
        "sign_convention": {
            "body_forward": "Unity body +z mapped to API +x",
            "body_lateral": "Unity body -x mapped to API +y",
            "force_basis": "signed slip curve along wheel directions",
            "yaw_moment": "x_forward*Fy_left - y_left*Fx_forward",
        },
        "sign_selection": {
            "initial_negative_force_basis_trial": {
                "result": "all global signed scales were negative",
                "interpretation": "opposite to measured body-force convention",
            },
            "selected_force_basis": "positive signed slip curve",
            "must_be_verified_by_direct_force_export": True,
        },
        "selection": {
            "speed_min_mps": MIN_SPEED_MPS,
            "all_four_wheels_grounded": True,
            "brake_torque_max_nm": BRAKE_THRESHOLD_NM,
            "accepted_rows": accepted_rows,
            "rejected_rows": rejected_rows,
        },
        "global_inversion": inversion,
        "front_rear_joint_inversion": axle_inversion,
        "front_rear_force_fit_and_moment_check": axle_force_fit,
        "regime_inversion": per_regime,
        "slip_samples": {
            "forward": _stats(sx for sx, _ in slips),
            "sideways": _stats(sy for _, sy in slips),
        },
        "limitations": [
            "WheelHit.force is a normal-load proxy, not measured tire Fx/Fy.",
            "Body acceleration and yaw acceleration are differentiated from the diagnostic trace.",
            "The corrected diagnostic exporter explicitly emits all body-axis inertia projections and declares body Y as the physical yaw axis.",
            "A fitted scale is not a tire coefficient or proof of the assumed force sign.",
            "No combined-slip law is promoted from this inversion alone.",
        ],
        "next_action": (
            "Compare force/moment residuals against the same regimes using the "
            "corrected body-Y inertia; add direct force export if the residual "
            "cannot be explained by measured load and configured curves."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-model", type=Path)
    args = parser.parse_args()
    report = analyze(args.trace, args.dump, args.load_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "accepted_rows": report["selection"]["accepted_rows"],
        "global_inversion": report["global_inversion"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
