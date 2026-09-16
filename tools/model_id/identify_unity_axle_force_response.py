#!/usr/bin/env python3
"""Recover an offline Unity axle lateral-force response without tire labels.

The Unity WheelCollider API does not expose the internal per-wheel forward or
sideways solver force. This diagnostic therefore:

1. reconstructs the measured body wrench from the exact Unity trace;
2. removes forward force reconstructed from wheel RPM, motor/brake torque, and
   the separately identified effective wheel rotational state; and
3. solves the remaining body lateral-force and yaw-moment equations for one
   front-axle and one rear-axle force at each usable sample.

The output compares those recovered forces with the serialized Unity
slip-curve/contact-load proxy. Coefficients are observability results only.
They are not friction coefficients, cornering stiffness values, or runtime
MPC parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar

try:
    # Direct command-line execution puts this directory on sys.path.
    from structured_vehicle_plant import unity_wheel_friction_value
except ModuleNotFoundError:
    # Package import is used by the regression tests and notebook tooling.
    from tools.model_id.structured_vehicle_plant import unity_wheel_friction_value


GRAVITY_MPS2 = 9.81
WHEEL_RADIUS_M = 0.059
WHEEL_DAMPING_NMS = 0.25
# Effective WheelCollider rotational response from the prior positive-drive
# screen. These remain explicit inputs to this inversion and are not promoted.
DEFAULT_I_EFFECTIVE = (0.000366, 0.000441, 0.000436)


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = quaternion / np.linalg.norm(quaternion, axis=1)[:, None]
    x, y, z, w = q.T
    result = np.empty((len(q), 3, 3), dtype=float)
    result[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[:, 0, 1] = 2.0 * (x * y - z * w)
    result[:, 0, 2] = 2.0 * (x * z + y * w)
    result[:, 1, 0] = 2.0 * (x * y + z * w)
    result[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[:, 1, 2] = 2.0 * (y * z - x * w)
    result[:, 2, 0] = 2.0 * (x * z - y * w)
    result[:, 2, 1] = 2.0 * (y * z + x * w)
    result[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def _derivative_segment(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Differentiate one continuous, fixed-step trace segment."""
    dt = np.diff(times)
    positive = dt[dt > 0.0]
    if not positive.size:
        return np.gradient(values, times)
    median_dt = float(np.median(positive))
    window = min(101, len(values) if len(values) % 2 else len(values) - 1)
    if window >= 7 and np.max(dt) <= 3.0 * median_dt:
        return savgol_filter(values, window, 2, deriv=1, delta=median_dt,
                             mode="interp")
    return np.gradient(values, times)


def _derivative(values: np.ndarray, times: np.ndarray,
                breakpoints: np.ndarray | None = None) -> np.ndarray:
    """Differentiate trace segments without crossing a simulator reset."""
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if breakpoints is None or not np.any(breakpoints):
        return _derivative_segment(values, times)
    starts = [0] + [int(index) for index in np.flatnonzero(breakpoints)]
    ends = starts[1:] + [len(values)]
    output = np.full_like(values, np.nan, dtype=float)
    for start, end in zip(starts, ends):
        if end > start:
            output[start:end] = _derivative_segment(
                values[start:end], times[start:end])
    # No derivative is valid across the discontinuity. The selector also
    # removes a short neighbourhood because the reset can affect several
    # diagnostic samples even though the state itself is only discontinuous
    # at the boundary.
    output[breakpoints] = np.nan
    return output


def _reset_boundaries(frame: pd.DataFrame) -> np.ndarray:
    """Detect teleports produced by a diagnostic experiment reset.

    The resetter preserves fixed-step and timestamp counters. Consequently a
    timestamp-gap test cannot identify it. A boundary is a large COM jump
    followed by a near-stationary body, with a substantial speed drop. The
    test is deliberately conservative so ordinary 1 kHz motion is retained.
    """
    position = frame[[
        "world_com_x_m", "world_com_y_m", "world_com_z_m",
    ]].to_numpy(float)
    velocity = frame[[
        "world_velocity_x_mps", "world_velocity_y_mps",
        "world_velocity_z_mps",
    ]].to_numpy(float)
    if len(frame) < 2:
        return np.zeros(len(frame), dtype=bool)
    position_jump = np.linalg.norm(np.diff(position, axis=0), axis=1)
    previous_speed = np.linalg.norm(velocity[:-1], axis=1)
    current_speed = np.linalg.norm(velocity[1:], axis=1)
    return np.r_[False, (position_jump > 0.2) &
                 (current_speed < 0.5) &
                 ((previous_speed - current_speed) > 1.0)]


def _reset_exclusion_mask(boundaries: np.ndarray,
                           radius: int = 10) -> np.ndarray:
    """Remove derivative-contaminated rows around reset edges."""
    excluded = np.zeros(len(boundaries), dtype=bool)
    for boundary in np.flatnonzero(boundaries):
        excluded[max(0, int(boundary) - radius):
                 min(len(boundaries), int(boundary) + radius + 1)] = True
    return excluded


def _inertia_matrix(frame: pd.DataFrame,
                    metadata: dict[str, Any]) -> np.ndarray:
    """Return Unity's full body-frame inertia tensor for every sample.

    Unity stores the principal moments and the rotation of that principal
    frame. The yaw-only inversion used by the first screen is useful as a
    baseline, but it drops the roll/pitch coupling terms that are present in
    this rigid body. New diagnostic traces contain the runtime tensor; older
    traces fall back to the serialized snapshot so they remain auditable.
    """
    inertia_columns = {
        "runtime_inertia_tensor_x_kgm2",
        "runtime_inertia_tensor_y_kgm2",
        "runtime_inertia_tensor_z_kgm2",
        "runtime_inertia_rotation_x",
        "runtime_inertia_rotation_y",
        "runtime_inertia_rotation_z",
        "runtime_inertia_rotation_w",
    }
    if inertia_columns.issubset(frame.columns):
        moments = frame[[
            "runtime_inertia_tensor_x_kgm2",
            "runtime_inertia_tensor_y_kgm2",
            "runtime_inertia_tensor_z_kgm2",
        ]].to_numpy(float)
        principal_rotation = _rotation_matrix(frame[[
            "runtime_inertia_rotation_x",
            "runtime_inertia_rotation_y",
            "runtime_inertia_rotation_z",
            "runtime_inertia_rotation_w",
        ]].to_numpy(float))
    else:
        rigid_body = metadata["vehicle"]["rigidBody"]
        moments = np.asarray([[float(rigid_body["inertiaTensor"][axis])
                               for axis in "xyz"]])
        moments = np.repeat(moments, len(frame), axis=0)
        principal = rigid_body["inertiaTensorRotation"]
        quaternion = np.asarray([[float(principal[axis]) for axis in "xyzw"]])
        principal_rotation = np.repeat(
            _rotation_matrix(quaternion), len(frame), axis=0)
    diagonal = np.zeros((len(frame), 3, 3), dtype=float)
    diagonal[:, range(3), range(3)] = moments
    # Unity's body tensor is R * diag(principal moments) * R^T. This
    # construction reproduces Rigidbody's body-y yaw value from the trace.
    return np.einsum("nij,njk,nlk->nil", principal_rotation, diagonal,
                     principal_rotation)


def _curve(slip: np.ndarray, values: tuple[float, ...]) -> np.ndarray:
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = values
    magnitude = np.abs(slip)
    value = np.where(
        magnitude <= extremum_slip,
        extremum_value * magnitude / extremum_slip,
        np.where(
            magnitude <= asymptote_slip,
            extremum_value + (magnitude - extremum_slip) /
            (asymptote_slip - extremum_slip) *
            (asymptote_value - extremum_value),
            asymptote_value,
        ),
    )
    return np.sign(slip) * stiffness * value


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(np.abs(values))),
        "p05": float(np.percentile(values, 5.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _trace_columns() -> list[str]:
    columns = [
        "fixed_time_s", "fixed_step", "world_velocity_x_mps",
        "world_velocity_y_mps", "world_velocity_z_mps",
        "world_angular_velocity_x_radps", "world_angular_velocity_y_radps",
        "world_angular_velocity_z_radps", "root_rotation_x",
        "root_rotation_y", "root_rotation_z", "root_rotation_w",
        "world_com_x_m", "world_com_y_m", "world_com_z_m",
        "applied_steering_rad",
        "applied_throttle_norm",
        "runtime_body_yaw_inertia_kgm2",
    ]
    for wheel in range(4):
        columns.extend([
            f"wheel{wheel}_{field}" for field in (
                "grounded", "rpm", "motor_torque_nm", "brake_torque_nm",
                "steer_angle_deg", "sprung_mass_kg", "world_pose_x_m",
                "world_pose_y_m", "world_pose_z_m",
                "forward_slip", "sideways_slip", "contact_force_n",
                "forward_dir_x", "forward_dir_y", "forward_dir_z",
                "sideways_dir_x", "sideways_dir_y", "sideways_dir_z",
                "contact_point_x_m", "contact_point_y_m",
                "contact_point_z_m", "contact_normal_x",
                "contact_normal_y", "contact_normal_z",
            )
        ])
    return columns


def _body_wrench(frame: pd.DataFrame, metadata: dict[str, Any],
                 angular_drag_model: str,
                 normal_force_model: str,
                 reset_boundaries: np.ndarray | None = None) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
    times = frame.fixed_time_s.to_numpy(float)
    rotations = _rotation_matrix(frame[[
        "root_rotation_x", "root_rotation_y", "root_rotation_z",
        "root_rotation_w",
    ]].to_numpy(float))
    world_velocity = frame[[
        "world_velocity_x_mps", "world_velocity_y_mps",
        "world_velocity_z_mps",
    ]].to_numpy(float)
    world_angular_velocity = frame[[
        "world_angular_velocity_x_radps", "world_angular_velocity_y_radps",
        "world_angular_velocity_z_radps",
    ]].to_numpy(float)
    velocity_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), world_velocity)
    acceleration_world = np.column_stack([
        _derivative(world_velocity[:, axis], times, reset_boundaries)
        for axis in range(3)
    ])
    acceleration_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), acceleration_world)
    angular_velocity_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), world_angular_velocity)
    angular_acceleration_world = np.column_stack([
        _derivative(world_angular_velocity[:, axis], times, reset_boundaries)
        for axis in range(3)
    ])
    angular_acceleration_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1),
        angular_acceleration_world)
    rigid_body = metadata["vehicle"]["rigidBody"]
    mass = float(rigid_body["mass"])
    drag = float(rigid_body["drag"])
    angular_drag = float(rigid_body["angularDrag"])
    inertia = _inertia_matrix(frame, metadata)
    force = mass * acceleration_body
    gravity_body = np.einsum(
        "nij,j->ni", rotations.transpose(0, 2, 1),
        np.asarray((0.0, -GRAVITY_MPS2, 0.0), dtype=float))
    # Newton-Euler balance is reconstructed for contact forces:
    # F_contact = m*a - m*g - F_drag. The horizontal gravity terms are
    # non-zero whenever the Unity body pitches or rolls.
    force -= mass * gravity_body
    if normal_force_model == "measured_contact_normal":
        normal_world = np.zeros_like(force)
        for wheel in range(4):
            load = frame[f"wheel{wheel}_contact_force_n"].to_numpy(float)
            normal = frame[[
                f"wheel{wheel}_contact_normal_{axis}" for axis in "xyz"
            ]].to_numpy(float)
            normal_world += load[:, None] * normal
        normal_body = np.einsum(
            "nij,nj->ni", rotations.transpose(0, 2, 1), normal_world)
        force -= normal_body
    elif normal_force_model != "none":
        raise ValueError(f"unknown normal force model: {normal_force_model}")
    force[:, 0] += mass * drag * velocity_body[:, 0]
    force[:, 2] += mass * drag * velocity_body[:, 2]
    angular_momentum = np.einsum("nij,nj->ni", inertia,
                                  angular_velocity_body)
    angular_moment = (np.einsum("nij,nj->ni", inertia,
                                 angular_acceleration_body) +
                      np.cross(angular_velocity_body, angular_momentum))
    if angular_drag_model == "inertia_scaled":
        angular_moment += angular_drag * angular_momentum
    elif angular_drag_model == "direct":
        angular_moment += angular_drag * angular_velocity_body
    elif angular_drag_model != "none":
        raise ValueError(f"unknown angular drag model: {angular_drag_model}")
    return rotations, velocity_body, np.column_stack((force[:, 0],
                                                        force[:, 2],
                                                        angular_moment[:, 1]))


def _collision_steps(path: Path | None) -> set[int]:
    if path is None:
        return set()
    frame = pd.read_csv(path, usecols=["fixed_step", "event",
                                       "this_collider"])
    frame = frame[(frame.event.isin(["enter", "stay"])) &
                  (frame.this_collider == "Chassis-1-solid1")]
    return set(frame.fixed_step.astype(int).tolist())


def _fit(proxy: np.ndarray, recovered: np.ndarray,
         indices: np.ndarray) -> dict[str, Any]:
    if indices.size < 16:
        return {"identified": False, "samples": int(indices.size)}
    split = indices.size // 2
    train = indices[:split]
    holdout = indices[split:]
    coefficients = np.linalg.lstsq(
        proxy[train], recovered[train], rcond=None)[0]
    error = recovered[holdout] - proxy[holdout] @ coefficients
    return {
        "identified": True,
        "train_rows": int(train.size),
        "holdout_rows": int(holdout.size),
        "train_coefficients": [float(value) for value in coefficients],
        "holdout_error": _stats(error),
    }


def _fit_design(design: np.ndarray, recovered: np.ndarray,
                indices: np.ndarray) -> dict[str, Any]:
    """Fit an explicit observability basis and score its time holdout."""
    if indices.size < max(16, design.shape[1] * 4):
        return {"identified": False, "samples": int(indices.size)}
    split = indices.size // 2
    train = indices[:split]
    holdout = indices[split:]
    coefficients, _, rank, singular = np.linalg.lstsq(
        design[train], recovered[train], rcond=None)
    error = recovered[holdout] - design[holdout] @ coefficients
    return {
        "identified": bool(rank == design.shape[1]),
        "rank": int(rank),
        "train_rows": int(train.size),
        "holdout_rows": int(holdout.size),
        "coefficients": [float(value) for value in coefficients],
        "holdout_error": _stats(error),
        "singular_values": [float(value) for value in singular],
    }


def _fit_load_power(curve_values: np.ndarray, loads: np.ndarray,
                    recovered: np.ndarray, indices: np.ndarray,
                    wheel_indices: tuple[int, int]) -> dict[str, Any]:
    """Screen an explicit power law for Unity load response.

    The normalized form is

      F = gain * 10 N * signed_curve * (load / 10 N)**p

    so p=1 and gain=1 reproduce the direct WheelHit.force proxy. The
    exponent is an explicit solver-response hypothesis, not a tire exponent.
    It is fit on the chronological training half and scored on the other half.
    """
    if indices.size < 32:
        return {"identified": False, "samples": int(indices.size)}
    split = indices.size // 2
    train = indices[:split]
    holdout = indices[split:]

    def basis(exponent: float, selected: np.ndarray) -> np.ndarray:
        selected_load = np.maximum(loads[selected][:, wheel_indices], 1.0e-6)
        return 10.0 * np.sum(
            curve_values[selected][:, wheel_indices] *
            np.power(selected_load / 10.0, exponent), axis=1)

    def fit_gain(exponent: float, selected: np.ndarray) -> float:
        design = basis(exponent, selected)
        return float(np.linalg.lstsq(
            design[:, None], recovered[selected], rcond=None)[0][0])

    def objective(exponent: float) -> float:
        design = basis(exponent, train)
        gain = fit_gain(exponent, train)
        residual = recovered[train] - gain * design
        return float(np.mean(residual * residual))

    fit = minimize_scalar(
        objective, bounds=(0.10, 2.00), method="bounded",
        options={"xatol": 1.0e-5})
    exponent = float(fit.x)
    gain = fit_gain(exponent, train)
    train_error = recovered[train] - gain * basis(exponent, train)
    holdout_error = recovered[holdout] - gain * basis(exponent, holdout)
    return {
        "identified": bool(fit.success),
        "train_rows": int(train.size),
        "holdout_rows": int(holdout.size),
        "load_exponent_p": exponent,
        "gain": gain,
        "normalized_load_reference_n": 10.0,
        "train_rmse": float(np.sqrt(np.mean(train_error * train_error))),
        "holdout_error": _stats(holdout_error),
        "direct_proxy_p_equals_1": {
            "gain": fit_gain(1.0, train),
            "train_rmse": float(np.sqrt(np.mean(
                (recovered[train] - fit_gain(1.0, train) *
                 basis(1.0, train)) ** 2))),
            "holdout_error": _stats(
                recovered[holdout] - fit_gain(1.0, train) *
                basis(1.0, holdout)),
        },
        "optimization_objective": "chronological training RMSE",
        "promotion": False,
    }


def analyze(trace: Path | list[Path], static: Path, output: Path,
            collision: Path | None = None,
            angular_drag_model: str = "inertia_scaled",
            normal_force_model: str = "measured_contact_normal",
            effective_inertia_values: tuple[float, float, float] =
            DEFAULT_I_EFFECTIVE,
            load_mode: str = "wheel_hit_force",
            rows_output: Path | None = None) -> dict[str, Any]:
    metadata = json.loads(static.read_text(encoding="utf-8"))
    if load_mode not in {"wheel_hit_force", "runtime_sprung_load",
                         "static_sprung_load"}:
        raise ValueError(f"unknown load mode: {load_mode}")
    requested_columns = _trace_columns()
    traces = trace if isinstance(trace, list) else [trace]
    frames = []
    for path in traces:
        available = set(pd.read_csv(path, nrows=0).columns)
        missing = set(requested_columns) - available
        optional = {
            field for field in requested_columns
            if (field.startswith("runtime_") or
                field.endswith("_sprung_mass_kg") or
                "_steer_angle_deg" in field or
                "_world_pose_" in field)
        }
        required = missing - optional
        if required:
            raise ValueError(f"trace lacks required columns: {sorted(required)}")
        frames.append(pd.read_csv(
            path, usecols=[field for field in requested_columns
                           if field in available]))
    frame = pd.concat(frames, ignore_index=True, sort=False)
    times = frame.fixed_time_s.to_numpy(float)
    reset_boundaries = _reset_boundaries(frame)
    reset_excluded = _reset_exclusion_mask(reset_boundaries)
    if load_mode == "runtime_sprung_load" and any(
            f"wheel{wheel}_sprung_mass_kg" not in frame.columns
            for wheel in range(4)):
        raise ValueError("runtime_sprung_load requires runtime sprung-mass fields")
    rotations, velocity_body, target = _body_wrench(
        frame, metadata, angular_drag_model, normal_force_model,
        reset_boundaries)
    count = len(frame)
    forward_wrench = np.zeros((count, 2), dtype=float)
    forward_force_body_z = np.zeros(count, dtype=float)
    side_matrix = np.zeros((count, 2, 2), dtype=float)
    proxy = np.zeros((count, 2), dtype=float)
    curve_values = np.zeros((count, 4), dtype=float)
    forward_curve_values = np.zeros((count, 4), dtype=float)
    wheel_omega = np.zeros((count, 4), dtype=float)
    wheel_angular_acceleration = np.zeros((count, 4), dtype=float)
    wheel_forward_force = np.zeros((count, 4), dtype=float)
    effective_inertia = np.zeros((count, 4), dtype=float)
    sideways_slips = np.zeros((count, 4), dtype=float)
    forward_slips = np.zeros((count, 4), dtype=float)
    contact_loads = np.zeros((count, 4), dtype=float)
    contact_arms_body = np.zeros((count, 4, 3), dtype=float)
    contact_normals_body = np.zeros((count, 4, 3), dtype=float)
    forward_dirs_body = np.zeros((count, 4, 3), dtype=float)
    sideways_dirs_body = np.zeros((count, 4, 3), dtype=float)
    wheel_pose_body_from_com = np.zeros((count, 4, 3), dtype=float)
    wheel_pose_body_y_rate = np.zeros((count, 4), dtype=float)
    for wheel in range(4):
        wheel_pose_world = frame[[
            f"wheel{wheel}_world_pose_{axis}_m" for axis in "xyz"
        ]].to_numpy(float)
        world_com = frame[[
            "world_com_x_m", "world_com_y_m", "world_com_z_m"
        ]].to_numpy(float)
        wheel_pose_body = np.einsum(
            "nij,nj->ni", rotations.transpose(0, 2, 1),
            wheel_pose_world - world_com)
        wheel_pose_body_from_com[:, wheel, :] = wheel_pose_body
        wheel_pose_body_y_rate[:, wheel] = _derivative(
            wheel_pose_body[:, 1], times, reset_boundaries)
        forward_direction = np.einsum(
            "nij,nj->ni", rotations.transpose(0, 2, 1), frame[[
                f"wheel{wheel}_forward_dir_{axis}" for axis in "xyz"
            ]].to_numpy(float))
        sideways_direction = np.einsum(
            "nij,nj->ni", rotations.transpose(0, 2, 1), frame[[
                f"wheel{wheel}_sideways_dir_{axis}" for axis in "xyz"
            ]].to_numpy(float))
        contact_normal = np.einsum(
            "nij,nj->ni", rotations.transpose(0, 2, 1), frame[[
                f"wheel{wheel}_contact_normal_{axis}" for axis in "xyz"
            ]].to_numpy(float))
        contact_arm = np.einsum(
            "nij,nj->ni", rotations.transpose(0, 2, 1),
            frame[[f"wheel{wheel}_contact_point_{axis}_m" for axis in "xyz"]]
            .to_numpy(float) - frame[[f"world_com_{axis}_m" for axis in "xyz"]]
            .to_numpy(float))
        contact_arms_body[:, wheel, :] = contact_arm
        contact_normals_body[:, wheel, :] = contact_normal
        forward_dirs_body[:, wheel, :] = forward_direction
        sideways_dirs_body[:, wheel, :] = sideways_direction
        wheel_omega[:, wheel] = (frame[f"wheel{wheel}_rpm"].to_numpy(float) *
                                 2.0 * np.pi / 60.0)
        effective_inertia[:, wheel] = np.where(
            velocity_body[:, 2] < 6.0, effective_inertia_values[0],
            np.where(velocity_body[:, 2] < 12.0,
                     effective_inertia_values[1], effective_inertia_values[2]))
        domega = _derivative(wheel_omega[:, wheel],
                             times, reset_boundaries)
        wheel_angular_acceleration[:, wheel] = domega
        sign = np.sign(wheel_omega[:, wheel])
        sign[sign == 0.0] = 1.0
        motor = frame[f"wheel{wheel}_motor_torque_nm"].to_numpy(float)
        brake = frame[f"wheel{wheel}_brake_torque_nm"].to_numpy(float)
        forward_force = ((motor - sign * brake -
                           effective_inertia[:, wheel] * domega -
                           WHEEL_DAMPING_NMS * wheel_omega[:, wheel]) /
                          WHEEL_RADIUS_M)
        wheel_forward_force[:, wheel] = forward_force
        forward_vector = forward_direction[:, [0, 2]] * forward_force[:, None]
        forward_wrench[:, 0] += forward_vector[:, 0]
        forward_force_body_z += forward_vector[:, 1]
        forward_wrench[:, 1] += (
            contact_arm[:, 2] * forward_vector[:, 0] -
            contact_arm[:, 0] * forward_vector[:, 1])

        # Unity's WheelHit sideways force direction is kept explicit. The
        # curve value supplies a signed scalar; no real-tire coefficient is
        # introduced here.
        side_unit = -sideways_direction[:, [0, 2]]
        side_moment_unit = (contact_arm[:, 2] * side_unit[:, 0] -
                            contact_arm[:, 0] * side_unit[:, 1])
        axle = 0 if wheel < 2 else 1
        side_matrix[:, 0, axle] += side_unit[:, 0]
        side_matrix[:, 1, axle] += side_moment_unit
        slip = frame[f"wheel{wheel}_sideways_slip"].to_numpy(float)
        forward_slip = frame[f"wheel{wheel}_forward_slip"].to_numpy(float)
        if load_mode == "wheel_hit_force":
            load = frame[f"wheel{wheel}_contact_force_n"].to_numpy(float)
        elif load_mode == "runtime_sprung_load":
            load = (frame[f"wheel{wheel}_sprung_mass_kg"].to_numpy(float) *
                    GRAVITY_MPS2)
        else:
            load = np.full(
                count,
                float(metadata["vehicle"]["wheels"][wheel]["sprungMass"]) *
                GRAVITY_MPS2,
                dtype=float)
        sideways_slips[:, wheel] = slip
        forward_slips[:, wheel] = forward_slip
        contact_loads[:, wheel] = load
        curve_value = _curve(slip, (0.01, 1.0, 0.1, 0.5, 1.0))
        forward_curve_value = _curve(
            forward_slip, (0.15, 0.9, 0.25, 0.58, 0.8))
        curve_values[:, wheel] = curve_value
        forward_curve_values[:, wheel] = forward_curve_value
        proxy[:, axle] += load * curve_value

    right_hand_side = np.column_stack((
        target[:, 0] - forward_wrench[:, 0],
        target[:, 2] - forward_wrench[:, 1],
    ))
    determinant = (side_matrix[:, 0, 0] * side_matrix[:, 1, 1] -
                   side_matrix[:, 0, 1] * side_matrix[:, 1, 0])
    recovered = np.full((count, 2), np.nan, dtype=float)
    finite_matrix = np.isfinite(determinant) & (np.abs(determinant) > 1.0e-12)
    recovered[finite_matrix] = np.linalg.solve(
        side_matrix[finite_matrix], right_hand_side[finite_matrix])

    steering = frame.applied_steering_rad.to_numpy(float)
    throttle = frame.applied_throttle_norm.to_numpy(float)
    transition = np.flatnonzero(
        (np.abs(np.diff(throttle, prepend=throttle[0])) > 1.0e-8) |
        (np.abs(np.diff(steering, prepend=steering[0])) > 1.0e-8))
    transition_mask = np.zeros(count, dtype=bool)
    for index in transition:
        transition_mask[index:min(count, index + 500)] = True
    grounded = np.ones(count, dtype=bool)
    for wheel in range(4):
        grounded &= frame[f"wheel{wheel}_grounded"].to_numpy() == 1
    body_steps = _collision_steps(collision)
    body_contact = np.asarray([
        int(step) in body_steps for step in frame.fixed_step.to_numpy()
    ], dtype=bool)
    base_usable = (
        grounded & ~transition_mask & (throttle > 0.05) &
        (np.abs(np.gradient(steering, times)) < 1.0e-5) &
        (velocity_body[:, 2] > 4.0) & (velocity_body[:, 2] < 16.0) &
        # Keep small high-speed raceline steering excursions; the simulator
        # schedule deliberately avoids impossible large steering at speed.
        (np.abs(steering) > 0.005) &
        np.isfinite(determinant) & (np.abs(determinant) > 1.0e-5) &
        np.all(np.isfinite(recovered), axis=1) &
        np.all(np.isfinite(proxy), axis=1)
    )
    usable = base_usable & ~body_contact & ~reset_excluded
    selected = np.flatnonzero(usable)

    def screen_bins(values: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for lower, upper in zip(edges[:-1], edges[1:]):
            indices = selected[(values[selected] >= lower) &
                               (values[selected] < upper)]
            label = f"{lower:.6g}_{upper:.6g}"
            output[label] = {
                "range": [float(lower), float(upper)],
                "rows": int(indices.size),
                "front": _fit(proxy[:, [0]], recovered[:, 0], indices),
                "rear": _fit(proxy[:, [1]], recovered[:, 1], indices),
            }
        return output

    def quantile_screen(values: np.ndarray) -> dict[str, Any]:
        if selected.size == 0:
            return {}
        edges = np.quantile(values[selected], [0.0, 1.0 / 3.0,
                                                2.0 / 3.0, 1.0])
        edges = np.unique(edges)
        if edges.size < 2:
            return {"constant": {"range": [float(edges[0]), float(edges[0])],
                                  "rows": int(selected.size),
                                  "front": _fit(proxy[:, [0]], recovered[:, 0], selected),
                                  "rear": _fit(proxy[:, [1]], recovered[:, 1], selected)}}
        return screen_bins(values, np.nextafter(edges, np.inf))

    front_load = contact_loads[:, 0] + contact_loads[:, 1]
    rear_load = contact_loads[:, 2] + contact_loads[:, 3]
    maximum_sideways_slip = np.max(np.abs(sideways_slips), axis=1)
    maximum_forward_slip = np.max(np.abs(forward_slips), axis=1)
    normalized_speed = (velocity_body[:, 2] - 8.0) / 8.0
    combined_slip_design = {}
    for axle, name in enumerate(("front", "rear")):
        combined_slip_design[name] = np.column_stack((
            proxy[:, axle],
            proxy[:, axle] * maximum_forward_slip,
            proxy[:, axle] * maximum_sideways_slip,
            proxy[:, axle] * normalized_speed,
        ))
    axle_load = np.column_stack((
        np.sum(contact_loads[:, (0, 1)], axis=1),
        np.sum(contact_loads[:, (2, 3)], axis=1),
    ))
    forward_curve_demand = np.zeros((count, 2), dtype=float)
    sideways_curve_demand = np.zeros((count, 2), dtype=float)
    for axle, wheel_indices in enumerate(((0, 1), (2, 3))):
        safe_load = np.maximum(axle_load[:, axle], 1.0e-6)
        forward_curve_demand[:, axle] = np.sum(
            contact_loads[:, wheel_indices] *
            np.abs(forward_curve_values[:, wheel_indices]), axis=1) / safe_load
        sideways_curve_demand[:, axle] = np.sum(
            contact_loads[:, wheel_indices] *
            np.abs(curve_values[:, wheel_indices]), axis=1) / safe_load
    combined_curve_demand = np.sqrt(
        forward_curve_demand ** 2 + sideways_curve_demand ** 2)
    curve_demand_design = {}
    for axle, name in enumerate(("front", "rear")):
        curve_demand_design[name] = np.column_stack((
            proxy[:, axle],
            proxy[:, axle] * forward_curve_demand[:, axle],
            proxy[:, axle] * forward_curve_demand[:, axle] ** 2,
            proxy[:, axle] * sideways_curve_demand[:, axle],
            proxy[:, axle] * combined_curve_demand[:, axle],
        ))
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_unity_axle_force_response_inversion",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "diagnostic_trace_only",
        "source": {
            "trace": ([str(path) for path in traces]
                      if len(traces) > 1 else str(traces[0])),
            "static_snapshot": str(static),
            "rows": int(count),
            "duration_s": float(times[-1] - times[0]),
            "body_wrench": "measured Unity body acceleration plus explicit Rigidbody drag compensation",
            "body_angular_wrench": (
                "full runtime inertia Euler equation with explicit angular "
                f"drag model '{angular_drag_model}'"),
            "normal_force_wrench": (
                f"explicit model '{normal_force_model}'; WheelHit.force "
                "times recorded contact normal is subtracted from the "
                "contact wrench"),
            "inertia_source": (
                "per-fixed-step Unity principal inertia and tensor rotation; "
                "serialized snapshot fallback for old traces"),
            "forward_force": "wheel torque balance using explicit prior effective Unity solver state",
            "effective_wheel_inertia_kgm2": list(effective_inertia_values),
            "wheel_damping_nms": WHEEL_DAMPING_NMS,
            "lateral_load_source": load_mode,
            "collision_split": "diagnostic only; chassis callbacks are excluded when supplied",
            "reset_segmentation": {
                "method": (
                    "COM position jump > 0.2 m, current speed < 0.5 m/s, "
                    "and speed drop > 1 m/s; derivative is segmented at "
                    "each boundary"),
                "detected_boundaries": int(np.count_nonzero(reset_boundaries)),
                "boundary_times_s": [
                    float(times[index]) for index in
                    np.flatnonzero(reset_boundaries)],
                "excluded_rows": int(np.count_nonzero(reset_excluded)),
                "exclusion_radius_rows": 10,
            },
        },
        "selection": {
            "usable_rows": int(selected.size),
            "body_contact_rows_excluded": int(
                np.count_nonzero(body_contact & base_usable)),
            "speed_mps": [float(np.min(velocity_body[selected, 2])),
                          float(np.max(velocity_body[selected, 2]))]
            if selected.size else [None, None],
            "steering_abs_rad": [float(np.min(np.abs(steering[selected]))),
                                 float(np.max(np.abs(steering[selected])))]
            if selected.size else [None, None],
            "minimum_absolute_steering_rad": 0.005,
            "determinant_abs_p05_p50_p95": [
                float(value) for value in np.percentile(
                    np.abs(determinant[selected]), [5.0, 50.0, 95.0])
            ] if selected.size else [None, None, None],
        },
        "recovered_lateral_force_n": {
            "front": _stats(recovered[selected, 0]) if selected.size else {},
            "rear": _stats(recovered[selected, 1]) if selected.size else {},
        },
        "curve_contact_load_proxy_n": {
            "front": _stats(proxy[selected, 0]) if selected.size else {},
            "rear": _stats(proxy[selected, 1]) if selected.size else {},
        },
        "chronological_holdout": {
            "front": _fit(proxy[:, [0]], recovered[:, 0], selected),
            "rear": _fit(proxy[:, [1]], recovered[:, 1], selected),
        },
        "load_power_screen": {
            "basis": (
                "gain * 10 N * signed Unity sideways curve * "
                "(wheel_load / 10 N)^p"),
            "front": _fit_load_power(
                curve_values, contact_loads, recovered[:, 0], selected,
                (0, 1)),
            "rear": _fit_load_power(
                curve_values, contact_loads, recovered[:, 1], selected,
                (2, 3)),
            "promotion": False,
        },
        "speed_bins": {},
        "dependence_screens": {
            "absolute_applied_steering_rad": screen_bins(
                np.abs(steering), np.asarray([0.02, 0.04, 0.08, 0.20])),
            "front_axle_contact_load_n_tertiles": quantile_screen(front_load),
            "rear_axle_contact_load_n_tertiles": quantile_screen(rear_load),
            "maximum_absolute_sideways_slip_tertiles": quantile_screen(
                maximum_sideways_slip),
            "maximum_absolute_forward_slip_tertiles": quantile_screen(
                maximum_forward_slip),
        },
        "explicit_combined_slip_screen": {
            "basis": [
                "curve_contact_load_proxy",
                "curve_contact_load_proxy * maximum_absolute_forward_slip",
                "curve_contact_load_proxy * maximum_absolute_sideways_slip",
                "curve_contact_load_proxy * ((speed_mps - 8) / 8)",
            ],
            "front": _fit_design(combined_slip_design["front"],
                                  recovered[:, 0], selected),
            "rear": _fit_design(combined_slip_design["rear"],
                                 recovered[:, 1], selected),
            "promotion": False,
        },
        "serialized_curve_demand_screen": {
            "basis": [
                "curve_contact_load_proxy",
                "proxy * weighted_abs_forward_curve_demand",
                "proxy * weighted_abs_forward_curve_demand^2",
                "proxy * weighted_abs_sideways_curve_demand",
                "proxy * weighted_combined_curve_demand",
            ],
            "front": _fit_design(curve_demand_design["front"],
                                  recovered[:, 0], selected),
            "rear": _fit_design(curve_demand_design["rear"],
                                 recovered[:, 1], selected),
            "promotion": False,
        },
        "acceptance": {
            "runtime_use": False,
            "friction_coefficient_identified": False,
            "cornering_stiffness_identified": False,
            "tire_peak_changed": False,
            "promotion": False,
        },
    }
    for lower, upper in ((4.0, 6.0), (6.0, 8.0), (8.0, 10.0),
                         (10.0, 12.0), (12.0, 16.0)):
        bin_indices = selected[(velocity_body[selected, 2] >= lower) &
                               (velocity_body[selected, 2] < upper)]
        result["speed_bins"][f"{lower:g}_{upper:g}mps"] = {
            "rows": int(bin_indices.size),
            "front": _fit(proxy[:, [0]], recovered[:, 0], bin_indices),
            "rear": _fit(proxy[:, [1]], recovered[:, 1], bin_indices),
        }
    if rows_output is not None:
        # Keep the causal inversion auditable without placing the large raw
        # Unity trace in a second artifact.  This table contains only the
        # selected, grounded, steady-command rows used by the screens above.
        def optional_column(name: str) -> np.ndarray:
            if name in frame.columns:
                return frame[name].to_numpy(float)[selected]
            return np.full(selected.size, np.nan, dtype=float)

        row_data: dict[str, np.ndarray] = {
            "fixed_step": frame.fixed_step.to_numpy()[selected],
            "fixed_time_s": times[selected],
            "speed_mps": velocity_body[selected, 2],
            "body_vx_mps": velocity_body[selected, 0],
            "body_vy_mps": velocity_body[selected, 1],
            "applied_steering_rad": steering[selected],
            "applied_throttle_norm": throttle[selected],
            "recovered_front_force_n": recovered[selected, 0],
            "recovered_rear_force_n": recovered[selected, 1],
            "front_proxy_n": proxy[selected, 0],
            "rear_proxy_n": proxy[selected, 1],
            "front_load_n": front_load[selected],
            "rear_load_n": rear_load[selected],
            "max_abs_forward_slip": maximum_forward_slip[selected],
            "max_abs_sideways_slip": maximum_sideways_slip[selected],
            "front_forward_curve_demand": forward_curve_demand[selected, 0],
            "rear_forward_curve_demand": forward_curve_demand[selected, 1],
            "front_sideways_curve_demand": sideways_curve_demand[selected, 0],
            "rear_sideways_curve_demand": sideways_curve_demand[selected, 1],
            "front_combined_curve_demand": combined_curve_demand[selected, 0],
            "rear_combined_curve_demand": combined_curve_demand[selected, 1],
            "runtime_body_yaw_inertia_kgm2": optional_column(
                "runtime_body_yaw_inertia_kgm2"),
            # target[:, 0] is Unity body-x force (the lateral axis for this
            # source vehicle), target[:, 1] is Unity body-z longitudinal
            # force, and target[:, 2] is the body-y yaw moment. Keep the
            # longitudinal channel explicit instead of folding it into the
            # lateral inversion or an effective drive gain.
            "target_body_x_force_n": target[selected, 0],
            "target_body_z_force_n": target[selected, 1],
            "target_body_y_moment_nm": target[selected, 2],
            "forward_torque_force_body_x_n": forward_wrench[selected, 0],
            "forward_torque_force_body_z_n": forward_force_body_z[selected],
            "forward_torque_moment_nm": forward_wrench[selected, 1],
        }
        for wheel in range(4):
            row_data[f"wheel{wheel}_load_n"] = contact_loads[selected, wheel]
            row_data[f"wheel{wheel}_sideways_slip"] = sideways_slips[selected, wheel]
            row_data[f"wheel{wheel}_forward_slip"] = forward_slips[selected, wheel]
            row_data[f"wheel{wheel}_sideways_curve"] = curve_values[selected, wheel]
            row_data[f"wheel{wheel}_forward_curve"] = forward_curve_values[selected, wheel]
            row_data[f"wheel{wheel}_omega_radps"] = wheel_omega[selected, wheel]
            row_data[f"wheel{wheel}_domega_radps2"] = wheel_angular_acceleration[
                selected, wheel]
            row_data[f"wheel{wheel}_effective_inertia_kgm2"] = effective_inertia[
                selected, wheel]
            row_data[f"wheel{wheel}_forward_force_from_torque_n"] = wheel_forward_force[
                selected, wheel]
            row_data[f"wheel{wheel}_motor_torque_nm"] = optional_column(
                f"wheel{wheel}_motor_torque_nm")
            row_data[f"wheel{wheel}_brake_torque_nm"] = optional_column(
                f"wheel{wheel}_brake_torque_nm")
            row_data[f"wheel{wheel}_steer_angle_deg"] = optional_column(
                f"wheel{wheel}_steer_angle_deg")
            row_data[f"wheel{wheel}_sprung_mass_kg"] = optional_column(
                f"wheel{wheel}_sprung_mass_kg")
            row_data[f"wheel{wheel}_world_pose_y_m"] = optional_column(
                f"wheel{wheel}_world_pose_y_m")
            row_data[f"wheel{wheel}_pose_body_from_com_x_m"] = (
                wheel_pose_body_from_com[selected, wheel, 0])
            row_data[f"wheel{wheel}_pose_body_from_com_y_m"] = (
                wheel_pose_body_from_com[selected, wheel, 1])
            row_data[f"wheel{wheel}_pose_body_from_com_z_m"] = (
                wheel_pose_body_from_com[selected, wheel, 2])
            row_data[f"wheel{wheel}_pose_body_y_rate_mps"] = (
                wheel_pose_body_y_rate[selected, wheel])
            row_data[f"wheel{wheel}_contact_force_n"] = optional_column(
                f"wheel{wheel}_contact_force_n")
            for axis_index, axis in enumerate("xyz"):
                row_data[f"wheel{wheel}_contact_arm_{axis}_m"] = contact_arms_body[
                    selected, wheel, axis_index]
                row_data[f"wheel{wheel}_contact_normal_body_{axis}"] = (
                    contact_normals_body[selected, wheel, axis_index])
                row_data[f"wheel{wheel}_forward_dir_body_{axis}"] = (
                    forward_dirs_body[selected, wheel, axis_index])
                row_data[f"wheel{wheel}_sideways_dir_body_{axis}"] = (
                    sideways_dirs_body[selected, wheel, axis_index])
        rows = pd.DataFrame(row_data)
        rows_output.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(rows_output, index=False, float_format="%.10g")
        result["row_diagnostics"] = {
            "path": str(rows_output),
            "rows": int(len(rows)),
            "purpose": "cross-schedule mechanism screen only",
            "promotion": False,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True,
                        help="trace CSV; repeat for split trace parts")
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--collision", type=Path)
    parser.add_argument(
        "--angular-drag-model", choices=("inertia_scaled", "direct", "none"),
        default="inertia_scaled",
        help=("angular drag torque screen: inertia_scaled uses "
              "I*omega, direct uses omega, none omits it"),
    )
    parser.add_argument(
        "--normal-force-model",
        choices=("measured_contact_normal", "none"),
        default="measured_contact_normal",
        help=("normal contact reaction screen; measured_contact_normal "
              "uses WheelHit.force times contact normal"),
    )
    parser.add_argument(
        "--effective-inertia", type=float, nargs=3,
        default=DEFAULT_I_EFFECTIVE, metavar=("LOW", "MID", "HIGH"),
        help=("effective wheel inertia for <6, 6-12, and 12-16 m/s; "
              "diagnostic override only"),
    )
    parser.add_argument(
        "--load-mode",
        choices=("wheel_hit_force", "runtime_sprung_load", "static_sprung_load"),
        default="wheel_hit_force",
        help="explicit load source used only for the lateral proxy screen",
    )
    parser.add_argument(
        "--rows-output", type=Path,
        help="optional compact selected-row CSV for cross-schedule screening",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.static, args.output, args.collision,
                     args.angular_drag_model, args.normal_force_model,
                     tuple(args.effective_inertia), args.load_mode,
                     args.rows_output)
    print(json.dumps({
        "output": str(args.output),
        "selection": result["selection"],
        "chronological_holdout": result["chronological_holdout"],
        "speed_bins": result["speed_bins"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
