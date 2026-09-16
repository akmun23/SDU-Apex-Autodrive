#!/usr/bin/env python3
"""Identify the measured Unity wheel-contact wrench without hiding parameters.

This is a diagnostic-only force-balance audit.  It reconstructs the horizontal
body wrench from the exact Unity rigid-body trace and compares it with the
serialized WheelFrictionCurve evaluated at the recorded per-wheel slips.  The
reported gains are explicit observability coefficients; they are not copied
into the vehicle plant or production MPC by this script.

The audit deliberately keeps separate:

* the measured ``WheelHit.force`` contact-load proxy;
* the exact settled ``sprungMass * g`` reference loads; and
* per-wheel force-direction/slip regressors.

This makes a load mismatch, slip-coordinate mismatch, and force-direction
mismatch visible instead of absorbing them into a tire peak or cornering
stiffness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from structured_vehicle_plant import unity_wheel_friction_value


WHEELS = range(4)
GRAVITY_MPS2 = 9.81
FORWARD_CURVE = (0.15, 0.9, 0.25, 0.58, 0.8)
SIDEWAYS_CURVE = (0.01, 1.0, 0.1, 0.5, 1.0)


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("zero-length Unity quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y))), dtype=float)


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mae": None, "rmse": None,
                "p50": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mae": float(np.mean(np.abs(finite))),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "p50": float(np.percentile(np.abs(finite), 50.0)),
        "p95": float(np.percentile(np.abs(finite), 95.0)),
        "max": float(np.max(np.abs(finite))),
    }


def _read_trace(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"trace contains no rows: {path}")
    return rows, fields


def _curve(slip: float, values: tuple[float, ...]) -> float:
    return unity_wheel_friction_value(slip, *values)


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    differences = np.diff(times)
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if positive.size == 0:
        raise ValueError("trace has no positive time intervals")
    dt = float(np.median(positive))
    window = min(101, len(values) if len(values) % 2 else len(values) - 1)
    if window >= 7:
        if window % 2 == 0:
            window -= 1
        return savgol_filter(values, window, 2, deriv=1, delta=dt,
                             mode="interp")
    return np.gradient(values, times)


def _body_vector(rows: list[dict[str, str]], prefix: str,
                 suffix: str) -> np.ndarray:
    values = []
    for row in rows:
        values.append([float(row[f"{prefix}_{axis}_{suffix}"])
                       for axis in "xyz"])
    return np.asarray(values, dtype=float)


def _inertia_matrix(rows: list[dict[str, str]],
                    metadata: dict[str, Any]) -> np.ndarray:
    """Reconstruct Unity's full body-frame inertia tensor per sample."""
    runtime_fields = {
        "runtime_inertia_tensor_x_kgm2",
        "runtime_inertia_tensor_y_kgm2",
        "runtime_inertia_tensor_z_kgm2",
        "runtime_inertia_rotation_x",
        "runtime_inertia_rotation_y",
        "runtime_inertia_rotation_z",
        "runtime_inertia_rotation_w",
    }
    if rows and runtime_fields.issubset(rows[0]):
        moments = np.asarray([
            [float(row[f"runtime_inertia_tensor_{axis}_kgm2"])
             for axis in "xyz"] for row in rows
        ], dtype=float)
        rotations = np.asarray([
            _rotation_matrix(np.asarray([
                float(row[f"runtime_inertia_rotation_{axis}"])
                for axis in "xyzw"
            ])) for row in rows
        ])
    else:
        rigid_body = metadata["vehicle"]["rigidBody"]
        moments = np.asarray([[
            float(rigid_body["inertiaTensor"][axis]) for axis in "xyz"
        ]] * len(rows), dtype=float)
        rotation = _rotation_matrix(np.asarray([
            float(rigid_body["inertiaTensorRotation"][axis])
            for axis in "xyzw"
        ]))
        rotations = np.repeat(rotation[None, :, :], len(rows), axis=0)
    diagonal = np.zeros((len(rows), 3, 3), dtype=float)
    diagonal[:, range(3), range(3)] = moments
    # This ordering reproduces Unity's runtime body-y inertia from the trace.
    return np.einsum("nij,njk,nlk->nil", rotations, diagonal, rotations)


def _body_wrench(rows: list[dict[str, str]], metadata: dict[str, Any]) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    times = np.asarray([float(row["fixed_time_s"]) for row in rows], dtype=float)
    world_velocity = _body_vector(rows, "world_velocity", "mps")
    world_angular_velocity = _body_vector(
        rows, "world_angular_velocity", "radps")
    quaternions = np.asarray([
        [float(row[f"root_rotation_{axis}"]) for axis in "xyzw"]
        for row in rows
    ], dtype=float)
    rotations = np.asarray([_rotation_matrix(value) for value in quaternions])
    acceleration_world = np.column_stack([
        _derivative(world_velocity[:, index], times)
        for index in range(3)
    ])
    acceleration_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), acceleration_world)
    angular_acceleration_world = np.column_stack([
        _derivative(world_angular_velocity[:, index], times)
        for index in range(3)
    ])
    angular_acceleration_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1),
        angular_acceleration_world)
    velocity_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), world_velocity)
    angular_velocity_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1),
        world_angular_velocity)

    rigid_body = metadata["vehicle"]["rigidBody"]
    mass = float(rigid_body["mass"])
    inertia = _inertia_matrix(rows, metadata)
    linear_drag = float(rigid_body["drag"])
    angular_drag = float(rigid_body["angularDrag"])
    # Unity Rigidbody drag is represented here as an explicit diagnostic
    # compensation.  The uncompensated and compensated residuals are both
    # reported, so this approximation cannot disappear into a wheel gain.
    body_force = mass * acceleration_body
    gravity_body = np.einsum(
        "nij,j->ni", rotations.transpose(0, 2, 1),
        np.asarray((0.0, -GRAVITY_MPS2, 0.0), dtype=float))
    # Reconstruct the horizontal contact force, not the total inertial force:
    # m*a = F_contact + F_drag + m*g.
    body_force -= mass * gravity_body
    normal_world = np.zeros_like(body_force)
    for wheel in WHEELS:
        contact_force = np.asarray([
            _float(row, f"wheel{wheel}_contact_force_n") or 0.0
            for row in rows
        ])
        contact_normal = np.asarray([[
            _float(row, f"wheel{wheel}_contact_normal_{axis}") or 0.0
            for axis in "xyz"
        ] for row in rows])
        normal_world += contact_force[:, None] * contact_normal
    normal_body = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), normal_world)
    # WheelFrictionCurve comparison is for tangential force. Remove the
    # measured normal reaction as well as gravity; on a tilted body their
    # horizontal projections largely cancel.
    body_force -= normal_body
    body_force[:, 0] += mass * linear_drag * velocity_body[:, 0]
    body_force[:, 2] += mass * linear_drag * velocity_body[:, 2]
    angular_momentum = np.einsum("nij,nj->ni", inertia,
                                  angular_velocity_body)
    angular_moment = (np.einsum("nij,nj->ni", inertia,
                                 angular_acceleration_body) +
                      np.cross(angular_velocity_body, angular_momentum) +
                      angular_drag * angular_momentum)
    return (times, velocity_body, body_force, angular_moment[:, 1],
            "per_fixed_step_full_inertia_euler", inertia[:, 1, 1])


def _wheel_regressors(rows: list[dict[str, str]],
                      metadata: dict[str, Any],
                      load_mode: str, sideways_force_sign: float,
                      forward_force_sign: float) -> tuple[
                          np.ndarray, np.ndarray, np.ndarray]:
    wheels = metadata["vehicle"]["wheels"]
    static_loads = np.asarray([
        float(wheel["sprungMass"]) * GRAVITY_MPS2 for wheel in wheels
    ])
    count = len(rows)
    # Two explicit columns per wheel: sideways and forward force. Each column
    # is a three-component wrench [Fx, Fz, My] at every sample.
    features = np.zeros((count, 8, 3), dtype=float)
    valid = np.ones(count, dtype=bool)
    for index, row in enumerate(rows):
        quaternion = np.asarray([
            float(row[f"root_rotation_{axis}"]) for axis in "xyzw"
        ])
        rotation = _rotation_matrix(quaternion)
        world_com = np.asarray([
            float(row[f"world_com_{axis}_m"]) for axis in "xyz"
        ])
        for wheel in WHEELS:
            if row.get(f"wheel{wheel}_grounded") != "1":
                valid[index] = False
                continue
            try:
                contact = np.asarray([
                    float(row[f"wheel{wheel}_contact_point_{axis}_m"])
                    for axis in "xyz"
                ])
                forward_dir = rotation.T @ np.asarray([
                    float(row[f"wheel{wheel}_forward_dir_{axis}"])
                    for axis in "xyz"
                ])
                sideways_dir = rotation.T @ np.asarray([
                    float(row[f"wheel{wheel}_sideways_dir_{axis}"])
                    for axis in "xyz"
                ])
                forward_slip = float(row[f"wheel{wheel}_forward_slip"])
                sideways_slip = float(row[f"wheel{wheel}_sideways_slip"])
                measured_load = float(row[f"wheel{wheel}_contact_force_n"])
            except (KeyError, TypeError, ValueError):
                valid[index] = False
                continue
            if not all(np.isfinite(value) for value in (
                    *contact, *forward_dir, *sideways_dir,
                    forward_slip, sideways_slip, measured_load)):
                valid[index] = False
                continue
            if load_mode == "wheel_hit_force":
                load = measured_load
            elif load_mode == "static_sprung_load":
                load = static_loads[wheel]
            elif load_mode == "runtime_sprung_load":
                try:
                    load = (float(row[f"wheel{wheel}_sprung_mass_kg"]) *
                            GRAVITY_MPS2)
                except (KeyError, TypeError, ValueError):
                    valid[index] = False
                    continue
            else:
                raise ValueError(f"unknown load mode: {load_mode}")
            # Contact point is used rather than the nominal collider center;
            # this preserves the measured suspension/contact geometry.
            arm = rotation.T @ (contact - world_com)

            def wrench(direction: np.ndarray, force: float,
                       direction_sign: float) -> np.ndarray:
                horizontal = direction * direction_sign * force
                # Unity y is vertical; only horizontal force and yaw moment
                # enter this planar audit.
                return np.asarray((horizontal[0], horizontal[2],
                                   arm[2] * horizontal[0] -
                                   arm[0] * horizontal[2]))

            side_force = load * _curve(sideways_slip, SIDEWAYS_CURVE)
            forward_force = load * _curve(forward_slip, FORWARD_CURVE)
            features[index, wheel * 2] = wrench(
                sideways_dir, side_force, sideways_force_sign)
            features[index, wheel * 2 + 1] = wrench(
                forward_dir, forward_force, forward_force_sign)
    return features, valid, static_loads


def _fit(features: np.ndarray, target: np.ndarray,
         indices: np.ndarray) -> dict[str, Any]:
    if indices.size < 16:
        return {"identified": False, "samples": int(indices.size)}
    # Stack the three wrench components so each scalar coefficient has to
    # explain force and yaw moment simultaneously.
    design = features[indices].transpose(0, 2, 1).reshape(-1, features.shape[1])
    response = target[indices].reshape(-1)
    coefficients, _, rank, singular = np.linalg.lstsq(
        design, response, rcond=None)
    residual = response - design @ coefficients
    prediction = np.einsum("nwc,w->nc", features[indices], coefficients)
    error = target[indices] - prediction
    return {
        "identified": bool(rank == features.shape[1]),
        "samples": int(indices.size),
        "rank": int(rank),
        "coefficients": {
            ("wheel%d_sideways_gain" % (index // 2)
             if index % 2 == 0 else
             "wheel%d_forward_gain" % (index // 2)): float(value)
            for index, value in enumerate(coefficients)
        },
        "wrench_rmse": {
            "force_x_n": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "force_z_n": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "yaw_moment_nm": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        },
        "wrench_mae": {
            "force_x_n": float(np.mean(np.abs(error[:, 0]))),
            "force_z_n": float(np.mean(np.abs(error[:, 1]))),
            "yaw_moment_nm": float(np.mean(np.abs(error[:, 2]))),
        },
        "stacked_rmse": float(np.sqrt(np.mean(residual * residual))),
        "design_singular_values": [float(value) for value in singular],
    }


def _score(features: np.ndarray, target: np.ndarray,
           coefficients: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    if not coefficients.get("identified") or indices.size == 0:
        return {"samples": int(indices.size), "identified_from_training": False}
    values = np.asarray(list(coefficients["coefficients"].values()), dtype=float)
    prediction = np.einsum("nwc,w->nc", features[indices], values)
    error = target[indices] - prediction
    return {
        "samples": int(indices.size),
        "identified_from_training": True,
        "wrench_rmse": {
            "force_x_n": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "force_z_n": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "yaw_moment_nm": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        },
        "wrench_mae": {
            "force_x_n": float(np.mean(np.abs(error[:, 0]))),
            "force_z_n": float(np.mean(np.abs(error[:, 1]))),
            "yaw_moment_nm": float(np.mean(np.abs(error[:, 2]))),
        },
    }


def _direct_error(features: np.ndarray, target: np.ndarray,
                  indices: np.ndarray) -> dict[str, Any]:
    prediction = np.sum(features[indices], axis=1)
    error = target[indices] - prediction
    return {
        "samples": int(indices.size),
        "wrench_rmse": {
            "force_x_n": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "force_z_n": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "yaw_moment_nm": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        },
        "wrench_mae": {
            "force_x_n": float(np.mean(np.abs(error[:, 0]))),
            "force_z_n": float(np.mean(np.abs(error[:, 1]))),
            "yaw_moment_nm": float(np.mean(np.abs(error[:, 2]))),
        },
    }


def _body_collision_steps(path: Path | None) -> set[int]:
    if path is None:
        return set()
    steps: set[int] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row.get("this_collider") == "Chassis-1-solid1" and
                    row.get("event") in ("enter", "stay")):
                try:
                    steps.add(int(row["fixed_step"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return steps


def analyze(trace: Path, static: Path, output: Path,
            derivative_window: int = 101,
            collision: Path | None = None) -> dict[str, Any]:
    del derivative_window  # Reserved for a future explicit sensitivity sweep.
    metadata = json.loads(static.read_text(encoding="utf-8"))
    rows, fields = _read_trace(trace)
    required = {
        "world_com_x_m", "world_com_y_m", "world_com_z_m",
        "world_velocity_x_mps", "world_velocity_y_mps",
        "world_velocity_z_mps", "world_angular_velocity_x_radps",
        "world_angular_velocity_y_radps", "world_angular_velocity_z_radps",
    }
    if not required.issubset(fields):
        raise ValueError("trace lacks body wrench fields")
    (times, velocity_body, target_force, target_moment,
     yaw_inertia_source, yaw_inertia) = _body_wrench(
        rows, metadata)
    target = np.column_stack((target_force[:, 0], target_force[:, 2],
                              target_moment))
    speed = velocity_body[:, 2]
    steering = np.asarray([float(row["applied_steering_rad"]) for row in rows])
    throttle = np.asarray([
        float(row["applied_throttle_norm"]) for row in rows
    ])
    grounded = np.asarray([
        all(row.get(f"wheel{wheel}_grounded") == "1" for wheel in WHEELS)
        for row in rows
    ])
    # Body acceleration is useful for force balance, but actuator transitions
    # create a separate transient problem. Exclude a measured 0.5 s after
    # every applied throttle/steering change when assessing the steady
    # WheelFrictionCurve semantics. The transient remains a separate future
    # identification target and is not hidden in this fit.
    transition = np.flatnonzero(
        (np.abs(np.diff(throttle, prepend=throttle[0])) > 1.0e-8) |
        (np.abs(np.diff(steering, prepend=steering[0])) > 1.0e-8))
    transition_mask = np.zeros(len(rows), dtype=bool)
    transition_steps = max(1, int(round(0.5 / 0.001)))
    for index in transition:
        transition_mask[index:min(len(rows), index + transition_steps)] = True
    stable_command = (~transition_mask & (throttle > 0.05) &
                      (np.abs(np.gradient(steering, times)) < 1.0e-5) &
                      (np.abs(np.gradient(throttle, times)) < 1.0e-5))
    usable = (grounded & stable_command & np.isfinite(speed) &
              np.isfinite(steering) &
              np.all(np.isfinite(target), axis=1) & (speed > 1.0) &
              (speed < 16.0))
    body_steps = _body_collision_steps(collision)
    body_contact = np.asarray([
        int(row["fixed_step"]) in body_steps for row in rows
    ], dtype=bool)
    selection_masks: dict[str, np.ndarray] = {"all": usable}
    if collision is not None:
        selection_masks["no_chassis_contact"] = usable & ~body_contact

    models_by_selection: dict[str, dict[str, Any]] = {}
    for selection_name, selection_mask in selection_masks.items():
        valid_results: dict[str, Any] = {}
        for load_mode in ("wheel_hit_force", "static_sprung_load",
                          "runtime_sprung_load"):
            for sideways_force_sign in (-1.0, 1.0):
                for forward_force_sign in (-1.0, 1.0):
                    features, feature_valid, static_loads = _wheel_regressors(
                        rows, metadata, load_mode, sideways_force_sign,
                        forward_force_sign)
                    selected = np.flatnonzero(selection_mask & feature_valid)
                    if selected.size < 32:
                        continue
                    # The first half is training and the second half is a
                    # chronological holdout. This is intentionally not random.
                    split = selected.size // 2
                    train = selected[:split]
                    holdout = selected[split:]
                    key = (f"{load_mode}_side_sign_{int(sideways_force_sign):+d}"
                           f"_forward_sign_{int(forward_force_sign):+d}")
                    fit = _fit(features, target, train)
                    valid_results[key] = {
                        "load_mode": load_mode,
                        "sideways_force_sign": sideways_force_sign,
                        "forward_force_sign": forward_force_sign,
                        "static_loads_n": [float(value)
                                            for value in static_loads],
                        "selection": {
                            "usable_rows": int(selected.size),
                            "train_rows": int(train.size),
                            "holdout_rows": int(holdout.size),
                            "speed_mps": [float(np.min(speed[selected])),
                                          float(np.max(speed[selected]))],
                            "steering_abs_rad": [
                                float(np.min(np.abs(steering[selected]))),
                                float(np.max(np.abs(steering[selected])))],
                        },
                        "direct_unit_gain": _direct_error(
                            features, target, holdout),
                        "fit_train": fit,
                        "fit_holdout": _score(
                            features, target, fit, holdout),
                    }
        models_by_selection[selection_name] = valid_results

    speed_bins = (1.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
    bin_results: dict[str, Any] = {}
    features, feature_valid, static_loads = _wheel_regressors(
        rows, metadata, "wheel_hit_force", -1.0, 1.0)
    for lower, upper in zip(speed_bins[:-1], speed_bins[1:]):
        selected = np.flatnonzero(usable & feature_valid &
                                  (speed >= lower) & (speed < upper))
        if selected.size < 32:
            continue
        split = selected.size // 2
        fit = _fit(features, target, selected[:split])
        bin_results[f"{lower:g}_{upper:g}mps"] = {
            "selection": {"rows": int(selected.size),
                          "train_rows": int(split),
                          "holdout_rows": int(selected.size - split)},
            "fit_train": fit,
            "fit_holdout": _score(features, target, fit, selected[split:]),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_unity_lateral_wrench_identification",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "diagnostic_trace_only",
        "source": {
            "trace": str(trace),
            "static_snapshot": str(static),
            "scene": metadata.get("sceneName"),
            "unity_version": metadata.get("unityVersion"),
            "trace_build_tag": metadata.get("simulatorBuildTag"),
            "rows": len(rows),
            "duration_s": float(times[-1] - times[0]),
            "derivative": "Savitzky-Golay order2 window101 over world velocity",
            "drag_compensation": "explicit measured Rigidbody drag; audit only",
            "yaw_inertia": {
                "source": yaw_inertia_source,
                "kgm2": {
                    "min": float(np.min(yaw_inertia)),
                    "max": float(np.max(yaw_inertia)),
                    "mean": float(np.mean(yaw_inertia)),
                },
            },
        },
        "target_wrench": {
            "components": ["body_force_x_n", "body_force_z_n",
                           "body_yaw_moment_y_nm"],
            "reference": (
                "rigid-body acceleration minus gravity and measured normal "
                "contact reaction plus explicit drag compensation"),
            "wheel_contact_arm": "measured contact point relative to runtime COM",
        },
        "selection": {
            "rows_all": len(rows),
            "rows_usable": int(np.count_nonzero(usable)),
            "rows_with_chassis_contact": int(
                np.count_nonzero(usable & body_contact)),
            "rows_usable_without_chassis_contact": int(
                np.count_nonzero(usable & ~body_contact)),
            "rows_excluded_as_actuator_transient": int(
                np.count_nonzero(grounded & ~stable_command)),
            "speed_mps": [float(np.min(speed[usable])) if np.any(usable) else None,
                          float(np.max(speed[usable])) if np.any(usable) else None],
        },
        "models": models_by_selection["all"],
        "models_by_selection": models_by_selection,
        "speed_bins": bin_results,
        "interpretation": {
            "coefficient_role": "explicit diagnostic observability gain, not a tire parameter",
            "collision_split_role": (
                "diagnostic regime comparison only; ContactPoint impulse and "
                "separation must be checked before treating callbacks as force"),
            "promotion": False,
            "next_decision": "compare holdout gains and residuals against fresh raceline-relevant runs before adding any causal state",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision", type=Path,
                        help="optional collision_contacts.csv for body-contact split")
    args = parser.parse_args()
    result = analyze(args.trace, args.static, args.output,
                     collision=args.collision)
    print(json.dumps({
        "output": str(args.output),
        "selection": result["selection"],
        "models": result["models"],
        "speed_bins": result["speed_bins"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
