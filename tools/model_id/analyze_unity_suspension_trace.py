#!/usr/bin/env python3
"""Extract explicit Unity suspension and wheel-contact quantities.

This is an offline diagnostic for the exact Unity competition-scene trace. It
uses only values exported by ``ModelIdentificationDiagnostics`` and does not
change the simulator, the vehicle parameters, or the production MPC.

The output deliberately reports a transparent compression/load relationship
instead of fitting a tire coefficient to absorb suspension or contact-model
errors.  The linear load screen is descriptive only; it is not a promoted
runtime model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


GRAVITY_MPS2 = 9.81
WHEELS = range(4)


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "p05": None, "p50": None,
                "p95": None, "min": None, "max": None}
    p05, p50, p95 = np.percentile(finite, [5.0, 50.0, 95.0])
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _quaternion_matrix(x: float, y: float, z: float,
                       w: float) -> np.ndarray:
    """Return the Unity local-to-world rotation matrix for x/y/z/w."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("zero-length Unity quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y))), dtype=float)


def _body_frame_wheel_position(row: dict[str, str], wheel: int) -> np.ndarray | None:
    root = np.asarray([
        _float(row, "root_position_x_m"),
        _float(row, "root_position_y_m"),
        _float(row, "root_position_z_m"),
    ], dtype=object)
    pose = np.asarray([
        _float(row, f"wheel{wheel}_world_pose_x_m"),
        _float(row, f"wheel{wheel}_world_pose_y_m"),
        _float(row, f"wheel{wheel}_world_pose_z_m"),
    ], dtype=object)
    quaternion = [
        _float(row, "root_rotation_x"), _float(row, "root_rotation_y"),
        _float(row, "root_rotation_z"), _float(row, "root_rotation_w"),
    ]
    if any(value is None for value in (*root, *pose, *quaternion)):
        return None
    rotation = _quaternion_matrix(*(float(value) for value in quaternion))
    return rotation.T @ (pose.astype(float) - root.astype(float))


def _linear_load_screen(compression: np.ndarray, rate: np.ndarray,
                        force: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(compression) & np.isfinite(rate) & np.isfinite(force)
    if int(np.count_nonzero(valid)) < 4:
        return {"samples": int(np.count_nonzero(valid)), "identified": False}
    design = np.column_stack((np.ones(int(np.count_nonzero(valid))),
                              compression[valid], rate[valid]))
    target = force[valid]
    coefficients, _, rank, singular = np.linalg.lstsq(design, target,
                                                        rcond=None)
    prediction = design @ coefficients
    residual = target - prediction
    variance = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "samples": int(target.size),
        "identified": bool(rank == design.shape[1]),
        "coefficients": {
            "offset_n": float(coefficients[0]),
            "spring_n_per_m": float(coefficients[1]),
            "damper_n_per_mps": float(coefficients[2]),
        },
        "rmse_n": float(np.sqrt(np.mean(residual * residual))),
        "mae_n": float(np.mean(np.abs(residual))),
        "r2": (float(1.0 - np.sum(residual * residual) / variance)
               if variance > 1.0e-12 else None),
        "design_rank": int(rank),
        "design_singular_values": [float(value) for value in singular],
        "runtime_promoted": False,
    }


def _metadata_summary(metadata: dict[str, Any], has_pose: bool) -> dict[str, Any]:
    vehicle = metadata["vehicle"]
    rigid_body = vehicle["rigidBody"]
    wheels = vehicle["wheels"]
    return {
        "scene": metadata.get("sceneName"),
        "unity_version": metadata.get("unityVersion"),
        "trace_build_tag": metadata.get("simulatorBuildTag"),
        "fixed_delta_time_s": metadata.get("fixedDeltaTime"),
        "yaw_axis": rigid_body.get("yawAxis"),
        "yaw_inertia_body_frame_kgm2": rigid_body.get("yawInertiaBodyFrame"),
        "body_inertia_projection_kgm2": {
            "x": rigid_body.get("bodyInertiaX"),
            "y": rigid_body.get("bodyInertiaY"),
            "z": rigid_body.get("bodyInertiaZ"),
        },
        "sprung_masses_kg": [wheel.get("sprungMass") for wheel in wheels],
        "sprung_loads_n": [
            float(wheel["sprungMass"]) * GRAVITY_MPS2 for wheel in wheels
        ],
        "wheel_pose_columns_present": has_pose,
        "exact_competition_scene_capture": (
            str(metadata.get("sceneName", "")).endswith(
                "ModelIdentification_Competition_Temp") and
            str(metadata.get("trace_build_tag", metadata.get(
                "simulatorBuildTag", ""))).startswith("exact_competition_")),
    }


def _settled_reference_y(settled_trace: Path | None,
                         wheel: int,
                         fallback_rows: list[dict[str, str]]) -> float:
    """Return the measured settled wheel pose, in the body frame.

    A separate stationary capture is preferred.  This matters because the
    combined campaign begins while the vehicle is still settling; using its
    first few hundred rows would incorrectly label that startup motion as a
    suspension deflection.
    """
    source_rows = fallback_rows
    if settled_trace is not None:
        with settled_trace.open(newline="", encoding="utf-8") as stream:
            source_rows = list(csv.DictReader(stream))
    values: list[float] = []
    for row in source_rows:
        if row.get(f"wheel{wheel}_grounded") != "1":
            continue
        position = _body_frame_wheel_position(row, wheel)
        if position is not None:
            values.append(float(position[1]))
    if not values:
        raise ValueError(f"no settled pose rows for wheel {wheel}")
    # The final part of the stationary capture is fully settled. If the
    # caller supplied a legacy trace without a separate capture, the same
    # rule still avoids the initial ungrounded/settling rows.
    tail = values[max(0, int(0.8 * len(values))):]
    return float(np.median(tail))


def _read_trace_rows(traces: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    if not traces:
        raise ValueError("at least one trace is required")
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for trace in traces:
        with trace.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            current_fields = list(reader.fieldnames or [])
            if not fieldnames:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(f"trace header mismatch: {trace}")
            rows.extend(reader)
    return rows, fieldnames


def analyze(trace: Path | list[Path], static: Path, output: Path,
            settled_trace: Path | None = None) -> dict[str, Any]:
    metadata = json.loads(static.read_text(encoding="utf-8"))
    traces = [trace] if isinstance(trace, Path) else trace
    rows, fieldnames = _read_trace_rows(traces)
    if not rows:
        raise ValueError("trace contains no rows")

    # The diagnostic writer does not duplicate its CSV header in the JSON
    # snapshot, so the CSV header is the authoritative pose-column check.
    has_pose = all(f"wheel{wheel}_world_pose_y_m" in fieldnames
                   for wheel in WHEELS)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_unity_suspension_contact_audit",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "diagnostic_trace_only",
        "source": {
            "trace": [str(path) for path in traces],
            "static_snapshot": str(static),
            "metadata": _metadata_summary(metadata, has_pose),
            "trace_rows": len(rows),
            "trace_columns": len(fieldnames),
            "pose_columns_present": has_pose,
            "settled_reference_trace": (str(settled_trace)
                                         if settled_trace is not None else None),
        },
        "selection": {
            "all_four_wheels_grounded": True,
            "minimum_body_forward_speed_mps": 0.0,
            "rows_with_complete_pose_and_load": 0,
        },
        "per_wheel": [],
        "acceptance": {
            "runtime_use": False,
            "load_state_promoted": False,
            "slip_coordinate_promoted": False,
            "requires_blind_holdout": True,
        },
    }
    if not has_pose:
        result["status"] = "offline_unity_suspension_contact_audit_pose_unavailable"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        return result

    times = np.asarray([_float(row, "fixed_time_s") or 0.0 for row in rows])
    for wheel in WHEELS:
        body_y: list[float] = []
        force: list[float] = []
        vertical_force: list[float] = []
        valid_indices: list[int] = []
        for index, row in enumerate(rows):
            if row.get(f"wheel{wheel}_grounded") != "1":
                continue
            position = _body_frame_wheel_position(row, wheel)
            contact_force = _float(row, f"wheel{wheel}_contact_force_n")
            normal_y = _float(row, f"wheel{wheel}_contact_normal_y")
            if position is None or contact_force is None:
                continue
            body_y.append(float(position[1]))
            force.append(contact_force)
            vertical_force.append(
                contact_force * normal_y if normal_y is not None else math.nan)
            valid_indices.append(index)

        if not valid_indices:
            result["per_wheel"].append({"wheel": wheel, "samples": 0})
            continue
        indices = np.asarray(valid_indices, dtype=int)
        y = np.asarray(body_y, dtype=float)
        reference_y = _settled_reference_y(settled_trace, wheel, rows)
        # Unity's wheel pose rises relative to the chassis as the suspension
        # is compressed. Positive deflection therefore means compression in
        # this report; this is a geometric coordinate, not a fitted tire term.
        compression = y - reference_y
        valid_times = times[indices]
        if len(compression) > 1:
            compression_rate = np.gradient(compression, valid_times)
        else:
            compression_rate = np.zeros_like(compression)
        raw_force = np.asarray(force, dtype=float)
        projected_force = np.asarray(vertical_force, dtype=float)
        result["selection"]["rows_with_complete_pose_and_load"] = max(
            result["selection"]["rows_with_complete_pose_and_load"],
            len(valid_indices))
        result["per_wheel"].append({
            "wheel": wheel,
            "samples": len(valid_indices),
            "body_frame_wheel_y_m": _stats(y),
            "compression_from_stationary_settled_reference_m": _stats(compression),
            "compression_rate_mps": _stats(compression_rate),
            "wheel_hit_force_n": _stats(raw_force),
            "wheel_hit_force_times_contact_normal_y_n": _stats(projected_force),
            "stationary_settled_reference_body_y_m": reference_y,
            "configured_spring_n_per_m": metadata["vehicle"]["wheels"][wheel][
                "suspensionSpring"]["spring"],
            "configured_damper_n_per_mps": metadata["vehicle"]["wheels"][wheel][
                "suspensionSpring"]["damper"],
            "descriptive_load_screen_raw_force": _linear_load_screen(
                compression, compression_rate, raw_force),
            "descriptive_load_screen_projected_force": _linear_load_screen(
                compression, compression_rate, projected_force),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument(
        "--settled-trace", type=Path,
        help="stationary exact-scene trace used as the zero-deflection pose")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.static, args.output, args.settled_trace)
    print(json.dumps({
        "output": str(args.output),
        "source": result["source"],
        "selection": result["selection"],
        "per_wheel": result["per_wheel"],
        "acceptance": result["acceptance"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
