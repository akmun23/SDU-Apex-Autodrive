#!/usr/bin/env python3
"""Measure prefab wheel-mesh collider proximity to an open-plane capture.

This is a geometry audit, not a contact-force reconstruction. It uses the
captured WheelCollider pose at each 1 ms row and the four enabled cylinder
MeshCollider transforms serialized on the F1TENTH wheel visuals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning.physx_vehicle_source import (
    unity_convex_mesh_plane_clearance_m,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
WHEELS = ("fl", "fr", "rl", "rr")
PLANE_NORMAL_WORLD = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
PLANE_OFFSET_M = 0.0
DEFAULT_PAIR_CONTACT_OFFSET_M = 0.020
BUILTIN_CYLINDER_JSON = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "unity_builtin_cylinder_mesh_2022.3.52f1.json")

# Local child transforms of the active, enabled built-in Cylinder MeshColliders
# under the four WheelCollider visual transforms in Assets/Prefabs/F1TENTH.
COLLIDER_CHILD_TRANSFORMS = {
    "fl": ((0.0008, 0.0007, 0.0001),
           (0.0077131568, 0.0077131568, 0.7070647, 0.7070647),
           (0.11, 0.022, 0.11)),
    "fr": ((0.0009, -0.0001, 0.0015),
           (0.0, 0.0, 0.7071068, 0.7071068),
           (0.10999997, 0.021999996, 0.110000014)),
    "rl": ((0.0, 0.0, 0.0),
           (0.0, 0.0, 0.7071068, 0.7071068),
           (0.109999985, 0.022, 0.11000002)),
    "rr": ((0.0, 0.0, 0.0),
           (0.0, 0.0, 0.7071068, 0.7071068),
           (0.10999999, 0.022, 0.11000003)),
}


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left.T
    rx, ry, rz, rw = right.T
    result = np.column_stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ))
    return result / np.linalg.norm(result, axis=1)[:, None]


def _rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, :3]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[:, 3, None] * twice_cross \
        + np.cross(xyz, twice_cross)


def _stats(clearance: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(clearance.size),
        "minimum_m": float(clearance.min()),
        "p05_m": float(np.quantile(clearance, 0.05)),
        "median_m": float(np.median(clearance)),
        "p95_m": float(np.quantile(clearance, 0.95)),
        "maximum_m": float(clearance.max()),
        "fraction_within_default_pair_contact_offset": float(
            np.mean(clearance <= DEFAULT_PAIR_CONTACT_OFFSET_M)),
        "fraction_with_negative_source_mesh_hull_clearance": float(
            np.mean(clearance < 0.0)),
    }


def analyze_capture(path: Path, stride: int = 1,
                    inspect_rows: tuple[int, ...] = ()) -> dict[str, Any]:
    if stride < 1:
        raise ValueError("stride must be positive")
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, separator, value = line[2:].strip().partition("=")
            if separator:
                metadata[key] = value
    if not math.isclose(float(metadata["fixed_delta_time_s"]), 0.001,
                        rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("capture must use the configured 1 ms physics grid")

    columns = ["sim_time_s", "pos_world_x_m", "pos_world_y_m",
               "pos_world_z_m", "rot_world_x", "rot_world_y",
               "rot_world_z", "rot_world_w"]
    for wheel in WHEELS:
        columns.extend(f"{wheel}_wheel_pose_body_{axis}" for axis in "xyz")
        columns.extend(f"{wheel}_wheel_pose_q{axis}" for axis in "xyzw")
    frame = pd.read_csv(path, comment="#", usecols=columns).iloc[::stride]

    actor_position = frame[["pos_world_x_m", "pos_world_y_m",
                            "pos_world_z_m"]].to_numpy(dtype=np.float64)
    actor_rotation = frame[["rot_world_x", "rot_world_y", "rot_world_z",
                            "rot_world_w"]].to_numpy(dtype=np.float64)
    actor_rotation /= np.linalg.norm(actor_rotation, axis=1)[:, None]
    cylinder_asset = json.loads(BUILTIN_CYLINDER_JSON.read_text(
        encoding="utf-8"))
    cylinder_vertices = np.asarray(
        cylinder_asset["vertices_unscaled_local_xyz"], dtype=np.float64)
    collider_clearances: dict[str, np.ndarray] = {}

    for wheel in WHEELS:
        wheel_position_body = frame[[
            f"{wheel}_wheel_pose_body_{axis}" for axis in "xyz"
        ]].to_numpy(dtype=np.float64)
        wheel_rotation_body = frame[[
            f"{wheel}_wheel_pose_q{axis}" for axis in "xyzw"
        ]].to_numpy(dtype=np.float64)
        wheel_rotation_body /= np.linalg.norm(
            wheel_rotation_body, axis=1)[:, None]
        wheel_rotation_world = _quat_mul(actor_rotation,
                                         wheel_rotation_body)
        child_position, child_rotation, child_scale = (
            COLLIDER_CHILD_TRANSFORMS[wheel])
        child_position_array = np.broadcast_to(
            np.asarray(child_position, dtype=np.float64),
            wheel_position_body.shape)
        child_rotation_array = np.broadcast_to(
            np.asarray(child_rotation, dtype=np.float64),
            wheel_rotation_world.shape)
        collider_rotation_world = _quat_mul(wheel_rotation_world,
                                            child_rotation_array)
        wheel_position_world = actor_position + _rotate(
            actor_rotation, wheel_position_body)
        child_offset_world = _rotate(wheel_rotation_world,
                                     child_position_array)
        cylinder_center_world = wheel_position_world + child_offset_world
        collider_clearances[wheel] = unity_convex_mesh_plane_clearance_m(
            cylinder_center_world,
            collider_rotation_world,
            cylinder_vertices,
            child_scale,
            (*PLANE_NORMAL_WORLD, PLANE_OFFSET_M))

    per_wheel = {wheel: _stats(values)
                 for wheel, values in collider_clearances.items()}
    all_clearances = np.concatenate(tuple(collider_clearances.values()))
    inspected = {}
    for row in inspect_rows:
        if row < 0 or row >= len(frame):
            raise ValueError(f"inspection row {row} is outside the capture")
        inspected[str(row)] = {
            "sim_time_s": float(frame.iloc[row]["sim_time_s"]),
            "clearance_m_by_wheel": {
                wheel: float(values[row])
                for wheel, values in collider_clearances.items()},
        }
    return {
        "capture": str(path.resolve().relative_to(ROOT)),
        "rows_read_after_stride": len(frame),
        "stride_rows": stride,
        "physics_step_s": 0.001,
        "static_plane_world_xyzd": (0.0, 1.0, 0.0, 0.0),
        "cylinder_mesh": {
            "unity_builtin_mesh_file_id": 10206,
            "unity_editor_version": cylinder_asset["unity_editor_version"],
            "resource_sha256": cylinder_asset["resource_sha256"],
            "vertex_count": len(cylinder_vertices),
            "support_model": "support of exact extracted source-mesh vertices (convex hull equivalent)",
        },
        "default_pair_contact_offset_m": DEFAULT_PAIR_CONTACT_OFFSET_M,
        "overall": _stats(all_clearances),
        "per_wheel": per_wheel,
        "inspection_rows_after_stride": inspected,
        "interpretation": (
            "Clearance of the extracted Unity source-mesh convex hull. The "
            "Unity-cooked collider shape and solver are not extracted. Being "
            "within the pair contact offset means PhysX may generate a "
            "contact pair; it does not prove an impulse was applied. No contact "
            "forces or solver state are inferred from this telemetry."),
        "inputs": [
            "current body pose",
            "current WheelCollider GetWorldPose position and rotation",
            "F1TENTH prefab wheel MeshCollider child transforms",
            "Unity DynamicsManager default contact offset",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--inspect-row", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_capture(args.capture, args.stride,
                             tuple(args.inspect_row))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
