#!/usr/bin/env python3
"""Validate and provenance-label a Unity model-identification parameter dump."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GUIDE = {
    "total_mass_kg": 3.906,
    "sprung_mass_kg": 3.470,
    "wheel_mass_each_kg": 0.109,
    "wheelbase_m": 0.324,
    "track_m": 0.236,
    "wheel_radius_m": 0.059,
    "com_x_from_rear_axle_m": 0.15532,
    "com_z_m": 0.01434,
    "longitudinal_extremum_slip": 0.15,
    "longitudinal_extremum_value": 0.72,
    "longitudinal_asymptote_slip": 0.25,
    "longitudinal_asymptote_value": 0.464,
    "lateral_extremum_slip": 0.01,
    "lateral_extremum_value": 1.00,
    "lateral_asymptote_slip": 0.10,
    "lateral_asymptote_value": 0.500,
}


def _close(actual: float, expected: float, tolerance: float = 1.0e-4) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def _body_yaw_inertia(moments: dict[str, float], rotation: dict[str, float]) -> float:
    x, y, z, w = (float(rotation[key]) for key in ("x", "y", "z", "w"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("inertia tensor rotation has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # Body-frame z components of R*ex, R*ey, and R*ez.
    rz0 = 2.0 * (x * z + w * y)
    rz1 = 2.0 * (y * z - w * x)
    rz2 = 1.0 - 2.0 * (x * x + y * y)
    return (float(moments["x"]) * rz0 * rz0 +
            float(moments["y"]) * rz1 * rz1 +
            float(moments["z"]) * rz2 * rz2)


def _check(name: str, actual: float, expected: float, tolerance: float = 1.0e-4) -> dict[str, Any]:
    return {
        "actual": actual,
        "expected_guide": expected,
        "absolute_difference": abs(actual - expected),
        "match": _close(actual, expected, tolerance),
    }


def analyze(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("diagnosticOnly") is not True:
        raise ValueError("dump is not marked diagnosticOnly")
    if payload.get("runtimeControlInput") is not False:
        raise ValueError("dump is marked as a runtime control input")
    vehicle = payload["vehicle"]
    rigid_body = vehicle["rigidBody"]
    wheels = vehicle["wheels"]
    if len(wheels) != 4:
        raise ValueError("dump must contain exactly four wheels")

    sprung_mass = float(rigid_body["mass"])
    wheel_mass = sum(float(wheel["mass"]) for wheel in wheels)
    total_mass = sprung_mass + wheel_mass
    com = rigid_body["centerOfMass"]
    derived_iz = _body_yaw_inertia(
        rigid_body["inertiaTensor"], rigid_body["inertiaTensorRotation"])
    checks = {
        "sprung_mass_kg": _check("sprung_mass_kg", sprung_mass, GUIDE["sprung_mass_kg"]),
        "wheel_mass_total_kg": _check(
            "wheel_mass_total_kg", wheel_mass, 4.0 * GUIDE["wheel_mass_each_kg"]),
        "total_mass_kg": _check("total_mass_kg", total_mass, GUIDE["total_mass_kg"]),
        "controller_wheelbase_m": _check(
            "controller_wheelbase_m", float(vehicle["wheelbaseM"]), GUIDE["wheelbase_m"]),
        "controller_track_m": _check(
            "controller_track_m", float(vehicle["trackWidthM"]), GUIDE["track_m"]),
        "controller_wheel_radius_m": _check(
            "controller_wheel_radius_m", float(vehicle["wheelRadiusControllerM"]),
            GUIDE["wheel_radius_m"]),
        "computed_yaw_inertia_kgm2": _check(
            "computed_yaw_inertia_kgm2", derived_iz,
            float(rigid_body["yawInertiaBodyFrame"]), 1.0e-5),
    }
    curve_checks: dict[str, Any] = {}
    forward = wheels[0]["forwardFriction"]
    sideways = wheels[0]["sidewaysFriction"]
    for prefix, curve, keys in (
        ("longitudinal", forward, (
            ("extremum_slip", "extremumSlip"),
            ("extremum_value", "extremumValue"),
            ("asymptote_slip", "asymptoteSlip"),
            ("asymptote_value", "asymptoteValue"),
        )),
        ("lateral", sideways, (
            ("extremum_slip", "extremumSlip"),
            ("extremum_value", "extremumValue"),
            ("asymptote_slip", "asymptoteSlip"),
            ("asymptote_value", "asymptoteValue"),
        )),
    ):
        curve_checks[prefix] = {
            name: _check(name, float(curve[field]), GUIDE[f"{prefix}_{name}"])
            for name, field in keys
        }

    return {
        "schema_version": 1,
        "status": "diagnostic_dump_analyzed",
        "input": str(path),
        "simulator_build_tag": payload.get("simulatorBuildTag", "unspecified"),
        "unity_version": payload.get("unityVersion", "unspecified"),
        "source_provenance": {
            "fixed_parameters": "unity_runtime_diagnostic_dump",
            "trajectory_ground_truth": "offline_only",
            "runtime_controller_consumption": False,
        },
        "derived": {
            "sprung_mass_kg": sprung_mass,
            "wheel_mass_total_kg": wheel_mass,
            "total_mass_kg": total_mass,
            "center_of_mass_unity_body_frame": com,
            "yaw_inertia_body_frame_kgm2": derived_iz,
            "static_normal_loads_N": {
                "front_axle": total_mass * 9.81 * 0.15532 / 0.324,
                "rear_axle": total_mass * 9.81 * (0.324 - 0.15532) / 0.324,
            },
        },
        "guide_crosscheck": checks,
        "friction_curve_guide_crosscheck": curve_checks,
        "next_action": (
            "Use dump values as fixed structural parameters; fit only actuator, "
            "drag, load-transfer, or residual terms that remain unexplained."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.dump)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
