#!/usr/bin/env python3
"""Validate and provenance-label a Unity model-identification parameter dump."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


GUIDE = {
    "total_mass_if_additive_kg": 3.906,
    "sprung_mass_kg": 3.470,
    "wheel_mass_each_kg": 0.109,
    "wheelbase_m": 0.324,
    "track_m": 0.236,
    "wheel_radius_m": 0.059,
    "controller_wheel_radius_m": 0.0325,
    "com_x_from_rear_axle_m": 0.15532,
    "rigidbody_com_local_y_m": 0.06434,
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


def _body_axis_inertia(moments: dict[str, float], rotation: dict[str, float],
                       axis: str) -> float:
    if axis not in ("x", "y", "z"):
        raise ValueError("body inertia axis must be x, y, or z")
    x, y, z, w = (float(rotation[key]) for key in ("x", "y", "z", "w"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("inertia tensor rotation has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation_matrix = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ]
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    # The diagonal element is the sum of each principal moment times the
    # squared component of that principal axis along the selected body axis.
    return sum(
        float(moments[principal]) * rotation_matrix[axis_index][index] ** 2
        for index, principal in enumerate(("x", "y", "z")))


def _check(name: str, actual: float, expected: float, tolerance: float = 1.0e-4) -> dict[str, Any]:
    return {
        "actual": actual,
        "expected_guide": expected,
        "absolute_difference": abs(actual - expected),
        "match": _close(actual, expected, tolerance),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


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
    mass_sum_if_additive = sprung_mass + wheel_mass
    com = rigid_body["centerOfMass"]
    # Unity uses x-right/z-forward; the model/API uses x-forward/y-left.
    com_api_x = float(com["z"])
    com_api_y = -float(com["x"])
    wheel_positions_api = [
        {
            "x_m": float(wheel["positionVehicleFrame"]["z"]) - com_api_x,
            "y_m": -float(wheel["positionVehicleFrame"]["x"]) - com_api_y,
        }
        for wheel in wheels
    ]
    front_positions = wheel_positions_api[:2]
    rear_positions = wheel_positions_api[2:]
    derived_wheelbase = (
        sum(position["x_m"] for position in front_positions) / 2.0 -
        sum(position["x_m"] for position in rear_positions) / 2.0)
    derived_track = abs(front_positions[0]["y_m"] -
                        front_positions[1]["y_m"])
    com_x_from_rear = -sum(position["x_m"] for position in rear_positions) / 2.0
    # WheelCollider.mass is a separate wheel parameter. The dump does not
    # establish that Unity adds it to Rigidbody.mass for body load/inertia, so
    # use Rigidbody.mass for this body-load report and expose the additive sum
    # only as an unresolved accounting alternative.
    front_axle_load = sprung_mass * 9.81 * com_x_from_rear / derived_wheelbase
    rear_axle_load = sprung_mass * 9.81 * (
        derived_wheelbase - com_x_from_rear) / derived_wheelbase
    body_inertias = {
        axis: _body_axis_inertia(
            rigid_body["inertiaTensor"],
            rigid_body["inertiaTensorRotation"], axis)
        for axis in ("x", "y", "z")
    }
    if rigid_body.get("yawAxis") != "body_y":
        raise ValueError(
            "diagnostic dump must explicitly declare Unity body Y as yawAxis")
    for axis in ("x", "y", "z"):
        field = f"bodyInertia{axis.upper()}"
        if field not in rigid_body:
            raise ValueError(f"diagnostic dump is missing corrected {field} field")
        if not math.isclose(
                body_inertias[axis], float(rigid_body[field]),
                rel_tol=1.0e-4, abs_tol=2.0e-6):
            raise ValueError(
                f"diagnostic {field} disagrees with inertia tensor projection")
    derived_iz = body_inertias["y"]
    wheel_diagnostic_completeness = []
    for wheel in wheels:
        spring = wheel.get("suspensionSpring")
        wheel_diagnostic_completeness.append({
            "name": wheel.get("name", "unspecified"),
            "sprung_mass_present": _optional_float(
                wheel.get("sprungMass")) is not None,
            "local_rotation_present": "localRotation" in wheel,
            "suspension_spring_present": isinstance(spring, dict) and all(
                key in spring for key in (
                    "spring", "damper", "targetPosition")),
        })
    rigid_body_diagnostic_completeness = {
        "max_angular_velocity_present": (
            _optional_float(rigid_body.get("maxAngularVelocity")) is not None),
    }
    diagnostic_fields_complete = (
        all(all(value for key, value in item.items() if key != "name")
            for item in wheel_diagnostic_completeness) and
        all(rigid_body_diagnostic_completeness.values()))
    checks = {
        "sprung_mass_kg": _check("sprung_mass_kg", sprung_mass, GUIDE["sprung_mass_kg"]),
        "wheel_mass_total_kg": _check(
            "wheel_mass_total_kg", wheel_mass, 4.0 * GUIDE["wheel_mass_each_kg"]),
        "mass_sum_if_additive_kg": _check(
            "mass_sum_if_additive_kg", mass_sum_if_additive,
            GUIDE["total_mass_if_additive_kg"]),
        "controller_wheelbase_m": _check(
            "controller_wheelbase_m", float(vehicle["wheelbaseM"]), GUIDE["wheelbase_m"]),
        "controller_track_m": _check(
            "controller_track_m", float(vehicle["trackWidthM"]), GUIDE["track_m"]),
        "physical_wheel_radius_m": _check(
            "physical_wheel_radius_m",
            sum(float(wheel["radius"]) for wheel in wheels) / len(wheels),
            GUIDE["wheel_radius_m"]),
        "controller_wheel_radius_m": _check(
            "controller_wheel_radius_m", float(vehicle["wheelRadiusControllerM"]),
            GUIDE["controller_wheel_radius_m"]),
        "computed_yaw_inertia_kgm2": _check(
            "computed_yaw_inertia_kgm2", derived_iz,
            float(rigid_body["yawInertiaBodyFrame"]), 1.0e-5),
    }
    curve_checks: dict[str, Any] = {}
    for wheel_index, wheel in enumerate(wheels):
        wheel_checks: dict[str, Any] = {}
        for prefix, curve, keys in (
            ("longitudinal", wheel["forwardFriction"], (
                ("extremum_slip", "extremumSlip"),
                ("extremum_value", "extremumValue"),
                ("asymptote_slip", "asymptoteSlip"),
                ("asymptote_value", "asymptoteValue"),
            )),
            ("lateral", wheel["sidewaysFriction"], (
                ("extremum_slip", "extremumSlip"),
                ("extremum_value", "extremumValue"),
                ("asymptote_slip", "asymptoteSlip"),
                ("asymptote_value", "asymptoteValue"),
            )),
        ):
            wheel_checks[prefix] = {
                name: _check(
                    name, float(curve[field]), GUIDE[f"{prefix}_{name}"])
                for name, field in keys
            }
        curve_checks[f"wheel_{wheel_index}"] = wheel_checks

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
            "rigidbody_mass_kg": sprung_mass,
            "center_of_mass_unity_body_frame": com,
            "center_of_mass_api_body_frame": {
                "x_m": com_api_x,
                "y_m": com_api_y,
            },
            "wheel_positions_api_body_frame": wheel_positions_api,
            "derived_contact_geometry": {
                "wheelbase_m": derived_wheelbase,
                "track_m": derived_track,
                "com_x_from_rear_axle_m": com_x_from_rear,
            },
            "controller_parameters": {
                "drive_type": vehicle.get("driveType"),
                "steer_type": vehicle.get("steerType"),
                "controller_wheel_radius_m": float(
                    vehicle["wheelRadiusControllerM"]),
                "controller_wheel_radius_role": (
                    "VehicleController.WheelRadius; used by the skid-steer "
                    "ExtendedDifferentialDrive branch, not the CAWD car "
                    "drive branch"),
                "competition_drive_type_uses_controller_radius": (
                    str(vehicle.get("driveType")) == "SkidSteer"),
                "physical_wheel_radius_m": sum(
                    float(wheel["radius"]) for wheel in wheels) / len(wheels),
            },
            "body_frame_inertia_kgm2": body_inertias,
            "yaw_axis": "body_y",
            "yaw_inertia_body_frame_kgm2": derived_iz,
            "static_normal_loads_N": {
                "front_axle": front_axle_load,
                "rear_axle": rear_axle_load,
                "front_each": front_axle_load / 2.0,
                "rear_each": rear_axle_load / 2.0,
            },
            "mass_accounting": {
                "rigidbody_mass_kg": sprung_mass,
                "wheel_mass_total_kg": wheel_mass,
                "sum_if_additive_kg": mass_sum_if_additive,
                "status": "explicit_body_and_wheel_accounting",
                "body_load_calculation_uses": "Rigidbody.mass only",
                "reason": (
                    "WheelCollider.mass is recorded separately; the dump and "
                    "source architecture do not prove that it is added to "
                    "Rigidbody.mass for body translation or inertia."),
            },
            "wheel_dynamics": [
                {
                    "wheel": index,
                    "name": wheel["name"],
                    "radius_m": float(wheel["radius"]),
                    "mass_kg": float(wheel["mass"]),
                    "sprung_mass_kg": _optional_float(
                        wheel.get("sprungMass")),
                    "suspension_distance_m": float(
                        wheel["suspensionDistance"]),
                    "wheel_damping_rate": float(
                        wheel["wheelDampingRate"]),
                    "force_app_point_distance_m": float(
                        wheel["forceAppPointDistance"]),
                    "local_rotation": wheel.get("localRotation"),
                    "suspension_spring": (
                        {
                            "spring_n_per_m": float(
                                wheel["suspensionSpring"]["spring"]),
                            "damper_ns_per_m": float(
                                wheel["suspensionSpring"]["damper"]),
                            "target_position": float(
                                wheel["suspensionSpring"]["targetPosition"]),
                        }
                        if isinstance(wheel.get("suspensionSpring"), dict)
                        else None),
                }
                for index, wheel in enumerate(wheels)
            ],
            "rigid_body_damping": {
                "drag": float(rigid_body["drag"]),
                "angular_drag": float(rigid_body["angularDrag"]),
                "max_angular_velocity_radps": _optional_float(
                    rigid_body.get("maxAngularVelocity")),
            },
            "diagnostic_field_completeness": {
                "required_dynamic_fields_present": diagnostic_fields_complete,
                "rigid_body": rigid_body_diagnostic_completeness,
                "wheels": wheel_diagnostic_completeness,
            },
            "solver_and_timing": {
                "fixed_delta_time_s": float(payload["fixedDeltaTime"]),
                "maximum_allowed_timestep_s": float(
                    payload["maximumAllowedTimestep"]),
                "time_scale": float(payload["timeScale"]),
                "gravity_mps2": payload["gravity"],
                "default_solver_iterations": int(
                    payload["defaultSolverIterations"]),
                "default_solver_velocity_iterations": int(
                    payload["defaultSolverVelocityIterations"]),
                "default_contact_offset_m": float(
                    payload["defaultContactOffset"]),
                "default_max_depenetration_velocity_mps": float(
                    payload["defaultMaxDepenetrationVelocity"]),
                "default_max_angular_speed_radps": float(
                    payload["defaultMaxAngularSpeed"]),
                "v_sync_count": int(payload["vSyncCount"]),
                "target_frame_rate": int(payload["targetFrameRate"]),
            },
        },
        "guide_crosscheck": checks,
        "friction_curve_guide_crosscheck": curve_checks,
        "next_action": (
            "Use Rigidbody.mass for body equations, retain WheelCollider.mass "
            "as a separate wheel accounting term, audit the controller radius "
            "semantics, then fit continuous drive and causal load terms."
            if diagnostic_fields_complete else
            "Rebuild the diagnostic player with the complete v2 static field "
            "set before using suspension or wheel-load values."
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
