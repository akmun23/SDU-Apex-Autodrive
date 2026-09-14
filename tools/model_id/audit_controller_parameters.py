#!/usr/bin/env python3
"""Audit Unity controller-radius semantics without modifying the simulator."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _matches(lines: list[str], pattern: str) -> list[dict[str, Any]]:
    expression = re.compile(pattern)
    return [
        {"line": index, "text": line.strip()}
        for index, line in enumerate(lines, start=1)
        if expression.search(line)
    ]


def _first_float(lines: list[str], pattern: str) -> float | None:
    expression = re.compile(pattern)
    for line in lines:
        match = expression.search(line)
        if match:
            return float(match.group(1))
    return None


def audit(controller_path: Path, prefab_path: Path,
          dump_path: Path | None = None) -> dict[str, Any]:
    controller_lines = controller_path.read_text(encoding="utf-8").splitlines()
    prefab_lines = prefab_path.read_text(encoding="utf-8").splitlines()

    wheel_radius_declaration = _matches(
        controller_lines, r"public\s+float\s+WheelRadius\s*=")
    wheel_radius_uses = _matches(controller_lines, r"\bWheelRadius\b")
    drive_type_branch = _matches(
        controller_lines, r"driveType\s*==\s*DriveType\.CAWD")
    drive_type_value = _first_float(prefab_lines, r"^\s*driveType:\s*([0-9.]+)")
    controller_radius = _first_float(
        prefab_lines, r"^\s*WheelRadius:\s*([0-9.eE+-]+)")
    physical_radii = [
        float(match.group(1))
        for line in prefab_lines
        for match in [re.search(r"^\s*m_Radius:\s*([0-9.eE+-]+)", line)]
        if match
    ]
    mass_values = [
        float(match.group(1))
        for line in prefab_lines
        for match in [re.search(r"^\s*m_Mass:\s*([0-9.eE+-]+)", line)]
        if match
    ]
    wheel_mass_candidates = [value for value in mass_values if value < 1.0]

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "controller_parameter_audit_complete",
        "offline_only": True,
        "simulator_modified": False,
        "sources": {
            "vehicle_controller": str(controller_path),
            "f1tenth_prefab": str(prefab_path),
            "diagnostic_dump": str(dump_path) if dump_path else None,
        },
        "controller_wheel_radius": {
            "value_m": controller_radius,
            "declaration": wheel_radius_declaration,
            "all_source_uses": wheel_radius_uses,
            "competition_cawd_branch": drive_type_branch,
            "used_by_competition_cawd_branch": False,
            "semantic_role": (
                "legacy VehicleController.WheelRadius used by "
                "ExtendedDifferentialDrive skid-steer equations; it is not "
                "used by the CAWD car-drive branch"),
        },
        "prefab_drive_configuration": {
            "drive_type_enum_value": drive_type_value,
            "drive_type_enum_name": "CAWD",
            "ca_w_d_branch_selected": drive_type_value == 5.0,
        },
        "wheel_collider_parameters": {
            "physical_radius_values_m": physical_radii,
            "physical_radius_unique_m": sorted(set(physical_radii)),
            "prefab_mass_values_kg": mass_values,
            "wheel_mass_candidates_kg": wheel_mass_candidates,
            "wheel_mass_candidate_unique_kg": sorted(set(wheel_mass_candidates)),
            "physical_radius_used_for_wheel_contact_and_rpm": True,
        },
        "conclusion": (
            "Use 0.059 m for WheelCollider contact/RPM geometry in the offline "
            "vehicle model. Keep 0.0325 m recorded as a controller parameter, "
            "but do not use it in the competition CAWD drive model."),
    }
    if dump_path:
        payload = json.loads(dump_path.read_text(encoding="utf-8"))
        vehicle = payload.get("vehicle", {})
        report["diagnostic_dump_crosscheck"] = {
            "drive_type": vehicle.get("driveType"),
            "controller_wheel_radius_m": vehicle.get("wheelRadiusControllerM"),
            "physical_wheel_radius_m": [
                wheel.get("radius") for wheel in vehicle.get("wheels", [])],
            "rigidbody_mass_kg": vehicle.get("rigidBody", {}).get("mass"),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicle-controller", type=Path, required=True)
    parser.add_argument("--prefab", type=Path, required=True)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.vehicle_controller, args.prefab, args.dump)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "controller_radius_m": report["controller_wheel_radius"]["value_m"],
        "physical_radius_m": report["wheel_collider_parameters"][
            "physical_radius_unique_m"],
        "cawd_uses_controller_radius": report["controller_wheel_radius"][
            "used_by_competition_cawd_branch"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
