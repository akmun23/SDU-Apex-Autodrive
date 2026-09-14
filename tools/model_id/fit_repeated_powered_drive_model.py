#!/usr/bin/env python3
"""Fit and cross-score the provisional continuous wheel-drive model.

This uses the existing offline wheel-state basis from
``fit_continuous_drive_model.py``. Repeats are kept as separate source-time
sequences: the first repeats are used for fitting and later repeats are held
out. No simulator truth is exported to runtime code by this tool.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fit_continuous_drive_model import (
    WHEEL_COUNT,
    _design,
    _fit,
    _powered_records,
    _read_trace,
    _replay,
    _stats,
    _wheel_series,
)


def _read_repeat(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    trace = path / "wheel_contact_trace.csv"
    dump = path / "simulator_parameters.json"
    rows = _read_trace(trace)
    with dump.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    return rows, metadata


def _powered_by_wheel(rows: list[dict[str, str]]) -> list[list[dict[str, float]]]:
    result = []
    for wheel in range(WHEEL_COUNT):
        series = _wheel_series(rows, wheel)
        if not series:
            result.append([])
            continue
        end_time = series[-1]["time_s"] + 1.0e-6
        result.append(_powered_records(series, ((0.0, end_time),)))
    return result


def _fit_wheel(train: list[dict[str, float]], validation: list[dict[str, float]]) -> dict[str, Any]:
    fit_report = _fit(train)
    coefficients = fit_report["coefficients"]
    coefficient_vector = np.asarray([
        coefficients["motor_torque_to_omega_dot_per_kgm2"],
        coefficients["contact_torque_to_omega_dot_per_kgm2"],
        coefficients["wheel_damping_per_kgm2"],
    ])
    features, targets = _design(validation)
    residual = features @ coefficient_vector - targets
    return {
        "fit": fit_report,
        "validation_derivative_fit": _stats(residual),
        "validation_samples": len(validation),
        "arbitrary_dt_replay": {
            str(period): _replay(validation, coefficients, period)
            for period in (0.001, 0.005, 0.010, 0.025, 0.050)
        },
    }


def fit_campaign(root: Path, output: Path) -> dict[str, Any]:
    repeat_dirs = sorted(
        path for path in root.glob("repeat_*")
        if (path / "wheel_contact_trace.csv").is_file() and
        (path / "simulator_parameters.json").is_file())
    if len(repeat_dirs) < 3:
        raise ValueError("at least three complete repeats are required")

    repeats: list[list[list[dict[str, float]]]] = []
    repeat_reports: list[dict[str, Any]] = []
    for directory in repeat_dirs:
        rows, metadata = _read_repeat(directory)
        records_by_wheel = _powered_by_wheel(rows)
        repeats.append(records_by_wheel)
        times = [float(row["fixed_time_s"]) for row in rows]
        dts = np.diff(np.asarray(times))
        requested = sorted({round(float(row["experiment_requested_throttle_norm"]), 6)
                            for row in rows})
        brakes = [float(row[f"wheel{wheel}_brake_torque_nm"])
                  for row in rows for wheel in range(WHEEL_COUNT)]
        repeat_reports.append({
            "directory": str(directory),
            "build_tag": metadata.get("simulatorBuildTag"),
            "rows": len(rows),
            "source_time_s": [times[0], times[-1]],
            "dt_s": {
                "median": float(np.median(dts)),
                "p95": float(np.percentile(dts, 95)),
                "max": float(np.max(dts)),
            },
            "requested_commands": requested,
            "maximum_brake_torque_nm": max(brakes),
            "powered_records_by_wheel": [len(records) for records in records_by_wheel],
        })

    split = max(1, len(repeats) - 2)
    wheel_reports: dict[str, Any] = {}
    for wheel in range(WHEEL_COUNT):
        train = [record for repeat in repeats[:split] for record in repeat[wheel]]
        validation = [record for repeat in repeats[split:] for record in repeat[wheel]]
        wheel_reports[f"wheel_{wheel}"] = _fit_wheel(train, validation)

    wheel_replay_p95 = [
        report["arbitrary_dt_replay"]["0.025"]["error_radps"]["p95"]
        for report in wheel_reports.values()
    ]

    report = {
        "schema_version": 1,
        "status": "continuous_powered_drive_model_e2_offline_only",
        "acceptance": "not_accepted",
        "ground_truth_use": "offline_identification_and_scoring_only",
        "campaign_root": str(root),
        "repeat_split": {
            "fit_repeats": [str(path) for path in repeat_dirs[:split]],
            "validation_repeats": [str(path) for path in repeat_dirs[split:]],
            "fit_rule": "first repeats pooled; final two independent repeats held out",
        },
        "actuator_semantics": {
            "zero_throttle_included": False,
            "maximum_validation_brake_torque_nm": max(
                report["maximum_brake_torque_nm"] for report in repeat_reports),
            "all_commands_positive": all(
                report["requested_commands"] and
                min(report["requested_commands"]) > 0.0
                for report in repeat_reports),
        },
        "repeat_reports": repeat_reports,
        "model": {
            "equation": (
                "domega = km*motor_torque - "
                "kx*radius*Fz*phi_x(Sx) - kd*omega"),
            "wheel_radius_m": 0.059,
            "contact_force": "WheelHit.force scalar proxy",
            "slip_basis": "Unity forward-slip diagnostic field",
            "coefficients_are_physical_inertia_claims": False,
        },
        "wheels": wheel_reports,
        "body_speed_recursive_prediction": {
            "status": "not_implemented",
            "reason": "this stage identifies wheel-state dynamics only"
        },
        "gate": {
            "repeatability_passed": all(
                report["rows"] >= 50000 and
                report["maximum_brake_torque_nm"] == 0.0
                for report in repeat_reports),
            "independent_wheel_replay_p95_radps_at_25ms": wheel_replay_p95,
            "independent_wheel_replay_passed": all(
                value is not None and math.isfinite(value) and value <= 1.0
                for value in wheel_replay_p95),
            "body_u_recursive_prediction_passed": False,
            "production_promotion": False,
        },
        "limitations": [
            "WheelHit.force is not a direct tire longitudinal force measurement.",
            "The model still uses measured slip and load as offline regressors.",
            "Body-speed recursion and actuator inverse are not implemented here.",
            "No MPC or simulator behavior is changed by this report.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = fit_campaign(args.campaign_root.resolve(), args.output.resolve())
    print(json.dumps({
        "status": report["status"],
        "repeat_count": len(report["repeat_reports"]),
        "output": str(args.output.resolve()),
        "production_promotion": report["gate"]["production_promotion"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
