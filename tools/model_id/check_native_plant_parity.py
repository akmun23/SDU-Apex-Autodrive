#!/usr/bin/env python3
"""Compare the Python reference plant with the native C replay plant."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from structured_vehicle_plant import PlantParameters, step


class CState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "x_m", "y_m", "yaw_rad", "u_mps", "v_mps", "r_radps",
        "steering_rad", "wheel_speed_mps")]


class CInput(ctypes.Structure):
    _fields_ = [("steering_target_norm", ctypes.c_float),
                ("throttle_norm", ctypes.c_float)]


class CParameters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_float) for name in (
            "mass_kg", "lf_m", "lr_m", "iz_kgm2", "max_steering_rad",
            "steering_rate_radps", "max_speed_mps", "force_max_n",
            "slip_gain_per_mps", "coast_speed_drag_n_per_mps",
            "cf_n_per_rad", "cr_n_per_rad", "df_n", "dr_n")
    ] + [("tire_model", ctypes.c_uint8),
         ("wheel_coefficients", ctypes.c_float * 6)]


def _parameters(parameters: PlantParameters) -> CParameters:
    result = CParameters()
    for name in (
            "mass_kg", "lf_m", "lr_m", "iz_kgm2", "max_steering_rad",
            "steering_rate_radps", "max_speed_mps", "force_max_n",
            "slip_gain_per_mps", "coast_speed_drag_n_per_mps",
            "cf_n_per_rad", "cr_n_per_rad", "df_n", "dr_n"):
        setattr(result, name, getattr(parameters, name))
    result.tire_model = 0 if parameters.tire_model == "linear_saturated" else 1
    result.wheel_coefficients = (ctypes.c_float * 6)(*parameters.wheel_coefficients)
    return result


def _state(values: np.ndarray) -> CState:
    return CState(*(float(value) for value in values))


def _state_array(state: CState) -> np.ndarray:
    return np.asarray([
        state.x_m, state.y_m, state.yaw_rad, state.u_mps,
        state.v_mps, state.r_radps, state.steering_rad,
        state.wheel_speed_mps,
    ], dtype=float)


def _compile(source_root: Path, output: Path) -> None:
    subprocess.run([
        "cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        "-shared", "-fPIC", str(source_root / "f1tenth_mpc/src/vehicle_plant.c"),
        "-I", str(source_root / "f1tenth_mpc/include"), "-lm", "-o", str(output),
    ], check=True)


def check(source_root: Path, lateral_report: Path,
          longitudinal_report: Path, output: Path) -> dict[str, object]:
    lateral = json.loads(lateral_report.read_text(encoding="utf-8"))
    longitudinal = json.loads(longitudinal_report.read_text(encoding="utf-8"))
    lateral_values = lateral["candidate_comparison"]["Y1_linear_saturated"]
    lateral_parameters = lateral_values["parameters"]["parameters"]
    longitudinal_parameters = longitudinal["models"]["wheel_dynamic"]["parameters"]
    parameters = PlantParameters.from_longitudinal_parameters(
        longitudinal_parameters).with_lateral(lateral_parameters)

    fixtures = [
        (np.asarray([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]), 0.0, 0.2, 0.025),
        (np.asarray([1.0, -0.4, 0.2, 4.0, 0.15, 0.3, 0.05, 4.5]), 0.5, 0.6, 0.024),
        (np.asarray([-2.0, 3.0, -1.2, 8.0, -0.2, -0.7, -0.2, 7.5]), -0.8, 0.0, 0.028),
        (np.asarray([4.0, 1.0, 2.7, 12.0, 0.3, 1.1, 0.35, 12.2]), 0.9, 0.8, 0.025),
    ]
    with tempfile.TemporaryDirectory(prefix="vehicle_plant_parity_") as temp:
        library = Path(temp) / "libvehicle_plant.so"
        _compile(source_root, library)
        native = ctypes.CDLL(str(library))
        native.vehicle_plant_step.argtypes = [
            ctypes.POINTER(CState), ctypes.POINTER(CInput), ctypes.c_float,
            ctypes.POINTER(CParameters), ctypes.POINTER(CState)]
        native.vehicle_plant_step.restype = None
        c_parameters = _parameters(parameters)
        fixture_reports = []
        maximum = 0.0
        for initial, steering, throttle, dt in fixtures:
            expected = step(initial, steering, throttle, dt, parameters)
            actual = CState()
            native.vehicle_plant_step(
                ctypes.byref(_state(initial)),
                ctypes.byref(CInput(steering, throttle)),
                dt, ctypes.byref(c_parameters), ctypes.byref(actual))
            actual_array = _state_array(actual)
            error = np.abs(expected - actual_array)
            maximum = max(maximum, float(np.max(error)))
            fixture_reports.append({
                "dt_s": dt,
                "max_abs_error": float(np.max(error)),
                "errors": [float(value) for value in error],
            })
    report = {
        "schema_version": 1,
        "status": "pass" if maximum <= 3.0e-5 else "fail",
        "ground_truth_use": "none",
        "python_reference": "tools/model_id/structured_vehicle_plant.py",
        "native_source": "f1tenth_mpc/src/vehicle_plant.c",
        "fixture_count": len(fixtures),
        "max_abs_error": maximum,
        "tolerance": 3.0e-5,
        "fixtures": fixture_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--longitudinal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(
        args.source_root, args.lateral_report,
        args.longitudinal_report, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
