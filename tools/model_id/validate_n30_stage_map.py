#!/usr/bin/env python3
"""Validate the offline N30 stage map's numerical and native parity gates.

The nonlinear Python map is the canonical offline reference.  The existing
native ``vehicle_plant`` is compared against that same scalar 2 ms reference
plant; it is not claimed to implement the not-yet-promoted fitted profile.
The report also checks the production-relevant two-substep map for finite and
bounded states and checks its finite-difference Jacobian.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structured_vehicle_plant as plant  # noqa: E402
from n30_stage_map import finite_difference_jacobian, f25  # noqa: E402


class CState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "x_m", "y_m", "yaw_rad", "u_mps", "v_mps", "r_radps",
        "steering_rad", "wheel_speed_mps")]


class CInput(ctypes.Structure):
    _fields_ = [("steering_target_norm", ctypes.c_float),
                ("throttle_norm", ctypes.c_float)]


class CParameters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "mass_kg", "lf_m", "lr_m", "iz_kgm2",
        "position_offset_from_velocity_point_x_m", "max_steering_rad",
        "steering_rate_radps", "max_speed_mps", "linear_damping_per_s",
        "angular_damping_per_s", "force_max_n", "hard_brake_force_n",
        "slip_gain_per_mps", "coast_speed_drag_n_per_mps", "cf_n_per_rad",
        "cr_n_per_rad", "df_n", "dr_n")]
    _fields_ += [("tire_model", ctypes.c_uint8),
                 ("wheel_dynamics_model", ctypes.c_uint8),
                 ("wheel_coefficients", ctypes.c_float * 6)]


def _c_parameters(parameters: plant.PlantParameters) -> CParameters:
    result = CParameters()
    for name in (
            "mass_kg", "lf_m", "lr_m", "iz_kgm2",
            "position_offset_from_velocity_point_x_m", "max_steering_rad",
            "steering_rate_radps", "max_speed_mps", "linear_damping_per_s",
            "angular_damping_per_s", "force_max_n", "hard_brake_force_n",
            "slip_gain_per_mps", "coast_speed_drag_n_per_mps",
            "cf_n_per_rad", "cr_n_per_rad", "df_n", "dr_n"):
        setattr(result, name, getattr(parameters, name))
    result.tire_model = 0 if parameters.tire_model == "linear_saturated" else 1
    result.wheel_dynamics_model = (
        1 if parameters.wheel_dynamics_kind == "continuous" else 0)
    result.wheel_coefficients = (ctypes.c_float * 6)(
        *parameters.wheel_coefficients)
    return result


def _c_state(values: np.ndarray) -> CState:
    return CState(*(float(value) for value in values))


def _array(state: CState) -> np.ndarray:
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


def validate(source_root: Path, output: Path) -> dict[str, object]:
    parameters = plant.PlantParameters()
    fixtures = [
        (np.asarray([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]), 0.0, 0.2),
        (np.asarray([1.0, -0.4, 0.2, 4.0, 0.15, 0.3, 0.05, 4.5]), 0.5, 0.6),
        (np.asarray([-2.0, 3.0, -1.2, 8.0, -0.2, -0.7, -0.2, 7.5]), -0.8, 0.0),
        (np.asarray([4.0, 1.0, 2.7, 12.0, 0.3, 1.1, 0.35, 12.2]), 0.9, 0.8),
    ]
    finite_jacobian = finite_difference_jacobian(
        fixtures[1][0], fixtures[1][1], fixtures[1][2], parameters)
    bounded_failures = 0
    for state, steering, throttle in fixtures:
        next_state = f25(state, steering, throttle, parameters)
        if (not np.all(np.isfinite(next_state)) or next_state[3] < 0.0 or
                next_state[3] > parameters.max_speed_mps or
                abs(next_state[4]) > 50.0 or abs(next_state[5]) > 100.0):
            bounded_failures += 1

    parity_errors: list[float] = []
    with tempfile.TemporaryDirectory(prefix="n30_stage_map_") as temp:
        library = Path(temp) / "libvehicle_plant.so"
        _compile(source_root, library)
        native = ctypes.CDLL(str(library))
        native.vehicle_plant_step.argtypes = [
            ctypes.POINTER(CState), ctypes.POINTER(CInput), ctypes.c_float,
            ctypes.POINTER(CParameters), ctypes.POINTER(CState)]
        native.vehicle_plant_step.restype = None
        c_parameters = _c_parameters(parameters)
        for state, steering, throttle in fixtures:
            expected = f25(state, steering, throttle, parameters,
                           internal_substep_s=0.002)
            actual = CState()
            native.vehicle_plant_step(
                ctypes.byref(_c_state(state)),
                ctypes.byref(CInput(steering, throttle)), ctypes.c_float(0.025),
                ctypes.byref(c_parameters), ctypes.byref(actual))
            parity_errors.append(float(np.max(np.abs(expected - _array(actual)))))

    result = {
        "schema_version": 1,
        "status": ("pass" if finite_jacobian["finite"] and
                    bounded_failures == 0 and max(parity_errors, default=float("inf"))
                    <= 3.0e-5 else "fail"),
        "simulator_modified": False,
        "production_mpc_modified": False,
        "stage_contract": {
            "commands": 30,
            "command_dt_s": 0.025,
            "physical_horizon_s": 0.75,
            "selected_internal_substep_s": 0.0125,
            "native_parity_reference_substep_s": 0.002,
        },
        "finite_bounded_gate": {
            "fixture_count": len(fixtures),
            "bounded_failures": bounded_failures,
        },
        "jacobian_gate": {
            "finite": finite_jacobian["finite"],
            "max_abs_jacobian": finite_jacobian["max_abs_jacobian"],
            "state_step": 1.0e-5,
            "input_step": 1.0e-5,
        },
        "python_c_scalar_reference_parity": {
            "fixture_count": len(fixtures),
            "max_abs_error": max(parity_errors, default=float("inf")),
            "tolerance": 3.0e-5,
            "reference": "scalar baseline at 2 ms internal integration",
            "fitted_profile_supported": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.source_root, args.output), indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
