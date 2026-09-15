#!/usr/bin/env python3
"""Run the native structured plant over complete V4 validation runs."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_native_plant_parity import (  # noqa: E402
    CInput,
    CParameters,
    CState,
    _compile,
    _parameters,
    _state,
    _state_array,
)
from fit_lateral_model import (  # noqa: E402
    HORIZONS_S,
    _error_vector,
    _horizon_key,
    _read_runs,
)
from structured_vehicle_plant import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    MAX_STEERING_RAD,
    PlantParameters,
)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mae": None, "median": None, "p90": None,
                "p95": None, "p99": None, "max": None, "bias": None,
                "rmse": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": int(data.size),
        "mae": float(np.mean(np.abs(data))),
        "median": float(np.median(np.abs(data))),
        "p90": float(np.percentile(np.abs(data), 90)),
        "p95": float(np.percentile(np.abs(data), 95)),
        "p99": float(np.percentile(np.abs(data), 99)),
        "max": float(np.max(np.abs(data))),
        "bias": float(np.mean(data)),
        "rmse": float(np.sqrt(np.mean(data * data))),
    }


def _native_function(library: ctypes.CDLL):
    function = library.vehicle_plant_step
    function.argtypes = [
        ctypes.POINTER(CState), ctypes.POINTER(CInput), ctypes.c_float,
        ctypes.POINTER(CParameters), ctypes.POINTER(CState)]
    function.restype = None
    return function


def _score(runs: dict[str, list[dict[str, float]]], function: Any,
           parameters: Any) -> dict[str, Any]:
    fields = ("x_m", "y_m", "position_m", "heading_rad", "u_mps", "v_mps",
              "r_radps", "steering_rad", "wheel_mps")
    output: dict[str, Any] = {}
    for horizon in HORIZONS_S:
        errors = {field: [] for field in fields}
        for rows in runs.values():
            for origin, first in enumerate(rows):
                if not np.isfinite(first["wheel_k_mps"]) or not np.isfinite(first["delta_k_rad"]):
                    continue
                state = np.asarray([
                    first["x_k_m"], first["y_k_m"], first["yaw_k_rad"],
                    first["u_k_mps"], first["v_k_mps"], first["r_k_radps"],
                    first["delta_k_rad"], first["wheel_k_mps"],
                ], dtype=float)
                segment = int(first["segment_id"])
                elapsed = 0.0
                index = origin
                while index < len(rows) and elapsed < horizon - 1.0e-10:
                    row = rows[index]
                    if int(row["segment_id"]) != segment:
                        break
                    native_next = CState()
                    function(
                        ctypes.byref(_state(state)),
                        ctypes.byref(CInput(
                            row["applied_steering_rad_k1"] / MAX_STEERING_RAD,
                            row["applied_throttle_norm_k1"])),
                        row["dt_sim_s"], ctypes.byref(parameters),
                        ctypes.byref(native_next))
                    state = _state_array(native_next)
                    elapsed += row["dt_sim_s"]
                    index += 1
                if index == origin or elapsed < horizon - 1.0e-10:
                    continue
                for field, value in _error_vector(state, rows[index - 1]).items():
                    errors[field].append(value)
        output[_horizon_key(horizon)] = {
            field: _stats(values) for field, values in errors.items()
        }
    return output


def replay(source_root: Path, accepted_root: Path, run_names: list[str],
           lateral_report: Path, longitudinal_report: Path,
           output: Path, longitudinal_model: str = "wheel_continuous",
           lateral_model: str = "Y2_tanh_fixed_iz",
           damping_profile: str = "unity_measured",
           use_manifest: bool = False,
           position_offset_from_velocity_point_x_m: float | None = None,
           coast_speed_drag_n_per_mps: float | None = None,
           linear_damping_per_s: float | None = None,
           angular_damping_per_s: float | None = None) -> dict[str, object]:
    lateral = json.loads(lateral_report.read_text(encoding="utf-8"))
    longitudinal = json.loads(longitudinal_report.read_text(encoding="utf-8"))
    lateral_candidates = lateral.get("candidate_comparison", {})
    if lateral_model not in lateral_candidates:
        raise ValueError(
            f"lateral report has no candidate named {lateral_model!r}; "
            f"available={sorted(lateral_candidates)}")
    lateral_parameters = lateral_candidates[lateral_model]["parameters"][
        "parameters"]
    if longitudinal_model not in longitudinal["models"]:
        raise ValueError(
            f"longitudinal report has no model named {longitudinal_model!r}")
    longitudinal_parameters = longitudinal["models"][longitudinal_model][
        "parameters"]
    if use_manifest:
        # Manifest resolution is explicit.  This prevents an offline A/B
        # replay from silently evaluating a different parameter set merely
        # because it uses the canonical model names.
        plant_parameters = PlantParameters.from_manifest()
    else:
        # Report-driven replay is the default for offline comparison: the
        # selected report files are the parameter authority for this run.
        plant_parameters = PlantParameters.from_longitudinal_parameters(
            longitudinal_parameters).with_lateral(lateral_parameters)
    overrides: dict[str, float] = {}
    if damping_profile == "unity_measured":
        overrides.update({
            "coast_speed_drag_n_per_mps": 0.0,
            "linear_damping_per_s": 0.273,
            "angular_damping_per_s": 0.1,
        })
    elif damping_profile == "fitted":
        overrides.update({
            "linear_damping_per_s": 0.0,
            "angular_damping_per_s": 0.0,
        })
    else:
        raise ValueError(
            f"unsupported damping profile {damping_profile!r}; "
            "expected unity_measured or fitted")
    if position_offset_from_velocity_point_x_m is not None:
        overrides["position_offset_from_velocity_point_x_m"] = float(
            position_offset_from_velocity_point_x_m)
    if coast_speed_drag_n_per_mps is not None:
        overrides["coast_speed_drag_n_per_mps"] = float(
            coast_speed_drag_n_per_mps)
    if linear_damping_per_s is not None:
        overrides["linear_damping_per_s"] = float(linear_damping_per_s)
    if angular_damping_per_s is not None:
        overrides["angular_damping_per_s"] = float(angular_damping_per_s)
    if overrides:
        plant_parameters = PlantParameters(
            **{**plant_parameters.__dict__, **overrides})
    runs = _read_runs(accepted_root, run_names)
    with tempfile.TemporaryDirectory(prefix="vehicle_plant_replay_") as temp:
        library_path = Path(temp) / "libvehicle_plant.so"
        _compile(source_root, library_path)
        library = ctypes.CDLL(str(library_path))
        function = _native_function(library)
        native_parameters = _parameters(plant_parameters)
        scores = _score(runs, function, native_parameters)
    report = {
        "schema_version": 1,
        "status": "native_candidate_replay_not_runtime_validated",
        "ground_truth_use": "offline_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "run_names": run_names,
        "state_definition": ["x_m", "y_m", "yaw_rad", "u_mps", "v_mps",
                              "r_radps", "steering_rad", "wheel_speed_mps"],
        "input_definition": ["steering_target_norm", "throttle_norm"],
        "plant_parameters": {
            "source_lateral_report": str(lateral_report),
            "source_longitudinal_report": str(longitudinal_report),
            "parameter_authority": str(DEFAULT_MANIFEST_PATH)
            if use_manifest else "source_reports",
            "candidate": lateral_model,
            "longitudinal_model": longitudinal_model,
            "damping_profile": damping_profile,
            "resolved": {
                key: value for key, value in plant_parameters.__dict__.items()
                if key != "wheel_coefficients"
            },
            "wheel_coefficients": list(plant_parameters.wheel_coefficients),
            "damping_overrides": overrides,
            "manifest_requested": use_manifest,
        },
        "native_source": "f1tenth_mpc/src/vehicle_plant.c",
        "recursive_scores": scores,
        "acceptance": {
            "native_replay": True,
            "production_mpc_parameters_updated": False,
            "blind_run_required": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--run-names", required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--longitudinal-report", type=Path, required=True)
    parser.add_argument(
        "--longitudinal-model", default="wheel_continuous",
        help="model key from the longitudinal benchmark (default: wheel_continuous)")
    parser.add_argument(
        "--lateral-model", default="Y2_tanh_fixed_iz",
        help="candidate key from the lateral benchmark (default: Y2_tanh_fixed_iz)")
    parser.add_argument(
        "--damping-profile", choices=("unity_measured", "fitted"),
        default="unity_measured",
        help="resolved offline damping profile (default: unity_measured)")
    parser.add_argument(
        "--use-manifest", action="store_true",
        help="explicitly resolve the canonical candidate from the manifest")
    parser.add_argument(
        "--position-offset-from-velocity-point-x-m", type=float, default=None,
        help="offline A/B override for the recorded pose-point lever arm [m]")
    parser.add_argument(
        "--coast-speed-drag-n-per-mps", type=float, default=None,
        help="offline A/B override for the fitted longitudinal drag force term")
    parser.add_argument(
        "--linear-damping-per-s", type=float, default=None,
        help="offline A/B override for explicit body linear damping [1/s]")
    parser.add_argument(
        "--angular-damping-per-s", type=float, default=None,
        help="offline A/B override for explicit yaw damping [1/s]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(
        args.source_root, args.accepted_root,
        [value.strip() for value in args.run_names.split(",") if value.strip()],
        args.lateral_report, args.longitudinal_report, args.output,
        args.longitudinal_model, args.lateral_model, args.damping_profile,
        args.use_manifest,
        args.position_offset_from_velocity_point_x_m,
        args.coast_speed_drag_n_per_mps,
        args.linear_damping_per_s, args.angular_damping_per_s)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "recursive_0.50s": report["recursive_scores"]["0.50s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
