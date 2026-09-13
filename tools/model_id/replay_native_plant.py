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
           output: Path) -> dict[str, object]:
    lateral = json.loads(lateral_report.read_text(encoding="utf-8"))
    longitudinal = json.loads(longitudinal_report.read_text(encoding="utf-8"))
    lateral_parameters = lateral["candidate_comparison"]["Y1_linear_saturated"][
        "parameters"]["parameters"]
    longitudinal_parameters = longitudinal["models"]["wheel_dynamic"]["parameters"]
    plant_parameters = PlantParameters.from_longitudinal_parameters(
        longitudinal_parameters).with_lateral(lateral_parameters)
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
            "candidate": "Y1_linear_saturated",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(
        args.source_root, args.accepted_root,
        [value.strip() for value in args.run_names.split(",") if value.strip()],
        args.lateral_report, args.longitudinal_report, args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "recursive_0.50s": report["recursive_scores"]["0.50s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
