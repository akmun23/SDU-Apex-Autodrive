#!/usr/bin/env python3
"""Check a GT-fitted yaw residual against the production C plant step."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

import yaml

from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.racing.score_yaw_residual_rollouts import ModelControl, ModelState, StageResult


ROOT = Path(__file__).resolve().parents[4]
DT_S = 0.025


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_candidate(base_config: Path, overlay_path: Path,
                       candidate: dict[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    params = config["/**"]["ros__parameters"]
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    overlay_params = overlay["/**"]["ros__parameters"]
    gate = candidate.get("support_gate")
    if gate is None:
        raise ValueError("candidate must have a declared support gate")
    expected = {
        "yaw_rate_residual_enabled": True,
        "yaw_rate_residual_gain": float(candidate["gain"]),
        "yaw_rate_residual_clip_radps2": float(candidate["correction_clip_radps2"]),
        "yaw_rate_residual_feature_mean": candidate["feature_mean"],
        "yaw_rate_residual_feature_scale": candidate["feature_scale"],
        "yaw_rate_residual_coefficients": candidate["coefficients_with_intercept"],
    }
    for source, suffix in (
        ("target_speed_zero_mps", "target_speed_zero_mps"),
        ("target_speed_full_mps", "target_speed_full_mps"),
        ("speed_deficit_full_mps", "speed_deficit_full_mps"),
        ("speed_deficit_zero_mps", "speed_deficit_zero_mps"),
        ("abs_steering_zero_rad", "abs_steering_zero_rad"),
        ("abs_steering_full_rad", "abs_steering_full_rad"),
        ("actual_speed_zero_mps", "actual_speed_zero_mps"),
        ("actual_speed_full_mps", "actual_speed_full_mps"),
        ("actual_speed_upper_full_mps", "actual_speed_upper_full_mps"),
        ("actual_speed_upper_zero_mps", "actual_speed_upper_zero_mps"),
        ("abs_steering_upper_full_rad", "abs_steering_upper_full_rad"),
        ("abs_steering_upper_zero_rad", "abs_steering_upper_zero_rad"),
    ):
        if source in gate:
            expected[f"yaw_rate_residual_{suffix}"] = float(gate[source])
    for name, expected_value in expected.items():
        actual = overlay_params.get(name)
        if isinstance(expected_value, list):
            if actual is None or len(actual) != len(expected_value) or any(
                not math.isclose(float(a), float(e), rel_tol=0.0, abs_tol=1e-12)
                for a, e in zip(actual, expected_value)
            ):
                raise ValueError(f"candidate overlay mismatch for {name}")
        elif isinstance(expected_value, bool):
            if actual is not expected_value:
                raise ValueError(f"candidate overlay mismatch for {name}")
        elif actual is None or not math.isclose(
                float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"candidate overlay mismatch for {name}")
    params.update(overlay_params)
    surface = Path(params.get("yaw_rate_response_surface_file", ""))
    if str(surface).startswith("/workspace/src/"):
        params["yaw_rate_response_surface_file"] = str(
            ROOT / str(surface).removeprefix("/workspace/src/"))
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-overlay", type=Path, required=True)
    parser.add_argument("--validation-samples", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    candidate_overlay = args.candidate_overlay.resolve()
    samples_path = args.validation_samples.resolve()
    base_config = args.base_config.resolve()
    library_path = args.library.resolve()
    trajectory_path = args.trajectory.resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    expected_config_hash = candidate.get("baseline_config_sha256")
    if expected_config_hash != sha256_file(base_config):
        raise ValueError("base MPC config hash differs from the fit provenance")

    library = ctypes.CDLL(str(library_path))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl), ctypes.c_float,
        ctypes.c_float,
    ]
    library.mpc_vehicle_model_step.restype = StageResult

    errors: list[float] = []
    worst_probe: dict[str, Any] | None = None
    invalid = 0
    config = configure_candidate(base_config, candidate_overlay, candidate)
    with tempfile.TemporaryDirectory(prefix="yaw-residual-c-parity-") as temp_dir:
        config_path = Path(temp_dir) / "mpc_candidate.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        with ProductionMpc(library_path, config_path, trajectory_path):
            with samples_path.open(newline="", encoding="utf-8") as stream:
                for probe_index, row in enumerate(csv.DictReader(stream)):
                    state = ModelState(
                        float(row["state_e_y_m"]),
                        float(row["state_e_psi_rad"]),
                        float(row["state_u_mps"]),
                        float(row["state_v_mps"]),
                        float(row["state_r_radps"]),
                        float(row["state_target_speed_mps"]),
                        float(row["state_steering_command_rad"]),
                        float(row["state_delayed_steering_command_1_rad"]),
                        float(row["state_delayed_steering_command_2_rad"]),
                        float(row["state_actual_steering_angle_rad"]),
                    )
                    control = ModelControl(
                        float(row["control_steering_rate_radps"]),
                        float(row["control_target_speed_rate_mps2"]),
                    )
                    stage = library.mpc_vehicle_model_step(
                        ctypes.byref(state), ctypes.byref(control),
                        ctypes.c_float(DT_S),
                        ctypes.c_float(float(row["path_curvature_inv_m"])),
                    )
                    if not stage.valid:
                        invalid += 1
                        continue
                    expected = (
                        float(row["baseline_prediction_radps"])
                        + DT_S * float(row["candidate_residual_rate_radps2"])
                    )
                    error = float(stage.next.r) - expected
                    errors.append(error)
                    if worst_probe is None or abs(error) > worst_probe["abs_error_radps"]:
                        worst_probe = {
                            "probe_index": probe_index,
                            "run_id": row["run_id"],
                            "time_s": float(row["time_s"]),
                            "speed_mps": float(row["speed_mps"]),
                            "abs_steering_rad": float(row["steering_abs_rad"]),
                            "support_weight": float(row["support_weight"]),
                            "baseline_yaw_radps": float(row["baseline_prediction_radps"]),
                            "candidate_residual_rate_radps2": float(
                                row["candidate_residual_rate_radps2"]),
                            "expected_yaw_radps": expected,
                            "production_c_yaw_radps": float(stage.next.r),
                            "abs_error_radps": abs(error),
                        }

    if not errors:
        raise ValueError("no valid C parity probes")
    absolute = [abs(value) for value in errors]
    report = {
        "candidate_sha256": sha256_file(candidate_path),
        "validation_samples_sha256": sha256_file(samples_path),
        "base_config_sha256": expected_config_hash,
        "library": str(library_path),
        "probe_count": len(errors),
        "invalid_c_steps": invalid,
        "max_abs_yaw_difference_radps": max(absolute),
        "rmse_yaw_difference_radps": math.sqrt(
            sum(value * value for value in errors) / len(errors)),
        "p99_abs_yaw_difference_radps": statistics.quantiles(
            absolute, n=100, method="inclusive")[98] if len(absolute) > 1 else absolute[0],
        "worst_probe": worst_probe,
        "pass": max(absolute) <= 2.0e-4 and invalid == 0,
        "comparison": "production C next yaw vs GT-fit candidate baseline plus saved gated correction",
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
