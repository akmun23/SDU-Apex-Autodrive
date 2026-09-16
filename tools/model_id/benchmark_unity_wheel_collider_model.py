#!/usr/bin/env python3
"""Benchmark the direct Unity WheelCollider lateral-law replay.

This is an offline diagnostic.  It uses the current candidate only for the
causal steering and longitudinal state paths, while replacing the fitted
axle-level tire peaks with the WheelCollider curve and per-wheel geometry
serialized by the Unity F1TENTH model.  It never edits or starts Unity and it
never changes the production MPC.

The result is intentionally a comparison artifact: a failure means that the
current state/normal-load abstraction is insufficient, not that a free tire
coefficient should be added back as compensation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_speed_regime_vehicle_model import (  # noqa: E402
    _load_runs,
    _raceline_context,
    _score,
)
from structured_vehicle_plant import (  # noqa: E402
    PlantParameters,
    UNITY_ACKERMANN_WHEELBASE_M,
    UNITY_COM_HEIGHT_M,
    UNITY_FRONT_WHEEL_X_M,
    UNITY_FORWARD_ASYMPTOTE_SLIP,
    UNITY_FORWARD_ASYMPTOTE_VALUE,
    UNITY_FORWARD_EXTREMUM_SLIP,
    UNITY_FORWARD_EXTREMUM_VALUE,
    UNITY_FORWARD_STIFFNESS,
    UNITY_SIDEWAYS_ASYMPTOTE_SLIP,
    UNITY_SIDEWAYS_ASYMPTOTE_VALUE,
    UNITY_SIDEWAYS_EXTREMUM_SLIP,
    UNITY_SIDEWAYS_EXTREMUM_VALUE,
    UNITY_SIDEWAYS_STIFFNESS,
    UNITY_TRACK_WIDTH_M,
    UNITY_REAR_WHEEL_X_M,
    UNITY_WHEEL_SPRUNG_MASSES_KG,
    step,
)


def _parameters(candidate: dict[str, Any]) -> PlantParameters:
    base = PlantParameters.from_manifest()
    values = candidate["candidate_parameters"]
    # Keep the measured causal actuator/longitudinal profile so this benchmark
    # isolates the lateral law.  The fitted lateral regime values are not
    # copied into the direct model.
    return replace(
        base,
        steering_dynamics_kind=str(values["steering_dynamics_kind"]),
        steering_lag_time_constant_s=float(
            values.get("steering_lag_time_constant_s", 0.0)),
        force_max_regimes_n=tuple(
            float(value) for value in values["force_max_regimes_n"]),
        slip_gain_regimes_per_mps=tuple(
            float(value) for value in values["slip_gain_regimes_per_mps"]),
        coast_speed_drag_n_per_mps=0.0,
        tire_model="unity_wheel_collider",
        unity_normal_load_mode="static_sprung_mass",
    )


def _parameters_with_load_screen(parameters: PlantParameters,
                                  normal_load_mode: str,
                                  sideways_slip_scale: float,
                                  longitudinal_model: str) -> PlantParameters:
    if longitudinal_model not in {"identified_force", "unity_forward_curve"}:
        raise ValueError(f"unsupported longitudinal model: {longitudinal_model}")
    parameters = replace(parameters,
                          unity_sideways_slip_scale=sideways_slip_scale,
                          unity_longitudinal_model=longitudinal_model)
    if normal_load_mode == "static_sprung_mass":
        return parameters
    if normal_load_mode == "mechanical_cg_transfer":
        return replace(parameters, unity_normal_load_mode=normal_load_mode)
    if normal_load_mode != "causal_linear_transfer":
        raise ValueError(f"unsupported normal-load mode: {normal_load_mode}")
    # Coefficients from the offline WheelHit contact-load screen.  They are
    # exposed as an explicit load-distribution mechanism rather than hidden in
    # the sideways curve.  Fresh exact-scene confirmation is still required.
    return replace(
        parameters,
        unity_normal_load_mode=normal_load_mode,
        unity_front_rear_load_transfer_bias_n=-6.383503267500638,
        unity_front_rear_load_transfer_gain_n_per_mps2=-1.4432096306172633,
        unity_left_right_load_transfer_bias_n=0.006084924592375423,
        unity_left_right_load_transfer_gain_n_per_mps2=2.360143182985202,
    )


def benchmark(candidate_path: Path, output: Path,
               origin_stride: int = 1,
               normal_load_mode: str = "static_sprung_mass",
               sideways_slip_scale: float = 1.0,
               longitudinal_model: str = "identified_force") -> dict[str, Any]:
    if origin_stride < 1:
        raise ValueError("origin_stride must be positive")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    runs = _load_runs(candidate["validation_specs"])
    context = _raceline_context(
        Path(candidate["operating_envelope"]["raceline_file"]))
    parameters = _parameters_with_load_screen(
        _parameters(candidate), normal_load_mode, sideways_slip_scale,
        longitudinal_model)
    scores = _score(runs, parameters, origin_stride, context)
    result = {
        "schema_version": 1,
        "status": "offline_direct_unity_wheel_collider_benchmark",
        "candidate": str(candidate_path),
        "simulator_modified": False,
        "production_mpc_modified": False,
        "ground_truth_use": "offline_endpoint_scoring_only",
        "normal_load_model": {
            "kind": normal_load_mode,
            "source": "exact competition-scene WheelCollider runtime diagnostic capture 20260916",
            "sprung_masses_kg": list(UNITY_WHEEL_SPRUNG_MASSES_KG),
            "sprung_mass_is_serialized_in_prefab": False,
            "requires_exact_competition_scene_refresh": False,
            "exact_scene_capture": "model_fits_v2/unity_exact_competition_static_capture_20260916",
            "dynamic_load_transfer_included": (
                normal_load_mode != "static_sprung_mass"),
            "coefficients_are_runtime_accepted": False,
        },
        "slip_coordinate_model": {
            "state_to_unity_sideways_slip_scale": sideways_slip_scale,
            "source": "offline WheelHit trace mapping diagnostic",
            "runtime_accepted": False,
        },
        "longitudinal_model": {
            "kind": longitudinal_model,
            "runtime_accepted": False,
            "wheel_state_is_still_identified_candidate": True,
        },
        "unity_source_anchors": {
            "wheelbase_for_ackermann_m": UNITY_ACKERMANN_WHEELBASE_M,
            "track_width_m": UNITY_TRACK_WIDTH_M,
            "com_height_m": UNITY_COM_HEIGHT_M,
            "wheel_locations_in_plant_frame_m": {
                "front_x": UNITY_FRONT_WHEEL_X_M,
                "rear_x": UNITY_REAR_WHEEL_X_M,
                "left_y": UNITY_TRACK_WIDTH_M * 0.5,
                "right_y": -UNITY_TRACK_WIDTH_M * 0.5,
            },
            "active_competition_scene_controller": {
                "drive_type": "CAWD",
                "drive_type_enum": 5,
                "steer_type": "FrontWheelSteer",
                "driving_mode": 1,
                "motor_torque_nm": 428.0,
                "steering_rate_degps": 183.346,
                "steering_limit_deg": 30.0,
            },
            "sideways_curve": {
                "extremum_slip": UNITY_SIDEWAYS_EXTREMUM_SLIP,
                "extremum_value": UNITY_SIDEWAYS_EXTREMUM_VALUE,
                "asymptote_slip": UNITY_SIDEWAYS_ASYMPTOTE_SLIP,
                "asymptote_value": UNITY_SIDEWAYS_ASYMPTOTE_VALUE,
                "stiffness": UNITY_SIDEWAYS_STIFFNESS,
            },
            "forward_curve_reference": {
                "extremum_slip": UNITY_FORWARD_EXTREMUM_SLIP,
                "extremum_value": UNITY_FORWARD_EXTREMUM_VALUE,
                "asymptote_slip": UNITY_FORWARD_ASYMPTOTE_SLIP,
                "asymptote_value": UNITY_FORWARD_ASYMPTOTE_VALUE,
                "stiffness": UNITY_FORWARD_STIFFNESS,
            },
            "source_files": [
                "/home/akselmo/Documents/GitHub/AutoDRIVE/Assets/Prefabs/F1TENTH/F1TENTH.prefab",
                "/home/akselmo/Documents/GitHub/AutoDRIVE/Assets/Scripts/VehicleController.cs",
            ],
        },
        "model_contract": {
            "tire_law": "piecewise_unity_wheel_friction_curve",
            "wheel_count": 4,
            "front_steering": "VehicleController Ackermann equations",
            "free_lateral_peak_or_cornering_stiffness": False,
            "horizon_s": 0.75,
            "horizon_commands": 30,
            "command_dt_s": 0.025,
        },
        "score_origin_stride": origin_stride,
        "scores": scores,
        "interpretation": {
            "purpose": "separate direct Unity law error from hidden fitted tire capacity",
            "not_for_runtime": True,
            "next_required_identification": [
                "fresh wheel-contact traces with the exact active competition scene",
                "explicit causal per-wheel normal-load state or validated load proxy",
                "forward WheelCollider torque/slip dynamics parity",
                "blind 0.75-second replay before any MPC migration",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument(
        "--normal-load-mode",
        choices=("static_sprung_mass", "causal_linear_transfer",
                 "mechanical_cg_transfer"),
        default="static_sprung_mass")
    parser.add_argument("--sideways-slip-scale", type=float, default=1.0)
    parser.add_argument(
        "--longitudinal-model",
        choices=("identified_force", "unity_forward_curve"),
        default="identified_force")
    args = parser.parse_args()
    result = benchmark(args.candidate, args.output, args.origin_stride,
                       args.normal_load_mode, args.sideways_slip_scale,
                       args.longitudinal_model)
    print(json.dumps({
        "output": str(args.output),
        "origin_stride": result["score_origin_stride"],
        "score_0.75s": result["scores"]["0.75s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
