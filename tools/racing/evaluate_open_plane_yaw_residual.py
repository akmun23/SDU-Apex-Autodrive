#!/usr/bin/env python3
"""Held-out open-plane replay for a P0-fitted yaw residual candidate.

The candidate starts from simulator truth at each window, then consumes only
the recorded steering-command sequence. Future simulator state is used solely
as a scoring label. This is development analysis, not runtime MPC code.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.racing.score_yaw_residual_rollouts import (
    DT_S,
    ModelControl,
    ModelState,
    StageResult,
    apply_yaw_correction,
    clone_state,
    metric,
    residual_rate,
    wrap_angle,
)


ROOT = Path(__file__).resolve().parents[2]
COM_X_M = 0.15532
STEERING_LIMIT_RAD = 0.5236
PROFILE_SPECS = {
    "subnet_highsteer_transients": {
        "phase_prefix": "subnet_c",
        "expected_phase_count": 20,
        "phase_duration_s": 2.0,
        "target_speed_mps": 7.5,
    },
    "isolated_highsteer_75_long": {
        "phase_prefix": "r1_",
        "expected_phase_count": 17,
        "phase_duration_s": 7.5,
        "target_speed_mps": 7.5,
    },
    "isolated_highsteer_multispeed": {
        "phase_prefix": "multispeed_v",
        "expected_phase_count": 51,
        "phase_duration_s": 7.5,
        "target_speed_mps": None,
    },
}
HORIZONS = {0.025: 1, 0.1: 4, 0.25: 10, 0.5: 20, 0.75: 30}
CHANNELS = ("u_mps", "v_rear_mps", "r_radps", "position_m", "heading_rad")


def finite_vector(values: Any, length: int) -> bool:
    return isinstance(values, np.ndarray) and values.shape == (length,) and np.isfinite(values).all()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth_state(sample: body.MotionSample) -> tuple[np.ndarray, np.ndarray] | None:
    rigid = sample.simulator_rigid_state
    pose = sample.simulator_pose_xyyaw
    if not finite_vector(rigid, 13) or not finite_vector(pose, 3):
        return None
    u_com, v_com = float(rigid[7]), float(rigid[8])
    yaw_rate = float(rigid[12])
    state = np.asarray((u_com, v_com - COM_X_M * yaw_rate, yaw_rate), dtype=np.float64)
    return state, np.asarray(pose, dtype=np.float64)


def command_angle(sample: body.MotionSample) -> float | None:
    if sample.actuator_history is None or len(sample.actuator_history) != 4:
        return None
    value = float(sample.actuator_history[2])
    return value if math.isfinite(value) else None


def heading_position(initial_pose: np.ndarray, state: ModelState,
                     progress: float) -> np.ndarray:
    yaw = float(initial_pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray((initial_pose[0] + progress * c - float(state.e_y) * s,
                       initial_pose[1] + progress * s + float(state.e_y) * c),
                      dtype=np.float64)


def signed_metric(values: list[float]) -> dict[str, float | int]:
    result = metric(values)
    result["bias"] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return result


def evaluate(bag_path: Path, model_path: Path, library_path: Path,
             config_path: Path, trajectory_path: Path, residual_gain: float,
             profile: str
             ) -> dict[str, Any]:
    capture = body.load_capture(bag_path)
    if (capture.aborted or capture.reason != "schedule complete"
            or capture.collision_count_end != capture.collision_count_start
            or capture.timing_faults != 0):
        raise ValueError("open-plane capture did not pass completion, collision, and timing gates")
    if capture.invalid_phase_count:
        raise ValueError("open-plane capture contains invalid phases")
    quality_stats = (capture.phase_stream_stats if capture.has_phase_markers
                     else capture.stream_stats)
    if not quality_stats or any(
        rate < 38.0 or p95_gap > 35.0 or max_gap > 120.0
        for rate, p95_gap, max_gap in quality_stats.values()
    ):
        raise ValueError("one or more active capture streams failed the 40 Hz quality gate")

    if profile not in PROFILE_SPECS:
        raise ValueError(f"unsupported open-plane profile: {profile}")
    profile_spec = PROFILE_SPECS[profile]
    phase_prefix = str(profile_spec["phase_prefix"])
    phase_duration_s = float(profile_spec["phase_duration_s"])
    fixed_target_speed = profile_spec["target_speed_mps"]
    phases: list[tuple[str, tuple[body.MotionSample, ...]]] = []
    for label, sequence in zip(capture.sequence_labels, capture.sequences):
        if label.startswith(phase_prefix):
            phases.append((label, sequence))
    expected_phase_count = int(profile_spec["expected_phase_count"])
    unique_phase_labels = {label for label, _ in phases}
    if len(unique_phase_labels) != expected_phase_count:
        raise ValueError(
            f"expected {expected_phase_count} unique {profile} phases, "
            f"found {len(unique_phase_labels)}")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("status") != "offline_one_step_candidate_only":
        raise ValueError("candidate file has an unexpected status")

    library = ctypes.CDLL(str(library_path.resolve()))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl), ctypes.c_float,
        ctypes.c_float,
    ]
    library.mpc_vehicle_model_step.restype = StageResult

    errors: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    high_steer_errors: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    per_phase_errors: dict[str, dict[tuple[float, str], list[float]]] = defaultdict(
        lambda: defaultdict(list))
    phase_input_summary: dict[str, dict[str, float]] = {}
    accepted_windows = 0
    skipped_windows = 0
    parity_max = 0.0

    with ProductionMpc(library_path, config_path, trajectory_path):
        phase_segment_counts: dict[str, int] = defaultdict(int)
        for label, _ in phases:
            phase_segment_counts[label] += 1
        for segment_index, (label, sequence) in enumerate(
                sorted(phases, key=lambda item: item[0]), start=1):
            samples = list(sequence)
            summary_key = (
                label if phase_segment_counts[label] == 1
                else f"{label}#segment_{segment_index}"
            )
            target_speed_mps = (
                float(label.split("_", 2)[1][1:])
                if fixed_target_speed is None
                else float(fixed_target_speed)
            )
            truth = [truth_state(sample) for sample in samples]
            commands = [command_angle(sample) for sample in samples]
            valid_truth = [item for item in truth if item is not None]
            valid_commands = [item for item in commands if item is not None]
            measured_steering = [float(sample.actuators[0]) for sample in samples
                                 if math.isfinite(float(sample.actuators[0]))]
            phase_input_summary[summary_key] = {
                "truth_speed_median_mps": float(np.median(
                    [float(item[0][0]) for item in valid_truth])),
                "truth_speed_p05_mps": float(np.quantile(
                    [float(item[0][0]) for item in valid_truth], 0.05)),
                "truth_speed_p95_mps": float(np.quantile(
                    [float(item[0][0]) for item in valid_truth], 0.95)),
                "measured_steering_median_rad": float(np.median(measured_steering)),
                "commanded_steering_median_rad": float(np.median(valid_commands)),
                "truth_abs_yaw_rate_p95_radps": float(np.quantile(
                    [abs(float(item[0][2])) for item in valid_truth], 0.95)),
            }
            valid_starts = [
                index for index, sample in enumerate(samples)
                if index >= 2
                and 0.0 <= sample.time_s <= phase_duration_s - max(HORIZONS)
                and index + max(HORIZONS.values()) < len(samples)
            ]
            for start in valid_starts:
                end = start + max(HORIZONS.values())
                if any(truth[index] is None or commands[index] is None
                       for index in range(start - 2, end + 1)):
                    skipped_windows += 1
                    continue
                if any(samples[index + 1].packet_sequence
                       != samples[index].packet_sequence + 1
                       for index in range(start - 2, end)):
                    skipped_windows += 1
                    continue

                initial_truth, initial_pose = truth[start]  # type: ignore[misc]
                steer = float(samples[start].actuators[0])
                if not math.isfinite(steer):
                    skipped_windows += 1
                    continue
                command_now = float(commands[start])
                initial = ModelState(
                    0.0, 0.0, float(initial_truth[0]), float(initial_truth[1]),
                    float(initial_truth[2]), target_speed_mps,
                    command_now, float(commands[start - 1]),
                    float(commands[start - 2]), steer,
                )
                baseline_state = clone_state(initial)
                candidate_state = clone_state(initial)
                base_progress = 0.0
                candidate_progress = 0.0
                staged: list[tuple[float, dict[str, float], dict[str, float]]] = []
                valid = True

                for step_index in range(max(HORIZONS.values())):
                    sample_index = start + step_index
                    q_delta = (float(commands[sample_index + 1])
                               - float(commands[sample_index])) / DT_S
                    control = ModelControl(q_delta, 0.0)
                    base_stage = library.mpc_vehicle_model_step(
                        ctypes.byref(baseline_state), ctypes.byref(control),
                        ctypes.c_float(DT_S), ctypes.c_float(0.0))
                    candidate_stage = library.mpc_vehicle_model_step(
                        ctypes.byref(candidate_state), ctypes.byref(control),
                        ctypes.c_float(DT_S), ctypes.c_float(0.0))
                    if not base_stage.valid or not candidate_stage.valid:
                        valid = False
                        break
                    if step_index == 0:
                        zero_next, zero_ds = apply_yaw_correction(
                            baseline_state, base_stage, control, DT_S, 0.0, 0.0)
                        parity = (
                            abs(float(zero_next.r) - float(base_stage.next.r)),
                            abs(float(zero_next.e_y) - float(base_stage.next.e_y)),
                            abs(wrap_angle(float(zero_next.e_psi)
                                           - float(base_stage.next.e_psi))),
                            abs(zero_ds - float(base_stage.delta_s_m)),
                        )
                        parity_max = max(parity_max, *parity)

                    correction = residual_gain * residual_rate(
                        model, candidate_state, control, 0.0,
                        float(candidate_stage.body_accel_mps2))
                    candidate_next, candidate_ds = apply_yaw_correction(
                        candidate_state, candidate_stage, control, DT_S, 0.0,
                        correction)
                    baseline_state = clone_state(base_stage.next)
                    candidate_state = candidate_next
                    base_progress += float(base_stage.delta_s_m)
                    candidate_progress += candidate_ds

                    horizon = next((seconds for seconds, count in HORIZONS.items()
                                    if step_index + 1 == count), None)
                    if horizon is None:
                        continue
                    truth_at_horizon = truth[sample_index + 1]
                    if truth_at_horizon is None:
                        valid = False
                        break
                    target_state, target_pose = truth_at_horizon
                    base_pose = heading_position(initial_pose, baseline_state,
                                                 base_progress)
                    candidate_pose = heading_position(
                        initial_pose, candidate_state, candidate_progress)
                    base_heading = wrap_angle(
                        float(initial_pose[2]) + float(baseline_state.e_psi)
                        - float(target_pose[2]))
                    candidate_heading = wrap_angle(
                        float(initial_pose[2]) + float(candidate_state.e_psi)
                        - float(target_pose[2]))
                    base_values = {
                        "u_mps": float(baseline_state.u) - float(target_state[0]),
                        "v_rear_mps": float(baseline_state.v) - float(target_state[1]),
                        "r_radps": float(baseline_state.r) - float(target_state[2]),
                        "position_m": float(np.linalg.norm(base_pose - target_pose[:2])),
                        "heading_rad": base_heading,
                    }
                    candidate_values = {
                        "u_mps": float(candidate_state.u) - float(target_state[0]),
                        "v_rear_mps": float(candidate_state.v) - float(target_state[1]),
                        "r_radps": float(candidate_state.r) - float(target_state[2]),
                        "position_m": float(np.linalg.norm(candidate_pose - target_pose[:2])),
                        "heading_rad": candidate_heading,
                    }
                    staged.append((horizon, base_values, candidate_values))

                if not valid or len(staged) != len(HORIZONS):
                    skipped_windows += 1
                    continue
                accepted_windows += 1
                # Classify the aggregate from the actual command samples,
                # not profile-specific label spelling (profiles may encode
                # their zero-angle phase differently).
                is_high_steer = (
                    float(np.median(np.abs(valid_commands))) >= 0.10)
                for horizon, base_values, candidate_values in staged:
                    for channel in CHANNELS:
                        baseline_error = base_values[channel]
                        candidate_error = candidate_values[channel]
                        key = (horizon, channel, "all")
                        errors[key].append(baseline_error)
                        errors[(horizon, channel, "candidate")].append(candidate_error)
                        if is_high_steer:
                            high_steer_errors[(horizon, channel, "production")].append(
                                baseline_error)
                            high_steer_errors[(horizon, channel, "candidate")].append(
                                candidate_error)
                        per_phase_errors[label][(horizon, channel)].append(
                            (baseline_error, candidate_error))

    by_horizon: dict[str, Any] = {}
    high_steer_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        for channel in CHANNELS:
            baseline = errors[(horizon, channel, "all")]
            candidate = errors[(horizon, channel, "candidate")]
            by_horizon[f"{horizon:g}s:{channel}"] = {
                "production": signed_metric(baseline),
                "candidate": signed_metric(candidate),
            }
            high_baseline = high_steer_errors[(horizon, channel, "production")]
            high_candidate = high_steer_errors[(horizon, channel, "candidate")]
            high_steer_by_horizon[f"{horizon:g}s:{channel}"] = {
                "production": signed_metric(high_baseline),
                "candidate": signed_metric(high_candidate),
            }

    per_phase: dict[str, Any] = {}
    for label, phase_values in per_phase_errors.items():
        per_phase[label] = {}
        for (horizon, channel), pairs in phase_values.items():
            per_phase[label][f"{horizon:g}s:{channel}"] = {
                "production": signed_metric([pair[0] for pair in pairs]),
                "candidate": signed_metric([pair[1] for pair in pairs]),
            }

    return {
        "schema_version": 1,
        "validation_profile": profile,
        "run_id": bag_path.parent.parent.name,
        "candidate_model_id": model.get("model_id"),
        "residual_gain": residual_gain,
        "support_gate": model.get("support_gate"),
        "inputs": {
            "bag": str(bag_path),
            "bag_sha256": sha256(bag_path),
            "candidate_model": str(model_path),
            "candidate_model_sha256": sha256(model_path),
            "production_library": str(library_path),
            "production_config_sha256": sha256(config_path),
            "trajectory_sha256": sha256(trajectory_path),
        },
        "run_quality": {
            "aborted": capture.aborted,
            "reason": capture.reason,
            "collisions_start_end": [capture.collision_count_start,
                                      capture.collision_count_end],
            "timing_faults": capture.timing_faults,
            "phase_count": capture.phase_count,
            "valid_phase_count": capture.valid_phase_count,
            "evaluated_phase_sequences": len(phases),
            "evaluated_unique_phases": len(unique_phase_labels),
            "stream_stats_hz_p95_ms_max_ms": {
                name: {"hz": values[0], "p95_gap_ms": values[1],
                       "max_gap_ms": values[2]}
                for name, values in capture.stream_stats.items()
            },
            "active_phase_stream_stats_hz_p95_ms_max_ms": {
                name: {"hz": values[0], "p95_gap_ms": values[1],
                       "max_gap_ms": values[2]}
                for name, values in capture.phase_stream_stats.items()
            },
        },
        "accepted_windows": accepted_windows,
        "skipped_windows": skipped_windows,
        "zero_correction_equation_parity_max_abs": parity_max,
        "future_truth_or_sensor_inputs": False,
        "scoring_horizons_s": list(HORIZONS),
        "whole_run_metrics": by_horizon,
        "high_steer_metrics_excluding_zero_steer_baseline": high_steer_by_horizon,
        "per_phase_metrics": per_phase,
        "per_phase_input_summary": phase_input_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-gain", type=float, default=0.15)
    parser.add_argument("--profile", choices=tuple(PROFILE_SPECS), required=True)
    args = parser.parse_args()
    if not 0.0 <= args.residual_gain <= 1.0:
        parser.error("--residual-gain must be between 0 and 1")
    paths = {
        key: value if value.is_absolute() else ROOT / value
        for key, value in {
            "bag_path": args.bag, "model_path": args.model,
            "library_path": args.library, "config_path": args.config,
            "trajectory_path": args.trajectory, "output_path": args.output,
        }.items()
    }
    report = evaluate(
        paths["bag_path"], paths["model_path"], paths["library_path"],
        paths["config_path"], paths["trajectory_path"], args.residual_gain,
        args.profile)
    output = paths["output_path"]
    if output.exists():
        parser.error(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    key_metrics = ("0.025s:r_radps", "0.1s:r_radps", "0.25s:r_radps",
                   "0.5s:r_radps", "0.75s:r_radps", "0.75s:position_m",
                   "0.75s:heading_rad")
    print(json.dumps({
        "run_id": report["run_id"],
        "accepted_windows": report["accepted_windows"],
        "skipped_windows": report["skipped_windows"],
        "zero_correction_equation_parity_max_abs": (
            report["zero_correction_equation_parity_max_abs"]),
        "whole_run_metrics": {
            key: report["whole_run_metrics"][key] for key in key_metrics
        },
        "high_steer_metrics": {
            key: report["high_steer_metrics_excluding_zero_steer_baseline"][key]
            for key in key_metrics
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
