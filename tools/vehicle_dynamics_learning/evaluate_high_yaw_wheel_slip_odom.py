#!/usr/bin/env python3
"""Paired source-node replay for the narrowly gated high-yaw odom candidate.

Replays only encoder and IMU topics. Ground truth is read from the closed bag
after replay for evaluation, never supplied to the observer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning import evaluate_sensor_observer_practice as practice
from tools.vehicle_dynamics_learning import prepare_dataset
from tools.vehicle_dynamics_learning.train_odom_u_residual import (
    _OdomCollector, _replay_one,
)


def _metrics(error: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(len(error)),
        "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "p95_abs": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
    }


def _score_windows(capture: body.Capture, states: dict[int, np.ndarray],
                   expected_laps: int | None) -> dict[str, Any]:
    if expected_laps is not None:
        quality = practice._receipt_gates(
            capture.path, expected_laps=expected_laps,
            allow_post_run_disconnect=True)
        _, labels, samples, _, _, invalid, unscored = practice._score_arrays(
            capture, quality["active_start_receipt_ns"],
            quality[f"lap{expected_laps}_receipt_ns"], [],
            *practice.WINDOW_STEPS)
        burn_in, score_steps = practice.WINDOW_STEPS
        truth = labels[:, burn_in:burn_in + score_steps, :3].reshape(-1, 3)
    else:
        quality_ok, quality_failures = prepare_dataset._quality(
            capture, fixed_packet_timebase=True)
        quality = {"fixed_packet_timebase_pass": quality_ok,
                   "failures": quality_failures}
        score_steps = 32
        samples = []
        truth_windows = []
        seen_stamps: set[int] = set()
        unique_valid_samples = 0
        sequence_context_samples = sum(
            len(sequence) for sequence in capture.sequences)
        # load_capture attaches 500 ms of prior context to each phase. That
        # context may appear in several phase sequences, so score only each
        # valid phase's own samples and count a packet timestamp once.
        for sequence in capture.sequences:
            phase_samples = []
            for sample in sequence:
                stamp = int(sample.source_stamp_ns)
                if sample.time_s < 0.0 or stamp in seen_stamps:
                    continue
                seen_stamps.add(stamp)
                phase_samples.append(sample)
            unique_valid_samples += len(phase_samples)
            for start in range(
                    0, len(phase_samples) - score_steps + 1, score_steps):
                window = phase_samples[start:start + score_steps]
                samples.extend(window)
                truth_windows.append(np.stack([item.state[:3] for item in window]))
        if not truth_windows:
            raise ValueError("capture has no complete 32-sample evaluation windows")
        truth = np.concatenate(truth_windows, axis=0)
        invalid = 0
        unscored = unique_valid_samples - len(samples)
        context_or_duplicate_samples_excluded = (
            sequence_context_samples - unique_valid_samples)

    stamps = [int(sample.source_stamp_ns) for sample in samples]
    missing = [stamp for stamp in stamps if stamp not in states]
    if missing:
        raise ValueError(f"source replay missed {len(missing)}/{len(stamps)} scored samples")
    estimate = np.stack([states[stamp] for stamp in stamps])
    if len(estimate) != len(truth) or len(estimate) % score_steps:
        raise ValueError("replayed states do not align to complete truth windows")

    error = estimate - truth
    high_yaw = np.abs(truth[:, 2]) >= 2.5
    motion = practice._integrated_motion_metrics(
        estimate, truth, samples, score_steps)
    return {
        "samples": len(stamps),
        "coverage": 1.0,
        "invalid_sensor_samples": int(invalid),
        "unscored_samples": int(unscored),
        "all": _metrics(error),
        "high_yaw_abs_r_ge_2p5": _metrics(error[high_yaw]) if np.any(high_yaw) else None,
        "truth_reset_local_integrated_motion": motion,
        "quality": quality,
        **({"unique_valid_phase_samples": unique_valid_samples,
           "context_or_duplicate_samples_excluded": (
               context_or_duplicate_samples_excluded)}
           if expected_laps is None else {}),
    }


def _paired_output_delta(capture: body.Capture,
                         baseline: dict[int, np.ndarray],
                         candidate: dict[int, np.ndarray]) -> dict[str, Any]:
    """Report whether the gated candidate actually changes published states."""
    truth_by_stamp: dict[int, np.ndarray] = {}
    for sequence in capture.sequences:
        for sample in sequence:
            stamp = int(sample.source_stamp_ns)
            if sample.time_s >= 0.0:
                truth_by_stamp.setdefault(stamp, sample.state[:3])
    stamps = sorted(baseline.keys() & candidate.keys() & truth_by_stamp.keys())
    if not stamps:
        raise ValueError("paired baseline/candidate replays have no common stamps")
    base = np.stack([baseline[stamp] for stamp in stamps])
    trial = np.stack([candidate[stamp] for stamp in stamps])
    delta = trial - base
    changed = np.any(np.abs(delta) > 1.0e-12, axis=1)

    truth = np.stack([truth_by_stamp[stamp] for stamp in stamps])
    high_yaw = np.abs(truth[:, 2]) >= 2.5

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        selected = delta[mask]
        if not len(selected):
            return {
                "samples": 0,
                "changed_samples": 0,
                "mean_abs_delta": None,
                "rmse_delta": None,
                "max_abs_delta": None,
            }
        return {
            "samples": int(np.count_nonzero(mask)),
            "changed_samples": int(np.count_nonzero(changed & mask)),
            "mean_abs_delta": np.mean(np.abs(selected), axis=0).tolist(),
            "rmse_delta": np.sqrt(np.mean(selected ** 2, axis=0)).tolist(),
            "max_abs_delta": np.max(np.abs(selected), axis=0).tolist(),
        }

    return {
        "common_samples": len(stamps),
        "exactly_unchanged_samples": int(np.count_nonzero(~changed)),
        "changed_samples": int(np.count_nonzero(changed)),
        "all": summarize(np.ones(len(stamps), dtype=bool)),
        "truth_abs_yaw_rate_ge_2p5": summarize(high_yaw),
        "truth_abs_yaw_rate_lt_2p5": summarize(~high_yaw),
    }


def evaluate(bag: Path, node_binary: Path, params_file: Path,
             output_dir: Path, rate: float, domain_id: int,
             expected_laps: int | None,
             candidate_params: list[str] | None = None,
             baseline_params: list[str] | None = None) -> dict[str, Any]:
    bag, node_binary, params_file, output_dir = map(
        Path.resolve, (bag, node_binary, params_file, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = body.load_capture(bag)
    if expected_laps is None:
        quality_ok, quality_failures = prepare_dataset._quality(
            capture, fixed_packet_timebase=True)
        if not quality_ok:
            raise ValueError(
                f"capture failed fixed-grid quality gates: {quality_failures}")
    else:
        # Practice bags do not have open-plane experiment manifests. Apply
        # the full-lap collision, timing-fault, and stream-cadence gates used
        # by the practice scorer instead of rejecting them as aborted
        # open-plane experiments.
        practice._receipt_gates(
            bag, expected_laps=expected_laps,
            allow_post_run_disconnect=True)

    os.environ.update({
        "ROS_DOMAIN_ID": str(domain_id),
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    })
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=[])
    collector = _OdomCollector(Node)()
    try:
        baseline_states = _replay_one(
            bag, node_binary, params_file, rate, domain_id, collector,
            output_dir / "baseline_logs",
            tuple(arg for value in [
                "use_high_yaw_wheel_slip_observer:=false",
                *(baseline_params or [])] for arg in ("-p", value)))
        candidate_overrides = candidate_params or [
            "use_high_yaw_wheel_slip_observer:=true"]
        candidate_states = _replay_one(
            bag, node_binary, params_file, rate, domain_id, collector,
            output_dir / "candidate_logs",
            tuple(arg for value in candidate_overrides for arg in ("-p", value)))
    finally:
        collector.destroy_node()
        rclpy.shutdown()

    baseline = _score_windows(capture, baseline_states, expected_laps)
    candidate = _score_windows(capture, candidate_states, expected_laps)
    state_stamps = sorted(baseline_states.keys() & candidate_states.keys())
    state_archive = output_dir / "paired_source_states.npz"
    np.savez_compressed(
        state_archive,
        source_stamp_ns=np.asarray(state_stamps, dtype=np.int64),
        baseline=np.stack([baseline_states[stamp] for stamp in state_stamps]),
        candidate=np.stack([candidate_states[stamp] for stamp in state_stamps]),
    )
    result = {
        "bag": str(bag),
        "node_binary": str(node_binary),
        "params_file": str(params_file),
        "rate": rate,
        "expected_laps": expected_laps,
        "sensor_inputs_only": [
            "/autodrive/roboracer_1/left_encoder",
            "/autodrive/roboracer_1/right_encoder",
            "/autodrive/roboracer_1/imu",
        ],
        "truth_used_only_for_scoring": True,
        "paired_source_states_archive": str(state_archive),
        "paired_candidate_output_delta": _paired_output_delta(
            capture, baseline_states, candidate_states),
        "baseline": baseline,
        "candidate": candidate,
    }
    output = output_dir / "paired_source_node_score.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--node-binary", type=Path, required=True)
    parser.add_argument("--params-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--domain-id", type=int, default=83)
    parser.add_argument("--expected-laps", type=int,
                        help="set for a complete practice capture; omit for open-plane")
    parser.add_argument(
        "--candidate-param", action="append", dest="candidate_params",
        help="candidate ROS parameter override NAME:=VALUE; repeat as needed")
    parser.add_argument(
      "--baseline-param", action="append", dest="baseline_params",
      help="baseline ROS parameter override NAME:=VALUE; repeat as needed")
    args = parser.parse_args()
    result = evaluate(args.bag, args.node_binary, args.params_file,
                      args.output_dir, args.rate, args.domain_id,
                      args.expected_laps, args.candidate_params,
                      args.baseline_params)
    print(json.dumps({
        "bag": result["bag"],
        "baseline_rmse": result["baseline"]["all"]["rmse"],
        "candidate_rmse": result["candidate"]["all"]["rmse"],
        "baseline_high_yaw_rmse": (
            result["baseline"]["high_yaw_abs_r_ge_2p5"]["rmse"]
            if result["baseline"]["high_yaw_abs_r_ge_2p5"] else None),
        "candidate_high_yaw_rmse": (
            result["candidate"]["high_yaw_abs_r_ge_2p5"]["rmse"]
            if result["candidate"]["high_yaw_abs_r_ge_2p5"] else None),
        "candidate_changed_samples": result["paired_candidate_output_delta"][
            "changed_samples"],
        "candidate_changed_high_yaw_samples": result[
            "paired_candidate_output_delta"]["truth_abs_yaw_rate_ge_2p5"][
                "changed_samples"],
        "output": str(args.output_dir / "paired_source_node_score.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
