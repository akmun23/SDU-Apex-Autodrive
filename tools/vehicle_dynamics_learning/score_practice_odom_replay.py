#!/usr/bin/env python3
"""Compare sensor-odometry replay with production odom and simulator truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning import evaluate_sensor_observer_practice as practice


def _metrics(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "rmse": None, "bias": None,
                "p95_abs": None}
    return {"samples": int(len(error)),
            "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "bias": np.mean(error, axis=0).tolist(),
            "p95_abs": np.quantile(np.abs(error), 0.95, axis=0).tolist()}


def score(raw_bag: Path, candidate_replay: Path, output: Path,
          baseline_replay: Path | None = None,
          expected_laps: int = 12) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite score: {output}")
    quality = practice._receipt_gates(
        raw_bag, expected_laps=expected_laps,
        allow_post_run_disconnect=(expected_laps == 12))
    capture = body.load_capture(raw_bag)
    direct = body._read_replayed_states(raw_bag, "/odom")
    candidate = body._read_replayed_states(candidate_replay, "/replayed_odom")
    baseline = (body._read_replayed_states(baseline_replay, "/replayed_odom")
                if baseline_replay else direct)

    x, y, samples, _, _, invalid, unscored = practice._score_arrays(
        capture, quality["active_start_receipt_ns"],
        quality[f"lap{expected_laps}_receipt_ns"], [],
        *practice.WINDOW_STEPS)
    sensor = x[:, practice.WINDOW_STEPS[0]:].reshape(-1, 10)
    truth = y[:, practice.WINDOW_STEPS[0]:, :3].reshape(-1, 3)
    stamps = [sample.source_stamp_ns for sample in samples]

    def exact(states: dict[int, np.ndarray], name: str) -> np.ndarray:
        missing = [stamp for stamp in stamps if stamp not in states]
        if missing:
            raise ValueError(f"{name} misses {len(missing)}/{len(stamps)} source stamps")
        return np.stack([states[stamp] for stamp in stamps])

    estimates = {
        "recorded_production_odom": exact(direct, "recorded /odom"),
        "candidate_replay": exact(candidate, "candidate replay"),
    }
    if baseline_replay:
        estimates["baseline_replay"] = exact(baseline, "baseline replay")
        direct_subset = estimates["recorded_production_odom"]
        replay_subset = estimates["baseline_replay"]
        baseline_max_difference = float(np.max(np.abs(direct_subset - replay_subset)))
        if baseline_max_difference > 1e-9:
            raise ValueError(
                "baseline replay does not reproduce recorded production /odom: "
                f"max abs difference={baseline_max_difference:g}")
    else:
        baseline_max_difference = None

    speed = np.hypot(truth[:, 0], truth[:, 1])
    steering = np.abs(sensor[:, 0])
    throttle = sensor[:, 1]
    encoder_residual = np.abs(sensor[:, 2:4].mean(axis=1) - truth[:, 0])
    lateral_accel_proxy = np.abs(truth[:, 0] * truth[:, 2])
    regions = {
        "all_scored_samples": np.ones(len(truth), dtype=bool),
        "speed_ge_4_mps": speed >= 4.0,
        "speed_ge_6_mps": speed >= 6.0,
        "speed_ge_6_and_steering_ge_0p30":
            (speed >= 6.0) & (steering >= 0.30),
        "speed_ge_6_and_lateral_accel_proxy_ge_6":
            (speed >= 6.0) & (lateral_accel_proxy >= 6.0),
        "steering_ge_0p30_rad": steering >= 0.30,
        "encoder_mismatch_ge_0p5_mps": encoder_residual >= 0.5,
        "encoder_mismatch_ge_1_mps": encoder_residual >= 1.0,
        "wheel_mismatch_and_high_throttle":
            (encoder_residual >= 0.5) & (throttle >= 0.30),
    }
    results = {}
    for name, estimate in estimates.items():
        error = estimate - truth
        results[name] = {
            "all": _metrics(error),
            "regions": {key: _metrics(error[mask])
                        for key, mask in regions.items() if np.any(mask)},
        }
    report = {
        "run_id": raw_bag.parents[1].name,
        "raw_bag": str(raw_bag.resolve()),
        "candidate_replay": str(candidate_replay.resolve()),
        "baseline_replay": str(baseline_replay.resolve()) if baseline_replay else None,
        "quality": quality,
        "comparison": {
            "truth": "bridge odometry rear-axle body twist, offline only",
            "sample_window": {"burn_in": practice.WINDOW_STEPS[0],
                              "score_steps": practice.WINDOW_STEPS[1]},
            "scored_samples": len(truth),
            "expected_laps": expected_laps,
            "exact_source_stamp_coverage": len(stamps),
            "baseline_replay_max_abs_difference_from_recorded_odom": baseline_max_difference,
            "invalid_sensor_samples": invalid,
            "unscored_short_or_tail_samples": unscored,
            "lateral_acceleration_regime": "|u*r| kinematic proxy, not tire force",
        },
        "estimators": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bag", type=Path, required=True)
    parser.add_argument("--candidate-replay", type=Path, required=True)
    parser.add_argument("--baseline-replay", type=Path)
    parser.add_argument("--expected-laps", type=int, default=12,
                        help="completed lap-count transitions required in the raw capture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(args.raw_bag, args.candidate_replay, args.output,
                       args.baseline_replay, args.expected_laps)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"practice replay scoring failed: {exc}\n")
    print(json.dumps({
        "run_id": result["run_id"],
        "samples": result["comparison"]["scored_samples"],
        "estimators": {name: value["all"]["rmse"]
                       for name, value in result["estimators"].items()},
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
