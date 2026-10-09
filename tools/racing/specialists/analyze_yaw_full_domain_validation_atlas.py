#!/usr/bin/env python3
"""Build a held-out, per-speed/steering/event yaw-error atlas.

Consumes the exact-two-only predictions from audit_yaw_full_domain_exact_two.
The speed/steering bins are analysis labels only; ground-truth speed is never
used to route or fit the predictor. Yaw-rate residual errors are also
integrated on the specified 25-ms packet grid to report response-window yaw
angle deviation, without using jittered receipt intervals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
               "yaw_full_domain_exact_two_v2")
DT_S = 0.025
STEERING_BIN_RAD = 0.1
SPEED_BIN_MPS = 1.0
ANGLE_LIMIT_DEG = 5.0


def _metric(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return {"samples": 0}
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
        "fraction_abs_error_at_most_0p1_radps": float(
            np.mean(absolute <= 0.1)),
    }


def _steer_bin(value: float) -> int:
    # 11 signed bins centered at -0.5, -0.4, ..., +0.5 rad.
    return int(np.clip(np.floor((value + 0.55) / STEERING_BIN_RAD), 0, 10))


def _phase_metadata(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for run_id, run in report["phase_audit"]["per_run"].items():
        for item in run["measurements"]:
            if (item.get("response_packet_count") != 2
                    or item.get("packet_sequence_contiguous") is not True):
                continue
            stimulus = item["stimulus"]
            key = (run_id, str(item["label"]))
            output[key] = {
                "run_id": run_id,
                "split": run["split"],
                "event": str(stimulus["event"]),
                "requested_speed_mps": float(stimulus["requested_speed_mps"]),
                "requested_abs_steering_rad": float(
                    stimulus["requested_abs_steering_rad"]),
                "turn_sign": int(stimulus["turn_sign"]),
                "family": str(stimulus["family"]),
            }
    return output


def _integrated_phase_errors(
        arrays: dict[str, np.ndarray], error: np.ndarray,
        metadata: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    run_ids = arrays["run_id"].astype(str)
    phases = arrays["response_phase"].astype(str)
    times = arrays["sample_time_ns"]
    for index, (run_id, phase) in enumerate(zip(run_ids, phases)):
        grouped[(run_id, phase)].append(index)

    rows = []
    for key, all_indices in grouped.items():
        meta = metadata.get(key)
        if meta is None:
            # Phase labels carry the transition suffix; retain a strict join
            # rather than guessing from the label text.
            continue
        all_indices.sort(key=lambda index: int(times[index]))
        indices = [index for index in all_indices if np.isfinite(error[index])]
        complete = len(indices) == len(all_indices)
        row = {
            "run_id": key[0], "response_phase": key[1], **meta,
            "samples": len(indices),
            "expected_samples": len(all_indices),
            "coverage_fraction": (float(len(indices) / len(all_indices))
                                  if all_indices else 0.0),
            "fully_supported": complete,
            "max_abs_cumulative_yaw_angle_error_deg": None,
            "terminal_yaw_angle_error_deg": None,
        }
        if complete and indices:
            angle_error_deg = np.cumsum(error[indices] * DT_S) * (180.0 / np.pi)
            row["max_abs_cumulative_yaw_angle_error_deg"] = float(
                np.max(np.abs(angle_error_deg)))
            row["terminal_yaw_angle_error_deg"] = float(angle_error_deg[-1])
        rows.append(row)
    return rows


def _cell_metrics(arrays: dict[str, np.ndarray], error: np.ndarray
                  ) -> dict[str, Any]:
    speed = arrays["gt_speed_mps"]
    steering = arrays["steering_feedback_rad"]
    valid = np.isfinite(error)
    speed_bins = np.floor(speed / SPEED_BIN_MPS).astype(int)
    steering_bins = np.asarray([_steer_bin(float(value))
                                for value in steering], dtype=np.int32)
    keys = set()
    for index in np.flatnonzero(valid):
        keys.add((int(speed_bins[index]), int(steering_bins[index])))

    output = {}
    for speed_bin, steering_bin in sorted(keys):
        mask = (valid
                & (speed_bins == speed_bin)
                & (steering_bins == steering_bin))
        center = -0.5 + steering_bin * STEERING_BIN_RAD
        output[f"v{speed_bin}-{speed_bin + 1}_steer{center:+.1f}"] = {
            "gt_speed_interval_mps": [float(speed_bin),
                                      float(speed_bin + SPEED_BIN_MPS)],
            "steering_bin_center_rad": center,
            "samples": int(mask.sum()),
            "independent_runs": int(len(set(arrays["run_id"][mask].astype(str)))),
            "yaw_rate_error": _metric(error[mask]),
        }
    return output


def _condition_metrics(arrays: dict[str, np.ndarray], error: np.ndarray,
                       condition: np.ndarray) -> dict[str, Any]:
    output = {}
    for label in sorted(set(condition.astype(str))):
        mask = condition.astype(str) == label
        output[label] = {
            "independent_runs": int(len(set(arrays["run_id"][mask].astype(str)))),
            **_metric(error[mask]),
        }
    return output


def analyze(directory: Path) -> dict[str, Any]:
    report_path = directory / "report.json"
    predictions_path = directory / "validation_predictions.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    policy = report["acceptance_policy"]
    if (policy.get("only_response_packet_count") != 2
            or not policy.get("requires_contiguous_packet_sequence")
            or policy.get("one_three_and_other_counts_excluded_from_fit_and_score")
            is not True):
        raise ValueError("source report does not enforce the exact-two policy")

    with np.load(predictions_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    ids = arrays["run_id"].astype(str)
    if len(set(ids)) != report["model"]["validation_runs_with_exact_two_rows"]:
        raise ValueError("validation prediction runs disagree with source report")

    target = arrays["target_yaw_residual_radps"]
    predictions = {
        "global": arrays["predicted_global_residual_radps"],
        "causal_event_specialists": arrays[
            "predicted_causal_event_residual_radps"],
        "local_speed_steering_event_experts": arrays[
            "predicted_local_residual_radps"],
    }
    metadata = _phase_metadata(report)
    result: dict[str, Any] = {
        "title": "Held-out exact-two yaw error atlas",
        "scope": "one-step teacher-forced response windows; not recursive full-run prediction",
        "packet_policy": "exactly two contiguous command-to-feedback packets only",
        "target": "GT yaw_rate[k+1] - current exact-source-time IMU yaw_rate[k]",
        "dt_for_angle_integration_s": DT_S,
        "angle_acceptance_limit_deg": ANGLE_LIMIT_DEG,
        "ground_truth_used_for_routing": False,
        "jittered_receipt_intervals_used_as_dt": False,
        "validation_runs": int(len(set(ids))),
        "validation_rows": int(len(ids)),
        "models": {},
    }
    for name, prediction in predictions.items():
        rate_error = prediction - target
        phase_rows = _integrated_phase_errors(arrays, rate_error, metadata)
        peaks = np.asarray([
            row["max_abs_cumulative_yaw_angle_error_deg"]
            for row in phase_rows if row["fully_supported"]], dtype=np.float64)
        terminals = np.asarray([
            row["terminal_yaw_angle_error_deg"] for row in phase_rows
            if row["fully_supported"]], dtype=np.float64)
        complete_phase_count = int(len(peaks))
        max_peak = float(np.max(peaks)) if complete_phase_count else None
        p95_peak = float(np.quantile(peaks, 0.95)) if complete_phase_count else None
        max_terminal = (float(np.max(np.abs(terminals)))
                        if complete_phase_count else None)
        result["models"][name] = {
            "yaw_rate_error": _metric(rate_error),
            "supported_rows": int(np.isfinite(rate_error).sum()),
            "row_coverage_fraction": float(np.isfinite(rate_error).mean()),
            "one_step_yaw_angle_increment_error_max_deg": float(
                np.max(np.abs(rate_error[np.isfinite(rate_error)]))
                * DT_S * 180.0 / np.pi),
            "integrated_response_phase": {
                "joined_phases": len(phase_rows),
                "fully_supported_phases": complete_phase_count,
                "partially_or_un_supported_phases": len(phase_rows) - complete_phase_count,
                "max_abs_cumulative_error_deg": max_peak,
                "p95_abs_cumulative_error_deg": p95_peak,
                "phases_over_5deg": (int(np.count_nonzero(peaks > ANGLE_LIMIT_DEG))
                                      if complete_phase_count else None),
                "max_abs_terminal_error_deg": max_terminal,
                "phases_terminal_over_5deg": (
                    int(np.count_nonzero(np.abs(terminals) > ANGLE_LIMIT_DEG))
                    if complete_phase_count else None),
                "by_phase": phase_rows,
            },
            "by_measured_response_event": _condition_metrics(
                arrays, rate_error, arrays["measured_response_event"]),
            "by_causal_event": _condition_metrics(
                arrays, rate_error, arrays["causal_event"]),
            "by_speed_and_signed_steering": _cell_metrics(arrays, rate_error),
        }

    # Quantify exact-two command-phase support independently of row scores.
    phase_support: dict[str, dict[str, Any]] = {}
    for meta in metadata.values():
        if meta["split"] not in ("train", "validation"):
            continue
        speed_bin = int(np.floor(meta["requested_speed_mps"]))
        steer_bin = min(4, int(np.floor(
            meta["requested_abs_steering_rad"] / 0.1)))
        key = f"{meta['split']}_v{speed_bin}-{speed_bin + 1}_abssteer{steer_bin}"
        row = phase_support.setdefault(key, {
            "split": meta["split"],
            "requested_speed_interval_mps": [speed_bin, speed_bin + 1],
            "requested_abs_steering_bin_rad": [steer_bin * 0.1,
                                                (steer_bin + 1) * 0.1],
            "phases": 0, "runs": set(), "events": defaultdict(int),
            "turn_signs": set(),
        })
        row["phases"] += 1
        # Recover run id directly from the metadata key, preserving provenance.
        # This avoids confusing duplicate phase settings in separate captures.
        row["runs"].add(meta["run_id"])
        row["events"][meta["event"]] += 1
        row["turn_signs"].add(meta["turn_sign"])
    result["exact_two_requested_phase_support"] = {
        key: {**value, "runs": len(value["runs"]),
              "capture_runs": sorted(value["runs"]),
              "events": dict(value["events"]),
              "turn_signs": sorted(value["turn_signs"])}
        for key, value in sorted(phase_support.items())
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    result = analyze(args.directory)
    output = args.directory / "validation_atlas.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("wrote", output)
    for name, row in result["models"].items():
        integrated = row["integrated_response_phase"]
        print(name, "rate", row["yaw_rate_error"], "angle window max/p95",
              integrated["max_abs_cumulative_error_deg"],
              integrated["p95_abs_cumulative_error_deg"], "phases >5deg",
              integrated["phases_over_5deg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
