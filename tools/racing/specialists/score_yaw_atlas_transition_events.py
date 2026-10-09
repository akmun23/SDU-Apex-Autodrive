#!/usr/bin/env python3
"""Score frozen yaw atlases by the intervention phase in a closed ROS bag.

The atlas phase label (turn-in/neutral/unwind) is not the same as the
experiment's commanded onset/unwind/reversal. This scorer joins each 25 ms
transition to its recorded probe interval and reports both levels separately.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

try:
    import fit_fullband_yaw_regime_atlas as atlas
    from score_frozen_yaw_atlas import _metric, _models, _predict
except ModuleNotFoundError:
    from tools.racing.specialists import fit_fullband_yaw_regime_atlas as atlas
    from tools.racing.specialists.score_frozen_yaw_atlas import (
        _metric, _models, _predict)

try:
    import analyze_open_plane_dynamics as dynamics
except ModuleNotFoundError:
    from tools import analyze_open_plane_dynamics as dynamics


def _feature_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    names = settings.get("feature_names", [])
    actuator_description = settings.get("steering_actuator_feature") or ""
    release_rule = settings.get("steering_release_rule")
    if release_rule is None:
        release_rule = ("same_sign_release"
                        if "same-sign magnitude release" in actuator_description
                        else "magnitude_decrease")
    return {
        "phase_threshold": float(settings["phase_threshold_rad2_per_s"]),
        "include_command_errors": bool(settings["command_tracking_errors_included"]),
        "include_command_rates": bool(settings["command_slew_rates_included"]),
        "include_rear_wheel_split": bool(settings["rear_wheel_split_included"]),
        "include_command_history": bool(settings.get("command_history_included", False)),
        "include_predicted_steering_change": bool(
            settings.get("predicted_steering_feedback_change_included", False)),
        "include_steering_release_indicator": (
            "predicted_steering_magnitude_decrease" in names
            or "predicted_same_sign_steering_release" in names
            or bool(settings.get("predicted_steering_release_indicator_included", False))),
        "steering_release_rule": release_rule,
    }


PHASE_TOPIC = "/open_plane_experiment/phase"
PROFILE_PREFIXES = (
    "probe_yawerr_highsteer_reversal_",
    "probe_yawerr_packet_phase_",
)
PROBE_LABEL_RE = re.compile(
    r"^probe_yawerr_(?:highsteer_reversal(?:_repeat)?_|packet_phase_r[0-9]+_)"
    r"(onset|unwind|reversal)_v([-+0-9.]+)_a([-+0-9.]+)_"
    r"turn([+-][0-9]+)_delay([-+0-9.]+)_(step|ramp)"
    r"([-+0-9.]+)s(?:_rep[0-9]+)?$")


def _read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "coefficient_models" not in report or "model" not in report:
        raise ValueError(f"not a frozen coefficient-atlas report: {path}")
    return report


def _probe_phases(
        bag_path: Path, expected_probe_count: int | None = 216
        ) -> tuple[list[Any], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{bag_path.resolve()}?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        phases, end = dynamics._phase_events(connection, topics)
    finally:
        connection.close()
    probes = [phase for phase in phases
              if phase.label.startswith(PROFILE_PREFIXES)]
    count_invalid = (not probes if expected_probe_count is None else
                     len(probes) != expected_probe_count)
    if (end.get("event") != "experiment_end" or end.get("aborted") is not False
            or end.get("reason") != "schedule complete"
            or end.get("quality_failures") or count_invalid
            or any(phase.valid is not True for phase in probes)):
        raise ValueError("bag is not the complete clean high-steer-reversal validation run")
    return probes, end


def _phase_lookup(times_ns: np.ndarray, phases: list[Any]) -> tuple[np.ndarray, list[str]]:
    ordered = sorted(phases, key=lambda phase: phase.start_ns)
    starts = np.asarray([phase.start_ns for phase in ordered], dtype=np.int64)
    ends = np.asarray([phase.end_ns for phase in ordered], dtype=np.int64)
    indices = np.searchsorted(starts, times_ns, side="right") - 1
    labels = [phase.label for phase in ordered]
    valid = indices >= 0
    safe = np.clip(indices, 0, max(0, len(ordered) - 1))
    valid &= times_ns < ends[safe]
    valid &= np.asarray([label.startswith(PROFILE_PREFIXES) for label in labels])[safe]
    return np.where(valid, indices, -1), labels


def _bucket(label: str) -> tuple[str, str]:
    match = PROBE_LABEL_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"unrecognized probe label: {label}")
    event, speed, steering, turn, delay, mode, duration = match.groups()
    return event, "/".join((event, f"v{speed}", f"a{steering}",
                            f"turn{turn}", f"delay{delay}",
                            f"{mode}{duration}s"))


def score(report_path: Path, run_id: str, bag_path: Path,
          compare_report_path: Path | None = None) -> dict[str, Any]:
    report = _read_report(report_path)
    other = _read_report(compare_report_path) if compare_report_path else None
    if other is not None:
        if (report["model"].get("phase_threshold_rad2_per_s")
                != other["model"].get("phase_threshold_rad2_per_s")):
            raise ValueError("atlas reports use different response-phase thresholds")

    candidates, source_audit = atlas._discover_run_series()
    matches = [row for row in candidates
               if row.run_id == run_id and row.split == "validation"]
    if len(matches) != 1:
        raise ValueError(f"expected one clean validation archive for {run_id!r}; found {len(matches)}")
    series = matches[0]
    config = report["model"]
    def make_rows(settings: dict[str, Any]):
        return atlas._make_rows(series, **_feature_kwargs(settings))

    features, delta, cells, phases, _, sequence_ids, frame_indices = make_rows(config)
    target = features[:, 0] + delta
    other_rows = make_rows(other["model"]) if other is not None else None
    if other_rows is not None:
        other_features, other_delta, other_cells, other_phases, _, other_sequences, other_frames = other_rows
        if (not np.array_equal(cells, other_cells)
                or not np.array_equal(phases, other_phases)
                or not np.array_equal(sequence_ids, other_sequences)
                or not np.array_equal(frame_indices, other_frames)
                or not np.allclose(delta, other_delta, rtol=0.0, atol=0.0)):
            raise ValueError("atlas reports do not produce aligned validation transitions")
    else:
        other_features = None
    with np.load(series.source, allow_pickle=False) as archive:
        times_ns = np.asarray(archive["sample_time_ns"], dtype=np.int64)
    if times_ns.shape != (len(series.frames),):
        raise ValueError("sample timestamp vector does not align with prepared run")
    probes, end = _probe_phases(bag_path)
    row_phase_indices, labels = _phase_lookup(times_ns[frame_indices], probes)
    next_phase_indices, _ = _phase_lookup(times_ns[frame_indices + 1], probes)
    row_phase_indices[next_phase_indices != row_phase_indices] = -1

    models_by_report = [_models(report)]
    if other is not None:
        models_by_report.append(_models(other))
    names = ["primary", "comparison"] if other is not None else ["primary"]
    totals: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    conditions: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    classifier_phase: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    worst_primary: list[dict[str, Any]] = []
    event_rows = 0
    event_labels: set[str] = set()

    rows = zip(features, target, cells, phases, frame_indices, row_phase_indices)
    if other_features is not None:
        rows = zip(features, target, cells, phases, frame_indices,
                   row_phase_indices, other_features)
    for values in rows:
        if other_features is None:
            x, truth, cell, model_phase, frame_index, event_index = values
            feature_rows = [x]
        else:
            (x, truth, cell, model_phase, frame_index, event_index,
             other_x) = values
            feature_rows = [x, other_x]
        if event_index < 0:
            continue
        event_label = labels[int(event_index)]
        event, condition = _bucket(event_label)
        event_rows += 1
        event_labels.add(event_label)
        keys = {
            "direct_cell": (int(cell[0]), int(cell[1])),
            "direct_phase": (int(cell[0]), int(cell[1]), int(model_phase)),
        }
        group = ("event=" + event)
        # Also retain the actual randomized speed/angle/direction/ramp condition.
        condition_group = "condition=" + condition
        classifier_group = f"event={event};atlas_phase={int(model_phase)}"
        for method, key in keys.items():
            row_errors: dict[str, float | None] = {}
            for model_name, models, model_x in zip(names, models_by_report, feature_rows):
                fitted = models.get(key)
                error = (None if fitted is None else
                         _predict(fitted, model_x) - float(truth))
                row_errors[model_name] = error
                if error is not None:
                    totals[method][group][model_name].append(error)
                    conditions[method][condition_group][model_name].append(error)
                    classifier_phase[method][classifier_group][model_name].append(error)
                if method == "direct_phase" and model_name == "primary" and error is not None:
                    worst_primary.append({
                        "absolute_error_radps": abs(error),
                        "signed_error_radps": error,
                        "sample_time_ns": int(times_ns[int(frame_index)]),
                        "event_label": event_label,
                        "event_type": event,
                        "atlas_response_phase": int(model_phase),
                        "measured_speed_mps": float(
                            atlas.SPEED_CENTERS[int(cell[0])] + x[2]),
                        "measured_steering_rad": float(
                            atlas.STEERING_CENTERS[int(cell[1])] + x[3]),
                        "target_yaw_rate_next_radps": float(truth),
                        "prediction_yaw_rate_next_radps": float(_predict(fitted, x)),
                        "current_yaw_rate_radps": float(x[0]),
                        "features": {
                            name: float(x[index])
                            for index, name in enumerate(config["feature_names"])
                        },
                    })
            if other is not None and all(value is not None for value in row_errors.values()):
                totals[method][group]["same_support_primary"].append(
                    float(row_errors["primary"]))
                totals[method][group]["same_support_comparison"].append(
                    float(row_errors["comparison"]))
                conditions[method][condition_group]["same_support_primary"].append(
                    float(row_errors["primary"]))
                conditions[method][condition_group]["same_support_comparison"].append(
                    float(row_errors["comparison"]))

    def summarize(groups: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
        return {group: {name: _metric(errors) for name, errors in sorted(rows.items())}
                for group, rows in sorted(groups.items())}

    output: dict[str, Any] = {
        "title": "One-step GT yaw atlas score by controlled steering event",
        "run_id": run_id,
        "bag": str(bag_path),
        "primary_report": str(report_path),
        "comparison_report": str(compare_report_path) if other else None,
        "target_period_ms": 25,
        "future_truth_used_as_input": False,
        "feature_contract": config["feature_names"],
        "comparison_feature_contract": (other["model"]["feature_names"]
                                        if other is not None else None),
        "probe_phases": len(probes),
        "probe_phase_rows_scored": event_rows,
        "unique_probe_conditions": len(event_labels),
        "experiment_end": end,
        "source_audit": source_audit,
        "by_event": {},
        "by_condition": {},
        "by_event_and_atlas_response_phase": {},
        "worst_primary_direct_phase_samples": [],
    }
    for method in ("direct_cell", "direct_phase"):
        output["by_event"][method] = summarize(totals[method])
        output["by_condition"][method] = summarize(conditions[method])
        output["by_event_and_atlas_response_phase"][method] = summarize(
            classifier_phase[method])
    output["worst_primary_direct_phase_samples"] = sorted(
        worst_primary, key=lambda row: row["absolute_error_radps"], reverse=True)[:30]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compare-report", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.report, args.run_id, args.bag, args.compare_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    print(f"scored {result['probe_phase_rows_scored']} transitions across "
          f"{result['unique_probe_conditions']} probe conditions")
    for method, events in result["by_event"].items():
        for event, metrics in events.items():
            print(f"{method} {event}: " + ", ".join(
                f"{name} n={row['samples']} RMSE={row['rmse_radps']:.5f} "
                f"p95={row['p95_abs_radps']:.5f}"
                for name, row in metrics.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
