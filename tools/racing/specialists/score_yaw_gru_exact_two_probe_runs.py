#!/usr/bin/env python3
"""Score the frozen causal yaw GRU on exact-two-packet probe rows only.

The model is not fitted or selected here. Simulator truth and packet-response
class are used only as offline targets/filters; the GRU input remains causal
sensor history and the command sequence. Test/final-test captures are never
admitted by the shared dataset discovery gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import evaluate_yaw_packet_response_conditioned_models as response
    import audit_yaw_full_domain_exact_two as exact_two
    import train_yaw_gru_trajectory_teacher as gru_teacher
    import train_yaw_multihorizon_teacher as tree_teacher
except ModuleNotFoundError:
    from tools.racing.specialists import (
        evaluate_yaw_packet_response_conditioned_models as response,
        audit_yaw_full_domain_exact_two as exact_two,
        train_yaw_gru_trajectory_teacher as gru_teacher,
        train_yaw_multihorizon_teacher as tree_teacher,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    ROOT / "live_runs/racing_model_diagnostics_20261007/"
    "yaw_gru_trajectory_teacher_v1/yaw_gru_trajectory_teacher.pt")
DEFAULT_RUNS = (
    "openplane_yaw_error_highsteer_reversal_validation_r03_20261008",
    "openplane_yaw_error_packet_phase_validation_r09_20261008",
)
DEFAULT_OUTPUT = (
    ROOT / "live_runs/racing_model_diagnostics_20261008/"
    "yaw_large_error_audit_v1/frozen_gru_exact_two_probe_scores.json")


def _metric(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
        "fraction_within_0p1": float(np.mean(absolute <= 0.1)),
    }


def _source_frame_times(series: Any) -> np.ndarray:
    """Mirror `_read_run`'s sequence filter/order for timestamp alignment."""
    source = ROOT / series.source
    with np.load(source, allow_pickle=False) as archive:
        run_id = str(series.run_id)
        run_index = archive["run_ids"].astype(str).tolist().index(run_id)
        sequence_run_index = np.asarray(
            archive["sequence_run_index"], dtype=np.int64)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        times = np.asarray(archive["sample_time_ns"], dtype=np.int64)
        selected: list[np.ndarray] = []
        for sequence_index in np.flatnonzero(sequence_run_index == run_index):
            begin, end = map(int, bounds[int(sequence_index)])
            if end - begin < gru_teacher.HISTORY_STEPS + gru_teacher.FUTURE_STEPS:
                continue
            selected.append(times[begin:end])
    if not selected:
        raise ValueError(f"{series.run_id}: no GRU-eligible sequence timestamps")
    return np.concatenate(selected)


def _predict_one_step(series: Any, model, normalizers: dict[str, Any]
                      ) -> dict[int, float]:
    run = gru_teacher._make_run(series)
    frame_times = _source_frame_times(series)
    if len(frame_times) != len(run.observations):
        raise ValueError(
            f"{series.run_id}: GRU timestamp and observation lengths differ")
    starts = run.starts
    time_by_start = frame_times[starts]
    if len(np.unique(time_by_start)) != len(time_by_start):
        raise ValueError(
            f"{series.run_id}: duplicate source timestamps prevent unambiguous join")

    predictions: list[np.ndarray] = []
    torch.set_num_threads(2)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(starts), gru_teacher.BATCH_SIZE):
            batch_starts = starts[offset:offset + gru_teacher.BATCH_SIZE]
            past, future, target = gru_teacher.gather_windows(run, batch_starts)
            past, future, _ = gru_teacher._normalize(
                past, future, target, normalizers)
            normalized_prediction = model(
                torch.from_numpy(past), torch.from_numpy(future)).numpy()
            yaw_prediction = (
                normalized_prediction[:, 0] * normalizers["yaw_scale"]
                + normalizers["yaw_center"])
            predictions.append(np.asarray(yaw_prediction, dtype=np.float64))
    prediction = np.concatenate(predictions)
    return {int(stamp): float(value)
            for stamp, value in zip(time_by_start, prediction)}


def _collect_exact_two_windows(series: Any
                               ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect a requested run's causal yaw targets around exact-two events."""
    rows = response._run_rows(series)
    bag = exact_two.alignment._bag_for_run(series)
    phases, experiment_end = exact_two._read_phases(bag)
    measured = exact_two._measure_run(series, phases, experiment_end)
    phase_by_label = {phase.label: phase for phase in phases}
    times = np.asarray(rows["sample_time_ns"], dtype=np.int64)
    records: dict[str, list[np.ndarray]] = {
        "time_ns": [], "target_next_yaw_rate": [], "event": [],
        "event_age_ms": [], "condition": [],
    }
    excluded: dict[str, int] = {}
    for item in measured["measurements"]:
        stimulus = item["stimulus"]
        if stimulus.get("channel") != "steering":
            continue
        count = item.get("response_packet_count")
        if count != 2 or item.get("packet_sequence_contiguous") is not True:
            label = str(count if count is not None else
                        item.get("classification", "unknown"))
            excluded[label] = excluded.get(label, 0) + 1
            continue
        command_ns = item.get("command_start_ns")
        phase = phase_by_label.get(str(item["phase_label"]))
        if (not isinstance(command_ns, int) or phase is None
                or phase.valid is not True):
            excluded["invalid_exact_two_metadata"] = (
                excluded.get("invalid_exact_two_metadata", 0) + 1)
            continue
        event = str(stimulus.get("event", "unknown"))
        event = "turn_in" if event == "onset" else event
        if event not in {"turn_in", "unwind", "reversal"}:
            excluded[f"unsupported_event_{event}"] = (
                excluded.get(f"unsupported_event_{event}", 0) + 1)
            continue
        lower = max(int(phase.start_ns),
                    command_ns - response.WINDOW_BEFORE_NS)
        upper = min(int(phase.end_ns),
                    command_ns + response.WINDOW_AFTER_NS)
        selected = np.flatnonzero((times >= lower) & (times <= upper))
        if not len(selected):
            excluded["empty_exact_two_window"] = (
                excluded.get("empty_exact_two_window", 0) + 1)
            continue
        records["time_ns"].append(times[selected])
        records["target_next_yaw_rate"].append(
            (rows["residual"][selected] + rows["imu_yaw"][selected])
            .astype(np.float32))
        records["event"].append(np.full(len(selected), event, dtype="U16"))
        records["event_age_ms"].append(
            ((times[selected] - command_ns) / 1.0e6).astype(np.float32))
        event_condition = (
            f"{item['phase_label']}::transition"
            f"{stimulus.get('transition_index', 0)}")
        records["condition"].append(np.full(
            len(selected), event_condition, dtype="U256"))
    nonempty = [values for values in records.values() if values]
    if not nonempty:
        raise ValueError(f"{series.run_id}: no exact-two steering rows")
    collected = {key: np.concatenate(values) for key, values in records.items()}
    return collected, {
        "bag": str(bag.relative_to(ROOT)),
        "response_measurements": len(measured["measurements"]),
        "exact_two_windows": int(len(set(collected["condition"].tolist()))),
        "exact_two_rows_before_gru_timestamp_join": int(
            len(collected["time_ns"])),
        "excluded_response_counts": excluded,
        "quality": {
            "aborted": bool(experiment_end.get("aborted", True)),
            "reason": experiment_end.get("reason"),
            "quality_failures": experiment_end.get("quality_failures", []),
        },
    }


def score(checkpoint_path: Path, run_ids: tuple[str, ...],
          output_path: Path,
          allow_unseen_train_run: bool = False) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    model = gru_teacher.make_model().cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizers = checkpoint["normalizers"]

    admitted, source_audit = tree_teacher._discover_series_with_safe_mixed_archives()
    available = {series.run_id: series for series in admitted}
    missing = sorted(set(run_ids) - available.keys())
    if missing:
        raise ValueError(f"requested validation captures not admitted: {missing}")
    checkpoint_report_path = checkpoint_path.with_name(
        "yaw_gru_trajectory_teacher_report.json")
    checkpoint_report = json.loads(
        checkpoint_report_path.read_text(encoding="utf-8"))
    checkpoint_training = set(checkpoint_report["training_runs"])
    overlap = sorted(set(run_ids) & checkpoint_training)
    if overlap:
        raise ValueError(
            f"refusing to score checkpoint-training captures: {overlap}")
    train_split_requested = [run_id for run_id in run_ids
                             if available[run_id].split != "validation"]
    if train_split_requested and not allow_unseen_train_run:
        raise ValueError(
            "train-split holdouts require --allow-unseen-train-run and must "
            "be absent from the checkpoint's recorded training list")

    results: dict[str, Any] = {}
    phase_audits: dict[str, Any] = {}
    for run_id in run_ids:
        series = available[run_id]
        rows, phase_audits[run_id] = _collect_exact_two_windows(series)
        prediction_by_time = _predict_one_step(
            series, model, normalizers)
        group_results = {}
        for event in ("turn_in", "unwind", "reversal"):
            indices = np.flatnonzero(rows["event"] == event)
            candidate_count = len(indices)
            matched = np.asarray([
                int(rows["time_ns"][index]) in prediction_by_time
                for index in indices], dtype=bool)
            indices = indices[matched]
            if not len(indices):
                group_results[f"{event}/2_packet"] = {"samples": 0}
                continue
            true_next = rows["target_next_yaw_rate"][indices].astype(
                np.float64)
            predicted_next = np.asarray([
                prediction_by_time[int(rows["time_ns"][index])]
                for index in indices], dtype=np.float64)
            error = predicted_next - true_next
            ages = rows["event_age_ms"][indices].astype(np.float64)
            temporal = (("before_command", -np.inf, 0.0),
                        ("0_to_25ms", 0.0, 25.0),
                        ("25_to_50ms", 25.0, 50.0),
                        ("50_to_150ms", 50.0, 150.0),
                        ("150ms_plus", 150.0, np.inf))
            group_results[f"{event}/2_packet"] = {
                **_metric(error),
                "matched_rows": int(np.count_nonzero(matched)),
                "candidate_rows_before_timestamp_join": int(candidate_count),
                "by_transition_age": {
                    name: _metric(error[(ages >= lower) & (ages < upper)])
                    for name, lower, upper in temporal
                    if np.any((ages >= lower) & (ages < upper))
                },
            }
        results[run_id] = group_results

    result = {
        "title": "Frozen causal GRU on exact-two-packet yaw transition probes",
        "status": "offline score only; checkpoint was not refitted or selected",
        "checkpoint": str(checkpoint_path),
        "checkpoint_best_epoch": 70,
        "checkpoint_training_run_overlap": sorted(
            set(run_ids) & checkpoint_training),
        "target": "simulator-GT next 25-ms yaw rate; error in rad/s",
        "packet_response_policy": (
            "only exactly two packets are included; all other counts are excluded"),
        "features": "causal sensor/actuator history and commanded input only",
        "future_feedback_or_ground_truth_used_as_input": False,
        "test_and_final_test_arrays_opened": False,
        "source_audit": source_audit,
        "phase_audits": {run_id: phase_audits[run_id] for run_id in run_ids},
        "validation_results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        result, indent=2, sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic)
        else str(value)) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "results": results}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-unseen-train-run", action="store_true",
                        help="score a clean train-split capture only if the frozen checkpoint report confirms it was not trained on")
    args = parser.parse_args()
    score(args.checkpoint, tuple(args.run_id or DEFAULT_RUNS), args.output,
          allow_unseen_train_run=args.allow_unseen_train_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
