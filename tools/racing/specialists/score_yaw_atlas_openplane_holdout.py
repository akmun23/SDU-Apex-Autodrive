#!/usr/bin/env python3
"""Score a frozen yaw atlas on a new reset-isolated Explore capture.

The capture must have been explicitly assigned final_test after recording. This
scorer never fits or selects a model; it evaluates one-step yaw predictions at
the current GT state and reports direct-cell and bilinear predictions
separately. It does not open any pre-existing test/final-test archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_open_plane_body_dynamics import load_capture
from fit_fullband_yaw_regime_atlas import (
    DT_S,
    RunSeries,
    _bilinear,
    _make_rows,
    _metrics,
    _predict,
)


ROOT = Path(__file__).resolve().parents[3]


def _load_final_test(dataset_dir: Path):
    manifest_path = dataset_dir / "manifest.json"
    archive_path = dataset_dir / "openplane_dynamics.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("runs", [])
    if len(rows) != 1:
        raise ValueError("holdout dataset must contain exactly one run")
    row = rows[0]
    if row.get("effective_split") != "final_test":
        raise ValueError("new holdout must be frozen as final_test before scoring")
    if (row.get("aborted") or row.get("reason") != "schedule complete"
            or not row.get("clean_stream_and_collision_gate")
            or row.get("quality_failures")
            or row.get("whole_bag_quality_failures")
            or int(row.get("timing_faults", -1)) != 0
            or any(int(value) != 0 for value in row.get("collisions", []))):
        raise ValueError("final-test capture failed the frozen clean-run gate")
    with np.load(archive_path, allow_pickle=False) as data:
        required = ("run_ids", "run_splits", "frames", "simulator_rigid_state",
                    "sequence_bounds", "sequence_labels", "sequence_reset_index",
                    "frame_reset_index", "dt_s")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"holdout archive is missing arrays: {missing}")
        run_ids = tuple(str(x) for x in data["run_ids"])
        splits = tuple(str(x) for x in data["run_splits"])
        if run_ids != (str(row.get("run_id")),) or splits != ("final_test",):
            raise ValueError("holdout archive identity/split disagrees with manifest")
        frames = np.asarray(data["frames"], dtype=np.float32)
        rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float32)
        bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
        labels = [str(x) for x in data["sequence_labels"]]
        sequence_reset = np.asarray(data["sequence_reset_index"], dtype=np.int64)
        frame_reset = np.asarray(data["frame_reset_index"], dtype=np.int64)
        dt = np.asarray(data["dt_s"], dtype=np.float32)
        if "packet_sequence" not in data.files:
            raise ValueError("holdout archive lacks packet identities")
        packets = np.asarray(data["packet_sequence"], dtype=np.int64)
    if (frames.ndim != 2 or frames.shape[1] != 9
            or rigid.shape != (len(frames), 13)
            or bounds.ndim != 2 or bounds.shape[1] != 2
            or len(labels) != len(bounds)
            or sequence_reset.shape != (len(bounds),)
            or frame_reset.shape != (len(frames),) or len(dt) != len(frames)
            or not np.allclose(dt, DT_S, rtol=0.0, atol=1.0e-7)
            or not np.isfinite(frames).all() or not np.isfinite(rigid).all()):
        raise ValueError("holdout arrays do not match the frozen 40 Hz schema")
    for begin, end in bounds:
        if np.any(np.diff(packets[int(begin):int(end)]) != 1):
            raise ValueError("holdout sequence contains a packet discontinuity")
    series = RunSeries(str(row["run_id"]), "final_test", str(archive_path),
                       frames, rigid, bounds)

    # Export labels are reset-epoch/lap labels, not maneuver labels. Recover
    # exact atlas phase markers from the source bag and join by simulator
    # packet identity. This is read-only scoring; it does not refit/select.
    bag_value = row.get("bag")
    if not isinstance(bag_value, str):
        raise ValueError("final-test manifest does not identify its source bag")
    capture = load_capture(ROOT / bag_value, include_nonvalid_phases=True)
    if (capture.aborted or capture.reason != "schedule complete"
            or capture.collision_count_start != capture.collision_count_end
            or capture.collision_count_end != 0 or capture.timing_faults):
        raise ValueError("raw final-test bag failed its clean-run gate")
    packet_to_reset: dict[int, int] = {}
    reset_to_packets: dict[int, set[int]] = defaultdict(set)
    reset_to_sequences: dict[int, list[int]] = defaultdict(list)
    for reset_id, (begin, end) in enumerate(bounds):
        begin, end = int(begin), int(end)
        local_reset_ids = np.unique(frame_reset[begin:end])
        if (len(local_reset_ids) != 1
                or int(local_reset_ids[0]) != int(sequence_reset[reset_id])):
            raise ValueError("holdout sequence crosses or mislabels a reset epoch")
        reset_epoch = int(local_reset_ids[0])
        reset_to_sequences[reset_epoch].append(reset_id)
        for packet_id in packets[begin:end]:
            if int(packet_id) in packet_to_reset:
                raise ValueError("packet identity occurs in multiple reset sequences")
            packet_to_reset[int(packet_id)] = reset_epoch
            reset_to_packets[reset_epoch].add(int(packet_id))
    phase_packets: dict[str, set[int]] = {}
    labels_by_sequence: dict[int, str] = {}
    for phase_label, phase_samples in zip(capture.sequence_labels,
                                          capture.sequences):
        if not phase_label.startswith("atlas_"):
            continue
        packet_ids = {int(sample.packet_sequence) for sample in phase_samples
                      if sample.packet_sequence >= 0}
        matched = {packet_to_reset[packet_id] for packet_id in packet_ids
                   if packet_id in packet_to_reset}
        if not packet_ids or len(matched) != 1:
            raise ValueError(
                f"phase {phase_label} does not map to exactly one reset sequence")
        reset_epoch = next(iter(matched))
        if phase_label in phase_packets or any(
                sequence_id in labels_by_sequence
                for sequence_id in reset_to_sequences[reset_epoch]):
            raise ValueError("holdout phase/reset mapping is not one-to-one")
        exported_packets = reset_to_packets[reset_epoch]
        overlap = packet_ids & exported_packets
        if len(overlap) / len(packet_ids) < 0.98:
            raise ValueError(f"phase packet join coverage too low: {phase_label}")
        for sequence_id in reset_to_sequences[reset_epoch]:
            labels_by_sequence[sequence_id] = phase_label
        phase_packets[phase_label] = packet_ids
    if len(phase_packets) != 24 or len(set(labels_by_sequence.values())) != 24:
        raise ValueError(
            f"expected 24 reset-isolated atlas probes, found {len(phase_packets)}")
    return series, labels_by_sequence, phase_packets, packets, row


def _restore_models(report: dict[str, Any]):
    unphased, phased = {}, {}
    for row in report["coefficient_models"]:
        model = {
            "mean": np.asarray(row["feature_mean"], dtype=np.float64),
            "coefficients": np.asarray(
                row["coefficients_on_delta_yaw_rate"], dtype=np.float64),
            "feature_scales": np.asarray(
                row["feature_scales"], dtype=np.float64),
        }
        if row["family"] == "cell":
            unphased[(int(row["speed_cell"]), int(row["steering_cell"]))] = model
        else:
            phased[(int(row["speed_cell"]), int(row["steering_cell"]),
                    int(row["phase"]))] = model
    return unphased, phased


def score(model_path: Path, dataset_dir: Path, output_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    dataset_dir = dataset_dir.resolve()
    output_path = output_path.resolve()
    model_bytes = model_path.read_bytes()
    model_report = json.loads(model_bytes)
    series, labels_by_sequence, phase_packets, packet_ids, manifest_run = (
        _load_final_test(dataset_dir))
    phase_threshold = float(model_report["model"].get(
        "phase_threshold_rad2_per_s", 0.05))
    include_command_errors = bool(model_report["model"].get(
        "command_tracking_errors_included", False))
    include_command_rates = bool(model_report["model"].get(
        "command_slew_rates_included", False))
    include_rear_wheel_split = bool(model_report["model"].get(
        "rear_wheel_split_included", False))
    x, delta, cells, phases, _, sequence_ids, frame_indices = _make_rows(
        series, phase_threshold, include_command_errors,
        include_command_rates, include_rear_wheel_split)
    unphased, phased = _restore_models(model_report)

    all_errors: dict[str, list[float]] = defaultdict(list)
    matched_errors: dict[str, list[float]] = defaultdict(list)
    sequence_errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    point_errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for features, yaw_delta, cell, phase, sequence_id, frame_index in zip(
            x, delta, cells, phases, sequence_ids, frame_indices):
        label = labels_by_sequence.get(int(sequence_id))
        if (label is None
                or int(packet_ids[int(frame_index)]) not in phase_packets[label]):
            continue
        speed_cell, steering_cell = map(int, cell)
        speed = float(0.25 + speed_cell * 0.5 + features[2])
        steering = float(-0.525 + steering_cell * 0.025 + features[3])
        target = float(features[0] + yaw_delta)
        predictions = {
            "persistence": float(features[0]),
            "direct_cell": (_predict(unphased[(speed_cell, steering_cell)], features)
                            if (speed_cell, steering_cell) in unphased else None),
            "direct_phase": (_predict(
                phased[(speed_cell, steering_cell, int(phase))], features)
                if (speed_cell, steering_cell, int(phase)) in phased else None),
            "bilinear_cell": _bilinear(
                unphased, features, speed, steering, int(phase), False),
            "bilinear_phase": _bilinear(
                phased, features, speed, steering, int(phase), True),
        }
        for method, prediction in predictions.items():
            if prediction is None:
                continue
            error = float(prediction - target)
            all_errors[method].append(error)
            matched_errors[method].append(error)
            if method != "persistence":
                matched_errors[f"persistence_same_{method}_support"].append(
                    float(features[0] - target))
            sequence_errors[label][method].append(error)
            if method != "persistence":
                sequence_errors[label][f"persistence_same_{method}_support"].append(
                    float(features[0] - target))
            # Explicit point windows: current truth speed/steering must lie
            # close to the new commanded operating point encoded in the label.
            try:
                match = re.fullmatch(
                    r"atlas_r\d+_v([0-9.]+)_a([0-9.]+)_turn([+-]\d)", label)
                if match is None:
                    raise ValueError("label does not match the frozen atlas format")
                target_speed, target_angle, direction = (
                    float(match.group(1)), float(match.group(2)),
                    int(match.group(3)))
            except ValueError:
                raise ValueError(f"cannot decode frozen test condition label: {label}")
            if (abs(speed - target_speed) <= 0.15
                    and abs(steering - direction * target_angle) <= 0.015):
                point_errors[label][method].append(error)
                if method != "persistence":
                    point_errors[label][f"persistence_same_{method}_support"].append(
                        float(features[0] - target))

    overall = {method: _metrics(np.asarray(errors, dtype=np.float64))
               for method, errors in sorted(all_errors.items())}
    per_sequence = {
        label: {method: _metrics(np.asarray(errors, dtype=np.float64))
                for method, errors in sorted(values.items())}
        for label, values in sorted(sequence_errors.items())
    }
    per_point = {
        label: {method: _metrics(np.asarray(errors, dtype=np.float64))
                for method, errors in sorted(values.items())}
        for label, values in sorted(point_errors.items())
    }
    result = {
        "title": "Frozen yaw-regime atlas on new reset-isolated open-plane points",
        "model_report": str(model_path.relative_to(ROOT)),
        "model_report_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "dataset": str(dataset_dir.relative_to(ROOT)),
        "run_id": manifest_run["run_id"],
        "split": "final_test",
        "fit_or_model_selection_performed": False,
        "source_manifest_run": manifest_run,
        "phase_packet_join": {
            "atlas_conditions": len(phase_packets),
            "reset_epochs_matched_one_to_one": len(phase_packets),
            "archive_sequences_joined": len(labels_by_sequence),
            "samples_scored_only_inside_marked_maneuver_phases": True,
        },
        "scored_new_point_sequences": len(per_sequence),
        "overall_one_step_yaw_rate_errors_radps": overall,
        "matched_support_one_step_errors_radps": {
            method: _metrics(np.asarray(errors, dtype=np.float64))
            for method, errors in sorted(matched_errors.items())
        },
        "per_test_condition_sequence": per_sequence,
        "near_target_operating_point_windows": per_point,
        "limits": [
            "Current-time GT state is used for this one-step plant transition score; no future truth is an input.",
            "This does not test free recursive rollout, sensor-only observer accuracy, or MPC tracking.",
            "A missing prediction means the frozen atlas lacked that cell support; no fallback was applied.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"score: {output_path}")
    print(json.dumps(overall, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.model_report, args.dataset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
