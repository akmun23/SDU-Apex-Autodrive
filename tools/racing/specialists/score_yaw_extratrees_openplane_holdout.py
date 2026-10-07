#!/usr/bin/env python3
"""Score a frozen ExtraTrees yaw atlas on a new Explore final-test capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from fit_fullband_yaw_extratrees import predict_transition
    from fit_fullband_yaw_regime_atlas import _make_rows, _metrics
    from score_yaw_atlas_openplane_holdout import _load_final_test
except ModuleNotFoundError:  # Importable both as a script and as a repo module.
    from tools.racing.specialists.fit_fullband_yaw_extratrees import (
        predict_transition,
    )
    from tools.racing.specialists.fit_fullband_yaw_regime_atlas import (
        _make_rows,
        _metrics,
    )
    from tools.racing.specialists.score_yaw_atlas_openplane_holdout import (
        _load_final_test,
    )


ROOT = Path(__file__).resolve().parents[3]
POINT_LABEL = re.compile(
    r"atlas_r\d+_v([0-9.]+)_a([0-9.]+)_turn([+-]\d)"
)


def score(model_path: Path, dataset_dir: Path, output_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    dataset_dir = dataset_dir.resolve()
    output_path = output_path.resolve()
    model_bytes = model_path.read_bytes()
    package = joblib.load(model_path)
    metadata_path = model_path.with_name(package.get(
        "manifest_file", "yaw_extratrees_fullband_v1_manifest.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    if metadata.get("model_sha256") != model_hash:
        raise ValueError("candidate model hash does not match its frozen manifest")
    supported_families = {
        "run_balanced_extra_trees_yaw_increment",
        "run_balanced_extra_trees_yaw_increment_lagged_history",
    }
    if (package.get("format_version") not in (1, 2)
            or package.get("model_family") not in supported_families
            or package.get("runtime_integration") != "none"):
        raise ValueError("unsupported or incorrectly identified model artifact")

    series, labels_by_sequence, phase_packets, packet_ids, manifest_run = (
        _load_final_test(dataset_dir)
    )
    x, yaw_delta, cells, phases, _, sequence_ids, frame_indices = _make_rows(
        series,
        float(package["phase_threshold_rad2_per_s"]),
        include_command_errors=True,
        include_command_rates=True,
        include_rear_wheel_split=False,
        include_lagged_history=bool(package.get("include_lagged_history", False)),
    )
    all_errors: dict[str, list[float]] = defaultdict(list)
    by_sequence: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_point: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unsupported = 0
    scored = 0
    for features, delta, cell, phase, sequence_id, frame_index in zip(
            x, yaw_delta, cells, phases, sequence_ids, frame_indices):
        label = labels_by_sequence.get(int(sequence_id))
        if label is None or int(packet_ids[int(frame_index)]) not in phase_packets[label]:
            continue
        key = (int(cell[0]), int(cell[1]), int(phase))
        prediction = predict_transition(package, key, features)
        if prediction is None:
            unsupported += 1
            continue
        scored += 1
        target = float(features[0] + delta)
        error = float(prediction - target)
        baseline_error = float(features[0] - target)
        all_errors["extra_trees"].append(error)
        all_errors["persistence"].append(baseline_error)
        by_sequence[label]["extra_trees"].append(error)
        by_sequence[label]["persistence"].append(baseline_error)

        match = POINT_LABEL.fullmatch(label)
        if match is None:
            raise ValueError(f"unrecognized frozen condition label: {label}")
        target_speed, target_angle, direction = (
            float(match.group(1)), float(match.group(2)), int(match.group(3))
        )
        measured_speed = float(0.25 + int(cell[0]) * 0.5 + features[2])
        measured_steering = float(-0.525 + int(cell[1]) * 0.025 + features[3])
        if (abs(measured_speed - target_speed) <= 0.15
                and abs(measured_steering - direction * target_angle) <= 0.015):
            by_point[label]["extra_trees"].append(error)
            by_point[label]["persistence"].append(baseline_error)

    per_sequence = {
        label: {name: _metrics(np.asarray(values, dtype=np.float64))
                for name, values in sorted(methods.items())}
        for label, methods in sorted(by_sequence.items())
    }
    per_point = {
        label: {name: _metrics(np.asarray(values, dtype=np.float64))
                for name, values in sorted(methods.items())}
        for label, methods in sorted(by_point.items())
    }
    result = {
        "title": "Frozen exact-cell ExtraTrees yaw atlas on new off-grid points",
        "model_file": str(model_path.relative_to(ROOT)),
        "model_sha256": model_hash,
        "dataset": str(dataset_dir.relative_to(ROOT)),
        "run_id": manifest_run["run_id"],
        "split": "final_test",
        "fit_or_model_selection_performed": False,
        "candidate_hyperparameters_were_selected_using_validation": True,
        "phase_packet_join": {
            "scored_maneuver_sequences": len(by_sequence),
            "packet_joined_by_reset_epoch": True,
            "scores_only_marked_maneuver_phases": True,
        },
        "supported_test_samples": scored,
        "unsupported_test_samples_abstained": unsupported,
        "one_step_yaw_rate_error_radps": {
            name: _metrics(np.asarray(values, dtype=np.float64))
            for name, values in sorted(all_errors.items())
        },
        "per_test_condition_sequence": per_sequence,
        "near_target_operating_point_windows": per_point,
        "limits": [
            "This is one-step prediction from current simulator-truth state, not recursive rollout or a sensor-only observer.",
            "Unsupported exact phase-cells abstain; no neighboring fallback is used.",
            "A single capture contains repeated conditions but only one independent run.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"score: {output_path}")
    print(json.dumps(result["one_step_yaw_rate_error_radps"], indent=2))
    print(f"supported: {scored}; abstained: {unsupported}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.model, args.dataset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
