#!/usr/bin/env python3
"""Freeze deterministic, common practice-run windows for WP2/WP3 scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DT_S = 0.025
CONTEXT_STEPS = 80
FUTURE_STEPS = 200
WINDOW_LIMIT_PER_RUN_CATEGORY = 8
SPEED_BINS = ((0.0, 5.0), (5.0, 7.0), (7.0, 9.0), (9.0, 12.0))
REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_CATEGORIES = (
    "straight", "brake", "turn_in", "apex", "exit", "high_steering",
    "wheel_body_mismatch_moderate", "wheel_body_mismatch_high",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_spread(indices: list[int], limit: int) -> list[int]:
    if limit <= 0:
        raise ValueError("selection limit must be positive")
    if len(indices) <= limit:
        return list(indices)
    slots = np.rint(np.linspace(0, len(indices) - 1, limit)).astype(int)
    return [indices[int(slot)] for slot in slots]


def _category_tags(index: int, speed: np.ndarray, steering: np.ndarray,
                   throttle_command: np.ndarray, mismatch: np.ndarray,
                   mismatch_median: float, mismatch_p90: float) -> list[str]:
    tags: list[str] = []
    value = float(speed[index])
    for low, high in SPEED_BINS:
        if low <= value < high or (value == high == 12.0):
            tags.append(f"speed_{low:g}_{high:g}_mps")
            break

    half_second = round(0.5 / DT_S)
    three_quarters = round(0.75 / DT_S)
    future_steer = np.abs(steering[index + 1:index + 1 + half_second])
    if len(future_steer) and np.max(future_steer) < 0.10:
        tags.append("straight")
    if value >= 2.0 and throttle_command[index] <= 0.001:
        tags.append("brake")

    steer_now = abs(float(steering[index]))
    if (len(future_steer) and steer_now < 0.10
            and np.max(future_steer) - steer_now >= 0.10
            and np.max(future_steer) >= 0.20):
        tags.append("turn_in")
    local_start = max(0, index - 10)
    local_end = min(len(steering), index + 11)
    if (steer_now >= 0.20
            and steer_now >= np.max(np.abs(steering[local_start:local_end]))):
        tags.append("apex")
    later_index = min(len(steering) - 1, index + half_second)
    if steer_now >= 0.20 and steer_now - abs(float(steering[later_index])) >= 0.10:
        tags.append("exit")
    if steer_now >= 0.40:
        tags.append("high_steering")
    mismatch_window = mismatch[index + 1:index + 1 + three_quarters]
    if len(mismatch_window):
        mismatch_max = float(np.max(mismatch_window))
        if mismatch_median <= mismatch_max < mismatch_p90:
            tags.append("wheel_body_mismatch_moderate")
        elif mismatch_max >= mismatch_p90:
            tags.append("wheel_body_mismatch_high")
    return tags


def _run_rows(archive: Any, run_index: int) -> np.ndarray:
    bounds = archive["sequence_bounds"]
    sequence_runs = archive["sequence_run_index"]
    sequences = [tuple(map(int, bounds[i])) for i in range(len(bounds))
                 if int(sequence_runs[i]) == run_index]
    sequences.sort()
    if not sequences:
        raise ValueError(f"run index {run_index} has no continuous sequences")
    indices = np.concatenate([np.arange(start, end, dtype=np.int64)
                              for start, end in sequences])
    packet_ids = archive["packet_sequence"][indices]
    if (len(indices) <= CONTEXT_STEPS + FUTURE_STEPS
            or np.any(np.diff(indices) != 1)
            or np.any(np.diff(packet_ids) != 1)
            or not np.allclose(archive["dt_s"][indices], DT_S,
                               rtol=0.0, atol=1e-7)):
        raise ValueError("practice run has a gap or insufficient common-window length")
    if (not np.all(archive["sensor_valid"][indices])
            or not np.isfinite(archive["simulator_pose_xyyaw"][indices]).all()
            or not np.isfinite(archive["simulator_rigid_state"][indices]).all()):
        raise ValueError("practice run has incomplete sensors or simulator labels")
    return indices


def _validate_source_chain(dataset_path: Path,
                           reports: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    current_dataset = dataset_path
    seen: set[Path] = set()
    matched: set[str] = set()
    chain = []
    while True:
        manifest_path = current_dataset.with_name("branch_view_manifest.json")
        if not manifest_path.is_file() or manifest_path in seen:
            break
        seen.add(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_run_id = str(manifest.get("source_run_id", ""))
        if source_run_id in reports:
            if manifest.get("bag_sha256") != reports[source_run_id]["bag_sha256"]:
                raise ValueError(f"strict report/bag hash mismatch for {source_run_id}")
            matched.add(source_run_id)
        chain.append({"path": str(manifest_path),
                      "sha256": _sha256(manifest_path)})
        base_value = manifest.get("base_dataset")
        if not base_value:
            break
        base_dataset = Path(base_value)
        current_dataset = (base_dataset if base_dataset.is_absolute()
                           else REPO_ROOT / base_dataset)
    if matched != set(reports):
        raise ValueError(
            "branch-view manifest chain does not authenticate all strict capture reports")
    return chain


def build(dataset_path: Path, validation_reports: list[Path],
          output_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen benchmark: {output_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    reports: dict[str, dict[str, Any]] = {}
    for report_path in validation_reports:
        report_path = report_path.resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("admitted") is not True
                or report.get("role") != "validation"
                or report.get("training_allowed") is not False
                or report.get("expected_laps") != 6):
            raise ValueError(f"not an admitted six-lap validation report: {report_path}")
        run_id = str(report.get("run_id", ""))
        if run_id in reports:
            raise ValueError(f"duplicate validation report for {run_id}")
        reports[run_id] = {
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "bag_sha256": report["bag_sha256"],
        }

    with np.load(dataset_path, allow_pickle=False) as archive:
        if (int(archive["schema_version"][0]) != 8
                or str(archive["dataset_role"][0]) != "race_domain"):
            raise ValueError("benchmark requires the frozen schema-8 race-domain view")
        run_ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
        run_indices = [i for i, (run_id, split) in enumerate(zip(run_ids, splits))
                       if split == "unseen_practice"
                       and run_id.startswith("practice_unseen_model_validation_")]
        if len(run_indices) < 2:
            raise ValueError("benchmark requires at least two fresh unseen practice runs")
        selected_run_ids = {str(run_ids[index]) for index in run_indices}
        expected_dataset_ids = {
            f"practice_unseen_model_validation_20261001_{run_id.rsplit('_', 1)[-1]}"
            for run_id in reports
        }
        if selected_run_ids != expected_dataset_ids:
            raise ValueError("strict report run IDs must exactly match benchmark practice runs")
        manifest_chain = _validate_source_chain(dataset_path, reports)

        train_run_indices = np.flatnonzero(splits == "train")
        train_rows = np.isin(archive["frame_run_index"], train_run_indices)
        frames_all = archive["frames"]
        train_mismatch = np.abs(
            0.5 * (frames_all[train_rows, 5] + frames_all[train_rows, 6])
            - frames_all[train_rows, 0])
        mismatch_median, mismatch_p90 = map(
            float, np.quantile(train_mismatch, (0.50, 0.90)))
        if (not np.isfinite(mismatch_median) or mismatch_median <= 0
                or not np.isfinite(mismatch_p90)
                or mismatch_p90 <= mismatch_median):
            raise ValueError("training split cannot define a wheel/body mismatch proxy")

        windows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        support: dict[str, dict[str, int]] = {}
        for run_index in run_indices:
            run_id = str(run_ids[run_index])
            indices = _run_rows(archive, run_index)
            frame = archive["frames"][indices]
            rigid = archive["simulator_rigid_state"][indices]
            speed = np.hypot(rigid[:, 7], rigid[:, 8])
            steering = frame[:, 3]
            yaw_rate = rigid[:, 12]
            mismatch = np.abs(0.5 * (frame[:, 5] + frame[:, 6]) - frame[:, 0])
            throttle_command = frame[:, 8]
            candidates: dict[str, list[int]] = {}
            last_start = len(frame) - FUTURE_STEPS - 1
            for index in range(CONTEXT_STEPS - 1, last_start + 1):
                tags = _category_tags(index, speed, steering,
                                      throttle_command, mismatch,
                                      mismatch_median, mismatch_p90)
                for tag in tags:
                    candidates.setdefault(tag, []).append(index)
            categories = [*(f"speed_{low:g}_{high:g}_mps"
                            for low, high in SPEED_BINS), *EVENT_CATEGORIES]
            support[run_id] = {}
            for category in categories:
                category_candidates = candidates.get(category, [])
                chosen = _select_spread(
                    category_candidates, WINDOW_LIMIT_PER_RUN_CATEGORY)
                support[run_id][category] = len(category_candidates)
                for index in chosen:
                    key = (run_id, index)
                    entry = windows_by_key.setdefault(key, {
                        "run_id": run_id,
                        "global_start_index": int(indices[index]),
                        "run_local_start_index": int(index),
                        "context_start_local_index": int(index - CONTEXT_STEPS + 1),
                        "context_steps": CONTEXT_STEPS,
                        "future_command_steps": FUTURE_STEPS,
                        "packet_sequence": int(archive["packet_sequence"][indices[index]]),
                        "sample_time_ns": int(archive["sample_time_ns"][indices[index]]),
                        "lap_count": int(archive["lap_count"][indices[index]]),
                        "initial_speed_mps": float(speed[index]),
                        "initial_steering_feedback_rad": float(steering[index]),
                        "initial_yaw_rate_rps": float(yaw_rate[index]),
                        "initial_wheel_body_mismatch_proxy_mps": float(mismatch[index]),
                        "categories": [],
                    })
                    if category not in entry["categories"]:
                        entry["categories"].append(category)

        windows = sorted(windows_by_key.values(),
                         key=lambda row: (row["run_id"], row["run_local_start_index"]))
        if not windows:
            raise ValueError("no deterministic benchmark windows were selected")
        report = {
            "schema_version": 1,
            "benchmark_id": "practice_transfer_benchmark_v1",
            "frozen": True,
            "training_or_checkpoint_selection_use": False,
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "branch_view_manifest_chain": manifest_chain,
            "independent_run_count": len(run_indices),
            "validation_reports": reports,
            "timebase_s": DT_S,
            "common_start_contract": {
                "same_run_id_and_packet_sequence_for_every_model": True,
                "context_steps": CONTEXT_STEPS,
                "future_command_steps": FUTURE_STEPS,
                "score_horizons_s": [0.5, 0.75, 1.0, 2.0],
                "all_context_sensor_and_future_simulator_labels_complete": True,
                "starts_are_selected_without_model_errors": True,
            },
            "category_definitions": {
                "straight": "max absolute steering feedback over the next 0.5 s is <0.10 rad",
                "brake": "initial speed >=2 m/s and recorded production throttle command is zero (active-brake mode)",
                "turn_in": "initial absolute steering <0.10 rad; over 0.5 s it grows by >=0.10 rad and reaches >=0.20 rad",
                "apex": "absolute steering >=0.20 rad and is a local maximum within +/-0.25 s",
                "exit": "initial absolute steering >=0.20 rad and falls by >=0.10 rad over the next 0.5 s",
                "high_steering": "absolute steering feedback >=0.40 rad, the existing race-domain steering-bin boundary",
                "wheel_body_mismatch_moderate": "next-0.75-s mismatch proxy reaches the train-split p50 but not p90",
                "wheel_body_mismatch_high": "next-0.75-s mismatch proxy reaches the train-split p90",
                "speed_bins_mps": [list(bounds) for bounds in SPEED_BINS],
                "max_windows_per_run_category": WINDOW_LIMIT_PER_RUN_CATEGORY,
                "selection": "deterministic evenly-spaced candidate indices within each run/category",
                "mismatch_training_median_mps": mismatch_median,
                "mismatch_training_p90_mps": mismatch_p90,
            },
            "candidate_support_by_run_and_category": support,
            "window_count_after_category_deduplication": len(windows),
            "windows": windows,
            "limitations": [
                "Practice captures reach only the observed range; empty speed/steering strata are unsupported, not extrapolated.",
                "Wheel/body mismatch is a measured slip proxy and is not a direct tire slip-angle or force label.",
                "Run-level uncertainty is based on independent captures, not the number of overlapping windows.",
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validation-report", type=Path, action="append",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.dataset, args.validation_report, args.output)
    print(json.dumps({
        "benchmark": report["benchmark_id"],
        "runs": report["independent_run_count"],
        "windows": report["window_count_after_category_deduplication"],
        "output": str(args.output),
        "candidate_support_by_run_and_category": report[
            "candidate_support_by_run_and_category"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
