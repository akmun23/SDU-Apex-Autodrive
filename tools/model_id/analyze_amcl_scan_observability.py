#!/usr/bin/env python3
"""Test whether recorded AMCL scans support an extra along-track correction.

This is an offline-only decision tool. It reconstructs the configured AMCL
likelihood-field score from the recorded LaserScan ranges and production map,
fits one bounded correction gain on the first chronological half, and scores
that unchanged gain on the second half. Simulator pose is used only as the
offline error target. Nothing here is imported by a runtime node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO_ROOT / (
    "f1tenth_planning/maps/autodrive_track_ftg_commit_20260909_025m.yaml")
DEFAULT_AMCL_CONFIG = REPO_ROOT / (
    "f1tenth_localization/config/gpu_amcl_cpp_params.yaml")
DEFAULT_ALPHA_GRID = np.linspace(0.0, 1.0, 21)
SEARCH_OFFSETS_M = np.arange(-0.20, 0.2001, 0.01)
MIN_SPEED_MPS = 0.5


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _payload(row: dict[str, str]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("payload_json", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _yaw_from_quaternion(payload: dict[str, Any], prefix: str = "") -> float | None:
    x = _finite(payload.get(f"{prefix}orientation_quaternion_x"))
    y = _finite(payload.get(f"{prefix}orientation_quaternion_y"))
    z = _finite(payload.get(f"{prefix}orientation_quaternion_z"))
    w = _finite(payload.get(f"{prefix}orientation_quaternion_w"))
    if None in (x, y, z, w):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"samples": 0, "mae": None, "rmse": None, "p95": None}

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lo, hi = math.floor(position), math.ceil(position)
        if lo == hi:
            return ordered[lo]
        weight = position - lo
        return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

    return {
        "samples": len(ordered),
        "mae": float(np.mean(ordered)),
        "rmse": float(np.sqrt(np.mean(np.square(ordered)))),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(ordered),
    }


class MapLikelihood:
    """CPU reproduction of AMCL's bilinear likelihood-field scan score."""

    def __init__(self, map_yaml: Path, amcl_yaml: Path):
        metadata = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
        image_path = (map_yaml.parent / metadata["image"]).resolve()
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        resolution = float(metadata["resolution"])
        negate = int(metadata.get("negate", 0))
        occupancy_probability = image / 255.0 if negate else (255.0 - image) / 255.0
        occupied = occupancy_probability > float(metadata["occupied_thresh"])
        # PGM row zero is the top of the image; OccupancyGrid row zero is the
        # bottom. The AMCL map processor stores row-major, bottom-up cells.
        occupied = np.flipud(occupied)
        if not np.any(occupied):
            raise ValueError(f"map has no occupied cells: {image_path}")
        distance = distance_transform_edt(~occupied) * resolution
        params_doc = yaml.safe_load(amcl_yaml.read_text(encoding="utf-8"))
        self.params = params_doc["gpu_amcl_cpp"]["ros__parameters"]
        self.distance = np.asarray(distance, dtype=np.float32)
        self.height, self.width = self.distance.shape
        self.resolution = resolution
        origin = metadata.get("origin", [0.0, 0.0, 0.0])
        self.origin_x, self.origin_y, self.origin_yaw = map(float, origin[:3])
        self.laser_min = float(self.params["laser_min_range"])
        self.laser_max = float(self.params["laser_max_range"])
        self.laser_ox = float(self.params["laser_offset_x"])
        self.laser_oy = float(self.params["laser_offset_y"])
        self.max_beams = int(self.params["max_beams"])
        self.sigma_hit = float(self.params["sigma_hit"])
        self.z_hit = float(self.params["z_hit"])
        self.z_rand = float(self.params["z_rand"])
        self.likelihood_scale = float(self.params["likelihood_scale"])

    def sampled_valid_range_count(self, ranges: np.ndarray) -> tuple[int, int]:
        """Count usable returns after the same stride used by AMCL."""
        step = max(1, len(ranges) // self.max_beams)
        sampled = np.asarray(ranges[::step], dtype=np.float32)
        valid = (np.isfinite(sampled) & (sampled >= self.laser_min) &
                 (sampled <= self.laser_max))
        return int(np.count_nonzero(valid)), int(len(sampled))

    def score_many(
            self, ranges: np.ndarray, angle_min: float,
            angle_increment: float, pose_x: np.ndarray,
            pose_y: np.ndarray, yaw: float) -> np.ndarray:
        """Return the configured normalized log score for each candidate pose."""
        beam_step = max(1, len(ranges) // self.max_beams)
        beam_indices = np.arange(0, len(ranges), beam_step)
        scan_ranges = np.asarray(ranges[beam_indices], dtype=np.float32)
        angles = angle_min + beam_indices * angle_increment
        valid = (np.isfinite(scan_ranges) & (scan_ranges >= self.laser_min) &
                 (scan_ranges <= self.laser_max))
        if not np.any(valid):
            return np.full(len(pose_x), -np.inf, dtype=np.float64)
        scan_ranges = scan_ranges[valid].astype(np.float64)
        angles = angles[valid]

        ct, st = math.cos(yaw), math.sin(yaw)
        laser_x = pose_x + self.laser_ox * ct - self.laser_oy * st
        laser_y = pose_y + self.laser_ox * st + self.laser_oy * ct
        beam_cos = np.cos(angles + yaw)[None, :]
        beam_sin = np.sin(angles + yaw)[None, :]
        endpoint_x = laser_x[:, None] + scan_ranges[None, :] * beam_cos
        endpoint_y = laser_y[:, None] + scan_ranges[None, :] * beam_sin

        dx, dy = endpoint_x - self.origin_x, endpoint_y - self.origin_y
        map_cos, map_sin = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        fx = (map_cos * dx + map_sin * dy) / self.resolution - 0.5
        fy = (-map_sin * dx + map_cos * dy) / self.resolution - 0.5
        in_bounds = ((fx >= -0.5) & (fx < self.width - 0.5) &
                     (fy >= -0.5) & (fy < self.height - 0.5))
        x0_raw, y0_raw = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        sx, sy = fx - x0_raw, fy - y0_raw
        x0 = np.clip(x0_raw, 0, self.width - 1)
        x1 = np.clip(x0_raw + 1, 0, self.width - 1)
        y0 = np.clip(y0_raw, 0, self.height - 1)
        y1 = np.clip(y0_raw + 1, 0, self.height - 1)
        d00 = self.distance[y0, x0]
        d10 = self.distance[y0, x1]
        d01 = self.distance[y1, x0]
        d11 = self.distance[y1, x1]
        d0 = d00 + sx * (d10 - d00)
        d1 = d01 + sx * (d11 - d01)
        distance = np.where(in_bounds, d0 + sy * (d1 - d0), self.laser_max)

        inv_2sigma2 = -0.5 / (self.sigma_hit * self.sigma_hit)
        normalizer = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)
        probability = (self.z_hit * normalizer *
                       np.exp(inv_2sigma2 * np.square(distance)) +
                       self.z_rand / self.laser_max)
        # AMCL divides by valid beam count before multiplying by scale.
        return (np.mean(np.log(np.maximum(probability, 1.0e-30)), axis=1) *
                max(self.likelihood_scale, 0.0))


def _read_truth_and_pose(events: list[dict[str, str]]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    timing = [row for row in events
              if row.get("topic") == "/autodrive/roboracer_1/bridge_packet_timing"]
    imu = [row for row in events if row.get("topic") == "/autodrive/roboracer_1/imu"]
    if len(timing) != len(imu) or not timing:
        raise ValueError("bridge-timing and IMU streams are not one-to-one")
    truth: dict[str, dict[str, float]] = {}
    for timing_row, imu_row in zip(timing, imu):
        stamp = imu_row.get("header_stamp_ns", "")
        payload = _payload(timing_row)
        values = {
            "x_m": _finite(payload.get("simulator_position_x")),
            "y_m": _finite(payload.get("simulator_position_y")),
            "yaw_rad": _yaw_from_quaternion(payload, "simulator_"),
            "simulation_time_s": _finite(payload.get("simulation_time_s")),
            "speed_mps": math.hypot(
                _finite(payload.get("simulator_linear_velocity_x")) or 0.0,
                _finite(payload.get("simulator_linear_velocity_y")) or 0.0),
        }
        if stamp and all(isinstance(values[key], float) for key in
                         ("x_m", "y_m", "yaw_rad", "simulation_time_s")):
            truth[stamp] = {key: float(value) for key, value in values.items()
                            if value is not None}

    pose: dict[str, dict[str, float]] = {}
    for row in events:
        if row.get("topic") != "/current_map_pose":
            continue
        stamp = row.get("header_stamp_ns", "")
        payload = _payload(row)
        values = {
            "x_m": _finite(payload.get("x_m")),
            "y_m": _finite(payload.get("y_m")),
            "yaw_rad": _finite(payload.get("yaw_rad")),
        }
        if stamp and all(value is not None for value in values.values()):
            pose[stamp] = {key: float(value) for key, value in values.items()
                           if value is not None}
    return truth, pose


def _score_summary(rows: list[dict[str, float]], alpha: float) -> dict[str, Any]:
    along_errors: list[float] = []
    cross_errors: list[float] = []
    yaw_errors: list[float] = []
    for row in rows:
        yaw = row["truth_yaw_rad"]
        dx = row["estimate_x_m"] + alpha * row["correction_x_m"] - row["truth_x_m"]
        dy = row["estimate_y_m"] + alpha * row["correction_y_m"] - row["truth_y_m"]
        along_errors.append(dx * math.cos(yaw) + dy * math.sin(yaw))
        cross_errors.append(-dx * math.sin(yaw) + dy * math.cos(yaw))
        yaw_errors.append(math.atan2(
            math.sin(row["estimate_yaw_rad"] - yaw),
            math.cos(row["estimate_yaw_rad"] - yaw)))
    return {
        "alpha": alpha,
        "along_abs_m": _summary(abs(value) for value in along_errors),
        "cross_abs_m": _summary(abs(value) for value in cross_errors),
        "yaw_abs_rad": _summary(abs(value) for value in yaw_errors),
        "along_signed_bias_m": float(np.mean(along_errors)) if along_errors else None,
    }


def analyze_run(
        run_dir: Path, map_yaml: Path = DEFAULT_MAP,
        amcl_yaml: Path = DEFAULT_AMCL_CONFIG,
        output_csv: Path | None = None) -> dict[str, Any]:
    events = _read_events(run_dir / "events.csv")
    truth_by_stamp, pose_by_stamp = _read_truth_and_pose(events)
    index_path = run_dir / "lidar_scan_index.csv"
    ranges_path = run_dir / "lidar_scan_ranges.f32le"
    if not index_path.is_file() or not ranges_path.is_file():
        raise ValueError("run is missing optional raw-LiDAR sidecar files")
    events_by_index = {int(row["event_index"]): row for row in events}
    model = MapLikelihood(map_yaml, amcl_yaml)
    output_rows: list[dict[str, Any]] = []
    unmatched_truth = unmatched_pose = 0
    scan_source_times: list[float] = []
    with index_path.open(newline="", encoding="utf-8") as stream:
        scans = list(csv.DictReader(stream))
    with ranges_path.open("rb") as range_stream:
        for index in scans:
            event_id = int(index["event_index"])
            event = events_by_index.get(event_id)
            if event is None or event.get("topic") != "/autodrive/roboracer_1/lidar":
                continue
            stamp = index.get("header_stamp_ns", "")
            truth = truth_by_stamp.get(stamp)
            estimate = pose_by_stamp.get(stamp)
            if truth is None:
                unmatched_truth += 1
                continue
            if estimate is None:
                unmatched_pose += 1
                continue
            scan_source_times.append(truth["simulation_time_s"])
            if truth["speed_mps"] < MIN_SPEED_MPS:
                continue
            count = int(index["range_count"])
            byte_length = int(index["byte_length"])
            offset = int(index["file_offset_bytes"])
            if byte_length != count * 4:
                raise ValueError(f"scan event {event_id}: inconsistent binary index")
            range_stream.seek(offset)
            ranges = np.fromfile(range_stream, dtype="<f4", count=count)
            if len(ranges) != count:
                raise ValueError(f"scan event {event_id}: truncated range sidecar")

            valid_beams, sampled_beams = model.sampled_valid_range_count(ranges)

            estimate_yaw = estimate["yaw_rad"]
            tangent_x, tangent_y = math.cos(estimate_yaw), math.sin(estimate_yaw)
            if valid_beams:
                candidates_x = estimate["x_m"] + SEARCH_OFFSETS_M * tangent_x
                candidates_y = estimate["y_m"] + SEARCH_OFFSETS_M * tangent_y
                angle_min = float(index["angle_min_rad"])
                angle_increment = float(index["angle_increment_rad"])
                scores = model.score_many(
                    ranges, angle_min, angle_increment, candidates_x,
                    candidates_y, estimate_yaw)
                center_score = float(model.score_many(
                    ranges, angle_min, angle_increment,
                    np.array([estimate["x_m"]]), np.array([estimate["y_m"]]),
                    estimate_yaw)[0])
                max_score = float(np.max(scores))
                tied = np.flatnonzero(scores >= max_score - 1.0e-7)
                best_i = int(tied[np.argmin(np.abs(SEARCH_OFFSETS_M[tied]))])
                correction = float(SEARCH_OFFSETS_M[best_i])
                plus_5cm = float(scores[np.argmin(np.abs(SEARCH_OFFSETS_M - 0.05))])
                minus_5cm = float(scores[np.argmin(np.abs(SEARCH_OFFSETS_M + 0.05))])
                score_gain = max_score - center_score
                score_curvature = 2.0 * center_score - plus_5cm - minus_5cm
                at_search_boundary = float(best_i in (0, len(scores) - 1))
            else:
                # Preserve the scan/pose row for data-quality accounting, but
                # never fit a correction from a scan with no sensor evidence.
                center_score = max_score = plus_5cm = minus_5cm = None
                correction = 0.0
                score_gain = score_curvature = None
                at_search_boundary = 0.0
            source_time = truth["simulation_time_s"]
            output_rows.append({
                "event_index": float(event_id),
                "header_stamp_ns": float(stamp),
                "simulation_time_s": source_time,
                "speed_mps": truth["speed_mps"],
                "truth_x_m": truth["x_m"],
                "truth_y_m": truth["y_m"],
                "truth_yaw_rad": truth["yaw_rad"],
                "estimate_x_m": estimate["x_m"],
                "estimate_y_m": estimate["y_m"],
                "estimate_yaw_rad": estimate_yaw,
                "baseline_along_error_m": (
                    (estimate["x_m"] - truth["x_m"]) * math.cos(truth["yaw_rad"]) +
                    (estimate["y_m"] - truth["y_m"]) * math.sin(truth["yaw_rad"])),
                "baseline_cross_error_m": (
                    -(estimate["x_m"] - truth["x_m"]) * math.sin(truth["yaw_rad"]) +
                    (estimate["y_m"] - truth["y_m"]) * math.cos(truth["yaw_rad"])),
                "correction_x_m": correction * tangent_x,
                "correction_y_m": correction * tangent_y,
                "correction_along_est_heading_m": correction,
                "sampled_valid_beam_count": valid_beams,
                "sampled_beam_count": sampled_beams,
                "center_log_likelihood": center_score,
                "best_log_likelihood": max_score,
                "best_score_gain": score_gain,
                "along_score_curvature_5cm": score_curvature,
                "best_offset_at_search_boundary": at_search_boundary,
                "score_plus_5cm": plus_5cm,
                "score_minus_5cm": minus_5cm,
            })

    if len(output_rows) < 200:
        raise ValueError(
            f"only {len(output_rows)} source-matched moving scans; need >=200")
    output_rows.sort(key=lambda row: row["simulation_time_s"])
    valid_output_rows = [row for row in output_rows
                         if row["sampled_valid_beam_count"] > 0]
    if output_csv is None:
        output_csv = run_dir / "amcl_scan_observability.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    intervals = np.diff(scan_source_times)
    source_timing = {
        "intervals": _summary(abs(value) for value in intervals),
        "period_p50_s": float(np.median(intervals)) if len(intervals) else None,
        "reversed_or_duplicate_count": int(np.count_nonzero(intervals <= 0.0)),
        "gaps_over_35ms_count": int(np.count_nonzero(intervals > 0.035)),
        "sub_15ms_count": int(np.count_nonzero(intervals < 0.015)),
    }
    if len(valid_output_rows) < 200:
        report = {
            "schema_version": 2,
            "offline_only": True,
            "purpose": (
                "Decide whether configured map likelihood provides a useful "
                "bounded along-track correction beyond current_map_pose"),
            "run_dir": str(run_dir),
            "map_yaml": str(map_yaml),
            "amcl_config": str(amcl_yaml),
            "recorded_scans": len(scans),
            "matched_moving_scans": len(output_rows),
            "scans_with_valid_sampled_returns": len(valid_output_rows),
            "scans_without_valid_sampled_returns": (
                len(output_rows) - len(valid_output_rows)),
            "unmatched_truth_scans": unmatched_truth,
            "unmatched_current_map_pose_scans": unmatched_pose,
            "speed_filter_mps": MIN_SPEED_MPS,
            "train_scans": 0,
            "holdout_scans": 0,
            "source_scan_timing": source_timing,
            "fit_gain_on_train": None,
            "holdout_baseline": None,
            "holdout_candidate": None,
            "decision": (
                "invalid_no_valid_lidar_returns" if not valid_output_rows else
                "invalid_insufficient_valid_lidar_returns"),
            "data_use": (
                "No correction fit is valid until the runtime scan stream has "
                "usable sampled returns; diagnose the LaserScan source and "
                "do not repeat this candidate test."),
            "analysis_csv": str(output_csv),
        }
        report_path = output_csv.with_suffix(".json")
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return report

    split_time = (valid_output_rows[0]["simulation_time_s"] +
                  valid_output_rows[-1]["simulation_time_s"]) / 2.0
    train = [row for row in valid_output_rows
             if row["simulation_time_s"] < split_time]
    holdout = [row for row in valid_output_rows
               if row["simulation_time_s"] >= split_time]
    if min(len(train), len(holdout)) < 100:
        raise ValueError("chronological split has fewer than 100 scans per half")

    train_scores = [
        (_score_summary(train, float(alpha))["along_abs_m"]["rmse"], float(alpha))
        for alpha in DEFAULT_ALPHA_GRID]
    fitted_alpha = min(train_scores, key=lambda item: float(item[0]))[1]
    baseline = _score_summary(holdout, 0.0)
    candidate = _score_summary(holdout, fitted_alpha)
    along_before = float(baseline["along_abs_m"]["p95"])
    along_after = float(candidate["along_abs_m"]["p95"])
    cross_before = float(baseline["cross_abs_m"]["p95"])
    cross_after = float(candidate["cross_abs_m"]["p95"])
    yaw_before = float(baseline["yaw_abs_rad"]["p95"])
    yaw_after = float(candidate["yaw_abs_rad"]["p95"])
    candidate_warrants_live_ab = (
        along_after <= 0.95 * along_before and
        cross_after <= 1.05 * max(cross_before, 1.0e-6) and
        yaw_after <= 1.05 * max(yaw_before, 1.0e-6))

    report = {
        "schema_version": 2,
        "offline_only": True,
        "purpose": (
            "Decide whether configured map likelihood provides a useful bounded "
            "along-track correction beyond current_map_pose"),
        "run_dir": str(run_dir),
        "map_yaml": str(map_yaml),
        "amcl_config": str(amcl_yaml),
        "recorded_scans": len(scans),
        "matched_moving_scans": len(output_rows),
        "scans_with_valid_sampled_returns": len(valid_output_rows),
        "scans_without_valid_sampled_returns": (
            len(output_rows) - len(valid_output_rows)),
        "unmatched_truth_scans": unmatched_truth,
        "unmatched_current_map_pose_scans": unmatched_pose,
        "speed_filter_mps": MIN_SPEED_MPS,
        "train_scans": len(train),
        "holdout_scans": len(holdout),
        "chronological_split_time_s": split_time,
        "source_scan_timing": source_timing,
        "fit_gain_on_train": fitted_alpha,
        "holdout_baseline": baseline,
        "holdout_candidate": candidate,
        "holdout_along_p95_relative_change": (
            along_after / along_before - 1.0 if along_before > 0.0 else None),
        "holdout_cross_p95_relative_change": (
            cross_after / cross_before - 1.0 if cross_before > 0.0 else None),
        "holdout_yaw_p95_relative_change": (
            yaw_after / yaw_before - 1.0 if yaw_before > 0.0 else None),
        "along_score_curvature_5cm": _summary(
            abs(float(row["along_score_curvature_5cm"]))
            for row in valid_output_rows),
        "zero_score_gain_fraction": float(np.mean([
            float(row["best_score_gain"]) <= 1.0e-6
            for row in valid_output_rows])),
        "search_boundary_fraction": float(np.mean([
            float(row["best_offset_at_search_boundary"])
            for row in valid_output_rows])),
        "decision": (
            "candidate_warrants_matched_live_ab" if candidate_warrants_live_ab
            else "reject_no_heldout_gain_do_not_repeat_capture"),
        "data_use": (
            "A passing candidate must next be implemented in AMCL and verified "
            "in a matched live A/B; this offline report alone is not promotion."),
        "analysis_csv": str(output_csv),
    }
    report_path = output_csv.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path,
                        help="recorder output directory with raw LiDAR sidecar")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--amcl-config", type=Path, default=DEFAULT_AMCL_CONFIG)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_dir, args.map, args.amcl_config, args.output_csv)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
