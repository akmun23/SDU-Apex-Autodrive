#!/usr/bin/env python3
"""Build canonical-sector metrics from complete P0 tracking runs."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.racing.track_projection import TrackProjection  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    position = (len(clean) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return clean[low] + (clean[high] - clean[low]) * (position - low)


def interpolate_periodic(x: float, xs: list[float], ys: list[float], period: float) -> float:
    x %= period
    position = bisect.bisect_right(xs, x)
    if position == 0:
        x0, x1 = xs[-1] - period, xs[0]
        y0, y1 = ys[-1], ys[0]
    elif position == len(xs):
        x0, x1 = xs[-1], xs[0] + period
        y0, y1 = ys[-1], ys[0]
    else:
        x0, x1 = xs[position - 1], xs[position]
        y0, y1 = ys[position - 1], ys[position]
    if x < x0:
        x += period
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def canonical_sector_intervals(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:
    period = float(config["period_m"])
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("sector period must be finite and positive")
    sectors = config["sectors"]
    if not sectors:
        raise ValueError("no sectors configured")
    intervals: list[dict[str, Any]] = []
    boundaries: set[float] = set()
    ids: set[str] = set()
    for sector in sectors:
        sector_id = str(sector["id"])
        if sector_id in ids:
            raise ValueError(f"duplicate sector id: {sector_id}")
        ids.add(sector_id)
        start = float(sector["start_track_s_m"]) % period
        end = float(sector["end_track_s_m"]) % period
        if math.isclose(start, end, abs_tol=1e-9):
            raise ValueError(f"sector {sector_id} has zero/full-loop ambiguity")
        boundaries.update((start, end))
        if end > start:
            intervals.append({"id": sector_id, "start": start, "end": end, "phase": sector["phase"]})
        else:
            intervals.append({"id": sector_id, "start": start, "end": period, "phase": sector["phase"]})
            intervals.append({"id": sector_id, "start": 0.0, "end": end, "phase": sector["phase"]})
    ordered = sorted(intervals, key=lambda item: (item["start"], item["end"]))
    cursor = 0.0
    for item in ordered:
        if item["start"] > cursor + 1e-6:
            raise ValueError(f"sector coverage gap [{cursor:.6f}, {item['start']:.6f})")
        if item["start"] < cursor - 1e-6:
            raise ValueError(f"overlapping sector intervals at {item['start']:.6f}")
        cursor = item["end"]
    if abs(cursor - period) > 1e-6:
        raise ValueError(f"sector coverage ends at {cursor:.6f}, expected {period:.6f}")
    return intervals, sorted(boundaries)


def sector_at(progress: float, period: float, intervals: list[dict[str, Any]]) -> str:
    position = progress % period
    for item in intervals:
        if item["start"] - 1e-9 <= position < item["end"] - 1e-9:
            return item["id"]
    # Assign the exact period endpoint to the first interval beginning at zero.
    for item in intervals:
        if math.isclose(position, item["end"], abs_tol=1e-9) and math.isclose(item["end"], period, abs_tol=1e-9):
            return item["id"]
    raise ValueError(f"no sector covers canonical progress {position:.9f} m")


def add_interval_time(
    totals: dict[str, float],
    u0: float,
    u1: float,
    dt: float,
    period: float,
    intervals: list[dict[str, Any]],
    boundaries: list[float],
) -> None:
    if dt <= 0.0:
        return
    if u1 <= u0 + 1e-10:
        totals[sector_at((u0 + u1) * 0.5, period, intervals)] += dt
        return
    distance = u1 - u0
    cursor = u0
    while cursor < u1 - 1e-10:
        sector_id = sector_at(cursor + min(1e-8, (u1 - cursor) * 0.25), period, intervals)
        modulo = cursor % period
        next_distances = [((boundary - modulo) % period) for boundary in boundaries]
        next_distances = [value for value in next_distances if value > 1e-8]
        step = min(next_distances, default=period)
        end = min(u1, cursor + step)
        if end <= cursor + 1e-10:
            raise ValueError("sector interval integration did not advance")
        totals[sector_id] += dt * (end - cursor) / distance
        cursor = end


def unwrap_rows(rows: list[dict[str, str]], projection: TrackProjection, period: float) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    previous_segment: int | None = None
    previous_modulo: float | None = None
    offset = 0.0
    previous_unwrapped: float | None = None
    for row in rows:
        x, y, t = number(row, "x_m"), number(row, "y_m"), number(row, "time_s")
        if x is None or y is None or t is None:
            continue
        estimate_yaw = number(row, "yaw_rate_radps") or 0.0
        # The report has yaw rate rather than pose heading. Projection is spatial;
        # a zero heading is used solely to select the nearest centerline segment.
        pose = projection.project(
            x, y, estimate_yaw,
            previous_segment=previous_segment,
            local_search_radius=24 if previous_segment is not None else None,
        )
        previous_segment = pose.segment_index
        modulo = pose.s_m
        if previous_modulo is not None:
            delta = modulo - previous_modulo
            if delta < -period * 0.5:
                offset += period
            elif delta > period * 0.5:
                offset -= period
        unwrapped = modulo + offset
        previous_modulo = modulo
        previous_unwrapped = unwrapped
        item: dict[str, Any] = dict(row)
        item["canonical_s_m"] = modulo
        item["unwrapped_s_m"] = unwrapped
        item["projection_distance_m"] = pose.distance_m
        projected.append(item)
    return projected


def sector_sample_summary(
    rows: list[dict[str, Any]],
    interval: dict[str, Any],
    schedule_s: list[float],
    schedule_speed: list[float],
    period: float,
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    selected = [row for row in rows if row["sector_id"] == interval["id"]]
    for row in selected:
        for key in (
            "truth_speed_mps", "reference_speed_mps", "speed_error_mps",
            "steering_command_rad", "steering_feedback_rad", "yaw_rate_radps",
            "ay_body_mps2", "tracking_error_m", "minimum_wall_clearance_m",
            "localization_normal_error_m",
        ):
            value = number(row, key)
            if value is not None:
                values[key].append(value)
    speeds = values["truth_speed_mps"]
    ref_speeds = values["reference_speed_mps"]
    steer_cmd = [abs(value) for value in values["steering_command_rad"]]
    steer_fb = [abs(value) for value in values["steering_feedback_rad"]]
    cte = [abs(value) for value in values["tracking_error_m"]]
    signed_speed_error = values["speed_error_mps"]
    phase_pairs = [
        (float(row["canonical_s_m"]), float(row["truth_speed_mps"]))
        for row in selected
        if number(row, "truth_speed_mps") is not None
    ]
    reference_range = max(
        (interpolate_periodic(s + shift, schedule_s, schedule_speed, period)
         for s, _ in phase_pairs for shift in (-0.75, 0.75)),
        default=0.0,
    ) - min(
        (interpolate_periodic(s + shift, schedule_s, schedule_speed, period)
         for s, _ in phase_pairs for shift in (-0.75, 0.75)),
        default=0.0,
    )
    best_reference_shift = None
    if len(phase_pairs) >= 8 and reference_range >= 0.20:
        phase_scores = []
        for step in range(-40, 41):
            shift = step * 0.025
            residuals = [
                measured - interpolate_periodic(s + shift, schedule_s, schedule_speed, period)
                for s, measured in phase_pairs
            ]
            phase_scores.append((statistics.fmean(value * value for value in residuals), shift))
        best_reference_shift = min(phase_scores)[1]
    return {
        "samples": len(selected),
        "entry_speed_mps": speeds[0] if speeds else None,
        "minimum_speed_mps": min(speeds) if speeds else None,
        "exit_speed_mps": speeds[-1] if speeds else None,
        "reference_speed_mean_mps": statistics.fmean(ref_speeds) if ref_speeds else None,
        "speed_error_mean_mps": statistics.fmean(signed_speed_error) if signed_speed_error else None,
        "speed_error_abs_p95_mps": percentile([abs(value) for value in signed_speed_error], 0.95),
        "best_reference_coordinate_shift_m": best_reference_shift,
        "peak_abs_steering_command_rad": max(steer_cmd) if steer_cmd else None,
        "peak_abs_steering_feedback_rad": max(steer_fb) if steer_fb else None,
        "steering_rate_abs_p95_radps": percentile(
            [abs(float(row["steering_rate_radps"])) for row in selected if row.get("steering_rate_radps") is not None], 0.95
        ),
        "steering_rate_abs_peak_radps": max(
            (abs(float(row["steering_rate_radps"])) for row in selected if row.get("steering_rate_radps") is not None),
            default=None,
        ),
        "peak_abs_yaw_rate_radps": max((abs(v) for v in values["yaw_rate_radps"]), default=None),
        "peak_abs_lateral_accel_mps2": max((abs(v) for v in values["ay_body_mps2"]), default=None),
        "cte_abs_p95_m": percentile(cte, 0.95),
        "wall_clearance_p05_m": percentile(values["minimum_wall_clearance_m"], 0.05),
        "wall_clearance_min_m": min(values["minimum_wall_clearance_m"], default=None),
        "localization_normal_abs_p95_m": percentile(
            [abs(value) for value in values["localization_normal_error_m"]], 0.95
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sectors", type=Path, required=True)
    parser.add_argument("--dual-reference-dir", type=Path, required=True)
    parser.add_argument("--run-report-dir", type=Path, action="append", required=True)
    parser.add_argument("--slip-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scored-lap-min", type=int, default=2)
    parser.add_argument("--scored-lap-max", type=int, default=11)
    args = parser.parse_args()

    sectors_path = args.sectors if args.sectors.is_absolute() else ROOT / args.sectors
    reference_dir = args.dual_reference_dir if args.dual_reference_dir.is_absolute() else ROOT / args.dual_reference_dir
    report_dirs = [p if p.is_absolute() else ROOT / p for p in args.run_report_dir]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    slip_path = args.slip_features if args.slip_features is None or args.slip_features.is_absolute() else ROOT / args.slip_features
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    config = yaml.safe_load(sectors_path.read_text(encoding="utf-8"))
    intervals, boundaries = canonical_sector_intervals(config)
    period = float(config["period_m"])
    ref_manifest = json.loads((reference_dir / "dual_reference_manifest.json").read_text(encoding="utf-8"))
    if not math.isclose(float(ref_manifest["canonical_track_length_m"]), period, rel_tol=0.0, abs_tol=1e-5):
        parser.error("sector period does not match dual-reference manifest")
    projection = TrackProjection.from_csv(
        ROOT / ref_manifest["canonical_centerline"], closed=True
    )
    if not math.isclose(projection.total_length, period, rel_tol=0.0, abs_tol=1e-5):
        parser.error("sector period does not match centerline geometry")
    geometry_rows = read_csv(reference_dir / "steering_path.csv")
    geometry_path_s = [float(point["path_s_m"]) for point in geometry_rows]
    geometry_track_s = [float(point["track_s_m"]) for point in geometry_rows]
    path_period = geometry_path_s[-1] + math.hypot(
        float(geometry_rows[-1]["x_m"]) - float(geometry_rows[0]["x_m"]),
        float(geometry_rows[-1]["y_m"]) - float(geometry_rows[0]["y_m"]),
    )
    schedule_rows = read_csv(reference_dir / "speed_schedule.csv")
    schedule_s = [float(row["track_s_m"]) for row in schedule_rows]
    schedule_speed = [float(row["target_speed_mps"]) for row in schedule_rows]

    slip_by_run_lap_sector: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    slip_burst_by_run_lap_sector: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    if slip_path is not None:
        for row in read_csv(slip_path):
            run_id = row.get("run_id", "")
            lap = row.get("lap_count", "")
            track_s = number(row, "track_s_m")
            if track_s is None or not (args.scored_lap_min <= int(lap or -1) <= args.scored_lap_max):
                continue
            sid = sector_at(track_s, period, intervals)
            mismatch = number(row, "wheel_minus_truth_u_mps")
            if mismatch is not None and row.get("valid_for_acceleration") == "True":
                slip_by_run_lap_sector[(run_id, lap, sid)].append(abs(mismatch))
            burst = number(row, "wheel_burst_rejected")
            if burst is not None:
                slip_burst_by_run_lap_sector[(run_id, lap, sid)].append(burst)

    sector_lap_rows: list[dict[str, Any]] = []
    data_coverage: list[dict[str, Any]] = []
    for report_dir in report_dirs:
        tracking_path = report_dir / "tracking_error.csv"
        controller_path = report_dir / "controller_saturation.csv"
        if not tracking_path.is_file() or not controller_path.is_file():
            parser.error(f"missing tracking/controller report under {report_dir}")
        run_rows = read_csv(tracking_path)
        run_id = run_rows[0].get("run_id", report_dir.parent.parent.name) if run_rows else report_dir.parent.parent.name
        scored = [
            row for row in run_rows
            if row.get("lap_count", "").isdigit()
            and args.scored_lap_min <= int(row["lap_count"]) <= args.scored_lap_max
        ]
        lap_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in scored:
            lap_groups[row["lap_count"]].append(row)
        controller_rows = read_csv(controller_path)
        event_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in controller_rows:
            if row.get("lap_number", "").isdigit():
                event_groups[row["lap_number"]].append(row)

        projected_by_lap: dict[str, list[dict[str, Any]]] = {}
        lap_time_sums: dict[str, dict[str, float]] = {}
        for lap, raw_rows in sorted(lap_groups.items(), key=lambda item: int(item[0])):
            raw_rows.sort(key=lambda row: float(row["time_s"]))
            rows = unwrap_rows(raw_rows, projection, period)
            if len(rows) < 3:
                continue
            # Normalize the first point to its canonical coordinate; each scored
            # lap should then span approximately one complete centerline period.
            u0 = float(rows[0]["unwrapped_s_m"])
            for row in rows:
                row["unwrapped_s_m"] = float(row["unwrapped_s_m"]) - u0 + float(rows[0]["canonical_s_m"])
                row["sector_id"] = sector_at(float(row["canonical_s_m"]), period, intervals)
                row["steering_rate_radps"] = None
            for previous, current in zip(rows, rows[1:]):
                dt = float(current["time_s"]) - float(previous["time_s"])
                first_steer = number(previous, "steering_feedback_rad")
                second_steer = number(current, "steering_feedback_rad")
                if 0.005 <= dt <= 0.10 and first_steer is not None and second_steer is not None:
                    current["steering_rate_radps"] = (second_steer - first_steer) / dt
            totals = {str(item["id"]): 0.0 for item in config["sectors"]}
            for first, second in zip(rows, rows[1:]):
                dt = float(second["time_s"]) - float(first["time_s"])
                if dt <= 0.0 or dt > 0.15:
                    continue
                add_interval_time(
                    totals,
                    float(first["unwrapped_s_m"]),
                    float(second["unwrapped_s_m"]),
                    dt,
                    period,
                    intervals,
                    boundaries,
                )
            projected_by_lap[lap] = rows
            lap_time_sums[lap] = totals
            span = float(rows[-1]["unwrapped_s_m"]) - float(rows[0]["unwrapped_s_m"])
            elapsed = float(rows[-1]["time_s"]) - float(rows[0]["time_s"])
            data_coverage.append({"run_id": run_id, "lap_count": int(lap), "sample_count": len(rows), "canonical_span_m": span, "elapsed_s": elapsed, "sector_time_sum_s": sum(totals.values())})

        for lap, rows in projected_by_lap.items():
            lap_events = event_groups.get(lap, [])
            for sector in config["sectors"]:
                sid = str(sector["id"])
                local_rows = [row for row in rows if row["sector_id"] == sid]
                if not local_rows:
                    raise ValueError(f"{run_id} lap {lap} has no samples in {sid}")
                metrics = sector_sample_summary(rows, {"id": sid}, schedule_s, schedule_speed, period)
                local_events: list[dict[str, str]] = []
                for event in lap_events:
                    progress = number(event, "progress_m")
                    if progress is None:
                        continue
                    path_s = progress % path_period
                    pos = bisect.bisect_right(geometry_path_s, path_s)
                    lo, hi = pos - 1, pos
                    if lo < 0:
                        lo, hi = len(geometry_path_s) - 1, 0
                        left_s, right_s = geometry_path_s[lo] - path_period, geometry_path_s[hi]
                        query_s = path_s - path_period if path_s > geometry_path_s[-1] else path_s
                        ratio = (query_s - left_s) / (right_s - left_s)
                        track_s = geometry_track_s[lo] + ratio * (geometry_track_s[hi] + period - geometry_track_s[lo])
                    elif hi >= len(geometry_path_s):
                        hi = 0
                        left_s, right_s = geometry_path_s[lo], path_period
                        ratio = (path_s - left_s) / (right_s - left_s)
                        track_s = geometry_track_s[lo] + ratio * (geometry_track_s[hi] + period - geometry_track_s[lo])
                    else:
                        ratio = (path_s - geometry_path_s[lo]) / (geometry_path_s[hi] - geometry_path_s[lo])
                        track_s = geometry_track_s[lo] + ratio * (geometry_track_s[hi] - geometry_track_s[lo])
                    if sector_at(track_s, period, intervals) == sid:
                        local_events.append(event)
                statuses = [event.get("status", "") for event in local_events]
                rejection_count = sum(status != "accepted_optimal" for status in statuses)
                repair_count = sum(event.get("corridor_repair", "").lower() == "true" for event in local_events)
                best_effort_count = sum(event.get("best_effort_action", "").lower() == "true" for event in local_events)
                rti2_count = sum(event.get("rti2_triggered", "").lower() == "true" for event in local_events)
                sl = slip_by_run_lap_sector.get((run_id.replace("practice_9g_parent_reproduced_20261006_", "p0_"), lap, sid), [])
                # Also accept canonical run IDs recorded in the replay analyzer.
                if not sl:
                    aliases = {"practice_9g_parent_reproduced_20261006_r01": "p0_r01", "practice_9g_parent_reproduced_20261006_r02": "p0_r02"}
                    sl = slip_by_run_lap_sector.get((aliases.get(run_id, run_id), lap, sid), [])
                burst_values = slip_burst_by_run_lap_sector.get(("p0_" + run_id[-3:], lap, sid), [])
                sector_lap_rows.append({
                    "run_id": run_id,
                    "lap_count": int(lap),
                    "sector": sid,
                    "phase": sector["phase"],
                    "sector_start_track_s_m": sector["start_track_s_m"],
                    "sector_end_track_s_m": sector["end_track_s_m"],
                    "sector_time_s": lap_time_sums[lap][sid],
                    **metrics,
                    "abs_wheel_minus_truth_u_mps_p95": percentile(sl, 0.95),
                    "slip_valid_samples": len(sl),
                    "wheel_burst_rejected_fraction": statistics.fmean(burst_values) if burst_values else None,
                    "mpc_reject_count": rejection_count,
                    "corridor_repair_count": repair_count,
                    "best_effort_count": best_effort_count,
                    "rti2_count": rti2_count,
                    "mpc_samples": len(local_events),
                })

    if not sector_lap_rows:
        parser.error("no scored sector rows were produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(sector_lap_rows[0])
    write_csv(output_dir / "sector_lap_metrics.csv", sector_lap_rows, fields)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sector_lap_rows:
        grouped[(row["run_id"], row["sector"])].append(row)
    run_sector_rows: list[dict[str, Any]] = []
    for (run_id, sid), rows in sorted(grouped.items()):
        durations = [float(row["sector_time_s"]) for row in rows]
        run_sector_rows.append({
            "run_id": run_id,
            "sector": sid,
            "laps": len(rows),
            "sector_time_mean_s": statistics.fmean(durations),
            "sector_time_median_s": statistics.median(durations),
            "sector_time_p10_s": percentile(durations, 0.10),
            "sector_time_p90_s": percentile(durations, 0.90),
            "entry_speed_mean_mps": statistics.fmean(float(row["entry_speed_mps"]) for row in rows if row["entry_speed_mps"] is not None),
            "minimum_speed_mean_mps": statistics.fmean(float(row["minimum_speed_mps"]) for row in rows if row["minimum_speed_mps"] is not None),
            "exit_speed_mean_mps": statistics.fmean(float(row["exit_speed_mps"]) for row in rows if row["exit_speed_mps"] is not None),
            "steering_peak_p95_rad": percentile([float(row["peak_abs_steering_feedback_rad"]) for row in rows if row["peak_abs_steering_feedback_rad"] is not None], 0.95),
            "cte_abs_p95_m": percentile([float(row["cte_abs_p95_m"]) for row in rows if row["cte_abs_p95_m"] is not None], 0.95),
            "wall_clearance_min_m": min((float(row["wall_clearance_min_m"]) for row in rows if row["wall_clearance_min_m"] is not None), default=None),
            "mpc_rejects_total": sum(int(row["mpc_reject_count"]) for row in rows),
        })
    write_csv(output_dir / "run_sector_summary.csv", run_sector_rows, list(run_sector_rows[0]))
    write_csv(output_dir / "lap_coverage.csv", data_coverage, list(data_coverage[0]))
    sector_ids = [str(item["id"]) for item in config["sectors"]]
    sector_budget_rows: list[dict[str, Any]] = []
    run_means: dict[str, dict[str, float]] = defaultdict(dict)
    for row in run_sector_rows:
        run_means[row["sector"]][row["run_id"]] = float(row["sector_time_mean_s"])
    equal_run_mean_by_sector = {
        sid: statistics.fmean(run_means[sid].values()) for sid in sector_ids
    }
    tracked_sector_sum = sum(equal_run_mean_by_sector.values())
    official_lap_times: list[float] = []
    for report_dir in report_dirs:
        for lap in read_csv(report_dir / "lap_times.csv"):
            if lap.get("lap_role") == "scored" and lap.get("complete", "").lower() == "true":
                time_value = number(lap, "lap_time_s")
                if time_value is not None:
                    official_lap_times.append(time_value)
    for sector in config["sectors"]:
        sid = str(sector["id"])
        observations = [row for row in sector_lap_rows if row["sector"] == sid]
        run_values = run_means[sid]
        sector_mean = equal_run_mean_by_sector[sid]
        sector_budget_rows.append({
            "sector": sid,
            "phase": sector["phase"],
            "start_track_s_m": sector["start_track_s_m"],
            "end_track_s_m": sector["end_track_s_m"],
            "run_count": len(run_values),
            "p0_run_means_s": ";".join(f"{value:.6f}" for _, value in sorted(run_values.items())),
            "p0_equal_run_mean_s": sector_mean,
            "p0_lap_p10_s": percentile([float(row["sector_time_s"]) for row in observations], 0.10),
            "p0_lap_p90_s": percentile([float(row["sector_time_s"]) for row in observations], 0.90),
            "p0_best_observed_s": min(float(row["sector_time_s"]) for row in observations),
            "p0_cte_abs_p95_m": percentile([float(row["cte_abs_p95_m"]) for row in observations if row["cte_abs_p95_m"] is not None], 0.95),
            "minimum_wall_clearance_m": min(float(row["wall_clearance_min_m"]) for row in observations if row["wall_clearance_min_m"] is not None),
            "mpc_rejections": sum(int(row["mpc_reject_count"]) for row in observations),
            "proportional_4p94_target_s": 4.94 * sector_mean / tracked_sector_sum,
        })
    write_csv(output_dir / "sector_time_budget.csv", sector_budget_rows, list(sector_budget_rows[0]))
    summary = {
        "schema_version": 1,
        "parent": config.get("parent"),
        "baseline_runs": sorted({row["run_id"] for row in sector_lap_rows}),
        "scored_lap_count_per_run": {run: sum(row["run_id"] == run for row in sector_lap_rows) // len(config["sectors"]) for run in {row["run_id"] for row in sector_lap_rows}},
        "sector_count": len(config["sectors"]),
        "canonical_period_m": period,
        "official_scored_lap_mean_s": statistics.fmean(official_lap_times) if official_lap_times else None,
        "sector_occupancy_mean_sum_s": tracked_sector_sum,
        "external_4p94_target_is_proportional_allocation": True,
        "sum_sector_durations_match_lap_elapsed": all(
            abs(float(row["sector_time_sum_s"]) - float(row["elapsed_s"])) < 0.10
            for row in data_coverage
        ),
        "lap_coverage": data_coverage,
        "outputs": ["sector_lap_metrics.csv", "run_sector_summary.csv", "lap_coverage.csv", "sector_time_budget.csv"],
        "caveats": [
            "Runs remain the independent comparison units; scored laps are repeated observations within each run.",
            "Track truth is used only offline for projection and evaluation.",
            "Wheel/body mismatch is an offline truth-labeled diagnostic, not a runtime control signal.",
            "best_reference_coordinate_shift_m minimizes local speed-profile error under a spatial shift; it is descriptive, not causal.",
            "The 4.94 s sector budget is a proportional target allocation, not a predicted achievable lap.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("parent", "baseline_runs", "scored_lap_count_per_run", "sector_count", "canonical_period_m", "sum_sector_durations_match_lap_elapsed", "outputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
