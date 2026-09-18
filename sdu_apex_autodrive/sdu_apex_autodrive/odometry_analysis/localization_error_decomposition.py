"""Decompose recorded localization error in the track Frenet frame.

This is an offline-only analysis.  It joins the existing source-stamped
localization report with the saved raceline and the bridge timing stream.  No
simulator truth is available to, or consumed by, a runtime node.

The decomposition uses the nearest raceline sample to the ground-truth pose as
the local track reference.  For each estimator it reports:

* signed cross-track and along-track position error;
* wrapped raceline ``s`` error;
* yaw error, source-time speed, and raceline curvature;
* summaries by estimator, track segment, speed, and curvature.

AMCL health and scan-alignment diagnostics are source-stamped inside their
Float64MultiArray payloads and joined to estimator samples within 1 ms. An
along-track gain counterfactual measures the immediate effect of changing the
latest applied scan correction; it is explicitly not a recurrent AMCL replay.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Iterable


TOPICS = ("/odom", "/ekf_odom", "/amcl_pose", "/current_map_pose")
HEALTH_FIELDS = (
    "correction_age_s",
    "correction_accepted",
    "rejected_scans",
    "degraded",
    "xy_variance",
    "yaw_variance",
    "scan_correction_distance_m",
    "scan_correction_yaw_rad",
    "applied_xy_correction_m",
    "applied_yaw_correction_rad",
    "scan_correction_x_m",
    "scan_correction_y_m",
    "applied_x_m",
    "applied_y_m",
    "source_stamp_s",
)
SCAN_FIELDS = (
    "scan_stamp_s",
    "matched_odom_stamp_s",
    "source_error_ms",
    "accepted",
    "queued",
    "dropped",
    "bracket_before_stamp_s",
    "bracket_after_stamp_s",
    "processing_dropped",
    "valid_sampled_beams",
    "sampled_beams",
    "likelihood_offset_m",
    "likelihood_score_gain",
    "likelihood_applied_m",
)


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _summary(values: Iterable[float], *, absolute: bool = False) -> dict[str, object]:
    finite = [value for value in values if math.isfinite(value)]
    if absolute:
        finite = [abs(value) for value in finite]
    if not finite:
        return {
            "samples": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "bias": None,
        }
    ordered = sorted(finite)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] + weight * (ordered[upper] - ordered[lower])

    return {
        "samples": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(ordered),
        "mean": sum(ordered) / len(ordered),
        "bias": sum(finite) / len(finite),
    }


def _read_events(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row.get("topic", ""), []).append(row)
    return grouped


def _payload(row: dict[str, str]) -> dict[str, object]:
    try:
        value = json.loads(row.get("payload_json", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_raceline(path: Path) -> dict[str, list[float] | float]:
    rows: list[tuple[float, float, float, float, float]] = []
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            raw = raw.strip()
            if not raw or raw.startswith("#") or raw.startswith("s_m"):
                continue
            values = raw.split(",")
            if len(values) < 5:
                continue
            rows.append((
                float(values[0]), float(values[1]), float(values[2]),
                float(values[3]), float(values[4]),
            ))
    if len(rows) < 3:
        raise ValueError("raceline has fewer than three numeric samples")
    s = [row[0] for row in rows]
    if any(b <= a for a, b in zip(s, s[1:])):
        raise ValueError("raceline s_m must be strictly increasing")
    spacing = s[-1] - s[-2]
    track_length = s[-1] + spacing
    return {
        "s_m": s,
        "x_m": [row[1] for row in rows],
        "y_m": [row[2] for row in rows],
        "psi_rad": [row[3] for row in rows],
        "kappa_radpm": [row[4] for row in rows],
        "track_length_m": track_length,
    }


def _build_spatial_index(raceline: dict[str, list[float] | float], cell_m: float = 0.5) -> dict[tuple[int, int], list[int]]:
    index: dict[tuple[int, int], list[int]] = {}
    xs = raceline["x_m"]
    ys = raceline["y_m"]
    assert isinstance(xs, list) and isinstance(ys, list)
    for i, (x, y) in enumerate(zip(xs, ys)):
        key = (math.floor(x / cell_m), math.floor(y / cell_m))
        index.setdefault(key, []).append(i)
    return index


def _nearest_raceline(
    x: float,
    y: float,
    raceline: dict[str, list[float] | float],
    spatial_index: dict[tuple[int, int], list[int]],
    cell_m: float = 0.5,
) -> tuple[int, float]:
    """Return nearest raceline index and squared distance."""
    cell = (math.floor(x / cell_m), math.floor(y / cell_m))
    candidates: list[int] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            candidates.extend(spatial_index.get((cell[0] + dx, cell[1] + dy), ()))
    if not candidates:
        xs = raceline["x_m"]
        ys = raceline["y_m"]
        assert isinstance(xs, list) and isinstance(ys, list)
        candidates = range(len(xs))
    xs = raceline["x_m"]
    ys = raceline["y_m"]
    assert isinstance(xs, list) and isinstance(ys, list)
    best = min(candidates, key=lambda i: (xs[i] - x) ** 2 + (ys[i] - y) ** 2)
    return best, (xs[best] - x) ** 2 + (ys[best] - y) ** 2


def _source_truth(events: dict[str, list[dict[str, str]]]) -> tuple[dict[str, dict[str, float]], list[float]]:
    timing = events.get("/autodrive/roboracer_1/bridge_packet_timing", [])
    imu = events.get("/autodrive/roboracer_1/imu", [])
    if len(timing) < 2 or len(timing) != len(imu):
        raise ValueError("bridge timing and IMU streams are incomplete")
    truth: dict[str, dict[str, float]] = {}
    source_times: list[float] = []
    for timing_row, imu_row in zip(timing, imu):
        bridge = _payload(timing_row)
        stamp = timing_row.get("header_stamp_ns", "") or imu_row.get("header_stamp_ns", "")
        if not stamp:
            continue
        values = {
            "x_m": _finite(bridge.get("simulator_position_x")),
            "y_m": _finite(bridge.get("simulator_position_y")),
            "u_mps": _finite(bridge.get("simulator_linear_velocity_x")),
            "v_mps": _finite(bridge.get("simulator_linear_velocity_y")),
            "source_time_s": _finite(bridge.get("simulation_time_s")),
        }
        if values["x_m"] is None or values["y_m"] is None:
            continue
        truth[stamp] = {key: float(value) for key, value in values.items() if value is not None}
        if values["source_time_s"] is not None:
            source_times.append(float(values["source_time_s"]))
    return truth, source_times


def _report_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bin_label(value: float, edges: list[float], *, prefix: str) -> str:
    index = bisect.bisect_right(edges, value) - 1
    index = max(0, min(index, len(edges) - 2))
    lower, upper = edges[index], edges[index + 1]
    upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
    return f"{prefix}[{lower:g},{upper_label})"


def _metric_block(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "samples": len(rows),
        "cross_track_error_m": _summary(
            [float(row["cross_track_error_m"]) for row in rows], absolute=True),
        "along_track_error_m": _summary(
            [float(row["along_track_error_m"]) for row in rows], absolute=True),
        "raceline_s_error_m": _summary(
            [float(row["raceline_s_error_m"]) for row in rows], absolute=True),
        "position_error_m": _summary(
            [float(row["position_error_m"]) for row in rows], absolute=True),
        "yaw_error_rad": _summary(
            [float(row["yaw_error_rad"]) for row in rows
             if isinstance(row.get("yaw_error_rad"), float)], absolute=True),
        "speed_mps": _summary(
            [float(row["speed_mps"]) for row in rows
             if isinstance(row.get("speed_mps"), float)]),
    }
    ages = [float(row["amcl_correction_age_s"]) for row in rows
            if isinstance(row.get("amcl_correction_age_s"), float)]
    if ages:
        result["amcl_correction_age_s"] = _summary(ages)
    return result


def _diagnostic_summary(
    events: dict[str, list[dict[str, str]]],
    topic: str,
    fields: tuple[str, ...],
) -> dict[str, object]:
    values_by_field: dict[str, list[float]] = {field: [] for field in fields}
    rows = []
    for event in events.get(topic, []):
        payload = _payload(event)
        data = payload.get("data")
        if not isinstance(data, list):
            continue
        rows.append(data)
        for index, field in enumerate(fields):
            if index < len(data):
                value = _finite(data[index])
                if value is not None:
                    values_by_field[field].append(value)
    result: dict[str, object] = {"samples": len(rows), "fields": {}}
    result["fields"] = {
        field: _summary(values, absolute=field.endswith(("_s", "_ms", "_m", "_rad")))
        for field, values in values_by_field.items()
    }
    return result


def _source_stamped_health(
    events: dict[str, list[dict[str, str]]],
) -> tuple[list[int], dict[int, float]]:
    """Return AMCL health source stamps and correction ages from schema v2."""
    stamps, values = _source_stamped_health_data(events)
    ages = {
        stamp: age for stamp, data in values.items()
        if (age := _finite(data[0])) is not None
    }
    return stamps, ages


def _source_stamped_health_data(
    events: dict[str, list[dict[str, str]]],
) -> tuple[list[int], dict[int, list[object]]]:
    """Return source-time keyed health payloads (field 15 is the source stamp)."""
    samples: list[tuple[int, list[object]]] = []
    for event in events.get("/amcl_localization_health", []):
        data = _payload(event).get("data")
        if not isinstance(data, list) or len(data) < 15:
            continue
        source_stamp_s = _finite(data[14])
        if source_stamp_s is None:
            continue
        samples.append((int(round(source_stamp_s * 1.0e9)), data))
    samples.sort()
    return [stamp for stamp, _ in samples], dict(samples)


def _nearest_stamped_value(
    stamps: list[int], values: dict[int, float], target_ns: int,
    tolerance_ns: int = 1_000_000,
) -> float | None:
    if not stamps:
        return None
    index = bisect.bisect_left(stamps, target_ns)
    candidates = []
    if index < len(stamps):
        candidates.append(stamps[index])
    if index:
        candidates.append(stamps[index - 1])
    match = min(candidates, key=lambda stamp: abs(stamp - target_ns))
    return values.get(match) if abs(match - target_ns) <= tolerance_ns else None


def _nearest_stamped_row(
    stamps: list[int], values: dict[int, list[object]], target_ns: int,
    tolerance_ns: int = 1_000_000,
) -> list[object] | None:
    if not stamps:
        return None
    index = bisect.bisect_left(stamps, target_ns)
    candidates = []
    if index < len(stamps):
        candidates.append(stamps[index])
    if index:
        candidates.append(stamps[index - 1])
    match = min(candidates, key=lambda stamp: abs(stamp - target_ns))
    return values[match] if abs(match - target_ns) <= tolerance_ns else None


def _amcl_along_track_gain_counterfactual(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Score an immediate 2x/0x along-track correction, not a replay."""
    accepted: list[dict[str, float]] = []
    for row in rows:
        if row.get("topic") != "/current_map_pose":
            continue
        values = {
            "error": _finite(row.get("along_track_error_m")),
            "applied": _finite(row.get("amcl_applied_along_track_m")),
            "accepted": _finite(row.get("amcl_correction_accepted")),
            "distance": _finite(row.get("amcl_scan_correction_distance_m")),
            "speed": _finite(row.get("speed_mps")),
        }
        if any(value is None for value in values.values()):
            continue
        if values["accepted"] < 0.5 or values["distance"] > 1.0:
            continue
        if abs(values["applied"]) < 1.0e-5:
            continue
        accepted.append({key: float(value) for key, value in values.items()})

    groups = {
        "all_accepted_local_scans": accepted,
        "speed_at_least_3_mps": [row for row in accepted if row["speed"] >= 3.0],
    }
    result: dict[str, object] = {}
    for name, samples in groups.items():
        actual = [row["error"] for row in samples]
        no_along_correction = [
            row["error"] - row["applied"] for row in samples]
        double_along_correction = [
            row["error"] + row["applied"] for row in samples]
        result[name] = {
            "samples": len(samples),
            "actual_along_track_error_m": _summary(actual, absolute=True),
            "zero_along_correction_counterfactual_m": _summary(
                no_along_correction, absolute=True),
            "double_applied_along_correction_counterfactual_m": _summary(
                double_along_correction, absolute=True),
            "double_increment_improves_fraction": (
                sum(abs(error + correction) < abs(error)
                    for error, correction in zip(actual,
                                                 [row["applied"] for row in samples])) /
                len(samples) if samples else None),
        }
    return {
        "offline_only": True,
        "method": (
            "For each accepted local scan, add or remove the already-applied "
            "along-track correction from the current error. No later AMCL "
            "state or scan is replayed, so this cannot qualify a gain change."),
        "current_config_effective_gain": (
            "approximately xy_gain * along_track_gain = 0.5 * 0.25 = 0.125 "
            "outside full-pose recovery"),
        "candidate_double_increment": "effective gain approximately 0.25",
        "groups": result,
    }


def build_report(
    run_dir: str | Path,
    report_csv: str | Path,
    raceline_csv: str | Path,
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    segment_count: int = 8,
) -> dict[str, object]:
    root = Path(run_dir)
    events = _read_events(root / "events.csv")
    truth_by_stamp, source_times = _source_truth(events)
    report = _report_rows(Path(report_csv))
    raceline = _read_raceline(Path(raceline_csv))
    spatial_index = _build_spatial_index(raceline)
    xs = raceline["x_m"]
    ys = raceline["y_m"]
    ss = raceline["s_m"]
    psis = raceline["psi_rad"]
    kappas = raceline["kappa_radpm"]
    track_length = float(raceline["track_length_m"])
    assert all(isinstance(value, list) for value in (xs, ys, ss, psis, kappas))

    rows: list[dict[str, object]] = []
    truth_nearest: dict[str, tuple[int, float]] = {}
    for source_stamp, truth in truth_by_stamp.items():
        truth_nearest[source_stamp] = _nearest_raceline(
            truth["x_m"], truth["y_m"], raceline, spatial_index)

    for raw in report:
        topic = raw.get("topic", "")
        stamp = raw.get("source_stamp_ns", "")
        if topic not in TOPICS or stamp not in truth_by_stamp or stamp not in truth_nearest:
            continue
        truth = truth_by_stamp[stamp]
        truth_index, truth_distance_sq = truth_nearest[stamp]
        estimate_x = _finite(raw.get("estimate_x_m"))
        estimate_y = _finite(raw.get("estimate_y_m"))
        truth_x = truth["x_m"]
        truth_y = truth["y_m"]
        if estimate_x is None or estimate_y is None:
            continue
        tangent_x = math.cos(psis[truth_index])
        tangent_y = math.sin(psis[truth_index])
        normal_x = -tangent_y
        normal_y = tangent_x
        dx = estimate_x - truth_x
        dy = estimate_y - truth_y
        estimate_index, _ = _nearest_raceline(
            estimate_x, estimate_y, raceline, spatial_index)
        s_error = ss[estimate_index] - ss[truth_index]
        if s_error > track_length / 2.0:
            s_error -= track_length
        elif s_error < -track_length / 2.0:
            s_error += track_length
        truth_yaw = _finite(raw.get("truth_yaw_rad"))
        estimate_yaw = _finite(raw.get("estimate_yaw_rad"))
        source_speed = math.hypot(truth.get("u_mps", 0.0), truth.get("v_mps", 0.0))
        row: dict[str, object] = {
            "topic": topic,
            "source_stamp_ns": stamp,
            "truth_track_s_m": ss[truth_index],
            "truth_track_heading_rad": psis[truth_index],
            "truth_track_kappa_radpm": kappas[truth_index],
            "truth_raceline_distance_m": math.sqrt(truth_distance_sq),
            "speed_mps": source_speed,
            "estimate_x_m": estimate_x,
            "estimate_y_m": estimate_y,
            "truth_x_m": truth_x,
            "truth_y_m": truth_y,
            "position_error_m": math.hypot(dx, dy),
            "cross_track_error_m": dx * normal_x + dy * normal_y,
            "along_track_error_m": dx * tangent_x + dy * tangent_y,
            "raceline_s_error_m": s_error,
            "yaw_error_rad": (
                _angle_diff(estimate_yaw, truth_yaw)
                if estimate_yaw is not None and truth_yaw is not None else None),
            "truth_raceline_index": truth_index,
            "estimate_raceline_index": estimate_index,
        }
        rows.append(row)

    by_topic: dict[str, object] = {}
    speed_edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, math.inf]
    curvature_edges = [0.0, 0.05, 0.10, 0.20, 0.40, math.inf]
    for topic in TOPICS:
        topic_rows = [row for row in rows if row["topic"] == topic]
        speed_bins: dict[str, list[dict[str, object]]] = {}
        curvature_bins: dict[str, list[dict[str, object]]] = {}
        segment_bins: dict[str, list[dict[str, object]]] = {}
        for row in topic_rows:
            speed_bins.setdefault(_bin_label(float(row["speed_mps"]), speed_edges, prefix="speed_mps"), []).append(row)
            curvature_bins.setdefault(
                _bin_label(abs(float(row["truth_track_kappa_radpm"])), curvature_edges, prefix="abs_kappa_radpm"),
                [],
            ).append(row)
            segment = min(segment_count - 1, int(float(row["truth_track_s_m"]) / track_length * segment_count))
            segment_bins.setdefault(f"segment_{segment:02d}", []).append(row)
        by_topic[topic] = {
            "overall": _metric_block(topic_rows),
            "by_speed": {key: _metric_block(value) for key, value in sorted(speed_bins.items())},
            "by_abs_curvature": {
                key: _metric_block(value) for key, value in sorted(curvature_bins.items())
            },
            "by_track_segment": {
                key: _metric_block(value) for key, value in sorted(segment_bins.items())
            },
        }

    source_dts = [b - a for a, b in zip(source_times, source_times[1:]) if b > a]
    health = _diagnostic_summary(events, "/amcl_localization_health", HEALTH_FIELDS)
    scan_alignment = _diagnostic_summary(events, "/amcl_scan_alignment", SCAN_FIELDS)
    health_stamps, health_by_stamp = _source_stamped_health_data(events)
    health_age_by_stamp = {
        stamp: age for stamp, data in health_by_stamp.items()
        if (age := (_finite(data[0]) if data else None)) is not None
    }
    if health_stamps:
        for row in rows:
            try:
                target_stamp_ns = int(row["source_stamp_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            age = _nearest_stamped_value(
                health_stamps, health_age_by_stamp, target_stamp_ns)
            if age is not None:
                row["amcl_correction_age_s"] = age
            if row.get("topic") != "/current_map_pose":
                continue
            health_data = _nearest_stamped_row(
                health_stamps, health_by_stamp, target_stamp_ns)
            if health_data is None:
                continue
            accepted = _finite(health_data[1]) if len(health_data) > 1 else None
            correction_distance = (
                _finite(health_data[6]) if len(health_data) > 6 else None)
            raw_x = _finite(health_data[10]) if len(health_data) > 10 else None
            raw_y = _finite(health_data[11]) if len(health_data) > 11 else None
            applied_x = _finite(health_data[12]) if len(health_data) > 12 else None
            applied_y = _finite(health_data[13]) if len(health_data) > 13 else None
            track_heading = _finite(row.get("truth_track_heading_rad"))
            if (track_heading is None or raw_x is None or raw_y is None or
                    applied_x is None or applied_y is None):
                continue
            tangent_x = math.cos(track_heading)
            tangent_y = math.sin(track_heading)
            row["amcl_correction_accepted"] = accepted
            row["amcl_scan_correction_distance_m"] = correction_distance
            row["amcl_raw_along_track_innovation_m"] = (
                raw_x * tangent_x + raw_y * tangent_y)
            row["amcl_applied_along_track_m"] = (
                applied_x * tangent_x + applied_y * tangent_y)
    scan_rows = []
    for event in events.get("/amcl_scan_alignment", []):
        data = _payload(event).get("data")
        if isinstance(data, list):
            scan_rows.append(data)
    report_out: dict[str, object] = {
        "schema_version": 2,
        "run_dir": str(root),
        "source_report_csv": str(report_csv),
        "raceline_csv": str(raceline_csv),
        "offline_only": True,
        "future_ground_truth_used": False,
        "reference_definition": {
            "track_frame": "nearest_raceline_to_ground_truth",
            "cross_track_sign": "dot(estimate-truth, [-sin(track_heading), cos(track_heading)])",
            "along_track_sign": "dot(estimate-truth, [cos(track_heading), sin(track_heading)])",
            "raceline_s_error_wrapped_to_half_track_length": True,
            "track_length_m": track_length,
            "segment_count": segment_count,
        },
        "sample_counts": {
            "source_truth_packets": len(truth_by_stamp),
            "localization_decomposition_rows": len(rows),
            "rows_by_topic": {topic: sum(row["topic"] == topic for row in rows) for topic in TOPICS},
        },
        "source_timing": {
            "dt_s": _summary(source_dts),
            "sequence_header_available": False,
        },
        "metrics": by_topic,
        "amcl_along_track_gain_counterfactual":
            _amcl_along_track_gain_counterfactual(rows),
        "amcl_diagnostics": {
            "health": health,
            "scan_alignment": scan_alignment,
            "state_age_limitations": {
                "exact_per_sample_correction_age_available": bool(health_stamps),
                "correction_age_summary_is_diagnostic_only": not bool(health_stamps),
                "source_stamped_health_samples": len(health_stamps),
                "scan_source_error_ms_is_source_stamped": bool(scan_rows),
                "reason": (
                    "AMCL health source stamps are available and joined within 1 ms."
                    if health_stamps else
                    "AMCL health messages have no source header. Their correction age "
                    "cannot be causally attached to a localization row without a "
                    "runtime source-stamped diagnostic contract."
                ),
            },
        },
        "acceptance": {
            "all_requested_topics_present": all(any(row["topic"] == topic for row in rows) for topic in TOPICS),
            "decomposition_complete": bool(rows),
            "state_age_exactly_joined": any(
                "amcl_correction_age_s" in row for row in rows),
        },
    }

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({field for row in rows for field in row})
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report_out, indent=2) + "\n", encoding="utf-8")
    return report_out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report-csv", type=Path, required=True)
    parser.add_argument("--raceline", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--segment-count", type=int, default=8)
    args = parser.parse_args(argv)
    report = build_report(
        args.run_dir,
        args.report_csv,
        args.raceline,
        output_json=args.output_json,
        output_csv=args.output_csv,
        segment_count=args.segment_count,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
