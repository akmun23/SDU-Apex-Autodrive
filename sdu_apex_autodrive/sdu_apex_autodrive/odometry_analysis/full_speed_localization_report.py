"""Score a source-stamped full-speed run without feeding truth to runtime.

The model-ID timing recorder stores sensor/runtime events while the bridge
diagnostic stream stores simulator truth in the same packet order.  This
offline report joins those streams by the source header stamp and reports the
state/timing gates from the MPC recovery handoff.  Simulator truth is never
read by a runtime node or controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _yaw_from_quaternion(payload: dict[str, object]) -> float | None:
    x = _finite(payload.get("simulator_orientation_quaternion_x"))
    y = _finite(payload.get("simulator_orientation_quaternion_y"))
    z = _finite(payload.get("simulator_orientation_quaternion_z"))
    w = _finite(payload.get("simulator_orientation_quaternion_w"))
    if None in (x, y, z, w):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "p50": None, "p90": None, "p95": None,
                "p99": None, "max": None, "mean": None, "bias": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] + weight * (ordered[upper] - ordered[lower])

    return {
        "samples": len(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
        "bias": sum(values) / len(values),
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


def _metric(values: list[float], *, absolute: bool = True) -> dict[str, object]:
    finite = [value for value in values if math.isfinite(value)]
    if absolute:
        return _summary([abs(value) for value in finite])
    return _summary(finite)


def build_report(run_dir: str | Path, output_json: str | Path | None = None,
                 output_csv: str | Path | None = None) -> dict[str, object]:
    root = Path(run_dir)
    events = _read_events(root / "events.csv")
    timing_rows = events.get("/autodrive/roboracer_1/bridge_packet_timing", [])
    imu_rows = events.get("/autodrive/roboracer_1/imu", [])
    if len(timing_rows) < 2 or len(timing_rows) != len(imu_rows):
        raise ValueError("bridge timing and IMU packet streams are incomplete")

    truth_by_stamp: dict[str, dict[str, object]] = {}
    timing_dts: list[float] = []
    command_lags: list[float] = []
    previous_time: float | None = None
    sequence_gaps = 0
    previous_sequence: int | None = None
    for timing, imu in zip(timing_rows, imu_rows):
        bridge = _payload(timing)
        source_stamp = imu.get("header_stamp_ns", "")
        truth = {
            "x_m": _finite(bridge.get("simulator_position_x")),
            "y_m": _finite(bridge.get("simulator_position_y")),
            "yaw_rad": _yaw_from_quaternion(bridge),
            "u_mps": _finite(bridge.get("simulator_linear_velocity_x")),
            "v_mps": _finite(bridge.get("simulator_linear_velocity_y")),
            "r_radps": _finite(bridge.get("simulator_angular_velocity_z")),
            "simulation_time_s": _finite(bridge.get("simulation_time_s")),
        }
        if source_stamp:
            truth_by_stamp[source_stamp] = truth
        source_time = truth["simulation_time_s"]
        if isinstance(source_time, float) and previous_time is not None:
            timing_dts.append(source_time - previous_time)
        if isinstance(source_time, float):
            previous_time = source_time
        sequence = _finite(bridge.get("packet_sequence"))
        if sequence is not None:
            integer_sequence = int(sequence)
            if previous_sequence is not None and integer_sequence != previous_sequence + 1:
                sequence_gaps += integer_sequence - previous_sequence - 1
            previous_sequence = integer_sequence
        request = _finite(bridge.get("request_sequence"))
        applied = _finite(bridge.get("applied_command_sequence"))
        if request is not None and applied is not None:
            command_lags.append(request - applied)

    first_truth = next(iter(truth_by_stamp.values()))
    initial_yaw = first_truth["yaw_rad"]
    if not isinstance(initial_yaw, float):
        raise ValueError("ground-truth stream has no finite initial yaw")
    initial_x = float(first_truth["x_m"] or 0.0)
    initial_y = float(first_truth["y_m"] or 0.0)

    report_rows: list[dict[str, object]] = []
    for topic in ("/odom", "/ekf_odom", "/amcl_pose", "/current_map_pose"):
        for row in events.get(topic, []):
            stamp = row.get("header_stamp_ns", "")
            truth = truth_by_stamp.get(stamp)
            if truth is None:
                continue
            payload = _payload(row)
            x = _finite(payload.get("x_m"))
            y = _finite(payload.get("y_m"))
            if None in (x, y, truth["x_m"], truth["y_m"]):
                continue
            if topic in ("/odom", "/ekf_odom"):
                # The local odom frame is initialized body-aligned at the
                # first coherent packet.  Convert it only for offline scoring.
                dx = x * math.cos(initial_yaw) - y * math.sin(initial_yaw)
                dy = x * math.sin(initial_yaw) + y * math.cos(initial_yaw)
                estimate_x = initial_x + dx
                estimate_y = initial_y + dy
                local_yaw = _finite(payload.get("yaw_rad"))
                estimate_yaw = (
                    _angle_diff(initial_yaw + local_yaw, 0.0)
                    if local_yaw is not None else None)
            else:
                estimate_x = x
                estimate_y = y
                estimate_yaw = _finite(payload.get("yaw_rad"))
            report_rows.append({
                "topic": topic,
                "source_stamp_ns": stamp,
                "truth_x_m": truth["x_m"],
                "truth_y_m": truth["y_m"],
                "estimate_x_m": estimate_x,
                "estimate_y_m": estimate_y,
                "position_error_m": math.hypot(
                    estimate_x - float(truth["x_m"]),
                    estimate_y - float(truth["y_m"])),
                "estimate_yaw_rad": estimate_yaw,
                "truth_yaw_rad": truth["yaw_rad"],
                "u_error_mps": (
                    _finite(payload.get("speed_mps")) - float(truth["u_mps"])
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("speed_mps")) is not None and
                    isinstance(truth["u_mps"], float) else None),
                "v_error_mps": (
                    _finite(payload.get("lateral_speed_mps")) - float(truth["v_mps"])
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("lateral_speed_mps")) is not None and
                    isinstance(truth["v_mps"], float) else None),
                "r_error_radps": (
                    _finite(payload.get("yaw_rate_radps")) - float(truth["r_radps"])
                    if topic in ("/odom", "/ekf_odom") and
                    _finite(payload.get("yaw_rate_radps")) is not None and
                    isinstance(truth["r_radps"], float) else None),
            })

    metrics: dict[str, object] = {}
    for topic in ("/odom", "/ekf_odom", "/amcl_pose", "/current_map_pose"):
        rows = [row for row in report_rows if row["topic"] == topic]
        metrics[topic] = {
            "position_error_m": _metric([
                float(row["position_error_m"]) for row in rows]),
            "yaw_error_rad": _metric([
                _angle_diff(float(row["estimate_yaw_rad"]),
                            float(row["truth_yaw_rad"]))
                for row in rows
                if isinstance(row["estimate_yaw_rad"], float) and
                isinstance(row["truth_yaw_rad"], float)]),
            "u_error_mps": _metric([
                float(row["u_error_mps"]) for row in rows
                if isinstance(row["u_error_mps"], float)]),
            "v_error_mps": _metric([
                float(row["v_error_mps"]) for row in rows
                if isinstance(row["v_error_mps"], float)]),
            "r_error_radps": _metric([
                float(row["r_error_radps"]) for row in rows
                if isinstance(row["r_error_radps"], float)]),
        }

    report: dict[str, object] = {
        "schema_version": 1,
        "run_dir": str(root),
        "offline_only": True,
        "future_ground_truth_used": False,
        "source_timing": {
            "packets": len(timing_rows),
            "dt_s": _metric(timing_dts, absolute=False),
            "sequence_gaps": sequence_gaps,
            "command_lag_packets": _metric(command_lags, absolute=False),
        },
        "metrics": metrics,
        "acceptance_targets": {
            "current_map_position_p95_m": 0.05,
            "current_map_yaw_p95_rad": 0.02,
            "odom_u_p95_mps": 0.10,
            "odom_v_p95_mps": 0.10,
            "odom_r_p95_radps": 0.05,
        },
        "acceptance": {
            "transport_sequence_clean": sequence_gaps == 0,
            "localization_targets_available": bool(report_rows),
            "full_speed_localization_pass": (
                metrics.get("/current_map_pose", {}).get("position_error_m", {}).get("p95")
                is not None and
                float(metrics["/current_map_pose"]["position_error_m"]["p95"]) < 0.05),
        },
    }

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({field for row in report_rows for field in row})
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report_rows)
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.run_dir, args.output_json, args.output_csv)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
