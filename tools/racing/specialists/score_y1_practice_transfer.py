#!/usr/bin/env python3
"""Score the guarded Y1 yaw candidate on existing practice bags, one-step only.

Legacy and Y1 production transitions receive identical recorded MPC
state/action inputs. Simulator truth is used only as the next-packet yaw label.
Rows at and after the first collision are excluded; no simulator is launched
and no sealed model split is opened.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import sqlite3
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.evaluate_open_plane_body_dynamics import load_capture


ROOT = Path(__file__).resolve().parents[3]
DT_S = 0.025
MPC_TOPIC = "/mpc/diagnostics"
COLLISION_TOPIC = "/autodrive/roboracer_1/collision_count"
YAW_TRANSITION_RESIDUAL_BAND = {
    "speed_lower_start_mps": 3.40,
    "speed_lower_full_mps": 3.45,
    "speed_upper_full_mps": 4.00,
    "speed_upper_end_mps": 4.05,
    "steering_lower_start_rad": 0.28,
    "steering_lower_full_rad": 0.30,
    "steering_upper_full_rad": 0.42,
    "steering_upper_end_rad": 0.45,
}


class ModelState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "e_y", "e_psi", "u", "v", "r", "target_speed",
        "steering_command", "delayed_steering_command_1",
        "delayed_steering_command_2", "actual_steering_angle")]


class ModelControl(ctypes.Structure):
    _fields_ = [("steering_rate", ctypes.c_float),
                ("target_speed_rate", ctypes.c_float)]


class StageResult(ctypes.Structure):
    _fields_ = [("next", ModelState), ("delta_s_m", ctypes.c_float),
                ("body_accel_mps2", ctypes.c_float),
                ("branch_flags", ctypes.c_uint),
                ("valid", ctypes.c_int)]


class YawResponseSurface(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("speed_count", ctypes.c_int),
        ("blend_q_start", ctypes.c_float),
        ("blend_q_end", ctypes.c_float),
        ("speed_blend_margin_mps", ctypes.c_float),
        ("low_speed_blend_margin_mps", ctypes.c_float),
        ("low_speed_support_fadeout_mps", ctypes.c_float),
        ("high_speed_support_fadein_mps", ctypes.c_float),
        ("speed_mps", ctypes.c_float * 5),
        ("q", ((ctypes.c_float * 10) * 2) * 5),
        ("yaw_rate_abs_rps", ((ctypes.c_float * 10) * 2) * 5),
        ("steering_blend_start_rad", ctypes.c_float),
        ("steering_blend_full_start_rad", ctypes.c_float),
        ("steering_blend_full_end_rad", ctypes.c_float),
        ("steering_blend_end_rad", ctypes.c_float),
        ("hold_response_time_constant_s", ctypes.c_float),
        ("hold_rate_full_radps", ctypes.c_float),
        ("hold_rate_zero_radps", ctypes.c_float),
    ]


def _load_y1_surface(path: Path, params: dict[str, Any]) -> YawResponseSurface:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(
            line for line in stream if not line.lstrip().startswith("#")))
    speeds = sorted({float(row["speed_knot_mps"]) for row in rows})
    steering_knots = (0.15, 0.20, 0.21, 0.22, 0.23,
                      0.25, 0.30, 0.35, 0.42, 0.50)
    if len(speeds) != 3 or len(rows) != 3 * 2 * len(steering_knots):
        raise ValueError("Y1 surface must contain three complete speed knots")

    surface = YawResponseSurface()
    surface.enabled = 1
    surface.speed_count = len(speeds)
    parameter_names = (
        "blend_q_start", "blend_q_end", "speed_blend_margin_mps",
        "low_speed_blend_margin_mps", "low_speed_support_fadeout_mps",
        "high_speed_support_fadein_mps", "steering_blend_start_rad",
        "steering_blend_full_start_rad", "steering_blend_full_end_rad",
        "steering_blend_end_rad", "hold_time_constant_s",
        "hold_rate_full_radps", "hold_rate_zero_radps",
    )
    for name in parameter_names:
        param_name = "yaw_rate_response_surface_" + name
        field_name = ("hold_response_time_constant_s" if name == "hold_time_constant_s"
                      else name)
        setattr(surface, field_name, float(params.get(param_name, 0.0)))

    direction_index = {-1: 0, 1: 1}
    for row_index, row in enumerate(rows):
        speed_index = row_index // (2 * len(steering_knots))
        within_speed = row_index % (2 * len(steering_knots))
        turn_sign = int(row["turn_sign"])
        direction = direction_index[turn_sign]
        knot = within_speed % len(steering_knots)
        steering = float(row["steering_rad"])
        if (abs(float(row["speed_knot_mps"]) - speeds[speed_index]) > 1e-6
                or abs(steering - steering_knots[knot]) > 1e-6
                or within_speed // len(steering_knots) != direction):
            raise ValueError("Y1 response surface rows are not in production order")
        surface.q[speed_index][direction][knot] = float(row["demand_q"])
        surface.yaw_rate_abs_rps[speed_index][direction][knot] = float(
            row["yaw_rate_abs_rps"])
        if knot == 0 and direction == 0:
            surface.speed_mps[speed_index] = speeds[speed_index]
    return surface


def _topic_rows(db: sqlite3.Connection, name: str) -> list[tuple[int, Any]]:
    item = db.execute("SELECT id,type FROM topics WHERE name=?", (name,)).fetchone()
    if item is None:
        raise ValueError(f"bag is missing required topic {name}")
    topic_id, type_name = int(item[0]), str(item[1])
    message_type = get_message(type_name)
    return [
        (int(stamp), deserialize_message(bytes(raw), message_type))
        for stamp, raw in db.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? "
            "ORDER BY timestamp,id", (topic_id,))
    ]


def _smoothstep(value: float, low: float, high: float) -> float:
    t = min(max((value - low) / (high - low), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _surface_gate(speed: float, steering: float, steering_rate: float) -> float:
    q = max(speed, 0.0) * abs(math.tan(steering))
    q_gate = _smoothstep(q, 0.30, 0.70)
    angle = abs(steering)
    steering_gate = _smoothstep(angle, 0.29, 0.30) * (
        1.0 - _smoothstep(angle, 0.42, 0.45))
    speed_gate = _smoothstep(speed, 2.40, 2.50) * (
        1.0 - _smoothstep(speed, 3.50, 3.60))
    rate_gate = 1.0 - _smoothstep(abs(steering_rate), 0.05, 0.15)
    return q_gate * steering_gate * speed_gate * rate_gate


def _band_gate(value: float, lower_start: float, lower_full: float,
               upper_full: float, upper_end: float) -> float:
    return (_smoothstep(value, lower_start, lower_full)
            * (1.0 - _smoothstep(value, upper_full, upper_end)))


def _residual_features(state: ModelState, control: ModelControl,
                       curvature: float, acceleration: float) -> np.ndarray:
    u = float(state.u)
    r = float(state.r)
    delta = float(state.actual_steering_angle)
    q_delta = float(control.steering_rate)
    v = float(state.v)
    q_speed = float(control.target_speed_rate)
    return np.asarray((
        u, r, delta, q_delta, v, curvature, acceleration, q_speed,
        u * delta * abs(delta), delta * abs(q_delta), v * abs(delta),
        abs(delta) * acceleration, delta * abs(q_speed),
    ), dtype=np.float64)


def _residual_rate(model: dict[str, Any], features: np.ndarray) -> float:
    means = np.asarray(model["feature_mean"], dtype=np.float64)
    scales = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(
        model["coefficients_with_intercept"], dtype=np.float64)
    if (means.shape != (13,) or scales.shape != (13,)
            or coefficients.shape != (14,) or not np.isfinite(means).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0.0)
            or not np.isfinite(coefficients).all()):
        raise ValueError("yaw residual model has invalid coefficient dimensions")
    correction = float(coefficients[0] + np.dot(
        coefficients[1:], (features - means) / scales))
    clip = float(model["correction_clip_radps2"])
    return float(np.clip(correction, -clip, clip))


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    error = prediction - truth
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "mae_radps": float(np.mean(np.abs(error))),
        "bias_radps": float(np.mean(error)),
        "p95_abs_radps": float(np.quantile(np.abs(error), 0.95)),
        "wrong_direction_count": int(np.sum(prediction * truth < 0.0)),
    }


def _merge_config(base_path: Path, overlay_path: Path) -> dict[str, Any]:
    base_doc = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    overlay_doc = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    base_params = dict(next(iter(base_doc.values())).get("ros__parameters", {}))
    overlay_params = next(iter(overlay_doc.values())).get("ros__parameters", {})
    base_params.update(overlay_params)
    return {"/**": {"ros__parameters": base_params}}


def _score_bag(path: Path, label: str, library: Any,
               candidate_surface: YawResponseSurface,
               residual_model: dict[str, Any] | None = None,
               residual_band: dict[str, float] | None = None,
               ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        diagnostic_rows = _topic_rows(db, MPC_TOPIC)
        collision_rows = _topic_rows(db, COLLISION_TOPIC)

    collision_counts = [int(row.data) for _, row in collision_rows]
    collision_start = collision_counts[0] if collision_counts else None
    first_collision_ns = next((stamp for stamp, msg in collision_rows
                               if collision_start is not None
                               and int(msg.data) > collision_start), None)
    if collision_start is None or collision_start != 0:
        return ({"run_id": label, "bag": str(path.relative_to(ROOT)),
                 "usable": False, "reason": "missing or initially nonzero collision count",
                 "collision_start": collision_start,
                 "collision_end": collision_counts[-1] if collision_counts else None}, [])

    capture = load_capture(path, include_nonvalid_phases=True,
                           continuous_phased_run=True)
    diagnostics_by_source_stamp: dict[int, tuple[int, dict[str, Any]]] = {}
    for receipt_ns, message in diagnostic_rows:
        try:
            payload = json.loads(message.data)
            source_stamp = int(payload["odom_source_stamp_ns"])
        except (AttributeError, KeyError, TypeError, ValueError,
                json.JSONDecodeError):
            continue
        diagnostics_by_source_stamp[source_stamp] = (receipt_ns, payload)

    samples: list[dict[str, Any]] = []
    skipped_no_diag = skipped_bad_payload = skipped_invalid_stage = 0
    skipped_collision = skipped_missing_truth = skipped_sequence_gap = 0
    library.vehicle_model_set_yaw_rate_response_surface.argtypes = [
        ctypes.POINTER(YawResponseSurface)]
    library.vehicle_model_set_yaw_rate_response_surface.restype = ctypes.c_int
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl),
        ctypes.c_float, ctypes.c_float]
    library.mpc_vehicle_model_step.restype = StageResult
    disabled_surface = YawResponseSurface()
    surface_enabled = True  # ProductionMpc initialization loaded the candidate.

    for sequence in capture.sequences:
      for current, following in zip(sequence, sequence[1:]):
        if (current.packet_sequence < 0 or following.packet_sequence < 0
                or following.packet_sequence != current.packet_sequence + 1):
            skipped_sequence_gap += 1
            continue
        if first_collision_ns is not None and following.receipt_ns >= first_collision_ns:
            skipped_collision += 1
            continue
        if (current.simulator_rigid_state is None
                or following.simulator_rigid_state is None):
            skipped_missing_truth += 1
            continue
        aligned = diagnostics_by_source_stamp.get(current.source_stamp_ns)
        if aligned is None:
            skipped_no_diag += 1
            continue
        diagnostic_receipt_ns, payload = aligned
        if first_collision_ns is not None and diagnostic_receipt_ns >= first_collision_ns:
            skipped_collision += 1
            continue
        try:
            state_values = payload.get("state", [])
            action = payload.get("first_action", [])
            first_prediction = next(
                (item for item in payload.get("predictions", [])
                 if int(item.get("n", -1)) == 1), None)
            baseline_state = (first_prediction or {}).get("state", [])
            path_curvature = float(payload["path_curvature_per_m"])
            if (len(state_values) < 10 or len(action) < 4 or len(baseline_state) < 5
                    or not np.isfinite(np.asarray(state_values[:10], dtype=float)).all()
                    or not np.isfinite(np.asarray(action[:4], dtype=float)).all()
                    or not math.isfinite(path_curvature)):
                skipped_bad_payload += 1
                continue
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            skipped_bad_payload += 1
            continue

        state = ModelState(*map(float, state_values[:10]))
        control = ModelControl(float(action[2]), float(action[3]))
        if surface_enabled:
            if not library.vehicle_model_set_yaw_rate_response_surface(
                    ctypes.byref(disabled_surface)):
                raise RuntimeError("failed to disable Y1 surface for legacy replay")
            surface_enabled = False
        legacy_stage = library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(control), DT_S, path_curvature)
        if not library.vehicle_model_set_yaw_rate_response_surface(
                ctypes.byref(candidate_surface)):
            raise RuntimeError("failed to enable Y1 surface for candidate replay")
        surface_enabled = True
        candidate_stage = library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(control), DT_S, path_curvature)
        if not legacy_stage.valid or not candidate_stage.valid:
            skipped_invalid_stage += 1
            continue

        truth_yaw_next = float(following.simulator_rigid_state[12])
        logged_runtime_yaw_next = float(baseline_state[4])
        legacy_yaw_next = float(legacy_stage.next.r)
        candidate_yaw_next = float(candidate_stage.next.r)
        speed = float(state.u)
        steering = float(state.actual_steering_angle)
        steering_rate = (float(state.delayed_steering_command_1) - steering) / DT_S
        gate = _surface_gate(speed, steering, steering_rate)
        sample = {
            "run_id": label,
            "packet_sequence": current.packet_sequence,
            "sample_dt_s": DT_S,
            "source_stamp_ns": current.source_stamp_ns,
            "diagnostic_receipt_skew_s": (
                diagnostic_receipt_ns - current.receipt_ns) / 1e9,
            "speed_mps": speed,
            "steering_rad": steering,
            "steering_rate_radps": steering_rate,
            "path_curvature_per_m": path_curvature,
            "surface_gate": gate,
            "logged_runtime_yaw_next_radps": logged_runtime_yaw_next,
            "legacy_yaw_next_radps": legacy_yaw_next,
            "candidate_yaw_next_radps": candidate_yaw_next,
            "truth_yaw_next_radps": truth_yaw_next,
            "legacy_error_radps": legacy_yaw_next - truth_yaw_next,
            "candidate_error_radps": candidate_yaw_next - truth_yaw_next,
            "runtime_candidate_parity_error_radps": (
                candidate_yaw_next - logged_runtime_yaw_next),
            "mpc_status": payload.get("status"),
            "collision_free_pair": True,
        }
        if residual_model is not None:
            features = _residual_features(
                state, control, path_curvature,
                float(candidate_stage.body_accel_mps2))
            residual_rate = _residual_rate(residual_model, features)
            residual_gate = 1.0
            if residual_band is not None:
                residual_gate = (
                    _band_gate(speed,
                               residual_band["speed_lower_start_mps"],
                               residual_band["speed_lower_full_mps"],
                               residual_band["speed_upper_full_mps"],
                               residual_band["speed_upper_end_mps"])
                    * _band_gate(abs(steering),
                                 residual_band["steering_lower_start_rad"],
                                 residual_band["steering_lower_full_rad"],
                                 residual_band["steering_upper_full_rad"],
                                 residual_band["steering_upper_end_rad"]))
            unbounded_yaw = candidate_yaw_next + DT_S * residual_rate
            bandlimited_yaw = candidate_yaw_next + (
                DT_S * residual_rate * residual_gate)
            sample.update({
                "yaw_residual_rate_radps2": residual_rate,
                "yaw_residual_band_gate": residual_gate,
                "y1_plus_residual_unbounded_yaw_next_radps": unbounded_yaw,
                "y1_plus_residual_unbounded_error_radps": (
                    unbounded_yaw - truth_yaw_next),
                "y1_plus_residual_bandlimited_yaw_next_radps": bandlimited_yaw,
                "y1_plus_residual_bandlimited_error_radps": (
                    bandlimited_yaw - truth_yaw_next),
            })
        samples.append(sample)

    def select(predicate: Any) -> list[dict[str, Any]]:
        return [sample for sample in samples if predicate(sample)]

    groups = {
        "whole_preimpact_run": select(lambda _: True),
        "supported_high_steering": select(lambda sample:
            2.40 <= sample["speed_mps"] <= 3.60
            and 0.30 <= abs(sample["steering_rad"]) <= 0.45),
        "yaw_transition_band": select(lambda sample:
            sample.get("yaw_residual_band_gate", 0.0) >= 0.05),
        "outside_yaw_transition_band": select(lambda sample:
            sample.get("yaw_residual_band_gate", 0.0) < 0.05),
        "candidate_materially_active": select(lambda sample:
            sample["surface_gate"] >= 0.05),
        "below_0p30_rad": select(lambda sample:
            abs(sample["steering_rad"]) < 0.30),
        "above_0p45_rad_or_speed_support": select(lambda sample:
            abs(sample["steering_rad"]) > 0.45
            or sample["speed_mps"] < 2.40 or sample["speed_mps"] > 3.60),
    }
    report: dict[str, Any] = {
        "run_id": label,
        "bag": str(path.relative_to(ROOT)),
        "bag_bytes": path.stat().st_size,
        "collision_start_end": [collision_start, collision_counts[-1]],
        "first_collision_receipt_ns": first_collision_ns,
        "pairing": "exact odom_source_stamp_ns plus next contiguous packet_sequence; fixed 25 ms step, receipt jitter is not used as sample dt",
        "truth_samples_scored": len(samples),
        "skipped": {
            "no_prior_mpc_diagnostic": skipped_no_diag,
            "invalid_mpc_payload": skipped_bad_payload,
            "invalid_candidate_model_step": skipped_invalid_stage,
            "collision_or_post_collision_pair": skipped_collision,
            "missing_simulator_truth": skipped_missing_truth,
            "nonconsecutive_packet_sequence": skipped_sequence_gap,
        },
        "groups": {},
        "usable": bool(samples),
    }
    for name, group in groups.items():
        if group:
            report["groups"][name] = {
                "legacy": _metrics(
                    np.asarray([row["legacy_yaw_next_radps"] for row in group]),
                    np.asarray([row["truth_yaw_next_radps"] for row in group])),
                "candidate": _metrics(
                    np.asarray([row["candidate_yaw_next_radps"] for row in group]),
                    np.asarray([row["truth_yaw_next_radps"] for row in group])),
                **({
                    "y1_plus_residual_unbounded": _metrics(
                        np.asarray([row[
                            "y1_plus_residual_unbounded_yaw_next_radps"]
                            for row in group]),
                        np.asarray([row["truth_yaw_next_radps"] for row in group])),
                    "y1_plus_residual_bandlimited": _metrics(
                        np.asarray([row[
                            "y1_plus_residual_bandlimited_yaw_next_radps"]
                            for row in group]),
                        np.asarray([row["truth_yaw_next_radps"] for row in group])),
                } if residual_model is not None else {}),
            }
    return report, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", action="append", required=True,
                        help="label=practice rosbag SQLite path; repeatable")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--overlay-config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--residual-model", type=Path,
                        help="optional frozen open-plane yaw residual JSON")
    args = parser.parse_args()

    library_path = args.library.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    sample_path = output_dir / "one_step_samples.csv"
    if report_path.exists() or sample_path.exists():
        raise FileExistsError(f"refusing to overwrite existing score in {output_dir}")

    bags: list[tuple[str, Path]] = []
    for item in args.bag:
        label, separator, raw_path = item.partition("=")
        if not separator or not label:
            raise ValueError("--bag must have form label=path")
        bag_path = Path(raw_path)
        if not bag_path.is_absolute():
            bag_path = ROOT / bag_path
        if not bag_path.is_file():
            raise FileNotFoundError(bag_path)
        bags.append((label, bag_path.resolve()))
    if len({label for label, _ in bags}) != len(bags):
        raise ValueError("practice bag labels must be unique")

    residual_model = None
    residual_band = None
    if args.residual_model is not None:
        residual_path = args.residual_model.resolve()
        residual_model = json.loads(residual_path.read_text(encoding="utf-8"))
        if residual_model.get("sample_period_s") != DT_S:
            raise ValueError("yaw residual sample period does not match fixed 25 ms")
        residual_band = dict(YAW_TRANSITION_RESIDUAL_BAND)

    merged_config = _merge_config(args.base_config.resolve(),
                                  args.overlay_config.resolve())
    params = merged_config["/**"]["ros__parameters"]
    params["yaw_rate_residual_enabled"] = False
    surface_path = ROOT / "f1tenth_mpc/config/yaw_response_surface_y1.csv"
    params["yaw_rate_response_surface_file"] = str(surface_path)
    candidate_surface = _load_y1_surface(surface_path, params)
    reports: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    with ExitStack() as stack:
        temp_dir = stack.enter_context(tempfile.TemporaryDirectory(
            prefix="y1-practice-score-", dir=output_dir))
        config_path = Path(temp_dir) / "merged_mpc.yaml"
        config_path.write_text(yaml.safe_dump(merged_config, sort_keys=False),
                               encoding="utf-8")
        controller = stack.enter_context(ProductionMpc(
            library_path, config_path, args.trajectory.resolve()))
        library = controller.library
        for label, bag_path in bags:
            report, samples = _score_bag(
                bag_path, label, library, candidate_surface,
                residual_model, residual_band)
            reports.append(report)
            all_samples.extend(samples)

    groups = {
        "whole_preimpact_run": lambda row: True,
        "supported_high_steering": lambda row: (
            2.40 <= row["speed_mps"] <= 3.60
            and 0.30 <= abs(row["steering_rad"]) <= 0.45),
        "candidate_materially_active": lambda row: row["surface_gate"] >= 0.05,
        "yaw_transition_band": lambda row: (
            row.get("yaw_residual_band_gate", 0.0) >= 0.05),
        "outside_yaw_transition_band": lambda row: (
            row.get("yaw_residual_band_gate", 0.0) < 0.05),
    }
    macro: dict[str, Any] = {}
    for name, predicate in groups.items():
        per_run = {}
        for label, _ in bags:
            group = [row for row in all_samples
                     if row["run_id"] == label and predicate(row)]
            if group:
                per_run_row = {
                    "legacy_rmse_radps": _metrics(
                        np.asarray([row["legacy_yaw_next_radps"] for row in group]),
                        np.asarray([row["truth_yaw_next_radps"] for row in group]))["rmse_radps"],
                    "candidate_rmse_radps": _metrics(
                        np.asarray([row["candidate_yaw_next_radps"] for row in group]),
                        np.asarray([row["truth_yaw_next_radps"] for row in group]))["rmse_radps"],
                }
                if residual_model is not None:
                    for prediction_name, output_name in (
                        ("unbounded_residual_rmse_radps",
                         "y1_plus_residual_unbounded_yaw_next_radps"),
                        ("bandlimited_residual_rmse_radps",
                         "y1_plus_residual_bandlimited_yaw_next_radps"),
                    ):
                        per_run_row[prediction_name] = _metrics(
                            np.asarray([row[output_name] for row in group]),
                            np.asarray([row["truth_yaw_next_radps"]
                                        for row in group]))["rmse_radps"]
                per_run[label] = per_run_row
        macro[name] = {"per_run_rmse": per_run}
        if per_run:
            deltas = np.asarray([
                row["candidate_rmse_radps"] - row["legacy_rmse_radps"]
                for row in per_run.values()], dtype=np.float64)
            macro[name]["macro_run_mean_candidate_minus_baseline_rmse_radps"] = float(
                np.mean(deltas))

    output = {
        "study": "practice transfer score for Y1 and optional open-plane yaw residual",
        "method": "same recorded MPC state/action; exact odom_source_stamp_ns joins to the current packet and the next contiguous packet is the truth label at fixed dt=25 ms; legacy and Y1 production C transitions are scored separately; optional residual is an offline correction only",
        "future_truth_used_as_model_input": False,
        "collision_policy": "exclude any current/future pair at or after first collision; reject captures starting with nonzero count",
        "timing_policy": "packet sequence defines adjacency; receipt jitter is diagnostic only and does not change model dt",
        "yaw_residual_model": (
            None if args.residual_model is None else
            str(args.residual_model.resolve().relative_to(ROOT))),
        "yaw_residual_model_sha256": (
            None if args.residual_model is None else
            hashlib.sha256(args.residual_model.resolve().read_bytes()).hexdigest()),
        "yaw_residual_band": residual_band,
        "runs": reports,
        "macro_by_run": macro,
        "limitations": [
            "Teacher-forced one-step practice transfer is not a closed-loop MPC track run.",
            "Logged actions were generated by each run's original controller, not the candidate MPC.",
            "The score covers only existing pre-impact bag data and supported sampled regimes.",
        ],
    }
    report_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if all_samples:
        with sample_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_samples[0]))
            writer.writeheader()
            writer.writerows(all_samples)
    print(json.dumps({
        "report": str(report_path.relative_to(ROOT)),
        "runs": [{
            "run_id": row["run_id"], "usable": row["usable"],
            "samples": row.get("truth_samples_scored", 0),
            "candidate_active": row.get("groups", {}).get(
                "candidate_materially_active", {}).get("candidate", {}).get("samples", 0),
            "candidate_active_rmse": row.get("groups", {}).get(
                "candidate_materially_active", {}).get("candidate", {}).get("rmse_radps"),
            "legacy_active_rmse": row.get("groups", {}).get(
                "candidate_materially_active", {}).get("legacy", {}).get("rmse_radps"),
        } for row in reports],
        "sample_csv": str(sample_path.relative_to(ROOT)) if all_samples else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
