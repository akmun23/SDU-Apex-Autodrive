#!/usr/bin/env python3
"""Replay N=30 predictions from legal live state and controller inputs.

The live-state rollout initializes from ``/current_map_pose``, ``/ekf_odom``,
and the two encoder topics. Effective steering is propagated causally from
the known zero-at-start state and controller commands. Simulator telemetry is
used only to align timestamps and score rollout endpoints; a separately named
oracle-initialization ablation is included for diagnosis, never runtime use.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sdu_apex_autodrive"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sdu_apex_autodrive.model_id_frames import decode_packet  # noqa: E402
from build_raceline_operating_envelope import (  # noqa: E402
    DEFAULT_RACELINE,
    _read_raceline,
)
from fit_speed_regime_vehicle_model import _raceline_context  # noqa: E402
from score_raceline_model import (  # noqa: E402
    _raceline_arrays,
    frenet_error,
    operating_class,
)
from structured_vehicle_plant import (  # noqa: E402
    PlantParameters,
    _steering_next,
    step,
)


DEFAULT_CANDIDATE = ROOT / (
    "sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/"
    "speed_regime_vehicle_candidate_unity_controller_source_075_20260916.json"
)
ENCODER_WHEEL_RADIUS_M = 0.059
HORIZON_STEPS = 30
COMMAND_DT_S = 0.025
MIN_DT_TOLERANT_S = 0.01499  # accommodates one float32 40 Hz boundary sample
MAX_DT_S = 0.035


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _events(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                payload = json.loads(row["payload_json"])
                arrival_ns = int(row["arrival_monotonic_ns"])
                stamp_ns = int(row["header_stamp_ns"] or 0)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            grouped.setdefault(row["topic"], []).append({
                "arrival_ns": arrival_ns,
                "stamp_ns": stamp_ns,
                "payload": payload,
            })
    for values in grouped.values():
        values.sort(key=lambda item: item["arrival_ns"])
    return grouped


def _finite(payload: dict[str, Any], field: str) -> float | None:
    try:
        value = float(payload[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)],
                        dtype=float)
    if finite.size == 0:
        return {"count": 0, "mae": None, "p50_abs": None,
                "p95_abs": None, "p99_abs": None, "max_abs": None,
                "bias": None, "rmse": None}
    absolute = np.abs(finite)
    return {
        "count": int(finite.size),
        "mae": float(np.mean(absolute)),
        "p50_abs": float(np.percentile(absolute, 50)),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "max_abs": float(np.max(absolute)),
        "bias": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
    }


def _candidate_parameters(candidate: dict[str, Any]) -> PlantParameters:
    base = PlantParameters.from_manifest()
    overrides = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in candidate["candidate_parameters"].items()
        if key in base.__dataclass_fields__
    }
    return replace(base, **overrides,
                   tire_model="regime_speed_combined_tanh")


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                args, cwd=ROOT, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    dirty = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "worktree_dirty": bool(dirty),
    }


def score(events_csv: Path, candidate_json: Path, output: Path,
          raceline_csv: Path = DEFAULT_RACELINE, origin_stride: int = 1,
          warmup_s: float = 5.0) -> dict[str, Any]:
    if origin_stride < 1:
        raise ValueError("origin_stride must be positive")
    groups = _events(events_csv)
    topic_names = {
        "packets": "/autodrive/roboracer_1/bridge_packet_timing",
        "odom": "/odom",
        "ekf": "/ekf_odom",
        "left_encoder": "/autodrive/roboracer_1/left_encoder",
        "right_encoder": "/autodrive/roboracer_1/right_encoder",
        "controller": "/cmd/speed",
        "map_pose": "/current_map_pose",
    }
    missing = [topic for topic in topic_names.values() if topic not in groups]
    if missing:
        raise ValueError(f"missing required live topics: {missing}")

    packets = groups[topic_names["packets"]]
    for name in ("odom", "ekf", "left_encoder", "right_encoder", "controller"):
        if len(groups[topic_names[name]]) != len(packets):
            raise ValueError(f"{name} count does not match source packet count")
    odom = groups[topic_names["odom"]]
    ekf = groups[topic_names["ekf"]]
    left = groups[topic_names["left_encoder"]]
    right = groups[topic_names["right_encoder"]]
    controller = groups[topic_names["controller"]]
    map_pose = groups[topic_names["map_pose"]]

    # Resolve one fixed ROS/source epoch from paired 40 Hz /odom and telemetry
    # rows. This is timestamp alignment only; no telemetry state enters rollout.
    offsets = [
        row["stamp_ns"] / 1.0e9 - float(packet["payload"]["simulation_time_s"])
        for packet, row in zip(packets, odom)
    ]
    epoch_offset_s = float(statistics.median(offsets))
    epoch_spread_s = max(abs(value - epoch_offset_s) for value in offsets)
    if epoch_spread_s > 1.0e-5:
        raise ValueError(
            "source sensor stamps do not share a stable one-to-one epoch "
            f"(spread={epoch_spread_s:.9g}s)")

    params = _candidate_parameters(
        json.loads(candidate_json.read_text(encoding="utf-8")))
    decoded_truth = [decode_packet(row["payload"]) for row in packets]
    source_times = [packet.time_s for packet in decoded_truth]
    dts = [later - earlier for earlier, later in
           zip(source_times, source_times[1:])]
    if any(not math.isfinite(dt) or dt <= 0.0 or dt > MAX_DT_S for dt in dts):
        raise ValueError("source time reversal, gap, or overlong transition")
    if any(_as_bool(row["payload"].get("sent_reset", False))
           for row in packets):
        raise ValueError("reset-containing runs must be segmented before scoring")
    packet_sequences = [int(row["payload"]["packet_sequence"]) for row in packets]
    if any(b != a + 1 for a, b in zip(packet_sequences, packet_sequences[1:])):
        raise ValueError("source packet sequence has a gap")

    # Index each legal ROS source by header stamp; never interpolate or borrow
    # a later sample. The controller callback is the availability cutoff.
    def exact_available(rows: list[dict[str, Any]], stamp_ns: int,
                        cutoff_ns: int) -> dict[str, Any] | None:
        matches = [row for row in rows
                   if abs(row["stamp_ns"] - stamp_ns) <= 1_000_000 and
                   row["arrival_ns"] <= cutoff_ns]
        return min(matches, key=lambda row: abs(row["stamp_ns"] - stamp_ns)) \
            if matches else None

    def latest_map(source_stamp_ns: int, cutoff_ns: int
                   ) -> dict[str, Any] | None:
        eligible = [row for row in map_pose
                    if 0 < row["stamp_ns"] <= source_stamp_ns and
                    row["arrival_ns"] <= cutoff_ns]
        return max(eligible, key=lambda row: row["stamp_ns"]) if eligible else None

    wheel_speed = [math.nan] * len(packets)
    sensor_pairs: list[tuple[dict[str, Any] | None,
                             dict[str, Any] | None,
                             dict[str, Any] | None,
                             dict[str, Any] | None]] = []
    legal_states: list[tuple[dict[str, Any] | None,
                             dict[str, Any] | None,
                             dict[str, Any] | None]] = []
    for i, packet in enumerate(packets):
        source_stamp_ns = int(round(
            (epoch_offset_s + source_times[i]) * 1.0e9))
        cutoff_ns = controller[i]["arrival_ns"]
        o = exact_available(odom, source_stamp_ns, cutoff_ns)
        e = exact_available(ekf, source_stamp_ns, cutoff_ns)
        l = exact_available(left, source_stamp_ns, cutoff_ns)
        r = exact_available(right, source_stamp_ns, cutoff_ns)
        sensor_pairs.append((o, e, l, r))
        legal_states.append((o, e, latest_map(source_stamp_ns, cutoff_ns)))
        if i == 0 or o is None or e is None or l is None or r is None:
            continue
        previous_stamp_ns = int(round(
            (epoch_offset_s + source_times[i - 1]) * 1.0e9))
        previous_l = exact_available(left, previous_stamp_ns, cutoff_ns)
        previous_r = exact_available(right, previous_stamp_ns, cutoff_ns)
        if previous_l is None or previous_r is None:
            continue
        dt = source_times[i] - source_times[i - 1]
        # JointState stores positions in a one-element array.
        try:
            l0, l1 = float(previous_l["payload"]["position"][0]), float(
                l["payload"]["position"][0])
            r0, r1 = float(previous_r["payload"]["position"][0]), float(
                r["payload"]["position"][0])
        except (IndexError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (l0, l1, r0, r1)):
            wheel_speed[i] = ENCODER_WHEEL_RADIUS_M * 0.5 * (
                (l1 - l0) / dt + (r1 - r0) / dt)

    # Reconstruct effective steering from the known start-at-zero actuator
    # state and causal source commands, using only command and source time.
    steering = [0.0] * len(packets)
    for i in range(1, len(packets)):
        command = packets[i - 1]["payload"]
        dt = source_times[i] - source_times[i - 1]
        target = _finite(command, "commanded_steering_norm")
        if target is None:
            steering[i] = math.nan
        elif math.isfinite(steering[i - 1]):
            steering[i] = _steering_next(steering[i - 1], target, dt, params)
        else:
            steering[i] = math.nan

    raceline = _raceline_arrays(_read_raceline(raceline_csv))
    envelope = _raceline_context(raceline_csv)
    horizon_steps = (4, 10, 20, HORIZON_STEPS)
    samples_by_step: dict[int, list[dict[str, Any]]] = {
        count: [] for count in horizon_steps
    }
    skipped = {
        "warmup": 0,
        "window": 0,
        "state_not_available_at_command": 0,
        "map_pose_missing_or_stale": 0,
        "encoder_or_steering_state_missing": 0,
        "invalid_window": 0,
    }
    map_pose_max_age_s = 0.25
    for i in range(0, len(packets) - HORIZON_STEPS, origin_stride):
        if source_times[i] < warmup_s:
            skipped["warmup"] += 1
            continue
        if i + HORIZON_STEPS >= len(packets):
            skipped["window"] += 1
            continue
        o, e, l, r = sensor_pairs[i]
        m = legal_states[i][2]
        if any(row is None for row in (o, e, l, r)):
            skipped["state_not_available_at_command"] += 1
            continue
        source_stamp_ns = int(round(
            (epoch_offset_s + source_times[i]) * 1.0e9))
        map_age_s = ((source_stamp_ns - m["stamp_ns"]) / 1.0e9
                     if m is not None else math.inf)
        if m is None or map_age_s < 0.0 or map_age_s > map_pose_max_age_s:
            skipped["map_pose_missing_or_stale"] += 1
            continue
        if (not math.isfinite(wheel_speed[i]) or
                not math.isfinite(steering[i])):
            skipped["encoder_or_steering_state_missing"] += 1
            continue
        window_dt = dts[i:i + HORIZON_STEPS]
        if (len(window_dt) != HORIZON_STEPS or
                any(dt < MIN_DT_TOLERANT_S or dt > MAX_DT_S for dt in window_dt)):
            skipped["invalid_window"] += 1
            continue

        pose = m["payload"]
        motion = e["payload"]
        state_values = [
            _finite(pose, "x_m"), _finite(pose, "y_m"),
            _finite(pose, "yaw_rad"), _finite(motion, "speed_mps"),
            _finite(motion, "lateral_speed_mps"),
            _finite(motion, "yaw_rate_radps"), steering[i], wheel_speed[i],
        ]
        if any(value is None or not math.isfinite(float(value))
               for value in state_values):
            skipped["state_not_available_at_command"] += 1
            continue
        state = np.asarray(state_values, dtype=float)
        origin_truth = decoded_truth[i]
        origin_error = frenet_error(
            float(pose["x_m"]), float(pose["y_m"]), float(pose["yaw_rad"]),
            origin_truth.position_x_m, origin_truth.position_y_m,
            origin_truth.yaw_quaternion_rad, raceline)

        # Named oracle ablation: truth is used at the initial state only, then
        # the same model and command sequence recurse without truth feedback.
        oracle = np.asarray([
            origin_truth.position_x_m, origin_truth.position_y_m,
            origin_truth.yaw_quaternion_rad, origin_truth.body_u_mps,
            origin_truth.body_v_mps, origin_truth.yaw_rate_radps,
            steering[i], wheel_speed[i],
        ], dtype=float)
        class_name = operating_class(
            origin_truth.position_x_m, origin_truth.position_y_m,
            origin_truth.body_u_mps, raceline, envelope["core_bins"],
            envelope["guard_bins"], envelope["speed_bin_width_mps"],
            envelope["curvature_bin_width_radpm"], 16.0)
        for j in range(i, i + HORIZON_STEPS):
            command = packets[j]["payload"]
            steering_target = _finite(command, "commanded_steering_norm")
            throttle = _finite(command, "commanded_throttle_norm")
            if steering_target is None or throttle is None:
                state = np.asarray([], dtype=float)
                break
            state = step(state, steering_target, throttle, dts[j], params)
            oracle = step(oracle, steering_target, throttle, dts[j], params)
            count = j - i + 1
            if count not in samples_by_step:
                continue
            if (state.shape != (8,) or oracle.shape != (8,) or
                    not np.all(np.isfinite(state)) or
                    not np.all(np.isfinite(oracle))):
                break
            endpoint = decoded_truth[i + count]
            endpoint_error = frenet_error(
                state[0], state[1], state[2], endpoint.position_x_m,
                endpoint.position_y_m, endpoint.yaw_quaternion_rad, raceline)
            oracle_error = frenet_error(
                oracle[0], oracle[1], oracle[2], endpoint.position_x_m,
                endpoint.position_y_m, endpoint.yaw_quaternion_rad, raceline)
            samples_by_step[count].append({
                "source_packet_sequence": packet_sequences[i],
                "origin_time_s": origin_truth.time_s,
                "horizon_s": endpoint.time_s - origin_truth.time_s,
                "speed_mps": origin_truth.body_u_mps,
                "operating_class": class_name,
                "current_map_pose_age_s": map_age_s,
                "origin_pose_e_cross_m": float(origin_error["e_cross_m"]),
                "origin_u_error_mps": float(
                    motion["speed_mps"] - origin_truth.body_u_mps),
                "origin_v_error_mps": float(
                    motion["lateral_speed_mps"] - origin_truth.body_v_mps),
                "origin_r_error_radps": float(
                    motion["yaw_rate_radps"] - origin_truth.yaw_rate_radps),
                "predicted_e_cross_error_m": float(
                    endpoint_error["e_cross_m"]),
                "predicted_u_error_mps": float(state[3] - endpoint.body_u_mps),
                "predicted_v_error_mps": float(state[4] - endpoint.body_v_mps),
                "predicted_r_error_radps": float(
                    state[5] - endpoint.yaw_rate_radps),
                "predicted_position_error_m": float(endpoint_error["position_m"]),
                "predicted_e_along_error_m": float(endpoint_error["e_along_m"]),
                "oracle_e_cross_error_m": float(oracle_error["e_cross_m"]),
                "oracle_u_error_mps": float(oracle[3] - endpoint.body_u_mps),
            })
        if state.shape != (8,) or not np.all(np.isfinite(state)):
            skipped["invalid_window"] += 1

    fields = (
        "origin_pose_e_cross_m", "origin_u_error_mps",
        "origin_v_error_mps", "origin_r_error_radps",
        "predicted_e_cross_error_m", "predicted_u_error_mps",
        "predicted_v_error_mps", "predicted_r_error_radps",
        "predicted_position_error_m", "predicted_e_along_error_m",
        "oracle_e_cross_error_m", "oracle_u_error_mps",
        "current_map_pose_age_s", "horizon_s",
    )

    def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {field: _stats(row[field] for row in rows) for field in fields}

    metrics_by_horizon: dict[str, Any] = {}
    for count, horizon_samples in samples_by_step.items():
        metrics_by_horizon[f"{count}_steps"] = {
            "step_count": count,
            "nominal_horizon_s": count * COMMAND_DT_S,
            "sample_count": len(horizon_samples),
            **{
                name: group_stats([
                    row for row in horizon_samples
                    if name == "all" or row["operating_class"] == name
                ])
                for name in ("all", "core", "guard")
            },
        }
    groups_out = {
        name: metrics_by_horizon[f"{HORIZON_STEPS}_steps"][name]
        for name in ("all", "core", "guard")
    }
    truth_max_speed = max((packet.body_u_mps for packet in decoded_truth),
                          default=0.0)
    state_diff: dict[str, list[float]] = {
        key: [] for key in ("speed_mps", "lateral_speed_mps", "yaw_rate_radps")
    }
    for o, e, *_ in sensor_pairs:
        if o is None or e is None:
            continue
        for key in ("speed_mps", "lateral_speed_mps", "yaw_rate_radps"):
            ov, ev = _finite(o["payload"], key), _finite(e["payload"], key)
            if ov is not None and ev is not None:
                state_diff[key].append(ev - ov)

    result = {
        "schema_version": 1,
        "status": "live_legal_state_n30_screen_not_production_validated",
        "experiment": "live_pure_pursuit_competition_no_diagnostic_logger",
        "git": _git_state(),
        "inputs": {
            "events_csv": str(events_csv),
            "events_sha256": _sha256(events_csv),
            "candidate_json": str(candidate_json),
            "candidate_sha256": _sha256(candidate_json),
            "raceline_csv": str(raceline_csv),
            "source_packet_count": len(packets),
            "source_sensor_clock_epoch_offset_s": epoch_offset_s,
            "source_sensor_epoch_spread_s": epoch_spread_s,
            "origin_stride_packets": origin_stride,
            "warmup_s": warmup_s,
        },
        "rollout_contract": {
            "steps": HORIZON_STEPS,
            "nominal_dt_s": COMMAND_DT_S,
            "nominal_horizon_s": HORIZON_STEPS * COMMAND_DT_S,
            "initial_state_fields": [
                "/current_map_pose x/y/yaw", "/ekf_odom u/v/r",
                "mean left/right encoder surface speed",
                "causally propagated steering state from command history",
            ],
            "command_fields": [
                "bridge timing topic commanded_steering_norm",
                "bridge timing topic commanded_throttle_norm",
            ],
            "future_simulator_truth_used_in_rollout": False,
            "truth_use": [
                "offline endpoint scoring only",
                "explicit oracle ablation initial state only",
                "offline operating-class assignment",
            ],
            "encoder_radius_m": ENCODER_WHEEL_RADIUS_M,
            "initial_effective_steering_seed_rad": 0.0,
            "pose_timestamp_policy": (
                "latest current_map_pose source stamp not later than rollout "
                "origin and received by the corresponding controller event"),
            "control_sequence_policy": (
                "the recorded causal controller command sequence is supplied "
                "as the known N=30 input; no future state is supplied"),
        },
        "source_quality": {
            "source_time_min_dt_s": min(dts, default=None),
            "source_time_median_dt_s": statistics.median(dts) if dts else None,
            "source_time_max_dt_s": max(dts, default=None),
            "source_intervals_just_below_15ms_float32_tolerance": sum(
                MIN_DT_TOLERANT_S <= dt < 0.015 for dt in dts),
            "maximum_truth_speed_mps_for_envelope_description_only": truth_max_speed,
            "odom_ekf_twist_difference": {
                key: _stats(values) for key, values in state_diff.items()
            },
        },
        "sample_count": len(samples_by_step[HORIZON_STEPS]),
        "skipped_origin_counts": skipped,
        "metrics_by_group": groups_out,
        "metrics_by_horizon": metrics_by_horizon,
        "acceptance": {
            "user_longitudinal_target_u_p95_below_0_10_mps": (
                groups_out["core"]["predicted_u_error_mps"]["p95_abs"]
                is not None and
                groups_out["core"]["predicted_u_error_mps"]["p95_abs"] < 0.10),
            "cross_track_p95_target_not_formally_specified": None,
            "speed_envelope_covers_12_to_16_mps": truth_max_speed >= 12.0,
            "production_mpc_migration": False,
            "reason": (
                "This is a single low-speed live screening run. It cannot pass "
                "the identified 12-16 m/s envelope or production blind/live "
                "acceptance gates."),
        },
        "simulator_modified": False,
        "production_mpc_modified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--raceline", type=Path, default=DEFAULT_RACELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    args = parser.parse_args()
    result = score(args.events_csv, args.candidate, args.output,
                   args.raceline, args.origin_stride, args.warmup_s)
    core = result["metrics_by_group"]["core"]
    print(json.dumps({
        "output": str(args.output),
        "sample_count": result["sample_count"],
        "core_predicted_e_cross_p95_m": core[
            "predicted_e_cross_error_m"]["p95_abs"],
        "core_predicted_u_p95_mps": core[
            "predicted_u_error_mps"]["p95_abs"],
        "core_oracle_u_p95_mps": core["oracle_u_error_mps"]["p95_abs"],
        "accepted_u_below_0_10_mps": result["acceptance"][
            "user_longitudinal_target_u_p95_below_0_10_mps"],
        "production_mpc_migration": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
