#!/usr/bin/env python3
"""Audit and fit yaw one-step models using only exact-two-packet responses.

This consumes already-admitted clean train/validation captures. Test and final
test splits are rejected by the shared corpus admission gate. For each labeled
probe, the intended actuator response is measured from raw packet-sequence
samples; only a command-to-feedback onset of exactly two consecutive simulator
packets contributes rows to fitting or scoring.

This is an offline research model. Simulator truth is a target/score only.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from tools import analyze_open_plane_dynamics as dynamics
from tools.evaluate_open_plane_body_dynamics import (
    STEERING_LIMIT_RAD, load_capture)

try:
    import audit_yaw_exact_packet_sensor_alignment as alignment
    import evaluate_yaw_packet_response_conditioned_models as rows_loader
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import audit_yaw_exact_packet_sensor_alignment as alignment
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as rows_loader
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_full_domain_exact_two_v2")
SEED_AUDIT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
              "yaw_full_domain_exact_two_v1/phase_audit.json")
HISTORY_LAGS = (0, 1, 2, 4)
WINDOW_BEFORE_NS = 100_000_000
WINDOW_AFTER_NS = 500_000_000
STEERING_COMMAND_THRESHOLD_RAD = 0.01
STEERING_FEEDBACK_THRESHOLD_RAD = 0.01
THROTTLE_COMMAND_THRESHOLD = 0.005
THROTTLE_FEEDBACK_THRESHOLD = 0.005
MIN_GLOBAL_ROWS = 200
MIN_EVENT_ROWS = 300
MIN_EVENT_RUNS = 3
MIN_LOCAL_ROWS = 160
MIN_LOCAL_RUNS = 2
MIN_LOCAL_ROWS_PER_RUN = 20
SPEED_BIN_MPS = 1.0
STEER_BIN_RAD = 0.10
ESTIMATOR = dict(n_estimators=240, max_depth=14, min_samples_leaf=5,
                 max_features=0.9, random_state=20261008, n_jobs=4)

GENERIC_STEERING_LABEL = re.compile(
    r"^probe_yawerr_(?P<family>.*?)_"
    r"(?P<event>onset|unwind|reversal)_v(?P<speed>[0-9.]+)_"
    r"a(?P<angle>[0-9.]+)_turn(?P<sign>[+-]1)_delay(?P<delay>[0-9.]+)_"
    r"(?P<mode>step|ramp)(?P<duration>[0-9.]+)s(?:_rep[0-9]+)?$")
GAP_STEERING_LABEL = re.compile(
    r"^probe_yawerr_gap_v(?P<speed>[0-9.]+)_a(?P<angle>[0-9.]+)_"
    r"turn(?P<sign>[+-]1)_delay(?P<delay>[0-9.]+)_(?P<mode>step|ramp)$")
THROTTLE_LABEL = re.compile(
    r"^probe_yawerr_(?P<family>wheel|lowwheel|midwheel)_"
    r"v(?P<speed>[0-9.]+)_a(?P<angle>[0-9.]+)_turn(?P<sign>[+-]1)_"
    r"(?P<direction>up|cut|down)(?:_d|)(?P<delta>[0-9.]+)?"
    r"(?:_ramp|_step|_step0\.025s|_ramp0\.300s)?$")
CRAWL_THROTTLE_LABEL = re.compile(
    r"^probe_yawerr_crawl_throttle_t(?P<throttle>[0-9]+)_"
    r"a(?P<angle>[+-][0-9.]+)_r[0-9]+$")
MULTI_STEERING_LABEL = re.compile(
    r"^(?P<family>lowyaw|yawgap|atlas)_r[0-9]+_v(?P<speed>[0-9.]+)_"
    r"a(?P<angle>[0-9.]+)(?:_rate(?P<mode>step|ramp))?_"
    r"turn(?P<sign>[+-]1)$")
MISMATCH_LABEL = re.compile(
    r"^yawmis_r[0-9]+_(?P<event>onset|reversal)_v(?P<speed>[0-9.]+)_"
    r"a(?P<angle>[0-9.]+)_(?P<mode>step|ramp)_turn(?P<sign>[+-]1)$")
LOWDYN_LABEL = re.compile(
    r"^(?P<family>lowdyn|yawdyn)_v(?P<speed>[0-9.]+)_"
    r"a(?P<angle>[0-9.]+)_turn(?P<sign>[+-]1)$")
SUBNET_LABEL = re.compile(
    r"^subnet_c[0-9]+_s(?P<speed>[0-9.]+)_d(?P<sign>[+-]1)_"
    r"a(?P<angle>[0-9.]+)_(?P<event>onset|turnin|unwind)_"
    r"(?P<load>brake|active_braking|acceleration|throttle_reduction)$")
HIGH_SPEED_ENVELOPE_LABEL = re.compile(
    r"^probe_yawenv_v(?P<speed>[0-9.]+)_a(?P<angle>[0-9.]+)_"
    r"turn(?P<sign>[+-]1)$")
SPARSE_CELL_SUPPORT_LABEL = re.compile(
    r"^probe_yawsupport_(?P<event>onset|unwind|reversal)_"
    r"v(?P<speed>[0-9.]+)_a(?P<angle>[0-9.]+)_"
    r"turn(?P<sign>[+-]1)_rep[0-9]+_idx(?P<index>[0-9]+)$")
SUPPORT_REPLICATION_LABEL = re.compile(
    r"^probe_yawsupportrep_(?P<event>onset|unwind|reversal)_"
    r"v(?P<speed>[0-9.]+)_a(?P<angle>[0-9.]+)_idx(?P<index>[0-9]+)$")


def _read_phases(bag: Path) -> tuple[list[Any], dict[str, Any]]:
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        return dynamics._phase_events(connection, topics)
    finally:
        connection.close()


def _stimulus(label: str, phase: Any) -> dict[str, Any] | None:
    """Return the designed command event for a known yaw probe label."""
    match = SUPPORT_REPLICATION_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        event = item["event"]
        angle = float(item["angle"])
        target = (angle if event == "onset" else 0.0
                  if event == "unwind" else -angle)
        return {
            "channel": "steering", "delay_s": 0.0,
            "event": "turn_in" if event == "onset" else event,
            "requested_speed_mps": float(item["speed"]),
            "requested_abs_steering_rad": angle,
            "turn_sign": 1, "target": target,
            "family": "exact_two_support_replication",
            "transition_index": int(item["index"]),
        }
    match = SPARSE_CELL_SUPPORT_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        event = item["event"]
        sign = 1 if item["sign"] == "+1" else -1
        angle = float(item["angle"])
        target = (sign * angle if event == "onset" else 0.0
                  if event == "unwind" else -sign * angle)
        return {
            "channel": "steering", "delay_s": 0.0,
            "event": "turn_in" if event == "onset" else event,
            "requested_speed_mps": float(item["speed"]),
            "requested_abs_steering_rad": angle,
            "turn_sign": sign, "target": target,
            "family": "sparse_cell_support",
            "transition_index": int(item["index"]),
        }
    match = HIGH_SPEED_ENVELOPE_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        sign = 1 if item["sign"] == "+1" else -1
        angle = float(item["angle"])
        return {
            "channel": "steering", "delay_s": 0.0, "event": "onset",
            "requested_speed_mps": float(item["speed"]),
            "requested_abs_steering_rad": angle,
            "turn_sign": sign, "target": sign * angle,
            "family": "high_speed_envelope",
        }
    match = GENERIC_STEERING_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        event = item["event"]
        sign = 1 if item["sign"] == "+1" else -1
        angle = float(item["angle"])
        target = (sign * angle if event == "onset" else 0.0
                  if event == "unwind" else -sign * angle)
        return {"channel": "steering", "delay_s": float(item["delay"]),
                "event": event, "requested_speed_mps": float(item["speed"]),
                "requested_abs_steering_rad": angle,
                "turn_sign": sign, "target": target,
                "family": item["family"]}
    match = GAP_STEERING_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        sign = 1 if item["sign"] == "+1" else -1
        angle = float(item["angle"])
        # This probe moves command through zero while the physical steering
        # starts at sign*angle; the response direction is measured from data.
        return {"channel": "steering", "delay_s": float(item["delay"]),
                "event": "command_gap", "requested_speed_mps": float(item["speed"]),
                "requested_abs_steering_rad": angle, "turn_sign": sign,
                "target": -sign * 0.0142, "family": "gap"}
    match = THROTTLE_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        direction = item["direction"]
        return {"channel": "throttle", "delay_s": 0.80,
                "event": "throttle_" + direction,
                "requested_speed_mps": float(item["speed"]),
                "requested_abs_steering_rad": float(item["angle"]),
                "turn_sign": 1 if item["sign"] == "+1" else -1,
                "target_direction": -1 if direction in ("cut", "down") else 1,
                "family": item["family"]}
    match = CRAWL_THROTTLE_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        return {"channel": "throttle", "delay_s": 0.0,
                "event": "crawl_throttle_step",
                "requested_speed_mps": 0.0,
                "requested_abs_steering_rad": abs(float(item["angle"])),
                "turn_sign": 1 if float(item["angle"]) >= 0 else -1,
                "target_direction": 1, "family": "crawl_throttle"}
    return None


def _stimuli_for_phase(phase: Any) -> list[dict[str, Any]]:
    """Describe each isolated steering/throttle command transition in a phase."""
    label = phase.label
    single = _stimulus(label, phase)
    if single is not None:
        return [single]
    match = MULTI_STEERING_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        family = item["family"]
        speed, angle = float(item["speed"]), float(item["angle"])
        sign = 1 if item["sign"] == "+1" else -1
        if family == "lowyaw":
            ramp = item["mode"] == "ramp"
            times = (0.0, 0.70, 1.20, 1.95)
            events = ("onset", "unwind", "reversal", "unwind")
            targets = (sign * angle, 0.0, -sign * angle, 0.0)
        else:
            times = (0.0, 0.70, 1.30, 2.10)
            events = ("onset", "unwind", "reversal", "unwind")
            targets = (sign * angle, 0.0, -sign * angle, 0.0)
            ramp = False
        return [{
            "channel": "steering", "delay_s": delay, "event": event,
            "requested_speed_mps": speed,
            "requested_abs_steering_rad": angle, "turn_sign": sign,
            "target": target, "family": family,
            "transition_index": index,
            "profile_mode": "ramp" if ramp else "step",
        } for index, (delay, event, target) in enumerate(
            zip(times, events, targets))]
    match = MISMATCH_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        speed, angle = float(item["speed"]), float(item["angle"])
        sign = 1 if item["sign"] == "+1" else -1
        mode = item["mode"]
        if item["event"] == "onset":
            main_delay, main_target = 0.20, sign * angle
            reset_delay = 1.20
        else:
            main_delay, main_target = 0.70, -sign * angle
            reset_delay = 1.50
        return [
            {"channel": "steering", "delay_s": main_delay,
             "event": item["event"], "requested_speed_mps": speed,
             "requested_abs_steering_rad": angle, "turn_sign": sign,
             "target": main_target, "family": "yawmis",
             "transition_index": 0, "profile_mode": mode},
            {"channel": "steering", "delay_s": reset_delay,
             "event": "unwind", "requested_speed_mps": speed,
             "requested_abs_steering_rad": angle, "turn_sign": sign,
             "target": 0.0, "family": "yawmis",
             "transition_index": 1, "profile_mode": mode},
        ]
    match = LOWDYN_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        speed, angle = float(item["speed"]), float(item["angle"])
        sign = 1 if item["sign"] == "+1" else -1
        if item["family"] == "yawdyn":
            times = (0.0, 0.75, 1.60, 2.35)
            targets = (sign * angle, 0.0, -sign * min(angle, 0.30), 0.0)
            events = ("onset", "unwind", "reversal", "unwind")
        else:
            times = (0.0, 0.50, 0.80, 1.25)
            targets = (sign * angle, 0.0, -sign * 0.30, 0.0)
            events = ("onset", "unwind", "reversal", "unwind")
        return [{
            "channel": "steering", "delay_s": delay, "event": event,
            "requested_speed_mps": speed,
            "requested_abs_steering_rad": angle, "turn_sign": sign,
            "target": target, "family": item["family"],
            "transition_index": index,
        } for index, (delay, event, target) in enumerate(
            zip(times, events, targets))]
    match = SUBNET_LABEL.fullmatch(label)
    if match:
        item = match.groupdict()
        speed, angle = float(item["speed"]), float(item["angle"])
        sign = 1 if item["sign"] == "+1" else -1
        event = "onset" if item["event"] == "turnin" else item["event"]
        steering_target = 0.0 if event == "unwind" else sign * angle
        throttle_direction = (1 if item["load"] == "acceleration" else -1)
        return [
            {"channel": "steering", "delay_s": 0.0, "event": event,
             "requested_speed_mps": speed,
             "requested_abs_steering_rad": angle, "turn_sign": sign,
             "target": steering_target, "family": "subnet",
             "transition_index": 0},
            {"channel": "throttle", "delay_s": 0.15,
             "event": "throttle_" + item["load"],
             "requested_speed_mps": speed,
             "requested_abs_steering_rad": angle, "turn_sign": sign,
             "target_direction": throttle_direction, "family": "subnet",
             "transition_index": 1},
        ]
    return []


def _channel_values(samples: list[Any], channel: str
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                               float, float]:
    times = np.asarray([sample.receipt_ns for sample in samples], dtype=np.int64)
    seq = np.asarray([sample.packet_sequence for sample in samples], dtype=np.int64)
    if channel == "steering":
        command = np.asarray([sample.actuators[3] * STEERING_LIMIT_RAD
                              for sample in samples], dtype=np.float64)
        feedback = np.asarray([sample.actuators[0] for sample in samples],
                              dtype=np.float64)
        return (times, seq, command, feedback,
                STEERING_COMMAND_THRESHOLD_RAD,
                STEERING_FEEDBACK_THRESHOLD_RAD)
    command = np.asarray([sample.actuators[2] for sample in samples],
                         dtype=np.float64)
    feedback = np.asarray([sample.actuators[1] for sample in samples],
                          dtype=np.float64)
    return (times, seq, command, feedback,
            THROTTLE_COMMAND_THRESHOLD, THROTTLE_FEEDBACK_THRESHOLD)


def _channel_value_cache(samples: list[Any]
                         ) -> dict[str, tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray,
                                              float, float]]:
    """Materialize packet channels once for a run's many transition queries."""
    times = np.asarray([sample.receipt_ns for sample in samples], dtype=np.int64)
    sequence = np.asarray([sample.packet_sequence for sample in samples],
                          dtype=np.int64)
    steering_command = np.asarray(
        [sample.actuators[3] * STEERING_LIMIT_RAD for sample in samples],
        dtype=np.float64)
    steering_feedback = np.asarray(
        [sample.actuators[0] for sample in samples], dtype=np.float64)
    throttle_command = np.asarray(
        [sample.actuators[2] for sample in samples], dtype=np.float64)
    throttle_feedback = np.asarray(
        [sample.actuators[1] for sample in samples], dtype=np.float64)
    return {
        "steering": (times, sequence, steering_command, steering_feedback,
                     STEERING_COMMAND_THRESHOLD_RAD,
                     STEERING_FEEDBACK_THRESHOLD_RAD),
        "throttle": (times, sequence, throttle_command, throttle_feedback,
                     THROTTLE_COMMAND_THRESHOLD,
                     THROTTLE_FEEDBACK_THRESHOLD),
    }


def _response_measurement(samples: list[Any], phase: Any,
                          stimulus: dict[str, Any],
                          channel_cache: dict[str, tuple[np.ndarray, ...]] | None = None
                          ) -> dict[str, Any]:
    channel = str(stimulus["channel"])
    values = (channel_cache if channel_cache is not None else
              _channel_value_cache(samples))
    times, sequence, command, feedback, command_threshold, feedback_threshold = (
        values[channel])
    scheduled_ns = int(phase.start_ns + float(stimulus["delay_s"]) * 1e9)
    command_base_begin = int(np.searchsorted(
        times, scheduled_ns - 150_000_000, side="left"))
    command_base_end = int(np.searchsorted(
        times, scheduled_ns - 25_000_000, side="left"))
    feedback_base_begin = int(np.searchsorted(
        times, scheduled_ns - 250_000_000, side="left"))
    feedback_base_end = int(np.searchsorted(
        times, scheduled_ns - 50_000_000, side="left"))
    if (command_base_end - command_base_begin < 2
            or feedback_base_end - feedback_base_begin < 2):
        return {"response_packet_count": None, "classification": "missing_baseline"}
    command_before = float(np.median(
        command[command_base_begin:command_base_end]))
    feedback_before = float(np.median(
        feedback[feedback_base_begin:feedback_base_end]))
    direction = float(np.sign(float(stimulus.get("target", command_before))
                              - command_before))
    if "target_direction" in stimulus:
        direction = float(stimulus["target_direction"])
    command_begin = int(np.searchsorted(
        times, scheduled_ns - 10_000_000, side="left"))
    command_end = min(
        int(np.searchsorted(times, scheduled_ns + 250_000_000, side="right")),
        int(np.searchsorted(times, int(phase.end_ns), side="left")))
    command_candidates = np.flatnonzero(
        np.abs(command[command_begin:command_end] - command_before)
        >= command_threshold)
    if not len(command_candidates):
        return {"response_packet_count": None, "classification": "no_command_departure"}
    command_index = command_begin + int(command_candidates[0])
    command_direction = float(np.sign(command[command_index] - command_before))
    if direction == 0.0:
        direction = command_direction
    # If the observed command trajectory contradicts the scheduled target,
    # trust its initial signed movement and keep that discrepancy in metadata.
    feedback_begin = command_index
    feedback_end = min(
        int(np.searchsorted(times, times[command_index] + 500_000_000,
                            side="right")),
        int(np.searchsorted(times, int(phase.end_ns), side="left")))
    moved = np.flatnonzero(
        ((feedback[feedback_begin:feedback_end] - feedback_before)
         * command_direction) >= feedback_threshold)
    if not len(moved):
        return {"response_packet_count": None,
                "classification": "no_feedback_departure",
                "command_start_ns": int(times[command_index]),
                "command_direction_matches_target": bool(direction == command_direction)}
    feedback_index = feedback_begin + int(moved[0])
    command_seq = int(sequence[command_index])
    feedback_seq = int(sequence[feedback_index])
    interval_begin = int(np.searchsorted(sequence, command_seq, side="left"))
    interval_end = int(np.searchsorted(sequence, feedback_seq, side="right"))
    interval = sequence[interval_begin:interval_end]
    contiguous = bool(len(interval) == feedback_seq - command_seq + 1
                      and np.all(np.diff(interval) == 1))
    response_steps = feedback_seq - command_seq
    return {
        "response_packet_count": int(response_steps),
        "classification": ("exact_two" if response_steps == 2 and contiguous
                            else "non_two_or_gap"),
        "channel": channel,
        "scheduled_ns": scheduled_ns,
        "command_start_ns": int(times[command_index]),
        "feedback_onset_ns": int(times[feedback_index]),
        "command_packet_sequence": command_seq,
        "feedback_packet_sequence": feedback_seq,
        "packet_sequence_contiguous": contiguous,
        "command_baseline": command_before,
        "feedback_baseline": feedback_before,
        "command_direction": command_direction,
        "command_direction_matches_target": bool(direction == command_direction),
        "latency_ms_on_packet_grid": float(response_steps * 25.0),
        "command_start_offset_ms": float((times[command_index] - scheduled_ns) / 1e6),
    }


def _measure_run(series: Any, phases: list[Any], end: dict[str, Any],
                 skip_phase_labels: set[str] | None = None
                 ) -> dict[str, Any]:
    bag = alignment._bag_for_run(series)
    capture = load_capture(bag, include_nonvalid_phases=True,
                           continuous_phased_run=True)
    samples = [sample for sequence in capture.sequences for sample in sequence
               if sample.packet_sequence >= 0]
    samples.sort(key=lambda sample: sample.packet_sequence)
    if len(samples) < 10:
        raise ValueError(f"{series.run_id}: insufficient raw packet samples")
    channel_cache = _channel_value_cache(samples)
    skip_phase_labels = skip_phase_labels or set()
    by_label = {phase.label: phase for phase in phases
                if _stimuli_for_phase(phase)}
    measurements = []
    excluded: Counter[str] = Counter()
    for label, phase in sorted(by_label.items()):
        if label in skip_phase_labels:
            continue
        if phase.valid is not True:
            excluded["invalid_phase"] += 1
            continue
        for spec in _stimuli_for_phase(phase):
            measurement = _response_measurement(
                samples, phase, spec, channel_cache=channel_cache)
            count = measurement.get("response_packet_count")
            if count != 2 or not measurement.get("packet_sequence_contiguous", False):
                excluded[str(measurement.get("classification", f"count_{count}"))] += 1
            measurements.append({
                "label": f"{label}::transition{spec.get('transition_index', 0)}",
                "phase_label": label,
                "phase_index": int(phase.index),
                "phase_start_ns": int(phase.start_ns),
                "phase_end_ns": int(phase.end_ns),
                "phase_valid": bool(phase.valid), "stimulus": spec,
                **measurement,
            })
    return {
        "run_id": series.run_id, "split": series.split,
        "bag": str(bag.relative_to(ROOT)),
        "phase_count": len(by_label),
        "phase_end_reason": end.get("reason"),
        "phase_end_aborted": end.get("aborted"),
        "packet_samples": len(samples),
        "measurements": measurements,
        "excluded_phase_counts": dict(excluded),
    }


def _metric(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return {"samples": 0}
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
        "fraction_abs_error_at_most_0p1": float(np.mean(absolute <= 0.1)),
        "fraction_abs_error_at_most_0p05": float(np.mean(absolute <= 0.05)),
    }


def _run_balanced_weights(run_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(run_ids.astype(str), return_counts=True)
    count_by_id = dict(zip(ids.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_id[str(run_id)] for run_id in run_ids],
                         dtype=np.float64)
    return weights * (len(ids) / weights.sum())


def _fit(x: np.ndarray, y: np.ndarray, run_ids: np.ndarray
         ) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(**ESTIMATOR).fit(
        x, y, sample_weight=_run_balanced_weights(run_ids))


def _bootstrap_run_rmse(errors: np.ndarray, run_ids: np.ndarray
                        ) -> dict[str, Any]:
    by_run = []
    for run_id in sorted(set(run_ids.astype(str))):
        values = errors[run_ids.astype(str) == run_id]
        if len(values):
            by_run.append(float(np.sqrt(np.mean(values ** 2))))
    if not by_run:
        return {"independent_runs": 0}
    data = np.asarray(by_run, dtype=np.float64)
    rng = np.random.default_rng(20261008)
    draws = rng.choice(data, (10000, len(data)), replace=True).mean(axis=1)
    return {
        "independent_runs": int(len(data)),
        "run_macro_rmse_radps": float(np.mean(data)),
        "run_macro_rmse_median_radps": float(np.median(data)),
        "bootstrap_95pct_ci": [float(np.quantile(draws, 0.025)),
                                float(np.quantile(draws, 0.975))],
        "per_run_rmse_radps": {run_id: float(np.sqrt(np.mean(
            errors[run_ids.astype(str) == run_id] ** 2)))
            for run_id in sorted(set(run_ids.astype(str)))},
    }


def _coverage(rows: dict[str, np.ndarray], mask: np.ndarray,
              speed_width: float = SPEED_BIN_MPS,
              steer_width: float = STEER_BIN_RAD) -> dict[str, Any]:
    speed = rows["gt_speed"][mask]
    steering = rows["steering"][mask]
    if not len(speed):
        return {"rows": 0, "occupied_speed_steering_cells": 0}
    speed_cell = np.floor(speed / speed_width).astype(int)
    steer_cell = np.floor((steering + 0.525) / steer_width).astype(int)
    counts = Counter(zip(speed_cell.tolist(), steer_cell.tolist()))
    return {
        "rows": int(len(speed)),
        "gt_speed_min_mps": float(np.min(speed)),
        "gt_speed_max_mps": float(np.max(speed)),
        "abs_steering_min_rad": float(np.min(np.abs(steering))),
        "abs_steering_max_rad": float(np.max(np.abs(steering))),
        "occupied_speed_steering_cells": len(counts),
        "cells_with_at_least_20_rows": int(sum(n >= 20 for n in counts.values())),
        "samples_by_1mps_speed_bin": {
            str(key): int(value) for key, value in sorted(Counter(speed_cell).items())},
        "samples_by_0p1rad_abs_steer_bin": {
            str(key): int(value) for key, value in sorted(Counter(
                np.floor(np.abs(steering) / 0.1).astype(int)).items())},
    }


def _collect_rows(series: Any, measurements: list[dict[str, Any]]) -> dict[str, Any]:
    rows = rows_loader._run_rows(series, history_lags=HISTORY_LAGS)
    eligible = [item for item in measurements
                if item.get("response_packet_count") == 2
                and item.get("packet_sequence_contiguous") is True]
    mask = np.zeros(len(rows["residual"]), dtype=bool)
    event_for_row = np.full(len(mask), "", dtype="U32")
    phase_for_row = np.full(len(mask), "", dtype="U256")
    for item in eligible:
        begin = int(item["command_start_ns"]) - WINDOW_BEFORE_NS
        end = int(item["command_start_ns"]) + WINDOW_AFTER_NS
        selected = ((rows["sample_time_ns"] >= begin)
                    & (rows["sample_time_ns"] <= end))
        selected &= np.asarray([
            int(item["phase_start_ns"]) <= int(time_ns) < int(item["phase_end_ns"])
            for time_ns in rows["sample_time_ns"]], dtype=bool)
        # Overlapping valid stimuli are not double-weighted.
        new = selected & ~mask
        mask |= selected
        event_for_row[new] = str(item["stimulus"]["event"])
        phase_for_row[new] = str(item["label"])
    rows["exact2_mask"] = mask
    rows["response_event"] = event_for_row
    rows["response_phase"] = phase_for_row
    rows["run_id"] = np.full(len(mask), series.run_id, dtype="U128")
    # Keep only admissible rows now; retaining every whole-run feature matrix
    # from every capture would needlessly multiply memory use.
    return {key: value[mask] if isinstance(value, np.ndarray)
            and value.shape[:1] == mask.shape else value
            for key, value in rows.items()}


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    run_audits: dict[str, dict[str, Any]] = {}
    cache_path = output / "phase_audit.json"
    admitted_by_id = {series.run_id: series for series in admitted}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_ids = set(cached.get("admitted_run_ids", []))
        if not cached_ids <= set(admitted_by_id):
            raise ValueError("cached phase audit contains runs outside the admitted corpus")
        run_audits = {
            run_id: row for run_id, row in cached["runs"].items()
            if run_id in admitted_by_id
        }
        print(f"loaded cached raw phase audit for {len(run_audits)} runs", flush=True)
    else:
        if SEED_AUDIT.is_file():
            seed = json.loads(SEED_AUDIT.read_text(encoding="utf-8"))
            seed_ids = set(seed.get("admitted_run_ids", []))
            if not seed_ids <= set(admitted_by_id):
                raise ValueError("seed audit contains runs outside the admitted corpus")
            run_audits = {
                run_id: row for run_id, row in seed.get("runs", {}).items()
                if run_id in admitted_by_id
            }
            print(f"reusing probe response audit for {len(run_audits)} runs; "
                  "measuring previously omitted yaw phase families", flush=True)
    for index, series in enumerate(admitted, 1):
        bag = alignment._bag_for_run(series)
        phases, end = _read_phases(bag)
        eligible = [phase for phase in phases if _stimuli_for_phase(phase)]
        if not eligible:
            continue
        prior = run_audits.get(series.run_id)
        existing_labels = set()
        if prior:
            existing_labels = {
                str(item.get("phase_label", item.get("label", "")))
                for item in prior.get("measurements", [])}
        new_labels = {phase.label for phase in eligible} - existing_labels
        if not new_labels:
            continue
        result = _measure_run(series, phases, end,
                              skip_phase_labels=existing_labels)
        if prior:
            result["measurements"] = (prior.get("measurements", [])
                                      + result["measurements"])
            result["excluded_phase_counts"] = dict(Counter(
                prior.get("excluded_phase_counts", {}))
                + Counter(result.get("excluded_phase_counts", {})))
        result["phase_count"] = len(eligible)
        run_audits[series.run_id] = result
        cache_path.write_text(json.dumps({
            "admitted_run_ids": sorted(admitted_by_id),
            "runs": run_audits,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"audited {index}/{len(admitted)} {series.run_id}: "
              f"{len(result['measurements'])} response transitions", flush=True)

    by_id = admitted_by_id
    training_parts = []
    validation_parts = []
    fit_load_errors = []
    for run_id, audit_row in run_audits.items():
        try:
            rows = _collect_rows(by_id[run_id], audit_row["measurements"])
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            fit_load_errors.append({"run_id": run_id, "error": str(exc)})
            continue
        part = {**rows, "split": audit_row["split"]}
        (training_parts if audit_row["split"] == "train"
         else validation_parts).append(part)
        print(f"features {run_id}: {int(np.count_nonzero(rows['exact2_mask']))} "
              "exact-two rows", flush=True)

    if not training_parts or not validation_parts:
        raise RuntimeError("no exact-two train/validation feature rows")

    def combine(parts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        keys = [key for key, value in parts[0].items()
                if isinstance(value, np.ndarray)]
        return {key: np.concatenate([part[key] for part in parts], axis=0)
                for key in keys}

    train = combine(training_parts)
    valid = combine(validation_parts)
    tr_mask = train["exact2_mask"]
    va_mask = valid["exact2_mask"]
    tr_ids = train["run_id"][tr_mask].astype(str)
    va_ids = valid["run_id"][va_mask].astype(str)
    x_train = train["x_timed"][tr_mask]
    y_train = train["residual"][tr_mask]
    x_valid = valid["x_timed"][va_mask]
    y_valid = valid["residual"][va_mask]
    if len(y_train) < MIN_GLOBAL_ROWS or len(y_valid) < 40:
        raise RuntimeError(f"exact-two rows too sparse: train={len(y_train)}, "
                           f"validation={len(y_valid)}")

    global_model = _fit(x_train, y_train, tr_ids)
    global_prediction = global_model.predict(x_valid).astype(np.float64)
    baseline_error = -y_valid.astype(np.float64)
    global_error = global_prediction - y_valid
    # Expert routing uses the causal event label built from current/past
    # steering command and feedback. The experiment's requested transition
    # label remains an evaluation stratum, never a model input or selector.
    event_labels = train["event"][tr_mask].astype(str)
    valid_events = valid["event"][va_mask].astype(str)
    response_events = valid["response_event"][va_mask].astype(str)
    event_models: dict[str, ExtraTreesRegressor] = {}
    event_support: dict[str, Any] = {}
    event_prediction = np.full(len(y_valid), np.nan, dtype=np.float64)
    for event in sorted(set(event_labels)):
        tm = event_labels == event
        vm = valid_events == event
        ids = len(set(tr_ids[tm]))
        if np.count_nonzero(tm) < MIN_EVENT_ROWS or ids < MIN_EVENT_RUNS:
            event_support[event] = {"train_rows": int(tm.sum()),
                                    "train_runs": ids, "fit": False}
            continue
        model = _fit(x_train[tm], y_train[tm], tr_ids[tm])
        event_models[event] = model
        event_prediction[vm] = model.predict(x_valid[vm])
        event_support[event] = {"train_rows": int(tm.sum()),
                                "train_runs": ids,
                                "validation_rows": int(vm.sum()), "fit": True}

    # Local experts are kept intentionally conservative and only exist where
    # independent training runs support the observable speed/steering cell.
    # Runtime-style routing is limited to observable wheel speed and measured
    # steering feedback; simulator GT is reserved for labels and score bins.
    tr_speed = np.floor(train["wheel_speed"][tr_mask] / SPEED_BIN_MPS).astype(int)
    tr_steer = np.floor((train["steering"][tr_mask] + 0.525) / STEER_BIN_RAD).astype(int)
    va_speed = np.floor(valid["wheel_speed"][va_mask] / SPEED_BIN_MPS).astype(int)
    va_steer = np.floor((valid["steering"][va_mask] + 0.525) / STEER_BIN_RAD).astype(int)
    local_models: dict[tuple[int, int, str], ExtraTreesRegressor] = {}
    local_support = Counter()
    for event in sorted(set(event_labels)):
        for speed_cell, steer_cell in sorted(set(zip(
                tr_speed[event_labels == event].tolist(),
                tr_steer[event_labels == event].tolist()))):
            mask = ((tr_speed == speed_cell) & (tr_steer == steer_cell)
                    & (event_labels == event))
            ids, counts = np.unique(tr_ids[mask], return_counts=True)
            keep_ids = ids[counts >= MIN_LOCAL_ROWS_PER_RUN]
            fit_mask = mask & np.isin(tr_ids, keep_ids)
            if (int(fit_mask.sum()) < MIN_LOCAL_ROWS
                    or len(keep_ids) < MIN_LOCAL_RUNS):
                continue
            key = (int(speed_cell), int(steer_cell), event)
            local_models[key] = _fit(x_train[fit_mask], y_train[fit_mask],
                                     tr_ids[fit_mask])
    local_prediction = np.full(len(y_valid), np.nan, dtype=np.float64)
    for key, model in local_models.items():
        s, d, event = key
        mask = (va_speed == s) & (va_steer == d) & (valid_events == event)
        if np.any(mask):
            local_prediction[mask] = model.predict(x_valid[mask])
    local_support_mask = np.isfinite(local_prediction)

    by_speed_steer: dict[str, Any] = {}
    actual_speed_cell = np.floor(
        valid["gt_speed"][va_mask] / SPEED_BIN_MPS).astype(int)
    actual_steer_cell = np.floor(
        (valid["steering"][va_mask] + 0.525) / STEER_BIN_RAD).astype(int)
    for speed_cell, steer_cell in sorted(set(zip(
            actual_speed_cell.tolist(), actual_steer_cell.tolist()))):
        mask = (actual_speed_cell == speed_cell) & (actual_steer_cell == steer_cell)
        if int(mask.sum()) < 10:
            continue
        key = f"v{speed_cell}-{speed_cell + 1}mps_steer{steer_cell}"
        by_speed_steer[key] = {
            "gt_speed_interval_mps": [float(speed_cell), float(speed_cell + 1)],
            "physical_steering_interval_rad": [
                float(-0.525 + steer_cell * STEER_BIN_RAD),
                float(-0.525 + (steer_cell + 1) * STEER_BIN_RAD)],
            "validation_rows": int(mask.sum()),
            "global": _metric(global_error[mask]),
            "event_local": _metric(event_prediction[mask] - y_valid[mask]
                                    ) if np.all(np.isfinite(event_prediction[mask]))
                                    else {"supported_rows": int(np.isfinite(
                                        event_prediction[mask]).sum()),
                                          "total_rows": int(mask.sum())},
            "regime_local": _metric(local_prediction[mask] - y_valid[mask]
                                     ) if np.all(np.isfinite(local_prediction[mask]))
                                     else {"supported_rows": int(np.isfinite(
                                         local_prediction[mask]).sum()),
                                           "total_rows": int(mask.sum())},
            "imu_persistence": _metric(baseline_error[mask]),
        }

    phase_hist = Counter(str(row.get("response_packet_count"))
                         if row.get("response_packet_count") is not None
                         else str(row.get("classification"))
                         for run in run_audits.values()
                         for row in run["measurements"])
    train_phase_meta = [row for run in run_audits.values()
                        if run["split"] == "train" for row in run["measurements"]
                        if row.get("response_packet_count") == 2
                        and row.get("packet_sequence_contiguous") is True]
    validation_phase_meta = [row for run in run_audits.values()
                             if run["split"] == "validation"
                             for row in run["measurements"]
                             if row.get("response_packet_count") == 2
                             and row.get("packet_sequence_contiguous") is True]

    def event_metrics(prediction: np.ndarray) -> dict[str, Any]:
        output = {}
        for event in sorted(set(valid_events)):
            mask = valid_events == event
            output[event] = {
                "validation_rows": int(mask.sum()),
                "global": _metric(global_error[mask]),
                "causal_event_expert": (
                    _metric(prediction[mask] - y_valid[mask])
                    if np.all(np.isfinite(prediction[mask])) else {
                        "supported_rows": int(np.isfinite(prediction[mask]).sum()),
                        "total_rows": int(mask.sum())}),
            }
        return output

    worst_indices = np.argsort(np.abs(global_error))[::-1][:100]
    worst_validation = [{
        "run_id": str(va_ids[index]),
        "sample_time_ns": int(valid["sample_time_ns"][va_mask][index]),
        "response_phase": str(valid["response_phase"][va_mask][index]),
        "causal_event": str(valid_events[index]),
        "measured_response_event": str(response_events[index]),
        "gt_speed_mps": float(valid["gt_speed"][va_mask][index]),
        "rear_wheel_mean_speed_mps": float(valid["wheel_speed"][va_mask][index]),
        "wheel_minus_body_speed_mps": float(
            valid["wheel_speed"][va_mask][index]
            - valid["gt_speed"][va_mask][index]),
        "steering_feedback_rad": float(valid["steering"][va_mask][index]),
        "gt_next_yaw_rate_radps": float(
            valid["residual"][va_mask][index]
            + valid["imu_yaw"][va_mask][index]),
        "predicted_next_yaw_rate_radps": float(
            global_prediction[index]
            + valid["imu_yaw"][va_mask][index]),
        "absolute_error_radps": float(abs(global_error[index])),
    } for index in worst_indices]

    model_bundle = {
        "title": "Research-only exact-two yaw models; not production-integrated",
        "global_extra_trees": global_model,
        "causal_event_experts": event_models,
        "observable_speed_steering_event_experts": local_models,
        "feature_key": "x_timed",
        "feature_names": [
            *(f"{name}_lag{lag * 25}ms"
              for lag in HISTORY_LAGS for name in atlas.OBSERVATION_NAMES),
            *atlas.DERIVED_NAMES,
            *rows_loader.TIMING_FEATURE_NAMES,
        ],
        "history_lags": list(HISTORY_LAGS),
        "history_lags_ms": [lag * 25 for lag in HISTORY_LAGS],
        "event_definition": "command_intent, computed from current and past command/feedback",
        "local_selector": {
            "speed": "causal rear-wheel mean speed in 1.0 m/s bins",
            "steering": "measured steering feedback in 0.10 rad bins",
            "event": "causal command_intent class",
            "uses_ground_truth_at_runtime": False,
        },
        "training_runs": sorted(set(tr_ids)),
        "training_rows": int(len(y_train)),
        "acceptance_packet_response_count": 2,
        "research_artifact_only": True,
    }
    joblib.dump(model_bundle, output / "yaw_models.joblib", compress=3)
    valid_selected = {key: value[va_mask] for key, value in valid.items()
                      if isinstance(value, np.ndarray)
                      and value.shape[:1] == va_mask.shape}
    np.savez_compressed(
        output / "validation_predictions.npz",
        run_id=va_ids,
        sample_time_ns=valid_selected["sample_time_ns"],
        response_phase=valid_selected["response_phase"],
        causal_event=valid_events,
        measured_response_event=response_events,
        gt_speed_mps=valid_selected["gt_speed"],
        rear_wheel_mean_speed_mps=valid_selected["wheel_speed"],
        steering_feedback_rad=valid_selected["steering"],
        current_imu_yaw_rate_radps=valid_selected["imu_yaw"],
        target_yaw_residual_radps=y_valid,
        predicted_global_residual_radps=global_prediction,
        predicted_causal_event_residual_radps=event_prediction,
        predicted_local_residual_radps=local_prediction,
    )

    report = {
        "title": "Observed-envelope yaw one-step models, exact-two response transitions only",
        "acceptance_policy": {
            "only_response_packet_count": 2,
            "requires_contiguous_packet_sequence": True,
            "one_three_and_other_counts_excluded_from_fit_and_score": True,
            "raw_observed_receipt_times_are_not_used_as_physical_dt": True,
            "physical_packet_interval_s": 0.025,
            "response_thresholds": {
                "steering_command_and_feedback_rad": 0.01,
                "throttle_command_and_feedback_norm": 0.005,
            },
            "row_window": {"before_command_ms": 100, "after_command_ms": 500},
            "target": "simulator truth yaw_rate[k+1] minus exact-packet IMU yaw_rate[k]",
            "future_truth_or_sensor_inputs": False,
        },
        "split_policy": {
            "admitted_run_inventory": source_audit,
            "admitted_runs": len(admitted),
            "run_counts_by_split": dict(Counter(s.split for s in admitted)),
            "test_and_final_test_used": False,
        },
        "phase_audit": {
            "runs_with_yaw_response_probes": len(run_audits),
            "source_phase_count_by_split": {
                split: int(sum(row["phase_count"] for row in run_audits.values()
                               if row["split"] == split))
                for split in ("train", "validation")},
            "response_count_histogram_including_exclusions": dict(phase_hist),
            "exact_two_response_events_by_split": {
                "train": len(train_phase_meta),
                "validation": len(validation_phase_meta)},
            "exact_two_response_event_conditions": {
                split: {
                    "requested_speed_min_mps": float(min(
                        x["stimulus"]["requested_speed_mps"] for x in values))
                    if values else None,
                    "requested_speed_max_mps": float(max(
                        x["stimulus"]["requested_speed_mps"] for x in values))
                    if values else None,
                    "requested_abs_steering_max_rad": float(max(
                        x["stimulus"]["requested_abs_steering_rad"] for x in values))
                    if values else None,
                    "events": dict(Counter(x["stimulus"]["event"] for x in values)),
                    "families": dict(Counter(x["stimulus"]["family"] for x in values)),
                }
                for split, values in (("train", train_phase_meta),
                                      ("validation", validation_phase_meta))},
            "per_run": run_audits,
        },
        "model": {
            "feature_contract": "causal sensor/actuator history; exact source-stamp IMU; GT is target only",
            "feature_width": int(x_train.shape[1]),
            "training_exact_two_rows": int(len(y_train)),
            "training_runs_with_exact_two_rows": int(len(set(tr_ids))),
            "validation_exact_two_rows": int(len(y_valid)),
            "validation_runs_with_exact_two_rows": int(len(set(va_ids))),
            "global_extra_trees": {
                "train_fit_rows": int(len(y_train)),
                "validation": _metric(global_error),
                "validation_run_clusters": _bootstrap_run_rmse(global_error, va_ids),
                "imu_persistence_validation": _metric(baseline_error),
                "per_response_event": {
                    event: {
                        "validation_rows": int(np.count_nonzero(valid_events == event)),
                        "metrics": _metric(global_error[valid_events == event]),
                    } for event in sorted(set(valid_events))},
            },
            "event_conditioned_extra_trees": {
                "routing": "causal command_intent from current/past inputs; not the phase label",
                "support": event_support,
                "validation": (_metric(event_prediction[np.isfinite(event_prediction)]
                                           - y_valid[np.isfinite(event_prediction)])
                                if np.any(np.isfinite(event_prediction)) else {"samples": 0}),
                "coverage_fraction": float(np.mean(np.isfinite(event_prediction))),
                "validation_by_causal_event": event_metrics(event_prediction),
            },
            "local_speed_steering_event_extra_trees": {
                "routing": "rear-wheel mean speed, measured steering feedback, causal command_intent",
                "speed_bin_width_mps": SPEED_BIN_MPS,
                "steering_bin_width_rad": STEER_BIN_RAD,
                "minimum_train_rows": MIN_LOCAL_ROWS,
                "minimum_independent_train_runs": MIN_LOCAL_RUNS,
                "minimum_rows_per_train_run": MIN_LOCAL_ROWS_PER_RUN,
                "fitted_expert_count": len(local_models),
                "validation_coverage_fraction": float(np.mean(local_support_mask)),
                "supported_validation": _metric(
                    local_prediction[local_support_mask] - y_valid[local_support_mask]),
            },
            "validation_by_actual_gt_speed_and_feedback_steering": by_speed_steer,
            "worst_100_global_validation_rows": worst_validation,
            "saved_model_bundle": "yaw_models.joblib",
            "saved_validation_predictions": "validation_predictions.npz",
        },
        "row_coverage": {
            "training_exact_two": _coverage(train, tr_mask),
            "validation_exact_two": _coverage(valid, va_mask),
        },
        "feature_load_errors": fit_load_errors,
        "interpretation": [
            "This is one-step yaw-rate prediction, not full-lap pose or offline-plant validation.",
            "Exact-two filtering is transition-level command-to-feedback response timing, not the separate exact-IMU source-stamp join.",
            "An unoccupied cell is unsupported; no interpolation or extrapolation claim is made.",
            "The validated domain is the measured feasible envelope, not the full rectangular Cartesian product.",
            "No production odometry/MPC integration or simulator run is performed by this analysis.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=lambda value:
                   value.item() if isinstance(value, np.generic) else str(value)) + "\n",
        encoding="utf-8")
    (output / "report.md").write_text(_markdown_summary(report), encoding="utf-8")
    print(output / "report.json", flush=True)
    print(json.dumps({
        "phase_count_histogram": dict(phase_hist),
        "exact_two_response_events": report["phase_audit"]["exact_two_response_events_by_split"],
        "train_rows": int(len(y_train)), "validation_rows": int(len(y_valid)),
        "global": report["model"]["global_extra_trees"]["validation"],
        "event_model_coverage": report["model"]["event_conditioned_extra_trees"]["coverage_fraction"],
        "event_model": report["model"]["event_conditioned_extra_trees"]["validation"],
        "local_coverage": report["model"]["local_speed_steering_event_extra_trees"]["validation_coverage_fraction"],
    }, indent=2), flush=True)
    return report


def _markdown_summary(report: dict[str, Any]) -> str:
    phase = report["phase_audit"]
    model = report["model"]
    overall = model["global_extra_trees"]["validation"]
    baseline = model["global_extra_trees"]["imu_persistence_validation"]
    lines = [
        "# Observed-envelope yaw audit: exact-two packet responses",
        "",
        "Generated from admitted clean training and validation captures only.",
        "Test/final-test data remained sealed. No simulator or production code was changed.",
        "",
        "## Data admitted",
        "",
        f"- Runs with yaw response probes: {phase['runs_with_yaw_response_probes']}",
        f"- Exact-two response events: {phase['exact_two_response_events_by_split']}",
        f"- Response count histogram (all classified phases): `{phase['response_count_histogram_including_exclusions']}`",
        f"- Exact-two training/validation rows: {model['training_exact_two_rows']} / {model['validation_exact_two_rows']}",
        f"- Requested exact-two range: train {phase['exact_two_response_event_conditions']['train']}; validation {phase['exact_two_response_event_conditions']['validation']}",
        "",
        "## Held-out validation",
        "",
        f"- Persistence baseline: RMSE {baseline.get('rmse_radps', float('nan')):.5f} rad/s; p95 {baseline.get('p95_abs_radps', float('nan')):.5f}; within 0.1: {baseline.get('fraction_abs_error_at_most_0p1', float('nan')):.3%}.",
        f"- Global ExtraTrees: RMSE {overall.get('rmse_radps', float('nan')):.5f} rad/s; p95 {overall.get('p95_abs_radps', float('nan')):.5f}; within 0.1: {overall.get('fraction_abs_error_at_most_0p1', float('nan')):.3%}.",
        f"- Run-macro RMSE: {model['global_extra_trees']['validation_run_clusters'].get('run_macro_rmse_radps', float('nan')):.5f} rad/s across {model['global_extra_trees']['validation_run_clusters'].get('independent_runs', 0)} independent runs.",
        f"- Event-specialist coverage: {model['event_conditioned_extra_trees']['coverage_fraction']:.1%}; local speed/steering expert coverage: {model['local_speed_steering_event_extra_trees']['validation_coverage_fraction']:.1%}.",
        "",
        "## Limits",
        "",
        "This is a held-out one-step yaw-rate result on exact-two response windows, not a complete 0–12 m/s × ±0.5 rad response model, recursive rollout, full-lap simulator, or production odometry integration. Cells without exact-two training support remain unsupported. The numeric report lists the measured envelope, split-level response counts, and per-cell validation metrics.",
        "",
        "Full per-run/per-phase/per-cell evidence is in `report.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--history-lags-ms", type=int, nargs="+", default=(0, 25, 50, 100),
        help="causal sensor-history lags in ms on the fixed 25-ms packet grid")
    args = parser.parse_args()
    history_lags_ms = tuple(args.history_lags_ms)
    if (not history_lags_ms or history_lags_ms[0] != 0
            or tuple(sorted(set(history_lags_ms))) != history_lags_ms
            or any(lag < 0 or lag % 25 for lag in history_lags_ms)):
        parser.error(
            "--history-lags-ms must be unique, increasing, start at 0, "
            "and contain only non-negative multiples of 25")
    global HISTORY_LAGS
    HISTORY_LAGS = tuple(lag // 25 for lag in history_lags_ms)
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
