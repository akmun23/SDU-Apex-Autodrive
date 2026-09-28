#!/usr/bin/env python3
"""Fit and whole-run validate command-to-feedback actuator models.

This is an offline identification diagnostic. It predicts measured steering
and throttle feedback from their command histories and the current actuator
state. It does not modify the simulator or controller and does not model tire
or vehicle-body forces.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body


DELAYS_S = (0.0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150)
TIME_CONSTANTS_S = (0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.500)
STEERING_RATES_RADPS = (1.6, 2.4, 3.2, 4.0, 4.8, 6.4)
THROTTLE_RATES_PER_S = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
ROLLOUT_SPACING_S = 0.500


@dataclass(frozen=True)
class Candidate:
    family: str
    delay_s: float
    tau_s: float
    rate_limit: float | None

    @property
    def label(self) -> str:
        rate = "none" if self.rate_limit is None else f"{self.rate_limit:g}"
        return (f"{self.family}(delay={self.delay_s:.3f}s,"
                f"tau={self.tau_s:.3f}s,rate={rate})")


@dataclass(frozen=True)
class TrainingRows:
    feedback_now: np.ndarray
    feedback_next: np.ndarray
    dt_s: np.ndarray
    delayed_commands: dict[float, np.ndarray]


def _channel_streams(capture: body.Capture, channel: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feedback_name = f"{channel}_feedback"
    command_name = f"{channel}_command"
    if feedback_name not in capture.actuator_streams or command_name not in capture.actuator_streams:
        raise ValueError(f"capture lacks raw receipt-time streams for {channel}")
    feedback_times_ns, feedback = capture.actuator_streams[feedback_name]
    command_times_ns, command = capture.actuator_streams[command_name]
    if channel == "steering":
        command = np.asarray([
            float(np.float32(value * body.STEERING_LIMIT_RAD))
            for value in command], dtype=float)
    return feedback_times_ns, feedback, command_times_ns, command


def _bounds(channel: str) -> tuple[float, float]:
    if channel == "steering":
        return -body.STEERING_LIMIT_RAD, body.STEERING_LIMIT_RAD
    return 0.0, 1.0


def _rate_grid(channel: str) -> tuple[float, ...]:
    return (STEERING_RATES_RADPS if channel == "steering"
            else THROTTLE_RATES_PER_S)


def _step(current: float, command: float, dt_s: float,
          candidate: Candidate, channel: str) -> float:
    if candidate.tau_s <= 0.0:
        change = command - current
    else:
        alpha = -math.expm1(-dt_s / candidate.tau_s)
        change = alpha * (command - current)
    if candidate.rate_limit is not None:
        maximum_change = candidate.rate_limit * dt_s
        change = min(maximum_change, max(-maximum_change, change))
    lower, upper = _bounds(channel)
    return min(upper, max(lower, current + change))


def _training_rows(capture: body.Capture, channel: str) -> TrainingRows:
    feedback_times_ns, feedback, command_times_ns, command = (
        _channel_streams(capture, channel))
    dt_s = np.diff(feedback_times_ns).astype(float) / 1e9
    valid = ((dt_s >= body.MIN_DT_S) & (dt_s <= body.MAX_DT_S)
             & np.isfinite(feedback[:-1]) & np.isfinite(feedback[1:]))
    if not np.any(valid):
        raise ValueError(f"no valid {channel} transitions in {capture.path}")
    transition_times_ns = feedback_times_ns[:-1]
    command_parts: dict[float, np.ndarray] = {}
    for delay in DELAYS_S:
        query_ns = transition_times_ns - int(round(delay * 1e9))
        command_indices = np.searchsorted(
            command_times_ns, query_ns, side="right") - 1
        command_indices = np.clip(command_indices, 0, len(command) - 1)
        command_parts[delay] = command[command_indices[valid]]
    return TrainingRows(
        feedback[:-1][valid], feedback[1:][valid], dt_s[valid], command_parts)


def _candidate_grid(channel: str) -> list[Candidate]:
    rates = _rate_grid(channel)
    candidates = [Candidate("zero_order", delay, 0.0, None)
                  for delay in DELAYS_S]
    candidates.extend(
        Candidate("first_order", delay, tau, None)
        for delay in DELAYS_S for tau in TIME_CONSTANTS_S)
    candidates.extend(
        Candidate("rate_limited", delay, 0.0, rate)
        for delay in DELAYS_S for rate in rates)
    if channel == "steering":
        # Keep the reported 3.2 rad/s steering-rate hypothesis visible rather
        # than allowing the free-rate grid to obscure it.
        candidates.extend(
            Candidate("guide_rate_limit", delay, 0.0, 3.2)
            for delay in DELAYS_S)
        candidates.extend(
            Candidate("guide_rate_with_filter", delay, tau, 3.2)
            for delay in DELAYS_S for tau in TIME_CONSTANTS_S)
    candidates.extend(
        Candidate("first_order_rate_limited", delay, tau, rate)
        for delay in DELAYS_S for tau in TIME_CONSTANTS_S for rate in rates)
    return candidates


def _predict_one_step(rows: TrainingRows, candidate: Candidate,
                      channel: str) -> np.ndarray:
    alpha = (np.ones_like(rows.dt_s) if candidate.tau_s <= 0.0 else
             -np.expm1(-rows.dt_s / candidate.tau_s))
    change = alpha * (rows.delayed_commands[candidate.delay_s]
                      - rows.feedback_now)
    if candidate.rate_limit is not None:
        max_change = candidate.rate_limit * rows.dt_s
        change = np.clip(change, -max_change, max_change)
    lower, upper = _bounds(channel)
    return np.clip(rows.feedback_now + change, lower, upper)


def _select_candidates(train_captures: list[body.Capture], channel: str
                       ) -> dict[str, tuple[Candidate, float]]:
    rows_by_run = [_training_rows(capture, channel)
                   for capture in train_captures]
    best: dict[str, tuple[Candidate, float]] = {}
    for candidate in _candidate_grid(channel):
        run_mse = []
        for rows in rows_by_run:
            residual = (_predict_one_step(rows, candidate, channel)
                        - rows.feedback_next)
            run_mse.append(float(np.mean(residual ** 2)))
        score = float(np.mean(run_mse))
        previous = best.get(candidate.family)
        if previous is None or score < previous[1]:
            best[candidate.family] = (candidate, score)
    # Report a true no-delay command passthrough as a fixed reference, not as
    # a train-selected candidate.
    baseline = Candidate("immediate_command", 0.0, 0.0, None)
    baseline_mse = np.mean([
        np.mean((_predict_one_step(rows, baseline, channel)
                 - rows.feedback_next) ** 2)
        for rows in rows_by_run
    ])
    best[baseline.family] = (baseline, float(baseline_mse))
    return best


def _nearest_raw_index(times_ns: np.ndarray, target_ns: int,
                       start: int = 0) -> int:
    right = max(start, int(np.searchsorted(times_ns, target_ns, side="left")))
    candidates = [i for i in (right - 1, right)
                  if start <= i < len(times_ns)]
    return min(candidates, key=lambda i: abs(int(times_ns[i]) - target_ns))


def _rollout_errors(capture: body.Capture, channel: str,
                    candidate: Candidate) -> dict[float, list[float]]:
    errors = {horizon: [] for horizon in HORIZONS_S}
    feedback_times_ns, feedback, command_times_ns, command = (
        _channel_streams(capture, channel))
    if len(feedback_times_ns) < 2:
        return errors
    if capture.has_phase_markers:
        starts = [int(np.searchsorted(feedback_times_ns, phase_start, side="left"))
                  for phase_start in capture.phase_start_times_ns]
        starts = sorted(set(index for index in starts
                            if index < len(feedback_times_ns)))
    else:
        starts = []
        next_time_ns = int(feedback_times_ns[0])
        final_time_ns = int(feedback_times_ns[-1])
        while next_time_ns <= final_time_ns - int(HORIZONS_S[-1] * 1e9):
            index = _nearest_raw_index(feedback_times_ns, next_time_ns)
            if not starts or index != starts[-1]:
                starts.append(index)
            next_time_ns += int(ROLLOUT_SPACING_S * 1e9)
    for start in starts:
        for horizon in HORIZONS_S:
            target_time_ns = int(feedback_times_ns[start]
                                 + round(horizon * 1e9))
            target_index = _nearest_raw_index(
                feedback_times_ns, target_time_ns, start)
            if abs(int(feedback_times_ns[target_index]) - target_time_ns) > 40_000_000:
                continue
            predicted = float(feedback[start])
            failed = False
            for index in range(start, target_index):
                dt_s = (int(feedback_times_ns[index + 1])
                        - int(feedback_times_ns[index])) / 1e9
                if not body.MIN_DT_S <= dt_s <= body.MAX_DT_S:
                    failed = True
                    break
                command_time_ns = (int(feedback_times_ns[index])
                                   - int(round(candidate.delay_s * 1e9)))
                command_index = max(0, int(np.searchsorted(
                    command_times_ns, command_time_ns, side="right")) - 1)
                predicted = _step(
                    predicted, float(command[command_index]), dt_s,
                    candidate, channel)
            if not failed:
                errors[horizon].append(
                    predicted - float(feedback[target_index]))
    return errors


def _phase_edge_errors(capture: body.Capture, channel: str,
                       candidate: Candidate) -> dict[str, list[float]]:
    """Break out receipt-time transitions near each experiment phase start."""
    groups = {"small command span": [], "large command span": []}
    if not capture.has_phase_markers:
        return groups
    feedback_times_ns, feedback, command_times_ns, command = (
        _channel_streams(capture, channel))
    threshold = 0.30 if channel == "steering" else 0.10
    seen: set[int] = set()
    for phase_start_ns in capture.phase_start_times_ns:
        lower = max(0, int(np.searchsorted(
            feedback_times_ns, phase_start_ns - 150_000_000, side="left")))
        upper = min(len(feedback_times_ns) - 1, int(np.searchsorted(
            feedback_times_ns, phase_start_ns + 100_000_000, side="right")))
        for index in range(lower, upper):
            current_time_ns = int(feedback_times_ns[index])
            if current_time_ns in seen:
                continue
            seen.add(current_time_ns)
            dt_s = (int(feedback_times_ns[index + 1]) - current_time_ns) / 1e9
            if not body.MIN_DT_S <= dt_s <= body.MAX_DT_S:
                continue
            history_start = int(np.searchsorted(
                command_times_ns, current_time_ns - 100_000_000, side="left"))
            history_end = int(np.searchsorted(
                command_times_ns, current_time_ns, side="right"))
            command_history = command[history_start:history_end]
            command_span = (float(np.max(command_history) - np.min(command_history))
                            if len(command_history) else 0.0)
            command_time_ns = (current_time_ns
                               - int(round(candidate.delay_s * 1e9)))
            command_index = max(0, int(np.searchsorted(
                command_times_ns, command_time_ns, side="right")) - 1)
            predicted = _step(float(feedback[index]),
                              float(command[command_index]), dt_s,
                              candidate, channel)
            group = ("large command span" if command_span >= threshold
                     else "small command span")
            groups[group].append(predicted - float(feedback[index + 1]))
    return groups


def _rmse(values: list[float]) -> float:
    return (float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))
            if values else math.nan)


def evaluate(source_paths: list[Path], holdout_path: Path,
             transfer_paths: list[Path], output_model: Path | None,
             steering_family: str, throttle_family: str) -> None:
    if len(source_paths) < 2:
        raise ValueError("use at least two complete training runs")
    source_captures = [body.load_capture(path) for path in source_paths]
    holdouts = [("whole-run holdout", body.load_capture(holdout_path))]
    holdouts.extend(("transfer holdout", body.load_capture(path))
                    for path in transfer_paths)
    source_resolved = {capture.path.resolve() for capture in source_captures}
    if any(capture.path.resolve() in source_resolved for _, capture in holdouts):
        raise ValueError("holdout bags must be independent of all training bags")
    for index, capture in enumerate(source_captures, 1):
        body._validate_capture(capture, f"source {index}")
    for role, capture in holdouts:
        body._validate_capture(capture, role)

    selected_by_channel: dict[str, dict[str, tuple[Candidate, float]]] = {}
    for channel in ("steering", "throttle"):
        selected = _select_candidates(source_captures, channel)
        selected_by_channel[channel] = selected
        print(f"{channel} command→feedback: train runs="
              + ", ".join(capture.path.parent.parent.name
                           for capture in source_captures))
        print("  one-step equal-run RMSE ranking:")
        ordered = sorted(selected.values(), key=lambda row: row[1])
        for candidate, mse in ordered:
            print(f"    {candidate.label}: train RMSE={math.sqrt(mse):.6f}")
        for role, capture in holdouts:
            print(f"  {role}: {capture.path.parent.parent.name}")
            for family in ("immediate_command", "zero_order", "first_order",
                           "rate_limited", "first_order_rate_limited",
                           "guide_rate_limit", "guide_rate_with_filter"):
                if family not in selected:
                    continue
                candidate, train_mse = selected[family]
                residuals = _rollout_errors(capture, channel, candidate)
                values = ", ".join(
                    f"{horizon * 1000:.0f}ms={_rmse(residuals[horizon]):.6f}"
                    f" (n={len(residuals[horizon])})"
                    for horizon in HORIZONS_S)
                print(f"    {candidate.label}; train={math.sqrt(train_mse):.6f}; "
                      f"rollout RMSE: {values}")
            if capture.has_phase_markers:
                for family in ("zero_order", "first_order",
                               "guide_rate_with_filter"):
                    if family not in selected:
                        continue
                    candidate, _ = selected[family]
                    edge_errors = _phase_edge_errors(
                        capture, channel, candidate)
                    edge_summary = ", ".join(
                        f"{group}={_rmse(edge_errors[group]):.6f}"
                        f" (n={len(edge_errors[group])})"
                        for group in edge_errors if edge_errors[group])
                    if edge_summary:
                        print(f"    phase-edge one-step {candidate.label}: "
                              + edge_summary)
    if output_model is not None:
        chosen = {"steering": steering_family, "throttle": throttle_family}
        channels = {}
        for channel, family in chosen.items():
            if family not in selected_by_channel[channel]:
                raise ValueError(f"{family!r} is not a candidate family for {channel}")
            candidate, train_mse = selected_by_channel[channel][family]
            channels[channel] = {
                "delay_s": candidate.delay_s,
                "time_constant_s": candidate.tau_s,
                "rate_limit_per_s": candidate.rate_limit,
                "output_bounds": list(_bounds(channel)),
                "selected_family": family,
                "balanced_training_rmse": math.sqrt(train_mse),
            }
        output_model.parent.mkdir(parents=True, exist_ok=True)
        output_model.write_text(json.dumps({
            "schema_version": 1,
            "fit_scope": "command-to-feedback actuator surrogate; offline only",
            "source_bags": [str(path.resolve()) for path in source_paths],
            "channel_models": channels,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"saved actuator candidate: {output_model}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bag", action="append", required=True,
                        type=Path, help="whole training run; repeat at least twice")
    parser.add_argument("--holdout-bag", required=True, type=Path,
                        help="independent whole run withheld from fitting")
    parser.add_argument("--transfer-bag", action="append", default=[],
                        type=Path, help="additional whole-run transfer test")
    parser.add_argument("--output-model", type=Path,
                        help="save the selected offline actuator parameters as JSON")
    parser.add_argument("--steering-family", default="guide_rate_with_filter",
                        choices=("immediate_command", "zero_order", "first_order",
                                 "rate_limited", "first_order_rate_limited",
                                 "guide_rate_limit", "guide_rate_with_filter"))
    parser.add_argument("--throttle-family", default="first_order",
                        choices=("immediate_command", "zero_order", "first_order",
                                 "rate_limited", "first_order_rate_limited"))
    args = parser.parse_args()
    try:
        evaluate(args.source_bag, args.holdout_bag, args.transfer_bag,
                 args.output_model, args.steering_family, args.throttle_family)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"actuator evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
