#!/usr/bin/env python3
"""Run WP19's one-seed, fixed-budget body/wheel target ablation.

This is a diagnostic black-box plant experiment. It trains only on complete
training runs, scores held-out validation and unseen-practice runs, and never
feeds future truth or encoder measurements into a recursive rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import DT_S
from tools.vehicle_dynamics_learning.signal_semantics import (
    REAR_AXLE_TO_COM_X_M,
    WHEEL_RADIUS_M,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928"
TASK_ROOT = (DATA_ROOT / "full_modeling_reset_20261001"
             / "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1")
SOURCE_ROOT = (DATA_ROOT / "full_modeling_reset_20261001"
               / "replacement_offline_sim_raceline_20261002"
               / "encoder_raw_state_teacher_v1")
FIXED_ROOT = TASK_ROOT / "encoder_fixed40hz_v1"
DEFAULT_OUTPUT = TASK_ROOT / "wp19_target_ablation_v2_common_encoder_mask"
DEFAULT_DYNAMIC = SOURCE_ROOT / "openplane_dynamics_raw_wheels.npz"
DEFAULT_DYNAMIC_FIXED = FIXED_ROOT / "openplane_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_PRACTICE = SOURCE_ROOT / "practice_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE_FIXED = FIXED_ROOT / "practice_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_DYNAMIC_PARENT = (DATA_ROOT / "full_modeling_reset_20261001"
                          / "replacement_offline_sim_raceline_20261002"
                          / "replacement_teacher_dataset_v1/openplane_dynamics.npz")
DEFAULT_PRACTICE_PARENT = (DATA_ROOT / "practice_transfer_validation_20261001_r03"
                           / "openplane_dynamics.npz")

BODY_TARGETS = ("effective_acceleration", "body_state_increment",
                "next_body_state")
WHEEL_TARGETS = ("wheel_acceleration", "wheel_rate_increment",
                 "next_wheel_rate", "encoder_angle_increment")
HISTORY_STEPS = 80
ROLLOUT_STEPS = 80
HIDDEN_SIZE = 32
SEED = 20261019
TRAIN_STEPS = 300
BATCH_SIZE = 8
EVAL_WINDOWS_PER_RUN = 64
BODY_NAMES = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")
ACTUATOR_NAMES = ("steering_feedback_rad", "throttle_feedback_norm")


@dataclass
class Capture:
    name: str
    source_path: Path
    fixed_path: Path
    parent_path: Path
    parent_sha256: str
    frames: np.ndarray
    dt_s: np.ndarray
    packet: np.ndarray
    bounds: np.ndarray
    sequence_run: np.ndarray
    sequence_reset: np.ndarray
    run_ids: np.ndarray
    splits: np.ndarray
    rigid: np.ndarray
    encoder_rate: np.ndarray
    encoder_valid: np.ndarray
    body: np.ndarray
    acceleration: np.ndarray
    input_features: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _observable_input_features(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 2 or frames.shape[1] != 9:
        raise ValueError("WP19 expects nine-channel aligned source frames")
    return np.column_stack((frames[:, :3], frames[:, 3:5], frames[:, 7:9]))


def _future_command_rows(capture: Capture, sequence_index: int,
                         source_row: int) -> np.ndarray:
    begin = int(capture.bounds[sequence_index, 0]) + source_row
    return capture.frames[begin + 1:begin + ROLLOUT_STEPS + 1, 7:9]


def load_capture(name: str, source_path: Path, fixed_path: Path,
                 parent_path: Path,
                 allowed_splits: set[str]) -> Capture:
    missing = [path for path in (source_path, fixed_path, parent_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    with np.load(source_path, allow_pickle=False) as source, \
            np.load(fixed_path, allow_pickle=False) as fixed:
        for key in ("run_ids", "run_splits", "sequence_bounds",
                    "sequence_run_index", "sequence_reset_index",
                    "packet_sequence", "dt_s", "frames",
                    "simulator_rigid_state"):
            if key not in source or key not in fixed:
                raise ValueError(f"{name}: missing aligned source/fixed field {key}")
            left, right = np.asarray(source[key]), np.asarray(fixed[key])
            if left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError(f"{name}: fixed sidecar row mismatch at {key}")
        required = {"encoder_raw_surface_mps", "encoder_raw_valid",
                    "encoder_raw_source_dataset_sha256"}
        if not required.issubset(source.files) or not required.issubset(fixed.files):
            raise ValueError(f"{name}: fixed-cadence encoder labels missing")
        source_frames = np.asarray(source["frames"], dtype=np.float32)
        dt_s = np.asarray(source["dt_s"], dtype=np.float32)
        packet = np.asarray(source["packet_sequence"], dtype=np.int64)
        bounds = np.asarray(source["sequence_bounds"], dtype=np.int64)
        sequence_run = np.asarray(source["sequence_run_index"], dtype=np.int32)
        sequence_reset = np.asarray(source["sequence_reset_index"], dtype=np.int32)
        run_ids = np.asarray(source["run_ids"]).astype(str)
        splits = np.asarray(source["run_splits"]).astype(str)
        rigid = np.asarray(source["simulator_rigid_state"], dtype=np.float32)
        encoder_rate = np.asarray(fixed["encoder_raw_surface_mps"],
                                  dtype=np.float32)
        encoder_valid = np.asarray(fixed["encoder_raw_valid"], dtype=bool)
        source_parent_hash = str(np.asarray(
            source["encoder_raw_source_dataset_sha256"]).astype(str)[0])
        fixed_parent_hash = str(np.asarray(
            fixed["encoder_raw_source_dataset_sha256"]).astype(str)[0])
    parent_file_hash = sha256_file(parent_path)
    if (source_parent_hash != fixed_parent_hash
            or source_parent_hash != parent_file_hash
            or len(source_parent_hash) != 64):
        raise ValueError(f"{name}: raw encoder parent provenance hash mismatch")
    if (source_frames.ndim != 2 or source_frames.shape[1] != 9
            or rigid.shape != (len(source_frames), 13)
            or encoder_rate.shape != (len(source_frames), 2)
            or encoder_valid.shape != (len(source_frames),)
            or dt_s.shape != (len(source_frames),)
            or bounds.shape != (len(sequence_run), 2)
            or len(sequence_reset) != len(bounds)
            or np.any(sequence_run < 0)
            or np.any(sequence_run >= len(run_ids))
            or len(splits) != len(run_ids)
            or np.any(bounds[:, 0] < 0)
            or np.any(bounds[:, 1] > len(source_frames))
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError(f"{name}: schema or encoder provenance mismatch")
    if not np.allclose(dt_s, DT_S, rtol=0.0, atol=1e-7):
        raise ValueError(f"{name}: expected fixed 25 ms sample periods")

    selected = np.isin(splits[sequence_run], list(allowed_splits))
    body = np.full((len(source_frames), 3), np.nan, dtype=np.float32)
    acceleration = np.full_like(body, np.nan)
    input_features = np.full((len(source_frames), 7), np.nan,
                              dtype=np.float32)
    for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
            zip(bounds, sequence_run)):
        if not selected[sequence_index]:
            continue
        begin, end, run = int(begin_raw), int(end_raw), int(run_raw)
        local_packet = packet[begin:end]
        if (end - begin < HISTORY_STEPS + ROLLOUT_STEPS + 1
                or not np.all(np.isfinite(rigid[begin:end]))
                or not np.all(np.isfinite(source_frames[begin:end]))
                or not np.all(np.isfinite(encoder_rate[begin:end][
                    encoder_valid[begin:end]]))
                or not np.allclose(dt_s[begin:end], DT_S, rtol=0.0, atol=1e-7)
                or np.any(np.diff(local_packet) != 1)):
            continue
        local_rigid = rigid[begin:end].astype(np.float64)
        u_com, v_com, yaw_rate = (local_rigid[:, 7], local_rigid[:, 8],
                                  local_rigid[:, 12])
        rear_v = v_com - REAR_AXLE_TO_COM_X_M * yaw_rate
        local_body = np.column_stack((u_com, rear_v, yaw_rate))
        body[begin:end] = local_body.astype(np.float32)
        local_accel = np.full_like(local_body, np.nan)
        du = np.diff(u_com) / DT_S
        dv_com = np.diff(v_com) / DT_S
        dr = np.diff(yaw_rate) / DT_S
        local_accel[:-1] = np.column_stack((
            du - yaw_rate[:-1] * v_com[:-1],
            dv_com + yaw_rate[:-1] * u_com[:-1], dr))
        acceleration[begin:end] = local_accel.astype(np.float32)
        # History uses the observable bridge odometry/actuator/command stream.
        # Simulator rigid-body truth is retained only for labels and the
        # explicitly known initial body state at each rollout start.
        input_features[begin:end] = _observable_input_features(
            source_frames[begin:end])
    return Capture(
        name=name, source_path=source_path, fixed_path=fixed_path,
        parent_path=parent_path, parent_sha256=source_parent_hash,
        frames=source_frames, dt_s=dt_s, packet=packet, bounds=bounds,
        sequence_run=sequence_run, sequence_reset=sequence_reset,
        run_ids=run_ids, splits=splits, rigid=rigid,
        encoder_rate=encoder_rate, encoder_valid=encoder_valid,
        body=body, acceleration=acceleration, input_features=input_features)


def collect_windows(captures: list[Capture], capture_indices: set[int],
                    split_names: set[str]) -> dict[str, list[tuple[int, int, int]]]:
    by_run: dict[str, list[tuple[int, int, int]]] = {}
    for capture_index in sorted(capture_indices):
        capture = captures[capture_index]
        for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
                zip(capture.bounds, capture.sequence_run)):
            begin, end, run_index = int(begin_raw), int(end_raw), int(run_raw)
            if str(capture.splits[run_index]) not in split_names:
                continue
            if not np.isfinite(capture.input_features[begin:end]).all():
                continue
            for source_row in range(begin + HISTORY_STEPS - 1,
                                    end - ROLLOUT_STEPS):
                by_run.setdefault(str(capture.run_ids[run_index]), []).append(
                    (capture_index, sequence_index, source_row - begin))
    return by_run


def select_eval_windows(
        by_run: dict[str, list[tuple[int, int, int]]],
        maximum_per_run: int = EVAL_WINDOWS_PER_RUN
        ) -> dict[str, list[tuple[int, int, int]]]:
    selected = {}
    for run_id, windows in sorted(by_run.items()):
        if len(windows) > maximum_per_run:
            indices = np.linspace(0, len(windows) - 1, maximum_per_run,
                                  dtype=np.int64)
            windows = [windows[index] for index in indices]
        selected[run_id] = windows
    return selected


def target_values(capture: Capture, sequence_index: int, source_row: int,
                  body_target: str, wheel_target: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    begin, end = map(int, capture.bounds[sequence_index])
    current = begin + source_row
    following = current + 1
    body_now = capture.body[current]
    body_next = capture.body[following]
    if body_target == "effective_acceleration":
        body_value = capture.acceleration[current]
    elif body_target == "body_state_increment":
        body_value = body_next - body_now
    elif body_target == "next_body_state":
        body_value = body_next
    else:
        raise ValueError(f"unknown body target {body_target}")

    wheel_now = capture.encoder_rate[current].astype(np.float64)
    wheel_next = capture.encoder_rate[following].astype(np.float64)
    valid_now = bool(capture.encoder_valid[current])
    valid_next = bool(capture.encoder_valid[following])
    # Use one paired transition population for all four target units so a
    # representation comparison cannot gain by receiving more labels.
    wheel_ok = valid_now and valid_next
    if wheel_target == "wheel_acceleration":
        wheel_value = (wheel_next - wheel_now) / DT_S
    elif wheel_target == "wheel_rate_increment":
        wheel_value = wheel_next - wheel_now
    elif wheel_target == "next_wheel_rate":
        wheel_value = wheel_next
    elif wheel_target == "encoder_angle_increment":
        wheel_value = wheel_next * DT_S / WHEEL_RADIUS_M
    else:
        raise ValueError(f"unknown wheel target {wheel_target}")
    return (np.asarray(body_value, dtype=np.float32),
            np.asarray(capture.frames[following, 3:5], dtype=np.float32),
            np.asarray(wheel_value, dtype=np.float32), wheel_ok)


def _run_balanced_normalizer(groups: dict[str, np.ndarray],
                             minimum: float) -> tuple[np.ndarray, np.ndarray]:
    """Pool run moments with equal weight per independent run."""
    moments = []
    for run_id, values in sorted(groups.items()):
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite).all(axis=1)]
        if not len(finite):
            continue
        mean = finite.mean(axis=0)
        second_moment = np.mean(finite * finite, axis=0)
        moments.append((mean, second_moment))
    if not moments:
        raise ValueError("WP19 run-balanced normalizer has no finite runs")
    mean = np.mean([item[0] for item in moments], axis=0)
    second = np.mean([item[1] for item in moments], axis=0)
    scale = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return mean.astype(np.float32), np.maximum(scale, minimum).astype(np.float32)


def training_statistics(captures: list[Capture],
                        train_windows: dict[str, list[tuple[int, int, int]]]
                        ) -> dict[str, Any]:
    input_groups: dict[str, list[np.ndarray]] = {}
    physical_body_groups: dict[str, list[np.ndarray]] = {}
    for capture in captures:
        for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
                zip(capture.bounds, capture.sequence_run)):
            begin, end, run = int(begin_raw), int(end_raw), int(run_raw)
            if str(capture.splits[run]) != "train":
                continue
            finite = np.isfinite(capture.input_features[begin:end]).all(axis=1)
            run_id = str(capture.run_ids[run])
            input_groups.setdefault(run_id, []).append(
                capture.input_features[begin:end][finite])
            physical_body_groups.setdefault(run_id, []).append(
                capture.body[begin:end][finite])
    if not input_groups:
        raise ValueError("WP19 has no finite training input rows")
    input_mean, input_scale = _run_balanced_normalizer(
        {run: np.concatenate(rows) for run, rows in input_groups.items()}, 1e-2)
    physical_body_mean, physical_body_scale = _run_balanced_normalizer(
        {run: np.concatenate(rows) for run, rows in physical_body_groups.items()},
        1e-2)

    # Cap each run's normalization sample equally; the optimizer itself draws
    # run-uniform windows, so long captures cannot dominate target scaling.
    sampled_refs: dict[str, list[tuple[int, int, int]]] = {}
    for run_id, refs in sorted(train_windows.items()):
        if len(refs) > 4096:
            indices = np.linspace(0, len(refs) - 1, 4096, dtype=np.int64)
            refs = [refs[index] for index in indices]
        sampled_refs[run_id] = refs

    body_groups: dict[str, dict[str, list[np.ndarray]]] = {
        name: {} for name in BODY_TARGETS}
    acceleration_groups: dict[str, list[np.ndarray]] = {}
    actuator_groups: dict[str, list[np.ndarray]] = {}
    wheel_groups: dict[str, dict[str, list[np.ndarray]]] = {
        name: {} for name in WHEEL_TARGETS}
    for run_id, run_windows in sampled_refs.items():
        body_values = {name: [] for name in BODY_TARGETS}
        acceleration_values, actuator_values = [], []
        wheel_values = {name: [] for name in WHEEL_TARGETS}
        for capture_index, sequence_index, source_row in run_windows:
            capture = captures[capture_index]
            begin = int(capture.bounds[sequence_index, 0]) + source_row
            body_now, body_next = capture.body[begin], capture.body[begin + 1]
            acceleration_values.append(capture.acceleration[begin])
            body_values["effective_acceleration"].append(
                capture.acceleration[begin])
            body_values["body_state_increment"].append(body_next - body_now)
            body_values["next_body_state"].append(body_next)
            actuator_values.append(capture.frames[begin + 1, 3:5])
            for wheel_target in WHEEL_TARGETS:
                _, _, wheel_value, wheel_ok = target_values(
                    capture, sequence_index, source_row,
                    "body_state_increment", wheel_target)
                if wheel_ok:
                    wheel_values[wheel_target].append(wheel_value)
        acceleration_groups[run_id] = np.asarray(acceleration_values)
        actuator_groups[run_id] = np.asarray(actuator_values)
        for target_name, values in body_values.items():
            body_groups[target_name][run_id] = np.asarray(values)
        for target_name, values in wheel_values.items():
            if values:
                wheel_groups[target_name][run_id] = np.asarray(values)

    body_minimum = {"effective_acceleration": 0.1,
                    "body_state_increment": 1e-3,
                    "next_body_state": 1e-2}
    body_by_mode = {
        mode: _run_balanced_normalizer(groups, body_minimum[mode])
        for mode, groups in body_groups.items()}
    acceleration_mean, acceleration_scale = body_by_mode[
        "effective_acceleration"]
    actuator_mean, actuator_scale = _run_balanced_normalizer(
        actuator_groups, 1e-2)
    wheel_by_mode = {}
    for wheel_target in WHEEL_TARGETS:
        groups = wheel_groups[wheel_target]
        if not groups:
            raise ValueError(f"WP19 has no valid training labels for {wheel_target}")
        minimum = 0.1 if wheel_target == "wheel_acceleration" else 1e-3
        if wheel_target == "next_wheel_rate":
            minimum = 1e-2
        wheel_by_mode[wheel_target] = _run_balanced_normalizer(groups, minimum)
    return {
        "input": (input_mean, input_scale),
        "physical_body": (physical_body_mean, physical_body_scale),
        "body": body_by_mode,
        "actuator": (actuator_mean, actuator_scale),
        "wheel": wheel_by_mode,
        "acceleration": (acceleration_mean, acceleration_scale),
        "training_target_rows": int(sum(map(len, sampled_refs.values()))),
        "training_normalization_runs": sorted(sampled_refs),
        "normalization_window_cap_per_run": 4096,
    }


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("WP19 requires PyTorch") from exc
    return torch, nn


def make_model(nn, input_dim: int = 7, hidden_size: int = HIDDEN_SIZE):
    class RecurrentTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = nn.GRUCell(input_dim, hidden_size)
            self.head = nn.Linear(hidden_size, 7)

        def step(self, feature, hidden):
            hidden = self.cell(feature, hidden)
            return self.head(hidden), hidden

    return RecurrentTeacher()


def _fit_one(model, torch, optimizer, captures, batch_refs, stats,
             body_target: str, wheel_target: str, device: str,
             gradient_log: list[float]) -> dict[str, float]:
    input_mean_np, input_scale_np = stats["input"]
    body_mean_np, body_scale_np = stats["body"][body_target]
    physical_mean_np, physical_scale_np = stats["physical_body"]
    accel_mean_np, accel_scale_np = stats["acceleration"]
    actuator_mean_np, actuator_scale_np = stats["actuator"]
    wheel_mean_np, wheel_scale_np = stats["wheel"][wheel_target]
    batch_size = len(batch_refs)
    history_np, body_init_np, actuator_init_np, command_init_np = [], [], [], []
    truth_body_np, acceleration_np = [], []
    actuator_targets_np, wheel_targets_np, wheel_masks_np = [], [], []
    wheel_init_np, next_commands_np = [], []
    for capture_index, sequence_index, source_row in batch_refs:
        capture = captures[capture_index]
        begin = int(capture.bounds[sequence_index, 0]) + source_row
        history_np.append(capture.input_features[
            begin - HISTORY_STEPS + 1:begin + 1])
        body_init_np.append(capture.body[begin])
        actuator_init_np.append(capture.frames[begin, 3:5])
        command_init_np.append(capture.frames[begin, 7:9])
        truth_body = capture.body[begin + 1:begin + ROLLOUT_STEPS + 1]
        truth_body_np.append(truth_body)
        acceleration_np.append(capture.acceleration[
            begin:begin + ROLLOUT_STEPS])
        actuator_targets_np.append(capture.frames[
            begin + 1:begin + ROLLOUT_STEPS + 1, 3:5])
        # After predicting t->t+1, the next transition consumes command t+1.
        next_commands_np.append(_future_command_rows(
            capture, sequence_index, source_row))
        rate_now = capture.encoder_rate[begin:begin + ROLLOUT_STEPS]
        rate_next = capture.encoder_rate[begin + 1:begin + ROLLOUT_STEPS + 1]
        valid_now = capture.encoder_valid[begin:begin + ROLLOUT_STEPS]
        valid_next = capture.encoder_valid[begin + 1:begin + ROLLOUT_STEPS + 1]
        if wheel_target == "wheel_acceleration":
            values = (rate_next - rate_now) / DT_S
            mask = valid_now & valid_next
        elif wheel_target == "wheel_rate_increment":
            values = rate_next - rate_now
            mask = valid_now & valid_next
        elif wheel_target == "next_wheel_rate":
            values = rate_next
            mask = valid_now & valid_next
        else:
            values = rate_next * DT_S / WHEEL_RADIUS_M
            mask = valid_now & valid_next
        wheel_targets_np.append(values)
        wheel_masks_np.append(mask)
        # A current measurement initializes the separate wheel-output stream;
        # it is never fed into the body transition or future model inputs.
        wheel_init_np.append(capture.frames[begin, 5:7])

    def tensor(values, dtype=torch.float32):
        return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)

    input_mean, input_scale = tensor(input_mean_np), tensor(input_scale_np)
    body_mean, body_scale = tensor(body_mean_np), tensor(body_scale_np)
    physical_mean, physical_scale = tensor(physical_mean_np), tensor(physical_scale_np)
    accel_mean, accel_scale = tensor(accel_mean_np), tensor(accel_scale_np)
    actuator_mean, actuator_scale = tensor(actuator_mean_np), tensor(actuator_scale_np)
    wheel_mean, wheel_scale = tensor(wheel_mean_np), tensor(wheel_scale_np)

    histories = (tensor(history_np) - input_mean) / input_scale
    body_initial = tensor(body_init_np)
    actuator_initial = tensor(actuator_init_np)
    command_initial = tensor(command_init_np)
    truth_body = tensor(truth_body_np)
    target_actuator = (tensor(actuator_targets_np) - actuator_mean) / actuator_scale
    target_wheel = (tensor(wheel_targets_np) - wheel_mean) / wheel_scale
    truth_accel = (tensor(acceleration_np) - accel_mean) / accel_scale
    wheel_mask = tensor(wheel_masks_np, dtype=torch.bool)
    wheel_initial = tensor(wheel_init_np)
    next_commands = tensor(next_commands_np)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    hidden = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
    for index in range(HISTORY_STEPS - 1):
        _, hidden = model.step(histories[:, index], hidden)
    body = body_initial
    actuator = actuator_initial
    command = command_initial
    wheel_rate = wheel_initial
    body_losses, actuator_losses, wheel_losses, consistency_losses = [], [], [], []
    for step_index in range(ROLLOUT_STEPS):
        physical_input = torch.cat((body, actuator, command), dim=1)
        feature = (physical_input - input_mean) / input_scale
        output, hidden = model.step(feature, hidden)
        target_head = output[:, :3]
        actuator_next_norm = output[:, 3:5]
        wheel_head = output[:, 5:7]
        body_value = target_head * body_scale + body_mean
        body_next, implied_accel = _advance_body_torch(
            torch, body, body_value, body_target)
        actuator_next = actuator_next_norm * actuator_scale + actuator_mean
        wheel_output = wheel_head * wheel_scale + wheel_mean
        if wheel_target == "wheel_acceleration":
            wheel_rate_next = wheel_rate + wheel_output * DT_S
            wheel_angle = wheel_rate_next * DT_S / WHEEL_RADIUS_M
        elif wheel_target == "wheel_rate_increment":
            wheel_rate_next = wheel_rate + wheel_output
            wheel_angle = wheel_rate_next * DT_S / WHEEL_RADIUS_M
        elif wheel_target == "next_wheel_rate":
            wheel_rate_next = wheel_output
            wheel_angle = wheel_rate_next * DT_S / WHEEL_RADIUS_M
        else:
            wheel_angle = wheel_output
            wheel_rate_next = wheel_angle * WHEEL_RADIUS_M / DT_S
        body_loss = torch.nn.functional.smooth_l1_loss(
            (body_next - physical_mean) / physical_scale,
            (truth_body[:, step_index] - physical_mean) / physical_scale,
            beta=1.0)
        # Acceleration-space consistency is directly supervised for A and
        # physics-derived from the transition for B/C, with the same weight.
        accel_norm = (implied_accel - accel_mean) / accel_scale
        consistency = torch.nn.functional.smooth_l1_loss(
            accel_norm, truth_accel[:, step_index], beta=1.0)
        actuator_loss = torch.nn.functional.smooth_l1_loss(
            actuator_next_norm, target_actuator[:, step_index], beta=1.0)
        mask = wheel_mask[:, step_index]
        if torch.any(mask):
            wheel_loss = torch.nn.functional.smooth_l1_loss(
                (wheel_output[mask] - wheel_mean) / wheel_scale,
                target_wheel[mask, step_index], beta=1.0)
        else:
            wheel_loss = body_loss.new_zeros(())
        body_losses.append(body_loss)
        actuator_losses.append(actuator_loss)
        wheel_losses.append(wheel_loss)
        consistency_losses.append(consistency)
        body = body_next
        actuator = actuator_next
        command = next_commands[:, step_index]
        wheel_rate = wheel_rate_next

    body_loss = torch.stack(body_losses).mean()
    actuator_loss = torch.stack(actuator_losses).mean()
    wheel_loss = torch.stack(wheel_losses).mean()
    consistency_loss = torch.stack(consistency_losses).mean()
    loss = body_loss + 0.25 * actuator_loss + 0.25 * wheel_loss \
        + 0.10 * consistency_loss
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0).detach().cpu())
    gradient_log.append(gradient_norm)
    optimizer.step()
    return {"loss": float(loss.detach().cpu()),
            "body": float(body_loss.detach().cpu()),
            "actuator": float(actuator_loss.detach().cpu()),
            "wheel": float(wheel_loss.detach().cpu()),
            "acceleration_consistency": float(consistency_loss.detach().cpu()),
            "gradient_norm_preclip": gradient_norm}


def _implied_acceleration_torch(torch, body_now, body_next):
    du, dv, dr = (body_next - body_now).unbind(dim=1)
    u, v_rear, yaw_rate = body_now.unbind(dim=1)
    alpha = dr / DT_S
    ax = du / DT_S - yaw_rate * (v_rear
        + REAR_AXLE_TO_COM_X_M * yaw_rate)
    ay = dv / DT_S + yaw_rate * u + REAR_AXLE_TO_COM_X_M * alpha
    return torch.stack((ax, ay, alpha), dim=1)


def _advance_body_torch(torch, body, body_value, body_target: str):
    if body_target == "effective_acceleration":
        ax, ay, alpha = body_value.unbind(dim=1)
        u, v_rear, yaw_rate = body.unbind(dim=1)
        body_next = body + DT_S * torch.stack((
            ax + yaw_rate * (v_rear
                + REAR_AXLE_TO_COM_X_M * yaw_rate),
            ay - yaw_rate * u - REAR_AXLE_TO_COM_X_M * alpha,
            alpha), dim=1)
        implied_accel = body_value
    elif body_target == "body_state_increment":
        body_next = body + body_value
        implied_accel = _implied_acceleration_torch(torch, body, body_next)
    elif body_target == "next_body_state":
        body_next = body_value
        implied_accel = _implied_acceleration_torch(torch, body, body_next)
    else:
        raise ValueError(f"unknown body target {body_target}")
    return body_next, implied_accel


def _advance_body_numpy(body: np.ndarray, body_value: np.ndarray,
                        body_target: str) -> np.ndarray:
    body = np.asarray(body, dtype=np.float64)
    value = np.asarray(body_value, dtype=np.float64)
    if body.shape != value.shape or body.shape[-1] != 3:
        raise ValueError("WP19 body state and target must have matching (..., 3) shape")
    if body_target == "effective_acceleration":
        ax, ay, alpha = np.moveaxis(value, -1, 0)
        u, v_rear, yaw_rate = np.moveaxis(body, -1, 0)
        increment = np.stack((
            ax + yaw_rate * (v_rear + REAR_AXLE_TO_COM_X_M * yaw_rate),
            ay - yaw_rate * u - REAR_AXLE_TO_COM_X_M * alpha,
            alpha), axis=-1)
        return body + DT_S * increment
    if body_target == "body_state_increment":
        return body + value
    if body_target == "next_body_state":
        return value.copy()
    raise ValueError(f"unknown body target {body_target}")


def _implied_acceleration_numpy(body_now: np.ndarray,
                                body_next: np.ndarray) -> np.ndarray:
    body_now = np.asarray(body_now, dtype=np.float64)
    body_next = np.asarray(body_next, dtype=np.float64)
    du, dv, dr = np.moveaxis(body_next - body_now, -1, 0)
    u, v_rear, yaw_rate = np.moveaxis(body_now, -1, 0)
    alpha = dr / DT_S
    ax = du / DT_S - yaw_rate * (v_rear
        + REAR_AXLE_TO_COM_X_M * yaw_rate)
    ay = dv / DT_S + yaw_rate * u + REAR_AXLE_TO_COM_X_M * alpha
    return np.stack((ax, ay, alpha), axis=-1)


def _predict_batch(model, torch, captures, refs, stats,
                   body_target: str, wheel_target: str, device: str
                   ) -> dict[str, np.ndarray]:
    input_mean, input_scale = stats["input"]
    actuator_mean, actuator_scale = stats["actuator"]
    wheel_mean, wheel_scale = stats["wheel"][wheel_target]
    model.eval()
    history_rows, initials, commands, rates = [], [], [], []
    truth_bodies, truth_actuators, truth_rates, valid_rates = [], [], [], []
    for capture_index, sequence_index, source_row in refs:
        capture = captures[capture_index]
        begin = int(capture.bounds[sequence_index, 0]) + source_row
        history_rows.append(capture.input_features[
            begin - HISTORY_STEPS + 1:begin + 1])
        initials.append(capture.body[begin])
        # The current transition uses command[t].  Subsequent transitions use
        # command[t+1], command[t+2], ... after the state advances.
        commands.append(_future_command_rows(
            capture, sequence_index, source_row))
        rates.append(capture.frames[begin, 5:7])
        truth_bodies.append(capture.body[begin + 1:begin + ROLLOUT_STEPS + 1])
        truth_actuators.append(capture.frames[
            begin + 1:begin + ROLLOUT_STEPS + 1, 3:5])
        truth_rates.append(capture.encoder_rate[
            begin + 1:begin + ROLLOUT_STEPS + 1])
        valid_rates.append(capture.encoder_valid[
            begin + 1:begin + ROLLOUT_STEPS + 1])
    device_tensor = lambda value: torch.as_tensor(
        np.asarray(value), dtype=torch.float32, device=device)
    input_mean_t, input_scale_t = device_tensor(input_mean), device_tensor(input_scale)
    actuator_mean_t = device_tensor(actuator_mean)
    actuator_scale_t = device_tensor(actuator_scale)
    wheel_mean_t, wheel_scale_t = device_tensor(wheel_mean), device_tensor(wheel_scale)
    body_mean_t, body_scale_t = (device_tensor(stats["body"][body_target][0]),
                                 device_tensor(stats["body"][body_target][1]))
    history = device_tensor(history_rows)
    body = device_tensor(initials)
    command_seq = device_tensor(commands)
    actuator = history[:, -1, 3:5] * input_scale_t[3:5] + input_mean_t[3:5]
    command = history[:, -1, 5:7] * input_scale_t[5:7] + input_mean_t[5:7]
    wheel_rate = device_tensor(rates)
    predicted_bodies, predicted_actuators = [], []
    predicted_rates, predicted_angles, predicted_wheel_targets = [], [], []
    hidden = torch.zeros(len(refs), HIDDEN_SIZE, device=device)
    with torch.no_grad():
        for index in range(HISTORY_STEPS - 1):
            normalized = (history[:, index] - input_mean_t) / input_scale_t
            _, hidden = model.step(normalized, hidden)
        for step_index in range(ROLLOUT_STEPS):
            physical_input = torch.cat((body, actuator, command), dim=1)
            normalized = (physical_input - input_mean_t) / input_scale_t
            output, hidden = model.step(normalized, hidden)
            body_value = output[:, :3] * body_scale_t + body_mean_t
            body_next, _ = _advance_body_torch(
                torch, body, body_value, body_target)
            actuator_next = output[:, 3:5] * actuator_scale_t + actuator_mean_t
            wheel_output = output[:, 5:7] * wheel_scale_t + wheel_mean_t
            if wheel_target == "wheel_acceleration":
                wheel_next = wheel_rate + wheel_output * DT_S
                angle = wheel_next * DT_S / WHEEL_RADIUS_M
            elif wheel_target == "wheel_rate_increment":
                wheel_next = wheel_rate + wheel_output
                angle = wheel_next * DT_S / WHEEL_RADIUS_M
            elif wheel_target == "next_wheel_rate":
                wheel_next = wheel_output
                angle = wheel_next * DT_S / WHEEL_RADIUS_M
            else:
                angle = wheel_output
                wheel_next = angle * WHEEL_RADIUS_M / DT_S
            predicted_bodies.append(body_next.cpu().numpy())
            predicted_actuators.append(actuator_next.cpu().numpy())
            predicted_rates.append(wheel_next.cpu().numpy())
            predicted_angles.append(angle.cpu().numpy())
            predicted_wheel_targets.append(wheel_output.cpu().numpy())
            body, actuator, wheel_rate = body_next, actuator_next, wheel_next
            command = command_seq[:, step_index]
    return {
        "body": np.transpose(np.asarray(predicted_bodies), (1, 0, 2)),
        "actuator": np.transpose(np.asarray(predicted_actuators), (1, 0, 2)),
        "wheel_rate": np.transpose(np.asarray(predicted_rates), (1, 0, 2)),
        "wheel_angle_increment": np.transpose(np.asarray(predicted_angles), (1, 0, 2)),
        "wheel_target": np.transpose(np.asarray(predicted_wheel_targets), (1, 0, 2)),
        "truth_body": np.asarray(truth_bodies),
        "truth_actuator": np.asarray(truth_actuators),
        "truth_wheel_rate": np.asarray(truth_rates),
        "wheel_valid": np.asarray(valid_rates),
    }


def _rmse(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(array ** 2))) if len(array) else None


def _cluster_ci(values: list[float], seed: int) -> list[float] | None:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        return None
    rng = np.random.default_rng(seed + len(sample))
    draw = rng.integers(0, len(sample), size=(5000, len(sample)))
    return np.quantile(sample[draw].mean(axis=1), (0.025, 0.975)).tolist()


SUPPORT_FEATURE_NAMES = (
    "speed_mps", "signed_steering_feedback_rad", "steering_rate_rps",
    "throttle_feedback_norm", "throttle_slew_per_s", "yaw_rate_rps",
    "wheel_body_mismatch_mps",
)


def _support_features(capture: Capture, sequence_index: int,
                      source_row: int) -> np.ndarray:
    begin = int(capture.bounds[sequence_index, 0]) + source_row
    observed = capture.frames[begin]
    previous = capture.frames[begin - 1]
    speed = float(np.hypot(observed[0], observed[1]))
    steering_rate = float((observed[3] - previous[3]) / DT_S)
    throttle_slew = float((observed[4] - previous[4]) / DT_S)
    # Match the established diagnostic definition: filtered rear-wheel mean
    # speed minus bridge rear-axle longitudinal odometry, absolute magnitude.
    mismatch = float(abs(np.mean(observed[5:7]) - observed[0]))
    return np.asarray((speed, observed[3], steering_rate, observed[4],
                       throttle_slew, observed[2], mismatch), dtype=np.float32)


def _score_split(model, torch, captures, windows_by_run, stats,
                 body_target: str, wheel_target: str, device: str,
                 seed_offset: int) -> dict[str, Any]:
    per_run = {}
    per_window = []
    for run_index, (run_id, refs) in enumerate(sorted(windows_by_run.items())):
        prediction = _predict_batch(model, torch, captures, refs, stats,
                                    body_target, wheel_target, device)
        body_error = prediction["body"] - prediction["truth_body"]
        actuator_error = prediction["actuator"] - prediction["truth_actuator"]
        wheel_error = prediction["wheel_rate"] - prediction["truth_wheel_rate"]
        run_report: dict[str, Any] = {
            "windows": len(refs),
            "one_step_body_rmse": {
                name: _rmse(body_error[:, 0, index])
                for index, name in enumerate(BODY_NAMES)},
            "one_step_actuator_rmse": {
                name: _rmse(actuator_error[:, 0, index])
                for index, name in enumerate(ACTUATOR_NAMES)},
            "body_recursive_rmse_at_horizon": {},
            "body_recursive_trajectory_rmse_to_horizon": {},
            "wheel_rate_rmse_at_2s": _rmse(np.where(
                prediction["wheel_valid"][:, -1, None], wheel_error[:, -1], np.nan)),
            "wheel_rate_recursive_trajectory_rmse": _rmse(
                np.where(prediction["wheel_valid"][:, :, None], wheel_error, np.nan)),
            "encoder_angle_increment_rmse": _rmse(np.where(
                prediction["wheel_valid"][:, :, None],
                prediction["wheel_angle_increment"]
                - prediction["truth_wheel_rate"] * DT_S / WHEEL_RADIUS_M,
                np.nan)),
            "nonfinite_recursive_values": int(
                np.count_nonzero(~np.isfinite(prediction["body"]))
                + np.count_nonzero(~np.isfinite(prediction["actuator"]))
                + np.count_nonzero(~np.isfinite(prediction["wheel_rate"]))),
            "maximum_predicted_body_speed_mps": _finite_max(np.linalg.norm(
                prediction["body"][:, :, :2], axis=2)),
            "maximum_absolute_yaw_rate_rps": _finite_max(np.abs(
                prediction["body"][:, :, 2])),
        }
        for horizon, step in (("0.75s", 30), ("1s", 40), ("2s", 80)):
            error = body_error[:, step - 1]
            run_report["body_recursive_rmse_at_horizon"][horizon] = {
                name: _rmse(error[:, index])
                for index, name in enumerate(BODY_NAMES)}
            run_report["body_recursive_trajectory_rmse_to_horizon"][horizon] = {
                name: _rmse(body_error[:, :step, index])
                for index, name in enumerate(BODY_NAMES)}
        for local_index, ref in enumerate(refs):
            capture_index, sequence_index, source_row = ref
            body_window_error = body_error[local_index]
            encoder_mask = prediction["wheel_valid"][local_index]
            wheel_window_error = wheel_error[local_index]
            angle_error = (prediction["wheel_angle_increment"][local_index]
                           - prediction["truth_wheel_rate"][local_index]
                           * DT_S / WHEEL_RADIUS_M)
            if np.any(encoder_mask):
                cumulative_angle_error = np.sum(
                    np.where(encoder_mask[:, None], angle_error, 0.0), axis=0)
                encoder_rate_rmse = _rmse(np.where(
                    encoder_mask[:, None], wheel_window_error, np.nan))
                encoder_increment_rmse = _rmse(np.where(
                    encoder_mask[:, None], angle_error, np.nan))
                cumulative_angle_rmse = _rmse(cumulative_angle_error)
            else:
                encoder_rate_rmse = None
                encoder_increment_rmse = None
                cumulative_angle_rmse = None
            per_window.append({
                "run_id": run_id,
                "capture": captures[capture_index].name,
                "sequence_index": int(sequence_index),
                "source_row_within_sequence": int(source_row),
                "support_features_observable_at_start": dict(zip(
                    SUPPORT_FEATURE_NAMES,
                    _support_features(captures[capture_index], sequence_index,
                                      source_row).astype(float).tolist())),
                "body_trajectory_rmse_2s": _rmse(body_window_error),
                "body_endpoint_abs_error_2s": {
                    name: float(abs(body_window_error[-1, index]))
                    for index, name in enumerate(BODY_NAMES)},
                "encoder_rate_rmse_2s": encoder_rate_rmse,
                "encoder_angle_increment_rmse_2s": encoder_increment_rmse,
                "encoder_cumulative_angle_error_2s_rad": (
                    cumulative_angle_error.astype(float).tolist()
                    if np.any(encoder_mask) else None),
                "encoder_cumulative_angle_rmse_2s_rad": cumulative_angle_rmse,
            })
        per_run[run_id] = run_report

    pooled: dict[str, Any] = {"independent_runs": len(per_run),
                              "per_run": per_run, "macro_run_metrics": {}}
    metric_paths = [
        ("one_step_body_rmse", name, "one_step_body_rmse", name)
        for name in BODY_NAMES]
    metric_paths += [
        ("one_step_actuator_rmse", name, "one_step_actuator_rmse", name)
        for name in ACTUATOR_NAMES]
    for horizon in ("0.75s", "1s", "2s"):
        metric_paths += [
            (f"body_endpoint_{horizon}", name,
             "body_recursive_rmse_at_horizon", horizon, name)
            for name in BODY_NAMES]
        metric_paths += [
            (f"body_trajectory_{horizon}", name,
             "body_recursive_trajectory_rmse_to_horizon", horizon, name)
            for name in BODY_NAMES]
    metric_paths += [("wheel_rate_endpoint_2s", "pair", "wheel_rate_rmse_at_2s")]
    metric_paths += [("wheel_rate_trajectory", "pair",
                      "wheel_rate_recursive_trajectory_rmse")]
    metric_paths += [("encoder_angle_increment", "pair",
                      "encoder_angle_increment_rmse")]
    for item in metric_paths:
        label, channel, *path = item
        values = {}
        for run_id, report in per_run.items():
            value: Any = report
            for key in path:
                value = value[key]
            if value is not None and np.isfinite(value):
                values[run_id] = float(value)
        array = list(values.values())
        pooled["macro_run_metrics"][f"{label}:{channel}"] = {
            "macro_run_mean": float(np.mean(array)) if array else None,
            "run_cluster_bootstrap_95pct_ci": _cluster_ci(
                array, SEED + seed_offset + len(label) + len(channel)),
            "independent_runs": len(array),
            "per_run": values,
        }
    pooled["per_window_recursive_errors"] = per_window
    pooled["support_feature_names"] = list(SUPPORT_FEATURE_NAMES)
    return pooled


def _finite_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if len(finite) else None


def _make_window_groups(captures: list[Capture]
                        ) -> tuple[dict[str, list[tuple[int, int, int]]],
                                   dict[str, list[tuple[int, int, int]]],
                                   dict[str, list[tuple[int, int, int]]]]:
    train = collect_windows(captures, {0}, {"train"})
    validation = collect_windows(captures, {0}, {"validation"})
    practice = collect_windows(captures, {1}, {"unseen_practice"})
    return train, select_eval_windows(validation), select_eval_windows(practice)


def run_ablation(dynamic: Path, dynamic_fixed: Path,
                 practice: Path, practice_fixed: Path,
                 dynamic_parent: Path, practice_parent: Path,
                 output: Path, steps: int = TRAIN_STEPS,
                 device: str = "auto") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WP19 result directory: {output}")
    if steps < 1:
        raise ValueError("WP19 training steps must be positive")
    captures = [
        load_capture("openplane", dynamic, dynamic_fixed, dynamic_parent,
                     {"train", "validation"}),
        load_capture("practice", practice, practice_fixed, practice_parent,
                     {"unseen_practice"}),
    ]
    train_windows, validation_windows, practice_windows = _make_window_groups(captures)
    if len(train_windows) < 3 or len(validation_windows) < 3:
        raise ValueError("WP19 needs >=3 train and >=3 held-out validation runs")
    if len(practice_windows) < 2:
        raise ValueError("WP19 requires the available two-run practice transfer view")
    stats = training_statistics(captures, train_windows)
    torch, nn = _torch_modules()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu":
        torch.set_num_threads(1)

    config = {
        "seed": SEED, "optimizer_steps_per_candidate": steps,
        "batch_size": BATCH_SIZE, "history_steps": HISTORY_STEPS,
        "history_seconds": HISTORY_STEPS * DT_S,
        "recursive_rollout_steps": ROLLOUT_STEPS,
        "recursive_rollout_seconds": ROLLOUT_STEPS * DT_S,
        "hidden_size": HIDDEN_SIZE,
        "same_encoder_capacity_and_optimizer_budget": True,
        "optimizer": "AdamW", "learning_rate": 3e-4,
        "weight_decay": 1e-5, "gradient_clip_norm": 1.0,
        "device": device,
        "loss_weights": {"body": 1.0, "actuator": 0.25,
                         "wheel_output": 0.25,
                         "body_acceleration_consistency": 0.10},
        "encoder_training_validity": "same adjacent current-and-next valid sidecar transitions for all W-A through W-D targets",
        "selected_history_note": (
            "WP18 selected no sufficient history; 2 s is held fixed here only "
            "to match the frozen parent for this controlled target ablation"),
    }
    output.mkdir(parents=True, exist_ok=False)
    all_candidate_results = []
    train_run_ids = sorted(train_windows)

    for candidate_index, body_target in enumerate(BODY_TARGETS):
        for wheel_index, wheel_target in enumerate(WHEEL_TARGETS):
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            model = make_model(nn).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                          weight_decay=1e-5)
            gradient_log: list[float] = []
            recent_losses: list[dict[str, float]] = []
            sample_rng = np.random.default_rng(SEED)
            for _ in range(steps):
                refs = []
                for _ in range(BATCH_SIZE):
                    run_id = train_run_ids[int(sample_rng.integers(
                        0, len(train_run_ids)))]
                    rows = train_windows[run_id]
                    refs.append(rows[int(sample_rng.integers(0, len(rows)))])
                recent_losses.append(_fit_one(
                    model, torch, optimizer, captures, refs, stats,
                    body_target, wheel_target, device, gradient_log))
            name = f"{candidate_index * len(WHEEL_TARGETS) + wheel_index:02d}_{body_target}__{wheel_target}"
            candidate_dir = output / name
            candidate_dir.mkdir()
            checkpoint = candidate_dir / "model.pt"
            torch.save({
                "state_dict": {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                "metadata": {
                    "body_target": body_target, "wheel_target": wheel_target,
                    "hidden_size": HIDDEN_SIZE,
                    "history_steps": HISTORY_STEPS,
                    "rollout_steps": ROLLOUT_STEPS,
                    "seed": SEED,
                    "training_runs": train_run_ids,
                    "validation_runs": sorted(validation_windows),
                    "practice_runs": sorted(practice_windows),
                },
            }, checkpoint)
            train_component_means = {
                key: float(np.mean([item[key] for item in recent_losses]))
                for key in recent_losses[0]}
            train_component_p90 = {
                key: float(np.quantile([item[key] for item in recent_losses], 0.90))
                for key in recent_losses[0]}
            result = {
                "candidate": name, "body_target": body_target,
                "wheel_target": wheel_target,
                "training_loss_mean": train_component_means,
                "training_loss_p90": train_component_p90,
                "gradient_norm_p50": float(np.quantile(gradient_log, 0.50)),
                "gradient_norm_p90": float(np.quantile(gradient_log, 0.90)),
                "gradient_norm_max": float(np.max(gradient_log)),
                "nonfinite_gradient_steps": int(np.count_nonzero(
                    ~np.isfinite(gradient_log))),
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation": _score_split(
                    model, torch, captures, validation_windows, stats,
                    body_target, wheel_target, device, candidate_index * 100
                    + wheel_index),
                "unseen_practice": _score_split(
                    model, torch, captures, practice_windows, stats,
                    body_target, wheel_target, device, candidate_index * 100
                    + wheel_index + 50),
            }
            all_candidate_results.append(result)
            progress = {
                "schema_version": 1,
                "status": "in_progress",
                "completed_candidates": len(all_candidate_results),
                "total_candidates": len(BODY_TARGETS) * len(WHEEL_TARGETS),
                "fixed_configuration": config,
                "candidates": all_candidate_results,
            }
            (output / "wp19_target_ablation_progress.json").write_text(
                json.dumps(progress, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

    report = {
        "schema_version": 1,
        "purpose": "WP19 controlled one-seed body-transition and rear-encoder output target ablation",
        "training_performed": True,
        "plant_promoted": False,
        "checkpoint_promoted": False,
        "test_or_final_test_opened_for_scoring": False,
        "future_truth_or_encoder_feedback_used_in_rollout": False,
        "current_wheel_measurement_policy": (
            "current filtered wheel output initializes only the separate wheel "
            "measurement-output branch; it is never fed to body transition"),
        "history_input_policy": (
            "observable bridge odometry, actuator feedback, and commands only; "
            "simulator truth is used only as supervised response and rollout "
            "initial state"),
        "body_frame": "rear axle; acceleration labels derived from simulator COM truth with explicit 0.15532 m rigid-body offset",
        "wheel_measurement_target": "fixed 25 ms causal encoder sidecar; angle-increment mode reconstructs rate using wheel radius and dt",
        "data": {
            "dynamic_dataset": str(dynamic.relative_to(ROOT)),
            "dynamic_sha256": sha256_file(dynamic),
            "dynamic_fixed_sidecar": str(dynamic_fixed.relative_to(ROOT)),
            "dynamic_fixed_sha256": sha256_file(dynamic_fixed),
            "dynamic_encoder_parent_dataset": str(
                dynamic_parent.relative_to(ROOT)),
            "dynamic_encoder_parent_sha256": sha256_file(dynamic_parent),
            "practice_dataset": str(practice.relative_to(ROOT)),
            "practice_sha256": sha256_file(practice),
            "practice_fixed_sidecar": str(practice_fixed.relative_to(ROOT)),
            "practice_fixed_sha256": sha256_file(practice_fixed),
            "practice_encoder_parent_dataset": str(
                practice_parent.relative_to(ROOT)),
            "practice_encoder_parent_sha256": sha256_file(practice_parent),
            "source_fixed_alignment_fields_equal": [
                "run_ids", "run_splits", "sequence_bounds",
                "sequence_run_index", "sequence_reset_index",
                "packet_sequence", "dt_s", "frames",
                "simulator_rigid_state"],
            "source_and_fixed_encoder_rates_expected_to_differ": True,
            "training_runs": train_run_ids,
            "validation_runs": sorted(validation_windows),
            "practice_runs": sorted(practice_windows),
            "training_windows_by_run": {
                run_id: len(rows) for run_id, rows in train_windows.items()},
            "validation_eval_windows_by_run": {
                run_id: len(rows) for run_id, rows in validation_windows.items()},
            "practice_eval_windows_by_run": {
                run_id: len(rows) for run_id, rows in practice_windows.items()},
        },
        "fixed_configuration": config,
        "target_statistics_train_split_only": {
            "physical_body": {
                "mean": stats["physical_body"][0].tolist(),
                "scale": stats["physical_body"][1].tolist()},
            "effective_acceleration": {
                "mean": stats["acceleration"][0].tolist(),
                "scale": stats["acceleration"][1].tolist()},
            "body": {key: {"mean": pair[0].tolist(),
                            "scale": pair[1].tolist()}
                     for key, pair in stats["body"].items()},
            "actuator": {"mean": stats["actuator"][0].tolist(),
                         "scale": stats["actuator"][1].tolist()},
            "wheel": {key: {"mean": pair[0].tolist(),
                            "scale": pair[1].tolist()}
                      for key, pair in stats["wheel"].items()},
            "input": {"mean": stats["input"][0].tolist(),
                      "scale": stats["input"][1].tolist()},
            "finite_training_transition_count": stats["training_target_rows"],
            "run_balanced_normalization_runs": stats[
                "training_normalization_runs"],
            "normalization_window_cap_per_run": stats[
                "normalization_window_cap_per_run"],
        },
        "candidates": all_candidate_results,
        "selection": {
            "status": "diagnostic results recorded; no automatic winner",
            "criteria": ["one-step body accuracy", "0.75–2 s recursive body accuracy",
                         "2 s encoder reconstruction", "numerical stability",
                         "target scale and gradient diagnostics"],
            "selection_note": (
                "Compare macro run-level validation results first, then check "
                "the two-run practice transfer separately. A single scalar "
                "or one-step score does not select a target representation."),
        },
    }
    report_path = output / "wp19_target_ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    (output / "wp19_target_ablation_progress.json").unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--dynamic-fixed", type=Path, default=DEFAULT_DYNAMIC_FIXED)
    parser.add_argument("--practice", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--practice-fixed", type=Path, default=DEFAULT_PRACTICE_FIXED)
    parser.add_argument("--dynamic-parent", type=Path, default=DEFAULT_DYNAMIC_PARENT)
    parser.add_argument("--practice-parent", type=Path, default=DEFAULT_PRACTICE_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_ablation(args.dynamic.resolve(), args.dynamic_fixed.resolve(),
                          args.practice.resolve(), args.practice_fixed.resolve(),
                          args.dynamic_parent.resolve(),
                          args.practice_parent.resolve(),
                          args.output.resolve(), args.steps, args.device)
    print(json.dumps({
        "report": str((args.output / "wp19_target_ablation_report.json").resolve()),
        "candidates": len(result["candidates"]),
        "train_runs": len(result["data"]["training_runs"]),
        "validation_runs": len(result["data"]["validation_runs"]),
        "practice_runs": len(result["data"]["practice_runs"]),
        "training_steps_per_candidate": result["fixed_configuration"][
            "optimizer_steps_per_candidate"],
        "device": result["fixed_configuration"]["device"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
