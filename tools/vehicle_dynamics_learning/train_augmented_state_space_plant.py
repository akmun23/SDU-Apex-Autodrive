#!/usr/bin/env python3
"""Train/evaluate WP22's single-seed augmented black-box plant candidate.

The default is the prescribed 120-step staged smoke, not an open-ended sweep.
Validation/test/final-test truth is never used as a model input or training
sample.  A failed pre-registered material-improvement gate ends this branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    HISTORY_STEPS,
    WHEEL_RADIUS_M,
    AugmentedStateSpacePlant,
    PlantConfig,
    encoder_rate_to_increment,
    load_checkpoint as load_plant_checkpoint,
    save_checkpoint,
)
from tools.vehicle_dynamics_learning.calibrate_wp20_support import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_DYNAMIC_PARENT,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    DEFAULT_PRACTICE_PARENT,
    _balanced_sample_bank,
    _normalize_runs,
    _training_feature_bank,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    BODY_NAMES,
    HIDDEN_SIZE,
    ROLLOUT_STEPS,
    SEED as WP19_SEED,
    TASK_ROOT,
    _make_window_groups,
    _run_balanced_normalizer,
    load_capture,
    make_model as make_wp19_model,
    select_eval_windows,
    sha256_file,
    training_statistics,
)


ROOT = Path(__file__).resolve().parents[2]
WP19_ROOT = TASK_ROOT / "wp19_target_ablation_v2_common_encoder_mask"
WP19_REPORT_PATH = WP19_ROOT / "wp19_target_ablation_report.json"
WP19_CHECKPOINT = (WP19_ROOT / "07_body_state_increment__encoder_angle_increment"
                   / "model.pt")
WP20_ROOT = TASK_ROOT / "wp20_support_calibration_v2_model_train_runs"
WP20_REPORT_PATH = WP20_ROOT / "wp20_support_calibration_report.json"
ACTUATOR_PARENT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    "/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002"
    "/encoder_raw_state_teacher_v1/edssm_gru_z32_e2_rollresidual_10s_yawonly_"
    "wheelonly_lowthrottle4_joint_lr3e5_seed101/best.pt")
DEFAULT_OUTPUT = TASK_ROOT / "wp22_augmented_state_space_seed101_smoke_v1"
EXPECTED_WP19_SHA256 = "045e98e1d0aafd9ec369bc6fdb1bdbce4e94491dec101a764f33195fe09e7541"
EXPECTED_ACTUATOR_PARENT_SHA256 = "8f84fa54306492bd9750e4e1e3fb0be014195491eccd8f4fd67f2943402bfe8e"
SEED = 101
BATCH_SIZE = 4
STAGE_STEPS = (40, 40, 40)
STAGE_HORIZONS = (20, 80, 200)
MAX_EVAL_WINDOWS_PER_RUN = 64
EVAL_HORIZONS = (("25ms", 1), ("250ms", 10), ("750ms", 30),
                 ("2s", 80), ("5s", 200), ("10s", 400))
BOOTSTRAP_REPLICATES = 5000
BODY_LOSS_WEIGHT = 1.0
ACCELERATION_CONSISTENCY_WEIGHT = 0.10
HEADING_LOSS_WEIGHT = 0.50
POSITION_LOSS_WEIGHT = 0.25
MEASUREMENT_LOSS_WEIGHT = 0.25
LATENT_LOSS_WEIGHT = 0.005
SUPPORT_LOSS_WEIGHT = 0.02
SYMMETRY_LOSS_WEIGHT = 0.0
POSITION_SCALE_M = 0.50
HEADING_SCALE_RAD = 0.10


@dataclass
class WP22Data:
    captures: list[Any]
    poses: list[np.ndarray]
    training_windows_80: dict[str, list[tuple[int, int, int]]]
    train_windows: dict[str, list[tuple[int, int, int]]]
    train_windows_by_horizon: dict[int, dict[str, list[tuple[int, int, int]]]]
    validation_windows: dict[str, list[tuple[int, int, int]]]
    practice_windows: dict[str, list[tuple[int, int, int]]]
    raw_sources: list[dict[str, np.ndarray]]
    training_runs: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wrap_angle(value: torch.Tensor | np.ndarray) -> Any:
    return torch.atan2(torch.sin(value), torch.cos(value)) if torch.is_tensor(value) \
        else np.arctan2(np.sin(value), np.cos(value))


def _load_data() -> tuple[WP22Data, dict[str, Any]]:
    wp19_report = _load_json(WP19_REPORT_PATH)
    if sha256_file(WP19_CHECKPOINT) != EXPECTED_WP19_SHA256:
        raise ValueError("selected WP19 B-B/W-D checkpoint hash changed")
    if wp19_report.get("experimental_status") == "superseded_not_for_selection":
        raise ValueError("WP22 refuses the superseded WP19 artifact")
    for path, report_key in (
            (DEFAULT_DYNAMIC, "dynamic_sha256"),
            (DEFAULT_DYNAMIC_FIXED, "dynamic_fixed_sha256"),
            (DEFAULT_PRACTICE, "practice_sha256"),
            (DEFAULT_PRACTICE_FIXED, "practice_fixed_sha256")):
        if sha256_file(path) != wp19_report["data"][report_key]:
            raise ValueError(f"WP22 refuses changed WP19 data input: {path}")
    wp20_report = _load_json(WP20_REPORT_PATH)
    if (wp20_report.get("selected_wp19_checkpoint_sha256") != EXPECTED_WP19_SHA256
            or not wp20_report.get("support_confidence_authorized")
            or wp20_report.get("gate_decision", {}).get("selected_support_definition")
            != "run_balanced_knn_distance"):
        raise ValueError("WP20 did not freeze the required calibrated kNN support")
    captures = [
        load_capture("openplane", DEFAULT_DYNAMIC, DEFAULT_DYNAMIC_FIXED,
                     DEFAULT_DYNAMIC_PARENT, {"train", "validation"}),
        load_capture("practice", DEFAULT_PRACTICE, DEFAULT_PRACTICE_FIXED,
                     DEFAULT_PRACTICE_PARENT, {"unseen_practice"}),
    ]
    raw_sources, poses = [], []
    for capture in captures:
        with np.load(capture.source_path, allow_pickle=False) as archive:
            raw_sources.append({
                "run_families": np.asarray(archive["run_families"]).astype(str),
                "training_families": np.asarray(
                    archive.get("training_families", archive["run_families"])).astype(str),
                "sequence_condition_id": np.asarray(
                    archive["sequence_condition_id"], dtype=np.int64),
                "training_family_names": np.asarray(
                    archive.get("training_family_names", np.empty(0))).astype(str),
                "training_family_probabilities": np.asarray(
                    archive.get("training_family_probabilities", np.empty(0)),
                    dtype=np.float64),
            })
            pose = np.asarray(archive["simulator_pose_xyyaw"], dtype=np.float32)
            poses.append(pose)
    train80, validation80, practice80 = _make_window_groups(captures)
    checkpoint = torch.load(WP19_CHECKPOINT, map_location="cpu", weights_only=True)
    training_runs = tuple(sorted(checkpoint["metadata"].get("training_runs", [])))
    if set(train80) != set(training_runs):
        raise ValueError("WP19 optimizer-run list does not match eligible training windows")
    train_windows_by_horizon = {}
    for horizon in (20, 80, 200):
        stage_windows = _collect_horizon_windows(
            captures, raw_sources, poses, {"train"}, horizon,
            capture_indices={0})
        train_windows_by_horizon[horizon] = {
            run_id: refs for run_id, refs in stage_windows.items()
            if run_id in training_runs}
        if len(train_windows_by_horizon[horizon]) < 5:
            raise ValueError(f"WP22 has too few independent runs at {horizon} steps")
    train_windows = train_windows_by_horizon[200]
    validation_windows = _collect_horizon_windows(
        captures, raw_sources, poses, {"validation"}, 400,
        capture_indices={0})
    validation_windows = _select_eval_windows(validation_windows,
                                               MAX_EVAL_WINDOWS_PER_RUN)
    practice_windows = _collect_horizon_windows(
        captures, raw_sources, poses, {"unseen_practice"}, 200,
        capture_indices={1})
    practice_windows = _select_eval_windows(practice_windows,
                                             MAX_EVAL_WINDOWS_PER_RUN)
    if len(validation_windows) < 3 or len(practice_windows) < 2:
        raise ValueError("WP22 requires six validation and two practice runs")
    return WP22Data(captures, poses, train80, train_windows,
                    train_windows_by_horizon,
                    validation_windows, practice_windows, raw_sources,
                    training_runs), wp20_report


def _collect_horizon_windows(captures, raw_sources, poses, splits: set[str],
                             horizon: int, capture_indices: set[int]
                             ) -> dict[str, list[tuple[int, int, int]]]:
    by_run: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for capture_index in sorted(capture_indices):
        capture = captures[capture_index]
        source = raw_sources[capture_index]
        conditions = source["sequence_condition_id"]
        for sequence_index, ((start_raw, end_raw), run_raw, condition) in enumerate(
                zip(capture.bounds, capture.sequence_run, conditions)):
            start, end, run = int(start_raw), int(end_raw), int(run_raw)
            run_id = str(capture.run_ids[run])
            if str(capture.splits[run]) not in splits:
                continue
            if end - start < HISTORY_STEPS + horizon:
                continue
            local_packet = capture.packet[start:end]
            if (not np.all(np.isfinite(capture.input_features[start:end]))
                    or not np.all(np.isfinite(capture.body[start:end]))
                    or not np.all(np.isfinite(poses[capture_index][start:end]))
                    or not np.allclose(capture.dt_s[start:end], DT_S,
                                       rtol=0.0, atol=1e-7)
                    or np.any(np.diff(local_packet) != 1)):
                continue
            # Keep the condition id in a side map attached to the call-local
            # index registry; no samples cross sequence/reset boundaries.
            for source_row in range(HISTORY_STEPS - 1,
                                    end - start - horizon):
                by_run[run_id].append((capture_index, sequence_index, source_row))
    return dict(by_run)


def _select_eval_windows(by_run: dict[str, list[tuple[int, int, int]]],
                         maximum: int) -> dict[str, list[tuple[int, int, int]]]:
    return select_eval_windows(by_run, maximum)


def _history_at(capture, sequence_index: int, source_row: int) -> np.ndarray:
    begin = int(capture.bounds[sequence_index, 0]) + source_row
    base = capture.input_features[begin - HISTORY_STEPS + 1:begin + 1]
    rates = capture.encoder_rate[begin - HISTORY_STEPS + 1:begin + 1]
    valid = capture.encoder_valid[begin - HISTORY_STEPS + 1:begin + 1]
    increments = rates * (DT_S / WHEEL_RADIUS_M)
    increments = np.where(valid[:, None], increments, 0.0)
    return np.column_stack((base, increments, valid.astype(np.float32))).astype(
        np.float32, copy=False)


def _balanced_stats(groups: dict[str, list[np.ndarray]], floor: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    arrays = {run: np.concatenate(values, axis=0) for run, values in groups.items()
              if values and sum(len(value) for value in values) > 0}
    return _run_balanced_normalizer(arrays, floor)


def _training_windows_and_stats(data: WP22Data, wp20_report: dict[str, Any]
                                ) -> tuple[PlantConfig, dict[str, Any], dict[str, Any]]:
    histories_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    states_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    commands_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    body_deltas_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    body_states_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    encoder_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    acceleration_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    sample_refs: dict[str, list[tuple[int, int, int]]] = {}
    for run_id, refs in sorted(data.training_windows_80.items()):
        if len(refs) > 512:
            indices = np.linspace(0, len(refs) - 1, 512, dtype=np.int64)
            refs = [refs[int(index)] for index in indices]
        sample_refs[run_id] = refs
        for capture_index, sequence_index, source_row in refs:
            capture = data.captures[capture_index]
            begin = int(capture.bounds[sequence_index, 0]) + source_row
            histories_by_run[run_id].append(_history_at(
                capture, sequence_index, source_row))
            states_by_run[run_id].append(np.concatenate((
                capture.body[begin], capture.frames[begin, 3:5]))[None, :])
            body_states_by_run[run_id].append(capture.body[begin][None, :])
            commands_by_run[run_id].append(capture.frames[begin, 7:9][None, :])
    for run_id, refs in sorted(data.training_windows_80.items()):
        if run_id not in data.training_runs:
            continue
        if len(refs) > 4096:
            refs = [refs[int(index)] for index in np.linspace(
                0, len(refs)-1, 4096, dtype=np.int64)]
        for capture_index, sequence_index, source_row in refs:
            capture = data.captures[capture_index]
            begin = int(capture.bounds[sequence_index, 0]) + source_row
            body_deltas_by_run[run_id].append(
                capture.body[begin + 1] - capture.body[begin])
            if capture.encoder_valid[begin + 1]:
                encoder_by_run[run_id].append(
                    capture.encoder_rate[begin + 1] * (DT_S / WHEEL_RADIUS_M))
            acceleration_by_run[run_id].append(
                capture.acceleration[begin][None, :])

    history_mean, history_scale = _balanced_stats(histories_by_run, 1e-2)
    state_mean, state_scale = _balanced_stats(states_by_run, 1e-2)
    normalized_body_states = {
        run_id: [((value[0] - state_mean[:3]) / state_scale[:3])
                 for value in values]
        for run_id, values in body_states_by_run.items()}
    body_state_limits, body_state_quantiles = _balanced_quantile_limits(
        normalized_body_states, margin=1.25)
    command_mean, command_scale = _balanced_stats(commands_by_run, 1e-2)
    body_limits, body_quantiles = _balanced_quantile_limits(
        body_deltas_by_run, margin=1.25)
    measurement_limits, measurement_quantiles = _balanced_quantile_limits(
        encoder_by_run, margin=1.25)
    acceleration_mean, acceleration_scale = _balanced_stats(
        acceleration_by_run, 0.1)

    support_norm = wp20_report["normalization_train_only"]
    support_meta = wp20_report["estimators"]["run_balanced_knn_distance"]
    support_thresholds = support_meta[
        "candidate_thresholds_by_validation_percentile"]
    supported_upper = float(support_thresholds["supported_upper_score"])
    weak_upper = float(support_thresholds["weak_support_upper_score"])
    training_bank, bank_counts = _training_feature_bank(
        data.captures, data.training_windows_80,
        allowed_run_ids=set(data.training_runs))
    if set(training_bank) != set(data.training_runs):
        raise ValueError("WP20 support bank does not match WP19 optimizer runs")
    sampled_bank = _balanced_sample_bank(training_bank)
    normalized_bank = _normalize_runs(
        sampled_bank, np.asarray(support_norm["mean"], dtype=np.float32),
        np.asarray(support_norm["scale"], dtype=np.float32))
    bank_run_ids = tuple(sorted(normalized_bank))
    support_points, support_run_index = [], []
    for index, run_id in enumerate(bank_run_ids):
        points = np.asarray(normalized_bank[run_id], dtype=np.float32)
        support_points.extend(points.tolist())
        support_run_index.extend([index] * len(points))

    if sha256_file(ACTUATOR_PARENT) != EXPECTED_ACTUATOR_PARENT_SHA256:
        raise ValueError("frozen parent actuator-fit checkpoint hash changed")
    # The frozen, hash-verified local checkpoint stores NumPy arrays in metadata.
    parent = torch.load(ACTUATOR_PARENT, map_location="cpu", weights_only=False)
    actuator_fit = parent["metadata"]["actuator_fit"]
    actuator_fit_summary = {
        name: {"delay_steps": int(actuator_fit[name]["delay_steps"]),
               "alpha": float(actuator_fit[name]["alpha"])}
        for name in ("steering", "throttle")}
    if actuator_fit_summary["steering"]["delay_steps"] != 1 \
            or actuator_fit_summary["throttle"]["delay_steps"] != 1:
        raise ValueError("WP21 parent actuator timing differs from frozen record")
    config = PlantConfig(
        history_steps=HISTORY_STEPS,
        latent_size=8,
        history_mean=tuple(history_mean.tolist()),
        history_scale=tuple(history_scale.tolist()),
        state_mean=tuple(state_mean.tolist()),
        state_scale=tuple(state_scale.tolist()),
        command_mean=tuple(command_mean.tolist()),
        command_scale=tuple(command_scale.tolist()),
        body_increment_limit=tuple(body_limits.tolist()),
        body_state_normalized_limit=tuple(body_state_limits.tolist()),
        encoder_increment_limit=tuple(measurement_limits.tolist()),
        steering_delay_steps=actuator_fit_summary["steering"]["delay_steps"],
        throttle_delay_steps=actuator_fit_summary["throttle"]["delay_steps"],
        steering_alpha=actuator_fit_summary["steering"]["alpha"],
        throttle_alpha=actuator_fit_summary["throttle"]["alpha"],
        support_mean=tuple(float(value) for value in support_norm["mean"]),
        support_scale=tuple(float(value) for value in support_norm["scale"]),
        support_bank=tuple(tuple(row) for row in support_points),
        support_run_index=tuple(support_run_index),
        support_run_ids=bank_run_ids,
        support_supported_upper=supported_upper,
        support_weak_upper=weak_upper,
        support_calibrated=True,
        support_calibration_scope=(
            "WP20 composite normalized 2-second body-trajectory-error rank; "
            "not a per-channel error bound"),
    )
    scales = {
        "body_state_scale": state_scale[:3].astype(float).tolist(),
        "acceleration_mean": acceleration_mean.astype(float).tolist(),
        "acceleration_scale": acceleration_scale.astype(float).tolist(),
        "measurement_loss_scale": measurement_limits.astype(float).tolist(),
        "position_scale_m": POSITION_SCALE_M,
        "heading_scale_rad": HEADING_SCALE_RAD,
    }
    diagnostics = {
        "train_runs": sorted(data.training_runs),
        "normalization_sample_windows_per_run": 512,
        "body_increment_q995_abs_by_channel": body_quantiles,
        "body_state_normalized_q995_abs_by_channel": body_state_quantiles,
        "encoder_increment_q995_abs_by_channel": measurement_quantiles,
        "bounded_residual_limits_with_25pct_margin": {
            "body_delta": body_limits.astype(float).tolist(),
            "encoder_increment": measurement_limits.astype(float).tolist()},
        "support_bank_rows": int(len(support_points)),
        "support_bank_run_counts": bank_counts,
        "support_thresholds": {"supported_upper": supported_upper,
                                "weak_upper": weak_upper},
        "actuator_fit": actuator_fit_summary,
        "actuator_fit_checkpoint_sha256": EXPECTED_ACTUATOR_PARENT_SHA256,
        "loss_scales": scales,
    }
    return config, scales, diagnostics


def _balanced_quantile_limits(groups: dict[str, list[np.ndarray]], margin: float
                              ) -> tuple[np.ndarray, list[list[float]]]:
    per_run = {}
    for run_id, values in sorted(groups.items()):
        if not values:
            continue
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if not np.isfinite(matrix).all():
            matrix = matrix[np.isfinite(matrix).all(axis=1)]
        if len(matrix) > 10000:
            matrix = matrix[np.linspace(0, len(matrix)-1, 10000,
                                        dtype=np.int64)]
        per_run[run_id] = np.quantile(np.abs(matrix), 0.995, axis=0)
    if len(per_run) < 3:
        raise ValueError("label-bound fitting requires >=3 independent training runs")
    run_quantiles = np.stack(list(per_run.values()))
    q995 = np.quantile(run_quantiles, 0.5, axis=0)
    limits = np.maximum(q995 * margin, 1e-5)
    return limits.astype(np.float32), run_quantiles.astype(float).tolist()


def _sample_windows(data: WP22Data, horizon: int, batch_size: int,
                    rng: np.random.Generator, counters: dict[str, Counter]
                    ) -> list[tuple[int, int, int]]:
    # Each curriculum stage draws from starts with a complete target horizon;
    # short captures are retained in the shorter stages only.
    groups = {run: refs for run, refs in data.train_windows_by_horizon[horizon].items()
              if run in data.training_runs and refs}
    family_for_run: dict[str, str] = {}
    conditions_by_run: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
    probabilities: dict[str, float] = {}
    source = data.raw_sources[0]
    run_families = source["training_families"]
    run_ids = data.captures[0].run_ids
    for index, run_id in enumerate(run_ids):
        if str(run_id) in groups:
            family_for_run[str(run_id)] = str(run_families[index])
    by_family: dict[str, list[str]] = defaultdict(list)
    for run_id in groups:
        by_family[family_for_run.get(run_id, "unclassified")].append(run_id)
    family_names = sorted(by_family)
    # Equal family probability preserves the requested family->run->condition
    # hierarchy without allowing a long family capture to dominate by rows.
    family_weights = np.ones(len(family_names), dtype=np.float64)
    family_weights /= family_weights.sum()
    theoretical_run_probabilities = {
        run_id: float(family_weights[family_names.index(family_for_run[run_id])]
                      / len(by_family[family_for_run[run_id]]))
        for run_id in groups}
    if max(theoretical_run_probabilities.values()) > 0.25:
        raise ValueError("family/run sampler would let one run dominate draws")
    for run_id, refs in groups.items():
        local: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for ref in refs:
            _, seq, _ = ref
            local[int(source["sequence_condition_id"][seq])].append(ref)
        conditions_by_run[run_id] = dict(local)
    batch = []
    for _ in range(batch_size):
        family_index = int(rng.choice(len(family_names), p=family_weights))
        family = family_names[family_index]
        run_id = str(rng.choice(by_family[family]))
        condition_ids = sorted(conditions_by_run[run_id])
        condition = int(rng.choice(condition_ids))
        refs = conditions_by_run[run_id][condition]
        ref = refs[int(rng.integers(len(refs)))]
        batch.append(ref)
        counters["family"][family] += 1
        counters["run"][run_id] += 1
        counters["condition"][f"{run_id}:{condition}"] += 1
    return batch


def _batch_arrays(data: WP22Data, refs, horizon: int):
    histories, initial_states, initial_poses = [], [], []
    commands, target_body, target_pose, target_encoder = [], [], [], []
    encoder_masks, target_accel = [], []
    for capture_index, sequence_index, source_row in refs:
        capture = data.captures[capture_index]
        begin = int(capture.bounds[sequence_index, 0]) + source_row
        end = begin + horizon
        histories.append(_history_at(capture, sequence_index, source_row))
        initial_states.append(np.r_[capture.body[begin], capture.frames[begin, 3:5]])
        initial_poses.append(data.poses[capture_index][begin])
        commands.append(capture.frames[begin:end, 7:9])
        target_body.append(capture.body[begin+1:end+1])
        target_pose.append(data.poses[capture_index][begin+1:end+1])
        target_encoder.append(capture.encoder_rate[begin+1:end+1]
                              * (DT_S / WHEEL_RADIUS_M))
        encoder_masks.append(capture.encoder_valid[begin+1:end+1])
        target_accel.append(capture.acceleration[begin:end])
    float_arrays = [histories, initial_states, initial_poses, commands,
                    target_body, target_pose, target_encoder, target_accel]
    result = [np.asarray(value, dtype=np.float32) for value in float_arrays]
    result.insert(7, np.asarray(encoder_masks, dtype=bool))
    return tuple(result)


def _implied_acceleration(predicted_body: torch.Tensor,
                          initial_body: torch.Tensor) -> torch.Tensor:
    current = torch.cat((initial_body[:, None, :], predicted_body[:, :-1]), dim=1)
    delta = predicted_body - current
    u, v, r = current.unbind(dim=-1)
    du, dv, dr = delta.unbind(dim=-1)
    # Rear-axle velocity labels converted from the aligned COM truth.
    from tools.vehicle_dynamics_learning.signal_semantics import (
        REAR_AXLE_TO_COM_X_M,
    )
    alpha = dr / DT_S
    ax = du / DT_S - r * (v + REAR_AXLE_TO_COM_X_M * r)
    ay = dv / DT_S + r * u + REAR_AXLE_TO_COM_X_M * alpha
    return torch.stack((ax, ay, alpha), dim=-1)


def _train_step(model: AugmentedStateSpacePlant, data: WP22Data,
                refs, horizon: int, scales: dict[str, Any], optimizer,
                device: torch.device,
                support_loss_weight: float = SUPPORT_LOSS_WEIGHT
                ) -> dict[str, float]:
    arrays = _batch_arrays(data, refs, horizon)
    tensors = [torch.as_tensor(value,
                               dtype=torch.bool if index == 7 else torch.float32,
                               device=device)
               for index, value in enumerate(arrays)]
    history, initial, pose, commands, truth_body, truth_pose, truth_enc, enc_mask, truth_accel = tensors
    model.train()
    model.reset(history, initial, pose)
    predicted = model.rollout(commands)
    body = predicted["states"][..., :3]
    body_scale = torch.tensor(scales["body_state_scale"], device=device)
    normalized_body_error = (body - truth_body) / body_scale
    body_loss = F.smooth_l1_loss(normalized_body_error, torch.zeros_like(body))
    predicted_pose = predicted["poses"]
    position_error = torch.linalg.vector_norm(
        predicted_pose[..., :2] - truth_pose[..., :2], dim=-1) / POSITION_SCALE_M
    heading_error = _wrap_angle(predicted_pose[..., 2] - truth_pose[..., 2]) \
        / HEADING_SCALE_RAD
    position_loss = F.smooth_l1_loss(position_error,
                                      torch.zeros_like(position_error))
    heading_loss = F.smooth_l1_loss(heading_error,
                                     torch.zeros_like(heading_error))
    encoder_scale = torch.tensor(scales["measurement_loss_scale"], device=device)
    encoder_error = (predicted["measurements"] - truth_enc) / encoder_scale
    if torch.any(enc_mask):
        measurement_loss = F.smooth_l1_loss(
            encoder_error[enc_mask], torch.zeros_like(encoder_error[enc_mask]))
    else:
        measurement_loss = body_loss.new_zeros(())
    acceleration_mean = torch.tensor(scales["acceleration_mean"], device=device)
    acceleration_scale = torch.tensor(scales["acceleration_scale"], device=device)
    implied = _implied_acceleration(body, initial[:, :3])
    acceleration_error = ((implied - acceleration_mean) / acceleration_scale
                          - (truth_accel - acceleration_mean) / acceleration_scale)
    acceleration_loss = F.smooth_l1_loss(
        acceleration_error, torch.zeros_like(acceleration_error))
    latent = predicted["latents"]
    latent_loss = (torch.mean(latent.square()) if latent.shape[-1]
                   else body_loss.new_zeros(()))
    if latent.shape[-1] and latent.shape[1] > 1:
        latent_loss = latent_loss + 0.1 * torch.mean(
            (latent[:, 1:] - latent[:, :-1]).square())
    body_limits = torch.tensor(model.config.body_increment_limit, device=device)
    normalized_increment = predicted["body_increments"] / body_limits
    low_support = 1.0 - predicted["support_confidence"].detach()
    support_loss = torch.mean(low_support[..., None]
                              * normalized_increment.square())
    loss = (BODY_LOSS_WEIGHT * body_loss
            + ACCELERATION_CONSISTENCY_WEIGHT * acceleration_loss
            + HEADING_LOSS_WEIGHT * heading_loss
            + POSITION_LOSS_WEIGHT * position_loss
            + MEASUREMENT_LOSS_WEIGHT * measurement_loss
            + LATENT_LOSS_WEIGHT * latent_loss
            + support_loss_weight * support_loss)
    if not torch.isfinite(loss):
        raise FloatingPointError("WP22 encountered a non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("WP22 encountered non-finite gradients")
    optimizer.step()
    return {"loss": float(loss.detach()), "body": float(body_loss.detach()),
            "position": float(position_loss.detach()),
            "heading": float(heading_loss.detach()),
            "measurement": float(measurement_loss.detach()),
            "acceleration_consistency": float(acceleration_loss.detach()),
            "latent": float(latent_loss.detach()),
            "support": float(support_loss.detach()),
            "gradient_norm_preclip": float(grad_norm.detach())}


def _integrate_numpy_batch(pose: np.ndarray, body: np.ndarray,
                           dt_s: float = DT_S) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    body = np.asarray(body, dtype=np.float64)
    x, y, heading = np.moveaxis(pose, -1, 0)
    u, v, yaw_rate = np.moveaxis(body, -1, 0)
    dtheta = yaw_rate * dt_s
    denominator = np.where(np.abs(yaw_rate) > 1e-7, yaw_rate, 1.0)
    dx_turn = (u * np.sin(dtheta) + v * (np.cos(dtheta) - 1.0)) / denominator
    dy_turn = (u * (1.0 - np.cos(dtheta)) + v * np.sin(dtheta)) / denominator
    dx = np.where(np.abs(yaw_rate) > 1e-7, dx_turn, u * dt_s)
    dy = np.where(np.abs(yaw_rate) > 1e-7, dy_turn, v * dt_s)
    return np.stack((x + np.cos(heading) * dx - np.sin(heading) * dy,
                     y + np.sin(heading) * dx + np.cos(heading) * dy,
                     np.arctan2(np.sin(heading + dtheta),
                                np.cos(heading + dtheta))), axis=-1).astype(np.float32)


def _predict_wp19_baseline(capture, pose_truth: np.ndarray, refs,
                           horizon: int, stats: dict[str, Any], model,
                           device: torch.device) -> dict[str, np.ndarray]:
    input_mean, input_scale = [torch.as_tensor(value, dtype=torch.float32,
                                               device=device)
                               for value in stats["input"]]
    body_mean, body_scale = [torch.as_tensor(value, dtype=torch.float32,
                                             device=device)
                             for value in stats["body"]["body_state_increment"]]
    actuator_mean, actuator_scale = [torch.as_tensor(value, dtype=torch.float32,
                                                     device=device)
                                    for value in stats["actuator"]]
    wheel_mean, wheel_scale = [torch.as_tensor(value, dtype=torch.float32,
                                               device=device)
                               for value in stats["wheel"]["encoder_angle_increment"]]
    hist, body, actuator, command, poses = [], [], [], [], []
    truth_body, truth_pose, truth_enc, masks = [], [], [], []
    for capture_index, sequence_index, source_row in refs:
        begin = int(capture.bounds[sequence_index, 0]) + source_row
        end = begin + horizon
        hist.append(capture.input_features[begin-HISTORY_STEPS+1:begin+1])
        body.append(capture.body[begin])
        actuator.append(capture.frames[begin, 3:5])
        command.append(capture.frames[begin:end, 7:9])
        poses.append(pose_truth[begin])
        truth_body.append(capture.body[begin+1:end+1])
        truth_pose.append(pose_truth[begin+1:end+1])
        truth_enc.append(capture.encoder_rate[begin+1:end+1]
                         * (DT_S / WHEEL_RADIUS_M))
        masks.append(capture.encoder_valid[begin+1:end+1])
    as_t = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32,
                                         device=device)
    hist_t, body_t, actuator_t, command_t = map(as_t, (hist, body, actuator, command))
    hidden = torch.zeros(len(refs), HIDDEN_SIZE, device=device)
    prediction_body, prediction_actuator, prediction_pose, prediction_encoder = [], [], [], []
    state = body_t
    act = actuator_t
    current_command = command_t[:, 0]
    pose_t = as_t(poses)
    model.eval()
    with torch.no_grad():
        for step in range(HISTORY_STEPS - 1):
            normalized = (hist_t[:, step] - input_mean) / input_scale
            _, hidden = model.step(normalized, hidden)
        for step in range(horizon):
            physical = torch.cat((state, act, current_command), dim=1)
            output, hidden = model.step((physical - input_mean) / input_scale,
                                        hidden)
            # Preserve the baseline's exact target interpretation in torch.
            state_next = state + output[:, :3] * body_scale + body_mean
            next_act = output[:, 3:5] * actuator_scale + actuator_mean
            # Evaluate its measurement head as a direct synthetic sensor output.
            encoder_next = output[:, 5:7] * wheel_scale + wheel_mean
            prediction_body.append(state_next.cpu().numpy())
            prediction_actuator.append(next_act.cpu().numpy())
            prediction_encoder.append(encoder_next.cpu().numpy())
            pose_next = torch.as_tensor(
                _integrate_numpy_batch(pose_t.cpu().numpy(), state.cpu().numpy()),
                device=device)
            prediction_pose.append(pose_next.cpu().numpy())
            state, act, pose_t = state_next, next_act, pose_next
            if step + 1 < horizon:
                current_command = command_t[:, step + 1]
    return {
        "body": np.asarray(prediction_body, dtype=np.float32).transpose(1, 0, 2),
        "pose": np.asarray(prediction_pose, dtype=np.float32).transpose(1, 0, 2),
        "encoder": np.asarray(prediction_encoder, dtype=np.float32).transpose(1, 0, 2),
        "truth_body": np.asarray(truth_body, dtype=np.float32),
        "truth_pose": np.asarray(truth_pose, dtype=np.float32),
        "truth_encoder": np.asarray(truth_enc, dtype=np.float32),
        "encoder_valid": np.asarray(masks, dtype=bool),
        "truth_actuator": np.asarray([
            capture.frames[int(capture.bounds[seq, 0])+row+1:
                           int(capture.bounds[seq, 0])+row+horizon+1, 3:5]
            for _, seq, row in refs], dtype=np.float32),
    }


def _predict_candidate(model, data: WP22Data, refs, horizon: int,
                       device: torch.device) -> dict[str, np.ndarray]:
    arrays = _batch_arrays(data, refs, horizon)
    tensors = [torch.as_tensor(value, dtype=torch.float32, device=device)
               for value in arrays]
    history, initial, pose, commands, truth_body, truth_pose, truth_enc, masks, _ = tensors
    model.eval()
    with torch.no_grad():
        model.reset(history, initial, pose)
        prediction = model.rollout(commands)
    return {
        "body": prediction["states"][..., :3].cpu().numpy(),
        "pose": prediction["poses"].cpu().numpy(),
        "encoder": prediction["measurements"].cpu().numpy(),
        "support_score": prediction["support_score"].cpu().numpy(),
        "truth_body": truth_body.cpu().numpy(),
        "truth_pose": truth_pose.cpu().numpy(),
        "truth_encoder": truth_enc.cpu().numpy(),
        "encoder_valid": masks.cpu().numpy().astype(bool),
        "actuator": prediction["states"][..., 3:5].cpu().numpy(),
        "support": prediction["support_confidence"].cpu().numpy(),
    }


def _per_window_metrics(prediction: dict[str, np.ndarray], horizon: int
                        ) -> list[dict[str, float]]:
    result = []
    for index in range(len(prediction["body"])):
        body_error = prediction["body"][index, :horizon] \
            - prediction["truth_body"][index, :horizon]
        pose_error = prediction["pose"][index, :horizon] \
            - prediction["truth_pose"][index, :horizon]
        pose_error[:, 2] = _wrap_angle(pose_error[:, 2])
        radial = np.linalg.norm(pose_error[:, :2], axis=1)
        result.append({
            "position_radial_trajectory_rmse_m": float(np.sqrt(np.mean(radial**2))),
            "position_endpoint_error_m": float(radial[-1]),
            "heading_trajectory_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2]**2))),
            "heading_endpoint_error_rad": float(abs(pose_error[-1, 2])),
            "u_rmse_mps": float(np.sqrt(np.mean(body_error[:, 0]**2))),
            "v_rmse_mps": float(np.sqrt(np.mean(body_error[:, 1]**2))),
            "yaw_rate_rmse_rps": float(np.sqrt(np.mean(body_error[:, 2]**2))),
            "u_endpoint_abs_error_mps": float(abs(body_error[-1, 0])),
            "yaw_endpoint_abs_error_rps": float(abs(body_error[-1, 2])),
            "encoder_angle_increment_rmse_rad": float(np.sqrt(np.mean(
                (prediction["encoder"][index, :horizon]
                 - prediction["truth_encoder"][index, :horizon])[
                     prediction["encoder_valid"][index, :horizon]] ** 2)))
                if np.any(prediction["encoder_valid"][index, :horizon]) else float("nan"),
            "maximum_predicted_speed_mps": float(np.max(np.linalg.norm(
                prediction["body"][index, :horizon, :2], axis=1))),
            "maximum_absolute_predicted_yaw_rate_rps": float(np.max(np.abs(
                prediction["body"][index, :horizon, 2]))),
        })
    return result


def _bootstrap_run_mean(values: dict[str, float], seed: int) -> list[float] | None:
    sample = np.asarray([value for value in values.values() if np.isfinite(value)])
    if not len(sample):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sample), size=(BOOTSTRAP_REPLICATES, len(sample)))
    return np.quantile(sample[draws].mean(axis=1), [0.025, 0.975]).astype(float).tolist()


def _evaluate(data: WP22Data, model: AugmentedStateSpacePlant,
              baseline_model, baseline_stats, device: torch.device,
              progress_label: str = "WP22"
              ) -> dict[str, Any]:
    split_specs = (("openplane_validation", data.validation_windows),
                   ("practice_transfer", data.practice_windows))
    report: dict[str, Any] = {"splits": {}, "paired_run_deltas": {}}
    per_method_run: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "candidate": {}, "wp19_parent": {}}
    for split_name, windows_by_run in split_specs:
        split_report = {"independent_runs": len(windows_by_run), "per_run": {}}
        for run_index, (run_id, refs) in enumerate(sorted(windows_by_run.items())):
            print(f"{progress_label} scoring {split_name} run {run_index + 1}/"
                  f"{len(windows_by_run)}: {run_id}", flush=True)
            max_horizon = min(400, min(
                int(data.captures[c].bounds[seq, 1]
                    - (int(data.captures[c].bounds[seq, 0]) + row))
                for c, seq, row in refs) - 1)
            max_horizon = max(1, max_horizon)
            candidate = _predict_candidate(model, data, refs, max_horizon, device)
            baseline = _predict_wp19_baseline(
                data.captures[refs[0][0]], data.poses[refs[0][0]], refs,
                max_horizon, baseline_stats, baseline_model, device)
            per_horizon = {}
            for horizon_name, requested in EVAL_HORIZONS:
                horizon = min(requested, max_horizon)
                if requested > max_horizon:
                    continue
                candidate_metrics = _per_window_metrics(candidate, horizon)
                baseline_metrics = _per_window_metrics(baseline, horizon)
                run_values = {}
                for metric in candidate_metrics[0]:
                    candidate_mean = float(np.mean([row[metric]
                                                    for row in candidate_metrics]))
                    baseline_mean = float(np.mean([row[metric]
                                                   for row in baseline_metrics]))
                    run_values[f"candidate_{metric}"] = candidate_mean
                    run_values[f"wp19_parent_{metric}"] = baseline_mean
                candidate_divergence = [
                    _first_divergence_seconds(candidate, index, 0.5)
                    for index in range(len(refs))]
                baseline_divergence = [
                    _first_divergence_seconds(baseline, index, 0.5)
                    for index in range(len(refs))]
                per_horizon[horizon_name] = {
                    "windows": len(refs),
                    "candidate_macro_window_metrics": {
                        key.removeprefix("candidate_"): value
                        for key, value in run_values.items()
                        if key.startswith("candidate_")},
                    "wp19_parent_macro_window_metrics": {
                        key.removeprefix("wp19_parent_"): value
                        for key, value in run_values.items()
                        if key.startswith("wp19_parent_")},
                    "candidate_first_divergence_0p5m_s": (
                        float(np.median([v for v in candidate_divergence
                                         if v is not None]))
                        if any(v is not None for v in candidate_divergence) else None),
                    "candidate_first_divergence_censored_windows": sum(
                        value is None for value in candidate_divergence),
                    "wp19_parent_first_divergence_0p5m_s": (
                        float(np.median([v for v in baseline_divergence
                                         if v is not None]))
                        if any(v is not None for v in baseline_divergence) else None),
                    "wp19_parent_first_divergence_censored_windows": sum(
                        value is None for value in baseline_divergence),
                    "candidate_nonfinite_count": int(
                        (not np.isfinite(candidate["body"][:, :horizon]).all())
                        + (not np.isfinite(candidate["pose"][:, :horizon]).all())),
                }
                for method, metrics in (("candidate", candidate_metrics),
                                        ("wp19_parent", baseline_metrics)):
                    per_method_run[method].setdefault(horizon_name, {})[run_id] = {
                        key: float(np.mean([item[key] for item in metrics]))
                        for key in metrics[0]}
            split_report["per_run"][run_id] = {
                "windows": len(refs), "horizons": per_horizon}
        split_report["macro_run_metrics"] = _macro_summary(
            split_report["per_run"])
        report["splits"][split_name] = split_report
    for split_name, _ in split_specs:
        selected_runs = data.validation_windows if split_name == "openplane_validation" \
            else data.practice_windows
        run_ids = sorted(selected_runs)
        for horizon in ("2s", "5s"):
            if not all(horizon in report["splits"][split_name]["per_run"][run]["horizons"]
                       for run in run_ids):
                continue
            deltas = {}
            for run_id in run_ids:
                candidate_value = per_method_run["candidate"][horizon][run_id][
                    "position_radial_trajectory_rmse_m"]
                baseline_value = per_method_run["wp19_parent"][horizon][run_id][
                    "position_radial_trajectory_rmse_m"]
                deltas[run_id] = baseline_value - candidate_value
            report["paired_run_deltas"][f"{split_name}_{horizon}_position_rmse_m"] = {
                "per_run_parent_minus_candidate": deltas,
                "macro_parent_minus_candidate": float(np.mean(list(deltas.values()))),
                "run_bootstrap_95pct_ci": _bootstrap_run_mean(
                    deltas, SEED + len(deltas)),
                "candidate_wins": int(sum(value > 0.0 for value in deltas.values())),
                "independent_runs": len(deltas),
            }
    report["gate"] = _material_gate(report)
    return report


def _first_divergence_seconds(prediction: dict[str, np.ndarray], index: int,
                              threshold_m: float) -> float | None:
    radial = np.linalg.norm(prediction["pose"][index, :, :2]
                            - prediction["truth_pose"][index, :, :2], axis=1)
    crossed = np.flatnonzero(radial > threshold_m)
    return float((crossed[0] + 1) * DT_S) if len(crossed) else None


def _macro_summary(per_run: dict[str, Any]) -> dict[str, Any]:
    horizons = sorted({h for run in per_run.values() for h in run["horizons"]},
                      key=lambda label: EVAL_HORIZONS.index((label, dict(EVAL_HORIZONS)[label])))
    summary = {}
    for horizon in horizons:
        run_data = {run_id: entry["horizons"][horizon]
                    for run_id, entry in per_run.items()
                    if horizon in entry["horizons"]}
        metric_names = next(iter(run_data.values()))["candidate_macro_window_metrics"].keys()
        summary[horizon] = {"independent_runs": len(run_data), "metrics": {}}
        for metric in metric_names:
            candidate = {run: values["candidate_macro_window_metrics"][metric]
                         for run, values in run_data.items()}
            parent = {run: values["wp19_parent_macro_window_metrics"][metric]
                      for run, values in run_data.items()}
            summary[horizon]["metrics"][metric] = {
                "candidate_macro_run_mean": float(np.mean(list(candidate.values()))),
                "candidate_run_bootstrap_95pct_ci": _bootstrap_run_mean(
                    candidate, SEED + len(metric)),
                "wp19_parent_macro_run_mean": float(np.mean(list(parent.values()))),
                "wp19_parent_run_bootstrap_95pct_ci": _bootstrap_run_mean(
                    parent, SEED + len(metric) + 1),
                "candidate_per_run": candidate,
                "wp19_parent_per_run": parent,
            }
    return summary


def _material_gate(report: dict[str, Any]) -> dict[str, Any]:
    validation = report["splits"]["openplane_validation"]["macro_run_metrics"]
    improvement = report["paired_run_deltas"].get(
        "openplane_validation_2s_position_rmse_m")
    if not improvement:
        return {"status": "not_evaluated", "passed": False}
    baseline = validation["2s"]["metrics"][
        "position_radial_trajectory_rmse_m"]["wp19_parent_macro_run_mean"]
    candidate = validation["2s"]["metrics"][
        "position_radial_trajectory_rmse_m"]["candidate_macro_run_mean"]
    relative = (baseline - candidate) / max(baseline, 1e-9)
    ci = improvement["run_bootstrap_95pct_ci"]
    majority = improvement["candidate_wins"] >= 4
    lower_ci_positive = ci is not None and ci[0] > 0.0
    channel_checks = {}
    for horizon in ("2s", "5s"):
        metrics = validation.get(horizon, {}).get("metrics", {})
        for metric, label in (("u_rmse_mps", "u"),
                              ("yaw_rate_rmse_rps", "yaw_rate")):
            value = metrics.get(metric)
            if value:
                base = value["wp19_parent_macro_run_mean"]
                cand = value["candidate_macro_run_mean"]
                channel_checks[f"{horizon}_{label}_relative_change"] = (
                    cand - base) / max(base, 1e-9)
    practice = report["splits"]["practice_transfer"]["macro_run_metrics"]
    practice_checks = {}
    for horizon in ("2s", "5s"):
        for metric, label in (("u_rmse_mps", "u"),
                              ("yaw_rate_rmse_rps", "yaw_rate")):
            values = practice.get(horizon, {}).get("metrics", {}).get(metric)
            if values:
                base = values["wp19_parent_macro_run_mean"]
                cand = values["candidate_macro_run_mean"]
                practice_checks[f"{horizon}_{label}_relative_change"] = (
                    cand - base) / max(base, 1e-9)
    no_large_regression = all(value <= 0.10 for value in channel_checks.values()) \
        and all(value <= 0.10 for value in practice_checks.values())
    validation_runs = report["splits"]["openplane_validation"]["per_run"]
    candidate_divergence = {}
    parent_divergence = {}
    for run_id, run_result in validation_runs.items():
        horizon_result = run_result["horizons"].get("5s")
        if horizon_result is None:
            continue
        cand_time = horizon_result["candidate_first_divergence_0p5m_s"]
        parent_time = horizon_result["wp19_parent_first_divergence_0p5m_s"]
        if cand_time is not None and parent_time is not None:
            candidate_divergence[run_id] = cand_time
            parent_divergence[run_id] = parent_time
    divergence_wins = sum(candidate_divergence[run] > parent_divergence[run]
                          for run in candidate_divergence)
    divergence_later = bool(
        candidate_divergence and divergence_wins >= 4
        and np.median(list(candidate_divergence.values()))
        > np.median(list(parent_divergence.values())))
    passed = (relative >= 0.20 and majority and lower_ci_positive
              and no_large_regression and divergence_later)
    return {
        "status": "candidate_survives_material_gate" if passed else "failed_material_gate",
        "passed": bool(passed),
        "pre_registered_primary_metric": "2-second run-macro radial position trajectory RMSE",
        "relative_improvement": float(relative),
        "required_relative_improvement": 0.20,
        "primary_candidate_wins": improvement["candidate_wins"],
        "required_validation_run_wins": 4,
        "paired_improvement_run_bootstrap_95pct_ci_m": ci,
        "required_improvement_ci_lower_above_zero": True,
        "validation_u_yaw_relative_error_change": channel_checks,
        "practice_transfer_u_yaw_relative_error_change": practice_checks,
        "no_gt_10pct_u_yaw_regression": bool(no_large_regression),
        "first_divergence_later_on_majority": divergence_later,
        "first_divergence_candidate_median_s": (
            float(np.median(list(candidate_divergence.values())))
            if candidate_divergence else None),
        "first_divergence_parent_median_s": (
            float(np.median(list(parent_divergence.values())))
            if parent_divergence else None),
        "first_divergence_candidate_wins": divergence_wins,
        "note": ("Research-only level-1 gate; passing does not authorize final "
                 "confirmation, production integration, or planner optimization."),
    }


def _theoretical_run_probability(data: WP22Data) -> float:
    families = data.raw_sources[0]["training_families"]
    run_ids = data.captures[0].run_ids
    by_family: dict[str, list[str]] = defaultdict(list)
    for run_id, family in zip(run_ids, families):
        if str(run_id) in data.training_runs:
            by_family[str(family)].append(str(run_id))
    if not by_family:
        raise ValueError("training-run family metadata is empty")
    return max(1.0 / (len(by_family) * len(runs))
               for runs in by_family.values())


def run_wp22(output: Path, stage_steps: tuple[int, int, int] = STAGE_STEPS,
             device: str = "cpu", variant: str = "A2",
             work_package: str = "WP22",
             support_loss_weight: float = SUPPORT_LOSS_WEIGHT
             ) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training directory: {output}")
    if len(stage_steps) != 3 or any(step < 1 for step in stage_steps):
        raise ValueError("requires positive Stage A/B/C optimizer-step counts")
    if variant not in ("A0", "A1", "A2", "D1", "D2", "D3"):
        raise ValueError("unsupported WP22/WP23/WP26 plant variant")
    if variant in ("A0", "A1") and work_package != "WP23":
        raise ValueError(f"{variant} is a WP23 structural ablation")
    if variant in ("D1", "D2", "D3") and work_package != "WP26":
        raise ValueError(f"{variant} is a WP26 mechanism ablation")
    if variant == "A2" and work_package != "WP22":
        raise ValueError("A2 is the existing WP22 comparator, not a new variant")
    if not np.isfinite(support_loss_weight) or support_loss_weight < 0.0:
        raise ValueError("support loss weight must be finite and nonnegative")
    data, wp20_report = _load_data()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(1)
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config, scales, diagnostics = _training_windows_and_stats(data, wp20_report)
    if variant == "A0":
        config = dataclass_replace(config, latent_enabled=False, latent_size=0)
    elif variant == "A1":
        config = dataclass_replace(config, body_transition_mode="direct_state")
    elif variant in ("D2", "D3"):
        config = dataclass_replace(config, latent_measurement_feedback=False)
    model = AugmentedStateSpacePlant(config).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    family_counts, run_counts, condition_counts = Counter(), Counter(), Counter()
    stage_reports, training_draws = [], []
    for stage_index, (step_count, horizon) in enumerate(zip(stage_steps,
                                                           STAGE_HORIZONS)):
        stage_started = time.perf_counter()
        print(f"{work_package} {variant} Stage {chr(ord('A') + stage_index)}: "
              f"{step_count} updates, "
              f"{horizon * DT_S:.2f}s recursive horizon, "
              f"{len(data.train_windows_by_horizon[horizon])} eligible runs",
              flush=True)
        losses = []
        counters = {"family": family_counts, "run": run_counts,
                    "condition": condition_counts}
        for step_index in range(step_count):
            refs = _sample_windows(data, horizon, BATCH_SIZE, rng, counters)
            training_draws.append([[int(part) for part in ref] for ref in refs])
            losses.append(_train_step(model, data, refs, horizon, scales,
                                      optimizer, device_obj, support_loss_weight))
            if (step_index + 1) % 10 == 0 or step_index + 1 == step_count:
                print(f"  update {step_index + 1}/{step_count}; "
                      f"loss={losses[-1]['loss']:.5f}; "
                      f"elapsed={time.perf_counter() - stage_started:.1f}s",
                      flush=True)
        stage_reports.append({
            "stage": chr(ord("A") + stage_index),
            "rollout_steps": horizon,
            "rollout_seconds": horizon * DT_S,
            "optimizer_steps": step_count,
            "eligible_independent_training_runs": sorted(
                data.train_windows_by_horizon[horizon]),
            "training_runs_without_a_complete_stage_window": sorted(
                set(data.training_runs)
                - set(data.train_windows_by_horizon[horizon])),
            "eligible_family_count": len(set(
                data.raw_sources[0]["training_families"][
                    [str(value) in data.train_windows_by_horizon[horizon]
                     for value in data.captures[0].run_ids]])),
            "mean_losses": {key: float(np.mean([row[key] for row in losses]))
                            for key in losses[0]},
            "loss_p90": {key: float(np.quantile([row[key] for row in losses], 0.9))
                         for key in losses[0]},
        })
    wp19_checkpoint = torch.load(WP19_CHECKPOINT, map_location=device_obj,
                                 weights_only=True)
    baseline_model = make_wp19_model(torch.nn).to(device_obj)
    baseline_model.load_state_dict(wp19_checkpoint["state_dict"], strict=True)
    baseline_stats = training_statistics(data.captures, data.training_windows_80)
    print(f"{work_package} {variant} training complete; scoring matched "
          "whole-run validation and "
          "practice-transfer windows.", flush=True)
    evaluation = _evaluate(data, model, baseline_model, baseline_stats,
                           device_obj, progress_label=f"{work_package} {variant}")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = (output / "checkpoint.pt" if work_package == "WP26"
                       else output / f"seed101_{variant}_candidate.pt")
    save_checkpoint(checkpoint_path, model, {
        "seed": SEED, "work_package": work_package, "variant": variant,
        "candidate_status": evaluation["gate"]["status"]})
    report = {
        "schema_version": 1,
        "work_package": work_package,
        "architecture_variant": variant,
        "purpose": ("single-seed matched WP23 structural ablation and material gate"
                    if work_package == "WP23" else
                    "fixed-budget WP26 mechanism ablation against frozen A2"
                    if work_package == "WP26" else
                    "single-seed staged recursive-training smoke and pre-registered material gate"),
        "candidate_promoted": False,
        "future_truth_or_feedback_used_as_rollout_input": False,
        "test_or_final_test_opened": False,
        "seed": SEED,
        "randomized_seed_policy": "one seed only; additional seeds blocked until candidate passes material gate",
        "training_runs": sorted(data.training_runs),
        "validation_runs": sorted(data.validation_windows),
        "practice_transfer_runs": sorted(data.practice_windows),
        "split_integrity": {
            "train_validation_intersection": sorted(set(data.training_runs)
                                                      & set(data.validation_windows)),
            "train_practice_intersection": sorted(set(data.training_runs)
                                                   & set(data.practice_windows)),
            "family_draw_counts": dict(family_counts),
            "run_draw_counts": dict(run_counts),
            "condition_draw_counts": dict(condition_counts),
            "single_run_draw_fraction_max": max(run_counts.values())
                / max(1, sum(run_counts.values())),
            "family_probabilities": {
                name: count / max(1, sum(family_counts.values()))
                for name, count in family_counts.items()},
            "family_sampling_rule": "uniform among available registered training families",
            "maximum_theoretical_run_probability": float(
                _theoretical_run_probability(data)),
        },
        "inputs": {
            "wp19_report_sha256": sha256_file(WP19_REPORT_PATH),
            "wp19_checkpoint_sha256": sha256_file(WP19_CHECKPOINT),
            "wp20_report_sha256": sha256_file(WP20_REPORT_PATH),
            "actuator_parent_sha256": EXPECTED_ACTUATOR_PARENT_SHA256,
            "training_datasets": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in (DEFAULT_DYNAMIC, DEFAULT_DYNAMIC_FIXED,
                             DEFAULT_PRACTICE, DEFAULT_PRACTICE_FIXED)],
        },
        "configuration": {
            "history_steps": HISTORY_STEPS,
            "history_seconds": HISTORY_STEPS * DT_S,
            "history_features": list(config.history_feature_names),
            "observable_state": ["u_rear_mps", "v_rear_mps", "yaw_rate_rps",
                                 "steering_feedback_rad", "throttle_feedback_norm"],
            "latent_size": config.latent_size,
            "latent_enabled": config.latent_enabled,
            "body_transition_mode": config.body_transition_mode,
            "target_representation": "WP19 B-B body-state increment + W-D encoder angle increment",
            "nominal": "zero body-increment hold with exact SE(2) pose integration",
            "training_stages": stage_reports,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW(lr=3e-4, weight_decay=1e-5); gradient norm clipped at 1",
            "loss_weights": {
                "body": BODY_LOSS_WEIGHT,
                "acceleration_consistency": ACCELERATION_CONSISTENCY_WEIGHT,
                "heading": HEADING_LOSS_WEIGHT,
                "position": POSITION_LOSS_WEIGHT,
                "measurement": MEASUREMENT_LOSS_WEIGHT,
                "latent": LATENT_LOSS_WEIGHT,
                "support": SUPPORT_LOSS_WEIGHT,
                "support_effective": support_loss_weight,
                "symmetry": SYMMETRY_LOSS_WEIGHT,
            },
            "support_regularization": "WP20 calibrated run-balanced kNN; higher penalty on residuals at low confidence",
            "latent_measurement_feedback": config.latent_measurement_feedback,
            "label_bounds": diagnostics,
        },
        "training_sampler": {
            "algorithm": "same WP22 family/run/condition sampler; seed 101",
            "draw_count": len(training_draws),
            "draws_by_step_and_batch": training_draws,
            "draws_sha256": hashlib.sha256(json.dumps(
                training_draws, separators=(",", ":")).encode("utf-8")).hexdigest(),
        },
        "evaluation": evaluation,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    report_filename = ("training_report.json" if work_package == "WP26" else
                       f"wp23_{variant}_training_report.json"
                       if work_package == "WP23" else "wp22_training_report.json")
    report_path = output / report_filename
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage-steps", type=int, nargs=3, default=STAGE_STEPS,
                        metavar=("A", "B", "C"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = run_wp22(args.output, tuple(args.stage_steps), args.device)
    print(json.dumps({
        "output": str(args.output), "checkpoint": report["checkpoint"],
        "gate": report["evaluation"]["gate"],
        "openplane_2s_position_rmse": report["evaluation"]["splits"][
            "openplane_validation"]["macro_run_metrics"].get("2s", {}).get(
                "metrics", {}).get("position_radial_trajectory_rmse_m"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
