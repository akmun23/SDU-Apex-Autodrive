#!/usr/bin/env python3
"""Compare the frozen deepSI SUBNET, WP19 parent, and plain-GRU ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import run_wp19_target_ablation as wp19
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    _predict_wp19_baseline,
)
from tools.vehicle_dynamics_learning.train_nssm import _model_type, _rollout


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
BODY_DATASET = RESET_ROOT / "body_sysid_v1.npz"
BODY_MANIFEST = RESET_ROOT / "body_sysid_v1_manifest.json"
STARTS_PATH = RESET_ROOT / "evaluation_starts.json"
COMPARATOR_MANIFEST = RESET_ROOT / "comparator_manifest.json"
SUBNET_DIR = RESET_ROOT / "deepsi_subnet_reference_v1"
SUBNET_CHECKPOINT = SUBNET_DIR / "SS_encoder_sdu_apex_subnet_reference_20261004_best.pth"
TRAINING_TRACE = SUBNET_DIR / "validation_trace.jsonl"
REFERENCE_REPORT = SUBNET_DIR / "subnet_reference_report.json"
DEFAULT_OUTPUT = SUBNET_DIR / "subnet_comparator_evaluation_v1.json"
PLAIN_GRU_DIR = ROOT / "live_runs/derived_dynamics_learning_20260928/nonlinear_model_tournament_gru_20260928"
PLAIN_GRU_MEMBERS = tuple(PLAIN_GRU_DIR / f"member_{index:02d}.pt"
                          for index in range(3))
WP19_CHECKPOINT = wp19.TASK_ROOT / (
    "wp19_target_ablation_v2_common_encoder_mask/"
    "07_body_state_increment__encoder_angle_increment/model.pt")
HORIZONS = (1, 10, 30, 40, 80, 150, 200, 400)
DT_S = 0.025
HISTORY = 12
BODY_NAMES = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")
POSE_METRICS = (
    "position_radial_trajectory_rmse_m",
    "position_endpoint_error_m",
    "heading_trajectory_rmse_rad",
    "heading_endpoint_abs_error_rad",
)
BOOTSTRAP_REPLICATES = 5000
SEED = 20261004


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap(values: dict[str, float], seed: int) -> dict[str, Any]:
    runs = sorted(values)
    array = np.asarray([values[run] for run in runs], dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("run-level bootstrap received non-finite values")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(array),
                            size=(BOOTSTRAP_REPLICATES, len(array)))
    means = array[selected].mean(axis=1)
    return {
        "independent_run_count": len(runs),
        "run_macro_mean": float(array.mean()),
        "run_cluster_bootstrap_95pct_ci": [
            float(x) for x in np.quantile(means, (0.025, 0.975))],
        "per_run": dict(zip(runs, array.astype(float).tolist())),
    }


def _exact_pose_rollout(body: np.ndarray, initial_body: np.ndarray,
                        initial_pose: np.ndarray) -> np.ndarray:
    """Apply the repository's exact constant-twist SE(2) pose step."""
    pose = np.asarray(initial_pose, dtype=np.float64).copy()
    state_before = np.asarray(initial_body, dtype=np.float64)
    output = np.empty_like(body, dtype=np.float64)
    for index, state_after in enumerate(np.asarray(body, dtype=np.float64)):
        u, v, yaw_rate = state_before
        dtheta = yaw_rate * DT_S
        if abs(yaw_rate) > 1.0e-7:
            dx_body = (u * math.sin(dtheta)
                       + v * (math.cos(dtheta) - 1.0)) / yaw_rate
            dy_body = (u * (1.0 - math.cos(dtheta))
                       + v * math.sin(dtheta)) / yaw_rate
        else:
            dx_body, dy_body = u * DT_S, v * DT_S
        pose = np.asarray((
            pose[0] + math.cos(pose[2]) * dx_body
            - math.sin(pose[2]) * dy_body,
            pose[1] + math.sin(pose[2]) * dx_body
            + math.cos(pose[2]) * dy_body,
            math.atan2(math.sin(pose[2] + dtheta),
                       math.cos(pose[2] + dtheta)),
        ), dtype=np.float64)
        output[index] = pose
        state_before = state_after
    return output


def _locate_frozen_starts(data, starts_doc: dict[str, Any], horizon: int
                          ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = data.captures[0]
    conditions = np.asarray(data.raw_sources[0]["sequence_condition_id"],
                            dtype=np.int64)
    run_lookup = {str(run): index for index, run in enumerate(capture.run_ids)}
    references = []
    for run_id, items in sorted(
            starts_doc["split_roles"]["development_validation"].items()):
        if run_id not in run_lookup:
            raise ValueError(f"frozen validation run missing from WP19 source: {run_id}")
        run_index = run_lookup[run_id]
        if str(capture.splits[run_index]) != "validation":
            raise ValueError(f"frozen run has a non-validation role: {run_id}")
        for item in items:
            condition = int(item["condition_id"])
            source_row = int(item["source_row"])
            candidates = np.flatnonzero(
                (capture.sequence_run == run_index)
                & (conditions == condition))
            eligible = [int(index) for index in candidates
                        if int(capture.bounds[index, 1]
                               - capture.bounds[index, 0]) >= source_row + horizon + 1]
            if len(eligible) != 1:
                raise ValueError(
                    f"expected one reset-safe comparator sequence for "
                    f"{run_id}/{condition}/{source_row}, found {len(eligible)}")
            sequence = eligible[0]
            if source_row < wp19.HISTORY_STEPS - 1:
                raise ValueError("frozen row does not support WP19's 80-step context")
            if (str(capture.splits[run_index]) != "validation"
                    or item.get("run_id") != run_id):
                raise ValueError("frozen start metadata disagrees with its comparator run")
            references.append({
                "run_id": run_id,
                "sequence_index": sequence,
                "source_row": source_row,
                "condition_id": condition,
                "absolute_row_body_dataset": int(item["absolute_row"]),
                "split_role": str(item["split_role"]),
            })
    counts = {run: sum(row["run_id"] == run for row in references)
              for run in sorted(starts_doc["split_roles"]["development_validation"])}
    if any(value != 64 for value in counts.values()):
        raise ValueError(f"the frozen-start cohort changed from 64/run: {counts}")
    return references, counts


def _subnet_rollout(model, capture, references: list[dict[str, Any]],
                    horizon: int, torch) -> np.ndarray:
    histories_u, histories_y, future_u = [], [], []
    for row in references:
        begin = int(capture.bounds[row["sequence_index"], 0]) + row["source_row"]
        histories_u.append(capture.frames[begin - HISTORY + 1:begin + 1, 3:5])
        histories_y.append(capture.body[begin - HISTORY + 1:begin + 1])
        future_u.append(capture.frames[begin + 1:begin + horizon + 1, 3:5])
    histories_u = np.asarray(histories_u, dtype=np.float32)
    histories_y = np.asarray(histories_y, dtype=np.float32)
    future_u = np.asarray(future_u, dtype=np.float32)
    u_norm = ((histories_u - np.asarray(model.norm.u0))
              / np.asarray(model.norm.ustd))
    y_norm = ((histories_y - np.asarray(model.norm.y0))
              / np.asarray(model.norm.ystd))
    encoded_history = np.concatenate((u_norm.reshape(len(references), -1),
                                      y_norm.reshape(len(references), -1)), axis=1)
    future_u_norm = ((future_u - np.asarray(model.norm.u0))
                     / np.asarray(model.norm.ustd))
    with torch.no_grad():
        state = model.encoder(torch.as_tensor(encoded_history, dtype=torch.float32))
        predictions = []
        for index in range(horizon):
            y = model.hn(state)
            predictions.append(y * torch.as_tensor(
                model.norm.ystd, dtype=torch.float32)
                + torch.as_tensor(model.norm.y0, dtype=torch.float32))
            u = torch.as_tensor(future_u_norm[:, index], dtype=torch.float32)
            state = model.fn(torch.cat((state, u), dim=1))
    return torch.stack(predictions, dim=1).cpu().numpy().astype(np.float64)


def _verify_subnet_vectorization(model, capture, references, torch
                                 ) -> float:
    # Compare a batched explicit SUBNET rollout against deepSI's own
    # single-system simulation to verify history ordering and state updates.
    probe = references[:2]
    batched = _subnet_rollout(model, capture, probe, 40, torch)
    errors = []
    for index, row in enumerate(probe):
        begin = int(capture.bounds[row["sequence_index"], 0]) + row["source_row"]
        start = begin - HISTORY + 1
        end = begin + 41
        system_data = __import__("deepSI").System_data(
            u=capture.frames[start:end, 3:5].astype(np.float32),
            y=capture.body[start:end].astype(np.float32), dt=DT_S)
        with torch.no_grad():
            reference = model.apply_experiment(system_data).y[HISTORY:]
        errors.append(np.max(np.abs(batched[index] - reference)))
    return float(max(errors))


def _gru_member_rollout(payload: dict[str, Any], model, capture,
                        references: list[dict[str, Any]], horizon: int,
                        torch, device) -> np.ndarray:
    means = np.asarray(payload["feature_mean"], dtype=np.float32)
    scales = np.asarray(payload["feature_scale"], dtype=np.float32)
    history_steps = int(payload["metadata"]["history_steps"])
    histories, futures, dts = [], [], []
    for row in references:
        begin = int(capture.bounds[row["sequence_index"], 0]) + row["source_row"]
        history = capture.frames[begin - history_steps + 1:begin + 1]
        future = capture.frames[begin + 1:begin + horizon + 1]
        if len(history) != history_steps or len(future) != horizon:
            raise ValueError("plain GRU window crossed a sequence boundary")
        histories.append((history - means) / scales)
        futures.append((future - means) / scales)
        dts.append(capture.dt_s[begin + 1:begin + horizon + 1])
    history_t = torch.as_tensor(np.asarray(histories), dtype=torch.float32,
                                device=device)
    future_t = torch.as_tensor(np.asarray(futures), dtype=torch.float32,
                               device=device)
    dts_t = torch.as_tensor(np.asarray(dts), dtype=torch.float32, device=device)
    with torch.no_grad():
        normalized = _rollout(model, history_t, future_t, dts_t, history_steps)
    physical = (normalized.cpu().numpy().astype(np.float64)
                * scales[:7][None, None, :]
                + means[:7][None, None, :])
    return physical[..., :3]


def _metrics_by_run(predictions: np.ndarray, truth_body: np.ndarray,
                    predicted_pose: np.ndarray, truth_pose: np.ndarray,
                    references: list[dict[str, Any]], train_std: np.ndarray
                    ) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    runs = sorted({row["run_id"] for row in references})
    for run_id in runs:
        indices = [index for index, row in enumerate(references)
                   if row["run_id"] == run_id]
        body_error = predictions[indices] - truth_body[indices]
        pose_error = predicted_pose[indices] - truth_pose[indices]
        pose_error[..., 2] = np.arctan2(np.sin(pose_error[..., 2]),
                                        np.cos(pose_error[..., 2]))
        radial = np.linalg.norm(pose_error[..., :2], axis=-1)
        body_rmse = np.sqrt(np.mean(body_error ** 2, axis=(0, 1)))
        result[run_id] = {
            "window_count": len(indices),
            "body_rmse_by_channel": dict(zip(BODY_NAMES, body_rmse.tolist())),
            "body_nrms_by_channel": dict(zip(
                BODY_NAMES, (body_rmse / train_std).tolist())),
            "body_macro_channel_nrms": float(np.mean(body_rmse / train_std)),
            "position_radial_trajectory_rmse_m": float(
                np.sqrt(np.mean(radial ** 2))),
            "position_endpoint_error_m": float(np.sqrt(
                np.mean(np.square(radial[:, -1])))),
            "heading_trajectory_rmse_rad": float(
                np.sqrt(np.mean(np.square(pose_error[..., 2])))),
            "heading_endpoint_abs_error_rad": float(
                np.sqrt(np.mean(np.square(pose_error[:, -1, 2])))),
        }
    return result


def _run_summary(per_run: dict[str, dict[str, Any]], seed: int
                 ) -> dict[str, Any]:
    flattened = {run: _flatten_metrics(row) for run, row in per_run.items()}
    metrics = list(next(iter(flattened.values())))
    output = {}
    for offset, metric in enumerate(metrics):
        output[metric] = _bootstrap(
            {run: float(row[metric]) for run, row in flattened.items()},
            seed + offset)
    return output


def _flatten_metrics(row: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten scalar and per-channel run metrics for run-cluster bootstrap."""
    flattened: dict[str, float] = {}
    for name, value in row.items():
        if name == "window_count":
            continue
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, key))
        else:
            flattened[key] = float(value)
    return flattened


def evaluate(output_path: Path = DEFAULT_OUTPUT, device_name: str = "cuda"
             ) -> dict[str, Any]:
    import deepSI
    import torch
    from torch import nn

    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    for path in (BODY_DATASET, BODY_MANIFEST, STARTS_PATH,
                 COMPARATOR_MANIFEST, SUBNET_CHECKPOINT, TRAINING_TRACE,
                 REFERENCE_REPORT,
                 WP19_CHECKPOINT, *PLAIN_GRU_MEMBERS):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for the GRU comparator but unavailable")

    starts_doc = json.loads(STARTS_PATH.read_text(encoding="utf-8"))
    start_runs = set(starts_doc["split_roles"]["development_validation"])
    freeze = json.loads(COMPARATOR_MANIFEST.read_text(encoding="utf-8"))
    comparators = {item["role"]: item for item in freeze["comparators"]}
    wp19_sha = comparators["wp19_parent"]["sha256"]
    if sha256(WP19_CHECKPOINT) != wp19_sha:
        raise ValueError("WP19 parent checkpoint differs from its frozen comparator hash")
    for member in PLAIN_GRU_MEMBERS:
        expected = next(item["sha256"] for item in freeze["comparators"]
                        if item["path"].endswith(member.name))
        if sha256(member) != expected:
            raise ValueError(f"plain-GRU comparator hash changed: {member.name}")
    trace = [json.loads(line) for line in TRAINING_TRACE.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    reference_report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    if sha256(SUBNET_CHECKPOINT) != reference_report["checkpoints"]["best_sha256"]:
        raise ValueError("SUBNET checkpoint hash disagrees with its fit report")
    if sha256(TRAINING_TRACE) != reference_report["training"]["validation_trace_sha256"]:
        raise ValueError("SUBNET validation trace hash disagrees with its fit report")
    if not trace:
        raise ValueError("SUBNET reference validation trace is empty")
    completed_epochs = int(round(float(trace[-1]["epoch"])))
    if completed_epochs < 1000 and reference_report.get("status") != "early_stopped_by_user":
        raise ValueError("an incomplete SUBNET fit lacks a user-authorized early-stop report")
    if int(reference_report["training"]["epochs_completed"]) != completed_epochs:
        raise ValueError("SUBNET final report and validation trace disagree on completed epochs")
    trace_scores = np.asarray([
        np.nan if item["validation_score"] is None else float(item["validation_score"])
        for item in trace], dtype=np.float64)
    valid_trace = (np.isfinite(trace_scores)
                   & (trace_scores < np.finfo(np.float64).max / 2.0))
    if not valid_trace.any():
        raise ValueError("SUBNET trace has no numerically usable validation score")
    best_trace = trace[int(np.argmin(np.where(valid_trace, trace_scores, np.inf)))]

    body_manifest = json.loads(BODY_MANIFEST.read_text(encoding="utf-8"))
    if sha256(BODY_DATASET) != body_manifest["dataset_sha256"]:
        raise ValueError("body dataset changed after the freeze")
    with np.load(BODY_DATASET, allow_pickle=False) as body_data:
        body_split = np.asarray(body_data["split"]).astype(str)
        if set(np.unique(body_split)) - {"train", "validation"}:
            raise ValueError("test/final-test rows found in comparator evaluation data")
        train_std = np.std(
            np.asarray(body_data["outputs"])[body_split == "train"], axis=0, ddof=0)
        if not np.isfinite(train_std).all() or np.any(train_std <= 1e-12):
            raise ValueError("training-only output scales are invalid")

    wp19_report_path = wp19.DEFAULT_OUTPUT / "wp19_target_ablation_report.json"
    wp19_report = json.loads(wp19_report_path.read_text(encoding="utf-8"))
    if wp19_report.get("experimental_status") == "superseded_not_for_selection":
        raise ValueError("refusing the superseded WP19 comparator")
    for path, report_key in (
            (wp19.DEFAULT_DYNAMIC, "dynamic_sha256"),
            (wp19.DEFAULT_DYNAMIC_FIXED, "dynamic_fixed_sha256")):
        if sha256(path) != wp19_report["data"][report_key]:
            raise ValueError(f"WP19 comparator source changed: {path}")
    capture = wp19.load_capture(
        "openplane", wp19.DEFAULT_DYNAMIC, wp19.DEFAULT_DYNAMIC_FIXED,
        wp19.DEFAULT_DYNAMIC_PARENT, {"train", "validation"})
    training_windows = wp19.collect_windows([capture], {0}, {"train"})
    if set(training_windows) != set(wp19_report["data"]["training_runs"]):
        raise ValueError("WP19 frozen checkpoint and comparator training runs differ")
    with np.load(capture.source_path, allow_pickle=False) as wp19_source:
        feature_names = wp19_source["feature_names"].astype(str).tolist()
        sequence_condition_ids = np.asarray(
            wp19_source["sequence_condition_id"], dtype=np.int64)
        source_pose = np.asarray(
            wp19_source["simulator_pose_xyyaw"], dtype=np.float64)
    if source_pose.shape != (len(capture.frames), 3):
        raise ValueError("WP19 source pose labels do not align with source frames")
    wp19_data = SimpleNamespace(
        captures=[capture],
        raw_sources=[{"sequence_condition_id": sequence_condition_ids}],
        poses=[source_pose],
        training_windows_80=training_windows)
    references, starts_per_run = _locate_frozen_starts(
        wp19_data, starts_doc, max(HORIZONS))
    if len(feature_names) != capture.frames.shape[1]:
        raise ValueError("WP19 feature schema and source-frame width differ")

    # Verify the six frozen run/condition/sample slices are bit-identical
    # between the SUBNET v2 source and WP19 source used for comparator rollout.
    v2_path = Path(body_manifest["source_dataset"])
    with np.load(v2_path, allow_pickle=False) as src:
        v2_runs = src["run_ids"].astype(str)
        v2_lookup = {run: index for index, run in enumerate(v2_runs)}
        v2_bounds = np.asarray(src["sequence_bounds"], dtype=np.int64)
        v2_seq_run = np.asarray(src["sequence_run_index"], dtype=np.int32)
        v2_conditions = np.asarray(src["sequence_condition_id"], dtype=np.int64)
        v2_frames = np.asarray(src["frames"], dtype=np.float32)
        v2_pose = np.asarray(src["simulator_pose_xyyaw"], dtype=np.float32)
        source_frame_equal_count = 0
        for row in references:
            run_id = row["run_id"]
            condition = row["condition_id"]
            source_row = row["source_row"]
            v2_run = v2_lookup[run_id]
            v2_candidates = np.flatnonzero(
                (v2_seq_run == v2_run) & (v2_conditions == condition))
            if len(v2_candidates) != 1:
                raise ValueError("v2 frozen-start condition is not unique")
            v2_sequence = int(v2_candidates[0])
            v2_start, v2_end = map(int, v2_bounds[v2_sequence])
            old_start, old_end = map(int, capture.bounds[row["sequence_index"]])
            if (v2_end - v2_start < source_row + max(HORIZONS) + 1
                    or old_end - old_start < source_row + max(HORIZONS) + 1):
                raise ValueError("mapped comparator start lacks its maximum horizon")
            left = v2_start + source_row
            right = old_start + source_row
            if (not np.array_equal(v2_frames[left - 11:left + 401],
                                   capture.frames[right - 11:right + 401])
                    or not np.array_equal(v2_pose[left:left + 401],
                                          wp19_data.poses[0][right:right + 401])):
                raise ValueError("mapped frozen start differs between source captures")
            source_frame_equal_count += 1

    # Load the saved reference checkpoint in the isolated, pinned deepSI env.
    subnet = deepSI.fit_systems.SS_encoder(nx=9, na=HISTORY, nb=HISTORY)
    subnet.__dict__ = torch.load(SUBNET_CHECKPOINT, map_location="cpu",
                                 weights_only=False)
    subnet.eval()
    subnet_model_sha = sha256(SUBNET_CHECKPOINT)

    # Model-specific training/validation overlap is explicit, not hidden.
    wp19_payload = torch.load(WP19_CHECKPOINT, map_location="cpu",
                              weights_only=False)
    wp19_meta = wp19_payload["metadata"]
    wp19_training_overlap = sorted(start_runs & set(wp19_meta["training_runs"]))
    wp19_validation_overlap = sorted(start_runs & set(wp19_meta["validation_runs"]))
    if set(wp19_meta["training_runs"]) != set(training_windows):
        raise ValueError("WP19 checkpoint training runs differ from recomputed run windows")
    if wp19_training_overlap:
        raise ValueError(f"WP19 training data overlap frozen evaluation: {wp19_training_overlap}")
    stats = wp19.training_statistics(
        wp19_data.captures, wp19_data.training_windows_80)
    wp19_model = wp19.make_model(nn, input_dim=7,
                                 hidden_size=int(wp19_meta["hidden_size"]))
    wp19_model.load_state_dict(wp19_payload["state_dict"])
    wp19_model.to(device).eval()

    gru_payloads = [torch.load(path, map_location="cpu", weights_only=False)
                    for path in PLAIN_GRU_MEMBERS]
    plain_training_runs = set().union(*(
        set(item["metadata"].get("training_runs", [])) for item in gru_payloads))
    plain_validation_runs = set().union(*(
        set(item["metadata"].get("validation_runs", [])) for item in gru_payloads))
    plain_training_overlap = sorted(start_runs & plain_training_runs)
    plain_validation_overlap = sorted(start_runs & plain_validation_runs)
    if plain_training_overlap:
        raise ValueError(f"plain-GRU training overlaps frozen evaluation: {plain_training_overlap}")

    gru_models = []
    for payload in gru_payloads:
        metadata = payload["metadata"]
        names = [str(value) for value in metadata["feature_names"]]
        if names != feature_names:
            raise ValueError("plain-GRU feature contract differs from evaluation source")
        model_type = _model_type(
            torch, nn,
            hidden_size=int(metadata["hidden_size"]),
            architecture=str(metadata["architecture"]),
            expert_count=int(metadata.get("expert_count", 1)),
            history_steps=int(metadata["history_steps"]),
            feature_count=len(names),
            feature_mean=payload["feature_mean"],
            feature_scale=payload["feature_scale"],
            body_acceleration_mean=metadata.get("body_acceleration_mean"),
            body_acceleration_scale=metadata.get("body_acceleration_scale"),
            integration_method=metadata.get("integration_method", "euler"),
            rear_axle_to_com_x_m=float(metadata.get("rear_axle_to_com_x_m", 0.0)),
        )
        model = model_type().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        gru_models.append(model)

    # A library-versus-vectorized check catches flattening or timestamp shifts.
    subnet_batch_probe = _subnet_rollout(subnet, capture, references[:2], 40, torch)
    deep_si_probe_max_abs = _verify_subnet_vectorization(
        subnet, capture, references, torch)
    if deep_si_probe_max_abs > 1e-5 or not np.isfinite(subnet_batch_probe).all():
        raise ValueError("manual batched SUBNET rollout differs from deepSI simulation")

    horizon = max(HORIZONS)
    subnet_body = _subnet_rollout(subnet, capture, references, horizon, torch)
    gru_member_bodies = [
        _gru_member_rollout(payload, model, capture, references, horizon,
                           torch, device)
        for payload, model in zip(gru_payloads, gru_models)]
    plain_gru_body = np.mean(np.stack(gru_member_bodies, axis=0), axis=0)
    wp19_refs = [(0, row["sequence_index"], row["source_row"])
                 for row in references]
    wp19_prediction = _predict_wp19_baseline(
        capture, wp19_data.poses[0], wp19_refs, horizon, stats,
        wp19_model, device)
    wp19_body = np.asarray(wp19_prediction["body"], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in
               (subnet_body, plain_gru_body, wp19_body)):
        raise FloatingPointError("a comparator produced non-finite body states")

    truth_body, truth_pose, initial_bodies, initial_poses = [], [], [], []
    for row in references:
        begin = int(capture.bounds[row["sequence_index"], 0]) + row["source_row"]
        truth_body.append(capture.body[begin + 1:begin + horizon + 1])
        truth_pose.append(wp19_data.poses[0][begin + 1:begin + horizon + 1])
        initial_bodies.append(capture.body[begin])
        initial_poses.append(wp19_data.poses[0][begin])
    truth_body = np.asarray(truth_body, dtype=np.float64)
    truth_pose = np.asarray(truth_pose, dtype=np.float64)
    initial_bodies = np.asarray(initial_bodies, dtype=np.float64)
    initial_poses = np.asarray(initial_poses, dtype=np.float64)

    bodies = {
        "deepSI_SUBNET": subnet_body,
        "WP19_parent": wp19_body,
        "plain_GRU_ensemble_mean": plain_gru_body,
    }
    poses = {}
    for name, body in bodies.items():
        poses[name] = np.stack([
            _exact_pose_rollout(body[index], initial_bodies[index],
                                initial_poses[index])
            for index in range(len(references))], axis=0)

    report_horizons: dict[str, Any] = {}
    for horizon_steps in HORIZONS:
        per_model = {}
        for model_index, (name, body) in enumerate(bodies.items()):
            pred_pose = poses[name]
            by_run = _metrics_by_run(
                body[:, :horizon_steps], truth_body[:, :horizon_steps],
                pred_pose[:, :horizon_steps], truth_pose[:, :horizon_steps],
                references, train_std)
            per_model[name] = {
                "run_macro": _run_summary(by_run, SEED + horizon_steps + model_index * 100),
                "per_run": by_run,
            }
        paired = {}
        subnet_runs = per_model["deepSI_SUBNET"]["per_run"]
        for comparator in ("WP19_parent", "plain_GRU_ensemble_mean"):
            comparator_runs = per_model[comparator]["per_run"]
            flat_subnet = {run: _flatten_metrics(row)
                           for run, row in subnet_runs.items()}
            flat_comparator = {run: _flatten_metrics(row)
                               for run, row in comparator_runs.items()}
            metrics = list(flat_subnet[next(iter(flat_subnet))])
            paired[comparator + "_minus_SUBNET"] = {
                metric: _bootstrap({
                    run: float(flat_comparator[run][metric]
                               - flat_subnet[run][metric])
                    for run in sorted(start_runs)},
                    SEED + horizon_steps + metrics.index(metric) + 1000)
                for metric in metrics
            }
        report_horizons[str(horizon_steps)] = {
            "duration_s": float(horizon_steps * DT_S),
            "per_model": per_model,
            "paired_run_differences": paired,
        }

    report = {
        "schema_version": 1,
        "study": "same-start whole-run comparison of deepSI SUBNET with frozen parents",
        "training_complete_epochs": int(round(float(trace[-1]["epoch"]))),
        "reference_report_sha256": sha256(REFERENCE_REPORT),
        "subnet_best_validation_epoch": float(best_trace["epoch"]),
        "subnet_best_validation_score": float(best_trace["validation_score"]),
        "subnet_checkpoint_sha256": subnet_model_sha,
        "subnet_source_dataset_sha256": sha256(BODY_DATASET),
        "wp19_checkpoint_sha256": sha256(WP19_CHECKPOINT),
        "comparator_manifest_sha256": sha256(COMPARATOR_MANIFEST),
        "wp19_training_overlap": wp19_training_overlap,
        "wp19_validation_overlap_checkpoint_selection": wp19_validation_overlap,
        "plain_gru_checkpoint_sha256": {
            path.name: sha256(path) for path in PLAIN_GRU_MEMBERS},
        "plain_gru_training_overlap": plain_training_overlap,
        "plain_gru_validation_overlap_checkpoint_selection": plain_validation_overlap,
        "evaluation_start_count_by_run": starts_per_run,
        "independent_run_count": len(starts_per_run),
        "mapped_source_rows_bit_identical": source_frame_equal_count,
        "deepSI_vs_vectorized_rollout_max_abs_error": deep_si_probe_max_abs,
        "training_output_std_for_NRMS": dict(zip(
            BODY_NAMES, train_std.astype(float).tolist())),
        "inputs_available_after_anchor": "actuator feedback for SUBNET; future commands only for recursive GRUs",
        "future_recorded_body_or_sensor_values_used": False,
        "test_or_final_test_used_for_training_or_scoring": False,
        "raw_source_contains_excluded_splits_not_used": True,
        "practice_source_opened": False,
        "pose_integrator": "exact constant-twist SE(2), left-held rear-axle body twist, dt=0.025 s",
        "horizons": report_horizons,
        "interpretation": (
            "Development comparison only: the SUBNET is selected using these whole-run "
            "validation runs. WP19 was also selected using these validation runs; the "
            "plain-GRU checkpoint has no training-run overlap. None of these starts "
            "is an independent final confirmation."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate(args.output, args.device)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "runs": report["independent_run_count"],
        "starts_per_run": report["evaluation_start_count_by_run"],
        "subnet_vs_deepsi_manual_max_abs": report[
            "deepSI_vs_vectorized_rollout_max_abs_error"],
        "horizons": {
            label: {
                name: values["run_macro"]["body_macro_channel_nrms"]
                for name, values in step["per_model"].items()
            }
            for label, step in report["horizons"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
