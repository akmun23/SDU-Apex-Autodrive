#!/usr/bin/env python3
"""Fit a causal, latent-free 40 Hz plant with truncated rollout training.

This is a research-only follow-up to the failed long-BPTT history transition.
It keeps the same frozen WP24 data and whole-run starts, but uses one-step
warm-start and short randomized simulation sections. It does not change any
production, simulator, odometry, localization, or MPC code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _normalization,
    _write_json,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    FROZEN_EVAL_STARTS,
    HISTORY_STEPS,
    SEED,
    SOURCE_COMMIT,
    TASK_ROOT,
    WP19_CHECKPOINT,
    _batch_arrays,
    _metrics,
    _predict_run,
    _rollout,
    _evaluate,
    HistoryTransition,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _load_data,
    _predict_wp19_baseline,
    _collect_horizon_windows,
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = TASK_ROOT / "next_phase_after_2129427/truncated_history_transition_v1"
L0_MAX_UPDATES = 1000
L1_MAX_UPDATES = 600
L2_MAX_UPDATES = 600
L3_MAX_UPDATES = 300
L4_MAX_UPDATES = 200
EVAL_EVERY = 100
L0_BATCH = 32
L1_BATCH = 16
L2_BATCH = 8
L3_BATCH = 4
L4_BATCH = 2
L3_GRADIENT_SEGMENT_STEPS = 80
L4_GRADIENT_SEGMENT_STEPS = 80
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MIN_LEARNING_RATE = 3e-5
EARLY_STOP_PATIENCE = 3
BOOTSTRAP_REPLICATES = 5000


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True,
                          text=True).stdout.strip()


def _horizon_choices(stage: str) -> tuple[int, ...]:
    choices = {"L0": (1,), "L1": (4, 10, 20),
               "L2": (20, 40, 80), "L3": (120, 160, 200),
               "L4": (240, 320, 400)}
    if stage not in choices:
        raise ValueError(f"unknown training stage {stage!r}")
    return choices[stage]


def _sample_short_windows(data, horizon: int, batch_size: int,
                         rng: np.random.Generator,
                         counters: dict[str, Counter]
                         ) -> list[tuple[int, int, int]]:
    """Reuse the frozen WP22 family/run/condition sampler at shorter horizons."""
    bucket = (20 if horizon <= 20 else 80 if horizon <= 80 else
              200 if horizon <= 200 else 400)
    groups = {run: refs for run, refs in data.train_windows_by_horizon[bucket].items()
              if run in data.training_runs and refs}
    source = data.raw_sources[0]
    run_families = source["training_families"]
    run_ids = data.captures[0].run_ids
    family_for_run: dict[str, str] = {}
    for index, run_id in enumerate(run_ids):
        if str(run_id) in groups:
            family_for_run[str(run_id)] = str(run_families[index])
    by_family: dict[str, list[str]] = defaultdict(list)
    conditions_by_run: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
    for run_id, refs in groups.items():
        family = family_for_run.get(run_id, "unclassified")
        by_family[family].append(run_id)
        local: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for ref in refs:
            _, sequence_index, _ = ref
            condition = int(source["sequence_condition_id"][sequence_index])
            local[condition].append(ref)
        conditions_by_run[run_id] = dict(local)
    family_names = sorted(by_family)
    if not family_names:
        raise ValueError(f"no training samples for {horizon}-step sections")
    family_weights = np.full(len(family_names), 1.0 / len(family_names))
    batch = []
    for _ in range(batch_size):
        family = family_names[int(rng.choice(len(family_names), p=family_weights))]
        run_id = str(rng.choice(by_family[family]))
        condition_ids = sorted(conditions_by_run[run_id])
        condition = int(rng.choice(condition_ids))
        options = conditions_by_run[run_id][condition]
        ref = options[int(rng.integers(len(options)))]
        batch.append(ref)
        counters["family"][family] += 1
        counters["run"][run_id] += 1
        counters["condition"][f"{run_id}:{condition}"] += 1
    return batch


def _validation_score(candidate: dict[str, float],
                      parent: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Equal-weight relative error score; lower is better, with no unit mixing."""
    metrics = ("position_radial_trajectory_rmse_m",
               "heading_trajectory_rmse_rad", "u_rmse_mps", "v_rmse_mps",
               "yaw_rate_rmse_rps")
    ratios = {name: float(candidate[name] / max(parent[name], 1e-8))
              for name in metrics}
    return float(np.mean(list(ratios.values()))), ratios


def _frozen_validation_runs(data, device: torch.device, norm_np: dict[str, np.ndarray],
                            config, parent_horizon: int = 80
                            ) -> tuple[dict[str, list], dict[str, dict]]:
    selected = _select_eval_windows(data.validation_windows, 64)
    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_runs = frozen["split_roles"]["development_validation"]
    if set(selected) != set(expected_runs):
        raise RuntimeError("development-validation run set differs from frozen WP24")

    model_checkpoint = torch.load(WP19_CHECKPOINT, map_location=device,
                                  weights_only=True)
    parent_model = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    parent_model.load_state_dict(model_checkpoint["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    parent_rollouts: dict[str, dict] = {}
    for run_id, refs in sorted(selected.items()):
        actual = {(int(sequence), int(row)) for _, sequence, row in refs}
        expected = {(int(item["sequence_index"]), int(item["source_row"]))
                    for item in expected_runs[run_id]}
        if actual != expected:
            raise RuntimeError(f"{run_id}: starts differ from frozen WP24 manifest")
        capture_index = int(refs[0][0])
        parent_rollouts[run_id] = _predict_wp19_baseline(
            data.captures[capture_index], data.poses[capture_index], refs,
            parent_horizon,
            wp19_stats, parent_model, device)
    return selected, parent_rollouts


def _score_checkpoint(data, model: HistoryTransition, selected: dict[str, list],
                      parent_rollouts: dict[str, dict], norm_np,
                      config, device: torch.device, horizon: int) -> dict[str, Any]:
    candidate_by_run: dict[str, dict[str, float]] = {}
    parent_by_run: dict[str, dict[str, float]] = {}
    for run_id, refs in sorted(selected.items()):
        capture_index = int(refs[0][0])
        prediction = _predict_run(data, refs, model, norm_np, config,
                                  horizon, device)
        truth_state, truth_pose = prediction[2], prediction[3]
        candidate_by_run[run_id] = _metrics(
            prediction[0], prediction[1], truth_state, truth_pose, horizon)
        baseline = parent_rollouts[run_id]
        parent_state = np.concatenate((baseline["body"][:, :horizon],
                                       truth_state[..., 3:5]),
                                      axis=-1)
        parent_by_run[run_id] = _metrics(
            parent_state, baseline["pose"][:, :horizon], truth_state,
            truth_pose, horizon,
            include_actuator=False)
    candidate_macro = {
        name: float(np.mean([metrics[name] for metrics in candidate_by_run.values()]))
        for name in candidate_by_run[next(iter(candidate_by_run))]
        if name in parent_by_run[next(iter(parent_by_run))]
    }
    parent_macro = {
        name: float(np.mean([metrics[name] for metrics in parent_by_run.values()]))
        for name in parent_by_run[next(iter(parent_by_run))]
    }
    score, ratios = _validation_score(candidate_macro, parent_macro)
    return {
        "horizon_steps": horizon,
        "independent_run_count": len(selected),
        "candidate_macro": candidate_macro,
        "wp19_parent_macro": parent_macro,
        "candidate_to_parent_ratio": ratios,
        "equal_weight_relative_score": score,
        "per_run_candidate": candidate_by_run,
        "per_run_parent": parent_by_run,
    }


def _fit(data, model: HistoryTransition, norm_np, config,
         device: torch.device, *, stages_to_run: tuple[str, ...] = ("L0", "L1", "L2"),
         sampler_seed: int = SEED, initial_clip: float = 10.0,
         score_initial_checkpoint: bool = False) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    norm = {
        "state_mean": torch.as_tensor(norm_np["state_mean"], device=device),
        "state_scale": torch.as_tensor(norm_np["state_scale"], device=device),
        "command_mean": torch.as_tensor(norm_np["command_mean"], device=device),
        "command_scale": torch.as_tensor(norm_np["command_scale"], device=device),
        "history_mean": torch.as_tensor(config.history_mean[:7], device=device),
        "history_scale": torch.as_tensor(config.history_scale[:7], device=device),
    }
    pose_integrator = PoseIntegrator(DT_S).to(device)
    validation_horizon = (400 if "L4" in stages_to_run else
                          200 if "L3" in stages_to_run else 80)
    selected, parent_rollouts = _frozen_validation_runs(
        data, device, norm_np, config, validation_horizon)
    rng = np.random.default_rng(sampler_seed)
    counters = {key: Counter() for key in ("family", "run", "condition")}
    draw_log: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    all_eval_records: list[dict[str, Any]] = []
    total_updates = 0
    calibrated_clip = initial_clip
    started = time.perf_counter()

    stage_specs = (
            ("L0", L0_MAX_UPDATES, L0_BATCH),
            ("L1", L1_MAX_UPDATES, L1_BATCH),
            ("L2", L2_MAX_UPDATES, L2_BATCH),
            ("L3", L3_MAX_UPDATES, L3_BATCH),
            ("L4", L4_MAX_UPDATES, L4_BATCH))
    for stage, max_updates, batch_size in stage_specs:
        if stage not in stages_to_run:
            continue
        choices = _horizon_choices(stage)
        stage_history: list[dict[str, Any]] = []
        raw_gradients: list[float] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_optimizer_state: dict[str, Any] | None = None
        best_score = float("inf")
        best_eval: dict[str, Any] | None = None
        stale_evaluations = 0
        stage_lr_reductions = 0
        model.train()

        if score_initial_checkpoint:
            if stage not in ("L2", "L3", "L4"):
                raise ValueError("initial checkpoint scoring is defined for L2/L3/L4 only")
            initial_horizon = (400 if stage == "L4" else
                               200 if stage == "L3" else 80)
            initial = _score_checkpoint(data, model, selected, parent_rollouts,
                                        norm_np, config, device, initial_horizon)
            initial.update({"stage": stage, "local_update": 0,
                            "total_updates": total_updates,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "checkpoint_role": "continuation_start"})
            best_eval = initial
            best_score = initial["equal_weight_relative_score"]
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            all_eval_records.append(initial)

        for local_update in range(1, max_updates + 1):
            horizon = choices[int(rng.integers(0, len(choices)))]
            refs = _sample_short_windows(data, horizon, batch_size, rng, counters)
            arrays = _batch_arrays(data, refs, horizon, norm_np, config, device)
            optimizer.zero_grad(set_to_none=True)
            segmented_backward = stage in ("L3", "L4")
            gradient_segment_steps = (L4_GRADIENT_SEGMENT_STEPS
                                      if stage == "L4" else
                                      L3_GRADIENT_SEGMENT_STEPS)
            result = _rollout(
                model, arrays, norm, horizon, pose_integrator,
                collect_loss=True,
                detach_every_steps=(gradient_segment_steps
                                    if segmented_backward else None),
                backward_segments=segmented_backward)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {stage} loss at update {local_update}")
            if not segmented_backward:
                loss.backward()
            raw_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0 if stage != "L2" else calibrated_clip))
            if not np.isfinite(raw_norm):
                raise FloatingPointError(f"non-finite {stage} gradient at {local_update}")
            optimizer.step()
            raw_gradients.append(raw_norm)
            total_updates += 1
            draw_log.append({
                "stage": stage,
                "local_update": local_update,
                "horizon_steps": horizon,
                "refs": [[int(value) for value in ref] for ref in refs],
            })
            if local_update % 20 == 0:
                stage_history.append({"update": local_update,
                                      "loss": float(loss.detach()),
                                      "raw_gradient_norm": raw_norm})

            if local_update % EVAL_EVERY != 0:
                continue
            eval_horizon = (1 if stage == "L0" else
                            400 if stage == "L4" else
                            200 if stage == "L3" else 80)
            scored = _score_checkpoint(data, model, selected, parent_rollouts,
                                       norm_np, config, device, eval_horizon)
            scored.update({"stage": stage, "local_update": local_update,
                           "total_updates": total_updates,
                           "learning_rate": optimizer.param_groups[0]["lr"]})
            all_eval_records.append(scored)
            current_score = scored["equal_weight_relative_score"]
            if current_score < best_score - 1e-6:
                best_score = current_score
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_eval = scored
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if stage == "L1" and local_update == 100:
                calibrated_clip = min(10.0, max(1.0,
                                                float(np.quantile(raw_gradients, 0.95))))
            if stale_evaluations >= 2 and stage_lr_reductions < 3:
                next_lr = max(MIN_LEARNING_RATE,
                              optimizer.param_groups[0]["lr"] * 0.5)
                if next_lr < optimizer.param_groups[0]["lr"]:
                    for group in optimizer.param_groups:
                        group["lr"] = next_lr
                    stage_lr_reductions += 1
                stale_evaluations = 0
            print(f"{stage} update={local_update} H={eval_horizon} "
                  f"score={current_score:.5f} best={best_score:.5f} "
                  f"pos_ratio={scored['candidate_to_parent_ratio']['position_radial_trajectory_rmse_m']:.3f} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
            if stale_evaluations >= EARLY_STOP_PATIENCE:
                break

        if (best_state is None or best_optimizer_state is None
                or best_eval is None):
            raise RuntimeError(f"{stage} produced no validation checkpoint")
        model.load_state_dict(best_state, strict=True)
        optimizer.load_state_dict(best_optimizer_state)
        stage_reports.append({
            "stage": stage,
            "horizon_choices_steps": list(choices),
            "max_updates": max_updates,
            "completed_updates": local_update,
            "batch_size": batch_size,
            "best_validation": best_eval,
            "raw_gradient_norm_p50": float(np.quantile(raw_gradients, 0.50)),
            "raw_gradient_norm_p95": float(np.quantile(raw_gradients, 0.95)),
            "gradient_clip_for_next_stage": calibrated_clip,
            "learning_rate_reductions": stage_lr_reductions,
            "training_trace_every_20_updates": stage_history,
        })

    draws_json = json.dumps(draw_log, separators=(",", ":"))
    return {
        "stages": stage_reports,
        "updates": total_updates,
        "sampler_counts": {key: dict(value) for key, value in counters.items()},
        "draw_log": draw_log,
        "draws_sha256": hashlib.sha256(draws_json.encode()).hexdigest(),
        "validation_checkpoint_history": all_eval_records,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(device_name: str = "cpu", output_root: Path = OUTPUT_ROOT,
        continuation_checkpoint: Path | None = None,
        long_rollout: bool = False,
        full_lap_rollout: bool = False) -> dict[str, Any]:
    if long_rollout and full_lap_rollout:
        raise ValueError("choose either 5-second or 10-second rollout continuation")
    if _git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"expected source {SOURCE_COMMIT}; found {_git_head()}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    sampler_seed = SEED if continuation_checkpoint is None else SEED + 1
    torch.manual_seed(sampler_seed)
    data, wp20 = _load_data()
    if full_lap_rollout:
        long_windows = _collect_horizon_windows(
            data.captures, data.raw_sources, data.poses, {"train"}, 400,
            capture_indices={0})
        data.train_windows_by_horizon[400] = {
            run_id: refs for run_id, refs in long_windows.items()
            if run_id in data.training_runs}
        if len(data.train_windows_by_horizon[400]) < 5:
            raise RuntimeError("10-second L4 needs at least five contiguous training runs")
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    for capture in data.captures:
        if not np.allclose(capture.dt_s, DT_S, rtol=0.0, atol=1e-7):
            raise RuntimeError("eligible capture is not fixed 40 Hz")
    model = HistoryTransition(norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    continuation_sha256 = None
    initial_clip = 10.0
    if continuation_checkpoint is not None:
        continuation_checkpoint = continuation_checkpoint.resolve()
        expected_hash = continuation_checkpoint.with_suffix(".sha256").read_text(
            encoding="utf-8").strip()
        continuation_sha256 = sha256_file(continuation_checkpoint)
        if continuation_sha256 != expected_hash:
            raise RuntimeError("continuation checkpoint hash does not match its sidecar")
        saved = torch.load(continuation_checkpoint, map_location=device,
                           weights_only=True)
        if saved.get("metadata", {}).get("source_commit") != SOURCE_COMMIT:
            raise RuntimeError("continuation checkpoint source commit differs")
        model.load_state_dict(saved["state_dict"], strict=True)
        parent_report = continuation_checkpoint.parent / "training_report.json"
        parent_training = json.loads(parent_report.read_text(encoding="utf-8"))
        initial_clip = float(parent_training["stages"][-1][
            "gradient_clip_for_next_stage"])
        stages_to_run = (("L4",) if full_lap_rollout else
                         ("L3",) if long_rollout else ("L2",))
    else:
        if long_rollout or full_lap_rollout:
            raise ValueError("L3/L4 continuation requires a verified checkpoint")
        stages_to_run = ("L0", "L1", "L2")
    training = _fit(data, model, norm_np, config, device,
                    stages_to_run=stages_to_run, sampler_seed=sampler_seed,
                    initial_clip=initial_clip,
                    score_initial_checkpoint=continuation_checkpoint is not None)
    final_evaluation = _evaluate(data, model, norm_np, config, device)

    output_root.mkdir(parents=True, exist_ok=False)
    draws = training.pop("draw_log")
    paths = {"openplane": DEFAULT_DYNAMIC,
             "openplane_fixed40hz_sidecar": DEFAULT_DYNAMIC_FIXED,
             "practice": DEFAULT_PRACTICE,
             "practice_fixed40hz_sidecar": DEFAULT_PRACTICE_FIXED}
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "training_runs": list(data.training_runs),
        "validation_runs": sorted(data.validation_windows),
        "practice_diagnostic_runs": sorted(data.practice_windows),
        "validation_starts": FROZEN_EVAL_STARTS.relative_to(ROOT).as_posix(),
        "training_sampler_draws_sha256": training["draws_sha256"],
        "files": {name: {"path": path.relative_to(ROOT).as_posix(),
                          "sha256": sha256_file(path)}
                  for name, path in paths.items()},
        "split_policy": "train updates only on training runs; exact WP24 whole-run validation starts select; practice diagnostic only; test/final-test excluded",
    }
    metadata = {
        "model": ("10-second explicit history, 25-ms nonlinear transition; "
                  "gradient detached every 2 seconds; no autonomous latent state"
                  if full_lap_rollout else
                  "5-second explicit history, 25-ms nonlinear transition; "
                  "gradient detached every 2 seconds; no autonomous latent state"
                  if long_rollout else
                  "2-second explicit causal history, 25-ms nonlinear transition; no autonomous latent state"),
        "source_commit": SOURCE_COMMIT,
        "seed": sampler_seed,
        "continued_from_checkpoint_sha256": continuation_sha256,
        "history_steps": HISTORY_STEPS,
        "history_seconds": HISTORY_STEPS * DT_S,
        "history_features": ["odom_u", "odom_v", "odom_yaw_rate",
                             "steering_feedback", "throttle_feedback",
                             "steering_command", "throttle_command"],
        "state": ["u_rear_mps", "v_rear_mps", "yaw_rate_rps",
                  "steering_feedback_rad", "throttle_feedback_norm"],
        "control_rate_hz": 40.0,
        "integrator_substep_hz": 40.0,
        "training": {"L0": {"max_updates": L0_MAX_UPDATES,
                             "horizons_steps": [1], "batch": L0_BATCH},
                     "L1": {"max_updates": L1_MAX_UPDATES,
                             "horizons_steps": [4, 10, 20], "batch": L1_BATCH},
                     "L2": {"max_updates": L2_MAX_UPDATES,
                             "horizons_steps": [20, 40, 80], "batch": L2_BATCH},
                     **({"L3": {"max_updates": L3_MAX_UPDATES,
                                 "horizons_steps": [120, 160, 200],
                                 "gradient_segment_steps": L3_GRADIENT_SEGMENT_STEPS,
                                 "batch": L3_BATCH}}
                        if long_rollout else {}),
                     **({"L4": {"max_updates": L4_MAX_UPDATES,
                                 "horizons_steps": [240, 320, 400],
                                 "gradient_segment_steps": L4_GRADIENT_SEGMENT_STEPS,
                                 "batch": L4_BATCH}}
                        if full_lap_rollout else {})},
        "loss": "same causal state/pose objective as the earlier history model; no support term and no sensor-output feedback",
        "future_truth_or_sensor_input": False,
        "normalization_train_only": {key: value.tolist()
                                     for key, value in norm_np.items()},
        "history_mean_first_7": list(config.history_mean[:7]),
        "history_scale_first_7": list(config.history_scale[:7]),
    }
    checkpoint = output_root / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    (output_root / "checkpoint.sha256").write_text(checkpoint_hash + "\n",
                                                    encoding="utf-8")
    _write_json(output_root / "checkpoint_metadata.json",
                {**metadata, "checkpoint_sha256": checkpoint_hash})
    _write_json(output_root / "dataset_manifest.json", manifest)
    _write_json(output_root / "training_report.json", training)
    _write_json(output_root / "training_sampler_draws.json", {
        "sha256": training["draws_sha256"], "draws": draws})
    _write_json(output_root / "evaluation_report.json", final_evaluation)
    report = {
        "study": ("10-second closed-loop training with 2-second truncated gradients"
                  if full_lap_rollout else
                  "5-second closed-loop training with 2-second truncated gradients"
                  if long_rollout else
                  "truncated one-step to 2-second training for explicit-history nonlinear plant"),
        "source_commit": SOURCE_COMMIT,
        "continued_from_checkpoint_sha256": continuation_sha256,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "control_rate_hz": 40.0,
        "internal_integration_rate_hz": 40.0,
        "gradient_segment_seconds": (
            L4_GRADIENT_SEGMENT_STEPS * DT_S if full_lap_rollout else
            L3_GRADIENT_SEGMENT_STEPS * DT_S if long_rollout else None),
        "training_updates": training["updates"],
        "selected_validation_checkpoint_per_stage": [
            {"stage": row["stage"], "updates": row["best_validation"]["total_updates"],
             "score": row["best_validation"]["equal_weight_relative_score"]}
            for row in training["stages"]],
        "evaluation": final_evaluation,
        "limitations": [
            "Only existing 40 Hz labels are used; this experiment does not infer 1 kHz physics.",
            "Validation selects checkpoints and is not an untouched confirmation set.",
            "Practice is diagnostic only; test/final-test runs are excluded.",
            "No candidate is accepted unless whole-run recursive accuracy materially improves.",
        ],
        "output_directory": output_root.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
    }
    _write_json(output_root / "study_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--continue-from", type=Path,
                        help="hash-verified checkpoint for an L2 or L3 continuation")
    parser.add_argument("--long-rollout", action="store_true",
                        help="train 3–5 s rollouts with 2 s gradient truncation")
    parser.add_argument("--full-lap-rollout", action="store_true",
                        help="train 6–10 s rollouts with 2 s gradient truncation")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    continuation = args.continue_from
    if continuation is not None and not continuation.is_absolute():
        continuation = ROOT / continuation
    result = run(args.device, output, continuation, args.long_rollout,
                 args.full_lap_rollout)
    print(json.dumps({"output": result["output_directory"],
                      "updates": result["training_updates"],
                      "checkpoint": result["checkpoint_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
