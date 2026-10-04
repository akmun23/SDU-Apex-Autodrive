#!/usr/bin/env python3
"""Train the no-latent rigid-acceleration model with held-out-safe high-steer data.

This is a single targeted transfer experiment, not a hyperparameter sweep. It
uses the prescribed 1-step / short-truncated / medium-truncated schedule and
keeps the two whole-run high-steering validation captures out of training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    EXPECTED_RUNS as HIGHSTEER_RUNS,
    HIGHSTEER_SOURCE,
    _capture_from_archive,
    _macro,
    _paired,
    _parent_metrics,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    OUTPUT_ROOT as WP28_ROOT,
    SEED,
    STAGES,
    FixedCapacityHistoryTransition,
    _draw_plan,
    _normalization,
    _training_pools,
    _train_one,
    _bootstrap_delta,
)
from tools.vehicle_dynamics_learning.run_rigid_acceleration_history_candidate import (
    COM_X_M,
    EVAL_HORIZONS,
    METRICS,
    _acceleration_normalization,
    _curve,
    _model_load,
)
from tools.vehicle_dynamics_learning.run_long_horizon_context_candidate import (
    LONG_STAGES,
    _eval_refs,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _collect_horizon_windows,
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)


OUTPUT_ROOT = (WP28_ROOT / "rigid_acceleration_highsteer_transfer_wp27_v1")
WP26_A2_EVALUATION_REPORT = (
    WP28_ROOT.parent / "wp26_mechanism_ablations/D1/evaluation_report.json")
HIGHSTEER_TRAIN_RUNS = {
    "openplane_highsteer_75_train_r02",
    "openplane_highsteer_75_train_r03_20261004",
}
HIGHSTEER_TRAIN_DATASET = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/highsteer_75_train_expanded_20261004"
    / "openplane_dynamics.npz")
BASE_ACCELERATION_CHECKPOINT = (
    WP28_ROOT / "rigid_acceleration_history_direct_supervision_5s_v1"
    / "checkpoint.pt")
BASE_WP28_CHECKPOINT = (
    WP28_ROOT / "long_rollout_5s_selected_context_v1" / "checkpoint.pt")
WP19_CHECKPOINT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
    / "07_body_state_increment__encoder_angle_increment/model.pt")


def _append_training_capture(data, dataset_path: Path) -> None:
    with np.load(dataset_path, allow_pickle=False) as archive:
        run_ids = np.asarray(archive["run_ids"]).astype(str)
        splits = np.asarray(archive["run_splits"]).astype(str)
        conditions = np.asarray(archive["sequence_condition_id"], dtype=np.int64)
        families = np.asarray(archive["run_families"]).astype(str)
        if (set(run_ids) != HIGHSTEER_TRAIN_RUNS
                or np.any(splits != "train")
                or len(conditions) != len(archive["sequence_bounds"])):
            raise ValueError("extra training archive has wrong run/split/sequence roster")
        source = {
            "run_families": families,
            "training_families": families.copy(),
            "sequence_condition_id": conditions,
            "training_family_names": np.empty(0, dtype="U1"),
            "training_family_probabilities": np.empty(0, dtype=np.float64),
        }
    capture, pose = _capture_from_archive(
        dataset_path, HIGHSTEER_TRAIN_RUNS, expected_split="train")
    capture_index = len(data.captures)
    data.captures.append(capture)
    data.poses.append(pose)
    data.raw_sources.append(source)
    for horizon in (20, 80, 200):
        windows = _collect_horizon_windows(
            data.captures, data.raw_sources, data.poses,
            {"train"}, horizon, capture_indices={capture_index})
        if set(windows) != HIGHSTEER_TRAIN_RUNS:
            raise ValueError(f"high-steer training archive lacks {horizon}-step windows")
        overlap = set(windows) & set(data.train_windows_by_horizon[horizon])
        if overlap:
            raise ValueError(f"high-steer training runs already exist: {sorted(overlap)}")
        data.train_windows_by_horizon[horizon].update(windows)
        if horizon == 80:
            data.training_windows_80.update(windows)
    data.training_runs = tuple(sorted(set(data.training_runs) | HIGHSTEER_TRAIN_RUNS))


def _load_acceleration_model(checkpoint_path: Path, norm_np,
                             device: torch.device
                             ) -> tuple[RigidAccelerationHistoryTransition, str]:
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    metadata = saved["metadata"]
    model = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"]),
    ).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, sha256_file(checkpoint_path)


def run(device_name: str = "cpu", output_root: Path = OUTPUT_ROOT,
        training_dataset: Path | None = HIGHSTEER_TRAIN_DATASET,
        long_horizon_training: bool = False
        ) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if (training_dataset is not None
            and sha256_file(training_dataset) == sha256_file(HIGHSTEER_SOURCE)):
        raise ValueError("training archive unexpectedly aliases held-out high-steer data")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    data, wp20 = _load_data()
    baseline_data = data
    baseline_config, _, _ = _training_windows_and_stats(baseline_data, wp20)
    baseline_norm = _normalization(baseline_data, baseline_config)
    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    validation_refs = _eval_refs(
        baseline_data, "validation", 0, expected_validation)
    practice_refs = _eval_refs(
        baseline_data, "unseen_practice", 1, expected_practice)

    wp19_saved = torch.load(WP19_CHECKPOINT, map_location=device, weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_saved["state_dict"], strict=True)
    wp19_stats = training_statistics(
        baseline_data.captures, baseline_data.training_windows_80)
    parent_by_horizon = {
        horizon: _parent_metrics(
            baseline_data, baseline_data.poses[0], validation_refs,
            wp19, wp19_stats, horizon, device)
        for horizon in (1, 20, 80, 200)}

    parent_acceleration, parent_accel_sha = _load_acceleration_model(
        BASE_ACCELERATION_CHECKPOINT, baseline_norm, device)
    parent_wp28 = FixedCapacityHistoryTransition(
        baseline_norm["delta_mean"], baseline_norm["delta_scale"]).to(device)
    wp28_sha = _model_load(BASE_WP28_CHECKPOINT, parent_wp28, device)

    train_runs = set(baseline_data.training_runs)
    extra_training_runs: set[str] = set()
    if training_dataset is not None:
        _append_training_capture(data, training_dataset)
        extra_training_runs = set(HIGHSTEER_TRAIN_RUNS)
    if train_runs & HIGHSTEER_RUNS or extra_training_runs & HIGHSTEER_RUNS:
        raise RuntimeError("training data overlaps independent validation runs")
    if long_horizon_training:
        config, _, _ = _training_windows_and_stats(data, wp20)
        norm_np = _normalization(data, config)
        acceleration_mean, acceleration_scale, acceleration_rows = (
            _acceleration_normalization(data))
    else:
        # Keep the exact WP28 train-only feature/target scales for both sides
        # of the WP27 data ablation. The added runs change only training support.
        config, norm_np = baseline_config, baseline_norm
        acceleration_mean, acceleration_scale, acceleration_rows = (
            _acceleration_normalization(baseline_data))
    pools, _ = _training_pools(data, horizon_steps=80)
    stages = LONG_STAGES if long_horizon_training else STAGES
    pools_by_stage = None
    if long_horizon_training:
        long_pools, _ = _training_pools(data, horizon_steps=200)
        pools_by_stage = {"L2": long_pools}
    plan, sampler_counts = _draw_plan(
        pools, SEED + 1, stages=stages, pools_by_stage=pools_by_stage)
    a2_yaw_reference = None
    if not long_horizon_training:
        a2_report = json.loads(WP26_A2_EVALUATION_REPORT.read_text(
            encoding="utf-8"))
        a2_yaw_reference = float(a2_report[
            "mechanism_gate_vs_frozen_A2"][
                "openplane_2s_yaw_A2_macro_run_mean_rps"])

    high_capture, high_pose = _capture_from_archive(HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    output_root.mkdir(parents=True)
    candidate, training = _train_one(
        data, "2.0s_highsteer_augmented_wp27_truncated",
        CONTEXT_STEPS["2.0s"], plan, validation_refs, norm_np, config,
        parent_by_horizon, device, stages=stages,
        initial_state_dict=(parent_acceleration.state_dict()
                            if not long_horizon_training else None),
        model_factory=lambda: RigidAccelerationHistoryTransition(
            norm_np["state_mean"], norm_np["state_scale"],
            acceleration_mean, acceleration_scale,
            norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
            dt_s=0.025, rear_axle_to_com_x_m=COM_X_M),
        wp27_policy=not long_horizon_training,
        wp27_a2_yaw_reference=a2_yaw_reference)

    checkpoint_path = output_root / "checkpoint.pt"
    torch.save({
        "state_dict": {name: value.detach().cpu()
                       for name, value in candidate.state_dict().items()},
        "metadata": {
            "model": "history-conditioned explicit planar acceleration transition; no autonomous latent recursion",
            "context_steps": CONTEXT_STEPS["2.0s"],
            "control_and_data_rate_hz": 40.0,
            "dt_s": 0.025,
            "rear_axle_to_com_x_m": COM_X_M,
            "acceleration_channels": ["body_com_ax_mps2",
                                      "body_com_ay_mps2", "yaw_accel_rps2"],
            "acceleration_mean_train_only": acceleration_mean.tolist(),
            "acceleration_scale_train_only": acceleration_scale.tolist(),
            "acceleration_supervision_weight": 1.0,
            "training_run_ids": sorted(data.training_runs),
            "added_highsteer_training_run_ids": sorted(extra_training_runs),
            "validation_run_ids": sorted(validation_refs),
            "future_sensor_or_truth_inputs": False,
            "production_integration": False,
        },
    }, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    (output_root / "checkpoint.sha256").write_text(checkpoint_sha + "\n",
                                                      encoding="utf-8")

    candidate_validation = _curve(
        baseline_data, validation_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    candidate_practice = _curve(
        baseline_data, practice_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    candidate_high = _curve(
        high_data, high_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    accel_validation = _curve(
        baseline_data, validation_refs, parent_acceleration,
        baseline_norm, baseline_config, CONTEXT_STEPS["2.0s"], device)
    accel_practice = _curve(
        baseline_data, practice_refs, parent_acceleration,
        baseline_norm, baseline_config, CONTEXT_STEPS["2.0s"], device)
    accel_high = _curve(
        high_data, high_refs, parent_acceleration,
        baseline_norm, baseline_config, CONTEXT_STEPS["2.0s"], device)
    wp28_validation = _curve(
        baseline_data, validation_refs, parent_wp28,
        baseline_norm, baseline_config, CONTEXT_STEPS["2.0s"], device)
    wp28_practice = _curve(
        baseline_data, practice_refs, parent_wp28,
        baseline_norm, baseline_config, CONTEXT_STEPS["2.0s"], device)
    wp28_high = _curve(
        high_data, high_refs, parent_wp28,
        baseline_norm, baseline_config, CONTEXT_STEPS["2.0s"], device)

    groups = {
        "development_validation": (candidate_validation, accel_validation, wp28_validation),
        "unseen_practice_diagnostic": (candidate_practice, accel_practice, wp28_practice),
        "heldout_highsteer": (candidate_high, accel_high, wp28_high),
    }
    report: dict[str, Any] = {
        "study": "WP27-style truncated training of no-latent rigid acceleration plant with independent high-steer training runs",
        "checkpoint_sha256": checkpoint_sha,
        "baseline_acceleration_checkpoint_sha256": parent_accel_sha,
        "baseline_wp28_checkpoint_sha256": wp28_sha,
        "data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "added_highsteer_train": (sha256_file(training_dataset)
                                      if training_dataset is not None else None),
            "heldout_highsteer_validation": sha256_file(HIGHSTEER_SOURCE),
        },
        "training_runs": sorted(data.training_runs),
        "added_highsteer_training_runs": sorted(extra_training_runs),
        "validation_runs": sorted(validation_refs),
        "heldout_highsteer_validation_runs": sorted(HIGHSTEER_RUNS),
        "training_schedule": [
            {"stage": stage, "horizons_steps": list(choices),
             "updates": updates, "batch_size": batch}
            for stage, choices, updates, batch in stages],
        "training_policy": {
            "name": ("WP27 curriculum with adaptive L2 clipping"
                     if not long_horizon_training
                     else "legacy 0.5-5 s comparator curriculum"),
            "adaptive_l2_clipping": not long_horizon_training,
            "initial_checkpoint_sha256": (parent_accel_sha
                                           if not long_horizon_training else None),
            "normalization_population": (
                "frozen WP28 training split shared across matched data conditions"
                if not long_horizon_training else
                "each training condition independently recomputed"),
            "a2_two_second_yaw_reference_rps": a2_yaw_reference,
            "stop_gate_source": (WP26_A2_EVALUATION_REPORT.relative_to(ROOT).as_posix()
                                 if a2_yaw_reference is not None else None),
        },
        "training": training,
        "midpoint_acceleration_normalization_train_rows_by_run": acceleration_rows,
        "sampler_counts": sampler_counts,
        "results": {},
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "limits": [
            "The additional data covers long holds at 7.5 m/s and signed steering levels; it does not fill the full 0-12 m/s high-steering surface.",
            "The high-steer evaluation uses two independent whole-run captures and remains diagnostic in run-count uncertainty.",
            "This is a 5-second stress evaluation, not a full-lap simulation validation.",
        ],
    }
    for name, (candidate_rows, accel_rows, wp28_rows) in groups.items():
        report["results"][name] = {
            "candidate": candidate_rows,
            "previous_acceleration_model": accel_rows,
            "WP28_direct_model": wp28_rows,
            "macro": {
                "candidate": {h: _macro(v) for h, v in candidate_rows.items()},
                "previous_acceleration_model": {h: _macro(v) for h, v in accel_rows.items()},
                "WP28_direct_model": {h: _macro(v) for h, v in wp28_rows.items()},
            },
            "paired_candidate_minus_previous_acceleration": {
                h: {metric: {
                    **_paired(candidate_rows[h], accel_rows[h], metric),
                    **_bootstrap_delta(candidate_rows[h], accel_rows[h], metric,
                                       SEED + index),
                } for index, metric in enumerate(METRICS)}
                for h in candidate_rows},
            "paired_candidate_minus_WP28": {
                h: {metric: {
                    **_paired(candidate_rows[h], wp28_rows[h], metric),
                    **_bootstrap_delta(candidate_rows[h], wp28_rows[h], metric,
                                       SEED + 100 + index),
                } for index, metric in enumerate(METRICS)}
                for h in candidate_rows},
        }
    _write_json(output_root / "wp27_highsteer_transfer_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--training-dataset", type=Path,
                        default=HIGHSTEER_TRAIN_DATASET)
    parser.add_argument("--no-extra-training-data", action="store_true",
                        help="run the matched curriculum control on the original training split")
    parser.add_argument("--long-horizon-training", action="store_true",
                        help="use the same 0.5-5 s training curriculum as the prior acceleration candidate")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    dataset = (None if args.no_extra_training_data else
               (args.training_dataset if args.training_dataset.is_absolute()
                else ROOT / args.training_dataset))
    report = run(args.device, output, dataset,
                 long_horizon_training=args.long_horizon_training)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "training_seconds": report["training"]["elapsed_seconds"],
        "development_5s": report["results"]["development_validation"]["macro"]["candidate"]["200"],
        "highsteer_5s": report["results"]["heldout_highsteer"]["macro"]["candidate"]["200"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
