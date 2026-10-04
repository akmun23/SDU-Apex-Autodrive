#!/usr/bin/env python3
"""Separate state-error and generated-history feedback on high-steer runs.

Oracle branches are offline interventions only. They quantify how a frozen
plant degrades when either its predicted state or its generated history is
allowed to feed back; they are not valid deployment rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    EXPECTED_RUNS,
    HIGHSTEER_SOURCE,
    _capture_from_archive,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    OUTPUT_ROOT,
    FixedCapacityHistoryTransition,
    _batch_arrays,
    _history_window,
    _metrics,
    _normalization,
    _torch_norm,
    advance_context,
    pack_history,
    sha256_file,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _load_data,
    _training_windows_and_stats,
)
from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)


CHECKPOINT = (OUTPUT_ROOT / "long_rollout_5s_selected_context_v1"
              / "checkpoint.pt")
OUTPUT = OUTPUT_ROOT / "history_feedback_intervention_diagnostic_v1.json"
HORIZON = 200
MODES = ("free_state_and_history", "recorded_history_only",
         "truth_state_only", "fully_teacher_forced")


def _recorded_history(capture, sequence_index: int, source_row: int,
                      context_steps: int, config):
    raw = _history_window(capture, sequence_index, source_row, context_steps)
    return pack_history(
        raw, context_steps,
        np.asarray(config.history_mean[:7], dtype=np.float32),
        np.asarray(config.history_scale[:7], dtype=np.float32))


def _score_intervention(data, refs, model, norm_np, config, context_steps,
                        device, mode: str, horizon: int
                        ) -> dict[str, dict[str, float]]:
    norm = _torch_norm(norm_np, config, device)
    integrator = PoseIntegrator(DT_S).to(device)
    output = {}
    model.eval()
    with torch.no_grad():
        for run_id, run_refs in sorted(refs.items()):
            arrays = _batch_arrays(
                data, run_refs, horizon, context_steps,
                norm_np, config, device)
            history, mask, state, pose, commands, truth, truth_pose = arrays
            capture = data.captures[int(run_refs[0][0])]
            current_truth = torch.cat((state[:, None], truth[:, :-1]), dim=1)
            predicted_states, predicted_poses = [], []
            for step in range(horizon):
                if mode in ("truth_state_only", "fully_teacher_forced"):
                    state = current_truth[:, step]
                if mode in ("recorded_history_only", "fully_teacher_forced"):
                    batches, masks = [], []
                    for _, sequence_index, source_row in run_refs:
                        packed, history_mask = _recorded_history(
                            capture, sequence_index, source_row + step,
                            context_steps, config)
                        batches.append(packed)
                        masks.append(history_mask)
                    history = torch.as_tensor(
                        np.asarray(batches), dtype=torch.float32, device=device)
                    mask = torch.as_tensor(
                        np.asarray(masks), dtype=torch.float32, device=device)

                delta = model(history, mask, state, commands[:, step])
                next_state = state + delta
                current_body = state[:, :3] * norm["state_scale"][:3] \
                    + norm["state_mean"][:3]
                next_body = next_state[:, :3] * norm["state_scale"][:3] \
                    + norm["state_mean"][:3]
                pose = integrator(pose, 0.5 * (current_body + next_body))
                if not torch.isfinite(next_state).all() or not torch.isfinite(pose).all():
                    raise FloatingPointError(
                        f"{mode} produced a non-finite rollout at {step + 1}")
                predicted_states.append(next_state)
                predicted_poses.append(pose)

                if step + 1 < horizon and mode in (
                        "free_state_and_history", "truth_state_only"):
                    physical_state = (next_state * norm["state_scale"]
                                      + norm["state_mean"])
                    physical_command = (commands[:, step + 1]
                                        * norm["command_scale"]
                                        + norm["command_mean"])
                    row = torch.cat((physical_state, physical_command), dim=-1)
                    row = (row - norm["history_mean"]) / norm["history_scale"]
                    history, mask = advance_context(
                        history, mask, row, context_steps)
                if mode not in ("truth_state_only", "fully_teacher_forced"):
                    state = next_state

            predicted_state = (torch.stack(predicted_states, dim=1).cpu().numpy()
                               * norm_np["state_scale"]
                               + norm_np["state_mean"])
            output[run_id] = _metrics(
                predicted_state,
                torch.stack(predicted_poses, dim=1).cpu().numpy(),
                truth.cpu().numpy() * norm_np["state_scale"]
                + norm_np["state_mean"],
                truth_pose.cpu().numpy(), horizon)
    return output


def evaluate(device_name: str = "cpu", output_path: Path = OUTPUT,
             checkpoint_path: Path = CHECKPOINT
             ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)

    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    metadata = saved.get("metadata", {})
    if "acceleration_mean_train_only" in metadata:
        model = RigidAccelerationHistoryTransition(
            norm_np["state_mean"], norm_np["state_scale"],
            np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
            np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
            norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
            dt_s=float(metadata["dt_s"]),
            rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"]),
        ).to(device)
        model_kind = "rigid acceleration"
    else:
        model = FixedCapacityHistoryTransition(
            norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
        model_kind = "WP28 direct transition"
    model.load_state_dict(saved["state_dict"], strict=True)
    expected_sha = (checkpoint_path.parent / "checkpoint.sha256").read_text(
        encoding="utf-8").strip()
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != expected_sha:
        raise RuntimeError("WP28 checkpoint hash does not match its sidecar")

    capture, pose = _capture_from_archive(HIGHSTEER_SOURCE, EXPECTED_RUNS)
    refs = _sample_all_valid_context_refs(
        capture, HORIZON, history_steps=80)
    holdout = SimpleNamespace(captures=[capture], poses=[pose])
    results = {
        mode: _score_intervention(
            holdout, refs, model, norm_np, config,
            CONTEXT_STEPS["2.0s"], device, mode, HORIZON)
        for mode in MODES}
    all_metrics = sorted(next(iter(next(iter(results.values())).values())).keys())
    report = {
        "study": f"{model_kind} free-rollout feedback path intervention on preserved high-steering runs",
        "checkpoint_sha256": checkpoint_sha,
        "source_capture_sha256": sha256_file(HIGHSTEER_SOURCE),
        "run_ids": sorted(EXPECTED_RUNS),
        "start_count_by_run": {run: len(run_refs)
                                for run, run_refs in refs.items()},
        "horizon_steps": HORIZON,
        "horizon_seconds": HORIZON * DT_S,
        "context_steps": CONTEXT_STEPS["2.0s"],
        "per_run_metrics": results,
        "run_macro_metrics": {
            mode: {metric: float(np.mean([
                values[metric] for values in by_run.values()]))
                   for metric in all_metrics}
            for mode, by_run in results.items()},
        "oracle_interventions_are_deployment_valid": False,
        "interpretation": {
            "free_state_and_history": "normal recursive prediction",
            "recorded_history_only": (
                "state remains recursively predicted; only admitted history is "
                "replaced each step by recorded causal history"),
            "truth_state_only": (
                "state is reset to simulator truth each step; generated history "
                "still feeds the model"),
            "fully_teacher_forced": (
                "both state and history are reset from recorded truth at every "
                "step; one-step fit diagnostic only"),
        },
        "simulator_launched": False,
        "production_integration": False,
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    checkpoint = (args.checkpoint if args.checkpoint.is_absolute()
                  else ROOT / args.checkpoint)
    report = evaluate(args.device, output, checkpoint)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "run_macro_metrics": report["run_macro_metrics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
