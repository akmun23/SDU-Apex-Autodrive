"""Command-to-feedback actuator model for the offline plant."""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    ActuatorFit,
    fit_actuator_dynamics,
)


def load_body_dataset(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _actuator_fit_view(data: dict[str, Any]) -> dict[str, Any]:
    inputs = np.asarray(data["inputs"], dtype=np.float64)
    commands = np.asarray(data["stored_commands"], dtype=np.float64)
    run_id = np.asarray(data["run_id"]).astype(str)
    splits = np.asarray(data["split"]).astype(str)
    bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
    sequence_runs = np.asarray(data["sequence_run_id"]).astype(str)
    conditions = np.asarray(data["sequence_condition_id"], dtype=np.int32)
    sequence_splits = np.asarray(data["sequence_split"]).astype(str)
    if (inputs.shape != (len(run_id), 2) or commands.shape != inputs.shape
            or splits.shape != (len(run_id),)
            or bounds.shape != (len(sequence_runs), 2)
            or conditions.shape != (len(sequence_runs),)
            or sequence_splits.shape != (len(sequence_runs),)):
        raise ValueError("body dataset actuator fields are not aligned")
    run_ids = list(dict.fromkeys(sequence_runs.tolist()))
    run_index = {run: index for index, run in enumerate(run_ids)}
    run_splits = []
    for run in run_ids:
        roles = set(splits[run_id == run].tolist())
        if len(roles) != 1:
            raise ValueError(f"run has multiple split labels: {run}")
        run_splits.append(roles.pop())
    sequence_run_index = np.asarray(
        [run_index[run] for run in sequence_runs], dtype=np.int32)
    frames = np.zeros((len(inputs), 9), dtype=np.float64)
    frames[:, 3:5] = inputs
    frames[:, 7:9] = commands
    return {
        "frames": frames,
        "bounds": bounds,
        "seq_run": sequence_run_index,
        "splits": np.asarray(run_splits),
        "run_ids": np.asarray(run_ids),
        "run_families": np.full(len(run_ids), "openplane", dtype="U32"),
        "sequence_condition_id": conditions,
        "sequence_split": sequence_splits,
    }


def fit_actuator_model(data: dict[str, Any]) -> tuple[ActuatorFit, dict[str, Any]]:
    """Select delay/first-order gain using training runs only, macro by run."""
    view = _actuator_fit_view(data)
    fit = fit_actuator_dynamics(view)
    return fit, view


def rollout_channel(initial_feedback: float, commands: np.ndarray,
                    start: int, horizon_steps: int, delay_steps: int,
                    alpha: float) -> np.ndarray:
    """Free-run one feedback channel; requires known command history for delay."""
    command = np.asarray(commands, dtype=np.float64)
    if (command.ndim != 1 or start < delay_steps or horizon_steps < 1
            or start + horizon_steps >= len(command)
            or delay_steps not in (0, 1)
            or not np.isfinite(initial_feedback)
            or not np.isfinite(command).all()
            or not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0):
        raise ValueError("invalid actuator rollout state or horizon")
    prediction = np.empty(horizon_steps + 1, dtype=np.float64)
    prediction[0] = initial_feedback
    for step in range(horizon_steps):
        command_index = start + step - delay_steps
        prediction[step + 1] = prediction[step] + alpha * (
            command[command_index] - prediction[step])
    return prediction

