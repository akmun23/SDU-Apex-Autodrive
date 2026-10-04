"""Focused numerical/causality checks for the research-only transition model."""

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    HISTORY_STEPS,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    HistoryTransition,
    _bootstrap,
    _metrics,
    _predicted_history_row,
    _rollout,
    _shift_history,
)


def test_history_shift_drops_only_oldest_row_and_appends_prediction() -> None:
    history = torch.arange(2 * HISTORY_STEPS * 7, dtype=torch.float32)
    history = history.reshape(2, HISTORY_STEPS, 7)
    row = torch.full((2, 7), -3.0)
    shifted = _shift_history(history, row)
    torch.testing.assert_close(shifted[:, :-1], history[:, 1:])
    torch.testing.assert_close(shifted[:, -1], row)


def test_predicted_history_row_uses_only_predicted_state_and_next_command() -> None:
    state = torch.tensor([[2.0, -0.5, 0.2, 0.1, 0.7]])
    command = torch.tensor([[0.3, 0.8]])
    norm = {
        "state_mean": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        "state_scale": torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0]),
        "command_mean": torch.tensor([0.1, 0.2]),
        "command_scale": torch.tensor([0.5, 2.0]),
        "history_mean": torch.zeros(7),
        "history_scale": torch.ones(7),
    }
    row = _predicted_history_row(state, command, norm)
    expected = torch.tensor([[5.0, -0.5, 0.2, 0.1, 0.7, 0.25, 1.8]])
    torch.testing.assert_close(row, expected)


def test_rollout_predictions_do_not_depend_on_future_truth_labels() -> None:
    torch.manual_seed(5)
    model = HistoryTransition(np.zeros(5, np.float32), np.ones(5, np.float32))
    history = torch.randn(2, HISTORY_STEPS, 7)
    state = torch.zeros(2, 5)
    pose = torch.zeros(2, 3)
    commands = torch.randn(2, 3, 2)
    target_a = torch.zeros(2, 3, 5)
    pose_a = torch.zeros(2, 3, 3)
    target_b = torch.full((2, 3, 5), 1000.0)
    pose_b = torch.full((2, 3, 3), -1000.0)
    norm = {
        "state_mean": torch.zeros(5), "state_scale": torch.ones(5),
        "command_mean": torch.zeros(2), "command_scale": torch.ones(2),
        "history_mean": torch.zeros(7), "history_scale": torch.ones(7),
    }
    model.eval()
    with torch.no_grad():
        a = _rollout(model, (history, state, pose, commands, target_a, pose_a),
                     norm, 3, PoseIntegrator())
        b = _rollout(model, (history, state, pose, commands, target_b, pose_b),
                     norm, 3, PoseIntegrator())
    torch.testing.assert_close(a["state"], b["state"])
    torch.testing.assert_close(a["pose"], b["pose"])


def test_segmented_rollout_preserves_forward_values_and_backpropagates() -> None:
    torch.manual_seed(17)
    model = HistoryTransition(np.zeros(5, np.float32), np.ones(5, np.float32))
    history = torch.randn(2, HISTORY_STEPS, 7)
    state = torch.zeros(2, 5)
    pose = torch.zeros(2, 3)
    commands = torch.randn(2, 5, 2)
    target = torch.full((2, 5, 5), 0.2)
    target_pose = torch.zeros(2, 5, 3)
    norm = {
        "state_mean": torch.zeros(5), "state_scale": torch.ones(5),
        "command_mean": torch.zeros(2), "command_scale": torch.ones(2),
        "history_mean": torch.zeros(7), "history_scale": torch.ones(7),
    }
    arrays = (history, state, pose, commands, target, target_pose)
    with torch.no_grad():
        full_values = _rollout(model, arrays, norm, 5, PoseIntegrator())
    model.zero_grad(set_to_none=True)
    segmented = _rollout(
        model, arrays, norm, 5, PoseIntegrator(), collect_loss=True,
        detach_every_steps=2, backward_segments=True)
    torch.testing.assert_close(segmented["state"], full_values["state"])
    torch.testing.assert_close(segmented["pose"], full_values["pose"])
    assert torch.isfinite(segmented["loss"])
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_run_bootstrap_resamples_metric_values_not_row_indexes() -> None:
    interval = _bootstrap({"run_a": -4.0, "run_b": -2.0, "run_c": -3.0}, 19)
    assert interval is not None
    assert -4.0 <= interval[0] <= interval[1] <= -2.0


def test_metric_averages_per_start_before_run_aggregation() -> None:
    predicted_state = np.zeros((2, 2, 5), dtype=np.float32)
    predicted_pose = np.zeros((2, 2, 3), dtype=np.float32)
    truth_state = np.zeros_like(predicted_state)
    truth_pose = np.zeros_like(predicted_pose)
    truth_pose[1, :, 0] = 2.0
    metrics = _metrics(predicted_state, predicted_pose, truth_state, truth_pose, 2,
                       include_actuator=False)
    assert metrics["position_radial_trajectory_rmse_m"] == 1.0
    assert metrics["position_endpoint_error_m"] == 1.0
