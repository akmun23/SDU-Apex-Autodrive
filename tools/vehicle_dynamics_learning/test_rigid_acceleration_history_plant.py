"""Math/interface checks for the acceleration-integrated offline plant."""

import numpy as np
import torch

from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
    integrate_planar_body,
    midpoint_acceleration_from_transition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    FEATURES,
    MAX_CONTEXT_STEPS,
)


def test_midpoint_labels_reconstruct_arbitrary_truth_transitions() -> None:
    rng = np.random.default_rng(20261004)
    current = rng.normal(size=(128, 3))
    following = rng.normal(size=(128, 3))
    dt_s = 0.025
    x_com = 0.15532

    acceleration = midpoint_acceleration_from_transition(
        current, following, dt_s, x_com)
    actual = integrate_planar_body(current, acceleration, dt_s, x_com)
    np.testing.assert_allclose(actual, following, rtol=0.0, atol=1e-12)


def test_implicit_midpoint_preserves_speed_under_coriolis_coupling() -> None:
    rng = np.random.default_rng(25)
    body = rng.normal(size=(128, 3))
    body[:, 2] *= 10.0
    acceleration = np.zeros_like(body)
    following = integrate_planar_body(body, acceleration, 0.025)
    v_com = body[:, 1] + 0.15532 * body[:, 2]
    v_com_next = following[:, 1] + 0.15532 * following[:, 2]
    np.testing.assert_allclose(
        np.hypot(body[:, 0], v_com), np.hypot(following[:, 0], v_com_next),
        rtol=0.0, atol=1e-12)


def test_planar_acceleration_rejects_invalid_state_or_timestep() -> None:
    with np.testing.assert_raises(ValueError):
        integrate_planar_body(np.zeros(2), np.zeros(3), 0.025)
    with np.testing.assert_raises(ValueError):
        integrate_planar_body(np.zeros(3), np.zeros(3), 0.0)
    with np.testing.assert_raises(ValueError):
        integrate_planar_body(np.array([0.0, np.nan, 0.0]), np.zeros(3), 0.025)


def test_transition_is_finite_differentiable_and_uses_implicit_rigid_step() -> None:
    state_mean = np.zeros(5, dtype=np.float32)
    state_scale = np.ones(5, dtype=np.float32)
    accel_mean = np.array([0.7, -0.4, 1.2], dtype=np.float32)
    accel_scale = np.ones(3, dtype=np.float32)
    actuator_mean = np.array([0.01, -0.02], dtype=np.float32)
    actuator_scale = np.ones(2, dtype=np.float32)
    model = RigidAccelerationHistoryTransition(
        state_mean, state_scale, accel_mean, accel_scale,
        actuator_mean, actuator_scale)
    history = torch.zeros(2, MAX_CONTEXT_STEPS, FEATURES)
    mask = torch.zeros(2, MAX_CONTEXT_STEPS)
    mask[:, -80:] = 1.0
    state = torch.tensor([[4.0, 0.3, 1.1, 0.2, 0.5],
                          [7.0, -0.4, -1.7, -0.1, 0.8]], requires_grad=True)
    command = torch.zeros(2, 2)

    delta = model(history, mask, state, command)
    expected_body = torch.as_tensor(integrate_planar_body(
        state.detach().numpy()[:, :3], np.broadcast_to(accel_mean, (2, 3)).copy(),
        0.025),
        dtype=torch.float32)
    torch.testing.assert_close(state.detach()[:, :3] + delta[:, :3], expected_body)
    torch.testing.assert_close(delta[:, 3:], torch.as_tensor(
        np.broadcast_to(actuator_mean, (2, 2)).copy(), dtype=torch.float32))
    predicted_next = state.detach() + delta.detach()
    recovered_acceleration = model.midpoint_acceleration_label(
        state.detach(), predicted_next)
    torch.testing.assert_close(recovered_acceleration,
                                   torch.as_tensor(np.broadcast_to(
                                       accel_mean, (2, 3)).copy(),
                                   dtype=torch.float32), atol=5e-5, rtol=0.0)
    assert delta.shape == (2, 5)
    assert torch.isfinite(delta).all()
    delta.square().sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
