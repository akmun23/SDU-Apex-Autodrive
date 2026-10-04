"""Focused structural tests for the shared-history recurrent plant."""

import numpy as np
import torch
from torch import nn

from tools.vehicle_dynamics_learning.consistent_history_teacher import make_model


def _model():
    return make_model(
        torch, nn,
        history_mean=np.zeros(11, dtype=np.float32),
        history_scale=np.ones(11, dtype=np.float32),
        state_mean=np.zeros(9, dtype=np.float32),
        state_scale=np.ones(9, dtype=np.float32),
        command_mean=np.zeros(2, dtype=np.float32),
        command_scale=np.ones(2, dtype=np.float32),
        acceleration_bounds=np.ones(5, dtype=np.float32),
        roll_oscillator_coefficients=np.asarray((0.0, -1.0, -0.3),
                                                dtype=np.float32),
        steering_delay_steps=1,
        steering_alpha=0.5,
        throttle_delay_steps=1,
        throttle_alpha=0.25,
    )()


def test_shared_gru_extension_matches_encoding_concatenated_history():
    torch.manual_seed(4)
    model = _model().eval()
    history = torch.randn(2, 80, 11)
    row = torch.randn(2, 1, 11)
    with torch.no_grad():
        _, hidden = model.history_gru(history)
        _, continued = model.history_gru(row, hidden)
        _, concatenated = model.history_gru(torch.cat((history, row), dim=1))
    torch.testing.assert_close(continued, concatenated, rtol=1e-5, atol=1e-6)


def test_rollout_uses_fixed_25ms_dynamics_and_has_finite_full_state():
    model = _model().eval()
    initial = torch.tensor([[4.0, 0.2, 0.5, 0.1, 0.3,
                             3.9, 4.1, 0.04, 0.02]], dtype=torch.float32)
    history = torch.zeros(1, 80, 11)
    history[:, -1, :9] = initial
    delayed = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
    commands = torch.tensor([[[0.6, 0.8], [0.5, 0.6]]], dtype=torch.float32)
    with torch.no_grad():
        states, acceleration, latent, gates = model.rollout(
            initial, delayed, history, commands)
    assert states.shape == (1, 2, 9)
    assert acceleration.shape == (1, 2, 5)
    assert latent.shape == (1, 2, 32)
    assert gates.shape == (1, 2, 1)
    assert torch.isfinite(states).all()
    # The zero-initialized acceleration head leaves wheel speeds unchanged;
    # body-frame transport and the fitted roll oscillator still advance.
    torch.testing.assert_close(states[0, 0, 5:7], initial[0, 5:7])
    assert not torch.equal(states[0, 0, :3], initial[0, :3])
    assert not torch.equal(states[0, 0, 7:9], initial[0, 7:9])
    torch.testing.assert_close(states[0, 0, 3], torch.tensor(0.15))
    torch.testing.assert_close(states[0, 0, 4], torch.tensor(0.325))
