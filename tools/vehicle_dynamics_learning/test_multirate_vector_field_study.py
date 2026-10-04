"""Focused numerical checks for the research-only multirate integrator."""

import numpy as np
import torch

from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    ContinuousStateVectorField,
    _ode_advance,
)


def _constant_field(rate: list[float]) -> ContinuousStateVectorField:
    model = ContinuousStateVectorField(np.asarray(rate, dtype=np.float32),
                                       np.zeros(5, dtype=np.float32))
    model.eval()
    return model


def test_midpoint_substeps_integrate_constant_state_rate_exactly() -> None:
    model = _constant_field([0.4, -0.2, 0.1, 0.3, -0.1])
    current = torch.tensor([[1.0, 0.5, -0.3, 0.2, 0.1]])
    previous = current.clone()
    command = torch.zeros((1, 2))
    pose = torch.zeros((1, 3))
    state_mean = torch.zeros(5)
    state_scale = torch.ones(5)
    expected = current + 0.025 * torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.1]])

    with torch.no_grad():
        for substeps in (1, 5, 25):
            state, _ = _ode_advance(model, current, previous, command, command,
                                    pose, state_mean, state_scale, substeps)
            torch.testing.assert_close(state, expected, rtol=2e-6, atol=2e-6)


def test_pose_integration_error_reduces_when_body_rate_changes_within_interval() -> None:
    model = _constant_field([4.0, 0.3, 4.0, 0.0, 0.0])
    current = torch.tensor([[4.0, 0.2, 1.0, 0.0, 0.0]])
    previous = current.clone()
    command = torch.zeros((1, 2))
    pose = torch.zeros((1, 3))
    state_mean = torch.zeros(5)
    state_scale = torch.ones(5)

    with torch.no_grad():
        coarse = _ode_advance(model, current, previous, command, command, pose,
                              state_mean, state_scale, 1)[1]
        fine = _ode_advance(model, current, previous, command, command, pose,
                            state_mean, state_scale, 25)[1]
        reference = _ode_advance(model, current, previous, command, command, pose,
                                 state_mean, state_scale, 1000)[1]

    coarse_error = torch.linalg.vector_norm(coarse - reference)
    fine_error = torch.linalg.vector_norm(fine - reference)
    assert torch.isfinite(reference).all()
    assert fine_error < 0.2 * coarse_error
