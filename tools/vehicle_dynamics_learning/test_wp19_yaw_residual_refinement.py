"""Small numerical checks for the WP19 yaw-only residual experiment."""

import torch

from tools.vehicle_dynamics_learning.run_wp19_yaw_residual_refinement import (
    YawResidual,
)


def test_zero_initialized_yaw_residual_preserves_parent_transition() -> None:
    torch.manual_seed(7)
    residual = YawResidual(yaw_increment_scale=0.08)
    inputs = torch.randn(6, 7)
    hidden = torch.randn(6, 32)
    base_delta = torch.randn(6, 3)
    correction = residual(inputs, hidden, base_delta)
    assert correction.shape == (6,)
    torch.testing.assert_close(correction, torch.zeros_like(correction))


def test_yaw_residual_output_is_finite_after_parameter_update() -> None:
    residual = YawResidual(yaw_increment_scale=0.08)
    optimizer = torch.optim.Adam(residual.parameters(), lr=1e-3)
    inputs = torch.randn(4, 7)
    hidden = torch.randn(4, 32)
    base_delta = torch.randn(4, 3)
    target = torch.full((4,), 0.02)
    loss = torch.nn.functional.mse_loss(
        residual(inputs, hidden, base_delta), target)
    loss.backward()
    optimizer.step()
    output = residual(inputs, hidden, base_delta)
    assert torch.isfinite(output).all()
    assert torch.all(output.abs() < 1.0)
