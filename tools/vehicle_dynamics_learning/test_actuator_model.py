import numpy as np
import pytest

from tools.vehicle_dynamics_learning.actuator_model import rollout_channel


def test_zero_delay_response_uses_current_command():
    result = rollout_channel(
        initial_feedback=0.0,
        commands=np.asarray([0.0, 1.0, 1.0, 1.0]),
        start=1,
        horizon_steps=2,
        delay_steps=0,
        alpha=0.5,
    )
    np.testing.assert_allclose(result, [0.0, 0.5, 0.75], rtol=0, atol=1e-12)


def test_one_sample_delay_uses_only_past_and_current_command():
    result = rollout_channel(
        initial_feedback=0.0,
        commands=np.asarray([0.0, 1.0, 1.0, 1.0]),
        start=1,
        horizon_steps=2,
        delay_steps=1,
        alpha=0.5,
    )
    np.testing.assert_allclose(result, [0.0, 0.0, 0.5], rtol=0, atol=1e-12)


def test_rollout_requires_prior_command_when_delay_is_one():
    with pytest.raises(ValueError, match="invalid actuator rollout"):
        rollout_channel(0.0, np.ones(4), start=0, horizon_steps=2,
                        delay_steps=1, alpha=0.5)
