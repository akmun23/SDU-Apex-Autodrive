from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    HISTORY_FEATURE_NAMES,
    ActuatorTransition,
    AugmentedStateSpacePlant,
    PlantConfig,
    PoseIntegrator,
    encoder_increment_to_rate,
    encoder_rate_to_increment,
    load_checkpoint,
    mirror_command,
    mirror_history,
    mirror_measurement,
    mirror_pose,
    mirror_state,
    save_checkpoint,
)


def _config(**overrides) -> PlantConfig:
    data = dict(
        support_bank=tuple(tuple(np.zeros(7)) for _ in range(12)),
        support_run_index=tuple(index // 3 for index in range(12)),
        support_run_ids=("a", "b", "c", "d"),
        support_calibrated=True,
    )
    data.update(overrides)
    return PlantConfig(**data)


def _history(batch: int = 1) -> torch.Tensor:
    result = torch.zeros(batch, 80, len(HISTORY_FEATURE_NAMES))
    result[..., 0] = 3.0
    result[..., 4] = 0.2
    result[..., 9] = 1.0
    return result


def _new_model(config: PlantConfig | None = None) -> AugmentedStateSpacePlant:
    torch.manual_seed(17)
    return AugmentedStateSpacePlant(config or _config()).eval()


def test_reset_rejects_zero_length_and_invalid_shapes() -> None:
    plant = _new_model()
    with pytest.raises(ValueError):
        plant.reset(torch.zeros(0, 10), torch.zeros(5), torch.zeros(3))
    with pytest.raises(ValueError):
        plant.reset(torch.zeros(80, 9), torch.zeros(5), torch.zeros(3))
    with pytest.raises(ValueError):
        plant.reset(torch.zeros(80, 10), torch.zeros(4), torch.zeros(3))


def test_deterministic_repeatability_from_same_reset() -> None:
    plant = _new_model()
    history = _history()[0]
    state = torch.tensor([3.0, 0.1, 0.2, 0.01, 0.2])
    pose = torch.tensor([1.0, -2.0, 0.3])
    commands = torch.tensor([[0.04, 0.3], [0.06, 0.35], [0.0, 0.2]])
    plant.reset(history, state, pose)
    first = plant.rollout(commands)
    plant.reset(history, state, pose)
    second = plant.rollout(commands)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)


def test_step_interface_has_no_future_sensor_or_truth_argument() -> None:
    parameters = set(inspect.signature(AugmentedStateSpacePlant.step).parameters)
    assert parameters == {"self", "command"}
    plant = _new_model()
    plant.reset(_history()[0], torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]),
                torch.zeros(3))
    state_a, measurement_a, _ = plant.step(torch.tensor([0.1, 0.2]))
    plant.reset(_history()[0], torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]),
                torch.zeros(3))
    state_b, measurement_b, _ = plant.step(torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(state_a, state_b)
    torch.testing.assert_close(measurement_a, measurement_b)


def test_latent_transition_can_exclude_generated_measurement_feedback() -> None:
    plant = _new_model(_config(latent_measurement_feedback=False))
    assert plant.latent_transition is not None
    assert plant.latent_transition.cell.input_size == 10
    plant.reset(_history()[0], torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]),
                torch.zeros(3))
    state, measurement, _ = plant.step(torch.tensor([0.1, 0.2]))
    assert torch.isfinite(state).all()
    assert torch.isfinite(measurement).all()


def test_pose_integrator_exact_straight_and_lateral_motion() -> None:
    integrate = PoseIntegrator(0.025)
    pose = torch.zeros(1, 3)
    next_pose = integrate(pose, torch.tensor([[2.0, 0.5, 0.0]]))
    torch.testing.assert_close(next_pose, torch.tensor([[0.05, 0.0125, 0.0]]),
                               atol=1e-7, rtol=0.0)


def test_pose_integrator_exact_constant_yaw_and_heading_wrap() -> None:
    integrate = PoseIntegrator(0.025)
    pose = torch.zeros(1, 3)
    state = torch.tensor([[2.0, 0.0, 1.0]])
    result = integrate(pose, state)
    torch.testing.assert_close(result[0, 0], torch.tensor(2.0 * np.sin(0.025),
                                                        dtype=torch.float32),
                               atol=1e-7, rtol=0.0)
    torch.testing.assert_close(result[0, 1], torch.tensor(
        2.0 * (1-np.cos(0.025)), dtype=torch.float32),
                               atol=1e-7, rtol=0.0)
    wrapped = integrate(torch.tensor([[0.0, 0.0, np.pi - 0.01]]), state)
    assert -np.pi <= wrapped[0, 2] <= np.pi
    torch.testing.assert_close(wrapped[0, 2], torch.tensor(-np.pi + 0.015,
                                                          dtype=torch.float32),
                               atol=1e-6, rtol=0.0)


def test_actuator_delay_zero_and_one_sample() -> None:
    feedback = torch.tensor([[0.1, 0.2]])
    command = torch.tensor([[0.5, 0.8]])
    previous = torch.tensor([[-0.3, 0.4]])
    immediate = ActuatorTransition(0.5, 0.25, 0, 0)
    delayed = ActuatorTransition(0.5, 0.25, 1, 1)
    torch.testing.assert_close(immediate(feedback, command, previous),
                               torch.tensor([[0.3, 0.35]]))
    torch.testing.assert_close(delayed(feedback, command, previous),
                               torch.tensor([[-0.1, 0.25]]))


def test_mirror_transforms_are_involutions() -> None:
    state = torch.randn(5)
    command = torch.randn(2)
    history = torch.randn(80, 10)
    pose = torch.randn(3)
    torch.testing.assert_close(mirror_state(mirror_state(state)), state)
    torch.testing.assert_close(mirror_command(mirror_command(command)), command)
    torch.testing.assert_close(mirror_history(mirror_history(history)), history)
    torch.testing.assert_close(mirror_pose(mirror_pose(pose)), pose)


def test_encoder_mirror_swaps_left_right_without_sign_flip() -> None:
    measurement = torch.tensor([0.3, 0.7])
    torch.testing.assert_close(mirror_measurement(measurement),
                               torch.tensor([0.7, 0.3]))
    torch.testing.assert_close(mirror_measurement(mirror_measurement(measurement)),
                               measurement)


def test_encoder_increment_and_surface_rate_reconstruct_each_other() -> None:
    rate = torch.tensor([[1.2, 2.4], [-0.6, 0.3]])
    torch.testing.assert_close(encoder_increment_to_rate(
        encoder_rate_to_increment(rate)), rate, atol=1e-7, rtol=0.0)


def test_model_outputs_finite_for_random_supported_states() -> None:
    plant = _new_model()
    history = _history(4)
    state = torch.randn(4, 5)
    state[:, 0] = torch.rand(4) * 8.0
    state[:, 3] = torch.rand(4) * 0.8 - 0.4
    state[:, 4] = torch.rand(4)
    plant.reset(history, state, torch.randn(4, 3))
    result = plant.rollout(torch.randn(4, 24, 2) * 0.2)
    assert all(torch.isfinite(value).all() for value in result.values())
    assert torch.max(torch.abs(result["states"][..., :3])) < 20.0
    assert torch.max(torch.abs(result["measurements"])) <= max(
        _config().encoder_increment_limit) + 1e-6


def test_support_api_is_bounded_and_respects_calibrated_thresholds() -> None:
    adapter = _new_model().support_estimator
    near = adapter(torch.zeros(1, 7))
    far = adapter(torch.ones(1, 7) * 10.0)
    assert 0.0 <= float(near["confidence"][0]) <= 1.0
    assert 0.0 <= float(far["confidence"][0]) <= 1.0
    assert near["score"][0] < far["score"][0]
    assert near["confidence"][0] >= far["confidence"][0]


def test_checkpoint_round_trip_preserves_rollout(tmp_path) -> None:
    plant = _new_model()
    plant.reset(_history()[0], torch.tensor([3.0, 0.1, 0.2, 0.0, 0.2]),
                torch.tensor([1.0, 2.0, 0.3]))
    expected = plant.rollout(torch.tensor([[0.1, 0.2], [0.0, 0.4]]))
    path = tmp_path / "plant.pt"
    save_checkpoint(path, plant, {"test": "wp21"})
    restored, extra = load_checkpoint(path)
    restored.eval()
    restored.reset(_history()[0], torch.tensor([3.0, 0.1, 0.2, 0.0, 0.2]),
                   torch.tensor([1.0, 2.0, 0.3]))
    actual = restored.rollout(torch.tensor([[0.1, 0.2], [0.0, 0.4]]))
    assert extra == {"test": "wp21"}
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0.0, atol=0.0)


def test_batched_and_single_rollouts_match() -> None:
    batched = _new_model()
    singles = _new_model()
    singles.load_state_dict(batched.state_dict())
    histories = _history(2)
    states = torch.tensor([[3.0, 0.1, 0.2, 0.0, 0.2],
                           [4.0, -0.2, -0.1, 0.1, 0.3]])
    poses = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.4]])
    commands = torch.zeros(2, 5, 2)
    commands[0, :, 0] = 0.1
    commands[1, :, 0] = -0.1
    batched.reset(histories, states, poses)
    result = batched.rollout(commands)
    singles.reset(histories[0], states[0], poses[0])
    first = singles.rollout(commands[0])
    for key in result:
        torch.testing.assert_close(result[key][0], first[key], rtol=1e-6, atol=1e-7)


def test_reset_clears_latent_and_all_recursive_buffers() -> None:
    plant = _new_model()
    history = _history()[0]
    state = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.2])
    pose = torch.zeros(3)
    commands = torch.full((8, 2), 0.2)
    plant.reset(history, state, pose)
    expected = plant.rollout(commands)
    plant.rollout(torch.full((9, 2), -0.3))
    plant.reset(history, state, pose)
    actual = plant.rollout(commands)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0.0, atol=0.0)


def test_batch_members_do_not_share_hidden_or_encoder_state() -> None:
    batched = _new_model()
    isolated = _new_model()
    isolated.load_state_dict(batched.state_dict())
    with torch.no_grad():
        batched.body_residual.head.weight.normal_(mean=0.0, std=0.02)
    isolated.load_state_dict(batched.state_dict())
    histories = _history(2)
    histories[1, :, 1] = 0.4
    states = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.2],
                           [3.0, 0.0, 0.0, 0.0, 0.2]])
    poses = torch.zeros(2, 3)
    commands = torch.zeros(2, 3, 2)
    batched.reset(histories, states, poses)
    batch_out = batched.rollout(commands)
    isolated.reset(histories[0], states[0], poses[0])
    single_out = isolated.rollout(commands[0])
    torch.testing.assert_close(batch_out["states"][0], single_out["states"])
    isolated.reset(histories[1], states[1], poses[1])
    second_out = isolated.rollout(commands[1])
    torch.testing.assert_close(batch_out["states"][1], second_out["states"])
    assert not torch.equal(batch_out["states"][0], batch_out["states"][1])


def test_wp23_a0_has_no_latent_state_but_keeps_recursive_api() -> None:
    config = _config(latent_enabled=False, latent_size=0)
    plant = _new_model(config)
    assert plant.history_encoder is None
    assert plant.latent_transition is None
    plant.reset(_history()[0], torch.tensor([3.0, 0.0, 0.0, 0.0, 0.2]),
                torch.zeros(3))
    result = plant.rollout(torch.zeros(1, 3, 2))
    assert result["latents"].shape == (3, 0)
    assert all(torch.isfinite(value).all() for value in result.values())


def test_wp23_a1_predicts_normalized_next_body_state_without_nominal_hold() -> None:
    config = _config(body_transition_mode="direct_state",
                     state_mean=(2.0, 0.1, -0.2, 0.0, 0.2),
                     state_scale=(1.0, 0.5, 0.25, 1.0, 1.0),
                     body_state_normalized_limit=(2.0, 2.0, 2.0))
    plant = _new_model(config)
    initial = torch.tensor([4.0, 0.4, 0.3, 0.0, 0.2])
    plant.reset(_history()[0], initial, torch.zeros(3))
    state, _, _ = plant.step(torch.zeros(2))
    torch.testing.assert_close(state[:3], torch.tensor([2.0, 0.1, -0.2]))
    assert not torch.equal(state[:3], initial[:3])
