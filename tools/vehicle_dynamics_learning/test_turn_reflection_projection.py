"""Mathematical checks for trajectory-level turn-reflection projection."""

import unittest

import torch

from tools.vehicle_dynamics_learning.turn_reflection import (
    reflect_acceleration_targets,
    reflect_commands,
    reflect_history_features,
    reflect_state_channels,
)
from tools.vehicle_dynamics_learning.turn_reflection_projection import (
    HighSteerStepwiseReflectionModel,
    TurnReflectionProjectedModel,
)


class AsymmetricToyModel:
    """Small deliberately asymmetric rollout to test the projection algebra."""

    history_state_size = 9
    include_roll_state = True
    include_raw_encoder_history = False
    include_wheel_innovation_history = False

    def rollout(self, initial, delayed, history, commands):
        state = initial
        states, accelerations, gates = [], [], []
        for command in commands.unbind(dim=1):
            steer, throttle = command.unbind(dim=-1)
            next_state = state.clone()
            next_state[..., 0] += 0.01 * (1.0 + 2.0 * steer + throttle)
            next_state[..., 1] += 0.02 * steer + 0.003 * state[..., 0]
            next_state[..., 2] += 0.04 * steer.square() + 0.01 * steer
            next_state[..., 3] = 0.3 * steer + 0.02 * delayed[..., 0]
            next_state[..., 4] = 0.5 * throttle + 0.01 * delayed[..., 1]
            next_state[..., 5] += 0.07 * throttle + 0.03 * steer
            next_state[..., 6] += 0.04 * throttle - 0.01 * steer
            next_state[..., 7] += 0.02 * steer + 0.001 * state[..., 8]
            next_state[..., 8] += 0.03 * steer + 0.002 * state[..., 7]
            states.append(next_state)
            accelerations.append(torch.stack((
                0.1 + steer, 0.2 * steer, 0.3 * steer.square() + steer,
                0.4 * throttle + steer, 0.5 * throttle - steer,
            ), dim=-1))
            gates.append(torch.sigmoid(steer + throttle))
            state = next_state
        state_sequence = torch.stack(states, dim=1)
        acceleration_sequence = torch.stack(accelerations, dim=1)
        gate_sequence = torch.stack(gates, dim=1)
        latent = torch.zeros((*state_sequence.shape[:2], 1),
                             dtype=state.dtype, device=state.device)
        return state_sequence, acceleration_sequence, latent, gate_sequence


class AsymmetricTransitionToyModel:
    """EDSSM-shaped transition fixture, intentionally not reflection-equivariant."""

    history_state_size = 9
    include_roll_state = True
    include_raw_encoder_history = False
    include_wheel_innovation_history = False
    dynamics_state_size = 9

    def __init__(self):
        self.state_mean = torch.zeros(9)
        self.state_scale = torch.ones(9)
        self.command_mean = torch.zeros(2)
        self.command_scale = torch.ones(2)
        self.acceleration_bounds = torch.ones(5)
        self.latent_transition = torch.nn.GRUCell(16, 4)

    def encode_history(self, history):
        return torch.zeros((len(history), 4), dtype=history.dtype,
                           device=history.device)

    def transition(self, state, delayed, latent, command):
        steer, throttle = command.unbind(dim=-1)
        acceleration = torch.stack((
            0.1 + steer, 0.2 * steer, 0.3 * steer.square() + steer,
            0.4 * throttle + steer, 0.5 * throttle - steer,
        ), dim=-1)
        next_state = state.clone()
        next_state[..., 0] += 0.01 * (1.0 + 2.0 * steer + throttle)
        next_state[..., 1] += 0.02 * steer + 0.003 * state[..., 0]
        next_state[..., 2] += 0.04 * steer.square() + 0.01 * steer
        next_state[..., 3] = 0.3 * steer + 0.02 * delayed[..., 0]
        next_state[..., 4] = 0.5 * throttle + 0.01 * delayed[..., 1]
        next_state[..., 5] += 0.07 * throttle + 0.03 * steer
        next_state[..., 6] += 0.04 * throttle - 0.01 * steer
        next_state[..., 7] += 0.02 * steer + 0.001 * state[..., 8]
        next_state[..., 8] += 0.03 * steer + 0.002 * state[..., 7]
        latent_input = torch.cat((state, command, acceleration), dim=-1)
        next_latent = self.latent_transition(latent_input, latent)
        gates = torch.sigmoid(steer + throttle)[:, None]
        return next_state, command, next_latent, acceleration, gates

    def rollout(self, initial, delayed, history, commands):
        state = initial
        latent = self.encode_history(history)
        states, accelerations, latents, gates = [], [], [], []
        for command in commands.unbind(dim=1):
            state, _, latent, acceleration, gate = self.transition(
                state, delayed, latent, command)
            delayed = command
            states.append(state)
            accelerations.append(acceleration)
            latents.append(latent)
            gates.append(gate)
        return (torch.stack(states, dim=1),
                torch.stack(accelerations, dim=1),
                torch.stack(latents, dim=1), torch.stack(gates, dim=1))


class TurnReflectionProjectionTest(unittest.TestCase):
    def test_projected_rollout_is_reflection_equivariant(self):
        generator = torch.Generator().manual_seed(17)
        initial = torch.randn((3, 9), generator=generator)
        delayed = torch.randn((3, 2), generator=generator)
        history = torch.randn((3, 5, 11), generator=generator)
        commands = torch.randn((3, 7, 2), generator=generator)
        model = TurnReflectionProjectedModel(AsymmetricToyModel())

        predicted = model.rollout(initial, delayed, history, commands)
        mirrored = model.rollout(
            reflect_state_channels(initial), reflect_commands(delayed),
            reflect_history_features(history, 9), reflect_commands(commands))

        torch.testing.assert_close(
            mirrored[0], reflect_state_channels(predicted[0]),
            rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            mirrored[1], reflect_acceleration_targets(predicted[1]),
            rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(mirrored[3], predicted[3],
                                   rtol=1e-6, atol=1e-6)

    def test_stepwise_projection_is_exact_parent_below_threshold(self):
        torch.manual_seed(3)
        initial = torch.randn((2, 9)) * 0.05
        delayed = torch.zeros((2, 2))
        history = torch.randn((2, 80, 11))
        commands = torch.zeros((2, 6, 2))
        commands[..., 0] = torch.tensor([0.12, -0.24])[:, None]
        commands[..., 1] = 0.3
        parent = AsymmetricTransitionToyModel()
        expected = parent.rollout(initial, delayed, history, commands)
        actual = HighSteerStepwiseReflectionModel(parent).rollout(
            initial, delayed, history, commands)
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(actual_value, expected_value,
                                       rtol=0.0, atol=0.0)

    def test_stepwise_projection_is_equivariant_when_high_steer_activates(self):
        torch.manual_seed(11)
        initial = torch.randn((2, 9)) * 0.05
        delayed = torch.randn((2, 2)) * 0.1
        history = torch.randn((2, 80, 11))
        commands = torch.randn((2, 5, 2)) * 0.2
        commands[..., 0] = torch.tensor([0.42, -0.38])[:, None]
        projected_model = HighSteerStepwiseReflectionModel(
            AsymmetricTransitionToyModel())
        predicted = projected_model.rollout(
            initial, delayed, history, commands)
        mirrored = projected_model.rollout(
            reflect_state_channels(initial), reflect_commands(delayed),
            reflect_history_features(history, 9), reflect_commands(commands))
        torch.testing.assert_close(
            mirrored[0], reflect_state_channels(predicted[0]),
            rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            mirrored[1], reflect_acceleration_targets(predicted[1]),
            rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(mirrored[3], predicted[3],
                                   rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
