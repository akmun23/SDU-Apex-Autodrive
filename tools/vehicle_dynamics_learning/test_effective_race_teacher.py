"""Mathematical tests for the EDSSM body and actuator integration."""

from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    ActuatorChannel,
    ActuatorFit,
    effective_model_type,
    generalized_acceleration_targets,
)


class EffectiveRaceTeacherMathTest(unittest.TestCase):
    def test_paired_comparison_labels_left_minus_right_sign_unambiguously(self):
        from tools.vehicle_dynamics_learning.compare_effective_race_teacher import (
            _paired_run_difference, _practice_windows,
        )

        result = _paired_run_difference(
            {"per_run": {"r1": 2.0, "r2": 4.0}},
            {"per_run": {"r1": 1.0, "r2": 3.0}})
        self.assertEqual(result["macro_run_difference_left_minus_right"], 1.0)
        self.assertEqual(
            result["paired_run_difference_left_minus_right_95pct_ci"],
            [1.0, 1.0])
        self.assertEqual(result["left_minus_right_percent_of_right"], 50.0)

    def test_long_horizon_practice_windows_do_not_cross_sequence_boundaries(self):
        from tools.vehicle_dynamics_learning.compare_effective_race_teacher import (
            _practice_windows,
        )

        data = {
            "frames": np.zeros((12, 9), dtype=np.float32),
            "bounds": np.asarray([[0, 8], [8, 12]], dtype=np.int64),
            "seq_run": np.asarray([0, 0], dtype=np.int64),
            "splits": np.asarray(["unseen_practice"]),
            "run_ids": np.asarray(["practice_run"]),
        }
        benchmark = {"windows": [
            {"run_id": "practice_run", "global_start_index": 3,
             "categories": []},
            {"run_id": "practice_run", "global_start_index": 7,
             "categories": []},
            {"run_id": "practice_run", "global_start_index": 9,
             "categories": []},
        ]}
        rows = _practice_windows(benchmark, data, horizon_steps=2)
        self.assertEqual([row["start"] for row in rows], [3, 9])

    def test_ten_second_raw_wheel_score_uses_full_truth_and_validity_horizon(self):
        from tools.vehicle_dynamics_learning.compare_effective_race_teacher import (
            _summarize_pair,
        )

        state = np.zeros((500, 7), dtype=np.float64)
        data = {
            "frames": np.zeros((500, 9), dtype=np.float64),
            "simulator_pose_xyyaw": np.zeros((500, 3), dtype=np.float64),
            "encoder_raw_valid": np.ones(500, dtype=bool),
            "run_ids": np.asarray(["run"]),
        }
        states = np.zeros((1, 400, 7), dtype=np.float64)
        poses = np.zeros((1, 400, 3), dtype=np.float64)
        report = _summarize_pair(
            states, poses, states, poses, data, state,
            [{"start": 50, "run": 0, "sequence_id": 0, "regimes": []}],
            "synthetic ten-second regression",
            rssm_truth_state=state,
            wheel_state_source="raw_encoder")
        self.assertIn("10s", report["horizons"])
        self.assertEqual(
            report["horizons"]["10s"]["wheel_rmse_mps"]["edssm"][
                "macro_run_mean"], 0.0)
        practice_report = _summarize_pair(
            states[:, :200], poses[:, :200], states[:, :200], poses[:, :200],
            data, state,
            [{"start": 50, "run": 0, "sequence_id": 0, "regimes": []}],
            "synthetic one-lap practice regression",
            rssm_truth_state=state,
            wheel_state_source="raw_encoder")
        self.assertIn("5s", practice_report["horizons"])
        self.assertNotIn("10s", practice_report["horizons"])

    def test_state_distillation_targets_only_yaw_and_rear_wheels(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
            _state_distillation_loss,
        )

        student = torch.tensor(
            [[[50.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.5]]],
            requires_grad=True)
        teacher = torch.zeros_like(student)
        loss = _state_distillation_loss(
            torch, nn, student, teacher, torch.ones(7))
        self.assertAlmostEqual(float(loss.detach()), 0.125)
        loss.backward()
        self.assertEqual(float(student.grad[0, 0, 0]), 0.0)
        self.assertNotEqual(float(student.grad[0, 0, 2]), 0.0)
        self.assertNotEqual(float(student.grad[0, 0, 5]), 0.0)
        self.assertNotEqual(float(student.grad[0, 0, 6]), 0.0)
        yaw_only = _state_distillation_loss(
            torch, nn, student.detach(), teacher, torch.ones(7), channels=(2,))
        self.assertAlmostEqual(float(yaw_only), 0.125)

    def test_low_throttle_longitudinal_weight_is_conditioned_on_speed_and_command(self):
        import torch
        from torch import nn

        from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
            _weighted_body_state_loss,
            _weighted_longitudinal_loss,
        )

        errors = torch.tensor([[[1.0], [3.0]]])
        targets = torch.tensor([[[4.0, 0.0, 0.0],
                                 [2.0, 0.0, 0.0]]])
        commands = torch.tensor([[[0.0, 0.03], [0.0, 0.03]]])
        weighted = _weighted_longitudinal_loss(
            torch, nn, errors, targets, commands, 5.0)
        unweighted = _weighted_longitudinal_loss(
            torch, nn, errors, targets, commands, 1.0)
        self.assertAlmostEqual(float(weighted), 5.0 / 6.0, places=6)
        self.assertAlmostEqual(float(unweighted), 1.5, places=6)
        body_errors = torch.cat((errors, torch.zeros((1, 2, 4))), dim=-1)
        body_targets = torch.cat((targets, torch.zeros((1, 2, 2))), dim=-1)
        body_weighted = _weighted_body_state_loss(
            torch, nn, body_errors, body_targets, commands, 5.0)
        body_unweighted = _weighted_body_state_loss(
            torch, nn, body_errors, body_targets, commands, 1.0)
        self.assertAlmostEqual(float(body_weighted), 1.0 / 6.0, places=6)
        self.assertAlmostEqual(float(body_unweighted), 0.3, places=6)

    def test_raw_encoder_rate_uses_only_latest_past_sample_pair(self):
        from tools.vehicle_dynamics_learning.build_encoder_state_view import (
            _latest_source_rate,
            _source_rate,
        )

        series = (
            np.asarray([0, 25_000_000, 50_000_000], dtype=np.int64),
            np.asarray([0.0, 1.0, 3.0], dtype=np.float64),
        )
        pair = _latest_source_rate(series, 40_000_000)
        self.assertEqual(pair, ((0, 0.0), (25_000_000, 1.0)))
        self.assertAlmostEqual(_source_rate(*pair), 0.059 / 0.025)
        self.assertIsNone(_latest_source_rate(series, 10_000_000))

    def test_raw_encoder_rate_uses_fixed_40hz_period_despite_stamp_jitter(self):
        from tools.vehicle_dynamics_learning.build_encoder_state_view import (
            _source_rate,
        )

        # A 20 ms header interval is alignment jitter; the physical encoder
        # cadence is still 40 Hz, so speed must use the fixed 25 ms period.
        self.assertAlmostEqual(
            _source_rate((1_000_000_000, 0.0), (1_020_000_000, 1.0)),
            0.059 / 0.025)
        # Keep rejecting gaps that are not plausibly adjacent 40 Hz samples.
        self.assertIsNone(
            _source_rate((1_000_000_000, 0.0), (1_036_000_000, 1.0)))

    def test_raw_encoder_state_replaces_only_valid_wheel_samples(self):
        from tools.vehicle_dynamics_learning.effective_race_teacher import (
            physical_state_from_dataset,
        )

        rigid = np.zeros((3, 13), dtype=np.float64)
        rigid[:, 7] = [1.0, 2.0, 3.0]
        rigid[:, 8] = [0.1, 0.2, 0.3]
        rigid[:, 12] = [0.4, 0.5, 0.6]
        frames = np.zeros((3, 9), dtype=np.float64)
        frames[:, 3:5] = [[0.01, 0.2], [0.02, 0.3], [0.03, 0.4]]
        frames[:, 5:7] = [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
        data = {
            "simulator_rigid_state": rigid,
            "frames": frames,
            "encoder_raw_surface_mps": np.asarray(
                [[10.0, 10.1], [20.0, 20.1], [30.0, 30.1]]),
            "encoder_raw_valid": np.asarray([True, False, True]),
        }

        filtered = physical_state_from_dataset(data)
        raw = physical_state_from_dataset(data, "raw_encoder")
        np.testing.assert_allclose(filtered[:, 5:7], frames[:, 5:7])
        np.testing.assert_allclose(raw[:, 5:7], [[10.0, 10.1], [2.0, 2.1],
                                                 [30.0, 30.1]])
        np.testing.assert_allclose(raw[:, :5], filtered[:, :5])

    def test_raw_encoder_history_does_not_change_recursive_state_layout(self):
        import torch
        from torch import nn

        actuator = ActuatorFit(ActuatorChannel(1, 0.5),
                               ActuatorChannel(1, 0.8), {})
        model_type = effective_model_type(
            torch, nn, np.zeros(12, np.float32), np.ones(12, np.float32),
            np.zeros(7, np.float32), np.ones(7, np.float32),
            np.zeros(2, np.float32), np.ones(2, np.float32),
            np.ones(5, np.float32), actuator, encoder="gru", latent_size=32,
            include_raw_encoder_history=True)
        model = model_type().eval()
        history = torch.zeros((1, 80, 12))
        latent = model.encode_history(history)
        state = torch.tensor([[3.0, 0.1, 0.2, 0.0, 0.1, 3.0, 3.1]])
        command = torch.tensor([[0.0, 0.1]])
        following, _, _, _, _ = model.transition(
            state, command, latent, command)
        self.assertTrue(model.include_raw_encoder_history)
        self.assertEqual(tuple(following.shape), (1, 7))

    def test_wheel_innovation_history_is_raw_minus_filtered_and_split_safe(self):
        from tools.vehicle_dynamics_learning.effective_race_teacher import (
            wheel_innovation_history_features,
        )

        frames = np.zeros((4, 9), dtype=np.float64)
        frames[:, 5:7] = [[1.0, 2.0], [3.0, 4.0],
                          [5.0, 6.0], [7.0, 8.0]]
        data = {
            "frames": frames,
            "encoder_raw_surface_mps": np.asarray(
                [[4.0, 7.0], [9.0, 10.0], [20.0, 21.0], [30.0, 31.0]]),
            "encoder_raw_valid": np.asarray([True, False, True, True]),
            "bounds": np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]),
            "seq_run": np.arange(4),
            "splits": np.asarray(
                ["train", "validation", "test", "final_test"]),
        }
        features = wheel_innovation_history_features(data)
        np.testing.assert_allclose(features, [[3.0, 5.0, 1.0],
                                              [0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0]])

    def test_generalized_targets_invert_body_frame_equations(self):
        dt = DT_S
        current = np.asarray([3.2, -0.7, 1.4, 0.0, 0.2, 3.0, 3.1])
        ax, ay, yaw_accel = 1.8, -2.1, 0.6
        following = current.copy()
        following[0] += dt * (ax + current[2] * current[1])
        following[1] += dt * (ay - current[2] * current[0])
        following[2] += dt * yaw_accel
        following[5:7] += dt * np.asarray([0.9, -0.4])
        states = np.stack((current, following))
        labels = generalized_acceleration_targets(
            states, np.asarray([[0, 2]]), np.asarray([dt, dt]))
        np.testing.assert_allclose(
            labels[0], [ax, ay, yaw_accel, 0.9, -0.4],
            rtol=0.0, atol=2e-6)
        self.assertTrue(np.isnan(labels[1]).all())

    def test_contact_slip_targets_remove_rear_contact_kinematics(self):
        from tools.vehicle_dynamics_learning.effective_race_teacher import (
            REAR_TRACK_WIDTH_M,
        )

        dt = DT_S
        current = np.asarray([4.0, 0.2, 0.5, 0.0, 0.2, 0.0, 0.0])
        half_track = REAR_TRACK_WIDTH_M / 2.0
        current_contact = np.asarray((
            current[0] - half_track * current[2],
            current[0] + half_track * current[2]))
        current_slip = np.asarray((0.1, -0.2))
        current[5:7] = current_contact + current_slip
        following = current.copy()
        following[0] += 0.12
        following[2] += 0.08
        following_contact = np.asarray((
            following[0] - half_track * following[2],
            following[0] + half_track * following[2]))
        slip_accel = np.asarray((2.0, -1.0))
        following[5:7] = following_contact + current_slip + dt * slip_accel
        labels = generalized_acceleration_targets(
            np.stack((current, following)), np.asarray([[0, 2]]),
            np.asarray([dt, dt]), "contact_slip")
        np.testing.assert_allclose(labels[0, 3:5], slip_accel,
                                   rtol=0.0, atol=2e-6)

    def test_contact_slip_transition_preserves_slip_while_body_moves(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.effective_race_teacher import (
            REAR_TRACK_WIDTH_M,
        )

        actuator = ActuatorFit(ActuatorChannel(0, 0.5),
                               ActuatorChannel(0, 0.5), {})
        model_type = effective_model_type(
            torch, nn, np.zeros(9, np.float32), np.ones(9, np.float32),
            np.zeros(7, np.float32), np.ones(7, np.float32),
            np.zeros(2, np.float32), np.ones(2, np.float32),
            np.ones(5, np.float32), actuator, encoder="gru", latent_size=16,
            wheel_dynamics_mode="contact_slip")
        model = model_type().eval()
        state = torch.tensor([[4.0, 0.5, 1.2, 0.1, 0.3, 4.1, 4.2]])
        command = torch.zeros((1, 2))
        next_state, _, _, _, _ = model.transition(
            state, command, torch.zeros((1, 16)), command)
        track_half = REAR_TRACK_WIDTH_M / 2.0
        current_contact = torch.tensor([[4.0 - track_half * 1.2,
                                         4.0 + track_half * 1.2]])
        next_contact = torch.stack((
            next_state[:, 0] - track_half * next_state[:, 2],
            next_state[:, 0] + track_half * next_state[:, 2]), dim=-1)
        expected_slip = state[:, 5:7] - current_contact
        torch.testing.assert_close(next_state[:, 5:7],
                                   next_contact + expected_slip,
                                   rtol=0.0, atol=1e-6)

    def test_zero_acceleration_nominal_keeps_rigid_coupling_and_actuators(self):
        import torch
        from torch import nn

        mean_history = np.zeros(9, dtype=np.float32)
        scale_history = np.ones(9, dtype=np.float32)
        mean_state = np.zeros(7, dtype=np.float32)
        scale_state = np.ones(7, dtype=np.float32)
        mean_command = np.zeros(2, dtype=np.float32)
        scale_command = np.ones(2, dtype=np.float32)
        acceleration_bounds = np.ones(5, dtype=np.float32)
        actuator = ActuatorFit(
            ActuatorChannel(delay_steps=1, alpha=0.5),
            ActuatorChannel(delay_steps=1, alpha=0.8), {})
        model_type = effective_model_type(
            torch, nn, mean_history, scale_history, mean_state, scale_state,
            mean_command, scale_command, acceleration_bounds, actuator,
            encoder="gru", latent_size=32)
        model = model_type().eval()
        state = torch.tensor([[4.0, 0.5, 1.2, 0.1, 0.3, 4.1, 4.2]])
        previous_command = torch.tensor([[0.0, 0.2]])
        command = torch.tensor([[0.4, 0.8]])
        latent = torch.zeros((1, 32))
        next_state, _, _, acceleration, _ = model.transition(
            state, previous_command, latent, command)
        expected = state.clone()
        expected[0, 0] += DT_S * state[0, 2] * state[0, 1]
        expected[0, 1] -= DT_S * state[0, 2] * state[0, 0]
        expected[0, 3] += 0.5 * (previous_command[0, 0] - state[0, 3])
        expected[0, 4] += 0.8 * (previous_command[0, 1] - state[0, 4])
        torch.testing.assert_close(next_state, expected, rtol=0.0, atol=1e-7)
        torch.testing.assert_close(acceleration, torch.zeros_like(acceleration))

    def test_soft_expert_router_is_causal_and_normalized(self):
        import torch
        from torch import nn

        zeros = np.zeros(9, dtype=np.float32)
        ones = np.ones(9, dtype=np.float32)
        actuator = ActuatorFit(
            ActuatorChannel(1, 0.4), ActuatorChannel(1, 0.9), {})
        model_type = effective_model_type(
            torch, nn, zeros, ones, np.zeros(7, np.float32),
            np.ones(7, np.float32), np.zeros(2, np.float32),
            np.ones(2, np.float32), np.ones(5, np.float32), actuator,
            encoder="gru", latent_size=32, expert_count=2)
        model = model_type().eval()
        state = torch.tensor([[4.0, 0.3, 0.8, 0.2, 0.4, 4.1, 4.2]])
        command = torch.tensor([[0.25, 0.5]])
        latent = torch.zeros((1, 32))
        _, _, _, _, gates = model.transition(state, command, latent, command)
        self.assertEqual(tuple(gates.shape), (1, 2))
        torch.testing.assert_close(gates.sum(dim=-1), torch.ones(1))
        self.assertTrue(torch.all(gates > 0.0).item())
        self.assertTrue(torch.all(gates < 1.0).item())

    def test_wheel_acceleration_weight_only_disables_noisy_rate_loss(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import _loss

        class FixedTransition:
            def rollout(self, initial, delayed, history, commands):
                states = torch.zeros((1, 1, 7), dtype=torch.float32)
                acceleration = torch.zeros((1, 1, 5), dtype=torch.float32)
                latent = torch.zeros((1, 1, 1), dtype=torch.float32)
                gates = torch.ones((1, 1, 1), dtype=torch.float32)
                return states, acceleration, latent, gates

        batch = (
            np.zeros((1, 80, 9), dtype=np.float32),
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1, 7), dtype=np.float32),
            np.asarray([[[0.0, 0.0, 0.0, 1.0, 1.0]]], dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
        )
        normalizers = {
            "state_scale": np.ones(7, dtype=np.float32),
            "acceleration_bounds": np.ones(5, dtype=np.float32),
        }
        _, original_components = _loss(
            torch, nn, FixedTransition(), batch, normalizers, "cpu", 1.0)
        _, masked_components = _loss(
            torch, nn, FixedTransition(), batch, normalizers, "cpu", 0.0)
        self.assertAlmostEqual(float(original_components["effective_acceleration"]), 0.2)
        self.assertAlmostEqual(float(masked_components["effective_acceleration"]), 0.0)
        self.assertAlmostEqual(float(masked_components["rear_wheel_state"]), 0.0)

    def test_longitudinal_head_only_loss_ignores_wheel_and_yaw_targets(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import _loss

        class FixedTransition:
            def rollout(self, initial, delayed, history, commands):
                states = torch.zeros((1, 1, 7), dtype=torch.float32)
                acceleration = torch.zeros((1, 1, 5), dtype=torch.float32)
                latent = torch.zeros((1, 1, 1), dtype=torch.float32)
                gates = torch.ones((1, 1, 1), dtype=torch.float32)
                return states, acceleration, latent, gates

        batch = (
            np.zeros((1, 80, 9), dtype=np.float32),
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.asarray([[[1.0, 0.0, 20.0, 0.0, 0.0, 50.0, -50.0]]],
                       dtype=np.float32),
            np.zeros((1, 1, 5), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
        )
        normalizers = {
            "state_scale": np.ones(7, dtype=np.float32),
            "acceleration_bounds": np.ones(5, dtype=np.float32),
        }
        total, _ = _loss(
            torch, nn, FixedTransition(), batch, normalizers, "cpu",
            output_head_only="longitudinal")
        self.assertAlmostEqual(float(total), 0.5)

    def test_longitudinal_head_distillation_preserves_other_motion_states(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
            _loss, _state_distillation_channels,
        )

        class FixedTransition:
            def rollout(self, initial, delayed, history, commands):
                return (
                    torch.zeros((1, 1, 7), dtype=torch.float32),
                    torch.zeros((1, 1, 5), dtype=torch.float32),
                    torch.zeros((1, 1, 1), dtype=torch.float32),
                    torch.ones((1, 1, 1), dtype=torch.float32),
                )

        batch = (
            np.zeros((1, 80, 9), dtype=np.float32),
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.asarray([[[1.0, 0.0, 99.0, 0.0, 0.0, 99.0, 99.0]]],
                       dtype=np.float32),
            np.zeros((1, 1, 5), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
        )
        normalizers = {
            "state_scale": np.ones(7, dtype=np.float32),
            "acceleration_bounds": np.ones(5, dtype=np.float32),
        }
        teacher_states = torch.tensor(
            [[[0.0, 2.0, 2.0, 0.0, 0.0, 4.0, 6.0]]],
            dtype=torch.float32)
        channels = _state_distillation_channels("longitudinal")
        self.assertEqual(channels, (1, 2, 5, 6))
        total, components = _loss(
            torch, nn, FixedTransition(), batch, normalizers, "cpu",
            output_head_only="longitudinal",
            teacher_state=teacher_states,
            state_distillation_weight=0.25,
            state_distillation_channels=channels)
        # Forward-speed loss remains 0.5. The parent-preservation term is the
        # mean Smooth-L1 of errors 2, 2, 4, and 6: 3.0 * 0.25.
        self.assertAlmostEqual(float(components["state_distillation"]), 3.0)
        self.assertAlmostEqual(float(total), 1.25)

    def test_wheel_head_distillation_preserves_parent_body_rollout(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
            _loss, _state_distillation_channels,
        )

        class FixedTransition:
            def rollout(self, initial, delayed, history, commands):
                return (
                    torch.zeros((1, 1, 7), dtype=torch.float32),
                    torch.zeros((1, 1, 5), dtype=torch.float32),
                    torch.zeros((1, 1, 1), dtype=torch.float32),
                    torch.ones((1, 1, 1), dtype=torch.float32),
                )

        batch = (
            np.zeros((1, 80, 9), dtype=np.float32),
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.asarray([[[50.0, 50.0, 50.0, 0.0, 0.0, 2.0, 2.0]]],
                       dtype=np.float32),
            np.zeros((1, 1, 5), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
        )
        normalizers = {
            "state_scale": np.ones(7, dtype=np.float32),
            "acceleration_bounds": np.ones(5, dtype=np.float32),
        }
        channels = _state_distillation_channels("wheel")
        self.assertEqual(channels, (0, 1, 2))
        parent_states = torch.tensor(
            [[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32)
        total, components = _loss(
            torch, nn, FixedTransition(), batch, normalizers, "cpu",
            wheel_state_weight=0.5,
            output_head_only="wheel",
            teacher_state=parent_states,
            state_distillation_weight=2.0,
            state_distillation_channels=channels)
        self.assertAlmostEqual(float(components["state_distillation"]), 1.5)
        self.assertAlmostEqual(float(total), 3.75)

    def test_longitudinal_wheel_head_loss_fits_speed_and_wheels_only(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import _loss

        class FixedTransition:
            def rollout(self, initial, delayed, history, commands):
                return (
                    torch.zeros((1, 1, 9), dtype=torch.float32),
                    torch.zeros((1, 1, 5), dtype=torch.float32),
                    torch.zeros((1, 1, 1), dtype=torch.float32),
                    torch.ones((1, 1, 1), dtype=torch.float32),
                )

        def make_batch(lateral_speed, yaw_rate):
            targets = np.asarray([[[1.0, lateral_speed, yaw_rate,
                                    0.0, 0.0, 2.0, 2.0, 0.0, 0.0]]],
                                 dtype=np.float32)
            return (
                np.zeros((1, 80, 9), dtype=np.float32),
                np.zeros((1, 9), dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 1, 2), dtype=np.float32),
                targets,
                np.zeros((1, 1, 5), dtype=np.float32),
                np.zeros((1, 2, 3), dtype=np.float32),
            )

        normalizers = {
            "state_scale": np.ones(9, dtype=np.float32),
            "acceleration_bounds": np.ones(5, dtype=np.float32),
        }
        model = FixedTransition()
        base, _ = _loss(torch, nn, model, make_batch(0.0, 0.0),
                        normalizers, "cpu",
                        wheel_state_weight=0.5,
                        output_head_only="longitudinal_wheel")
        changed_non_targets, _ = _loss(
            torch, nn, model, make_batch(30.0, -40.0), normalizers, "cpu",
            wheel_state_weight=0.5,
            output_head_only="longitudinal_wheel")
        self.assertAlmostEqual(float(base), 1.25)
        torch.testing.assert_close(base, changed_non_targets)

    def test_terminal_heading_penalty_is_additive_and_requires_longitudinal_wheel_mode(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import _loss

        class FixedTransition:
            def rollout(self, initial, delayed, history, commands):
                return (
                    torch.zeros((1, 1, 9), dtype=torch.float32),
                    torch.zeros((1, 1, 5), dtype=torch.float32),
                    torch.zeros((1, 1, 1), dtype=torch.float32),
                    torch.ones((1, 1, 1), dtype=torch.float32),
                )

        targets = np.zeros((1, 1, 9), dtype=np.float32)
        batch = (
            np.zeros((1, 80, 9), dtype=np.float32),
            np.zeros((1, 9), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            targets,
            np.zeros((1, 1, 5), dtype=np.float32),
            np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]],
                       dtype=np.float32),
        )
        normalizers = {
            "state_scale": np.ones(9, dtype=np.float32),
            "acceleration_bounds": np.ones(5, dtype=np.float32),
        }
        model = FixedTransition()
        base, components = _loss(
            torch, nn, model, batch, normalizers, "cpu",
            output_head_only="longitudinal_wheel")
        weighted, weighted_components = _loss(
            torch, nn, model, batch, normalizers, "cpu",
            output_head_only="longitudinal_wheel",
            terminal_heading_loss_weight=5.0)
        expected = 5.0 * float(weighted_components["terminal_heading"])
        self.assertGreater(float(components["terminal_heading"]), 0.0)
        self.assertAlmostEqual(float(weighted - base), expected, places=6)
        pose_upweighted, pose_components = _loss(
            torch, nn, model, batch, normalizers, "cpu",
            output_head_only="longitudinal_wheel",
            pose_secondary_weight=0.3)
        self.assertAlmostEqual(
            float(pose_upweighted - base),
            0.2 * float(pose_components["pose_secondary"]), places=6)
        with self.assertRaises(ValueError):
            _loss(torch, nn, model, batch, normalizers, "cpu",
                  output_head_only="longitudinal",
                  terminal_heading_loss_weight=5.0)

    def test_signed_mismatch_sampler_separates_opposite_slip_regimes(self):
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
            _build_window_sampler,
        )

        row_count = 118
        frames = np.zeros((row_count, 9), dtype=np.float64)
        frames[:, 0] = 5.0
        frames[:, 5:7] = 5.0
        # The only eligible 5 s starts are 79, 86, and 87. Keep equal
        # mismatch magnitude but opposite signs so only sign stratification
        # can distinguish driven wheelspin from wheels slower than the body.
        frames[79, 5:7] = 3.0
        frames[86:88, 5:7] = 7.0
        data = {
            "frames": frames,
            "simulator_pose_xyyaw": np.zeros((row_count, 3)),
            "bounds": np.asarray([[0, row_count]], dtype=np.int64),
            "seq_run": np.asarray([0], dtype=np.int64),
            "splits": np.asarray(["train"]),
            "sequence_condition_id": np.asarray([0], dtype=np.int64),
            "run_ids": np.asarray(["synthetic_run"]),
            "training_families": np.asarray(["synthetic_family"]),
            "training_family_names": np.asarray(["synthetic_family"]),
            "training_family_probabilities": np.asarray([1.0]),
        }
        state = np.zeros((row_count, 7), dtype=np.float64)
        state[:, 0] = 5.0
        state[:, 5:7] = 5.0
        state[79, 5:7] = 3.0
        state[86:88, 5:7] = 7.0
        _, magnitude_only = _build_window_sampler(
            data, state, 30, "train", stride_steps=7)
        _, signed = _build_window_sampler(
            data, state, 30, "train", stride_steps=7,
            stratify_signed_wheel_mismatch=True)
        family = "synthetic_family"
        self.assertEqual(
            magnitude_only["family_condition_sampler"]
            ["condition_count_by_family"][family], 1)
        self.assertEqual(
            signed["family_condition_sampler"]
            ["condition_count_by_family"][family], 2)

    def test_sampler_throttle_domain_is_explicit_and_inclusive(self):
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
            _build_window_sampler,
        )

        row_count = 118
        frames = np.zeros((row_count, 9), dtype=np.float64)
        frames[:, 0] = 4.0
        frames[:, 5:7] = 4.0
        frames[:, 8] = 0.75
        data = {
            "frames": frames,
            "simulator_pose_xyyaw": np.zeros((row_count, 3)),
            "bounds": np.asarray([[0, row_count]], dtype=np.int64),
            "seq_run": np.asarray([0], dtype=np.int64),
            "splits": np.asarray(["train"]),
            "sequence_condition_id": np.asarray([0], dtype=np.int64),
            "run_ids": np.asarray(["high_throttle_run"]),
            "training_families": np.asarray(["synthetic_family"]),
            "training_family_names": np.asarray(["synthetic_family"]),
            "training_family_probabilities": np.asarray([1.0]),
        }
        state = np.zeros((row_count, 7), dtype=np.float64)
        state[:, 0] = 4.0
        state[:, 5:7] = 4.0
        with self.assertRaises(ValueError):
            _build_window_sampler(data, state, 30, "train", stride_steps=7)
        _, full_throttle = _build_window_sampler(
            data, state, 30, "train", stride_steps=7,
            max_throttle_command=1.0)
        self.assertGreater(full_throttle["eligible_windows"], 0)
        self.assertEqual(full_throttle["command_domain"],
                         "future throttle command <=1")
        with self.assertRaises(ValueError):
            _build_window_sampler(
                data, state, 30, "train", max_throttle_command=1.01)

    def test_checkpoint_selection_includes_recursive_wheel_error(self):
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
            _selection_score,
        )

        def report(wheel_5s):
            horizons = {
                name: {
                    "body": {"macro_run_mean": 0.1},
                    "wheel": {"macro_run_mean": 0.1},
                    "position_m": {"macro_run_mean": 0.1},
                    "heading_rad": {"macro_run_mean": 0.1},
                }
                for name in ("0.75s", "2s", "5s")
            }
            horizons["5s"]["wheel"]["macro_run_mean"] = wheel_5s
            return {"horizons": horizons, "hard_regimes": {}}

        lower, _ = _selection_score(report(0.1))
        higher, _ = _selection_score(report(0.5))
        self.assertGreater(higher, lower)

    def test_body_head_objective_retains_recursive_wheel_and_pose_losses(self):
        import torch
        from torch import nn
        from tools.vehicle_dynamics_learning.train_effective_race_teacher import _loss

        class FixedTransition:
            def rollout(self, initial, delayed, history, commands):
                return (
                    torch.zeros((1, 1, 7), dtype=torch.float32),
                    torch.zeros((1, 1, 5), dtype=torch.float32),
                    torch.zeros((1, 1, 1), dtype=torch.float32),
                    torch.ones((1, 1, 1), dtype=torch.float32),
                )

        batch = (
            np.zeros((1, 80, 9), dtype=np.float32),
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0]]],
                       dtype=np.float32),
            np.full((1, 1, 5), 20.0, dtype=np.float32),
            np.asarray([[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]],
                       dtype=np.float32),
        )
        normalizers = {
            "state_scale": np.ones(7, dtype=np.float32),
            "acceleration_bounds": np.ones(5, dtype=np.float32),
        }
        total, _ = _loss(
            torch, nn, FixedTransition(), batch, normalizers, "cpu",
            wheel_state_weight=0.5, output_head_only="body")
        # Wheel state Smooth-L1 is 1.5, mean position Smooth-L1 is .75;
        # the body-head objective keeps both as recursive side constraints.
        self.assertAlmostEqual(float(total), 0.825, places=6)

    def test_roll_state_is_integrated_from_predicted_rate(self):
        import torch
        from torch import nn

        actuator = ActuatorFit(
            ActuatorChannel(delay_steps=1, alpha=0.5),
            ActuatorChannel(delay_steps=1, alpha=0.8), {})
        model_type = effective_model_type(
            torch, nn, np.zeros(11, np.float32), np.ones(11, np.float32),
            np.zeros(9, np.float32), np.ones(9, np.float32),
            np.zeros(2, np.float32), np.ones(2, np.float32),
            np.ones(5, np.float32), actuator, encoder="gru",
            latent_size=16, include_roll_state=True)
        model = model_type().eval()
        state = torch.tensor([[4.0, 0.5, 1.2, 0.1, 0.3,
                               4.1, 4.2, 0.08, 0.4]])
        command = torch.zeros((1, 2))
        next_state, _, _, _, _ = model.transition(
            state, command, torch.zeros((1, 16)), command)
        torch.testing.assert_close(next_state[:, 7], torch.tensor([0.09]),
                                   rtol=0.0, atol=1e-7)
        torch.testing.assert_close(next_state[:, 8], state[:, 8],
                                   rtol=0.0, atol=1e-7)

    def test_coupled_roll_rate_uses_predicted_lateral_acceleration(self):
        import torch
        from torch import nn

        actuator = ActuatorFit(ActuatorChannel(0, 0.5),
                               ActuatorChannel(0, 0.5), {})
        model_type = effective_model_type(
            torch, nn, np.zeros(11, np.float32), np.ones(11, np.float32),
            np.zeros(9, np.float32), np.ones(9, np.float32),
            np.zeros(2, np.float32), np.ones(2, np.float32),
            np.ones(5, np.float32), actuator, encoder="gru", latent_size=16,
            include_roll_state=True, couple_roll_acceleration=True)
        model = model_type().eval()
        self.assertEqual(model.roll_rate_transition[0].in_features, 28)
        with torch.no_grad():
            model.roll_rate_transition[0].weight.zero_()
            model.roll_rate_transition[0].bias.zero_()
            model.roll_rate_transition[0].weight[0, -1] = 1.0
            model.roll_rate_transition[2].weight.zero_()
            model.roll_rate_transition[2].bias.zero_()
            model.roll_rate_transition[2].weight[0, 0] = 1.0
            state = torch.zeros((1, 9))
            command = torch.zeros((1, 2))
            latent = torch.zeros((1, 16))
            model.acceleration_heads[1].bias.fill_(1.0)
            positive, *_ = model.transition(state, command, latent, command)
            model.acceleration_heads[1].bias.fill_(-1.0)
            negative, *_ = model.transition(state, command, latent, command)
        self.assertGreater(float(positive[0, 8]), float(negative[0, 8]))

    def test_fitted_roll_oscillator_uses_predicted_lateral_acceleration(self):
        import torch
        from torch import nn

        actuator = ActuatorFit(ActuatorChannel(1, 0.5),
                               ActuatorChannel(1, 0.5), {})
        model_type = effective_model_type(
            torch, nn, np.zeros(9, np.float32), np.ones(9, np.float32),
            np.zeros(9, np.float32), np.ones(9, np.float32),
            np.zeros(2, np.float32), np.ones(2, np.float32),
            np.ones(5, np.float32), actuator, encoder="gru", latent_size=16,
            include_roll_state=True, couple_roll_acceleration=True,
            roll_residual_mode=True,
            roll_oscillator_coefficients=np.asarray((1.0, -2.0, -3.0)))
        model = model_type().eval()
        with torch.no_grad():
            model.acceleration_heads[1].bias.fill_(1.0)
            state = torch.tensor([[4.0, 0.5, 1.2, 0.1, 0.3,
                                   4.1, 4.2, 0.08, 0.4]])
            command = torch.zeros((1, 2))
            next_state, _, _, acceleration, _ = model.transition(
                state, command, torch.zeros((1, 16)), command)
        expected_rate = 0.4 + DT_S * (
            float(acceleration[0, 1]) - 2.0 * 0.08 - 3.0 * 0.4)
        self.assertAlmostEqual(float(next_state[0, 8]), expected_rate, places=6)
        expected_roll = 0.08 + 0.5 * DT_S * (0.4 + expected_rate)
        self.assertAlmostEqual(float(next_state[0, 7]), expected_roll, places=6)


if __name__ == "__main__":
    unittest.main()
