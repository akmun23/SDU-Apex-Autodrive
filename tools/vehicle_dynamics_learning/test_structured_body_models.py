#!/usr/bin/env python3
"""Focused mathematical and architecture checks for the offline body model."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from run_structured_body_cv import (
    BODY_TEACHER_FORCE_CHANNELS,
    _aggregate_cv,
    _coverage_profile,
    _grouped_folds,
    _train_loss,
)
from structured_body_models import (
    REAR_AXLE_TO_COM_X_M,
    acceleration_statistics,
    generalized_accelerations,
    integrate_body_state,
    make_structured_model,
    rollout_structured_body,
)
from train_nssm import (
    _integrate_rear_axle_pose,
    _initialize_model,
    _model_type,
    _rollout,
    _torch,
)
from tools.evaluate_open_plane_body_dynamics import _roll_pitch_from_quaternion
from tools.evaluate_open_plane_body_dynamics import STREAM_TOPICS
from tools.vehicle_dynamics_learning.prepare_dataset import _quality


class StructuredBodyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.nn = _torch()

    def test_body_frame_transport_terms_and_acceleration_recovery(self):
        torch = self.torch
        body = torch.tensor([[2.0, 0.0, 1.0]])
        zero_acceleration = torch.zeros((1, 3))
        dt = torch.tensor([0.01])
        next_body = integrate_body_state(torch, body, zero_acceleration, dt)
        np.testing.assert_allclose(
            next_body.numpy(),
            [[2.0 + 0.01 * REAR_AXLE_TO_COM_X_M, -0.02, 1.0]],
            rtol=0.0, atol=1e-7)

        accel = np.asarray([[0.5, 1.0, 2.0]], dtype=np.float32)
        current = np.zeros((1, 9), dtype=np.float32)
        following = np.zeros_like(current)
        current[0, :3] = [2.0, 0.0, 1.0]
        next_state = integrate_body_state(
            torch, torch.as_tensor(current[:, :3]), torch.as_tensor(accel), dt)
        following[0, :3] = next_state.numpy()[0]
        recovered = generalized_accelerations(current, following,
                                              np.asarray([0.01]))
        np.testing.assert_allclose(recovered, accel, rtol=0.0, atol=2e-5)

    def test_structured_gru_integrates_rear_axle_com_offset(self):
        torch = self.torch
        mean = np.zeros(9, dtype=np.float32)
        scale = np.ones(9, dtype=np.float32)
        model_type = _model_type(
            torch, self.nn, 16, "structured_gru", 1, 16, 9,
            mean, scale, np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32), "euler", REAR_AXLE_TO_COM_X_M)
        model = model_type()
        for parameter in model.rate.parameters():
            torch.nn.init.zeros_(parameter)
        feature = torch.zeros((1, 9))
        feature[0, :3] = torch.tensor([2.0, 0.0, 1.0])
        next_state, _ = model.advance(
            feature, torch.zeros((1, 16)), torch.tensor([0.01]))
        np.testing.assert_allclose(
            next_state.detach().numpy()[0, :3],
            [2.0 + 0.01 * REAR_AXLE_TO_COM_X_M, -0.02, 1.0],
            rtol=0.0, atol=1e-7)

    def test_gru_lift_to_mixture_preserves_initial_transition(self):
        torch, nn = self.torch, self.nn
        torch.manual_seed(211)
        mean = np.linspace(-0.5, 0.5, 9, dtype=np.float32)
        scale = np.linspace(0.5, 1.5, 9, dtype=np.float32)
        gru_type = _model_type(torch, nn, 16, "gru", 1, 16, 9)
        source = gru_type().eval()
        payload = {
            "state_dict": source.state_dict(),
            "metadata": {"architecture": "gru"},
        }
        mixture_type = _model_type(
            torch, nn, 16, "mixture", 4, 16, 9, mean, scale)
        mixture = mixture_type().eval()
        mode = _initialize_model(mixture, payload, "mixture")
        self.assertEqual(mode, "gru_lifted_to_4_identical_mixture_experts")

        feature = torch.randn((7, 9))
        hidden = torch.randn((7, 16))
        dt = torch.full((7,), 0.025)
        with torch.no_grad():
            expected, _ = source.advance(feature, hidden, dt)
            actual, _ = mixture.advance(feature, hidden, dt)
        torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-6)

    def test_planar_pose_loss_integrates_rear_axle_twist_midpoint(self):
        torch = self.torch
        history = torch.zeros((1, 2, 9))
        history[0, -1, 0] = 2.0
        history[0, -1, 2] = 1.0
        predicted = torch.zeros((1, 2, 7), requires_grad=True)
        with torch.no_grad():
            predicted[:, :, 0] = 2.0
            predicted[:, :, 2] = 1.0
        pose = torch.zeros((1, 3, 3))
        pose[0, 0] = torch.tensor([1.0, 2.0, 0.4])
        dts = torch.full((1, 2), 0.1)
        integrated = _integrate_rear_axle_pose(
            torch, predicted, history, dts, pose,
            np.zeros(9, dtype=np.float32),
            np.ones(9, dtype=np.float32))
        expected = np.asarray([[
            [1.0 + 0.2 * np.cos(0.45), 2.0 + 0.2 * np.sin(0.45)],
            [1.0 + 0.2 * (np.cos(0.45) + np.cos(0.55)),
             2.0 + 0.2 * (np.sin(0.45) + np.sin(0.55))],
        ]], dtype=np.float32)
        np.testing.assert_allclose(integrated.detach().numpy(), expected,
                                   rtol=0.0, atol=1e-6)
        integrated.square().sum().backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_imu_quaternion_roll_pitch_conversion(self):
        roll = 0.31
        pitch = -0.17
        converted_roll = _roll_pitch_from_quaternion(
            np.sin(roll / 2.0), 0.0, 0.0, np.cos(roll / 2.0))
        converted_pitch = _roll_pitch_from_quaternion(
            0.0, np.sin(pitch / 2.0), 0.0, np.cos(pitch / 2.0))
        self.assertAlmostEqual(converted_roll[0], roll, places=6)
        self.assertAlmostEqual(converted_roll[1], 0.0, places=6)
        self.assertAlmostEqual(converted_pitch[0], 0.0, places=6)
        self.assertAlmostEqual(converted_pitch[1], pitch, places=6)
        self.assertIsNone(_roll_pitch_from_quaternion(0.0, 0.0, 0.0, 0.0))

    def test_aborted_capture_is_never_exported_as_clean_training_data(self):
        streams = {topic: (39.9, 25.0, 47.0) for topic in STREAM_TOPICS}
        capture = SimpleNamespace(
            aborted=True,
            collision_count_start=0,
            collision_count_end=0,
            timing_faults=0,
            phase_stream_stats=streams,
            stream_stats={},
            sequences=((object(),),),
        )
        clean, failures = _quality(capture)
        self.assertFalse(clean)
        self.assertIn("experiment_aborted", failures)

    def test_acceleration_scaling_and_rollout_gradients_all_latent_sizes(self):
        torch = self.torch
        feature_mean = np.zeros(9, dtype=np.float32)
        feature_scale = np.ones(9, dtype=np.float32)
        accel_stats = acceleration_statistics(np.asarray(
            [[-1.0, 0.0, -2.0], [1.0, 0.0, 2.0]], dtype=np.float32))
        self.assertEqual(accel_stats.sample_count, 2)
        self.assertTrue(np.all(accel_stats.scale >= [0.5, 0.5, 1.0]))
        history = torch.randn((3, 16, 9), requires_grad=False)
        future = torch.randn((3, 32, 9), requires_grad=False)
        dt = torch.full((3, 32), 0.025)
        architectures = (("accel_mlp", 0), ("latent", 2), ("latent", 4),
                         ("latent", 8), ("linear_residual", 0))
        for architecture, latent_dim in architectures:
            coefficients = (np.zeros((11, 3), dtype=np.float32)
                            if architecture == "linear_residual" else None)
            model = make_structured_model(
                torch, self.nn, architecture, latent_dim, 9, 16,
                feature_mean, feature_scale, accel_stats.mean,
                accel_stats.scale, coefficients)
            prediction = rollout_structured_body(model, history, future, dt)
            self.assertEqual(tuple(prediction.shape), (3, 32, 3))
            loss = prediction.square().mean()
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters()
                         if parameter.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all()
                                for g in gradients), architecture)

    def test_grouped_folds_keep_duplicates_together_and_balance_families(self):
        run_ids = np.asarray([
            "openplane_rootless_30_full_surface_20260927_01",
            "openplane_rootless_40_full_surface_20260927_01",
            "openplane_full_input_excitation_20260927_train1",
            "openplane_full_input_excitation_20260927_train2",
            "openplane_transition_4mps_20260926",
            "openplane_combined_slip_5mps_20260926",
        ])
        frames = np.zeros((len(run_ids) * 64, 9), dtype=np.float32)
        frames[:, 0] = np.tile(np.linspace(0.0, 8.0, 64), len(run_ids))
        frames[:, 3] = np.tile(np.linspace(-0.5, 0.5, 64), len(run_ids))
        bounds = np.asarray([[i * 64, (i + 1) * 64]
                             for i in range(len(run_ids))], dtype=np.int64)
        data = {
            "frames": frames,
            "bounds": bounds,
            "seq_run": np.arange(len(run_ids), dtype=np.int32),
            "run_ids": run_ids,
            "splits": np.asarray(["train"] * len(run_ids)),
        }
        manifest = {str(run_id): {"fingerprint": f"fingerprint-{index}"}
                    for index, run_id in enumerate(run_ids)}
        manifest[str(run_ids[1])]["fingerprint"] = manifest[
            str(run_ids[0])]["fingerprint"]
        run_fold, _, families = _grouped_folds(
            data, list(range(len(run_ids))), manifest, 4, 19)
        self.assertEqual(run_fold[0], run_fold[1])
        self.assertEqual(len(set(run_fold.values())), 4)
        self.assertEqual(families[str(run_ids[0])],
                         families[str(run_ids[1])])

    def test_gru_control_uses_measured_exogenous_channels_and_body_only_loss(self):
        torch, nn = self.torch, self.nn
        torch.manual_seed(31)
        model = _model_type(torch, nn, 16, "gru", 1, 16, 9)()
        history = torch.randn((2, 16, 9))
        future = torch.randn((2, 1, 9))
        dt = torch.full((2, 1), 0.025)

        predicted = _rollout(
            model, history, future, dt, 16,
            teacher_force_channels=BODY_TEACHER_FORCE_CHANNELS)
        torch.testing.assert_close(
            predicted[:, 0, BODY_TEACHER_FORCE_CHANNELS],
            future[:, 0, BODY_TEACHER_FORCE_CHANNELS])

        changed_exogenous = future.clone()
        changed_exogenous[:, :, BODY_TEACHER_FORCE_CHANNELS] += 100.0
        loss_original = _train_loss(torch, nn, model, "gru", history,
                                    future, dt, 16)
        loss_changed = _train_loss(torch, nn, model, "gru", history,
                                   changed_exogenous, dt, 16)
        torch.testing.assert_close(loss_original, loss_changed)

    def test_attitude_conditioning_never_reads_future_imu_attitude(self):
        torch = self.torch
        torch.manual_seed(7)
        mean = np.zeros(13, dtype=np.float32)
        scale = np.ones(13, dtype=np.float32)
        acceleration = acceleration_statistics(np.asarray(
            [[-1.0, -1.0, -2.0], [1.0, 1.0, 2.0]], dtype=np.float32))
        history = torch.randn((2, 16, 13))
        future_a = torch.randn((2, 32, 13))
        future_b = future_a.clone()
        future_b[:, :, 9:13] += 100.0
        dts = torch.full((2, 32), 0.025)
        model = make_structured_model(
            torch, self.nn, "latent", 4, 13, 16, mean, scale,
            acceleration.mean, acceleration.scale)
        held = (9, 10, 11, 12)
        prediction_a = rollout_structured_body(
            model, history, future_a, dts, held)
        prediction_b = rollout_structured_body(
            model, history, future_b, dts, held)
        torch.testing.assert_close(prediction_a, prediction_b,
                                   rtol=0.0, atol=1e-6)

    def test_coverage_report_is_strict_json_with_open_ended_bins(self):
        frames = np.zeros((2, 9), dtype=np.float32)
        frames[:, 0] = [1.0, 8.2]
        frames[:, 3] = [0.1, 0.5]
        data = {
            "frames": frames,
            "bounds": np.asarray([[0, 2]], dtype=np.int64),
            "seq_run": np.asarray([0], dtype=np.int32),
        }
        profile = _coverage_profile(data, [0])
        self.assertIn("8-10mps__0.42+rad", profile["speed_steering_cells"])
        self.assertEqual(profile["speed_bins"]["10+mps"], 0)
        json.dumps(profile, allow_nan=False)

    def test_cv_aggregate_preserves_each_architecture_and_horizon(self):
        def run_record(value):
            return {"30": {"all": {
                "windows": 5,
                "rmse": {"u_mps": value, "v_mps": value / 2,
                         "yaw_rate_rps": value / 3},
            }}}

        results = {
            "gru": {"run_a": run_record(1.0), "run_b": run_record(2.0)},
            "latent": {"run_a": run_record(0.8), "run_b": run_record(2.2)},
        }
        summary = _aggregate_cv(results, bootstrap_count=100, seed=17)
        self.assertEqual(set(summary), {"gru", "latent"})
        self.assertEqual(set(summary["gru"]), {"1", "5", "10", "20", "30"})
        self.assertEqual(
            summary["gru"]["30"]["regions"]["all"]["u_mps"][
                "independent_runs"], 2)
        self.assertEqual(
            summary["latent"]["30"]["regions"]["all"]["u_mps"][
                "macro_run_rmse"], 1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
