#!/usr/bin/env python3
"""Focused causal-window tests for the swerve sensor observer evaluator."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor

from tools.vehicle_dynamics_learning.evaluate_swerve_sensor_observer import (
    SENSOR_LAGS,
    _load_frozen_checkpoint,
    _integrated_lap_metrics,
    _integrated_sequence_metrics,
    _equal_run_sample_weights,
    _lagged_sensor_matrix,
    _regime_metrics,
    _simulator_rear_axle_targets,
    evaluate,
)


class SwerveSensorObserverTests(unittest.TestCase):
    def test_frozen_checkpoint_requires_exact_feature_and_lag_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.joblib"
            model = DummyRegressor(strategy="constant", constant=0.0).fit([[0.0]], [0.0])
            joblib.dump({
                "models_u_v": [model, model],
                "sensor_feature_names": ["encoder", "imu"],
                "sensor_lags": SENSOR_LAGS,
                "training_run_ids": ["train-r01"],
                "dataset": "train.npz",
            }, path)

            checkpoint = _load_frozen_checkpoint(path, ["encoder", "imu"])

            self.assertEqual(checkpoint["training_run_ids"], ["train-r01"])
            with self.assertRaisesRegex(ValueError, "feature names"):
                _load_frozen_checkpoint(path, ["encoder"])
            broken = joblib.load(path)
            broken["sensor_lags"] = (0, 1)
            joblib.dump(broken, path)
            with self.assertRaisesRegex(ValueError, "lags"):
                _load_frozen_checkpoint(path, ["encoder", "imu"])

    def test_frozen_evaluation_scores_validation_only_without_refitting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "validation.npz"
            checkpoint_path = root / "frozen.joblib"
            output_dir = root / "score"
            sensor_names = [
                "rear_left_surface_mps", "rear_right_surface_mps",
                "steering_feedback_rad", "imu_yaw_rate_rps",
            ]
            model_u = DummyRegressor(strategy="constant", constant=2.0).fit(
                [[0.0]], [2.0])
            model_v = DummyRegressor(strategy="constant", constant=0.1).fit(
                [[0.0]], [0.1])
            joblib.dump({
                "models_u_v": [model_u, model_v],
                "sensor_feature_names": sensor_names,
                "sensor_lags": SENSOR_LAGS,
                "training_run_ids": ["train-r01"],
                "dataset": "training.npz",
                "runtime_status": "diagnostic_only_not_integrated",
            }, checkpoint_path)
            original_checkpoint = checkpoint_path.read_bytes()
            count = 16
            rigid = np.zeros((count, 13), dtype=np.float32)
            rigid[:, 7] = 2.0
            rigid[:, 8] = 0.1
            np.savez_compressed(
                dataset,
                schema_version=np.asarray([7], dtype=np.int32),
                frames=np.zeros((count, 3), dtype=np.float32),
                sensor_frames=np.column_stack((
                    np.full(count, 2.0), np.full(count, 2.0),
                    np.zeros(count), np.zeros(count),
                )).astype(np.float32),
                sensor_valid=np.ones(count, dtype=bool),
                sequence_bounds=np.asarray([[0, count]], dtype=np.int64),
                sequence_run_index=np.asarray([0], dtype=np.int32),
                run_ids=np.asarray(["unseen-r01"]),
                run_splits=np.asarray(["validation"]),
                sensor_feature_names=np.asarray(sensor_names),
                simulator_pose_xyyaw=np.column_stack((
                    np.arange(count) * 0.05, np.zeros(count), np.zeros(count),
                )).astype(np.float32),
                lap_count=np.full(count, -1, dtype=np.int32),
                dt_s=np.full(count, 0.025, dtype=np.float32),
                simulator_rigid_state=rigid,
            )

            report = evaluate(dataset, output_dir, frozen_checkpoint=checkpoint_path)

            self.assertEqual(report["checkpoint_mode"], "frozen_evaluation_only")
            self.assertEqual(report["validation_runs"], ["unseen-r01"])
            self.assertEqual(report["samples"]["primary_dataset_train"], 0)
            self.assertGreater(report["samples"]["validation"], 0)
            self.assertEqual(checkpoint_path.read_bytes(), original_checkpoint)
            self.assertFalse((output_dir / "swerve_sensor_observer.joblib").exists())

    def test_equal_run_weights_equalize_total_capture_mass(self) -> None:
        ids = np.asarray(["short", "long", "long", "long", "short"])

        weights = _equal_run_sample_weights(ids)

        self.assertAlmostEqual(float(weights[ids == "short"].sum()), 2.5)
        self.assertAlmostEqual(float(weights[ids == "long"].sum()), 2.5)

    def test_lagged_features_never_cross_sequence_boundary(self) -> None:
        sensors = np.arange(40, dtype=np.float32).reshape(20, 2)
        valid = np.ones(20, dtype=bool)
        bounds = np.asarray([[0, 10], [10, 20]], dtype=np.int64)
        run_by_sequence = np.asarray([0, 1], dtype=np.int32)

        features, frame_index, row_run_index, row_sequence_index = _lagged_sensor_matrix(
            sensors, valid, bounds, run_by_sequence)

        self.assertEqual(frame_index.tolist(), [8, 9, 18, 19])
        self.assertEqual(row_run_index.tolist(), [0, 0, 1, 1])
        self.assertEqual(row_sequence_index.tolist(), [0, 0, 1, 1])
        self.assertEqual(features.shape, (4, 2 * len(SENSOR_LAGS)))
        # At index 18, the oldest feature is row 10, not row 2 from the
        # preceding reset-bounded sequence.
        self.assertEqual(features[2, -2:].tolist(), sensors[10].tolist())

    def test_invalid_sensor_in_history_excludes_scored_row(self) -> None:
        sensors = np.ones((20, 2), dtype=np.float32)
        valid = np.ones(20, dtype=bool)
        valid[10] = False

        _, frame_index, _, _ = _lagged_sensor_matrix(
            sensors, valid, np.asarray([[0, 20]]), np.asarray([0]))

        self.assertNotIn(18, frame_index.tolist())
        self.assertIn(19, frame_index.tolist())

    def test_regime_metrics_count_independent_runs(self) -> None:
        truth = np.asarray([[4.0, 0.1], [4.0, -0.1], [6.0, 0.2]])
        candidate = truth + np.asarray([[0.1, 0.0], [0.1, 0.0], [0.2, 0.0]])
        baseline = truth + np.asarray([[0.3, -0.1], [0.3, 0.1], [0.4, -0.2]])
        speed = np.asarray([4.0, 4.5, 6.0])
        steering = np.asarray([0.05, -0.09, 0.15])
        runs = np.asarray([0, 1, 1])

        rows = _regime_metrics(
            truth, candidate, baseline, speed, steering, runs,
            speed_edges=(0.0, 5.0, 7.0),
            steering_edges=(0.0, 0.1, 0.2))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["independent_runs"], 2)
        self.assertEqual(rows[0]["per_run"]["0"]["samples"], 1)
        self.assertEqual(rows[0]["per_run"]["1"]["samples"], 1)
        self.assertLess(rows[0]["u_candidate_rmse_mps"],
                        rows[0]["u_wheel_mean_baseline_rmse_mps"])
        self.assertEqual(rows[1]["samples"], 1)

    def test_pose_integration_uses_truth_only_as_sequence_anchor(self) -> None:
        indices = np.asarray([0, 1, 2], dtype=np.int64)
        sequence_ids = np.asarray([0, 0, 0], dtype=np.int32)
        run_ids = np.asarray([0, 0, 0], dtype=np.int32)
        bounds = np.asarray([[0, 3]], dtype=np.int64)
        pose = np.asarray([[2.0, 3.0, 0.0],
                           [2.05, 3.0, 0.0],
                           [2.10, 3.0, 0.0]])
        velocity = np.tile(np.asarray([[2.0, 0.0]]), (3, 1))

        result = _integrated_sequence_metrics(
            indices, sequence_ids, run_ids, bounds,
            velocity, velocity, pose, np.zeros(3), np.full(3, 0.025),
            rear_axle_to_com_x_m=0.0)

        self.assertAlmostEqual(result["sequences"][0]["candidate_path_rmse_m"], 0.0)
        self.assertAlmostEqual(result["sequences"][0]["candidate_endpoint_error_m"], 0.0)

    def test_lap_segment_integration_uses_truth_only_as_segment_anchor(self) -> None:
        indices = np.arange(20, dtype=np.int64)
        velocity = np.tile(np.asarray([[2.0, 0.0]]), (20, 1))
        pose = np.column_stack((0.05 * indices, np.zeros(20), np.zeros(20)))

        result = _integrated_lap_metrics(
            indices, np.zeros(20, dtype=np.int32), np.full(20, 4, dtype=np.int32),
            velocity, pose, np.zeros(20), np.full(20, 0.025),
            np.asarray(["practice-test"]))

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["lap_count"], 4)
        self.assertAlmostEqual(result["segments"][0]["candidate_path_rmse_m"], 0.0)
        self.assertAlmostEqual(result["segments"][0]["candidate_endpoint_error_m"], 0.0)

    def test_simulator_com_velocity_is_shifted_to_rear_axle(self) -> None:
        rigid = np.zeros((1, 13), dtype=np.float32)
        rigid[0, 7:9] = [3.0, 1.0]
        rigid[0, 12] = 2.0

        target = _simulator_rear_axle_targets(rigid)

        self.assertAlmostEqual(float(target[0, 0]), 3.0)
        self.assertAlmostEqual(float(target[0, 1]),
                               1.0 - 2.0 * 0.15532, places=6)
        self.assertAlmostEqual(float(target[0, 2]), 2.0)


if __name__ == "__main__":
    unittest.main()
