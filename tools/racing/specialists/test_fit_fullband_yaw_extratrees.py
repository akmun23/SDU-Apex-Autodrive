#!/usr/bin/env python3
"""Focused checks for the frozen nonlinear yaw-transition candidate."""

from __future__ import annotations

import unittest

import numpy as np

from tools.racing.specialists.fit_fullband_yaw_extratrees import (
    _feature_schema,
    _run_balanced_weights,
    predict_transition,
)


class _FixedDeltaEstimator:
    def predict(self, features: np.ndarray) -> np.ndarray:
        self.last_features = np.asarray(features)
        return np.asarray((0.125,), dtype=np.float64)


class FullbandYawExtraTreesTest(unittest.TestCase):
    def test_run_balancing_gives_each_capture_equal_total_weight(self) -> None:
        ids = np.asarray(("short", "short", "long", "long", "long",
                          "long", "long", "long"), dtype=object)
        weights = _run_balanced_weights(ids)
        self.assertAlmostEqual(float(weights[ids == "short"].sum()), 1.0)
        self.assertAlmostEqual(float(weights[ids == "long"].sum()), 1.0)

    def test_feature_schema_is_current_sample_v11_schema(self) -> None:
        names, scales = _feature_schema()
        self.assertEqual(len(names), 14)
        self.assertEqual(scales.shape, (14,))
        self.assertIn("steering_command_rate_radps", names)
        self.assertNotIn("future", " ".join(names).lower())
        history_names, history_scales = _feature_schema(
            include_lagged_history=True)
        self.assertEqual(len(history_names), 17)
        self.assertEqual(history_scales.shape, (17,))
        self.assertEqual(history_names[-3:], (
            "previous_yaw_rate_increment_radps",
            "previous_steering_rate_radps",
            "previous_throttle_rate_per_s",
        ))

    def test_exact_supported_key_predicts_and_missing_key_abstains(self) -> None:
        estimator = _FixedDeltaEstimator()
        package = {
            "feature_names": ("current_yaw_rate", "other"),
            "models": {
                (4, 20, -1): {
                    "estimator": estimator,
                    "feature_mean": np.zeros(2),
                    "feature_scales": np.ones(2),
                },
            },
        }
        features = np.asarray((0.4, -0.2))
        self.assertAlmostEqual(
            predict_transition(package, (4, 20, -1), features), 0.525)
        self.assertIsNone(predict_transition(package, (4, 20, 1), features))
        with self.assertRaises(ValueError):
            predict_transition(package, (4, 20, -1), np.ones(3))


if __name__ == "__main__":
    unittest.main()
