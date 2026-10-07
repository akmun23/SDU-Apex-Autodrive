#!/usr/bin/env python3
"""Focused mathematical checks for the local yaw-atlas fitter."""

from __future__ import annotations

import unittest

import numpy as np

from fit_fullband_yaw_regime_atlas import (
    COM_X_M,
    RunSeries,
    _fit_model,
    _make_rows,
    _rear_lateral_velocity,
    _predict,
)


class YawRegimeAtlasMathTest(unittest.TestCase):
    def test_rigid_body_lateral_speed_uses_body_frame_com_offset(self) -> None:
        rigid = np.zeros((2, 13), dtype=np.float64)
        rigid[:, 3:7] = (0.0, 0.0, 0.0, 1.0)
        rigid[:, 8] = (0.5, -0.2)
        rigid[:, 12] = (2.0, -1.0)
        expected = rigid[:, 8] - COM_X_M * rigid[:, 12]
        np.testing.assert_allclose(_rear_lateral_velocity(rigid), expected)

    def test_next_yaw_increment_is_adjacent_and_fixed_step(self) -> None:
        frames = np.zeros((6, 9), dtype=np.float64)
        frames[:, 0] = 4.0
        frames[:, 3] = 0.1
        frames[:, 4] = 0.2
        frames[:, 5:7] = 4.0
        rigid = np.zeros((6, 13), dtype=np.float64)
        rigid[:, 6] = 1.0
        rigid[:, 7] = 4.0
        rigid[:, 12] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        series = RunSeries("synthetic", "train", "synthetic",
                           frames, rigid, np.asarray(((0, 6),)))
        x, delta, cells, phases, _, sequence_ids, frame_indices = _make_rows(series)
        self.assertEqual(len(delta), 3)
        np.testing.assert_allclose(x[:, 0], (0.2, 0.3, 0.4))
        np.testing.assert_allclose(delta, (0.1, 0.1, 0.1), atol=1e-12)
        np.testing.assert_array_equal(cells[:, 0], (8, 8, 8))
        np.testing.assert_array_equal(phases, (0, 0, 0))
        np.testing.assert_array_equal(sequence_ids, (0, 0, 0))
        np.testing.assert_array_equal(frame_indices, (2, 3, 4))

    def test_command_tracking_errors_are_current_sample_features(self) -> None:
        frames = np.zeros((6, 9), dtype=np.float64)
        frames[:, 0] = 4.0
        frames[:, 3] = 0.1
        frames[:, 4] = 0.2
        frames[:, 5:7] = 4.0
        frames[:, 7] = 0.15
        frames[:, 8] = 0.3
        frames[3:, 7:9] = 0.45
        rigid = np.zeros((6, 13), dtype=np.float64)
        rigid[:, 6] = 1.0
        rigid[:, 7] = 4.0
        rigid[:, 12] = np.linspace(0.0, 0.5, 6)
        series = RunSeries("synthetic", "train", "synthetic",
                           frames, rigid, np.asarray(((0, 6),)))

        x, *_ = _make_rows(series, include_command_errors=True,
                           include_command_rates=True)

        self.assertEqual(x.shape, (3, 14))
        np.testing.assert_allclose(x[0, -4:], (0.05, 0.1, 0.0, 0.0))
        np.testing.assert_allclose(x[1, -2:], (12.0, 6.0))

    def test_lagged_history_features_use_only_prior_25ms_samples(self) -> None:
        frames = np.zeros((7, 9), dtype=np.float64)
        frames[:, 0] = 4.0
        frames[:, 3] = (0.0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.21)
        frames[:, 4] = (0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80)
        frames[:, 5:7] = 4.0
        rigid = np.zeros((7, 13), dtype=np.float64)
        rigid[:, 6] = 1.0
        rigid[:, 7] = 4.0
        rigid[:, 12] = (0.0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36)
        series = RunSeries("synthetic", "train", "synthetic",
                           frames, rigid, np.asarray(((0, 7),)))

        features, *_ = _make_rows(series, include_lagged_history=True)

        # First row corresponds to k=2: all appended inputs come from k-1/k-2.
        np.testing.assert_allclose(features[0, -3:], (0.01, 0.4, 4.0))

    def test_current_imu_roll_features_are_aligned_and_invalid_rows_are_skipped(self) -> None:
        frames = np.zeros((7, 9), dtype=np.float64)
        frames[:, 0] = 4.0
        frames[:, 3] = 0.1
        frames[:, 4] = 0.2
        frames[:, 5:7] = 4.0
        rigid = np.zeros((7, 13), dtype=np.float64)
        rigid[:, 6] = 1.0
        rigid[:, 7] = 4.0
        rigid[:, 12] = np.arange(7) * 0.01
        attitude = np.zeros((7, 4), dtype=np.float64)
        attitude[:, 0] = np.arange(7) * 0.02
        attitude[:, 2] = np.arange(7) * 0.1
        valid = np.ones(7, dtype=bool)
        valid[3] = False
        series = RunSeries("synthetic", "train", "synthetic", frames,
                           rigid, np.asarray(((0, 7),)), attitude, valid)

        features, _, _, _, _, _, indices = _make_rows(
            series, include_imu_roll=True, require_imu_roll=True)

        # k=3 is excluded; all retained values come from the current frame.
        np.testing.assert_array_equal(indices, (2, 4, 5))
        np.testing.assert_allclose(features[:, -2:],
                                   ((0.04, 0.2), (0.08, 0.4), (0.10, 0.5)))

    def test_roll_features_require_an_attitude_stream(self) -> None:
        frames = np.zeros((6, 9), dtype=np.float64)
        rigid = np.zeros((6, 13), dtype=np.float64)
        series = RunSeries("synthetic", "train", "synthetic", frames,
                           rigid, np.asarray(((0, 6),)))

        with self.assertRaisesRegex(ValueError, "no IMU attitude"):
            _make_rows(series, require_imu_roll=True)

    def test_signed_rear_wheel_difference_is_not_averaged_away(self) -> None:
        frames = np.zeros((6, 9), dtype=np.float64)
        frames[:, 0] = 4.0
        frames[:, 3] = 0.1
        frames[:, 4] = 0.2
        frames[:, 5] = 4.2
        frames[:, 6] = 3.8
        rigid = np.zeros((6, 13), dtype=np.float64)
        rigid[:, 6] = 1.0
        rigid[:, 7] = 4.0
        rigid[:, 12] = np.linspace(0.0, 0.5, 6)
        series = RunSeries("synthetic", "train", "synthetic",
                           frames, rigid, np.asarray(((0, 6),)))

        x, *_ = _make_rows(series, include_rear_wheel_split=True)

        self.assertEqual(x.shape, (3, 11))
        np.testing.assert_allclose(x[:, -1], 0.4)

    def test_fixed_feature_scales_prevent_tiny_variance_amplification(self) -> None:
        x = np.zeros((80, 10), dtype=np.float64)
        x[:, 0] = np.linspace(-0.5, 0.5, len(x))
        x[:, 4] = np.linspace(-1.0e-4, 1.0e-4, len(x))
        target = 0.01 * x[:, 0] + 0.02 * x[:, 4]
        run_ids = np.asarray(["r1"] * 40 + ["r2"] * 40, dtype=object)
        model = _fit_model(x, target, run_ids)
        self.assertIsNotNone(model)
        prediction = _predict(model, np.asarray((0.0, 0, 0, 0, 0.1,
                                                  0, 0, 0, 0, 0)))
        self.assertTrue(np.isfinite(prediction))
        self.assertLess(abs(prediction), 0.1)


if __name__ == "__main__":
    unittest.main()
