from __future__ import annotations

import unittest

import numpy as np

from tools.racing.offline.localization.amcl_surrogate import (
    EmpiricalAmclSurrogate,
    LocalizationRun,
)


def _runs(duration_s: float = 4.0) -> tuple[LocalizationRun, ...]:
    times = np.arange(0.0, duration_s + 1e-9, 0.025)
    residual = np.zeros((len(times), 3), dtype=np.float64)
    return tuple(LocalizationRun(name, times, residual, 0.010)
                 for name in ("practice_a", "practice_b"))


class EmpiricalAmclSurrogateTest(unittest.TestCase):
    def test_anchor_and_measured_latency_are_applied(self) -> None:
        surrogate = EmpiricalAmclSurrogate(
            _runs(), seed=42, rollout_seconds=1.0)
        plant_start = np.asarray((5.0, 2.0, 0.2))
        map_start = np.asarray((100.0, 200.0, 0.3))
        surrogate.reset(
            time_s=1.0,
            initial_plant_pose_xyyaw=plant_start,
            initial_localization_pose_xyyaw=map_start,
            rollout_seconds=1.0,
        )
        self.assertIsNone(surrogate.step(
            1.005, np.asarray((5.02, 2.0, 0.2))))
        output = surrogate.step(1.025, np.asarray((5.1, 2.0, 0.2)))
        self.assertIsNotNone(output)
        expected_delta = 0.6 * 0.1 * np.asarray((np.cos(0.1), np.sin(0.1)))
        np.testing.assert_allclose(output[:2], map_start[:2] + expected_delta,
                                   atol=1e-9)
        self.assertAlmostEqual(output[2], map_start[2], places=9)

    def test_trace_queries_do_not_silently_return_zero(self) -> None:
        surrogate = EmpiricalAmclSurrogate(
            _runs(), seed=7, rollout_seconds=1.0)
        surrogate.reset(rollout_seconds=1.0)
        surrogate._trace_origin_s = surrogate._run.time_s[-1] - 0.005
        with self.assertRaises(RuntimeError):
            surrogate._sample_trace(surrogate._run.time_s[-1] + 0.01)

    def test_reset_rejects_short_training_trace_for_requested_horizon(self) -> None:
        surrogate = EmpiricalAmclSurrogate(
            _runs(0.5), seed=3, rollout_seconds=0.2)
        with self.assertRaisesRegex(ValueError, "too short"):
            surrogate.reset(rollout_seconds=0.6)


if __name__ == "__main__":
    unittest.main()
