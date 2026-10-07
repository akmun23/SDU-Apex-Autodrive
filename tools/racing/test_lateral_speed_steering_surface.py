from __future__ import annotations

import unittest

import numpy as np

from tools.racing.analyze_lateral_speed_steering_surface import (
    condition_quantiles,
    steering_bin,
    training_capability,
    validation_score,
)


class LateralSpeedSteeringSurfaceTest(unittest.TestCase):
    @staticmethod
    def _data(run_values: tuple[float, float, float], samples_per_condition: int = 20):
        run, sequence, values = [], [], []
        for run_index, run_value in enumerate(run_values):
            for condition in range(3):
                run.extend([run_index] * samples_per_condition)
                sequence.extend([run_index * 3 + condition] * samples_per_condition)
                values.extend([run_value] * samples_per_condition)
        return {
            "run": np.asarray(run, dtype=np.int32),
            "seq": np.asarray(sequence, dtype=np.int32),
        }, np.asarray(values, dtype=np.float64)

    def test_steering_bins_use_absolute_angle_and_include_limit(self) -> None:
        values = steering_bin(np.asarray([-0.5236, -0.1, 0.0, 0.1, 0.5236]))
        np.testing.assert_array_equal(values, [5, 1, 0, 1, 5])

    def test_training_cap_weights_independent_runs_not_frame_count(self) -> None:
        data, values = self._data((1.0, 2.0, 4.0))
        fit = training_capability(
            data, np.ones(values.size, dtype=bool), values,
            np.asarray(["run-a", "run-b", "run-c"]), 20)
        self.assertTrue(fit["support_pass"])
        self.assertEqual(fit["supported_runs"], 3)
        self.assertAlmostEqual(fit["candidate_cap_mps2"], 1.8)

    def test_training_cap_uses_lateral_acceleration_magnitude(self) -> None:
        data, values = self._data((-1.0, -2.0, -4.0))
        fit = training_capability(
            data, np.ones(values.size, dtype=bool), values,
            np.asarray(["run-a", "run-b", "run-c"]), 20)
        self.assertAlmostEqual(fit["candidate_cap_mps2"], 1.8)

    def test_validation_cap_is_not_passed_when_a_run_exceeds_limit(self) -> None:
        data, values = self._data((1.0, 1.0, 1.0))
        values[-20:] = 3.0
        result = validation_score(
            data, np.ones(values.size, dtype=bool), values,
            candidate_cap=2.0, minimum_samples=20,
            run_names=np.asarray(["run-a", "run-b", "run-c"]),
        )
        self.assertEqual(result["status"], "fail")

    def test_short_sequences_do_not_create_support(self) -> None:
        data, values = self._data((1.0, 2.0, 4.0), samples_per_condition=4)
        sequence_rows, eligible = condition_quantiles(
            data, np.ones(values.size, dtype=bool), values, 5)
        self.assertEqual(sequence_rows, {})
        self.assertEqual(eligible.size, 0)


if __name__ == "__main__":
    unittest.main()
