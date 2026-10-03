"""Mathematical checks for the recursive-divergence diagnostic."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.diagnose_recursive_first_divergence import (
    _first_crossing_step,
    _per_run_curves,
)


class RecursiveDivergenceDiagnosticTest(unittest.TestCase):
    def test_first_crossing_is_one_based_and_strict(self):
        self.assertEqual(_first_crossing_step(np.asarray([1.0, 1.9, 2.1]), 2.0), 3)
        self.assertIsNone(_first_crossing_step(np.asarray([1.0, 2.0]), 2.0))

    def test_run_curves_weight_runs_equally_not_by_window_count(self):
        error = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        result = _per_run_curves(error, np.asarray([0, 0, 1]),
                                 np.asarray(["run_a", "run_b"]))
        self.assertEqual(result["macro_run_rmse"], [2.0, 3.0])
        self.assertEqual(result["per_run_bias"]["run_a"], [1.0, 2.0])
        self.assertEqual(result["per_run_bias"]["run_b"], [3.0, 4.0])

    def test_invalid_crossing_inputs_rejected(self):
        with self.assertRaises(ValueError):
            _first_crossing_step(np.ones((2, 2)), 1.0)
        with self.assertRaises(ValueError):
            _first_crossing_step(np.ones(2), -1.0)


if __name__ == "__main__":
    unittest.main()
