import unittest

import numpy as np
from scipy import sparse

from tools.mpc_qp_oracle import minimum_uniform_box_relaxation


class MinimumUniformBoxRelaxationTest(unittest.TestCase):
    def setUp(self):
        # Exact equality x = 1 conflicts with the box x <= 0. A second
        # unconstrained variable keeps the LP shape representative.
        self.matrix = sparse.csr_matrix([[1.0, 0.0], [1.0, 0.0]])
        self.lower = np.array([1.0, -np.inf])
        self.upper = np.array([1.0, 0.0])

    def test_relaxing_the_conflicting_variable_restores_feasibility(self):
        result = minimum_uniform_box_relaxation(
            self.matrix, self.lower, self.upper, 1,
            lambda variables: variables == 0)
        self.assertEqual(result["status"], "feasible")
        self.assertAlmostEqual(result["minimum_uniform_relaxation"], 1.0)

    def test_unrelated_box_family_does_not_hide_infeasibility(self):
        result = minimum_uniform_box_relaxation(
            self.matrix, self.lower, self.upper, 1,
            lambda variables: variables == 1)
        self.assertEqual(result["status"], "infeasible")
        self.assertIsNone(result["minimum_uniform_relaxation"])

    def test_coupled_footprint_rows_are_never_relaxed_as_boxes(self):
        # x=1,y=0, box x<=0 can be widened, but x+y<=.5 remains hard.
        matrix = sparse.csr_matrix([[1., 0.], [0., 1.], [1., 0.], [1., 1.]])
        lower = np.array([1., 0., -np.inf, -np.inf])
        upper = np.array([1., 0., 0., .5])
        result = minimum_uniform_box_relaxation(matrix, lower, upper, 2,
            lambda variables: variables == 0)
        self.assertEqual(result["status"], "infeasible")
        upper[-1] = 2.
        result = minimum_uniform_box_relaxation(matrix, lower, upper, 2,
            lambda variables: variables == 0)
        self.assertEqual(result["status"], "feasible")
        self.assertAlmostEqual(result["minimum_uniform_relaxation"], 1.)


if __name__ == "__main__":
    unittest.main()
