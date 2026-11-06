"""Verify the external solver adapter, multiplier signs and strict certificates."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import casadi as ca
except ModuleNotFoundError:
    ca = None
    reference = None
else:
    from tools import reference_mpc_source_ipopt as reference


class QuadraticPhysicalOracle:
    def __init__(self, *_):
        self.calls = 0

    def evaluate(self, controls):
        self.calls += 1
        command = np.asarray(controls).ravel()
        z = command / reference.CONTROL_SCALE
        target = np.tile([0., 2.], 30)
        target[0] = -.2
        gradient = z - target
        jacobian = np.zeros((480, 60))
        margins = np.ones(480)
        jacobian[0, 0] = 1.
        margins[0] = z[0]
        row_types = ["body"] * 360 + ["box"] * 120
        for column in range(60):
            scale = reference.CONTROL_SCALE[column]
            lower, upper = (-.5, .5) if column % 2 == 0 else (.001, 1.)
            jacobian[360 + 2 * column, column] = scale
            jacobian[361 + 2 * column, column] = -scale
            margins[360 + 2 * column] = command[column] - lower
            margins[361 + 2 * column] = upper - command[column]
        return {"valid": True, "original_cost": float(.5 * gradient @ gradient),
            "hard_margins": margins.tolist(), "gradient_scaled": gradient.tolist(),
            "hard_jacobian_scaled": jacobian.tolist(), "row_types": row_types,
            "minimum_clearance_m": float(margins[0]), "minimum_rate_margin": 1.,
            "minimum_box_margin": float(margins[360:].min()), "max_steering_rad": .5}

    def close(self):
        pass


@unittest.skipIf(ca is None, "Install optional tools/requirements-source-reference.txt")
class IpoptSourceAdapter(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(reference, "SourceOracle", QuadraticPhysicalOracle)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.controls = np.tile([.05, .5], (30, 1)).astype(np.float32)
        self.problem = reference.SourceProblem(Path(), Path(), self.controls)

    def test_exact_callback_jacobian_has_original_scaled_cross_interface(self):
        function = reference.SourceFunction(self.problem)
        z = self.controls.ravel() / reference.CONTROL_SCALE
        value = function(z)
        matrix = np.asarray(function.jacobian()(z, value))
        score = self.problem.value(z)
        np.testing.assert_allclose(matrix[0], score["gradient_scaled"])
        np.testing.assert_array_equal(matrix[1:], np.asarray(score["hard_jacobian_scaled"])[:360])
        calls = self.problem.oracle.calls
        function(z)
        self.assertEqual(calls, self.problem.oracle.calls)

    def test_lower_upper_bound_duals_return_to_original_row_units(self):
        lam_g = np.zeros(360)
        lam_g[0] = -2.
        lam_x = np.zeros(60)
        lam_x[0], lam_x[1] = -.3, .5
        dual = self.problem.multipliers(self.problem.score, lam_g, lam_x)
        self.assertEqual(dual[0], 2.)
        self.assertAlmostEqual(dual[360], 2.)
        self.assertEqual(dual[361], 0.)
        self.assertEqual(dual[362], 0.)
        self.assertAlmostEqual(dual[363], 2.)
        np.testing.assert_allclose(
            np.asarray(self.problem.score["hard_jacobian_scaled"])[360:].T @ dual[360:], -lam_x)

    def test_ipopt_solves_known_active_constraint_and_independent_kkt(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            np.savetxt(case / "initial.controls", self.controls)
            (case / "source_context.mctx").touch()
            (case / "mpc_source_nlp_oracle").touch()
            result = reference.solve(case, case, case / "output", 100)
            self.assertEqual(result["ipopt_return_status"], "Solve_Succeeded")
            self.assertTrue(result["qualified"], result["final"])
            self.assertLess(result["final"]["stationarity_inf_scaled"], 1e-4)
            self.assertTrue(result["history"])
            self.assertFalse(result["witness_initialization"])


if __name__ == "__main__":
    unittest.main()
