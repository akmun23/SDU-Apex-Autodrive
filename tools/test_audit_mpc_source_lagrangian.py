"""Curvature certification must detect cross terms and reject branch changes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import audit_mpc_source_lagrangian as curvature
from tools.qualify_mpc_source_phase1 import CONTROL_SCALE


class QuadraticOracle:
    crosses_branch = False
    hessian_error = 0.

    def __init__(self, *_):
        self.calls = 0
        self.matrix = np.eye(60)
        self.matrix[::2, 1::2] += .01
        self.matrix[1::2, ::2] += .01

    def evaluate(self, controls, **options):
        self.calls += 1
        z = np.asarray(controls).ravel() / CONTROL_SCALE
        result = {"valid": True, "qp_original_duals_trustworthy": True,
            "physical_multipliers": [0.] * 480, "max_steering_rad": .5,
            "gradient_scaled": (self.matrix @ z).tolist(),
            "hard_jacobian_scaled": np.zeros((480, 60)).tolist(),
            "hard_margins": [1.] * 480,
            "original_cost": float(.5 * z @ self.matrix @ z),
            "physical_branch_changes": {"test_switch": int(self.crosses_branch)},
            "peak_branch_changes": 0, "geometry_branch_changes": 0}
        if 'constraint_direction' in options:
            direction=np.asarray(options['constraint_direction']).ravel()/CONTROL_SCALE
            result['constraint_path_model']={"valid": True,
                "first": [0.]*480,"second": [0.]*480,
                "original_objective_first": float((self.matrix @ z) @ direction),
                "original_objective_second": float(direction @ self.matrix @ direction)}
        if 'lagrangian_hessian_multipliers' in options:
            result['lagrangian_hessian']={"valid": True,
                "scaled_row_major": (self.matrix + self.hessian_error*np.eye(60)).ravel().tolist(),
                "source_directional_evaluations": 1833,
                "quadratic_consistency_relative": 0.}
        return result

    def close(self):
        pass


class LagrangianCurvatureAudit(unittest.TestCase):
    def run_audit(self, crosses_branch=False):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            np.savetxt(case / "initial.controls", np.tile([.1, .5], (30, 1)))
            with patch.object(curvature, "SourceOracle", QuadraticOracle), \
                 patch.object(QuadraticOracle, "crosses_branch", crosses_branch):
                return curvature.audit(case, Path())

    def test_smooth_coupled_quadratic_passes_full_hessian_and_hvps(self):
        result = self.run_audit()
        self.assertTrue(result["passed"])
        self.assertGreater(result["steering_throttle_cross_curvature_norm"], .1)
        self.assertTrue(all(check["relative_error"] < 1e-6
                            for check in result["hessian_vector_checks"]))

    def test_small_fd_error_cannot_hide_different_source_branches(self):
        result = self.run_audit(crosses_branch=True)
        self.assertLess(result["hessian_antisymmetry_relative"], 1e-6)
        self.assertFalse(result["passed"])
        self.assertTrue(all(not check["same_active_branches"] for check in result["columns"]))

    def test_consistent_symmetric_but_wrong_analytical_hessian_fails(self):
        with patch.object(QuadraticOracle, "hessian_error", 1.):
            result = self.run_audit()
        self.assertTrue(result["finite_difference_reference_passed"])
        self.assertFalse(result["passed"])
        self.assertTrue(all(check["relative_error"] > .1
                            for check in result["hessian_vector_checks"]))


if __name__ == "__main__":
    unittest.main()
