"""Fail-closed qualification of original-source feasibility and KKT claims."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import competition_failure_benchmark as benchmark
from tools import qualify_mpc_source_phase1 as qualification


class PhysicalCertificates(unittest.TestCase):
    def score(self):
        jacobian = np.zeros((480, 60))
        jacobian[0, 0] = 1.
        return {"valid": True, "original_cost": 123.,
            "minimum_clearance_m": 0., "minimum_rate_margin": 1., "minimum_box_margin": 1.,
            "hard_margins": [0.] + [1.] * 479,
            "gradient_scaled": [2.] + [0.] * 59,
            "hard_jacobian_scaled": jacobian.tolist()}

    def test_feasibility_does_not_certify_stationarity(self):
        result = qualification.physical_metrics(self.score(), [0.] * 480)
        self.assertTrue(result["physically_feasible"])
        self.assertEqual(result["stationarity_inf_scaled"], 2.)
        self.assertFalse(result["kkt_converged"])

    def test_normalized_stationarity_uses_physical_duals(self):
        result = qualification.physical_metrics(self.score(), [2.] + [0.] * 479)
        self.assertTrue(result["kkt_converged"])
        self.assertEqual(result["stationarity_inf_scaled"], 0.)
        self.assertEqual(result["complementarity_inf"], 0.)

    def test_any_physical_wall_penetration_fails(self):
        score = self.score()
        score["minimum_clearance_m"] = score["hard_margins"][0] = -1e-12
        result = qualification.physical_metrics(score, [2.] + [0.] * 479)
        self.assertLess(result["maximum_hard_violation"], qualification.PRIMAL_TOLERANCE)
        self.assertFalse(result["physically_feasible"])
        self.assertFalse(result["kkt_converged"])

    def test_negative_or_missing_duals_cannot_certify_kkt(self):
        score = self.score()
        score["gradient_scaled"][0] = -2.
        bad = qualification.physical_metrics(score, [-2.] + [0.] * 479)
        self.assertEqual(bad["stationarity_inf_scaled"], 0.)
        self.assertFalse(bad["kkt_converged"])
        self.assertFalse(qualification.physical_metrics(score)["kkt_converged"])

    def test_noncomplementary_duals_cannot_certify_kkt(self):
        score = self.score()
        score["minimum_clearance_m"] = score["hard_margins"][0] = .1
        result = qualification.physical_metrics(score, [2.] + [0.] * 479)
        self.assertEqual(result["stationarity_inf_scaled"], 0.)
        self.assertAlmostEqual(result["complementarity_inf"], .2)
        self.assertFalse(result["kkt_converged"])

    def test_budget_and_single_replay_cannot_qualify(self):
        with self.assertRaises(ValueError):
            qualification.qualify(Path(), Path(), Path(), iterations=1, repeat=1)
        with self.assertRaises(ValueError):
            qualification.qualify(Path(), Path(), Path(), iterations=2001)


class AffineFeasibilityAudit(unittest.TestCase):
    def problem(self, rows, margins):
        matrix = np.zeros((len(rows), 60))
        matrix[:, 0] = rows
        return {"hard_margins": margins, "hard_jacobian_scaled": matrix.tolist(),
                "max_steering_rad": .5}, [[0., .5] for _ in range(30)]

    def test_affine_witness_is_independent_of_racing_hessian(self):
        score, controls = self.problem([1., -1.], [-1., 2.])
        result = qualification.affine_feasibility(score, controls)
        self.assertEqual(result["classification"], "feasible")
        self.assertLessEqual(result["witness_maximum_violation"], 1e-8)

    def test_inconsistent_rows_have_positive_dual_certified_relaxation(self):
        score, controls = self.problem([1., -1.], [-1., 0.])
        result = qualification.affine_feasibility(score, controls)
        self.assertEqual(result["classification"], "infeasible")
        self.assertAlmostEqual(result["minimum_scaled_common_relaxation"], .5)
        self.assertAlmostEqual(result["dual_lower_bound"], .5)
        self.assertFalse(result["physical_unrecoverability_proven"])

    def test_legal_control_bounds_are_part_of_the_lp(self):
        score, controls = self.problem([1.], [-4.])
        self.assertEqual(qualification.affine_feasibility(score, controls)["classification"], "infeasible")


class NativePhase1Qualification(unittest.TestCase):
    def test_entire_corpus_is_checked_but_failed_recovery_never_passes(self):
        tools = Path(os.environ.get("MPC_SOURCE_BENCHMARK_TOOLS", benchmark.DEFAULT_TOOLS))
        names = ("mpc_source_live_fixture", "mpc_source_exact_offline_optimize",
                 "mpc_source_feasibility_witness_probe", "mpc_source_nlp_oracle")
        if not all((tools / name).is_file() for name in names):
            if os.environ.get("MPC_SOURCE_BENCHMARK_REQUIRE_NATIVE"):
                self.fail("Build original-source native qualification tools")
            self.skipTest("Native source tools unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            # A single iteration deliberately leaves known failures. The
            # replay-consistency gate may pass; Phase 1 must still fail.
            result = qualification.qualify(benchmark.DEFAULT_BUNDLE, output, tools,
                iterations=1, repeat=2, perturbations=1)
            self.assertEqual(len(result["epochs"]), 22)
            self.assertEqual(result["kkt_cases"], 0)
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["perturbations"]), 5)
            self.assertFalse(result["perturbations_qualified"])
            self.assertFalse(result["feasibility_qualified"])
            self.assertFalse(result["witness_initialization"])
            self.assertFalse(result["physical_unrecoverability_proven"])
            self.assertIsNone(result["wall_clock_deadline"])
            self.assertTrue(all(row["deterministic"] for row in result["epochs"]))
            for row in result["epochs"]:
                case = output / f"epoch_{row['record_index']}"
                check = benchmark.read_json(case / "phase1_qualification.json.gz")
                self.assertTrue(check["recorded_warm_start_verified"])
                self.assertEqual(len(check["iterations"]), 1)
                self.assertGreater(check["oracle_evaluations"], 0)
                for iteration in check["iterations"]:
                    self.assertTrue(iteration["checks"]["racing_cost_parity"])
                    self.assertTrue(iteration["checks"]["physical_violation_parity"])
                    self.assertTrue(iteration["checks"]["accepted_step_parity"])
                    self.assertGreater(iteration["exact_nonlinear_evaluations"], 0)
            # Validate the whole condensed directional source model on a
            # corpus-derived fixture, so this proof needs no ignored /tmp
            # snapshot. Clearance metrics must exclude rate/box row units.
            from tools.audit_mpc_source_linearization_domain import audit
            domain = audit(output/"epoch_775", tools, 0, [.001])
            self.assertTrue(domain["path_curvature_selected_source_branch_valid"])
            self.assertLess(domain["path_curvature_first_derivative_agreement_inf"],1e-5)
            self.assertLess(domain["objective_directional_gradient_agreement_abs"]/
                max(1.,abs(domain["original_objective_directional_first"])),1e-6)
            small = next(row for row in domain["fractions"] if row["fraction"] == .001)
            self.assertLess(small["maximum_absolute_quadratic_constraint_model_error_m"],
                            small["maximum_absolute_constraint_model_error_m"])
            self.assertLess(abs(small["quadratic_racing_cost_error"]),
                            abs(small["affine_racing_cost_error"]))
            oracle = qualification.SourceOracle(tools, output/"epoch_775")
            try:
                controls = np.loadtxt(output/"epoch_775/initial.controls").tolist()
                original = oracle.evaluate(controls)
                self.assertEqual(original["objective"],original["original_cost"])
                rejected = oracle.evaluate(controls, preferred_clearance_weight=1e6)
                self.assertFalse(rejected["valid"])
                self.assertEqual(rejected["reason"],"extra_reserve_is_not_original_objective")
            finally:
                oracle.close()
            # The full source reference must agree with independent
            # gradient differences, including steering/throttle coupling.
            # Polarization symmetry alone cannot establish correctness.
            from tools.audit_mpc_source_lagrangian import audit as audit_lagrangian
            lagrangian = audit_lagrangian(output/"epoch_769", tools)
            self.assertTrue(lagrangian["analytical_hessian_valid"])
            self.assertTrue(lagrangian["passed"])
            self.assertTrue(all(check["relative_error"] < .05
                                for check in lagrangian["hessian_vector_checks"]))
            # Comparing with an actual, immutable prior result must leave
            # that report intact and compare each original racing cost.
            baseline = output / "phase1_report.json"
            before = baseline.read_bytes()
            other = qualification.qualify(benchmark.DEFAULT_BUNDLE, output / "repeat", tools,
                iterations=1, repeat=2, baseline_report=baseline, perturbations=1, jobs=2)
            self.assertEqual(baseline.read_bytes(), before)
            self.assertTrue(all(row["baseline_racing_cost_delta"] == 0. for row in other["epochs"]))
            self.assertEqual(other["perturbations"],result["perturbations"])


if __name__ == "__main__":
    unittest.main()
