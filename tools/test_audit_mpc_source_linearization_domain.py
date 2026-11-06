"""Original source hard-constraint Taylor model and branch-identity regression.

Requires a real captured source fixture, never a made-up surrogate plant.
"""
from pathlib import Path
import os
import unittest
import numpy as np

from tools.audit_mpc_source_linearization_domain import audit, path_error_decomposition

ROOT=Path(__file__).resolve().parents[1]


class ConstraintPathIdentityTest(unittest.TestCase):
    def test_smooth_quadratic_is_all_curvature(self):
        t=np.linspace(0.,1.,9)
        parts=path_error_decomposition(t,(2.+3.*t+4.*t*t)[:,None],
            (3.+8.*t)[:,None],np.full((len(t),1),8.),[""]*8)
        np.testing.assert_allclose(parts["smooth_curvature_integral"],[4.],atol=1e-14)
        np.testing.assert_allclose(sum(parts["estimated_switch_and_unresolved_contributions"].values()),[0.],atol=1e-14)
        self.assertLess(parts["identity_closure_inf"],1e-14)

    def test_continuous_kink_has_derivative_jump(self):
        t=np.linspace(0.,1.,4)
        parts=path_error_decomposition(t,np.maximum(0.,t-.5)[:,None],
            (t>.5).astype(float)[:,None],np.zeros((4,1)),["","tire_slip",""])
        np.testing.assert_allclose(parts["smooth_curvature_integral"],[0.])
        np.testing.assert_allclose(parts["estimated_switch_and_unresolved_contributions"]["tire_slip"],[.5])
        self.assertLess(parts["identity_closure_inf"],1e-14)

    def test_discrete_jump_is_not_smooth_curvature(self):
        t=np.linspace(0.,1.,4)
        parts=path_error_decomposition(t,(t>.5).astype(float)[:,None],
            np.zeros((4,1)),np.zeros((4,1)),["","substep_schedule",""])
        np.testing.assert_allclose(parts["smooth_curvature_integral"],[0.])
        np.testing.assert_allclose(parts["estimated_switch_and_unresolved_contributions"]["substep_schedule"],[1.])

    def test_smooth_cubic_unresolved_term_shrinks_on_refinement(self):
        def evaluate(n):
            t=np.linspace(0.,1.,n+1)
            return path_error_decomposition(t,(t**3)[:,None],(3.*t*t)[:,None],
                (6.*t)[:,None],[""]*n)
        coarse,fine=evaluate(16),evaluate(32)
        self.assertLess(abs(fine["smooth_curvature_integral"][0]-1.),
                        abs(coarse["smooth_curvature_integral"][0]-1.))
        self.assertLess(fine["identity_closure_inf"],1e-14)


class SourceCurvatureDomainAuditTest(unittest.TestCase):
    def test_real_source_second_order_never_certified_across_switches(self):
        path=Path(os.environ.get(
            "MPC_SOURCE_CURVATURE_RECORDED_CASE",
            str(ROOT/"artifacts/competition_phase1_verified/epoch_775")))
        tools=Path(os.environ.get("MPC_SOURCE_CURVATURE_TOOLS",str(ROOT/"build/f1tenth_mpc")))
        if not (path/"replay.json.gz").is_file() or not (
                tools/"mpc_source_nlp_oracle").is_file():
            self.skipTest("No native original source fixture and oracle available")
        report=audit(path,tools,0,[.001,.01,.1])
        self.assertEqual(report["original_hard_inequality_count"],480)
        self.assertTrue(report["path_curvature_selected_source_branch_valid"])
        self.assertLess(report["path_curvature_first_derivative_agreement_inf"],1e-4)
        full=report["fractions"][-1]
        self.assertFalse(full["source_branch_identity_preserved_to_candidate"])
        self.assertFalse(full["second_order_prediction_qualified_at_candidate"])
        self.assertGreater(full["source_substep_schedule_changed_events"],0)
        self.assertGreater(
            full["maximum_absolute_quadratic_constraint_model_error_m"],0.)
        self.assertFalse(report["physical_dynamics_modified"])
        self.assertFalse(report["alternative_controls_generated"])


if __name__=="__main__":
    unittest.main()
