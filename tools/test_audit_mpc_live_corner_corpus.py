import unittest
from tools.audit_mpc_live_corner_corpus import classify

def case(stage, passes):
    return {"nonlinear_failure_stage": stage, "passes": passes}

def qp(feasible, status):
    return {"linear_feasible": feasible, "raw_status": status}

class TestMpcLiveCornerClassification(unittest.TestCase):
    def test_current_state_violation_is_distinct_from_qp_infeasibility(self):
        self.assertEqual(classify(case(0, {"r1": qp(False, 1)})),
                         "current_epoch_corridor_violation")

    def test_independent_linearized_infeasibility(self):
        self.assertEqual(classify(case(2, {"r1": qp(False, 1),
                                            "r2": qp(False, 1)})),
                         "infeasible_local_linearization")
        self.assertEqual(classify(case(2, {"r1": qp(True, 1),
                                            "r2": qp(False, 1)})),
                         "some_passes_linearly_infeasible")

    def test_inexact_convergence_not_conflated_with_physical_viability(self):
        self.assertEqual(classify(case(19, {"r1": qp(True, 1),
                                             "r2": qp(True, 0)})),
                         "solvable_linearization_but_admm_not_converged")
        self.assertEqual(classify(case(19, {"r1": qp(True, 0)})),
                         "linearized_qp_solved_but_nonlinear_rollout_rejected")
        self.assertEqual(classify(case(19, {})), "unclassified")

if __name__ == "__main__":
    unittest.main()
