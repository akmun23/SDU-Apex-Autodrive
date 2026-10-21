import unittest

from tools.refine_mpc_qp_oracle_cases import select_refinement_rows


def row(case_id, fixture, converged, feasibility, primal, dual):
    return {
        "case_id": case_id,
        "fixture": fixture,
        "oracle_converged": converged,
        "oracle_feasibility_lp_status": feasibility,
        "oracle_primal_residual": primal,
        "oracle_dual_residual": dual,
    }


class SelectRefinementRowsTest(unittest.TestCase):
    def test_all_unresolved_preserves_all_supported_cases(self):
        rows = [
            row("b", "03.qp", False, None, 2.0, 1.0),
            row("a", "02.qp", False, "feasible", 0.2, 0.1),
            row("a", "01.qp", False, None, 1.0, 0.5),
            row("a", "04.qp", True, None, 0.0, 0.0),
            row("a", "05.qp", False, "infeasible", 9.0, 9.0),
        ]
        selected = select_refinement_rows(rows, all_unresolved=True)
        self.assertEqual(
            [(item["case_id"], item["fixture"]) for item in selected],
            [("a", "01.qp"), ("a", "02.qp"), ("b", "03.qp")],
        )

    def test_default_keeps_representative_screen(self):
        rows = [row("a", f"{i:02}.qp", False, None, float(i), 0.0)
                for i in range(5)]
        selected = select_refinement_rows(rows, all_unresolved=False)
        self.assertEqual([item["fixture"] for item in selected],
                         ["04.qp", "02.qp"])


if __name__ == "__main__":
    unittest.main()
