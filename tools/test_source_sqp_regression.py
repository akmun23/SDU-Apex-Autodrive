"""Original-source constrained-SQP regression: never confuse feasible with converged.

Fixtures are source-only snapshots captured on this development host.
When a checkout lacks the ignored live_runs corpus, these tests skip rather
than silently fabricate dynamics or overwrite the established race benchmark.
"""
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT / "build/f1tenth_mpc/mpc_source_exact_offline_optimize"
CORPUS=ROOT / "live_runs/mpc_source_second_order_20261104_01"
SEEDS=ROOT / "tools/source_exact_fixture_seeds"


class SourceSQPRegression(unittest.TestCase):
    def run_case(self, name, steps=6):
        path=CORPUS/name
        controls=SEEDS/(name+".controls")
        if not EXE.is_file() or not (path/"source_context.mctx").is_file() or not controls.is_file():
            self.skipTest("Recorded source-SQP fixture or local tool unavailable")
        env=os.environ.copy()
        env.update(MPC_SOURCE_NLP_MIN_CLEARANCE_M="0.03",
                   MPC_SOURCE_NLP_QP_REGULARIZATION="10000",
                   MPC_SOURCE_NLP_QP_SOFT_WEIGHT="1000000",
                   MPC_SOURCE_NLP_RECORD_PRIMAL_TRACE="1")
        run=subprocess.run([str(EXE),str(path),str(controls),str(steps)],
                           cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(run.returncode,0,run.stderr[-3000:])
        data=json.loads(run.stdout)
        self.assertEqual(data["schema"],"source-exact-single-nlp-sqp/v1")
        self.assertFalse(data["single_full_step_per_sqp"])
        self.assertTrue(data["alpha_search"])
        self.assertFalse(data["fallback_controller"])
        self.assertFalse(data["multiple_shooting"])
        self.assertTrue(data["source_feasible_final"],name)
        self.assertGreaterEqual(data["final_corridor_margin_m"],0.)
        self.assertLess(data["final_cost"],data["initial_cost"])
        for row in data["history"]:
            self.assertLessEqual(row["cost_after"],row["cost_before"])
        self.assertGreater(data["accepted_sqp_iterations"],0)
        self.assertLessEqual(data["accepted_sqp_iterations"],len(data["history"]))
        soc=sum(i["second_order_correction_attempted"] for i in data["history"])
        self.assertLessEqual(data["exact_source_full_step_evaluations"],len(data["history"])+soc+sum(i["safeguarded_evaluations"] for i in data["history"]))
        for iteration in data["history"]:
            self.assertGreaterEqual(iteration["qp_iterations"],1)
            self.assertGreaterEqual(iteration["original_nlp_stationarity_inf"],0)
            self.assertGreaterEqual(iteration["original_nlp_complementarity_inf"],0)
            self.assertGreaterEqual(iteration["original_nlp_inequality_residual_inf"],0)
        # A QP multiplier calculated at U_old cannot certify the final
        # accepted U_new. The solver must clear stale terminal duals
        # until a fresh original-source linearization is available.
        if data["history"] and data["history"][-1]["accepted"]:
            self.assertEqual(data["final_physical_multipliers"], [])
        if data["original_nlp_kkt_converged"]:
            last=data["history"][-1]
            self.assertLessEqual(last["original_nlp_stationarity_inf"],1e-2)
            self.assertLessEqual(last["original_nlp_complementarity_inf"],1e-4)
            self.assertLessEqual(last["original_nlp_inequality_residual_inf"],6e-7)
        return data

    def test_known_corner_16(self):
        data=self.run_case("case_16")
        # A sub-3cm SOFT reserve deficit does not mean wall collision.
        # Case16 initially has positive physical wall clearance but
        # a small elastic reserve violation, which SQP reduces to zero.
        self.assertTrue(data["source_feasible_initial"])
        self.assertGreater(data["initial_constraint_violation"],0.)

    def test_live_fast_17(self):
        data=self.run_case("case_live_fast_17")
        self.assertTrue(data["source_feasible_initial"])


if __name__=="__main__":
    unittest.main()
