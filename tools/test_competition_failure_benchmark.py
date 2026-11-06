"""Recorded competition oracle, chronology, cause classification and replay."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools import competition_failure_benchmark as benchmark


class CompetitionFailureBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = benchmark.DEFAULT_BUNDLE
        cls.sequence = benchmark.read_json(cls.bundle/"sequence.json.gz")

    def test_recorded_episode_and_command_chronology(self):
        s = self.sequence
        self.assertEqual(len(s["epochs"]), 22)
        self.assertEqual(s["roles"], {"last_feasible": 756, "first_infeasible": 757,
                                     "terminal_failure": 769, "last_recorded": 775})
        self.assertEqual(s["maximum_no_publication_streak"], 19)
        # Progress is slightly nonmonotone across different command leads.
        self.assertGreater(s["epochs"][3]["diagnostic"]["progress_m"],
                           s["epochs"][4]["diagnostic"]["progress_m"])
        for e in s["epochs"]:
            d = e["diagnostic"]
            self.assertEqual(len(d["plant_internal_state"]), 22)
            self.assertEqual(len(d["state"]), 10)
            self.assertEqual(len(d["nominal_controls"]), 30)
            self.assertEqual(d["plant_state_flags"], [1, 1, 1])
            self.assertTrue(all(u[1] >= .001 for u in d["nominal_controls"]))

    def test_deadline_and_feasibility_are_independent(self):
        d = self.sequence["epochs"][3]["diagnostic"]
        self.assertIn("deadline_exceeded", benchmark.classify(d))
        self.assertIn("recorded_iterate_primal_infeasible", benchmark.classify(d))
        self.assertNotIn("feasible_solution_available", benchmark.classify(d))

    def test_oracle_parity_never_mixes_initial_and_final_iterates(self):
        d = self.sequence["epochs"][0]["diagnostic"]
        s = d["solver"]
        seed = {"initial_cost": s["source_exact_nlp_original_cost_initial"]}
        check = benchmark.parity(seed, d)
        self.assertTrue(check["passed"])
        self.assertFalse(check["recorded_final_primal_available"])
        seed["recorded_final_score"] = {
            "cost": s["source_exact_nlp_original_cost_initial"],
            "minimum_clearance_m": s["source_exact_nlp_original_minimum_corner_margin_m"],
            "elastic_violation": s["source_exact_nlp_original_constraint_violation"]}
        self.assertFalse(benchmark.parity(seed, d)["passed"])

    def test_bundle_hashes_and_baseline_are_frozen(self):
        manifest = benchmark.read_json(self.bundle/"manifest.json")
        for name, expected in manifest["bundle_sha256"].items():
            self.assertEqual(benchmark.sha(self.bundle/name), expected, name)
        baseline = benchmark.read_json(self.bundle/"oracle_baseline.json.gz")
        self.assertEqual(len(baseline["epochs"]), 22)
        self.assertEqual(baseline["primal_trace_origin"],
                         "offline_reconstruction_not_historical_live_iterates")
        for epoch in baseline["epochs"].values():
            self.assertEqual(len(epoch["frozen_problem"]["references"]), 31)
            self.assertFalse(epoch["frozen_problem"]["terminal_speed_viability_active"])

    def test_witnesses_exclude_zero_and_preserve_unresolved_search(self):
        evidence = benchmark.read_json(self.bundle/"investigation.json.gz")
        searches = evidence["all_recorded_epoch_reachability"]["searches"]
        self.assertEqual(len(searches), 22)
        self.assertEqual(sum(r["best_minimum_clearance_m"] >= 0. for r in searches), 21)
        self.assertEqual(searches[-1]["record_index"], 775)
        self.assertLess(searches[-1]["best_minimum_clearance_m"], 0.)
        self.assertFalse(searches[-1]["infeasibility_proven"])
        for r in evidence["admissible_reachability"]["searches"]:
            self.assertFalse(r["brake_branch_searched"])
            self.assertTrue(all(.001 <= u[1] <= 1. for u in r["controls"]))

    def test_default_filter_sqp_uses_original_objective_and_physics(self):
        # Primary recorded controls, one default SQP, independent source
        # verification. Preferred reserve changes must not change J or U.
        import json
        tools = Path(os.environ.get(
            "MPC_SOURCE_BENCHMARK_TOOLS", benchmark.DEFAULT_TOOLS))
        names = ("mpc_source_live_fixture",
                 "mpc_source_exact_offline_optimize",
                 "mpc_source_feasibility_witness_probe",
                 "mpc_source_nlp_oracle")
        if not all((tools / name).is_file() for name in names):
            if os.environ.get("MPC_SOURCE_BENCHMARK_REQUIRE_NATIVE"):
                self.fail("Build original-source oracle and benchmark tools")
            self.skipTest("Native competition benchmark tools unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            # Independently reconstruct source state, reference and exact
            # physics from the tracked original replay, not an improved seed.
            report = benchmark.replay(
                self.bundle, work, tools, "recorded", 1)
            self.assertTrue(report["passed"])
            env = benchmark.solver_environment(self.sequence["parameters"], controls_only_trace=True)
            for epoch, iterations in ((765, 500), (769, 500)):
                case = work / f"epoch_{epoch}"
                output = case / "original_objective.controls"
                run = subprocess.run(
                    [str(tools / "mpc_source_exact_offline_optimize"),
                     str(case), "-", str(iterations), str(output)],
                    cwd=benchmark.ROOT, env=env, capture_output=True,
                    text=True, timeout=120)
                self.assertEqual(run.returncode, 0, run.stderr[-2000:])
                result = json.loads(run.stdout)
                self.assertTrue(result["source_feasible_final"], epoch)
                if result["original_nlp_kkt_converged"]:
                    self.assertEqual(result["termination_reason"], "original_nlp_kkt")
                    self.assertTrue(result["history"][-1]["original_duals_trustworthy"])
                self.assertTrue(result["alpha_search"])
                self.assertFalse(result["fixed_alpha_grid"])
                self.assertFalse(result["fallback_controller"])
                self.assertEqual(result["nonlinear_acceptance_objective"],"original_racing_cost")
                if epoch == 765:
                    variant = dict(env, MPC_SOURCE_NLP_MIN_CLEARANCE_M="0", MPC_SOURCE_NLP_QP_SOFT_WEIGHT="10")
                    changed = subprocess.run([str(tools / "mpc_source_exact_offline_optimize"),
                        str(case), "-", str(iterations), str(case / "reserve_variant.controls")],
                        env=variant, capture_output=True,text=True,check=True,timeout=120)
                    variant_result = json.loads(changed.stdout)
                    self.assertEqual(output.read_bytes(),(case / "reserve_variant.controls").read_bytes())
                    self.assertEqual(result["final_cost"],variant_result["final_cost"])
                controls = [list(map(float, line.split()))
                            for line in output.read_text().splitlines()
                            if line.strip()]
                self.assertEqual(len(controls), 30)
                oracle = subprocess.run(
                    [str(tools / "mpc_source_nlp_oracle"), str(case)],
                    input=json.dumps({
                        "controls": controls,
                        "derivatives": False}) + "\n",
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(
                    oracle.returncode, 0, oracle.stderr[-2000:])
                original = json.loads(oracle.stdout)
                self.assertTrue(original["valid"])
                self.assertGreaterEqual(
                    original["minimum_clearance_m"], 0.)
                self.assertGreaterEqual(
                    original["minimum_rate_margin"], -1e-7)
                self.assertGreaterEqual(
                    original["minimum_box_margin"], -1e-7)
                self.assertAlmostEqual(
                    original["original_cost"], result["final_cost"],
                    delta=1e-4 * max(1, abs(result["final_cost"])))

    def test_native_recorded_replay_and_live_capture(self):
        tools = Path(os.environ.get("MPC_SOURCE_BENCHMARK_TOOLS", benchmark.DEFAULT_TOOLS))
        required = [tools/name for name in ("mpc_source_live_fixture",
                    "mpc_source_exact_offline_optimize", "mpc_source_feasibility_witness_probe",
                    "mpc_source_benchmark_capture_check")]
        if not all(p.is_file() for p in required):
            if os.environ.get("MPC_SOURCE_BENCHMARK_REQUIRE_NATIVE"):
                self.fail("Build native benchmark tools before running this qualification test")
            self.skipTest("native tools unavailable; CLI replay always fails on missing tools")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = benchmark.replay(self.bundle, output, tools, "recorded", 2)
            self.assertTrue(report["passed"])
            self.assertEqual(report["unresolved_reachability_epochs"], [])
            self.assertEqual(len(report["recorded_witness_regressions"]), 25)
            for index in (756, 757, 769):
                check = subprocess.run([str(required[-1]), str(output/f"epoch_{index}")],
                                       capture_output=True, text=True, timeout=30)
                self.assertEqual(check.returncode, 0, check.stdout+check.stderr)


if __name__ == "__main__":
    unittest.main()
