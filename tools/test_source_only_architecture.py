"""Prevent reintroducing retired MPC numerics, fallbacks or alpha sampling."""
from pathlib import Path
import re
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MPC = ROOT / "f1tenth_mpc"


class SourceOnlyArchitecture(unittest.TestCase):
    def test_single_numerical_library(self):
        cm = (MPC / "CMakeLists.txt").read_text()
        sources = cm.split("add_library(mpc_core STATIC", 1)[1].split(")", 1)[0]
        forbidden = (
            "riccati_", "mpc_rti_qp.cpp", "mpc_linearization.cpp",
            "mpc_source_reduced_sqp.cpp", "mpc_source_nonlinear_optimizer.cpp",
            "mpc_terminal_viability.cpp", "mpc_successor_viability.cpp",
            "multiple_shooting", "mpc_source_first_step_viability.cpp",
            "mpc_controller_recovery.cpp",
        )
        for term in forbidden:
            with self.subTest(retired=term):
                self.assertNotIn(term, sources)
        self.assertIn("mpc_source_exact_offline_solver.cpp", sources)
        self.assertIn("mpc_source_exact_qp.cpp", sources)

    def test_no_legacy_solver_files(self):
        for name in (
            "mpc_rti_qp.cpp", "mpc_source_reduced_sqp.cpp",
            "mpc_source_nonlinear_optimizer.cpp",
            "riccati_solver.cpp", "riccati_factorization.cpp",
            "mpc_controller_recovery.cpp", "mpc_source_exact_nlp_terminal_optimize.cpp",
        ):
            with self.subTest(name=name):
                self.assertFalse((MPC / "src" / name).exists())

    def test_no_dead_riccati_interfaces(self):
        self.assertFalse((MPC / "include/riccati_solver.h").exists())
        self.assertFalse((MPC / "include/mpc_qp_fixture_format.hpp").exists())
        for source in ("include/mpc_types.h", "src/mpc_rti_internal.hpp",
                       "src/mpc_rti.cpp"):
            text=(MPC / source).read_text()
            with self.subTest(source=source):
                self.assertNotIn("RiccatiAdmmConfig_t",text)
                self.assertNotIn("RiccatiStepData_t",text)
                self.assertNotIn("RiccatiAdmmState_t",text)
        cycle=(MPC / "include/mpc_rti.h").read_text().split(
            "} MpcRtiCycleConfiguration_t;",1)[0].split(
            "typedef struct\n{\n    MpcRtiConfiguration_t model;",1)[-1]
        for obsolete in ("native_source_reduced_sqp",
                         "source_exact_nlp_parallel_trials",
                         "source_exact_nlp_max_trials_per_iteration",
                         "enable_nonlinear_line_search",
                         "max_sqp_iterations"):
            with self.subTest(deleted_mode=obsolete):
                self.assertNotIn(obsolete,cycle)

    def test_one_full_nonlinear_step_per_iteration(self):
        code = (MPC / "src/mpc_source_exact_offline_solver.cpp").read_text()
        self.assertIn("source_exact_constrained_qp_direction(", code)
        self.assertIn("exact_source_rollout(", code)
        self.assertIn("qp->direction[k][0]", code)
        self.assertNotIn("kQpAlphaOrder", code)
        self.assertNotIn("trial_alpha(", code)
        self.assertNotIn("parallel_trials", code)
        self.assertNotIn("prioritize_source_qp_trial_scales", code)
        self.assertNotIn("maximum_source_trials_per_iteration", code)

    def test_runtime_no_alternate_control_publication(self):
        code = (MPC / "src/mpc_controller_node_impl.hpp").read_text()
        self.assertIn("mpc_rti_solve_cycle(", code)
        self.assertIn("MPC_RTI_CYCLE_ACCEPTED_FEASIBLE", code)
        self.assertNotIn("try_publish_prior_plan(", code)
        self.assertNotIn("try_publish_candidate_prefix(", code)
        self.assertNotIn("try_publish_viability_recovery(", code)
        self.assertNotIn("declare_parameter<bool>(\"native_source_reduced_sqp\"", code)
        self.assertNotIn("declare_parameter<bool>(\"source_exact_nlp_live\"", code)

    def test_source_input_envelope_initialized_after_legacy_cleanup(self):
        node=(MPC / "src/mpc_controller_node_impl.hpp").read_text()
        validation=(MPC / "src/mpc_rti.cpp").read_text()
        self.assertIn("rti_config_.model.max_steering_command_step_rad =", node)
        self.assertIn("2.f * rti_config_.model.max_steering_rad;", node)
        self.assertIn("rti_config_.model.max_throttle_command_step =", node)
        self.assertIn("MPC_THROTTLE_MAX - MPC_THROTTLE_MIN_DRIVE;", node)
        self.assertIn(
            "configuration->max_steering_command_step_rad > 0.f", validation)
        self.assertIn(
            "configuration->max_throttle_command_step > 0.f", validation)

    def test_profiles_have_no_retired_settings(self):
        forbidden = (
            "admm_", "fallback_", "viability_recovery",
            "line_search", "multiple_shooting", "command_increment_qp",
            "feasibility_restoration", "quality_prefix_",
            "source_exact_nlp_max_trials",
        )
        for name in ("mpc_practice.yaml", "mpc_competition.yaml"):
            params = yaml.safe_load((MPC / "config" / name).read_text())[
                "/**"]["ros__parameters"]
            self.assertLessEqual(params["end_to_end_cycle_budget_ms"], 22.)
            self.assertLessEqual(params["rti_cycle_budget_ms"], 22.)
            self.assertEqual(params["source_exact_nlp_extra_clearance_m"], .03)
            for key in params:
                with self.subTest(profile=name, param=key):
                    self.assertFalse(any(s in key for s in forbidden), key)


if __name__ == "__main__":
    unittest.main()
