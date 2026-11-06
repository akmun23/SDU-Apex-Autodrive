#!/usr/bin/env python3
"""Tests for source SQP live nonlinear model-agreement audit."""
import json
import tempfile
import unittest
from pathlib import Path
from analyze_mpc_qp_agreement import report, accepted_cost_precision_gate

class AgreementAuditTests(unittest.TestCase):
    def test_current_source_iterations_compare_same_controls_only(self):
        full=dict(full_step_evaluated=True,full_step_accepted=True,
                  original_cost_before=10.,full_step_original_cost=7.,
                  qp_predicted_cost_change=-4.,linearization_ms=3.,
                  qp_ms=2.,source_rollout_ms=.5,total_ms=6.)
        corrected=dict(full,full_step_accepted=False,
                       full_step_original_cost=12.,original_cost_after=8.)
        unevaluated=dict(full,full_step_evaluated=False,
                         full_step_accepted=False,full_step_original_cost=0.)
        rows=[dict(status="accepted_feasible",command_published=True,
                   solver=dict(source_iterations=[full,corrected,unevaluated])),
              dict(status="waiting_for_solver_budget",command_published=False,
                   solver=dict(source_iterations=[]))]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"current.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            data=report(p)
        self.assertEqual(data["telemetry_cycles"],1)
        self.assertEqual(data["cost_model"]["samples"],2)
        self.assertEqual(data["cost_model"]["sign_mismatch_count"],1)
        self.assertEqual(data["accepted_cost_model"]["samples"],1)
        self.assertEqual(data["accepted_cost_model"]["absolute_cost_error"]["p99"],1.)
        self.assertEqual(data["iteration_timing"]["total_ms"]["count"],3)
        self.assertIn("not measured Unity cost",data["interpretation"])

    def test_full_step_false_safe_rejection_and_cost_error(self):
        rows=[
            dict(status="accepted_feasible",command_published=True,
                 solver=dict(native_source_sqp_enabled=True,
                     native_source_sqp_full_step_model_agreement=[
                         dict(valid=True,predicted_wall_violation_m2=0.,
                              exact_source_wall_violation_m2=.06,
                              maximum_wall_linearization_defect_m=.34,
                              accepted_alpha=.0625,
                              racing_cost_model_valid=True,
                              qp_predicted_racing_cost_change=-10.,
                              exact_source_racing_cost_change=-4.,
                              racing_cost_prediction_error=6.,
                              accepted_racing_cost_model_valid=True,
                              accepted_qp_predicted_racing_cost_change=-3.,
                              accepted_exact_source_racing_cost_change=-2.,
                              accepted_racing_cost_prediction_error=1.),
                         dict(valid=False)])),
            dict(status="waiting_for_solver_budget",command_published=False,
                 solver=dict(native_source_sqp_enabled=True,
                     native_source_sqp_full_step_model_agreement=[
                         dict(valid=True,predicted_wall_violation_m2=.003,
                              exact_source_wall_violation_m2=.001,
                              maximum_wall_linearization_defect_m=.02,
                              accepted_alpha=0.5),
                         dict(valid=True,predicted_wall_violation_m2=.001,
                              exact_source_wall_violation_m2=.002,
                              maximum_wall_linearization_defect_m=.01,
                              accepted_alpha=0.125)],
                     native_source_sqp_cost_prediction_valid=True,
                     native_source_sqp_predicted_cost_change=-10.,
                     native_source_sqp_actual_cost_change=-6.)),
            dict(status="rejected_nonlinear_rollout",command_published=False,
                 solver=dict(native_source_sqp_enabled=True,
                     native_source_sqp_full_step_model_agreement=[
                         dict(valid=False),dict(valid=False)])),
        ]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"samples.jsonl"
            p.write_text("\n".join(json.dumps({"payload":r}) for r in rows)+"\n")
            data=report(p)
        self.assertEqual(data["cycles"],3)
        self.assertTrue(data["model_agreement_instrumented"])
        self.assertEqual(data["longest_nonpublication_streak"],2)
        self.assertEqual(data["per_pass"]["SQP_1"]["samples"],2)
        self.assertEqual(data["per_pass"]["SQP_1"]["false_safe_count"],1)
        self.assertEqual(data["per_pass"]["SQP_2"]["samples"],1)
        self.assertEqual(data["per_pass"]["SQP_2"]["false_safe_count"],0)
        self.assertEqual(data["cost_model"]["samples"],2)
        self.assertEqual(data["cost_model"]["absolute_cost_error"]["median"],5.)
        self.assertEqual(data["per_pass"]["SQP_1"]["racing_cost_prediction_samples"],1)
        self.assertEqual(data["per_pass"]["SQP_2"]["racing_cost_prediction_samples"],0)
        self.assertEqual(data["accepted_cost_model"]["samples"],1)
        self.assertEqual(data["accepted_cost_model"]["absolute_cost_error"]["median"],1.)
        self.assertEqual(data["accepted_cost_model"]["absolute_cost_error"]["p99"],1.)
        self.assertEqual(data["per_pass"]["SQP_1"][
            "accepted_racing_cost_prediction_abs_error"]["median"],1.)
        self.assertEqual(len(data["worst_disagreement_epochs"]),1)

    def test_precision_gate_rejects_missing_insufficient_and_bad_model(self):
        summary={"accepted_cost_model":{"absolute_cost_error":None}}
        self.assertFalse(accepted_cost_precision_gate(
            summary,max_p99=100.,minimum_samples=1)[0])
        summary["accepted_cost_model"]["absolute_cost_error"]={
            "count":58,"p99":56.03903,
        }
        self.assertFalse(accepted_cost_precision_gate(
            summary,max_p99=100.,minimum_samples=100)[0])
        self.assertTrue(accepted_cost_precision_gate(
            summary,max_p99=100.,minimum_samples=58)[0])
        summary["accepted_cost_model"]["absolute_cost_error"]["p99"]=101.
        self.assertFalse(accepted_cost_precision_gate(
            summary,max_p99=100.,minimum_samples=58)[0])

    def test_uninstrumented_is_not_misclassified_as_agreement(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"old.jsonl"
            p.write_text(json.dumps(dict(status="accepted_feasible",
                solver=dict(native_source_sqp_enabled=True)))+"\n")
            data=report(p)
        self.assertFalse(data["model_agreement_instrumented"])
        self.assertEqual(data["per_pass"]["SQP_1"]["samples"],0)
        self.assertIsNone(data["per_pass"]["SQP_1"]["wall_prediction_abs_error_m2"])
        self.assertIsNone(data["accepted_cost_model"]["absolute_cost_error"])

    def test_rejects_invalid_data_and_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"bad.jsonl"
            p.write_text("{not valid}\n")
            with self.assertRaisesRegex(ValueError,"line 1"):
                report(p)
            p.write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"positive thresholds"):
                report(p,defect_threshold_m=-1.)

if __name__=="__main__":
    unittest.main()
