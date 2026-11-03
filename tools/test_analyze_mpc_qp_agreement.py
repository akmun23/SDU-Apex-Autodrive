#!/usr/bin/env python3
"""Tests for source SQP live nonlinear model-agreement audit."""
import json
import tempfile
import unittest
from pathlib import Path
from analyze_mpc_qp_agreement import report

class AgreementAuditTests(unittest.TestCase):
    def test_full_step_false_safe_rejection_and_cost_error(self):
        rows=[
            dict(status="accepted_feasible",command_published=True,
                 solver=dict(native_source_sqp_enabled=True,
                     native_source_sqp_full_step_model_agreement=[
                         dict(valid=True,predicted_wall_violation_m2=0.,
                              exact_source_wall_violation_m2=.06,
                              maximum_wall_linearization_defect_m=.34,
                              accepted_alpha=.0625),
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
        self.assertEqual(data["cost_model"]["samples"],1)
        self.assertEqual(data["cost_model"]["absolute_cost_error"]["median"],4.)
        self.assertEqual(len(data["worst_disagreement_epochs"]),1)

    def test_uninstrumented_is_not_misclassified_as_agreement(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"old.jsonl"
            p.write_text(json.dumps(dict(status="accepted_feasible",
                solver=dict(native_source_sqp_enabled=True)))+"\n")
            data=report(p)
        self.assertFalse(data["model_agreement_instrumented"])
        self.assertEqual(data["per_pass"]["SQP_1"]["samples"],0)
        self.assertIsNone(data["per_pass"]["SQP_1"]["wall_prediction_abs_error_m2"])

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
