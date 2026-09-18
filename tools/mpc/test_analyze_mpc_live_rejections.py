import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze_mpc_live_rejections as analyzer


class MpcDiagnosticAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "status": "rejected_residual",
            "source_stamp_ns": 1_000_000_000,
            "control_ros_stamp_ns": 1_025_000_000,
            "source_age_s": 0.025,
            "progress_m": 12.5,
            "path_curvature_per_m": -0.2,
            "state": [0.1, -0.2, 6.0, 0.3, -0.5, 5.5, -0.15, 0.4, -2.0],
            "first_action": [0.0, 0.0, -1.2, -8.0],
            "control_time_prediction": {
                "mode": "ct2", "age_s": 0.025, "command_changes_used": 2,
                "command_fallback": False, "source_map_pose": [1.0, 2.0, 0.3],
            },
            "solver": {
                "iterations": 100, "primal_residual": 0.04,
                "dual_residual": 0.02, "max_regularization": 1e-5,
                "regularization_count": 1, "nonsmooth_columns": 28,
                "rho_start": 7.0, "rho_final": 14.0,
                "rho_u_start": 7.0, "rho_u_final": 14.0,
                "rho_change_count": 1, "factorization_count": 3,
                "factorization_time_ns": 200_000,
            },
            "nonlinear_failure_stage": 14,
            "rti_iterations_used": 1,
            "rti2_triggered": False,
        }

    def test_unwraps_recorder_string_and_flattens_telemetry(self):
        raw = json.dumps({"value": json.dumps(self.payload)})
        payload = analyzer._nested_payload(raw)
        self.assertEqual(payload["status"], "rejected_residual")
        row = analyzer.flatten_diagnostic(
            {"event_index": "42", "header_stamp_ns": ""}, payload)
        self.assertEqual(row["event_index"], "42")
        self.assertEqual(row["max_residual"], 0.04)
        self.assertEqual(row["abs_curvature_per_m"], 0.2)
        self.assertEqual(row["lateral_accel_proxy_mps2"], 3.0)
        self.assertEqual(row["operating_mode"], "braking_command")
        self.assertTrue(row["steering_reversal"])
        self.assertEqual(analyzer.rejection_category(row), "solver_residual")

    def test_nested_non_linear_failure_category_is_explicit(self):
        row = {"cycle_status": "rejected_nonlinear_rollout",
               "nonlinear_failure_reason": "corridor"}
        self.assertEqual(analyzer.rejection_category(row), "nonlinear_corridor")

    def test_event_reader_and_reports_classify_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "events.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "event_index", "topic", "header_stamp_ns", "payload_json"])
                writer.writeheader()
                writer.writerow({
                    "event_index": "7", "topic": "/mpc_shadow/diagnostics",
                    "header_stamp_ns": "", "payload_json": json.dumps(
                        {"data": json.dumps(self.payload)})})
                accepted = dict(self.payload, status="accepted_optimal")
                writer.writerow({
                    "event_index": "8", "topic": "/mpc/diagnostics",
                    "header_stamp_ns": "", "payload_json": json.dumps(accepted)})
            rows, malformed = analyzer.read_cycles(source)
            self.assertEqual(malformed, 0)
            self.assertEqual(len(rows), 2)
            report = analyzer.write_outputs(rows, malformed, root / "report")
            self.assertEqual(report["rejected"], 1)
            self.assertEqual(report["rejection_classification_coverage_pct"], 100.0)
            self.assertTrue(report["meets_95pct_classification_gate"])
            with (root / "report" / "mpc_cycles.csv").open(
                    newline="", encoding="utf-8") as stream:
                cycle_rows = list(csv.DictReader(stream))
            self.assertEqual(len(cycle_rows), 2)
            self.assertTrue((root / "report" / "stratified_rates.csv").exists())
            self.assertTrue((root / "report" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
