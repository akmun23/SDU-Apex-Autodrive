import csv
import tempfile
import unittest
from pathlib import Path

from tools.export_mpc_solver_inputs_from_bag import (
    controls_to_rows, export_control_sequence, export_rows, to_solver_row,
)


class SolverInputExportTests(unittest.TestCase):
    def sample(self):
        return {
            "source_stamp_ns": 123456789,
            "progress_m": 12.5,
            "active_speed_ceiling_mps": 16.0,
            "state": [0.1, -0.2, 3.0, 0.4, 0.5, 0.01, 0.02, 0.03, 0.4, 0.5],
            "plant_internal_state": list(map(float, range(22))),
            "plant_state_flags": [1, 0, 1],
        }

    def test_export_row_is_exactly_the_established_38_column_order(self):
        row = to_solver_row(self.sample())
        self.assertEqual(len(row), 38)
        self.assertEqual(row[:3], [123456789.0, 12.5, 16.0])
        self.assertEqual(row[3:13], [0.1, -0.2, 3.0, 0.4, 0.5, 0.03, 0.01, 0.02, 0.4, 0.5])
        self.assertEqual(row[13:16], [0.0, 0.0, 0.0])
        self.assertEqual(row[16:], list(map(float, range(22))))

    def test_export_records_skipped_rows_and_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inputs.csv"
            bad = {"status": "waiting_for_localization"}
            report = export_rows([self.sample(), bad], output, 0, 1)
            with output.open(newline="") as stream:
                rows = list(csv.reader(line for line in stream
                                       if not line.startswith("#")))
        self.assertEqual(report["solver_rows_exported"], 1)
        self.assertEqual(report["solver_rows_skipped"][0]["diagnostic_index"], 1)
        self.assertEqual(len(rows[0]), 38)
        self.assertEqual(report["source_kind"],
                         "saved_state_reconstruction_not_live_solver_memory")

    def test_control_sequence_export_preserves_all_candidate_stages(self):
        controls = [[(stage - 15) * 0.001, 0.1 + stage * 0.001]
                    for stage in range(30)]
        diag = {"candidate_controls": controls,
                "status": "accepted_candidate_prefix",
                "progress_m": 40.3088661,
                "source_stamp_ns": 123456789,
                "issued_command_stamp_ns": 123456790,
                "command_published": True}
        self.assertEqual(controls_to_rows(diag, "candidate"), controls)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed.csv"
            report = export_control_sequence([diag], output, 0, "candidate")
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0]["stage"], "1")
        self.assertAlmostEqual(float(rows[0]["command_steering_rad"]),
                               controls[0][0])
        self.assertAlmostEqual(float(rows[-1]["command_throttle"]),
                               controls[-1][1])
        self.assertEqual(report["source_kind"],
                         "exact_saved_mpc_diagnostic_sequence")

    def test_control_sequence_rejects_incomplete_or_nonfinite_data(self):
        with self.assertRaisesRegex(ValueError, "30 candidate_controls"):
            controls_to_rows({"candidate_controls": [[0.0, 0.1]]}, "candidate")
        bad = {"nominal_controls": [[0.0, 0.1]] * 29 + [[float("nan"), 0.2]]}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            controls_to_rows(bad, "nominal")


if __name__ == "__main__":
    unittest.main()
