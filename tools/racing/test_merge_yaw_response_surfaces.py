#!/usr/bin/env python3
"""Focused tests for measured yaw-surface combination and grid validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.racing.merge_yaw_response_surfaces import merge_surfaces


class MergeYawSurfacesTest(unittest.TestCase):
    def test_replaces_old_low_speed_rows_and_preserves_high_speed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "low.csv"
            high = root / "high.csv"
            output = root / "merged.csv"
            header = "speed_knot_mps,turn_sign,steering_rad,demand_q,yaw_rate_abs_rps\n"
            low.write_text(
                header + "3.0,-1,0.3,0.9,1.2\n3.0,1,0.3,0.9,1.2\n"
                "3.8,-1,0.3,1.1,1.4\n3.8,1,0.3,1.1,1.4\n",
                encoding="utf-8")
            high.write_text(
                header + "2.9,-1,0.3,0.9,3.0\n2.9,1,0.3,0.9,3.0\n"
                "4.0,-1,0.3,1.2,1.5\n4.0,1,0.3,1.2,1.5\n"
                "5.0,-1,0.3,1.5,1.7\n5.0,1,0.3,1.5,1.7\n",
                encoding="utf-8")
            report = merge_surfaces(
                low, high, output, 3.8, keep_speed_knots={3.0, 4.0, 5.0})
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("2.90000000,-1", text)
            self.assertNotIn("3.80000000,-1", text)
            self.assertIn("4.00000000,-1", text)
            self.assertEqual(report["merged_speed_knots_mps"], [3.0, 4.0, 5.0])
            self.assertEqual(report["row_count"], 6)

    def test_rejects_incomplete_steering_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low = root / "low.csv"
            high = root / "high.csv"
            header = "speed_knot_mps,turn_sign,steering_rad,demand_q,yaw_rate_abs_rps\n"
            low.write_text(header + "3.0,-1,0.3,0.9,1.2\n", encoding="utf-8")
            high.write_text(header + "4.0,1,0.3,1.2,1.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same steering grid"):
                merge_surfaces(low, high, root / "out.csv", 3.0)


if __name__ == "__main__":
    unittest.main()
