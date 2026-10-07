#!/usr/bin/env python3
"""Checks that holdout joins and scores exclude prepended history context."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_yaw_atlas_openplane_holdout import (
    _expected_atlas_labels,
    _phase_window_packet_ids,
)


class YawHoldoutWindowTest(unittest.TestCase):
    def test_packet_join_window_excludes_negative_phase_history(self) -> None:
        samples = (
            SimpleNamespace(packet_sequence=10, time_s=-0.50),
            SimpleNamespace(packet_sequence=11, time_s=-0.025),
            SimpleNamespace(packet_sequence=12, time_s=0.0),
            SimpleNamespace(packet_sequence=13, time_s=0.025),
            SimpleNamespace(packet_sequence=-1, time_s=0.05),
        )
        self.assertEqual(_phase_window_packet_ids(samples), {12, 13})

    def test_highsteer_final_matrix_has_all_signed_repeats(self) -> None:
        labels = _expected_atlas_labels(
            "openplane_yaw_atlas_extratrees_highsteer_final_20261007_r01")
        self.assertEqual(len(labels), 16)
        self.assertIn("atlas_r02_v8.62_a0.4270_turn-1", labels)
        self.assertIn("atlas_r01_v9.12_a0.3020_turn+1", labels)

    def test_lowangle_final_matrix_remains_24_probes(self) -> None:
        labels = _expected_atlas_labels(
            "openplane_yaw_atlas_extratrees_final_20261007_r01")
        self.assertEqual(len(labels), 24)
        self.assertIn("atlas_r01_v4.62_a0.1080_turn-1", labels)


if __name__ == "__main__":
    unittest.main()
