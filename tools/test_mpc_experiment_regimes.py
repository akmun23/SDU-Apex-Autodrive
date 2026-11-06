#!/usr/bin/env python3
"""Regression tests for keeping MPC measurement regimes distinct."""

from __future__ import annotations

import unittest

from tools.benchmark_mpc_replay import experiment_regime_for_memory_mode


class MpcExperimentRegimeTest(unittest.TestCase):
    def test_replay_memory_modes_are_not_live_closed_loop(self) -> None:
        self.assertEqual(experiment_regime_for_memory_mode("sequence"),
                         "sequence_replay")
        self.assertEqual(experiment_regime_for_memory_mode("independent"),
                         "identical_source_state")
        self.assertEqual(experiment_regime_for_memory_mode("nominal-cold"),
                         "identical_source_state")

    def test_unknown_mode_is_not_silently_mislabeled(self) -> None:
        with self.assertRaises(ValueError):
            experiment_regime_for_memory_mode("live")


if __name__ == "__main__":
    unittest.main()
