from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tools.racing.offline.controller.production_mpc import ProductionMpc


REPO = Path(__file__).resolve().parents[4]
OUTPUTS = (
    "f1tenth_mpc/config/mpc_competition.yaml",
    "f1tenth_planning/trajectories/autodrive_practice_20260924_b/"
    "autodrive_practice_20260924_b_mintime_raceline.csv",
)


class ProductionMpcTest(unittest.TestCase):
    def test_production_config_trajectory_and_cycle(self):
        library = Path(
            "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
            "production_mpc_build/libproduction_mpc.so")
        self.assertTrue(library.is_file(),
                        "build the production MPC shim before running this test")
        config, trajectory = (REPO / relative for relative in OUTPUTS)
        with ProductionMpc(library, config, trajectory) as controller:
            self.assertGreater(controller.lap_length_m, 1.0)
            projection = controller.project(np.asarray((-0.727576, -6.459126,
                                                        -2.424012)))
            self.assertLess(projection[3], 0.01)
            state = np.asarray((
                0.0, 0.0, 2.72, 0.0, -1.0, 2.72,
                -0.2, -0.2, -0.2, -0.2, 0.0, 0.0,
            ))
            result = controller.step(state, 0.0, 16.0)
            self.assertGreaterEqual(result.status, 0)
            self.assertTrue(np.isfinite((
                result.steering_command_rad, result.target_speed_mps,
                result.primal_residual, result.dual_residual,
            )).all())
            controller.reset()
            repeated = controller.step(state, 0.0, 16.0)
            self.assertTrue(np.isfinite(repeated.target_speed_mps))


if __name__ == "__main__":
    unittest.main()
