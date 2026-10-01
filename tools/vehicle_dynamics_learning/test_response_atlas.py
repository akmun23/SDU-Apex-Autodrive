"""Focused invariants for atlas reductions and mirror-parity comparisons."""

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.response_atlas import (
    _finite_row_mean,
    _mirror_symmetry_analysis,
)


class ResponseAtlasTests(unittest.TestCase):
    def test_finite_row_mean_preserves_all_missing_rows_as_nan(self):
        values = np.asarray([[1.0, np.nan], [np.nan, np.nan], [2.0, 4.0]])
        result = _finite_row_mean(values)

        np.testing.assert_allclose(result[[0, 2]], [1.0, 3.0])
        self.assertTrue(np.isnan(result[1]))

    def test_mirror_analysis_uses_even_and_odd_response_parity(self):
        base = {
            "run_id": "run_a",
            "throttle_start_norm": 0.0,
            "throttle_target_norm": 0.5,
            "window_index": 1,
            "delta_u_rear_mps": 1.0,
            "mean_ax_sim_body_mps2": 2.0,
            "mean_wheel_surface_mps": 3.0,
            "mean_wheel_body_mismatch_mps": 0.5,
            "delta_v_com_mps": 0.2,
            "delta_yaw_rate_rps": 0.4,
            "mean_yaw_acceleration_rps2": 0.6,
            "mean_ay_sim_body_mps2": 1.2,
            "mean_roll_rad": 0.1,
        }
        positive = {**base, "steering_command_rad": 0.4}
        negative = {
            **base,
            "steering_command_rad": -0.4,
            "delta_v_com_mps": -0.2,
            "delta_yaw_rate_rps": -0.4,
            "mean_yaw_acceleration_rps2": -0.6,
            "mean_ay_sim_body_mps2": -1.2,
            "mean_roll_rad": -0.1,
        }

        result = _mirror_symmetry_analysis([positive, negative])

        self.assertEqual(result["all_steering"]["matched_pair_count"], 1)
        self.assertEqual(result["high_steering"]["matched_pair_count"], 1)
        for metric in result["all_steering"]["metrics"].values():
            self.assertEqual(metric["macro_mean_relative_parity_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
