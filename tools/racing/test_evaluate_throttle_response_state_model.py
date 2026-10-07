import unittest

import numpy as np

from tools.racing.evaluate_throttle_response_state_model import (
    _kernel_fit_predict,
    _phase_inputs,
)


class ThrottleResponseStateModelTest(unittest.TestCase):
    def test_phase_inputs_are_current_and_pre_stimulus_only(self):
        phase = {
            "stimulus_state": {
                "speed_mps": 9.2,
                "steering_feedback_rad": 0.15,
                "vy_mps": -0.2,
                "yaw_rate_rps": 0.7,
                "throttle_feedback_norm": 0.41,
            },
            "pre_window": {"median": {
                "mean_abs_rear_wheel_residual_mps": 0.12,
                "common_rear_wheel_residual_mps": -0.03,
                "rear_wheel_residual_asymmetry_mps": 0.04,
                "abs_rigid_body_lateral_acceleration_mps2": 6.2,
                "abs_imu_roll_rad": 0.03,
                "imu_roll_rad": -0.03,
                "imu_roll_rate_rps": 0.02,
                "abs_imu_roll_rate_rps": 0.02,
            }},
            "post_stimulus_sensor_value_that_must_not_be_used": 999.0,
        }
        inputs = _phase_inputs(phase, target_speed=9.0, turn_sign=1.0)
        self.assertAlmostEqual(inputs["speed_error_mps"], 0.2)
        self.assertAlmostEqual(inputs["turn_v_mps"], -0.2)
        self.assertAlmostEqual(inputs["turn_yaw_rate_rps"], 0.7)
        self.assertAlmostEqual(inputs["wheel_residual_abs_mps"], 0.12)
        self.assertNotIn(
            "post_stimulus_sensor_value_that_must_not_be_used", inputs)

    def test_rbf_kernel_predicts_a_nonlinear_response_surface(self):
        train_x = np.linspace(-2.0, 2.0, 41).reshape(-1, 1)
        train_y = np.square(train_x[:, 0])
        query_x = np.asarray([[-1.75], [-0.5], [0.25], [1.4]])
        prediction = _kernel_fit_predict(
            train_x, train_y, query_x, length_scale=0.55, alpha=1.0e-7)
        np.testing.assert_allclose(prediction, np.square(query_x[:, 0]),
                                   atol=0.015, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
