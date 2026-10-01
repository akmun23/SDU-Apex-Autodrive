import unittest

import numpy as np

from tools.vehicle_dynamics_learning.race_domain_coverage import (
    _command_direction,
    _interval_labels,
)


class RaceDomainCoverageTest(unittest.TestCase):
    def test_speed_bin_edges_are_half_open_except_declared_cap(self):
        edges = (0.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0)
        values = np.asarray([0.0, 3.0, 5.0, 9.0, 11.0, 12.0])
        result = _interval_labels(values, edges, "mps")
        self.assertEqual(result.tolist(), [
            "0-3mps", "3-5mps", "5-7mps", "9-10mps", "11-12mps",
            "11-12mps",
        ])

    def test_zero_throttle_is_active_brake_not_unobserved_coast(self):
        command = np.asarray([0.0, 0.3, 0.6, 0.6, 0.3, -0.1])
        acceleration = np.asarray([-2.0] * len(command))
        bounds = np.asarray([[0, len(command)]])
        result = _command_direction(command, acceleration, bounds)
        self.assertEqual(result.tolist(), [
            "active_brake_zero_throttle", "accelerating", "accelerating",
            "steady_positive_drive", "throttle_reduction",
            "negative_unsupported",
        ])


if __name__ == "__main__":
    unittest.main()
