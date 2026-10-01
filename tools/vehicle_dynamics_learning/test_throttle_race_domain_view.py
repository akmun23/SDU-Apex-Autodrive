import unittest

import numpy as np

from tools.vehicle_dynamics_learning.build_throttle_race_domain_view import (
    _retained_prefix,
)


class ThrottleRaceDomainViewTest(unittest.TestCase):
    def test_first_over_cap_sample_is_excluded_and_prefix_preserved(self):
        speed = np.asarray([0.0, 8.0, 11.9, 12.0, 12.01, 8.0])
        valid = np.ones(len(speed), dtype=bool)
        stop, reason = _retained_prefix(speed, valid)
        self.assertEqual(stop, 4)
        self.assertEqual(reason, "speed_cap")

    def test_invalid_sample_stops_sequence_without_reordering_or_skipping(self):
        speed = np.asarray([2.0, 3.0, 4.0, 5.0])
        valid = np.asarray([True, True, False, True])
        stop, reason = _retained_prefix(speed, valid)
        self.assertEqual(stop, 2)
        self.assertEqual(reason, "invalid_sample")

    def test_exact_12_mps_sample_is_inside_the_race_domain(self):
        stop, reason = _retained_prefix(
            np.asarray([11.9, 12.0]), np.asarray([True, True]))
        self.assertEqual((stop, reason), (2, "source_end"))


if __name__ == "__main__":
    unittest.main()
