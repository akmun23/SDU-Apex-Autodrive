"""Regression for independent Unity-kinematics impact-like annotations.

These diagnostic thresholds do not declare contact and do not remove any
window from the full unmodified source-vs-Unity scoring.
"""
import unittest

from tools.qualify_source_unity_horizon import impact_like_transitions


class ImpactAnnotationsTest(unittest.TestCase):
    def test_clean_25ms_free_motion_has_no_events(self):
        rows = {
            0: {"u": 6.0, "yaw": 0.0},
            25_000_000: {"u": 5.96, "yaw": 0.05},
            50_000_000: {"u": 5.90, "yaw": 0.10},
        }
        self.assertEqual(impact_like_transitions(rows), [])

    def test_impulsive_braking_or_heading_change_is_retained_as_event(self):
        rows = {
            0: {"u": 5.0, "yaw": 0.01},
            25_000_000: {"u": 0.61, "yaw": 0.75},
            50_000_000: {"u": 0.60, "yaw": 0.76},
        }
        self.assertEqual(impact_like_transitions(rows), [25_000_000])

    def test_irregular_sampling_does_not_mislabel_motion(self):
        rows = {
            0: {"u": 6.0, "yaw": 0.0},
            100_000_000: {"u": 4.0, "yaw": 0.4},
        }
        self.assertEqual(impact_like_transitions(rows), [])


if __name__ == "__main__":
    unittest.main()
