import unittest
from pathlib import Path

from tools.watch_sim_run import lap_target_satisfied, stall_watch_armed


class MpcViabilityStatusTest(unittest.TestCase):
    def test_new_full_horizon_recovery_is_accepted_by_supervisor(self):
        import json
        from types import SimpleNamespace
        from tools.watch_sim_run import SimRunWatcher
        watcher = SimpleNamespace(mpc_fault_reason=None, mpc_active=False,
                                  last_progress_time=0.0)
        message = SimpleNamespace(data=json.dumps(
            {"status": "accepted_viability_recovery"}))
        SimRunWatcher._mpc_diagnostics_callback(watcher, message)
        self.assertIsNone(watcher.mpc_fault_reason)
        self.assertTrue(watcher.mpc_active)


class LapTargetDistanceGuardTest(unittest.TestCase):
    def test_counter_alone_is_not_a_completed_lap(self):
        self.assertFalse(lap_target_satisfied(2, 8, 6, 14.8, 20.0))

    def test_requires_counter_and_minimum_route_distance(self):
        self.assertFalse(lap_target_satisfied(2, 7, 6, 200.0, 20.0))
        self.assertFalse(lap_target_satisfied(2, 8, 6, 119.99, 20.0))
        self.assertTrue(lap_target_satisfied(2, 8, 6, 120.0, 20.0))

    def test_zero_distance_threshold_preserves_counter_only_mode(self):
        self.assertTrue(lap_target_satisfied(0, 5, 5, 0.0, 0.0))


class MpcStartupStallGuardTest(unittest.TestCase):
    def test_mpc_warmup_does_not_count_as_stalled_motion(self):
        self.assertFalse(stall_watch_armed(True, False))

    def test_mpc_motion_watch_arms_after_accepted_cycle(self):
        self.assertTrue(stall_watch_armed(True, True))

    def test_non_mpc_watch_retains_existing_stall_behavior(self):
        self.assertTrue(stall_watch_armed(False, False))


if __name__ == "__main__":
    unittest.main()
