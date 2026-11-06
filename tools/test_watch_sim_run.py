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


class SimulationNonfatalMpcDiagnosticTest(unittest.TestCase):
    def test_rejected_corridor_is_recorded_without_disabling_mpc(self):
        import io
        import json
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        from tools.watch_sim_run import SimRunWatcher

        message=SimpleNamespace(data=json.dumps({
            "status":"rejected_nonlinear_rollout",
            "nonlinear_failure_reason":"corridor",
            "nonlinear_failure_stage":25}))
        watcher=SimpleNamespace(
            mpc_fault_reason=None, mpc_active=True,
            continue_after_mpc_rejection=True, mpc_rejection_count=0)
        output=io.StringIO()
        with redirect_stdout(output):
            SimRunWatcher._mpc_diagnostics_callback(watcher,message)
        self.assertEqual(watcher.mpc_rejection_count,1)
        self.assertTrue(watcher.mpc_active)
        self.assertIn("MPC_REJECTION_OBSERVED",output.getvalue())
        self.assertNotIn("SCREEN_ABORT",output.getvalue())

    def test_default_supervisor_remains_fatal_on_rejection(self):
        import io
        import json
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        from tools.watch_sim_run import SimRunWatcher
        watcher=SimpleNamespace(
            mpc_fault_reason=None,mpc_active=False,
            continue_after_mpc_rejection=False,mpc_rejection_count=0)
        message=SimpleNamespace(data=json.dumps({
            "status":"rejected_nonlinear_rollout",
            "nonlinear_failure_reason":"corridor",
            "nonlinear_failure_stage":4}))
        with redirect_stdout(io.StringIO()):
            SimRunWatcher._mpc_diagnostics_callback(watcher,message)
        self.assertEqual(watcher.mpc_fault_reason,
                         "rejected_nonlinear_rollout; nonlinear=corridor@4")
        self.assertEqual(watcher.mpc_rejection_count,1)


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
