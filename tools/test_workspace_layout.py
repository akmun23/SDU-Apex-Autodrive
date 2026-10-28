"""Keep generated/vendor trees out of ROS discovery and Git source exports."""

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER = ROOT / 'SDU_Apex_IPOPT_PhysX_MinTime_v2'


class WorkspaceLayoutTests(unittest.TestCase):
    def test_non_ros_roots_have_exportable_discovery_markers(self):
        for name in ('external', 'live_runs', 'build_task', OPTIMIZER.name):
            with self.subTest(name=name):
                marker = ROOT / name / 'COLCON_IGNORE'
                self.assertTrue(marker.is_file())
                ignored = subprocess.run(
                    ['git', 'check-ignore', '--no-index', '--quiet', str(marker)],
                    cwd=ROOT, capture_output=True)
                self.assertEqual(ignored.returncode, 1,
                                 'marker must survive source checkout')

    def test_required_optimizer_overlays_are_exportable(self):
        # These are transitively imported by recover/solver_overlay_v251.
        # They previously existed only on the development machine because
        # the package-wide ignore rule hid them from Git.
        for name in ('solver_overlay_condensed_v242.py',
                     'solver_overlay_eliminated_v243.py',
                     'vehicle_mechanics_overlay_v250.py',
                     'wheel_model_overlay_v241.py',
                     'wheel_model_overlay_v250.py',
                     'wheel_model_overlay_v251.py'):
            with self.subTest(name=name):
                path = OPTIMIZER / name
                self.assertTrue(path.is_file())
                ignored = subprocess.run(
                    ['git', 'check-ignore', '--no-index', '--quiet', str(path)],
                    cwd=ROOT, capture_output=True)
                self.assertEqual(ignored.returncode, 1)

    def test_generated_copies_and_live_outputs_are_not_tracked(self):
        tracked = subprocess.check_output(
            ['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        forbidden = ('optimized_', OPTIMIZER.name + '/_working_solver',
                     OPTIMIZER.name + '/_restored_git_history/',
                     OPTIMIZER.name + '/min_curvature_tradeoff_')
        self.assertFalse([p for p in tracked if p.startswith(forbidden)])


if __name__ == '__main__':
    unittest.main()
