#!/usr/bin/env python3
"""Recovery contract checks; never launch IPOPT or a simulator."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from watch_ipopt import observed_state


class RecoveryTests(unittest.TestCase):
    def test_inactive_service_overrides_stale_running_record(self):
        for state in ('inactive', 'failed', 'unknown', 'not-found'):
            with self.subTest(state=state):
                self.assertTrue(observed_state('running', state).startswith(
                    'interrupted / service '))

    def test_active_service_and_finished_record_are_not_interrupted(self):
        self.assertEqual(observed_state('running', 'active'),
                         'running / service active')
        self.assertEqual(observed_state('finished', 'inactive'),
                         'finished / service inactive')

    def test_bad_restart_options_fail_before_creating_output(self):
        launcher = Path(__file__).with_name('run_ipopt_unattended.sh')
        with tempfile.TemporaryDirectory() as directory:
            seed = Path(directory) / 'seed.npz'
            seed.touch()
            output = Path(directory) / 'must_not_exist'
            for options in (['--trust-factors'], ['--trust-factors', ''],
                            ['--trust-factors', '4,2'],
                            ['--trust-factors', '4'],
                            ['--trust-factors', '4,nan'],
                            ['--trust-factors', '0,4'],
                            ['--trust-factors', '4,4'], ['--bogus']):
                with self.subTest(options=options):
                    result = subprocess.run(
                        ['bash', str(launcher), str(output), str(seed), *options],
                        capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
