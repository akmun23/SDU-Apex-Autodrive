"""Observer fixtures describe file formats, not measured vehicle performance."""
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from tools.optimizer_dashboard import (Dashboard, LogObserver, Workspace,
                                       handler_for, json_bytes, process_info)


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root/'starts/clockwise/campaign/from100_90_phase000_resample/solve_000'
        self.folder.mkdir(parents=True)
        self.campaign_path = self.root/'starts/clockwise/campaign/campaign.json'
        self.context = dict(key='from100_90_phase000_resample', output=str(self.folder), refinement=0)
        self.write('optimizer.json', dict(status='running', pid=0,
            stage='adaptive_minimum_time_search', active_start='clockwise',
            contract=dict(iterations=1500, workers=8, stagnation_iterations=250, direction='clockwise'),
            geometry_starts=[]))
        self.write(self.campaign_path, dict(active=self.context, attempts=[]))
        self.log = self.root/'optimizer.log'
        self.log.write_text('')
        self.dashboard = Dashboard(self.root)

    def write(self, path, data):
        path = Path(path)
        if not path.is_absolute():
            path = self.root/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def start_log(self):
        self.log.write_text(json.dumps(dict(event='solve', **self.context))+'\n'+
            json.dumps(dict(stage='production-command witness', trial_lap_s=2.25,
                            initial_residual=.5, max_iterations=1500))+'\n')

    def test_snapshot_decodes_real_npz_metadata_without_claiming_running(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(iteration=5, trial_lap_s=2.25,
                                         max_constraint_violation=.12))+'\n')
            stream.write('  6  1.234567e-04 1.0e-05 2.0e-06 -10 0.5 - 1 1h 1\n')
        np.savez_compressed(self.folder/'best_iterate.npz', iteration=5,
                            max_constraint_violation=.12, state=np.zeros((21, 91)))
        result = self.dashboard.snapshot()
        self.assertEqual(result['iteration'], 6)
        self.assertEqual(result['checkpoint']['iteration'], 5)
        self.assertEqual(result['checkpoint']['residual'], .12)
        self.assertEqual(result['active']['trial_lap_s'], 2.25)
        self.assertEqual(result['status'], 'not_running')
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['residuals'][-1]['residual'], .12)
        self.assertEqual(result['ipopt']['primal_infeasibility'], 1e-5)

    def test_never_mix_ipopt_scaled_primal_with_actual_constraint_residual(self):
        reader = LogObserver(self.log)
        reader.parse('{"iteration":5,"max_constraint_violation":0.4}')
        reader.parse('  5  9.5e-05 1.0e-08 1.0e-12 -10 0.1 - 1 1h 1')
        self.assertEqual(reader.points[-1]['residual'], .4)
        self.assertEqual(reader.ipopt['primal_infeasibility'], 1e-8)

    def test_partial_lines_and_new_solve_do_not_retain_old_iteration(self):
        reader = LogObserver(self.log)
        self.log.write_text('{"iteration":5,"max_constraint_violation":0.4}')
        reader.poll()
        self.assertIsNone(reader.live_iteration)
        with self.log.open('a') as stream:
            stream.write('\n')
        reader.poll()
        self.assertEqual(reader.live_iteration, 5)
        with self.log.open('a') as stream:
            stream.write('{"event":"solve","output":"new"}\n')
        reader.poll()
        self.assertIsNone(reader.live_iteration)
        self.assertEqual(list(reader.points), [])
        self.log.write_text('{"iteration":0,"max_constraint_violation":0.8}\n')
        reader.poll()
        self.assertEqual(reader.live_iteration, 0)
        self.assertEqual(len(reader.points), 1)

    def test_campaign_change_before_log_header_hides_previous_solve(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write('{"iteration":70,"max_constraint_violation":0.03}\n')
        next_context = dict(key='from90_81_phase020_resample', output=str(self.root/'new_solve'))
        self.write(self.campaign_path, dict(active=next_context))
        result = self.dashboard.snapshot()
        self.assertIsNone(result['iteration'])
        self.assertEqual(result['residuals'], [])
        self.assertAlmostEqual(result['active']['trial_lap_s'], 2.025)
        self.assertEqual(result['active']['phase'], 20)

    def test_read_only_snapshot_does_not_touch_any_workspace_file(self):
        self.start_log()
        before = {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                  for path in self.root.rglob('*') if path.is_file()}
        self.dashboard.snapshot()
        after = {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)

    def test_model_only_exports_preserve_validity_flags(self):
        manifest = dict(lap_s=2.5, direction='clockwise', source_only=True,
                        unity_validated=False, mpc_validated=False, runtime_promoted=False,
                        files={})
        self.write('best.json', manifest)
        self.write('exports/abc/manifest.json', manifest)
        result = self.dashboard.snapshot()
        self.assertEqual(result['history'][0]['lap_s'], 2.5)
        self.assertFalse(result['best']['unity_validated'])
        self.assertFalse(result['best']['mpc_validated'])
        json_bytes(result)  # Strict JSON: never NaN / Infinity in the browser API.

    def test_bad_checkpoint_is_reported_without_killing_dashboard(self):
        (self.folder/'best_iterate.npz').write_bytes(b'interrupted file')
        result = self.dashboard.snapshot()
        self.assertIsNone(result['checkpoint'])
        self.assertTrue(any('Checkpoint unavailable' in value for value in result['warnings']))

    def test_map_and_reference_are_read_in_their_own_frame(self):
        (self.root/'input').mkdir()
        Image.new('L', (20, 30), 255).save(self.root/'input/map.pgm')
        self.write('input/map.yaml', dict(origin=[3, -4, .7], resolution=.05))
        csv = self.root/'exports/abc/line.csv'
        csv.parent.mkdir(parents=True)
        np.savetxt(csv, [[0, 3, -4], [1, 3.5, -3.5]], delimiter=',')
        self.write('best.json', dict(files=dict(raceline=str(csv)), witness_sha256='abc'))
        data = self.dashboard.track()
        self.assertEqual(data['origin'], [3, -4, .7])
        self.assertEqual(data['resolution'], .05)
        self.assertEqual((data['width'], data['height']), (20, 30))
        self.assertEqual(data['layers'][0]['points'], [[3., -4.], [3.5, -3.5]])
        self.assertTrue(self.dashboard.map_payload()[0].startswith(b'\x89PNG'))

    def test_file_reads_reject_traversal_and_symlinks_outside_workspace(self):
        workspace = Workspace(self.root)
        with self.assertRaises(ValueError):
            workspace.path('../secret')
        (self.root/'outside').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            workspace.path('outside/secret')
        self.write('best.json', dict(files=dict(raceline='/etc/passwd')))
        with self.assertRaises(ValueError):
            self.dashboard.download('raceline.csv')
        with self.assertRaises(FileNotFoundError):
            self.dashboard.download('../../etc/passwd')

    def test_pid_reuse_does_not_claim_optimizer_running(self):
        with patch.object(Path, 'read_bytes', return_value=b'/usr/bin/other-process\x00'), \
             patch.object(Path, 'read_text', return_value='State:\tS\nVmRSS:\t120 kB\nThreads:\t1\n'):
            self.assertFalse(process_info(123)['alive'])

    def test_http_endpoints_are_local_read_only_and_no_generic_file_server(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.dashboard))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}'
        with urlopen(url+'/api/snapshot') as response:
            self.assertEqual(json.load(response)['name'], self.root.name)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        with urlopen(url+'/') as response:
            self.assertIn(b'Raceline optimizer', response.read())
        cases = [('/etc/passwd', None, 404), ('/stop', b'', 501),
                 ('/api/snapshot', None, 403)]
        for path, data, code in cases:
            headers = {'Origin': 'https://attacker.invalid'} if code == 403 else {}
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(url+path, data=data, headers=headers))
            self.assertEqual(caught.exception.code, code)


if __name__ == '__main__':
    unittest.main()
