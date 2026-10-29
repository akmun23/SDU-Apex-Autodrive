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

    def test_time_prediction_uses_its_own_residuals_and_is_never_a_certified_trial(self):
        self.start_log()
        self.write(self.campaign_path, dict(active=None, attempts=[], active_predictor=dict(
            output=str(self.folder), incumbent_sha256='fixture')))
        with self.log.open('a') as stream:
            for event in (dict(event='minimum_time_prediction', output=str(self.folder)),
                          dict(stage='minimum-time predictor setup', source_lap_s=2.25,
                               max_iterations=200),
                          dict(stage='minimum-time predictor', iteration=5,
                               predicted_lap_s=2.22, max_constraint_violation=.002)):
                stream.write(json.dumps(event)+'\n')
        result = self.dashboard.snapshot()
        self.assertTrue(result['active']['time_predictor'])
        self.assertEqual(result['active']['trial_lap_s'], 2.22)
        self.assertEqual(result['iteration'], 5)
        self.assertEqual(result['residuals'][-1]['residual'], .002)
        self.assertEqual(result['passed_attempts'], 0)

    def test_idle_between_trials_accepts_null_active(self):
        self.write(self.campaign_path, dict(active=None, attempts=[]))
        result = self.dashboard.snapshot()
        self.assertIsNone(result['active']['trial_lap_s'])

    def test_feasibility_and_ipopt_raw_use_separate_graphs(self):
        html = Path(__file__).with_name('optimizer_dashboard.html').read_text()
        actual = html.split('function residualChart(){', 1)[1].split('function ipoptChart(){', 1)[0]
        raw = html.split('function ipoptChart(){', 1)[1].split('function lapChart(){', 1)[0]
        self.assertIn('id="residualChart"', html)
        self.assertIn('id="ipoptChart"', html)
        self.assertIn('threshold=-4', actual)
        self.assertNotIn('ipopt_history', actual)
        self.assertIn('ipopt_history', raw)
        self.assertNotIn('threshold=-4', raw)
        self.assertIn('Separate logarithmic scale', html)

    def test_intentional_stagnation_is_reported_without_claiming_optimizer_crash(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write('RuntimeError: command feasibility stagnated; durable best iterate saved\n')
        result = self.dashboard.snapshot()
        event = next(event for event in result['events']
                     if event.get('event')=='feasibility_stagnation')
        self.assertEqual(event['event'], 'feasibility_stagnation')
        self.assertEqual(result['status'], 'not_running')
        self.assertTrue(any('RuntimeError: command feasibility stagnated' in line
                            for line in result['log_lines']))

    def test_raw_ipopt_history_updates_without_actual_residual_callback(self):
        self.start_log()
        with self.log.open('a') as stream:
            stream.write('  1  1.234567e-04 1.0e-02 2.0e-04 -3 0.5 - 1 1h 1\n')
            stream.write('  2  1.234567e-04 3.0e-03 2.0e-04 -3 0.5 - 1 1h 1\n')
        result = self.dashboard.snapshot()
        self.assertEqual(result['iteration'], 2)
        self.assertEqual(len(result['ipopt_history']), 2)
        self.assertEqual(result['ipopt_history'][-1]['primal_infeasibility'], .003)
        self.assertEqual(result['residuals'], [])
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(event='solve',output='different-solve'))+'\n')
        self.dashboard.log.poll()
        self.assertEqual(list(self.dashboard.log.ipopt_points), [])

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

    def test_command_seed_campaign_extracts_and_caches_missing_centerline(self):
        # Command-seed workflows intentionally have no geometry_starts; this
        # must not silently remove the centerline layer from the viewer.
        from time import monotonic, sleep
        (self.root/'input').mkdir()
        Image.new('L', (20, 30), 255).save(self.root/'input/map.pgm')
        self.write('input/map.yaml', dict(origin=[0, 0, 0], resolution=.05))
        points = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        with patch('tools.optimizer_dashboard.map_centerline', return_value=points) as extract:
            self.dashboard.track()
            end = monotonic()+2.
            while self.dashboard.centerline_status == 'extracting' and monotonic() < end:
                sleep(.005)
            data = self.dashboard.track()
            self.assertEqual(data['centerline_status'], 'ready')
            self.assertEqual(data['layers'][0]['kind'], 'centerline')
            self.assertEqual(data['layers'][0]['points'], points)
            self.dashboard.track()
            extract.assert_called_once()

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

    def test_unity_queue_exposes_trial_metrics_and_honest_worker_status(self):
        self.write('mpc_validation/state.json', dict(
            pending=['def567'], worker=dict(status='running'),
            attempts=[dict(id='abc123', model_lap_s=7.35, status='completed',
                           run_id='apex_mpc_abc1234567890abc_a01', attempt=1,
                           metrics=dict(average_scored_s=7.6,
                                        scored_lap_times_s=[7.6]*10))]))
        result=self.dashboard.snapshot()['mpc_trials']
        self.assertEqual(result['pending'],1)
        self.assertEqual(result['attempts'][0]['metrics']['average_scored_s'],7.6)
        self.assertEqual(result['attempts'][0]['status'],'completed')
        self.assertFalse(result['supervisor_alive'])
        self.assertEqual(result['total_trials'], 1)

    def test_only_latest_unity_attempt_is_sent_to_gui(self):
        self.write('mpc_validation/state.json', dict(
            pending=[], worker=dict(status='idle'),
            attempts=[
                dict(id='original', model_lap_s=11.275, status='completed',
                     attempt=1, started_epoch_s=100.,
                     metrics=dict(completed_scored_laps=10)),
                dict(id='latest', model_lap_s=8.25, status='mpc_rejected',
                     attempt=1, started_epoch_s=300.,
                     reason='rejected_nonlinear_rollout; nonlinear=corridor@2',
                     metrics=dict(completed_scored_laps=1)),
                dict(id='middle', model_lap_s=9.15, status='mpc_rejected',
                     attempt=1, started_epoch_s=200.,
                     metrics=dict(completed_scored_laps=1))]))
        result=self.dashboard.snapshot()['mpc_trials']
        self.assertEqual(len(result['attempts']),1)
        self.assertEqual(result['attempts'][0]['id'],'latest')
        self.assertEqual(result['attempts'][0]['metrics']['completed_scored_laps'],1)

    def test_unity_lap_count_has_independent_height_and_no_clipping(self):
        html = Path(__file__).with_name('optimizer_dashboard.html').read_text()
        self.assertEqual(html.count('function renderMpc(data){'),1)
        self.assertNotIn('class="mpc-progress-label"',html)
        self.assertIn("mpcAppend(summary,top,quick,bar)",html)
        self.assertIn('mpc-summary-meta',html)
        self.assertIn("'Warm-up: '+mpcNum(m.warmup_lap_s)",html)
        self.assertIn('Warm-up not completed',html)
        self.assertIn("slice(0,1)",html)

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
        cases = [('/etc/passwd', None, 404), ('/stop', b'', 403),
                 ('/api/snapshot', None, 403)]
        for path, data, code in cases:
            headers = {'Origin': 'https://attacker.invalid'} if code == 403 else {}
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(url+path, data=data, headers=headers))
            self.assertEqual(caught.exception.code, code)

    def test_retry_endpoint_requires_same_origin_and_json(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(self.dashboard))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}'
        for origin in (None, 'https://attacker.invalid'):
            headers = {'Content-Type': 'application/json'}
            if origin:
                headers['Origin'] = origin
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(url+'/api/retry', data=b'{}', headers=headers))
            self.assertEqual(caught.exception.code, 403)
        with patch.object(self.dashboard, 'retry', return_value=dict(status='queued')) as retry:
            request = Request(url+'/api/retry', data=b'{"id":"aaaaaaaaaaaaaaaa"}',
                              headers={'Origin': url, 'Content-Type': 'application/json'})
            with urlopen(request) as response:
                self.assertEqual(response.status, 202)
                self.assertEqual(json.load(response), dict(status='queued'))
            retry.assert_called_once_with('aaaaaaaaaaaaaaaa')

    def test_retry_refuses_stopped_supervisor_and_invalid_candidates(self):
        with self.assertRaisesRegex(RuntimeError, 'supervisor'):
            self.dashboard.retry('a'*16)
        with patch('tools.optimizer_dashboard.validation_supervisor', return_value=True):
            with self.assertRaises(ValueError):
                self.dashboard.retry('../unsafe')
            with self.assertRaises(ValueError):
                self.dashboard.retry('a'*16)


if __name__ == '__main__':
    unittest.main()
