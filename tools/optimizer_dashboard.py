#!/usr/bin/env python3
"""Localhost dashboard for map-only raceline optimizer workspaces.

No optimizer imports, physics builds or subprocess solver calls. The single
write action queues a verified, previously failed Unity trial for the supervisor.
The observer is deliberately outside the optimizer's implementation-hash contract.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import webbrowser

import numpy as np
from PIL import Image
import yaml


REPO = Path(__file__).resolve().parents[1]
HTML = Path(__file__).with_suffix('.html')
ITERATION = re.compile(
    r'^\s*(\d+)(r?)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+')
TRIAL = re.compile(r'(?:from\d+_)?(\d+)_phase(\d+)_(.+)')
def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode()


class Workspace:
    """Confine all manifest-selected reads to one explicitly selected workspace."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}

    def path(self, value):
        path = Path(value)
        path = (self.root/path if not path.is_absolute() else path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('manifest path leaves the selected workspace')
        return path

    def read(self, path, loader):
        path = self.path(path)
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        key = (path, loader)
        if key not in self.cache or self.cache[key][0] != stamp:
            self.cache[key] = (stamp, loader(path))
        return self.cache[key][1]

    def json(self, path):
        return self.read(path, read_json)


def read_json(path):
    result = json.loads(path.read_text(), parse_constant=lambda _: None)
    if not isinstance(result, dict):
        raise ValueError('expected a JSON object')
    return result


class LogObserver:
    """Incremental bounded log reader; never reread a growing multi-day log."""
    def __init__(self, path):
        self.path = path
        self.identity = None
        self.offset = 0
        self.partial = b''
        self.lines = deque(maxlen=90)
        self.events = deque(maxlen=30)
        self.points = deque(maxlen=1600)
        self.ipopt_points = deque(maxlen=2000)
        self.solve = {}
        self.output = None
        self.completed = False
        self.live_iteration = None
        self.ipopt = {}
        self.initial_read = True
        self.last_change_seen = None
        self.log_mtime = None

    def parse(self, line):
        line = line.rstrip()
        if not line:
            return
        self.lines.append(line[:2000])
        # This specific callback exception is an INTENTIONAL IPOPT
        # early-exit after checkpointing. Preserve the original stack trace
        # in the raw logfile, but describe the outcome correctly to the user.
        if line.startswith('RuntimeError: command feasibility stagnated; durable best iterate saved'):
            self.events.append(dict(event='feasibility_stagnation',
                                    iteration=self.live_iteration))
        try:
            item = json.loads(line, parse_constant=lambda _: None)
        except (ValueError, TypeError):
            item = None
        if isinstance(item, dict):
            if item.get('event') in ('solve', 'minimum_time_prediction'):
                self.output = item.get('output')
                self.points.clear()
                self.ipopt_points.clear()
                self.solve = {}
                self.completed = False
                self.live_iteration = None
                self.ipopt = {}
            if item.get('stage') in ('production-command witness', 'minimum-time predictor setup'):
                self.solve = item
                self.points.clear()
                self.ipopt_points.clear()
                self.live_iteration = None
                self.ipopt = {}
                self.completed = False
            if item.get('stage') == 'minimum-time predictor':
                self.solve['predicted_lap_s'] = item.get('predicted_lap_s')
            if 'iteration' in item and 'max_constraint_violation' in item:
                self.live_iteration = int(item['iteration'])
                self.add_point(item['iteration'], item['max_constraint_violation'])
            if item.get('event') or (item.get('stage') and
                                     item['stage'] != 'production-command witness'):
                self.events.append(item)
            if 'status' in item and 'report' in item:
                self.completed = True
            return
        match = ITERATION.match(line)
        if match:
            iteration, restoration, objective, primal, dual = match.groups()
            self.live_iteration = int(iteration)
            # IPOPT inf_pr can be internally scaled. The callback evaluates the
            # actual bounds; mixing these numbers would falsely imply feasibility.
            self.ipopt = dict(iteration=int(iteration), objective=finite(objective),
                              primal_infeasibility=finite(primal),
                              dual_infeasibility=finite(dual), restoration=bool(restoration))
            self.ipopt_points.append(dict(self.ipopt))

    def add_point(self, iteration, residual, objective=None, dual=None, restoration=False):
        point = dict(iteration=int(iteration), residual=finite(residual),
                     objective=finite(objective), dual=finite(dual), restoration=restoration)
        if self.points and self.points[-1]['iteration'] == point['iteration']:
            previous = self.points.pop()
            point = {k: previous[k] if v is None else v for k, v in point.items()}
        self.points.append(point)

    def poll(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if self.identity != identity or stat.st_size < self.offset:
            self.offset = max(0, stat.st_size-1024*1024)
            self.partial = b''
            self.lines.clear()
            self.events.clear()
            self.points.clear()
            self.ipopt_points.clear()
            self.solve = {}
            self.output = None
            self.live_iteration = None
            self.ipopt = {}
            self.completed = False
            self.initial_read = True
            self.identity = identity
        if stat.st_size > self.offset:
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                chunk = stream.read(1024*1024)
            discard_first = self.initial_read and self.offset > 0
            self.offset += len(chunk)
            chunks = (self.partial+chunk).split(b'\n')
            self.partial = chunks.pop()[-8192:]
            if discard_first and chunks:
                chunks.pop(0)
            for line in chunks:
                self.parse(line.decode('utf-8', errors='replace'))
            self.last_change_seen = time.time()
            self.initial_read = False
        self.log_mtime = stat.st_mtime


def process_info(pid):
    """Check command identity as well as PID (PID reuse must not mean 'running')."""
    if not isinstance(pid, int) or pid <= 0:
        return dict(alive=False, verified=False)
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                      if ':' in line)
        verified = ('track_optimizer.py' in command or 'continue_direct_challenger.py' in command)
        return dict(alive=verified and not status.get('State', '').strip().startswith('Z'),
                    verified=verified, pid=pid,
                    rss_mb=float(status.get('VmRSS', '0 kB').split()[0])/1024,
                    threads=int(status.get('Threads', '0').strip()))
    except (OSError, ValueError):
        return dict(alive=False, verified=False, pid=pid)


def checkpoint_data(path):
    # Read metadata only; do not decompress 21-state shooting arrays for status.
    with np.load(path, allow_pickle=False) as data:
        return dict(iteration=int(data['iteration']),
                    residual=finite(data['max_constraint_violation']))


def validation_supervisor(workspace, state):
    pid = state.get('supervisor', {}).get('pid')
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        proc = Path('/proc')/str(pid)
        args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        return (any(Path(arg).name == 'optimizer_mpc_validation.py' for arg in args)
                and (proc.joinpath('cwd').resolve(strict=True) /
                     args[args.index('--workspace')+1]).resolve() == workspace)
    except (OSError, ValueError, IndexError):
        return False


def reference_points(path):
    values = np.loadtxt(path, delimiter=',', comments='#', ndmin=2)
    values = values[np.isfinite(values[:, 1:3]).all(axis=1)]
    if not len(values):
        return []
    stride = max(1, math.ceil(len(values)/1600))
    return values[::stride, 1:3].tolist()


def map_centerline(image):
    """Same map-only extraction as geometry bootstrap; no optimization or writes."""
    if __package__:
        from . import build_mapped_raceline as builder
    else:
        import build_mapped_raceline as builder
    import map_track
    # Initialize the reusable geometry helper exactly as builder.build does,
    # without invoking its curvature optimization, plotting or file output.
    builder.map_track_module, builder.map_to_track = map_track, map_track.map_to_track
    grid = map_track.MapGrid.load(image)
    track, _ = builder._extract_map_track(grid, image, min(.12, grid.resolution*2.4))
    return track.xy[::max(1, math.ceil(len(track.xy)/1600))].tolist()


class Dashboard:
    def __init__(self, root):
        self.workspace = Workspace(root)
        self.log = LogObserver(self.workspace.root/'optimizer.log')
        self.lock = threading.RLock()
        self.last_poll = 0.
        self.snapshot_cache = None
        self.warnings = []
        self.centerline_key = None
        self.centerline_points = []
        self.centerline_status = 'pending'
        self.centerline_message = ''

    def derived_centerline(self):
        paths = [self.workspace.path('input/map.'+ext) for ext in ('pgm', 'yaml')]
        key = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        if key != self.centerline_key:
            self.centerline_key = key
            self.centerline_points = []
            self.centerline_status = 'extracting'
            self.centerline_message = 'Extracting the centerline from this map…'
            def extract():
                try:
                    points, status, message = map_centerline(paths[0]), 'ready', ''
                except Exception as error:
                    points, status, message = [], 'unavailable', str(error)
                with self.lock:
                    if self.centerline_key == key:
                        self.centerline_points, self.centerline_status, self.centerline_message = points, status, message
            threading.Thread(target=extract, name='dashboard-map-centerline', daemon=True).start()
        return self.centerline_points

    def optional(self, path, default=None):
        try:
            return self.workspace.json(path)
        except (OSError, ValueError, TypeError) as error:
            if not isinstance(error, FileNotFoundError):
                self.warnings.append(f'{Path(path).name}: {error}')
            return {} if default is None else default

    def snapshot(self):
        with self.lock:
            if self.snapshot_cache is not None and time.monotonic()-self.last_poll < 1.:
                return self.snapshot_cache
            self.warnings = []
            state = self.optional('optimizer.json')
            best = self.optional('best.json')
            self.log.poll()
            now = time.time()
            active_name = state.get('active_start')
            campaign = (self.optional(f'starts/{active_name}/campaign/campaign.json')
                        if active_name else {})
            predicting = campaign.get('active_predictor')
            active = ((predicting or campaign.get('active') or {})
                      if state.get('stage') == 'adaptive_minimum_time_search' else (
                          state.get('starts', {}).get(active_name, {}).get('active') or {}))
            context = dict(active)
            checkpoint = None
            report = None
            if active.get('output'):
                try:
                    folder = self.workspace.path(active['output'])
                    saved = folder/('best_predictor.npz' if predicting else 'best_iterate.npz')
                    if saved.is_file():
                        checkpoint = dict(self.workspace.read(saved, checkpoint_data),
                                          saved_epoch_s=saved.stat().st_mtime,
                                          path=str(saved.relative_to(self.workspace.root)))
                    report = self.optional(folder/'report.json') or None
                    context['output'] = str(folder.relative_to(self.workspace.root))
                except (OSError, ValueError, KeyError) as error:
                    self.warnings.append(f'Checkpoint unavailable: {error}')
            # A campaign can select the next solve before its solver header appears.
            # Never show the previous solve's iteration/residual as the new one's.
            matches = bool(active.get('output') and self.log.output and
                           str(Path(active['output'])) == self.log.output)
            # Bootstrap has no campaign 'solve' event; match its explicit duration.
            if not self.log.output and state.get('stage') == 'periodic_bootstrap':
                matches = self.log.solve.get('trial_lap_s') == state.get('trial_lap_s')
            solver = dict(self.log.solve) if matches else {}
            points = list(self.log.points) if matches else []
            ipopt_points = list(self.log.ipopt_points) if matches else []
            trial = TRIAL.fullmatch(active.get('key', ''))
            trial_s = finite(solver.get('predicted_lap_s') if predicting else solver.get('trial_lap_s'))
            if trial_s is None:
                trial_s = int(trial[1])*.025 if trial else (
                    finite(state.get('trial_lap_s')) if state.get('stage') == 'periodic_bootstrap' else None)
            history = []
            for path in sorted((self.workspace.root/'exports').glob('*/manifest.json')):
                item = self.optional(path)
                lap = finite(item.get('lap_s'))
                if lap is not None and item.get('source_only'):
                    history.append(dict(lap_s=lap, epoch_s=path.stat().st_mtime,
                                        id=path.parent.name))
            history.sort(key=lambda item: item['epoch_s'])
            validation = self.optional(best.get('files', {}).get('validation', 'missing'))
            proc = process_info(state.get('pid'))
            status = state.get('status', 'waiting')
            if status == 'running' and not proc['alive']:
                status = 'not_running'
            age = max(0., now-self.log.log_mtime) if self.log.log_mtime is not None else None
            log_status = ('not_running' if not proc['alive'] else
                          'quiet' if age is None or age > 90 else 'active')
            attempts = campaign.get('attempts', [])
            context.update(trial_lap_s=trial_s, phase=int(trial[2]) if trial else None,
                           strategy='minimum-time predictor' if predicting else trial[3] if trial else None,
                           time_predictor=bool(predicting))
            best_summary = {key: best.get(key) for key in (
                'lap_s', 'direction', 'source_only', 'unity_validated', 'mpc_validated', 'runtime_promoted')}
            best_summary['available_downloads'] = [key for key in (
                'raceline', 'raceline_with_witness', 'validation') if best.get('files', {}).get(key)]
            mpc_state = self.optional('mpc_validation/state.json', {})
            supervisor_alive = validation_supervisor(self.workspace.root, mpc_state)
            mpc_attempts = []
            for trial in sorted(
                mpc_state.get('attempts', []),
                key=lambda attempt: (finite(attempt.get('started_epoch_s')) or
                                     finite(attempt.get('finished_epoch_s')) or
                                     finite(attempt.get('queued_epoch_s')) or 0.,
                                     int(attempt.get('attempt') or 0)),
                reverse=True,
            )[:1]:
                compact = {key: trial.get(key) for key in (
                    'id', 'model_lap_s', 'run_id', 'status', 'reason', 'attempt',
                    'started_epoch_s', 'finished_epoch_s', 'analysis_error',
                    'trajectory_sha256', 'mpc_validated', 'unity_trial_completed',
                    'mpc_profile_sha256', 'decoder_compatibility_note')}
                metrics = trial.get('metrics') or {}
                compact['metrics'] = {key: metrics.get(key) for key in (
                    'warmup_lap_s', 'scored_lap_times_s', 'completed_scored_laps',
                    'per_official_lap_tracking',
                    'total_scored_s', 'average_scored_s', 'best_scored_s',
                    'worst_scored_s', 'stddev_scored_s', 'tracking',
                    'absolute_error_distributions', 'status_counts', 'cycle_reason_counts',
                    'nonlinear_failure_reason_counts', 'controller_cycles',
                    'score_cutoff_reason', 'gate_timing_ms', 'jacobian_profile',
                    'solver_profile', 'solver_termination_counts',
                    'solver_stop_reason_counts', 'fixed_command_rollout',
                    'first_collision_from_start_s', 'first_unpublished_mpc_rejection_from_start_s')}
                if compact['status'] == 'running':
                    run_id = compact.get('run_id') or ''
                    # Only read logs from explicitly named auto-trial directories.
                    if re.fullmatch(r'apex_mpc_[a-f0-9]{16}_a[0-9]{2,}', run_id):
                        watcher = REPO/'live_runs'/run_id/'watcher.log'
                        run_dir = watcher.parent
                        compact['live_phase'] = ('analyzing' if (run_dir/'analysis.log').exists()
                                                 else 'driving' if watcher.exists()
                                                 else 'launching' if (run_dir/'controller.log').exists()
                                                 else 'building')
                        if watcher.is_file() and not watcher.is_symlink():
                            try:
                                lines = watcher.read_text(errors='replace').splitlines()
                                counts = [int(m.group(1)) for line in lines[-300:]
                                          if (m := re.fullmatch(r'lap_count=(\d+)', line))]
                                compact['official_laps_seen'] = max(counts, default=0)
                                aborts = [line for line in lines[-100:] if 'SCREEN_ABORT:' in line]
                                if aborts:
                                    compact['live_failure'] = aborts[-1][:400]
                            except OSError:
                                pass
                mpc_attempts.append(compact)
            mpc_trials = dict(enabled=bool(mpc_state), pending=len(mpc_state.get('pending', [])),
                              supervisor_alive=supervisor_alive,
                              retry_requests=len(list((self.workspace.root/'mpc_validation/retry_requests').glob('*.json'))),
                              optimizer_pause=self.optional('mpc_validation/optimizer_pause.json', {}),
                              total_trials=len(mpc_state.get('attempts', [])),
                              successful_trials=sum(bool(a.get('mpc_validated')) for a in mpc_state.get('attempts', [])),
                              worker=mpc_state.get('worker', {'status': 'not_configured'}),
                              attempts=mpc_attempts)
            result = dict(workspace=str(self.workspace.root), name=self.workspace.root.name,
                          now_epoch_s=now, status=status, recorded_status=state.get('status'),
                          stage=state.get('stage', 'waiting'), active_start=active_name,
                          direction=state.get('contract', {}).get('direction'),
                          contract=state.get('contract', {}), process=proc,
                          log_age_s=age, activity=log_status, best=best_summary,
                          validation=validation, history=history, active=context,
                          solver=solver, iteration=self.log.live_iteration if matches else None,
                          ipopt=dict(self.log.ipopt) if matches else {},
                          checkpoint=checkpoint, residuals=points, ipopt_history=ipopt_points, solve_report=report,
                          completed_attempts=len(attempts),
                          passed_attempts=sum(bool(v.get('certified')) for v in attempts),
                          geometry_starts=state.get('geometry_starts', []),
                          attempts=attempts[-12:][::-1], events=list(self.log.events)[::-1],
                          log_lines=list(self.log.lines), warnings=list(self.warnings), mpc_trials=mpc_trials,
                          error=state.get('error'))
            self.snapshot_cache = result
            self.last_poll = time.monotonic()
            return result

    def retry(self, ident):
        """Queue only an existing failed certificate; never execute shell input."""
        if __package__:
            from .optimizer_mpc_validation import certified_exports, request_retry
        else:
            from optimizer_mpc_validation import certified_exports, request_retry
        with self.lock:
            state = self.optional('mpc_validation/state.json', {})
            if not validation_supervisor(self.workspace.root, state):
                raise RuntimeError('Automatic test supervisor is not running')
            if not isinstance(ident, str) or not re.fullmatch(r'[a-f0-9]{16}', ident):
                raise ValueError('Choose a recorded candidate to retry')
            attempt = next((a for a in reversed(state.get('attempts', [])) if a.get('id') == ident), None)
            if not attempt or attempt.get('status') in ('running', 'completed'):
                raise ValueError('Only a failed recorded test can be retried')
            if ident in state.get('pending', []):
                return dict(status='already_queued')
            if not any(item['id'] == ident for item in certified_exports(self.workspace.root)):
                raise ValueError('Source certificate no longer verifies')
            for path in (self.workspace.root/'mpc_validation/retry_requests').glob('*.json'):
                if self.optional(path).get('witness_prefix') == ident:
                    return dict(status='already_queued')
            request_retry(self.workspace.root, ident)
            self.last_poll = 0.
            return dict(status='queued')

    def map_payload(self):
        config = self.workspace.read('input/map.yaml', read_map_config)
        return self.workspace.read('input/map.pgm', map_png), config

    def track(self):
        with self.lock:
            best = self.optional('best.json')
            state = self.optional('optimizer.json')
            layers = []
            starts = state.get('geometry_starts', [])
            if starts:
                try:
                    path = self.workspace.path(starts[0]['csv']).with_name('centerline.csv')
                except ValueError:
                    path = self.workspace.path('geometry/unity_reverse_seed/centerline.csv')
                if path.is_file():
                    layers.append(dict(name='Extracted centerline', kind='centerline',
                                       points=self.workspace.read(path, reference_points)))
            centerline_status, centerline_message = 'ready', ''
            if not layers:
                points = self.derived_centerline()
                centerline_status, centerline_message = self.centerline_status, self.centerline_message
                if points:
                    layers.append(dict(name='Extracted map centerline', kind='centerline', points=points))
            reference = best.get('files', {}).get('raceline')
            if reference:
                layers.append(dict(name='Best source-validated line', kind='best',
                                   points=self.workspace.read(reference, reference_points)))
            _, config = self.map_payload()
            with Image.open(self.workspace.path('input/map.pgm')) as image:
                return dict(layers=layers, width=image.width, height=image.height,
                            resolution=float(config['resolution']), origin=config['origin'],
                            centerline_status=centerline_status, centerline_message=centerline_message,
                            best_id=best.get('witness_sha256'))

    def download(self, name):
        mapping = {'raceline.csv': 'raceline', 'commands.csv': 'raceline_with_witness',
                   'validation.json': 'validation'}
        if name not in mapping:
            raise FileNotFoundError(name)
        best = self.workspace.json('best.json')
        return self.workspace.path(best['files'][mapping[name]])


def read_map_config(path):
    return yaml.safe_load(path.read_text())


def map_png(path):
    buffer = io.BytesIO()
    with Image.open(path) as source:
        source.save(buffer, format='PNG')
    return buffer.getvalue()


def handler_for(dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            # Same-origin JSON only: no cross-site form/DNS-rebinding control.
            host = self.headers.get('Host')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if host not in allowed or self.headers.get('Origin') != 'http://'+host:
                self.send_error(403, 'Same-origin requests only')
                return
            if self.path != '/api/retry':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if self.headers.get('Content-Type') != 'application/json' or not 0 < length <= 256:
                    raise ValueError('Expected a small JSON request')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a JSON object')
                result, status = dashboard.retry(payload.get('id')), 202
            except (ValueError, TypeError) as error:
                result, status = dict(error=str(error)), 400
            except (OSError, RuntimeError) as error:
                result, status = dict(error=str(error)), 503
            body = json_bytes(result)
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Localhost-only, no CORS or generic file serving.
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed:
                self.send_error(403, 'Localhost only')
                return
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://{value}' for value in allowed}:
                self.send_error(403, 'Same-origin requests only')
                return
            path = urlsplit(self.path).path
            disposition = None
            try:
                if path == '/':
                    body, mime = HTML.read_bytes(), 'text/html; charset=utf-8'
                elif path == '/api/snapshot':
                    body, mime = json_bytes(dashboard.snapshot()), 'application/json'
                elif path == '/api/track':
                    body, mime = json_bytes(dashboard.track()), 'application/json'
                elif path == '/api/health':
                    body, mime = json_bytes(dict(workspace=str(dashboard.workspace.root))), 'application/json'
                elif path == '/map.png':
                    with dashboard.lock:
                        body, _ = dashboard.map_payload()
                    mime = 'image/png'
                elif path.startswith('/download/'):
                    with dashboard.lock:
                        selected = dashboard.download(path.removeprefix('/download/'))
                        body = selected.read_bytes()
                    mime = 'application/json' if selected.suffix == '.json' else 'text/csv'
                    disposition = f'attachment; filename="{selected.name}"'
                else:
                    self.send_error(404)
                    return
            except (OSError, KeyError, ValueError, TypeError) as error:
                body, mime = json_bytes(dict(error=str(error))), 'application/json'
                self.send_response(503)
            else:
                self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                             "style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if disposition:
                self.send_header('Content-Disposition', disposition)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def select_workspace(value):
    if value:
        root = Path(value).resolve()
    else:
        choices = [path.parent for path in (REPO/'live_runs').glob('optimizer_*/optimizer.json')]
        if len(choices) != 1:
            raise ValueError('Specify a workspace directory. Found: '+', '.join(str(v) for v in choices))
        root = choices[0]
    if not (root/'optimizer.json').is_file():
        raise ValueError(f'No optimizer.json in {root}')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', nargs='?', help='optimizer output; auto-select only if exactly one exists')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--serve', action='store_true', help='run the read-only HTTP observer in this terminal')
    parser.add_argument('--no-open', action='store_true', help='print URL without opening a browser')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    try:
        root = select_workspace(args.workspace)
    except ValueError as error:
        parser.error(str(error))
    url = f'http://127.0.0.1:{args.port}/'
    if args.serve:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(Dashboard(root)))
        server.daemon_threads = True
        print(f'Read-only optimizer dashboard: {url}\nWorkspace: {root}', flush=True)
        try:
            server.serve_forever(poll_interval=.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    try:
        with urlopen(url+'api/health', timeout=1) as response:
            existing = json.load(response)
    except OSError:
        existing = None
    if existing is not None and existing.get('workspace') != str(root):
        parser.error(f'Port {args.port} serves another workspace; select --port with an unused port')
    unit = 'sdu-optimizer-dashboard-'+hashlib.sha256(f'{root}:{args.port}'.encode()).hexdigest()[:12]
    if existing is None:
        import sys
        subprocess.run(['systemctl', '--user', 'reset-failed', unit+'.service'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(['systemd-run', '--user', f'--unit={unit}', '--property=Nice=10',
                        '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
                        sys.executable, '-u', str(Path(__file__).resolve()), str(root),
                        '--serve', '--port', str(args.port)], check=True)
        for _ in range(30):
            try:
                with urlopen(url+'api/health', timeout=.5) as response:
                    if json.load(response).get('workspace') != str(root):
                        raise RuntimeError('Another dashboard has claimed the port')
                break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError(f'Dashboard did not start; inspect journalctl --user -u {unit}')
    print(f'Dashboard: {url}\nRead-only. Closing the browser does not stop IPOPT.\n'
          f'Stop this viewer only: systemctl --user stop {unit}.service')
    if not args.no_open:
        webbrowser.open(url)


if __name__ == '__main__':
    main()
