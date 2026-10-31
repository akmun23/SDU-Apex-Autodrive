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
try:
    from .source_witness_catalog import certified_catalog
    from .raceline_clearance import declared_tolerance, NUMERICAL_TOLERANCE_M
except ImportError:  # Standalone python tools/optimizer_dashboard.py
    from source_witness_catalog import certified_catalog
    from raceline_clearance import declared_tolerance, NUMERICAL_TOLERANCE_M


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


def candidate_acceptance(checkpoint, replay, report, *, predicting=False, margin=.05):
    """Explain current-solve gates without borrowing the incumbent's certificate.

    An early replay checks its own iterate, not necessarily best_iterate.npz.
    Missing measurements stay pending; a small NLP residual proves no wall gap.
    """
    checkpoint, report = checkpoint or {}, report or {}
    residual = finite(report.get('max_constraint_violation', checkpoint.get('residual')))
    threshold = 1.e-4 if predicting else 1.e-5
    gates = []

    def gate(ident, label, value, limit, comparison='<=', unit='', detail='', tolerance_m=0.):
        value = finite(value)
        passed = value <= limit if value is not None and comparison == '<=' else (
            value >= limit-tolerance_m if value is not None else None)
        gates.append(dict(id=ident, label=label, status=(
            'pending' if passed is None else 'passed' if passed else 'failed'),
            value=value, limit=limit, comparison=comparison, unit=unit, detail=detail,
            tolerance_m=tolerance_m))

    gate('constraints', 'Relaxed predictor constraints' if predicting else 'Model constraints',
         residual, threshold,
         detail=('Completed selected witness' if report else
                 f"Saved checkpoint, iteration {checkpoint.get('iteration', 'unknown')}"))
    if predicting:
        return dict(status='initializer_only', gates=gates,
            summary='Predictor only: even passing relaxed constraints cannot certify a raceline. Exact-grid repair and source replay are required.',
            checkpoint_iteration=checkpoint.get('iteration'), replay_iteration=None,
            replay_matches_checkpoint=False)

    rollout = report.get('rollout') or {}
    origin = 'Completed selected witness' if rollout else (
        f"Latest early replay, iteration {replay.get('iteration')}" if replay else 'No current-solve replay recorded')
    replay_iteration = replay.get('iteration') if replay and not rollout else None
    sample = (dict(clearance_slack_m=rollout.get('minimum_1ms_footprint_slack_m'),
                   source_xy_error_m=rollout.get('full_lap_xy_max_m'),
                   periodic_scaled_defect=report.get('max_periodicity_defect'))
              if rollout else replay or {})
    gate('closure', 'Periodic lap closure', sample.get('periodic_scaled_defect'),
         1.e-5, detail=origin)
    if rollout and finite(report.get('max_constraint_violation')) is not None:
        # A completed OCP residual already includes its periodic equalities;
        # it is not a separate measured closure error, so invent no value.
        if gates[-1]['value'] is None:
            gates[-1].update(status='passed' if residual <= 1.e-5 else 'failed',
                             detail='Included in completed model-constraint check')
    gate('drift', 'Full-lap recursive XY drift', sample.get('source_xy_error_m'),
         .05, unit='m', detail=origin)
    extra = finite(sample.get('clearance_slack_m'))
    target = finite(sample.get('target_slack_m')) or 0.
    tolerance=(declared_tolerance(report if report else replay)
               if report or replay else NUMERICAL_TOLERANCE_M)
    gate('clearance', 'Conservative map clearance (1 ms)',
         None if extra is None else margin+extra, margin+target,
         comparison='>=', unit='m', detail=origin, tolerance_m=tolerance or 0.)
    if tolerance is None:
        gates[-1].update(status='failed',detail=origin+'; invalid declared numerical tolerance')
    elif extra is not None:
        # Match the verifier's raw-slack comparison even on the tolerance boundary.
        gates[-1]['status']='passed' if extra>=target-tolerance else 'failed'
    # A solve report is a preliminary source gate, not the separately exported
    # campaign certificate. Do not mark that later stage complete prematurely.
    gates.append(dict(id='archive', label='Independent certificate and archive',
        status='pending', value=None, limit=None, comparison=None, unit='',
        detail='Runs after solve completion; no current-solve export confirmed here'))
    failures = []
    for g in gates:
        if g['status'] != 'failed':
            continue
        text = g['label']
        if g['id'] == 'clearance':
            miss = g['limit']-g['value']
            scale, unit = ((1.e3, 'mm') if miss >= 1.e-3 else
                           (1.e6, 'µm') if miss >= 1.e-6 else
                           (1.e9, 'nm') if miss >= 1.e-9 else (1.e12, 'pm'))
            text += f" (short by {miss*scale:.2f} {unit})"
        failures.append(text)
    if failures:
        summary = 'Not accepted: '+', '.join(failures)+'.'
    elif any(g['status']=='pending' for g in gates[:-1]):
        summary = 'Not accepted yet: replay/closure checks are still pending for this solve.'
    else:
        summary = 'Source checks passed; independent certification and archiving are still pending.'
    return dict(status='blocked' if failures else 'pending', summary=summary, gates=gates,
                measurement=(rollout.get('clearance_measurement') if rollout else
                             (replay or {}).get('clearance_measurement')),
                clearance_numerical_tolerance_m=tolerance,
                checkpoint_iteration=checkpoint.get('iteration'), replay_iteration=replay_iteration,
                replay_matches_checkpoint=(replay_iteration is not None and
                    replay_iteration == checkpoint.get('iteration')))


def describe_solve(solver, active, acceptance, points, checkpoint, report, iteration):
    """Separate the optimization goal from evidence about particular iterates.

    Feasibility is not optimality. Only compare objective values from this
    solve's unscaled callbacks; IPOPT's displayed objective may be scaled.
    """
    mode = 'time' if active.get('time_predictor') else solver.get('objective_mode')
    lap = finite(active.get('trial_lap_s'))
    quality = mode in ('clearance', 'smooth')
    label, goal = {
        'clearance': ('Increase wall clearance',
                      'Increase minimum wall clearance; command smoothness is secondary.'),
        'smooth': ('Smooth steering and throttle',
                   'Reduce steering and throttle changes while retaining model feasibility.'),
        'time': ('Explore a shorter lap time',
                 'Optimize lap duration; this predictor still needs exact-command repair and replay.'),
        'repair': ('Find a model-feasible trajectory',
                   'Repair the trajectory at the requested duration and clearance.'),
    }.get(mode, ('Preparing the next solve', 'Waiting for this solve\'s objective to be reported.'))
    if mode == 'repair' and active.get('strategy') == 'adaptive':
        label = 'Repair an alternative corner trajectory'
        goal = 'Make the adaptive corner candidate feasible, including the added 1 ms wall checks.'
    elif mode == 'repair' and active.get('strategy') == 'proposal_restoration':
        label = 'Restore an improved quality proposal'
        goal = 'Restore model feasibility for an exploratory clearance or smoothness proposal.'
    fixed_duration = quality or (mode == 'repair' and 'index' in active)
    if lap is not None:
        goal = (f'Fixed lap time: {lap:.3f} s. ' if fixed_duration else
                f'Current trial: {lap:.3f} s. ')+goal
    why = ('Passing feasibility does not finish this task: the solver is searching for a better '
           'objective at the same lap time, not trying to make the starting point pass again.'
           if quality else 'A small model residual alone does not prove the full replay will pass.')
    if mode:
        why += (' The solve ends on convergence, configured stagnation or iteration limits, '
                'or an explicit replay/refinement interruption.')
    if report:
        why = 'Solve finished. The selected result must still be independently certified and archived.'

    checkpoint = checkpoint or {}
    saved = checkpoint.get('iteration')
    replay = acceptance.get('replay_iteration')
    evidence = (f'Live iteration: {iteration if iteration is not None else "not reported"}. '
                f'Saved checkpoint: {saved if saved is not None else "none"}. '
                f'Latest source replay: {replay if replay is not None else "none"}. '
                'Table results belong to the stated saved/replayed points, not automatically to the live iterate.')
    if report:
        evidence = 'The table describes the completed selected witness, not every iterate explored during the solve.'

    source_passed = all(g['status'] == 'passed' for g in acceptance.get('gates', [])[:4])
    seed_passed = (quality and not report and saved == 0 and replay == 0 and
                   len(acceptance.get('gates', [])) >= 4 and source_passed)
    progress = ('Starting point passed source checks at iteration 0. '
                'No improved feasible checkpoint has been saved yet.' if seed_passed else
                acceptance.get('summary', 'Waiting for verification evidence.'))
    if quality and not report and saved is not None and saved > 0:
        progress += ' A later saved checkpoint is not by itself a certified clearance or smoothness gain.'

    initial = next((p for p in points if p['iteration'] == 0), {})
    latest = points[-1] if points else {}
    saved_point = next((p for p in reversed(points) if p['iteration'] == saved), {})
    return dict(mode=mode, label=label, goal=goal, why_continuing=why,
                fixed_duration=fixed_duration, evidence=evidence, progress=progress,
                feasible_seed_checked=seed_passed,
                objective=dict(initial=finite(initial.get('objective')),
                    latest=finite(latest.get('objective')),
                    latest_iteration=latest.get('iteration'),
                    latest_residual=finite(latest.get('residual')),
                    saved=finite(saved_point.get('objective')), saved_iteration=saved))


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
        self.unidentified_header = False
        self.completed = False
        self.live_iteration = None
        self.ipopt = {}
        self.source_gate = None
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
                self.unidentified_header = False
                self.points.clear()
                self.ipopt_points.clear()
                self.solve = {}
                self.completed = False
                self.live_iteration = None
                self.ipopt = {}
                self.source_gate = None
            if item.get('stage') in ('production-command witness', 'minimum-time predictor setup'):
                # Headers own their identity: repair/restoration calls need not
                # have an outer campaign event, and the tail may omit that event.
                self.unidentified_header = bool(self.solve) and not item.get('output')
                if item.get('output'):
                    self.output = item['output']
                self.solve = item
                self.points.clear()
                self.ipopt_points.clear()
                self.live_iteration = None
                self.ipopt = {}
                self.completed = False
                self.source_gate = None
            if item.get('event') == 'early_source_checkpoint_gate':
                self.source_gate = dict(item, output=self.output)
            if item.get('stage') == 'minimum-time predictor':
                self.solve['predicted_lap_s'] = item.get('predicted_lap_s')
            if 'iteration' in item and 'max_constraint_violation' in item:
                self.live_iteration = int(item['iteration'])
                self.add_point(item['iteration'], item['max_constraint_violation'],
                               objective=item.get('objective'))
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
            self.unidentified_header = False
            self.live_iteration = None
            self.ipopt = {}
            self.completed = False
            self.source_gate = None
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


def mpc_failure_diagnostic(run_dir):
    """Extract the last exact-rollout rejection without loading a whole bag/log.

    The cycle sidecar is generated by the bag report decoder and carries the
    state and both candidate/prior prefix validation results. Keep only the
    last 256 KiB in memory; this remains cheap for multi-lap recordings.
    """
    path = Path(run_dir) / 'mpc_cycle_detail_diagnostics.jsonl'
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 256 * 1024))
            chunk = stream.read()
    except OSError:
        return None
    rows = []
    last_recovery = None
    for line in chunk.splitlines()[-256:]:
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        payload = row.get('payload') if isinstance(row, dict) else None
        if not isinstance(payload, dict):
            continue
        selection = payload.get('quality_aware_prefix_selection') or {}
        if selection.get('attempted') and selection.get('choice') in (
                'candidate', 'retained_plan', 'retained_recovery_prefix'):
            candidate_cost = finite(selection.get('candidate_objective'))
            prior_cost = finite(selection.get('prior_objective'))
            last_recovery = dict(
                progress_m=finite(payload.get('progress_m')),
                action_source=payload.get('action_source'),
                choice=selection.get('choice'),
                near_boundary_override=bool(
                    selection.get('near_boundary_override', False)),
                candidate_clearance_m=finite(
                    selection.get('candidate_minimum_corridor_slack_m')),
                prior_clearance_m=finite(
                    selection.get('prior_minimum_corridor_slack_m')),
                objective_ratio=(candidate_cost / prior_cost
                                 if candidate_cost is not None and
                                 prior_cost is not None and prior_cost > 0
                                 else None),
            )
        if payload.get('status') == 'rejected_nonlinear_rollout':
            rows.append(payload)
    if not rows:
        return None
    payload = rows[-1]
    state = payload.get('state')
    reference = payload.get('reference') or {}
    solver = payload.get('solver') or {}
    candidate = payload.get('candidate_prefix_revalidation') or {}
    prior = payload.get('prior_plan_revalidation') or {}
    slacks = payload.get('candidate_corridor_slack_by_stage_m') or []
    invalid_slacks = [item for item in slacks if isinstance(item, dict) and
                      any(finite(item.get(side)) is not None and
                          finite(item.get(side)) < 0.0 for side in ('lower', 'upper'))]
    state_values = state if isinstance(state, list) else []
    def state_at(index):
        return finite(state_values[index]) if len(state_values) > index else None
    return dict(
        progress_m=finite(payload.get('progress_m')),
        curvature_per_m=finite(payload.get('path_curvature_per_m')),
        cross_track_m=state_at(0), heading_error_rad=state_at(1),
        forward_speed_mps=state_at(2), lateral_speed_mps=state_at(3),
        yaw_rate_radps=state_at(4),
        reference_speed_mps=finite(reference.get('speed_reference_mps')),
        reference_acceleration_mps2=finite(reference.get('trajectory_acceleration_mps2')),
        candidate_invalid_stage=(invalid_slacks[0].get('stage') if invalid_slacks else None),
        candidate_invalid_slack_m=(min(
            [finite(invalid_slacks[0].get(side)) for side in ('lower', 'upper')
             if finite(invalid_slacks[0].get(side)) is not None], default=None)
            if invalid_slacks else None),
        candidate_failure_stage=finite(candidate.get('failure_stage')),
        prior_failure_stage=finite(prior.get('failure_stage')),
        iterations=finite(solver.get('iterations')),
        iteration_limit=finite(solver.get('max_iterations_configured')),
        admm_stop_reason=solver.get('admm_stop_reason'),
        primal_residual=finite(solver.get('primal_residual')),
        dual_residual=finite(solver.get('dual_residual')),
        end_to_end_control_ms=finite(payload.get('end_to_end_control_ms')),
        cycle_budget_ms=finite(payload.get('end_to_end_cycle_budget_ms')),
        candidate_exact_valid=candidate.get('exact_valid'),
        prior_exact_valid=prior.get('exact_valid'),
        last_recovery_decision=last_recovery,
    )


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
        self.cross_campaign_cache = None
        self.cross_campaign_scanned_at = 0.

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

    def log_identity(self, active):
        """Match a solve explicitly, or prove an in-flight refinement handoff.

        Already-running workers can have headers without an output path. In
        that case a same-duration guess is NOT sufficient: require the same
        polishing family, source, duration, margin and exact reported cut union
        (or a named quality proposal). New headers carry the path directly.
        """
        if not active.get('output') or not self.log.output:
            return None
        try:
            current = self.workspace.path(active['output'])
            announced = self.workspace.path(self.log.output)
            if not self.log.unidentified_header:
                return 'explicit output' if current == announced else None
            if self.log.solve.get('output') or current == announced:
                return None
            family = re.fullmatch(r'(polish_\d+)(?:_restore)?(?:_cut_\d+)?(?:_\d+)?', current.name)
            if (not family or current.parent != announced.parent or
                    not re.fullmatch(re.escape(family[1])+r'(?:_restore)?(?:_cut_\d+)?(?:_\d+)?', announced.name)):
                return None
            solver = self.log.solve
            cuts = {tuple(c) for c in active.get('source_clearance_cuts', [])}
            if (solver.get('source_clearance_cuts') != len(cuts) or
                    finite(solver.get('trial_lap_s')) != finite(active.get('trial_lap_s'))):
                return None
            for path in sorted(current.parent.glob(family[1]+'*/report.json')):
                if (path.parent == current or not re.fullmatch(
                        re.escape(family[1])+r'(?:_restore)?(?:_cut_\d+)?(?:_\d+)?', path.parent.name)):
                    continue
                parent = self.workspace.json(path)
                if (not solver.get('source_sha256') or
                        parent.get('source_plant_sha256') != solver['source_sha256'] or
                        finite(parent.get('trial_lap_s')) != finite(active.get('trial_lap_s')) or
                        finite(parent.get('required_margin_m')) != finite(active.get('margin'))):
                    continue
                requested = {tuple(c) for c in parent.get('requested_source_clearance_cuts', [])}
                is_restore = '_restore' in current.name
                parent_restore = '_restore' in path.parent.name
                if ('_cut_' in current.name and is_restore == parent_restore and requested and
                        cuts == requested | {tuple(c) for c in parent.get('source_clearance_cuts', [])}):
                    return 'verified refinement handoff'
                if (is_restore and '_cut_' not in current.name and
                        solver.get('objective_mode') == 'repair' and
                        parent.get('objective_mode') in ('clearance', 'smooth') and
                        active.get('proposal') and self.workspace.path(active['proposal']) ==
                        path.parent/'best_quality_proposal.npz'):
                    return 'verified proposal handoff'
        except (OSError, ValueError, TypeError):
            return None
        return None

    def snapshot(self):
        with self.lock:
            if self.snapshot_cache is not None and time.monotonic()-self.last_poll < 1.:
                return self.snapshot_cache
            self.warnings = []
            state = self.optional('optimizer.json')
            best = self.optional('best.json')
            portfolio = self.optional('portfolio.json')
            self.log.poll()
            now = time.time()
            active_name = state.get('active_start')
            campaign = (self.optional(f'starts/{active_name}/campaign/campaign.json')
                        if active_name else {})
            # Multi-incumbent campaigns run one child branch synchronously.
            # Prefer that branch's actual solver checkpoints, not a stale
            # parent solve or the old incumbent's metrics.
            elite_job = campaign.get('active_elite') or {}
            child_campaign = {}
            if elite_job.get('output'):
                try:
                    child_folder = self.workspace.path(elite_job['output'])
                    child_campaign = self.optional(child_folder/'campaign.json')
                except (ValueError, OSError) as error:
                    self.warnings.append(f'Elite branch inaccessible: {error}')
            active_campaign = child_campaign if child_campaign else campaign
            polishing = campaign.get('active_elite_polish') or {}
            polish_archive = {}
            if polishing.get('output'):
                try:
                    polish_archive=self.optional(
                        self.workspace.path(polishing['output'])/'elite.json')
                except (OSError, ValueError) as error:
                    self.warnings.append(f'Elite polishing state inaccessible: {error}')
            polish_active=polish_archive.get('active_attempt') or {}
            # Parent polishing may preempt a child branch retained for resume.
            # That child's pending predictor is not the currently running job.
            if polish_active:
                active_campaign = campaign
            predicting = None if polish_active else active_campaign.get('active_predictor')
            active = ((polish_active or predicting or active_campaign.get('active') or {})
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
            identity = self.log_identity(active)
            matches = identity is not None
            context['identity_source'] = identity
            # Bootstrap has no campaign 'solve' event; match its explicit duration.
            if not self.log.output and state.get('stage') == 'periodic_bootstrap':
                matches = self.log.solve.get('trial_lap_s') == state.get('trial_lap_s')
            if active.get('output') and self.log.solve and not matches:
                self.warnings.append('Live solver identity does not match the active manifest; '
                                     'withholding unassociated iteration and replay data.')
            solver = dict(self.log.solve) if matches else {}
            points = list(self.log.points) if matches else []
            ipopt_points = list(self.log.ipopt_points) if matches else []
            trial = TRIAL.fullmatch(active.get('key', ''))
            trial_s = finite(solver.get('predicted_lap_s') if predicting else solver.get('trial_lap_s'))
            if trial_s is None:
                trial_s = int(trial[1])*.025 if trial else finite(active.get('trial_lap_s'))
            if trial_s is None:
                trial_s = (
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
            source_digest=validation.get('source_plant_sha256')
            map_file=self.workspace.root/'input/map.pgm'
            if now-self.cross_campaign_scanned_at>20. or self.cross_campaign_cache is None:
                try:
                    map_digest=(hashlib.sha256(map_file.read_bytes()).hexdigest()
                                if map_file.is_file() else None)
                    self.cross_campaign_cache=certified_catalog(
                        self.workspace.root.parent,source_sha256=source_digest,
                        map_sha256=map_digest,max_results=25)
                except (OSError,ValueError) as error:
                    self.warnings.append(f'Historical certificate scan unavailable: {error}')
                    self.cross_campaign_cache=dict(candidates=[],pareto=[],total=0)
                self.cross_campaign_scanned_at=now
            historic=self.cross_campaign_cache
            proc = process_info(state.get('pid'))
            status = state.get('status', 'waiting')
            if status == 'running' and not proc['alive']:
                status = 'not_running'
            age = max(0., now-self.log.log_mtime) if self.log.log_mtime is not None else None
            log_status = ('not_running' if not proc['alive'] else
                          'quiet' if age is None or age > 90 else 'active')
            attempts = active_campaign.get('attempts', [])
            elites = []
            for archive_path in (campaign.get('elite_archives') or {}).values():
                try:
                    archive = self.optional(archive_path)
                    elites.extend(archive.get('accepted', []))
                except (OSError, ValueError) as error:
                    self.warnings.append(f'Elite archive unavailable: {error}')
            elite_progress = dict(
                enabled=bool(state.get('contract', {}).get('elite_refinements')),
                polishing=polishing,
                polishing_attempts=len(polish_archive.get('attempts',[])),
                polishing_total=(sum(len(polish_archive.get('contract',{}).get(key,[]))
                                    for key in ('quality_objectives','control_seeds',
                                                'extra_margin_m','geometry_seeds'))
                                 +polish_archive.get('contract',{}).get('adaptive_regions',0)),
                adaptive_regions=polish_archive.get('contract',{}).get('adaptive_regions',0),
                adaptive_hotspots=polish_archive.get('adaptive_hotspots',[]),
                adaptive_history=[x for x in polish_archive.get('attempts',[])
                                  if x.get('strategy')=='adaptive'][-8:],
                polishing_active=polish_active,
                parent_attempts=len(campaign.get('attempts', [])),
                current_attempts=len(attempts),
                archived_alternatives=len(elites)+len(polish_archive.get('accepted',[])),
                active_branch=elite_job,
                schedule=campaign.get('elite_schedule', {}),
                child_status=child_campaign.get('status') if child_campaign else None,
                explored_basins=(len(set((campaign.get('elite_schedule') or {}).get('visits', {})) |
                                      ({elite_job['sha256']} if elite_job.get('sha256') else set()))
                                 if campaign.get('elite_schedule') else
                                 len(campaign.get('elite_explored', []))),
                failed_time_starts=sum(not a.get('certified') for a in
                    campaign.get('attempts', [])+(attempts if child_campaign else [])),
                archive_count=len(campaign.get('elite_archives', {})),
                historic_certified_count=historic.get('total',0),
                historic_pareto_count=historic.get('pareto_count',0),
                historical_quality=historic,
                current_saved_quality=report.get('quality') if report else None,
                recent_alternatives=(elites+polish_archive.get('accepted',[]))[-6:])
            context.update(trial_lap_s=trial_s, phase=int(trial[2]) if trial else None,
                           strategy=('minimum-time predictor' if predicting else
                                     trial[3] if trial else active.get('strategy')),
                           time_predictor=bool(predicting))
            # Explain real gains separately from solver activity. The local
            # portfolio is exported from this exact map/source workspace.
            roomiest = portfolio.get('roomiest')
            if not roomiest:
                roomiest = max(elites+polish_archive.get('accepted', []),
                    key=lambda e: e.get('clearance_slack_m', 0.), default=None)
            required = finite(validation.get('required_margin_m'))
            required = .05 if required is None else required
            fastest_slack = finite((validation.get('rollout') or {}).get(
                'minimum_1ms_footprint_slack_m'))
            speed_floor, last_speed_gain = float('inf'), None
            for item in history:
                if item['lap_s'] < speed_floor-1.e-9:
                    speed_floor, last_speed_gain = item['lap_s'], item['epoch_s']
            progress = dict(
                nlp_threshold=1.e-4 if predicting else 1.e-5,
                trial_certified=False,
                fastest_clearance_cm=None if fastest_slack is None else
                    (required+fastest_slack)*100.,
                roomiest_lap_s=roomiest.get('lap_s') if roomiest else None,
                roomiest_clearance_cm=(
                    (roomiest.get('required_margin_m', required)+
                     roomiest['clearance_slack_m'])*100. if roomiest else None),
                last_speed_gain_age_s=(now-last_speed_gain if last_speed_gain else None),
                certified_choices=portfolio.get('certified_count', len(history)),
                pareto_choices=len(portfolio.get('pareto', [])))
            current_gate = self.log.source_gate if matches else None
            replay_margin = next((value for value in (
                finite((report or {}).get('required_margin_m')), finite(active.get('margin')),
                finite(active_campaign.get('contract', {}).get('required_margin_m')), required)
                if value is not None), required)
            progress['acceptance'] = candidate_acceptance(
                checkpoint, current_gate, report, predicting=bool(predicting),
                margin=replay_margin)
            progress['task'] = describe_solve(
                solver, context, progress['acceptance'], points, checkpoint, report,
                self.log.live_iteration if matches else None)
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
                run_id = compact.get('run_id') or ''
                # Only read artifacts from explicitly named auto-trial directories.
                if re.fullmatch(r'apex_mpc_[a-f0-9]{16}_a[0-9]{2,}', run_id):
                    run_dir = (REPO/'live_runs'/run_id).resolve()
                    if run_dir.is_relative_to((REPO/'live_runs').resolve()):
                        failure = mpc_failure_diagnostic(run_dir)
                        if failure is not None:
                            compact['failure_diagnostic'] = failure
                    if compact['status'] == 'running':
                        watcher = run_dir/'watcher.log'
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
                          elite_progress=elite_progress,
                          search_progress=progress,
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
            portfolio = self.optional('portfolio.json')
            roomiest = portfolio.get('roomiest') or {}
            if roomiest.get('manifest'):
                room = self.optional(roomiest['manifest'])
                room_reference = room.get('files', {}).get('raceline')
                if room_reference and room_reference != reference:
                    layers.append(dict(name='Roomiest certified alternative', kind='roomiest',
                        points=self.workspace.read(room_reference, reference_points)))
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
