"""Map-only, restartable production-model raceline search.

PGM + ROS YAML -> centerline -> geometry starts -> periodic command bootstrap
-> adaptive IPOPT campaign -> independently checked best-so-far exports.
No controller, recording, previous raceline, or target lap time is required.
"""
from __future__ import annotations

import argparse
import ctypes as C
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

from command_campaign import run_campaign, unused_directory, budget_increase_only
from command_ocp import (certify_witness, production_command_map, save_command_checkpoint,
                         save_json_checkpoint, solve_witness)
from physics import SourcePlant

REPO = Path(__file__).resolve().parents[1]
DT = .025


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_map(path, metadata=None):
    """Resolve only the supplied map pair; never select another track's seed."""
    path = Path(path).expanduser().resolve()
    metadata = (Path(metadata).expanduser().resolve() if metadata else
                path if path.suffix.lower() in ('.yaml', '.yml') else path.with_suffix('.yaml'))
    config = yaml.safe_load(metadata.read_text())
    if not isinstance(config, dict) or 'image' not in config:
        raise ValueError('map YAML requires image, resolution and origin')
    image = Path(config['image'])
    if not image.is_absolute():
        image = metadata.parent/image
    image = image.resolve()
    if path.suffix.lower() not in ('.yaml', '.yml'):
        if file_hash(path) != file_hash(image):
            raise ValueError('supplied image does not match the YAML image')
        image = path
    if (not math.isfinite(float(config.get('resolution', 0.))) or
            float(config.get('resolution', 0.)) <= 0.):
        raise ValueError('map resolution must be finite and positive')
    origin = np.asarray(config.get('origin', []), dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('map origin must contain finite x, y, yaw')
    return image, metadata, config


def default_output(image, config):
    key = hashlib.sha256(Path(image).read_bytes()+
                         json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    return REPO/'live_runs'/f'optimizer_{Path(image).stem}_{key}'


def implementation_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in (Path(__file__), root/'command_ocp.py', root/'command_campaign.py',
                 root/'map_track.py', root/'min_curvature_seed.py', root/'physics.py',
                 root/'path_frame.py', root/'command_realizability.py',
                 root.parent/'tools/build_mapped_raceline.py'):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def bundle_map(image, config, assets):
    """Never associate a fresh map contract with stale/incomplete workspace input."""
    import cv2
    assets.mkdir(parents=True, exist_ok=True)
    bundled_image, bundled_yaml = assets/'map.pgm', assets/'map.yaml'
    expected = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if expected is None or expected.ndim != 2:
        raise ValueError('input map must be a readable grayscale image')
    expected_config = dict(config, image='map.pgm')
    if bundled_image.exists():
        actual = cv2.imread(str(bundled_image), cv2.IMREAD_UNCHANGED)
        if actual is None or not np.array_equal(actual, expected):
            raise ValueError('workspace contains a different map image; choose a new output directory')
    if bundled_yaml.exists() and yaml.safe_load(bundled_yaml.read_text()) != expected_config:
        raise ValueError('workspace contains different map metadata; choose a new output directory')
    if not bundled_image.exists():
        temporary = assets/'map.tmp.pgm'
        if not cv2.imwrite(str(temporary), expected):
            raise ValueError('cannot bundle input map')
        os.replace(temporary, bundled_image)
    if not bundled_yaml.exists():
        save_json_checkpoint(bundled_yaml, expected_config)
    return bundled_image, bundled_yaml


def verified_unity_direction(reference):
    """Get travel orientation from a previously COMPLETED official Unity lap.

    The ROS map's mathematical winding is not the competition driving-direction
    convention. A bare occupancy grid has no authoritative travel direction.
    """
    reference = Path(reference).resolve(strict=True)
    table = np.loadtxt(reference, delimiter=',', comments='#')
    if table.ndim != 2 or table.shape[1] < 4 or len(table) < 25:
        raise ValueError('Unity driving reference must contain a complete race loop')
    xy = table[:, 1:3]
    yaw = np.unwrap(table[:, 3])
    if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        raise ValueError('non-finite driving reference')
    winding = round((yaw[-1]-yaw[0])/(2*math.pi))
    signed_area_twice = float(np.sum(xy[:, 0]*np.roll(xy[:, 1],-1) -
                                    xy[:, 1]*np.roll(xy[:, 0],-1)))
    if abs(winding) != 1 or winding*signed_area_twice <= 0.:
        raise ValueError('ambiguous Unity-verified direction or degenerate reference')
    geometry_orientation = 'counterclockwise' if winding > 0 else 'clockwise'
    return dict(reference=str(reference), sha256=file_hash(reference),
                winding=winding, geometry_orientation=geometry_orientation,
                signed_area_m2=signed_area_twice/2)


def require_travel_orientation(witness, winding, *, verified=False):
    with np.load(witness, allow_pickle=False) as data:
        x = np.asarray(data['state'])
    if x.ndim != 2 or x.shape[0] != 21 or not np.isfinite(x).all():
        raise ValueError('invalid direction witness')
    area_twice = float(np.sum(x[0, :-1]*x[1, 1:]-x[1, :-1]*x[0, 1:]))
    actual = round((x[2,-1]-x[2,0])/(2*math.pi))
    if actual != winding or area_twice*winding <= 0.:
        if verified:
            raise ValueError('witness travels opposite to verified Unity racing direction')
        raise ValueError('only clockwise racelines may be exported')


def source_parameters(repo):
    """Read initializer constants from the same plant, not a second config."""
    source = (repo/'f1tenth_mpc/src/unity_source_plant.cpp').read_text()
    names = ('kWheelbaseM', 'kWheelRadiusM', 'kComForwardFromRearAxleM',
             'kWheelDampingRate', 'kMotorTorqueNm')
    result = {}
    for name in names:
        value = re.search(r'constexpr double '+name+r'\s*=\s*([\d.eE+-]+)\s*;', source)
        if value is None:
            raise ValueError(f'cannot read source initializer parameter {name}')
        result[name] = float(value[1])
    return result


def geometric_seed(reference, destination, repo, speed_scale=1.):
    """Explicitly uncertified initial guess; IPOPT must solve every transition.

    Geometry supplies tangent/yaw and a kinematic initialization, never fitted
    vehicle response or measurement data. Jounce is initialized by our source
    implementation; its full history remains in the shooting states.
    """
    table = np.loadtxt(reference, delimiter=',', comments='#')
    if (table.ndim != 2 or table.shape[1] < 9 or len(table) < 12 or
            not np.isfinite(table).all() or np.any(table[:, 5] <= 0.)):
        raise ValueError('invalid geometry initializer')
    xy = table[:, 1:3]
    closed = np.vstack([xy, xy[0]])
    ds = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(ds <= 1.e-8):
        raise ValueError('degenerate geometry initializer')
    station = np.r_[0., np.cumsum(ds)]
    speed = np.clip(table[:, 5]*speed_scale, .5, 12.)
    clock = np.r_[0., np.cumsum(2*ds/(speed+np.roll(speed, -1)))]
    count = max(30, math.ceil(clock[-1]/DT))
    query = np.arange(count+1)*clock[-1]/count
    s = np.interp(query, clock, station)
    curve = CubicSpline(station, closed, bc_type='periodic')
    point, tangent, second = curve(s), curve(s, 1), curve(s, 2)
    yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    winding = round((yaw[-1]-yaw[0])/(2*math.pi))
    if abs(winding) != 1:
        raise ValueError('map centerline must describe one closed circuit')
    curvature = (tangent[:, 0]*second[:, 1]-tangent[:, 1]*second[:, 0]) / np.maximum(
        np.linalg.norm(tangent, axis=1)**3, 1.e-12)
    u = np.interp(s, station, np.r_[speed, speed[0]])
    p = source_parameters(repo)
    r = u*curvature
    steering = np.clip(np.arctan(p['kWheelbaseM']*curvature), -math.pi/6, math.pi/6)
    q = np.clip(4*p['kWheelDampingRate']*u/(p['kWheelRadiusM']*p['kMotorTorqueNm']), .001, 1.)
    plant = SourcePlant(repo, repo/'build_task/source_witness')
    plant.reset(0., 0., 0., float(np.mean(u)))
    for _ in range(20):
        plant.step(0., float(np.mean(q)))
    physical = np.repeat(np.asarray(plant.input.state[:18])[:, None], count+1, axis=1)
    physical[0], physical[1], physical[2] = u, p['kComForwardFromRearAxleM']*r, r
    physical[3:7] = u/p['kWheelRadiusM']
    physical[13] = steering
    reconcile = plant.lib.unity_source_plant_reconcile_suspension_jounce_c
    reconcile.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_int)]
    reconcile.restype = C.c_int
    for k in range(count+1):
        raw = (C.c_double*26)(*physical[:, k], *([0.]*8))
        initialized = C.c_int(0)
        if not reconcile(raw, C.byref(initialized)):
            raise ValueError('source suspension initialization failed')
        physical[14:18, k] = raw[14:18]
    states = np.vstack([point.T, yaw, physical])
    states[:, -1] = states[:, 0]
    states[2, -1] += winding*2*math.pi
    controls = np.vstack([steering[:-1], q[:-1]])
    save_command_checkpoint(destination, states, controls,
                            'uncertified-map-geometric-initializer', float('inf'), -1)
    return count


def publish_best(output, witness, lap_s, state, *, source_digest,
                 verified_direction=None):
    """Publish only independently certified immutable bundles; atomic pointer."""
    report_path = witness.with_name('report.json')
    checked = json.loads(report_path.read_text())
    if (checked.get('status') != 'source_witness_passed' or
            checked.get('source_plant_sha256') != source_digest):
        raise ValueError('cannot publish an uncertified or stale-source witness')
    require_travel_orientation(witness,
        verified_direction['winding'] if verified_direction else -1,
        verified=verified_direction is not None)
    if state.get('best_lap_s') is not None and lap_s >= state['best_lap_s']-1.e-9:
        return
    directory = output/'exports'/file_hash(witness)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    files = {'witness': witness, 'validation': report_path}
    reference = checked.get('reference_export', {})
    for key in ('source_witness_csv', 'path'):
        if reference.get(key):
            files['raceline_with_witness' if key == 'source_witness_csv' else 'raceline'] = Path(reference[key])
    # Current export_reference calls its plain CSV 'csv'; keep schema explicit.
    if reference.get('csv'):
        files['raceline'] = Path(reference['csv'])
    published = {}
    for key, source in files.items():
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = directory/source.name
        if not target.exists():
            temporary = target.with_name(target.name+'.tmp')
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        published[key] = str(target)
    if 'raceline' not in published:
        raise ValueError('certified witness has no exported raceline')
    summary = dict(lap_s=lap_s, files=published, witness_sha256=file_hash(witness),
                   source_plant_sha256=source_digest,
                   direction='unity_verified' if verified_direction else 'clockwise',
                   direction_reference_sha256=(verified_direction['sha256'] if verified_direction else None),
                   source_only=True,
                   unity_validated=False, mpc_validated=False, runtime_promoted=False)
    save_json_checkpoint(directory/'manifest.json', summary)
    save_json_checkpoint(output/'best.json', summary)
    state.update(best_lap_s=lap_s, best_manifest=str(directory/'manifest.json'))


def require_clockwise(witness):
    """Historical map-only default; prefer verified Unity reference on IROS."""
    return require_travel_orientation(witness, -1)


def geometry_starts(map_yaml, output, directions, cancel):
    """Extract the map centerline, then vary curvature/length initialization."""
    spec = importlib.util.spec_from_file_location('mapped_raceline', REPO/'tools/build_mapped_raceline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    starts = []
    for direction in directions:
        if cancel(): raise KeyboardInterrupt
        folder = output/direction
        report = folder/'report.json'
        reference = folder/'initial_raceline.csv'
        centerline = folder/'centerline.csv'
        if not report.exists() or not centerline.exists():
            try:
                module.build(argparse.Namespace(map=map_yaml, output_dir=folder,
                    speed=1.2, direction=direction, driven_trace=None, trace_cols=None))
            except (ValueError, RuntimeError) as error:
                print(json.dumps(dict(event='direction_start_rejected',
                    direction=direction, reason=str(error))), flush=True)
                continue
        starts.append((direction, reference))
        # Geometry objective weights define different starting basins only.
        # They are NOT lateral-acceleration or vehicle-feasibility limits.
        from min_curvature_seed import generate
        for penalty in (.05, .2):
            if cancel(): raise KeyboardInterrupt
            path = folder/f'length_{penalty:g}.csv'
            if not path.exists():
                try:
                    generate(map_yaml.with_suffix('.pgm'), centerline, path,
                        spacing=min(.12, float(json.loads(report.read_text())['map_resolution_m'])*2.4),
                        margin=.05, max_offset=2.5, offset_step=.025,
                        harmonics=16, midline_smoothing_m=.24,
                        length_penalty_per_m2=penalty, preserve_source_geometry=True)
                except (RuntimeError, ValueError) as error:
                    print(json.dumps(dict(event='geometry_start_rejected',
                        direction=direction, penalty=penalty, reason=str(error))), flush=True)
                    if path.exists():
                        path.rename(unused_directory(path.with_suffix('.rejected.csv')))
                    continue
            starts.append((f'{direction}_length_{penalty:g}', path))
    return starts


def run_track(*, image, metadata, config, output, workers=8, iterations=1500,
              phases=4, refinements=6, stagnation_iterations=250, prepare_only=False,
              bootstrap_only=False, cancel=None, driving_reference=None, hybrid_time=False,
              command_seed=None):
    cancel = cancel or (lambda: False)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output/'optimizer.json'
    verified = verified_unity_direction(driving_reference) if driving_reference else None
    geometry_direction = verified['geometry_orientation'] if verified else 'clockwise'
    _, digest = production_command_map(REPO, REPO/'build_task/command_ocp')
    contract = dict(image_sha256=file_hash(image), metadata_sha256=file_hash(metadata),
        source_sha256=digest, optimizer_sha256=implementation_digest(),
        workers=workers, iterations=iterations, stagnation_iterations=stagnation_iterations, phases=phases,
        refinements=refinements,
        direction='unity_verified' if verified else 'clockwise',
        direction_reference_sha256=verified['sha256'] if verified else None,
        expected_winding=verified['winding'] if verified else -1, schema=2 if verified else 1)
    contract['hybrid_time_search'] = hybrid_time
    if command_seed is not None:
        command_seed = Path(command_seed).resolve()
        contract['command_seed_sha256'] = file_hash(command_seed)
    state = json.loads(status_path.read_text()) if status_path.exists() else dict(
        schema='sdu-map-optimizer/v1', contract=contract, best_lap_s=None,
        best_manifest=None, starts={}, status='running', source_only=True,
        global_minimum_proven=False, runtime_promoted=False)
    if state['contract'] != contract:
        if not budget_increase_only(state['contract'], contract):
            raise ValueError('map/source/settings changed; choose a new output directory')
        state.setdefault('budget_updates', []).append(dict(
            previous=state['contract'], requested=contract, epoch_s=time.time()))
        state['contract'] = contract
        for context in state['starts'].values():
            if context['witness'] is None:
                context.setdefault('budget_limited_attempts', []).extend(context['attempts'])
                context['attempts'] = []
    if state.get('best_manifest'):
        previous_best = json.loads(Path(state['best_manifest']).read_text())
        require_travel_orientation(Path(previous_best['files']['witness']),
            verified['winding'] if verified else -1, verified=verified is not None)
    def save(**values):
        state.update(values, pid=os.getpid(), updated_epoch_s=time.time())
        save_json_checkpoint(status_path, state)
    def event(stage, **values):
        save(stage=stage, **values)
        print(json.dumps(dict(stage=stage, best_lap_s=state['best_lap_s'], **values)), flush=True)
    def promote(path, lap):
        publish_best(output, path, lap, state, source_digest=digest,
                     verified_direction=verified)
        save()
    try:
        save(status='running')
        # Bundle metric metadata as well as pixels; another track's origin or
        # resolution must never be silently paired with a saved shooting state.
        bundled_image, bundled_yaml = bundle_map(image, config, output/'input')
        event('source_seed_validation' if command_seed else 'centerline_and_geometry')
        starts = ([('source_seed', command_seed)] if command_seed else
                  geometry_starts(bundled_yaml, output/'geometry', (geometry_direction,), cancel))
        if verified:
            # User-facing basin names describe simulator driving authority, not
            # the map XY coordinate system's mathematical winding.
            starts = [(name.replace(geometry_direction, 'unity_race_direction'),path)
                      for name,path in starts]
        if not starts:
            raise ValueError('no safe closed-circuit geometry extracted from supplied map')
        save(geometry_starts=[] if command_seed else [dict(name=name, csv=str(path)) for name, path in starts])
        if prepare_only:
            event('prepared', status='prepared')
            return state
        for name, reference in starts:
            if cancel(): raise KeyboardInterrupt
            current = state['starts'].setdefault(name, dict(attempts=[], witness=None))
            base = output/'starts'/name
            base.mkdir(parents=True, exist_ok=True)
            if command_seed is not None and current['witness'] is None:
                immutable = output/'input/source_seed.npz'
                if not immutable.exists():
                    shutil.copyfile(command_seed, immutable)
                if file_hash(immutable) != contract['command_seed_sha256']:
                    raise ValueError('workspace contains a different command seed')
                checked_dir = unused_directory(base/'certified')
                checked = certify_witness(REPO, immutable, bundled_image, checked_dir)
                if checked['status'] != 'source_witness_passed':
                    raise ValueError('provided command seed fails current map/source certification')
                current['witness'] = str(checked_dir/'witness.npz')
                promote(Path(current['witness']), checked['trial_lap_s'])
                save()
            if current['witness'] is None:
                for scale in (1., 1.5, .75):
                    key = f'speed_{scale:g}'
                    previous = current.get('active')
                    if key in current['attempts']:
                        continue
                    folder = base/key
                    folder.mkdir(exist_ok=True)
                    seed = folder/'initializer.npz'
                    if not seed.exists():
                        geometric_seed(reference, seed, REPO, scale)
                    if previous and previous['key'] == key:
                        last = Path(previous['output'])
                        if (last/'report.json').exists():
                            seed = last/'witness.npz'
                        elif (last/'best_iterate.npz').exists():
                            seed = last/'best_iterate.npz'
                        elif previous.get('seed'):
                            seed = Path(previous['seed'])
                    with np.load(seed, allow_pickle=False) as data:
                        count = data['controls'].shape[1]
                    cuts = set(tuple(v) for v in (previous or {}).get('cuts', [])) if (
                        previous and previous['key'] == key) else set()
                    best_residual = float('inf')
                    for repair in range(refinements+1):
                        if cancel(): raise KeyboardInterrupt
                        destination = unused_directory(folder/'solve')
                        current['active'] = dict(key=key, output=str(destination),
                                                 seed=str(seed), cuts=sorted(cuts))
                        event('periodic_bootstrap', active_start=name, intervals=count,
                              trial_lap_s=count*DT, initializer=str(seed), repair=repair)
                        report = solve_witness(repo=REPO, checkpoint=None, mesh=None, audit=None,
                            map_path=bundled_image, output=destination, count=count,
                            iterations=iterations, cpu_seconds=1.e20, command_seed=seed,
                            workers=workers, clearance_cuts=sorted(cuts),
                            stagnation_iterations=stagnation_iterations, cancel_requested=cancel)
                        seed = destination/'witness.npz'
                        residual = report['max_constraint_violation']
                        if report['status'] == 'source_witness_passed':
                            checked_dir = unused_directory(folder/'certified')
                            checked = certify_witness(REPO, seed, bundled_image, checked_dir)
                            if checked['status'] == 'source_witness_passed':
                                current['witness'] = str(checked_dir/'witness.npz')
                                promote(Path(current['witness']), checked['trial_lap_s'])
                                break
                        discovered = {(int(c['interval']), int(c['tick'])) for c in
                                      report['rollout']['violating_source_ticks']}
                        has_cuts = bool(discovered-cuts) and residual <= 1.e-3
                        cuts.update(discovered if has_cuts else ())
                        if not has_cuts and residual >= best_residual*.98:
                            break
                        best_residual = min(best_residual, residual)
                    current['attempts'].append(key)
                    current.pop('active', None)
                    save()
                    if current['witness'] is not None:
                        break
            if current['witness'] is None:
                continue
            if bootstrap_only:
                event('bootstrap_verified', status='prepared')
                return state
            event('adaptive_minimum_time_search', active_start=name)
            campaign = base/'campaign'
            result = run_campaign(repo=REPO, seed=Path(current['witness']),
                map_path=bundled_image, output=campaign, target_seconds=None,
                phases=phases, workers=workers, iterations=iterations, refinements=refinements,
                stagnation_iterations=stagnation_iterations, coarse_fraction=.1,
                hybrid_time=hybrid_time,
                resume=(campaign/'campaign.json').exists(), cancel_requested=cancel,
                on_improvement=promote)
            current['campaign_status'] = result['status']
            save()
        terminal = ('no_validated_raceline' if state['best_lap_s'] is None else
                    'search_domain_reached' if any(v.get('campaign_status') ==
                        'search_domain_reached' for v in state['starts'].values()) else 'plateau')
        event('finished', status=terminal)
        return state
    except KeyboardInterrupt:
        event('interrupted', status='interrupted')
        raise
    except BaseException as error:
        event('failed', status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        lock.close()


def show_status(output):
    path = Path(output)/'optimizer.json'
    if not path.exists():
        print(f'Not initialized yet: {output}')
        return
    state = json.loads(path.read_text())
    best = state.get('best_lap_s')
    print(f"Stage: {state['stage']} | Status: {state['status']}")
    print(f"Budgets: {state['contract']['iterations']} iterations/solve | "
          f"{state['contract'].get('stagnation_iterations', 50)} iterations of stagnation patience")
    print('Best source-validated lap: '+(f'{best:.3f} s' if best is not None else 'none yet'))
    print(f"Starting basin: {state.get('active_start', '—')} | Last state save: "
          f"{time.time()-state['updated_epoch_s']:.0f} s ago")
    process = Path('/proc')/str(state.get('pid', 0))
    if process.exists():
        fields = process.joinpath('status').read_text().splitlines()
        print('Process: '+', '.join(v.strip() for v in fields if
              v.startswith(('State:', 'VmRSS:', 'VmSwap:'))))
    else:
        print('Process: not running (saved status may predate a power loss)')
    for path in sorted((Path(output)/'starts').glob('*/campaign/campaign.json')):
        from command_campaign import show_status as campaign_status
        print(f'\n{path.parent.parent.name}')
        campaign_status(path.parent)
    active = state.get('active_start')
    context = state.get('starts', {}).get(active, {}).get('active')
    if context:
        checkpoint = Path(context['output'])/'best_iterate.npz'
        print(f"Bootstrap trial: {state.get('trial_lap_s')} s | Output: {context['output']}")
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as data:
                print(f"Saved iteration: {int(data['iteration'])} | residual: "
                      f"{float(data['max_constraint_violation']):.6g}")
    print(f"Best output manifest: {Path(output)/'best.json'} (exists only after validation)")
    log = Path(output)/'optimizer.log'
    if log.exists():
        print(f'Log last changed: {time.time()-log.stat().st_mtime:.0f} s ago')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map', type=Path, help='PGM/image or ROS map YAML')
    parser.add_argument('yaml', nargs='?', type=Path, help='explicit metric YAML for the supplied image')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--driving-reference', type=Path, help=
        'previously completed official Unity raceline; defines required travel direction')
    parser.add_argument('--command-seed', type=Path,
                        help='Continue from a saved witness; independently recertified on the supplied map')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--iterations', type=int, default=1500, help='per solve; improving primals are restarted')
    parser.add_argument('--stagnation-iterations', type=int, default=250,
                        help='patience without significant feasibility improvement (minimum 5)')
    parser.add_argument('--phases', type=int, default=4)
    parser.add_argument('--refinements', type=int, default=6)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--bootstrap-only', action='store_true')
    parser.add_argument('--hybrid-time-search', action='store_true',
                        help='Add local minimum-time proposals followed by exact command repair')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--describe', action='store_true', help='print resolved output/service metadata; no solve')
    args = parser.parse_args()
    if (not 1 <= args.workers <= (os.cpu_count() or 1) or args.iterations < 1
            or args.phases < 1 or args.refinements < 0 or args.stagnation_iterations < 5):
        parser.error('invalid solver search settings')
    image, metadata, config = resolve_map(args.map, args.yaml)
    driving_reference = args.driving_reference
    if driving_reference is None and image.stem == 'autodrive_iros_2026_compete':
        # Existing two-lap Unity/MPC result is authoritative for IROS direction.
        driving_reference = REPO/'live_runs/iros_mpc_initial_20261029_02/raceline.csv'
    if driving_reference is not None:
        verified = verified_unity_direction(driving_reference)
        # Never resume the geometrically opposite archived campaign.
        output = (args.output or (default_output(image, config).with_name(
            default_output(image, config).name+'_unity_'+verified['sha256'][:10]))).resolve()
    else:
        output = (args.output or default_output(image, config)).resolve()
    if args.command_seed is not None and args.output is None:
        output = output.with_name(output.name+'_seed_'+file_hash(args.command_seed)[:10])
    if args.hybrid_time_search and args.output is None:
        output = output.with_name(output.name+'_hybrid')
    if args.describe:
        print(json.dumps(dict(output=str(output), image=str(image), metadata=str(metadata),
            unit='sdu-track-optimizer-'+hashlib.sha256(str(output).encode()).hexdigest()[:12])))
        return 0
    if args.status or args.watch:
        try:
            while True:
                if args.watch: print('\033[2J\033[H', end='')
                show_status(output)
                if not args.watch: return 0
                time.sleep(2.)
        except KeyboardInterrupt:
            return 0
    stop = [False]
    def interrupted(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        state = run_track(image=image, metadata=metadata, config=config, output=output,
            workers=args.workers, iterations=args.iterations, phases=args.phases,
            refinements=args.refinements, stagnation_iterations=args.stagnation_iterations,
            prepare_only=args.prepare_only, bootstrap_only=args.bootstrap_only,
            driving_reference=driving_reference,
            hybrid_time=args.hybrid_time_search,
            command_seed=args.command_seed,
            cancel=lambda: stop[0])
        return 0 if state['status'] in ('plateau', 'prepared', 'search_domain_reached') else 3
    except KeyboardInterrupt:
        print('Stopped; all completed certifications and durable iterates retained.', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
