#!/usr/bin/env python3
"""Restore the IPOPT minimum-time OCP and export an unvalidated target.

This is NOT the stochastic `optimize.py` from the previous archive.

- Source model: historical CasADi/IPOPT 16-state Unity-wheel minimum-time OCP,
  patched with checked-in Unity/PhysX-derived equations and parameters.
- The target contains geometry, speed, and curvature, not witness commands.
- Native witness replay is diagnostic only; it is neither MPC tracking nor
  Unity validation. The discrete source-native plant is not Unity's native
  WheelCollider/contact solver.
"""
from __future__ import annotations
import argparse
import csv
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import traceback
import numpy as np
import yaml

from recover import load_modules, REF
from source_vehicle import current_model,bind_current_wheel_moment_arms
from map_track import MapGrid, map_to_track
from native_validate import replay,export_mpc
from native_seed import generate as generate_native_seed
from phase1_campaign import (continue_phase1,
                             load_phase1_resume_report,
                             validate_restart_checkpoint)
from seed_projection import (load_native_seed, project_algebraics,
                             write_audit, validate_seed_provenance)


OPTIMIZER_TEMPORAL_SEMANTICS = {
    'formulation': 'spatial OCP; minimize integrated lap time from ds/sdot',
    'control_mesh': (
        'controls are optimized over spatial nodes; no fixed-rate runtime '
        'command schedule is imposed'),
    '25ms_runtime_command_hold': False,
    '1ms_unity_fixed_step_role': (
        'source physics parameter for the wheel-spin recurrence only; its '
        'constant-torque scalar recurrence is analytically composed over '
        'each variable-duration spatial segment, not used as the optimizer '
        'time grid; within-segment changing tire forces remain collocation '
        'approximations'),
}


def annotate_spatial_temporal_semantics(
        report: dict, *, unity_source_fixed_step_s: float) -> None:
    """Keep reports explicit that OCP space/time is not an MPC tick schedule."""
    report['temporal_semantics'] = dict(OPTIMIZER_TEMPORAL_SEMANTICS)
    wheel = report.get('wheel_spin_discretization')
    if isinstance(wheel, dict):
        wheel['reason'] = (
            'The source implicit-Euler scalar wheel recurrence is composed '
            'for collocated segment torques over the variable spatial-segment '
            'travel time. The 5/1 substep branch is selected once from the '
            'segment-start speed; within-segment speed-triggered branch '
            'changes and changing tire reaction are not resolved. Fractional '
            'segment durations use the smooth analytic extension. This is '
            'not a Unity-per-tick replay.')
        wheel['optimizer_time_grid_s'] = None
        wheel['unity_fixed_step_s'] = float(unity_source_fixed_step_s)
        wheel['branch_selection_per_spatial_segment'] = True
        wheel['within_segment_unity_branch_changes_reproduced'] = False

if hasattr(signal, 'SIGUSR1'):
    # Always install this before any long solve. An optional handler made a
    # progress inspection terminate a valid background solve with SIGUSR1.
    # Stack inspection must never depend on an undocumented launch flag.
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


def validate_geometry(track,grid,margin):
    xy=np.column_stack((track.x,track.y))
    # Geometry seed must live in known free space; unlike native rollout we
    # only check centerline here (IPOPT constrains body footprint separately).
    probe=xy[:,None,:]+np.linspace(0,1,6,endpoint=False)[None,:,None]*(
        np.roll(xy,-1,axis=0)-xy)[:,None,:]
    min_center=float(np.min(grid.distance(probe.reshape(-1,2))))
    if min_center < margin+.1365:
        raise ValueError('Original reference/resampled centerline unsafe for IPOPT: '
                         f'min wall distance={min_center:.3f}m; need >= {margin+.1365:.3f}m')
    return min_center


def print_feasibility_diagnostics(report:dict, trust_limit:float)->None:
    """Print *solver iterate* constraints, not just the default Ipopt summary."""
    diag=report.get('nlp_diagnostics',{})
    print('IPOPT Phase-I diagnostics:',flush=True)
    status=report.get('solver',{}).get('return_status','unknown')
    print('  IPOPT status:',status,flush=True)
    if status == 'initial_guess_diagnostic_only':
        initial=diag.get('initial_guess',{})
        raw=diag.get('initial_raw_dynamic_defects',{})
        print('  Initial NLP violation:',
              initial.get('max_constraint_violation'),
              '(elastic seed check; not a feasible lap)',flush=True)
        print('  Non-finite initial constraints:',
              initial.get('nonfinite_constraint_count'),flush=True)
        print('  Maximum unelasticized dynamic defect:',
              diag.get('initial_raw_dynamic_max_defect'),flush=True)
        print('  Raw-defect values mix physical units and scaled collocation residuals; '
              'they are not one physical error metric.', flush=True)
        physical=sorted(
            ((name,values) for name,values in raw.items()
             if not name.startswith('periodic_collocation_state:')),
            key=lambda pair: pair[1].get('max_abs',0.0), reverse=True)
        for name,values in physical[:5]:
            print(f"  {name}: max={values.get('max_abs')}, "
                  f"RMSE={values.get('rmse')}, "
                  f"worst={values.get('worst_constraint_label', 'unknown')}",
                  flush=True)
        source = report.get('initial_guess_source', {})
        if source:
            print('  Actual seed source:', source.get('kind', 'unknown'),
                  source.get('path', ''), flush=True)
        print('  Initial-only diagnostic: IPOPT was not called; no candidate '
              'was certified.',flush=True)
        return
    print('  Max NLP violation:',diag.get('max_constraint_violation'),flush=True)
    worst=diag.get('worst_constraints',[])
    if not worst:
        print('  No labeled constraint diagnostics available; inspect JSON report.',flush=True)
        return
    for item in worst[:8]:
        print(f"  {item.get('label', 'unknown')}: violation={item.get('violation')} ",flush=True)
    if worst[0].get('label')=='captured_motion_trust_region':
        print('  Unexpected captured-motion trust-bound violation during Phase I. '
              'The v2.2.3 numerical overlay disables this artificial bound; '
              'verify your working solver overlay and provenance.',flush=True)


def validate_initial_diagnostic_seed_mode(*, diagnose_initial_only: bool,
                                          seed_file: Path | None,
                                          seed_backend: str) -> None:
    """Prevent a native-seed request from silently becoming a synthetic audit."""
    if (diagnose_initial_only and seed_file is None
            and seed_backend != 'synthetic'):
        raise ValueError(
            '--diagnose-initial-only does not generate a native/source warm-start. '
            'Provide --seed-file for an existing captured seed, or explicitly '
            'select --seed-backend synthetic to audit the synthetic initializer.')


def validate_native_diagnostic_capture_mode(*, capture: bool,
                                            seed_file: Path | None,
                                            seed_backend: str,
                                            skip_native: bool,
                                            setup_check: bool,
                                            diagnose_initial_only: bool) -> None:
    """Keep invasive native capture explicit and tied to a fresh native lap."""
    if not capture:
        return
    if seed_backend != 'native':
        raise ValueError('--capture-native-diagnostics requires --seed-backend native')
    if seed_file is not None or skip_native or setup_check or diagnose_initial_only:
        raise ValueError(
            '--capture-native-diagnostics requires a fresh native warm-start trial; '
            'remove --seed-file, --skip-native, --setup-check, and '
            '--diagnose-initial-only')


def competition_wall_margin(repo_root: Path) -> float:
    """Read the active MPC corridor margin so target and runtime agree."""
    config_path = Path(repo_root) / 'f1tenth_mpc/config/mpc_competition.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    parameters = config.get('/**', {}).get('ros__parameters', {})
    value = parameters.get('corridor_margin_m')
    if value is None:
        raise ValueError(f'Missing corridor_margin_m in {config_path}')
    margin = float(value)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f'Invalid corridor_margin_m in {config_path}: {value!r}')
    return margin


def parse_trust_region_factors(value: str) -> tuple[float, ...]:
    """Parse one or more increasing IPOPT trust-region continuation radii."""
    try:
        factors = tuple(float(item.strip()) for item in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'trust-region factors must be comma-separated numbers') from error
    if (not factors or not np.isfinite(factors).all()
            or any(factor <= 1.0 for factor in factors)
            or any(right <= left for left, right in zip(factors, factors[1:]))):
        raise argparse.ArgumentTypeError(
            'trust-region factors must be finite, >1, and strictly increasing')
    return factors


def validate_speed_profile_seed_args(args: argparse.Namespace) -> None:
    """Validate speed-profile initialization against the selected seed source.

    A speed profile may initialize a synthetic solve, or replace the speed in
    a fully source-feasible optimizer checkpoint.  Requiring the latter to be
    labeled ``synthetic`` is misleading and made a valid continuation look as
    if it were discarding its source-derived checkpoint.
    """
    from_centerline = bool(args.seed_speed_from_centerline)
    from_csv = args.seed_speed_csv is not None
    fraction = float(getattr(args, 'checkpoint_speed_seed_fraction', 1.0))
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError('--checkpoint-speed-seed-fraction must be in (0, 1]')
    if fraction != 1.0 and (
            args.optimizer_seed_file is None or not (from_centerline or from_csv)):
        raise ValueError(
            '--checkpoint-speed-seed-fraction requires a speed profile and '
            '--optimizer-seed-file')
    if from_centerline and from_csv:
        raise ValueError(
            'choose either --seed-speed-from-centerline or --seed-speed-csv')
    if not (from_centerline or from_csv):
        return
    if args.seed_file is not None:
        raise ValueError(
            'speed-profile initialization cannot be combined with --seed-file; '
            'use --optimizer-seed-file for a source-feasible checkpoint')
    if (args.phase1_warmstart_report is not None or
            args.phase1_resume_report is not None):
        raise ValueError(
            'speed-profile initialization cannot be combined with Phase-I '
            'warm-start or resume reports')
    if (args.optimizer_seed_file is None and
            args.seed_backend != 'synthetic'):
        raise ValueError(
            'speed-profile initialization without --optimizer-seed-file '
            'requires --seed-backend synthetic')
    if getattr(args, 'seed_speed_cap_to_reference_lateral_demand', False):
        if args.seed_speed_csv is None or args.optimizer_seed_file is None:
            raise ValueError(
                '--seed-speed-cap-to-reference-lateral-demand requires both '
                '--seed-speed-csv and --optimizer-seed-file')


def path_curvature_from_lateral_offset(track, lateral_offset_m: np.ndarray
                                       ) -> np.ndarray:
    """Return signed Menger curvature of the checkpoint's sampled XY path."""
    offset = np.asarray(lateral_offset_m, dtype=np.float64)
    if offset.shape != (int(track.count),) or not np.isfinite(offset).all():
        raise ValueError('checkpoint lateral offset does not match the track mesh')
    psi = np.asarray(track.psi, dtype=np.float64)
    x = np.asarray(track.x, dtype=np.float64) - np.sin(psi) * offset
    y = np.asarray(track.y, dtype=np.float64) + np.cos(psi) * offset
    points = np.column_stack((x, y))
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    first = points - previous
    second = following - points
    chord = following - previous
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    denominator = (np.linalg.norm(first, axis=1) *
                   np.linalg.norm(second, axis=1) *
                   np.linalg.norm(chord, axis=1))
    if np.any(denominator <= 1.0e-12):
        raise ValueError('checkpoint path has a degenerate curvature sample')
    curvature = 2.0 * cross / denominator
    if not np.isfinite(curvature).all():
        raise ValueError('checkpoint path curvature is non-finite')
    return curvature


def cap_speed_seed_to_reference_lateral_demand(
        speed_mps: np.ndarray, path_curvature_per_m: np.ndarray,
        reference_lateral_demand_mps2: float) -> tuple[np.ndarray, dict]:
    """Cap only an initial speed guess to an existing line's demand envelope.

    The returned profile is not an OCP constraint or a claim about the
    vehicle's physical lateral limit; the full source equations still decide
    feasibility in Phase I and the minimum-time NLP.
    """
    speed = np.asarray(speed_mps, dtype=np.float64)
    curvature = np.asarray(path_curvature_per_m, dtype=np.float64)
    if (speed.ndim != 1 or speed.size < 2 or speed.shape != curvature.shape
            or not np.isfinite(speed).all()
            or not np.isfinite(curvature).all() or np.any(speed <= 0.0)):
        raise ValueError('speed/curvature seed arrays must be finite and aligned')
    if (not math.isfinite(reference_lateral_demand_mps2) or
            reference_lateral_demand_mps2 <= 0.0):
        raise ValueError('reference lateral demand must be finite and positive')
    curvature_floor = 1.0e-8
    speed_cap = np.sqrt(reference_lateral_demand_mps2 /
                        np.maximum(np.abs(curvature), curvature_floor))
    capped = np.minimum(speed, speed_cap)
    return capped, {
        'method': 'initial_speed_cap_from_reference_u2_abs_kappa',
        'reference_peak_lateral_demand_mps2': float(
            reference_lateral_demand_mps2),
        'candidate_path_peak_lateral_demand_before_mps2': float(
            np.max(speed * speed * np.abs(curvature))),
        'candidate_path_peak_lateral_demand_after_mps2': float(
            np.max(capped * capped * np.abs(curvature))),
        'clipped_sample_count': int(np.count_nonzero(capped < speed - 1.0e-9)),
        'profile_min_mean_max_mps': [float(np.min(capped)),
                                     float(np.mean(capped)),
                                     float(np.max(capped))],
        'is_constraint': False,
    }


def blend_speed_seed_with_checkpoint(
        target_speed_mps: np.ndarray, checkpoint_speed_mps: np.ndarray,
        fraction: float) -> tuple[np.ndarray, dict]:
    """Move a checkpoint speed seed partway toward a target profile.

    This is a continuation initializer only. It does not constrain the OCP or
    assert a physical lateral-acceleration limit.
    """
    target = np.asarray(target_speed_mps, dtype=np.float64)
    checkpoint = np.asarray(checkpoint_speed_mps, dtype=np.float64)
    if (target.ndim != 1 or target.size < 2 or target.shape != checkpoint.shape
            or not np.isfinite(target).all()
            or not np.isfinite(checkpoint).all()
            or np.any(target <= 0.0) or np.any(checkpoint <= 0.0)):
        raise ValueError(
            'target/checkpoint speed profiles must be positive, finite, and aligned')
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError('checkpoint speed-seed fraction must be in (0, 1]')
    blended = checkpoint + fraction * (target - checkpoint)
    return blended, {
        'method': 'linear_speed_profile_homotopy_initializer',
        'fraction_toward_target_profile': float(fraction),
        'checkpoint_profile_mean_mps': float(np.mean(checkpoint)),
        'target_profile_mean_mps': float(np.mean(target)),
        'blended_profile_mean_mps': float(np.mean(blended)),
        'is_constraint': False,
    }


def load_reference_speed_seed(path: Path, track, *,
                              return_lateral_demand_cap: bool = False):
    """Resample a CSV speed column onto the optimizer mesh for initialization only.

    This intentionally reads speed without importing reference geometry into
    the OCP constraints.  The path and speed remain free optimization
    variables; source-dynamics feasibility is still checked by IPOPT.
    """
    header: list[str] | None = None
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                candidate = [item.strip().lower()
                             for item in line.lstrip("# ").split(",")]
                if "s_m" in candidate and any(
                        name in candidate for name in
                        ("body_forward_speed_mps", "vx_mps", "speed_mps",
                         "velocity_mps")):
                    header = candidate
                continue
            try:
                row = [float(item.strip()) for item in line.split(",")]
            except ValueError:
                continue
            rows.append(row)

    if header is None or not rows:
        raise ValueError(
            f"Reference speed seed requires a commented CSV header and data: {path}")
    speed_name = next(name for name in
                      ("body_forward_speed_mps", "vx_mps", "speed_mps",
                       "velocity_mps")
                      if name in header)
    if len(rows[0]) != len(header) or any(len(row) != len(header) for row in rows):
        raise ValueError("Reference speed seed CSV rows do not match its header")
    table = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(table).all():
        raise ValueError("Reference speed seed contains non-finite values")
    source_s = table[:, header.index("s_m")].copy()
    speed = table[:, header.index(speed_name)].copy()
    source_curvature = None
    if return_lateral_demand_cap:
        curvature_name = next((name for name in
                               ('kappa_radpm', 'curvature_radpm',
                                'path_curvature_per_m')
                               if name in header), None)
        if curvature_name is None:
            raise ValueError(
                'A lateral-demand-capped speed seed requires a CSV '
                'curvature column')
        source_curvature = table[:, header.index(curvature_name)].copy()
    if (len(source_s) < 3 or np.any(np.diff(source_s) <= 0.0)
            or source_s[0] < -1.0e-8 or np.any(speed <= 0.0)):
        raise ValueError("Reference speed seed needs increasing s and positive speeds")

    # CSV station is open at the seam.  Scale its arc-length coordinate to
    # the smoothed optimizer track length, then interpolate periodically.
    source_s -= source_s[0]
    if "x_m" in header and "y_m" in header:
        xy = table[:, [header.index("x_m"), header.index("y_m")]]
        closing_distance = float(np.linalg.norm(xy[0] - xy[-1]))
        if closing_distance < 1.0e-6:
            source_s = source_s[:-1]
            speed = speed[:-1]
            if source_curvature is not None:
                source_curvature = source_curvature[:-1]
            xy = xy[:-1]
            closing_distance = float(np.linalg.norm(xy[0] - xy[-1]))
    else:
        closing_distance = float(np.median(np.diff(source_s)))
    source_length = float(source_s[-1] + closing_distance)
    if not math.isfinite(source_length) or source_length <= 0.0:
        raise ValueError("Reference speed seed has invalid lap length")
    source_s = source_s * (float(track.length) / source_length)
    periodic_s = np.r_[source_s, float(track.length)]
    periodic_speed = np.r_[speed, speed[0]]
    result = np.interp(np.asarray(track.s, dtype=np.float64),
                       periodic_s, periodic_speed)
    if result.shape != (int(track.count),) or not np.isfinite(result).all():
        raise ValueError("Resampled reference speed seed is invalid")
    if source_curvature is None:
        return result
    lateral_demand_cap = float(np.max(
        np.square(speed) * np.abs(source_curvature)))
    if not math.isfinite(lateral_demand_cap) or lateral_demand_cap <= 0.0:
        raise ValueError('Reference speed/curvature profile has no valid lateral demand')
    return result, lateral_demand_cap


_OPTIMIZER_SEED_FIELDS = (
    ('ey_m', 'ey_m'), ('epsi_rad', 'epsi_rad'), ('u_mps', 'speed_mps'),
    ('v_mps', 'lateral_speed_mps'), ('yaw_rate_radps', 'yaw_rate_radps'),
    ('wheel_omega_radps', 'wheel_omega_radps'),
    ('steering_rad', 'steering_rad'),
    ('steering_rate_radps', 'steering_rate_radps'),
    ('throttle_norm', 'throttle_norm'), ('ax_mps2', 'ax_mps2'),
    ('ay_mps2', 'ay_mps2'), ('normal_load_n', 'normal_load_n'),
    ('body_height_m', 'body_height_m'),
    ('body_height_rate_mps', 'body_height_rate_mps'),
    ('roll_rad', 'roll_rad'), ('roll_rate_radps', 'roll_rate_radps'),
    ('pitch_rad', 'pitch_rad'), ('pitch_rate_radps', 'pitch_rate_radps'),
)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                          default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def preserve_optimizer_seed_inputs(modification: dict, *, speed_replaced: bool,
                                   reproject_requested: bool) -> bool:
    """Keep an unmodified full raw-stage primal, including its saved torque.

    The downstream 'model consistent' initialization flag disables the
    steady-force throttle guess. It does NOT certify this seed: the complete
    current source-constraint audit still decides whether Phase I is needed.
    Partial blends and coordinate conversions retain normal projection.
    """
    if speed_replaced or reproject_requested:
        return False
    if not modification:
        return True
    return (
        modification.get('full_differential_state_transferred') is True
        and modification.get('fraction') == 1.0
        and modification.get('com_path_converted_to_rear_axle') is False
        and modification.get('clipped_speed_samples') == 0
    )


def write_optimizer_seed_checkpoint(candidate: dict, path: Path, *, track,
                                    map_path: Path, centerline_path: Path,
                                    model) -> None:
    """Save a complete, provenance-bound source-model iterate for continuation."""
    count = int(track.count)
    station = np.asarray(track.s, dtype=np.float64)
    payload = {
        'seed_sample_s': station,
        'metadata_json': np.asarray(json.dumps({
            'schema': 1,
            'map_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest(),
            'centerline_sha256': hashlib.sha256(
                centerline_path.read_bytes()).hexdigest(),
            'track_count': count,
            'track_length_m': float(track.length),
            'model_sha256': _sha256_json(model.metadata()),
        }, sort_keys=True)),
    }
    for seed_name, candidate_name in _OPTIMIZER_SEED_FIELDS:
        values = np.asarray(candidate[candidate_name], dtype=np.float64)
        expected_shape = (4, count) if seed_name == 'wheel_omega_radps' \
            or seed_name == 'normal_load_n' else (count,)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(
                f'optimizer checkpoint field {candidate_name} has invalid '
                f'shape/data: {values.shape}, expected {expected_shape}')
        payload[seed_name] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_optimizer_seed_checkpoint(path: Path, *, track, map_path: Path,
                                   centerline_path: Path, model,
                                   allow_mesh_resample: bool = False
                                   ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load a provenance-matched optimizer checkpoint.

    Mesh changes are rejected unless explicitly enabled. When enabled, the
    source station grid and arrays are returned unchanged; solve_bottom_up
    performs periodic interpolation onto its new mesh before Phase I proves
    the resampled state feasible.
    """
    with np.load(path, allow_pickle=False) as archive:
        if 'metadata_json' not in archive or 'seed_sample_s' not in archive:
            raise ValueError(f'incomplete optimizer seed checkpoint: {path}')
        metadata = json.loads(str(archive['metadata_json'].item()))
        station = np.asarray(archive['seed_sample_s'], dtype=np.float64)
        expected_metadata = {
            'schema': 1,
            'map_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest(),
            'centerline_sha256': hashlib.sha256(
                centerline_path.read_bytes()).hexdigest(),
            'track_length_m': float(track.length),
            'model_sha256': _sha256_json(model.metadata()),
        }
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if key == 'track_length_m':
                matches = (isinstance(actual, (int, float))
                           and math.isclose(float(actual), expected,
                                            rel_tol=0.0, abs_tol=1.0e-9))
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f'optimizer seed checkpoint {key} mismatch: '
                    f'{actual!r} != {expected!r}')
        source_count = metadata.get('track_count')
        if (not isinstance(source_count, int) or source_count != len(station)
                or station.ndim != 1 or len(station) < 2
                or not np.isfinite(station).all()
                or station[0] < -1.0e-8
                or station[-1] >= track.length
                or np.any(np.diff(station) <= 0.0)):
            raise ValueError('optimizer seed checkpoint has an invalid source mesh')
        same_mesh = (station.shape == (track.count,)
                     and np.allclose(station, track.s, rtol=0.0, atol=1.0e-9))
        if not same_mesh and not allow_mesh_resample:
            raise ValueError('optimizer seed checkpoint mesh does not match OCP')
        seed = {}
        for seed_name, _ in _OPTIMIZER_SEED_FIELDS:
            if seed_name not in archive:
                raise ValueError(
                    f'optimizer seed checkpoint lacks {seed_name}')
            values = np.asarray(archive[seed_name], dtype=np.float64)
            expected_shape = (4, source_count) if seed_name in {
                'wheel_omega_radps', 'normal_load_n'} else (track.count,)
            if not same_mesh and seed_name not in {
                    'wheel_omega_radps', 'normal_load_n'}:
                expected_shape = (source_count,)
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(
                    f'optimizer seed checkpoint {seed_name} is malformed')
            seed[seed_name] = values.copy()
        if 'seed_modification_json' in archive:
            modification = json.loads(
                str(archive['seed_modification_json'].item()))
            if (not isinstance(modification, dict)
                    or not isinstance(modification.get('modified_fields'), list)
                    or not all(isinstance(name, str)
                               for name in modification['modified_fields'])):
                raise ValueError(
                    'optimizer seed modification metadata is malformed')
            seed['_seed_modification'] = modification
    return seed, station.copy()


def replace_optimizer_seed_speed_profile(
        seed: dict[str, np.ndarray], speed_mps: np.ndarray, *,
        minimum_mps: float, maximum_mps: float,
        mode: str = 'speed-only',
        steering_rate_limit_radps: float | None = None
        ) -> dict[str, np.ndarray]:
    """Replace only forward speed in a complete checkpoint seed.

    This is an initialization experiment, not a constraint. ``time-scaled``
    also rescales time derivatives and accelerations as an initial guess for
    the same spatial trajectory. Neither mode changes the source equations;
    Phase I must re-establish them before time optimization.
    """
    if not isinstance(seed, dict) or 'u_mps' not in seed:
        raise ValueError('checkpoint seed must contain u_mps')
    speed = np.asarray(speed_mps, dtype=np.float64)
    original_speed = np.asarray(seed['u_mps'], dtype=np.float64)
    if (speed.ndim != 1 or speed.shape != original_speed.shape
            or not np.isfinite(speed).all()):
        raise ValueError('replacement speed profile must be finite and match u_mps')
    if (not math.isfinite(minimum_mps) or not math.isfinite(maximum_mps)
            or minimum_mps >= maximum_mps
            or np.any(speed < minimum_mps) or np.any(speed > maximum_mps)):
        raise ValueError('replacement speed profile lies outside OCP speed bounds')
    if mode not in {'speed-only', 'time-scaled'}:
        raise ValueError(f'unsupported checkpoint speed-seed mode: {mode!r}')
    result = {name: np.asarray(values, dtype=np.float64).copy()
              for name, values in seed.items()}
    if mode == 'time-scaled':
        ratio = speed / original_speed
        rate_fields = (
            'v_mps', 'yaw_rate_radps', 'steering_rate_radps',
            'body_height_rate_mps', 'roll_rate_radps', 'pitch_rate_radps')
        for name in rate_fields:
            if name not in result or result[name].shape != speed.shape:
                raise ValueError(
                    f'time-scaled checkpoint lacks compatible {name}')
            result[name] *= ratio
        if (steering_rate_limit_radps is not None
                and (not math.isfinite(steering_rate_limit_radps)
                     or steering_rate_limit_radps <= 0.0)):
            raise ValueError('steering-rate limit must be finite and positive')
        if steering_rate_limit_radps is not None:
            result['steering_rate_radps'] = np.clip(
                result['steering_rate_radps'],
                -steering_rate_limit_radps, steering_rate_limit_radps)
        wheel_speed = result.get('wheel_omega_radps')
        if (wheel_speed is None or wheel_speed.shape != (4, len(speed))):
            raise ValueError('time-scaled checkpoint lacks four wheel-speed rows')
        result['wheel_omega_radps'] *= ratio[None, :]
        for name in ('ax_mps2', 'ay_mps2'):
            if name not in result or result[name].shape != speed.shape:
                raise ValueError(
                    f'time-scaled checkpoint lacks compatible {name}')
            result[name] *= np.square(ratio)
    result['u_mps'] = speed.copy()
    return result


def run_phase1_or_diagnose(old, track, model, kwargs, *, output: Path,
                           phase: int, max_restarts: int,
                           minimum_improvement: float,
                           diagnose_initial_only: bool,
                           continuation=continue_phase1):
    """Keep report-only diagnostics out of the Phase-I continuation wrapper.

    ``solve_bottom_up`` has an early ``diagnose_initial_only`` return before
    installing or invoking IPOPT. Calling it directly also avoids printing a
    misleading "solve attempt" and writing restart/continuation artifacts.
    The NLP graph is still assembled to measure its constraints; no numerical
    optimization is performed in this mode.
    """
    if diagnose_initial_only:
        candidate, report = old.solve_bottom_up(track, model, **kwargs)
        return candidate, report, []
    return continuation(
        old, track, model, kwargs, output=output, phase=phase,
        max_restarts=max_restarts, minimum_improvement=minimum_improvement)


def write_midpoint_witness(candidate:dict, path:Path)->None:
    """Persist the source-model witness values checked between OCP knots."""
    state=np.asarray(candidate['midpoint_state'],dtype=np.float64)
    n=state.shape[1]
    columns=[
        's_ref_m','speed_u_mps','lateral_v_mps','yaw_rate_radps',
        'wheel_omega_fl_radps','wheel_omega_fr_radps',
        'wheel_omega_rl_radps','wheel_omega_rr_radps',
        'witness_steering_rad','witness_steering_rate_radps',
        'witness_throttle_norm','body_height_m','body_height_rate_mps',
        'roll_rad','roll_rate_radps','pitch_rad','pitch_rate_radps',
        'normal_load_fl_n','normal_load_fr_n','normal_load_rl_n',
        'normal_load_rr_n','slipx_fl','slipx_fr','slipx_rl','slipx_rr',
        'slipy_fl','slipy_fr','slipy_rl','slipy_rr',
        'force_x_body_n','force_y_body_n','force_z_body_n',
        'moment_x_body_nm','moment_y_body_nm','moment_z_body_nm',
        'du_dt_mps2','dv_dt_mps2','yaw_acceleration_radps2',
        'world_vertical_acceleration_mps2','roll_acceleration_radps2',
        'pitch_acceleration_radps2']
    arrays=(candidate['midpoint_wheel_omega_radps'],
            candidate['midpoint_normal_load_n'],
            candidate['midpoint_slip_x'],candidate['midpoint_slip_y'])
    if any(np.asarray(array).shape!=(4,n) for array in arrays):
        raise ValueError('midpoint feasibility witness arrays have inconsistent shapes')
    row_count=n
    if any(np.asarray(candidate[key]).shape!=(row_count,) for key in (
            'midpoint_s_ref_m','midpoint_steering_rad',
            'midpoint_steering_rate_radps','midpoint_throttle_norm')):
        raise ValueError('midpoint feasibility witness scalars have inconsistent shapes')
    if np.asarray(candidate['midpoint_force_summary']).shape!=(6,n):
        raise ValueError('midpoint force summary has inconsistent shape')
    if np.asarray(candidate['midpoint_acceleration_summary']).shape!=(6,n):
        raise ValueError('midpoint acceleration summary has inconsistent shape')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(columns)
        for k in range(n):
            writer.writerow([
                candidate['midpoint_s_ref_m'][k],
                *state[2:5,k],
                *candidate['midpoint_wheel_omega_radps'][:,k],
                candidate['midpoint_steering_rad'][k],
                candidate['midpoint_steering_rate_radps'][k],
                candidate['midpoint_throttle_norm'][k],
                *state[10:16,k],
                *candidate['midpoint_normal_load_n'][:,k],
                *candidate['midpoint_slip_x'][:,k],
                *candidate['midpoint_slip_y'][:,k],
                *candidate['midpoint_force_summary'][:,k],
                *candidate['midpoint_acceleration_summary'][:,k],
            ])
    return path


def run():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=(
        'CasADi/IPOPT spatial minimum-time target optimization with a '
        'source-derived wheel feasibility witness'))
    ap.add_argument('--repo-root',type=Path,default=Path('..'))
    ap.add_argument('--map',type=Path,default=None,help='PGM with matching ROS .yaml; default Porto practice map')
    ap.add_argument('--centerline',type=Path,default=None,help='CSV reference only; optimizer changes geometry')
    ap.add_argument('--output',type=Path,default=Path('optimized_practice_ipopt'))
    ap.add_argument('--audit-time-checkpoint', type=Path,
                    help='Rebuild a matching feasible seed NLP and audit saved primal/duals; no time solve')
    ap.add_argument('--physx-binary',type=Path,default=None)
    ap.add_argument('--mesh-spacing',type=float,default=.06,help='spatial OCP knot spacing (m); source feasibility is checked at knots and interval midpoints, so max sample gap is half this value')
    ap.add_argument('--max-speed',type=float,default=16.,help='current MPC command envelope, not real-car limit')
    ap.add_argument('--iterations',type=int,default=1200,help='IPOPT minimum-time stage iterations')
    ap.add_argument('--feasibility-iterations',type=int,default=500)
    ap.add_argument('--phase1-max-cpu-seconds', type=float, default=0.0,
                    help=('optional IPOPT Phase-I CPU-time cap; 0 leaves the '
                          'iteration budget as the only limit'))
    ap.add_argument('--time-max-cpu-seconds', type=float, default=0.0,
                    help=('optional IPOPT CPU-time cap for each minimum-time '
                          'continuation stage; 0 leaves its iteration budget'))
    ap.add_argument('--time-hessian-approximation',
                    choices=('limited-memory', 'exact'),
                    default='limited-memory',
                    help='Hessian method for the minimum-time NLP stages')
    ap.add_argument('--time-trust-region-factors',
                    type=parse_trust_region_factors, default=None,
                    help=('comma-separated increasing trust radii for time '
                          'continuation; defaults to the optimizer schedule'))
    ap.add_argument('--minimum-trust-region-reference', type=float, default=1.0,
                    help=('minimum normalized seed-distance reference used to '
                          'scale time-stage trust radii'))
    ap.add_argument('--phase1-restarts',type=int,default=0,help='controlled Phase-I restarts from best hard-feasible but nonzero-defect checkpoint')
    ap.add_argument('--phase1-min-defect-improvement',type=float,default=1.e-5,help='minimum max-defect progress required to restart Phase I')
    ap.add_argument('--phase1-hessian', choices=('limited-memory','exact'), default='limited-memory', help='Use IPOPT L-BFGS for stiff/non-smooth Phase-I feasibility; original exact Hessian remains available')
    ap.add_argument('--phase1-formulation', choices=('elastic','direct','objective','source-equalities','condensed','eliminated'), default='elastic', help='Phase-I model: source-equalities keeps Unity-derived dynamics exact and minimizes normalized displacement from the seed; objective minimizes residuals then restores exact equalities after a strict gate; direct keeps equivalent bounded residual rows')
    ap.add_argument('--feasibility-trust-region',type=float,default=200.,help='Elastic Phase-I normalized squared-distance cap from the measured seed; condensed Phase-I uses four times this value as its hard numeric trust radius')
    ap.add_argument('--diagnose-initial-only',action='store_true',help=(
        'audit an existing --seed-file or explicitly selected synthetic seed; '
        'does not acquire native/source warm-start data and never runs IPOPT'))
    ap.add_argument('--phase1-warmstart-report', type=Path, default=None,
                    help='Use a verified v2.4.2 Phase-I checkpoint to initialize eliminated Phase I')
    ap.add_argument('--phase1-resume-report', type=Path, default=None,
                    help=('resume Phase I from a saved hard-feasible but '
                          'dynamics-infeasible report; map, centerline, exact '
                          'spatial mesh, vehicle model, optimizer source, and '
                          'Phase-I formulation must all match'))
    ap.add_argument('--seed-file',type=Path,default=None, help='Reuse a native_warmstart.npz for the same OCP mesh and map; skips driving warm-start')
    ap.add_argument('--optimizer-seed-file', type=Path, default=None,
                    help=('Continue from a previously exported, fully '
                          'source-feasible IPOPT checkpoint; map, centerline, '
                          'mesh, and vehicle-model fingerprints must match'))
    ap.add_argument('--reproject-optimizer-seed', action='store_true',
                    help=('recompute algebraic and periodic suspension seed '
                          'values when the checkpoint records intentional '
                          'state/geometry modifications'))
    ap.add_argument('--seed-speed-from-centerline', action='store_true',
                    help=('use centerline CSV speed only as an initial '
                          'profile; with --optimizer-seed-file replace '
                          'checkpoint speed and preserve its other states'))
    ap.add_argument('--seed-speed-csv', type=Path, default=None,
                    help=('use speed from this same-track trajectory CSV as '
                          'an initial profile without changing the selected '
                          'centerline geometry or its checkpoint fingerprint'))
    ap.add_argument('--seed-speed-cap-to-reference-lateral-demand',
                    action='store_true',
                    help=('initializer only: cap transferred speeds using the '
                          'reference CSV peak u^2*|kappa| evaluated on the '
                          'optimizer checkpoint path; adds no OCP constraint'))
    ap.add_argument('--checkpoint-speed-seed-mode',
                    choices=('speed-only', 'time-scaled'),
                    default='time-scaled',
                    help=('when replacing checkpoint speed, rescale the '
                          'associated rate/acceleration initial values; '
                          'Phase I still enforces the full source equations'))
    ap.add_argument('--checkpoint-speed-seed-fraction', type=float, default=1.0,
                    help=('fraction in (0,1] of the requested speed-profile '
                          'change to apply to a source checkpoint; <1 enables '
                          'gradual Phase-I continuation only'))
    ap.add_argument('--resample-optimizer-seed', action='store_true',
                    help=('allow a provenance-matched optimizer checkpoint '
                          'from a different spatial mesh; the new mesh is '
                          'periodically interpolated and must pass Phase I'))
    ap.add_argument('--no-seed-force-projection',action='store_true',help='Diagnostic: keep measured finite-difference acceleration and PhysX loads')
    ap.add_argument('--seed-backend',choices=('native','source','synthetic'),default='native',
                    help='native PhysX flying-lap seed (default), C++ source-derived seed, or legacy synthetic initialization')
    ap.add_argument('--wall-margin',type=float,default=None,
                    help='extra body-to-wall clearance in metres; defaults to the active MPC profile')
    ap.add_argument('--native-max-seconds',type=float,default=18.)
    ap.add_argument('--capture-native-diagnostics',action='store_true',
                    help=('during a fresh native warm-start trial, save per-wheel '
                          'contact and full returned state/math snapshots as JSONL; '
                          'then exit before IPOPT; this launches the native simulator'))
    ap.add_argument('--warmup-laps',type=int,default=1)
    ap.add_argument('--print-level',type=int,default=4)
    ap.add_argument('--allow-history-fetch',action='store_true')
    ap.add_argument('--restore-only',action='store_true',help='restore exact historical OCP source to package cache and exit')
    ap.add_argument('--setup-check',action='store_true',help='restore and check map/model imports but skip IPOPT')
    ap.add_argument('--skip-native',action='store_true',
                    help='skip source-native replay; export remains unvalidated')
    args=ap.parse_args()
    try:
        validate_initial_diagnostic_seed_mode(
            diagnose_initial_only=args.diagnose_initial_only,
            seed_file=args.seed_file or args.optimizer_seed_file,
            seed_backend=args.seed_backend)
        validate_native_diagnostic_capture_mode(
            capture=args.capture_native_diagnostics,
            seed_file=args.seed_file or args.optimizer_seed_file,
            seed_backend=args.seed_backend,
            skip_native=args.skip_native,
            setup_check=args.setup_check,
            diagnose_initial_only=args.diagnose_initial_only)
    except ValueError as error:
        ap.error(str(error))
    args.repo_root=args.repo_root.resolve()
    wall_margin_overridden = args.wall_margin is not None
    runtime_wall_margin = competition_wall_margin(args.repo_root)
    if args.wall_margin is None:
        args.wall_margin = runtime_wall_margin
    if not math.isclose(args.wall_margin, runtime_wall_margin,
                        rel_tol=0.0, abs_tol=1.0e-12):
        print('WARNING: optimizer wall margin differs from the active MPC '
              f'profile ({args.wall_margin:.3f} m vs '
              f'{runtime_wall_margin:.3f} m).', file=sys.stderr)
    if args.map is None:
        args.map=args.repo_root/'f1tenth_planning/maps/autodrive_practice_20260924_b.pgm'
    args.map=args.map.resolve()
    if args.centerline is None:
        if args.map.name=='autodrive_practice_20260924_b.pgm':
            args.centerline=args.repo_root/'f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv'
        elif args.map.name=='autodrive_track_ftg_commit_20260909_025m.pgm':
            args.centerline=args.repo_root/'f1tenth_planning/trajectories/autodrive_mintime_sim_5p0_dense/autodrive_mintime_raceline.csv'
        else:
            ap.error('For a custom PGM, supply a centerline CSV (--centerline) with correct map correspondences')
    args.centerline=args.centerline.resolve()
    try:
        validate_speed_profile_seed_args(args)
    except ValueError as error:
        ap.error(str(error))
    if (args.phase1_resume_report is not None
            and (args.phase1_warmstart_report is not None
                 or args.seed_file is not None
                 or args.optimizer_seed_file is not None
                 or args.seed_speed_from_centerline
                 or args.seed_speed_csv is not None
                 or args.capture_native_diagnostics
                 or args.setup_check or args.restore_only
                 or args.diagnose_initial_only)):
        ap.error('--phase1-resume-report is a standalone optimization mode; '
                 'remove other seed and diagnostic/setup options')
    reference_speed_seed_path = (
        args.seed_speed_csv.resolve() if args.seed_speed_csv is not None else
        args.centerline if args.seed_speed_from_centerline else None)
    args.output=args.output.resolve()
    if (args.mesh_spacing<.06 or args.mesh_spacing>.24
            or not math.isfinite(args.wall_margin) or args.wall_margin<0
            or args.max_speed<.4 or args.iterations<1
            or args.feasibility_iterations<1
            or not math.isfinite(args.phase1_max_cpu_seconds)
            or args.phase1_max_cpu_seconds < 0.0
            or not math.isfinite(args.time_max_cpu_seconds)
            or args.time_max_cpu_seconds < 0.0
            or not math.isfinite(args.feasibility_trust_region)
            or args.feasibility_trust_region<=0
            or not math.isfinite(args.minimum_trust_region_reference)
            or args.minimum_trust_region_reference<=0
            or args.native_max_seconds<.5
            or args.warmup_laps<0):
        ap.error('invalid optimization bounds (mesh-spacing .06-.24, positive speed/time/iterations)')
    print('== Recover verified original IPOPT spatial min-time source ==',flush=True)
    old, wheel, track_mod, vendor, solver_overlay = load_modules(
        args.repo_root, root, args.allow_history_fetch, args.phase1_hessian, args.phase1_formulation)
    print(f'Original commit: {REF}',flush=True)
    print(f'Working CasADi solver: {Path(old.__file__)}',flush=True)
    print(f'Phase-I Hessian: {args.phase1_hessian}; historical source preserved: '
          f'{solver_overlay.get("v249_baseline_preserved", False)}',flush=True)
    min_drive_throttle = float(solver_overlay['mpc_throttle_min_drive'])
    print(f'Optimizer input envelope: throttle [{min_drive_throttle:.3f}, 1.000]; '
          'positive drive only, matching production MPC.', flush=True)
    if args.restore_only:
        print('Restored successfully; original files are unchanged.');return 0
    print('== Resolve current PhysX/Unity source parameters ==',flush=True)
    model, model_meta=current_model(args.repo_root,wheel)
    model_meta['current_wheel_moment_arms']=bind_current_wheel_moment_arms(args.repo_root,old,wheel)
    grid=MapGrid.load(args.map)
    # Seed verification also detects mismatched map/raceline pairs.
    init_ref=map_to_track(grid,args.centerline,spacing=.12)
    track=track_mod.load_centerline_csv(args.centerline,args.mesh_spacing)
    reference_speed_seed = None
    reference_lateral_demand_cap_mps2 = None
    optimizer_seed_speed_replaced = False
    hybrid_seed_checkpoint_path = None
    if reference_speed_seed_path is not None:
        try:
            loaded_seed = load_reference_speed_seed(
                reference_speed_seed_path, track,
                return_lateral_demand_cap=(
                    args.seed_speed_cap_to_reference_lateral_demand))
            if args.seed_speed_cap_to_reference_lateral_demand:
                (reference_speed_seed,
                 reference_lateral_demand_cap_mps2) = loaded_seed
            else:
                reference_speed_seed = loaded_seed
        except (OSError, ValueError) as error:
            ap.error(str(error))
        print('Reference speed used for initialization only: '
              f'min={np.min(reference_speed_seed):.3f}, '
              f'mean={np.mean(reference_speed_seed):.3f}, '
              f'max={np.max(reference_speed_seed):.3f} m/s; '
              f'source={reference_speed_seed_path}; path, speed, and controls '
              'remain optimization variables.',
              flush=True)
        if reference_lateral_demand_cap_mps2 is not None:
            print('Reference-line peak u^2*|kappa| initializer cap: '
                  f'{reference_lateral_demand_cap_mps2:.3f} m/s^2; '
                  'not an OCP or vehicle-limit constraint.', flush=True)
    min_center=validate_geometry(track,grid,args.wall_margin)
    track_stations=np.asarray(track.s,dtype=np.float64)
    segment_lengths=np.diff(np.r_[track_stations,
                                  track_stations[0]+track.length])
    max_source_feasibility_sample_gap_m=float(np.max(segment_lengths)*0.5)
    print(f'Map {args.map.name} | OCP length {track.length:.3f} m | nodes {track.count} '
          f'| reference min clearance {min_center:.3f} m',flush=True)
    print(f'Source feasibility samples at knots + interval midpoints; maximum '
          f'spatial gap {max_source_feasibility_sample_gap_m:.4f} m.',flush=True)
    print(f'Motor {model.motor_torque_nm} Nm; wheel radius {model.wheel_radius_m} m; '
          f'throttle [{min_drive_throttle:.3f}, 1.000]; '
          f'steering ±{model.steering_limit_rad:.4f} rad',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    # A normal CLI solve must retain useful feasible intermediate iterates,
    # not only a possibly infeasible max-iteration terminal point. Callers can
    # still choose another durable root explicitly (unattended launcher).
    if not args.diagnose_initial_only and args.audit_time_checkpoint is None:
        os.environ.setdefault('SDU_IPOPT_STAGE_CAPTURE_DIR',
                              str((args.output/'time_stage_iterates').resolve()))
        os.environ.setdefault('SDU_IPOPT_TIME_CHECKPOINT_STRIDE',
                              '5' if args.iterations <= 100 else '25')
    reference_track_mesh_path = args.output/'optimizer_reference_track_mesh.csv'
    track_mod.export_track_csv(track, reference_track_mesh_path)
    run_info={
        'optimizer':'CasADi/IPOPT restored from original wheel-level minimum-time collocation',
        'repo_root':str(args.repo_root),'map':str(args.map),
        'map_sha256':hashlib.sha256(args.map.read_bytes()).hexdigest(),
        'centerline':str(args.centerline),
        'centerline_sha256':hashlib.sha256(args.centerline.read_bytes()).hexdigest(),
        'optimizer_reference_track_mesh': {
            'path':str(reference_track_mesh_path),
            'sha256':hashlib.sha256(
                reference_track_mesh_path.read_bytes()).hexdigest(),
            'nodes':int(track.count),
            'length_m':float(track.length),
            'purpose':'exact resampled OCP spline mesh for geometry-seed previews',
        },
        'restore_commit':REF,'phase1_solver_overlay':solver_overlay,'model':model_meta,
        'wall_clearance_m': args.wall_margin,
        'active_mpc_wall_clearance_m': runtime_wall_margin,
        'wall_clearance_source': ('command_line_override' if wall_margin_overridden
                                  else 'f1tenth_mpc/config/mpc_competition.yaml'),
        'time_optimization': {
            'variable_scaling': os.environ.get('SDU_IPOPT_VARIABLE_SCALING', '0') == '1',
            'barrier_strategy': os.environ.get('SDU_IPOPT_BARRIER_STRATEGY', 'adaptive'),
            'trust_geometry': os.environ.get('SDU_IPOPT_TRUST_GEOMETRY', 'node-mean'),
            'hessian_approximation': args.time_hessian_approximation,
            'trust_region_factors': (None if args.time_trust_region_factors is None
                                     else list(args.time_trust_region_factors)),
            'minimum_trust_region_reference':
                args.minimum_trust_region_reference,
            'max_iterations_per_stage': args.iterations,
        },
        'phase1_max_cpu_seconds': float(args.phase1_max_cpu_seconds),
        'time_max_cpu_seconds_per_stage': float(args.time_max_cpu_seconds),
        'temporal_semantics': OPTIMIZER_TEMPORAL_SEMANTICS,
        'optimizer_actuator_contract': {
            'throttle_min_drive': float(solver_overlay['mpc_throttle_min_drive']),
            'throttle_max': 1.0,
            'minimum_throttle_matches_production_mpc': True,
            'source': 'f1tenth_mpc/include/mpc_types.h MPC_THROTTLE_MIN_DRIVE',
        },
        'requested_warmstart_backend':args.seed_backend,
        'reference_speed_seed': (
            None if reference_speed_seed is None else {
                'source_csv': str(reference_speed_seed_path),
                'column': 'body_forward_speed_mps/vx_mps/speed_mps/velocity_mps',
                'used_as_constraint': False,
                'used_as_initial_state_only': True,
                'minimum_mps': float(np.min(reference_speed_seed)),
                'mean_mps': float(np.mean(reference_speed_seed)),
                'maximum_mps': float(np.max(reference_speed_seed)),
                'application': ('replace_checkpoint_u_mps_only'
                                if optimizer_seed_speed_replaced
                                else 'fresh_synthetic_speed_initializer'),
            }),
        'native_preflight_replay_does_not_certify_Unity':True,
        'source_feasibility_sample_gap_max_m':
            max_source_feasibility_sample_gap_m,
        'source_feasibility_sampling_is_continuous_certification':False,
        'stage':'setup_complete',
        'limitations':('Spatial OCP uses a source-derived wheel/contact model, '
            'with force, suspension, and wheel feasibility constraints at OCP '
            'nodes and each interval midpoint. This finite collocation is not '
            'continuous-time certification or exact Unity 1 ms scene integration. '
            'The exported target is not MPC-tracking-validated.'),
    }
    (args.output/'provenance.json').write_text(json.dumps(run_info,indent=2)+'\n')
    if args.setup_check:
        print('SETUP CHECK PASS (no IPOPT or PhysX evaluation launched)');return 0
    if (args.output/'ipopt_candidate_pass_00').exists():
        ap.error('Refusing to overwrite previous candidate; choose a fresh --output directory')
    if args.seed_file is not None and args.seed_backend == 'synthetic':
        ap.error('--seed-file requires --seed-backend native or source')
    if args.optimizer_seed_file is not None and args.seed_file is not None:
        ap.error('--optimizer-seed-file and --seed-file are mutually exclusive')
    # Native measured state initialization replaces the repeatedly stalled
    # curvature-derived seed. Never pass an incomplete or colliding lap to IPOPT.
    first_seed=None
    optimizer_seed = None
    optimizer_seed_modification = {}
    if args.optimizer_seed_file is not None:
        optimizer_seed, optimizer_seed_s = load_optimizer_seed_checkpoint(
            args.optimizer_seed_file.resolve(), track=track, map_path=args.map,
            centerline_path=args.centerline, model=model,
            allow_mesh_resample=args.resample_optimizer_seed)
        optimizer_seed_modification = optimizer_seed.pop(
            '_seed_modification', {})
        optimizer_seed_mesh_resampled = not (
            optimizer_seed_s.shape == np.asarray(track.s).shape
            and np.allclose(optimizer_seed_s, track.s, rtol=0.0, atol=1.0e-9))
        seed_reports=[{
            'backend':'same-run IPOPT checkpoint',
            'path':str(args.optimizer_seed_file.resolve()),
            'provenance_checked':True,
            'target_node_count':track.count,
            'source_node_count':int(len(optimizer_seed_s)),
            'mesh_resampled':optimizer_seed_mesh_resampled,
            'resampling_requires_phase1_recertification':
                optimizer_seed_mesh_resampled,
            'seed_modification': optimizer_seed_modification or None,
            'source_feasible_at_load': not bool(optimizer_seed_modification),
            'requires_phase1_recertification': bool(
                optimizer_seed_mesh_resampled or optimizer_seed_modification),
        }]
        if reference_speed_seed is not None:
            if optimizer_seed_mesh_resampled:
                ap.error('speed-profile seeding with a checkpoint '
                         'requires an exact-mesh seed; omit '
                         '--resample-optimizer-seed')
            if args.seed_speed_cap_to_reference_lateral_demand:
                if reference_lateral_demand_cap_mps2 is None:
                    ap.error('lateral-demand seed cap was requested without a '
                             'reference profile curvature')
                try:
                    path_curvature_seed = path_curvature_from_lateral_offset(
                        track, optimizer_seed['ey_m'])
                    uncapped_speed = reference_speed_seed.copy()
                    reference_speed_seed, cap_report = (
                        cap_speed_seed_to_reference_lateral_demand(
                            reference_speed_seed, path_curvature_seed,
                            reference_lateral_demand_cap_mps2))
                except ValueError as error:
                    ap.error(str(error))
                run_info['reference_speed_seed'][
                    'uncapped_profile_min_mean_max_mps'] = [
                        float(np.min(uncapped_speed)),
                        float(np.mean(uncapped_speed)),
                        float(np.max(uncapped_speed))]
                run_info['reference_speed_seed'][
                    'checkpoint_path_lateral_demand_cap'] = cap_report
                run_info['reference_speed_seed'].update({
                    'minimum_mps': float(np.min(reference_speed_seed)),
                    'mean_mps': float(np.mean(reference_speed_seed)),
                    'maximum_mps': float(np.max(reference_speed_seed)),
                })
                print('Curvature-aware checkpoint speed initializer: '
                      f'min/mean/max={np.min(reference_speed_seed):.3f}/'
                      f'{np.mean(reference_speed_seed):.3f}/'
                      f'{np.max(reference_speed_seed):.3f} m/s; clipped '
                      f'{cap_report["clipped_sample_count"]} nodes; '
                      f'peak demand {cap_report["candidate_path_peak_lateral_demand_after_mps2"]:.3f} '
                      'm/s^2.', flush=True)
            speed_blend_report = None
            if args.checkpoint_speed_seed_fraction != 1.0:
                try:
                    reference_speed_seed, speed_blend_report = (
                        blend_speed_seed_with_checkpoint(
                            reference_speed_seed, optimizer_seed['u_mps'],
                            args.checkpoint_speed_seed_fraction))
                except ValueError as error:
                    ap.error(str(error))
                run_info['reference_speed_seed'][
                    'checkpoint_speed_profile_homotopy'] = speed_blend_report
                print('Checkpoint speed homotopy initializer: '
                      f'fraction={args.checkpoint_speed_seed_fraction:.3f}, '
                      f'mean={speed_blend_report["checkpoint_profile_mean_mps"]:.3f}'
                      ' -> '
                      f'{speed_blend_report["blended_profile_mean_mps"]:.3f} m/s '
                      '(target '
                      f'{speed_blend_report["target_profile_mean_mps"]:.3f} m/s).',
                      flush=True)
            try:
                optimizer_seed = replace_optimizer_seed_speed_profile(
                    optimizer_seed, reference_speed_seed,
                    minimum_mps=0.35, maximum_mps=args.max_speed,
                    mode=args.checkpoint_speed_seed_mode,
                    steering_rate_limit_radps=(
                        model.steering_rate_radps
                        if args.checkpoint_speed_seed_mode == 'time-scaled'
                        else None))
            except ValueError as error:
                ap.error(str(error))
            optimizer_seed_speed_replaced = True
            run_info['reference_speed_seed']['application'] = (
                'replace_checkpoint_speed_profile_with_'
                + args.checkpoint_speed_seed_mode)
            hybrid_seed_checkpoint_path = (
                args.output / 'initial_optimizer_seed_speed_replaced.npz')
            checkpoint_candidate = {
                candidate_name: optimizer_seed[seed_name]
                for seed_name, candidate_name in _OPTIMIZER_SEED_FIELDS}
            write_optimizer_seed_checkpoint(
                checkpoint_candidate, hybrid_seed_checkpoint_path,
                track=track, map_path=args.map,
                centerline_path=args.centerline, model=model)
            seed_reports[0]['speed_profile_replacement'] = {
                'source_csv': str(reference_speed_seed_path),
                'mode': args.checkpoint_speed_seed_mode,
                'modified_fields': (
                    ['u_mps'] if args.checkpoint_speed_seed_mode == 'speed-only'
                    else ['u_mps', 'v_mps', 'yaw_rate_radps',
                          'wheel_omega_radps', 'steering_rate_radps',
                          'body_height_rate_mps', 'roll_rate_radps',
                          'pitch_rate_radps', 'ax_mps2', 'ay_mps2']),
                'preserved_fields': [
                    seed_name for seed_name, _ in _OPTIMIZER_SEED_FIELDS
                    if seed_name != 'u_mps'],
                'speed_bounds_mps': [0.35, float(args.max_speed)],
                'profile_min_mps': float(np.min(reference_speed_seed)),
                'profile_mean_mps': float(np.mean(reference_speed_seed)),
                'profile_max_mps': float(np.max(reference_speed_seed)),
                'checkpoint_speed_profile_homotopy': speed_blend_report,
                'steering_rate_clamped_to_source_bound': (
                    args.checkpoint_speed_seed_mode == 'time-scaled'),
                'derived_checkpoint': str(hybrid_seed_checkpoint_path),
                'source_equations_must_be_recertified': True,
            }
            print('Hybrid seed: kept the full checkpoint and applied '
                  f'{args.checkpoint_speed_seed_mode} speed-profile seeding; '
                  f'saved {hybrid_seed_checkpoint_path}.', flush=True)
        print('== Reusing verified source-feasible IPOPT checkpoint: '
              + str(args.optimizer_seed_file.resolve()) + ' ==', flush=True)
    elif args.seed_file is not None:
        provenance_audit=validate_seed_provenance(args.seed_file,args.map,args.centerline)
        first_seed=load_native_seed(
            args.seed_file.resolve(), track.count,
            source_length_m=provenance_audit.get('source_track_length_m'),
            target_length_m=track.length)
        seed_reports=[{'backend':args.seed_backend,'completed_flying_lap':True,
                       'source':'reused_native_warmstart_file',
                       'path':str(args.seed_file.resolve()),
                       'provenance_check':provenance_audit,
                       'target_node_count':track.count,
                       'mesh_resampled':len(first_seed['u_mps']) != int(
                           provenance_audit.get('source_node_count', track.count)),
                       'note':'matching map/centerline checksums and lap length verified; '
                             'uniform periodic mesh resampled only when node count differs'}]
        print('== Reusing physics warm-start file: '+str(args.seed_file.resolve())+' ==',flush=True)
    elif (optimizer_seed is None and args.phase1_resume_report is None
          and args.seed_backend != 'synthetic'
          and not args.diagnose_initial_only):
        first_seed,seed_reports=generate_native_seed(
            args.repo_root,args.physx_binary,args.seed_backend,init_ref,grid,
            track,args.output/'warmstart_physx',args.wall_margin,
            capture_native_diagnostics=args.capture_native_diagnostics)
        if first_seed is None:
            (args.output/'report.json').write_text(json.dumps({
                'status':'no_complete_safe_native_seed',
                'backend':args.seed_backend,'trials':seed_reports,
                'native_validated':False,'lap_target_s':4.9},indent=2)+'\n')
            print('No complete, wall-clear physics warm-start lap. '
                  'IPOPT was NOT launched; inspect warmstart_physx/native_seed_report.json '
                  'and seed_trace_*.csv. Fix the path follower or use '
                  '--seed-backend synthetic for the old behavior.',file=sys.stderr)
            return 5
        print('== Complete PhysX motion seed loaded; optimizing from observed '
              'wheel/body states instead of synthetic curvature dynamics ==',flush=True)
        run_info['warmstart_preflight']=seed_reports
        run_info['warmstart_state_path']=str(args.output/'warmstart_physx/native_warmstart.npz')
        if args.capture_native_diagnostics:
            run_info['stage'] = 'native_diagnostics_capture_complete'
            run_info['ipopt_invoked'] = False
            run_info['native_witness_replay_invoked'] = False
        (args.output/'provenance.json').write_text(json.dumps(run_info,indent=2)+'\n')
    if args.capture_native_diagnostics:
        (args.output/'report.json').write_text(json.dumps({
            'status':'native_diagnostics_capture_complete',
            'warmstart_preflight':seed_reports,
            'capture_manifest':str(
                args.output/'warmstart_physx/native_diagnostics_manifest.json'),
            'ipopt_invoked':False,
            'native_witness_replay_invoked':False,
            'unity_validated':False,
        },indent=2)+'\n')
        print('Native diagnostics captured; IPOPT and witness replay were not run.',
              flush=True)
        return 0
    if optimizer_seed is not None:
        run_info['optimizer_checkpoint_seed'] = seed_reports[0]
        initial_guess_source = {
            'kind': (
                'provenance_matched_checkpoint_with_geometry_modification'
                if optimizer_seed_modification else
                'source_feasible_checkpoint_with_reference_speed_replaced'
                if optimizer_seed_speed_replaced else
                'source_feasible_optimizer_checkpoint'),
            'path':str(args.optimizer_seed_file.resolve()),
            'requested_backend':args.seed_backend,
            'provenance_check':'map, centerline, spatial mesh, and vehicle model matched',
            'modified_fields': (
                list(optimizer_seed_modification.get('modified_fields', []))
                + ([] if not optimizer_seed_speed_replaced else
                   seed_reports[0]['speed_profile_replacement']['modified_fields'])
            ),
            'preserved_full_state_checkpoint': (
                not optimizer_seed_modification),
            'seed_modification': optimizer_seed_modification or None,
            'derived_checkpoint_path': (
                None if hybrid_seed_checkpoint_path is None
                else str(hybrid_seed_checkpoint_path)),
        }
    elif first_seed is not None:
        # Also persist reused-seed provenance and any verified periodic mesh
        # resampling; otherwise a later report could hide which initial grid
        # supplied its state trajectory.
        run_info['warmstart_preflight'] = seed_reports
        run_info['warmstart_state_path'] = str(
            args.seed_file.resolve() if args.seed_file is not None else
            args.output/'warmstart_physx/native_warmstart.npz')
        (args.output/'provenance.json').write_text(
            json.dumps(run_info, indent=2)+'\n')
        initial_guess_source = {
            'kind': ('seed_file' if args.seed_file is not None
                     else f'{args.seed_backend}_warmstart'),
            'path': str(args.seed_file.resolve() if args.seed_file is not None
                        else args.output/'warmstart_physx/native_warmstart.npz'),
            'requested_backend': args.seed_backend,
        }
    elif args.phase1_resume_report is not None:
        initial_guess_source = {
            'kind': 'saved_phase1_dynamics_infeasible_checkpoint',
            'path': str(args.phase1_resume_report.resolve()),
            'requested_seed_backend_ignored': args.seed_backend,
            'note': ('The saved checkpoint is only an initialization for a '
                     'new strict Phase-I solve; it is not a feasible lap.'),
        }
    else:
        initial_guess_source = {
            'kind': ('synthetic_reference_speed_profile'
                     if reference_speed_seed is not None else 'synthetic'),
            'path': None,
            'requested_backend': args.seed_backend,
            'reference_speed_seed_used': reference_speed_seed is not None,
            'explicitly_selected': args.seed_backend == 'synthetic',
            'note': 'No captured/source warm-start state was supplied; '
                    'the optimizer constructed its synthetic curvature seed.',
        }
    run_info['actual_initial_guess_source'] = initial_guess_source
    (args.output/'provenance.json').write_text(
        json.dumps(run_info, indent=2)+'\n')
    phase1_checkpoint_seed = None
    if args.phase1_resume_report is not None:
        try:
            phase1_checkpoint_seed = load_phase1_resume_report(
                args.phase1_resume_report,
                map_path=args.map,
                centerline_path=args.centerline,
                mesh_path=reference_track_mesh_path,
                track=track,
                model_metadata=model.metadata(),
                optimizer_source_sha256=(
                    solver_overlay.get('optimizer_source_sha256')),
                phase1_formulation=args.phase1_formulation)
        except (OSError, ValueError, KeyError, TypeError) as error:
            ap.error(str(error))
        run_info['phase1_resume_source'] = {
            'report': str(args.phase1_resume_report.resolve()),
            'raw_max_dynamic_defect': phase1_checkpoint_seed[3],
            'hard_feasible_only': True,
            'strict_original_dynamics_feasible': False,
            'source_and_mesh_verified': True,
        }
        (args.output/'provenance.json').write_text(
            json.dumps(run_info, indent=2)+'\n')
        print('== Resuming Phase I from provenance-matched checkpoint: max '
              f'dynamics defect {phase1_checkpoint_seed[3]:.9g}; this is not '
              'a feasible lap ==', flush=True)
    elif args.phase1_warmstart_report is not None:
        if args.phase1_formulation != 'eliminated' or args.seed_file is None:
            ap.error('--phase1-warmstart-report requires eliminated formulation and a matching --seed-file')
        old_report = json.loads(args.phase1_warmstart_report.read_text())
        if (old_report.get('phase1_strict_feasible') is not False
                or old_report.get('solver', {}).get('feasible_incumbent') is not False
                or old_report.get('phase1_restart_checkpoint') is None):
            ap.error('Checkpoint source is not a verified failed Phase-I report')
        if old_report.get('unity_vehicle_model') != json.loads(json.dumps(model.metadata())):
            ap.error('Checkpoint source vehicle model differs from current source parameters')
        source_track = old_report.get('track', {})
        if (int(source_track.get('count', -1)) != track.count
                or abs(float(source_track.get('length_m', -1)) - track.length) > 1.e-6):
            ap.error('Checkpoint source track differs from current OCP')
        phase1_checkpoint_seed = validate_restart_checkpoint(
            old_report['phase1_restart_checkpoint'], track.count,
            track.length, 1)
        print('== Reusing verified dynamics-infeasible checkpoint: max defect '
              f'{phase1_checkpoint_seed[3]:.9g} (not a racing lap) ==', flush=True)

    print(f'== IPOPT OCP: {track.count} nodes, drive-only throttle envelope, '
          f'{args.feasibility_iterations} feasibility + {args.iterations} time iterations, '
          f'Phase-I CPU cap={args.phase1_max_cpu_seconds:g}s, '
          f'time-stage cap={args.time_max_cpu_seconds:g}s/stage, '
          f'Phase-I seed-distance cap={args.feasibility_trust_region:g}, seed={args.seed_backend}, '
          f'Phase-I {args.phase1_formulation} dynamics + safe spatial evaluations + simple bounds, '
          f'Phase-I Hessian={args.phase1_hessian} ==',flush=True)
    phase = 0
    print('== IPOPT feasibility -> minimum-time solve ==',flush=True)
    for _ in range(1):
        kwargs=dict(
            wall_clearance_m=args.wall_margin,
            feasibility_trust_region=args.feasibility_trust_region,
            diagnose_initial_only=args.diagnose_initial_only,
            max_speed_mps=args.max_speed,
            feasibility_max_iterations=args.feasibility_iterations,
            time_optimization_max_iterations=args.iterations,
            time_hessian_approximation=args.time_hessian_approximation,
            minimum_trust_region_reference=(
                args.minimum_trust_region_reference),
            print_level=args.print_level)
        if args.time_trust_region_factors is not None:
            kwargs['time_trust_region_factors'] = (
                args.time_trust_region_factors)
        if reference_speed_seed is not None and optimizer_seed is None:
            kwargs['initial_speed_seed_mps'] = reference_speed_seed
        if phase1_checkpoint_seed is not None:
            cp_seed, cp_station, cp_lengths, cp_defect, _ = phase1_checkpoint_seed
            kwargs.update(initial_state_seed=cp_seed,
                          initial_state_seed_s=cp_station,
                          phase_lengths_initial=cp_lengths,
                          initial_state_seed_is_model_consistent=True)
        elif optimizer_seed is not None:
            kwargs.update(initial_state_seed=optimizer_seed,
                          initial_state_seed_s=optimizer_seed_s,
                          initial_state_seed_is_model_consistent=
                              preserve_optimizer_seed_inputs(
                                  optimizer_seed_modification,
                                  speed_replaced=optimizer_seed_speed_replaced,
                                  reproject_requested=args.reproject_optimizer_seed))
        elif first_seed is not None:
            # The native measured differential state is not algebraically
            # consistent with the source-translated symbolic force model.
            # Compute the *symbolic* suspension loads and tire-force-derived
            # accelerations before seeding IPOPT, but retain native u/v/yaw.
            if not args.no_seed_force_projection:
                seed,seed_audit=project_algebraics(
                    first_seed,model,wheel,
                    suspension_loads_fn=old._unity_suspension_loads_numpy,
                    body_acceleration_fn=old._unity_body_acceleration_numpy,
                    minimum_drive_throttle=min_drive_throttle)
            else:
                seed={k:np.asarray(v).copy() for k,v in first_seed.items()}
                seed['throttle_norm'] = np.clip(
                    seed['throttle_norm'], min_drive_throttle, 1.0)
                seed_audit={'type':'unprojected_native_state',
                    'minimum_drive_throttle_applied': min_drive_throttle}
            write_audit(args.output/'warmstart_physx/seed_projection_audit.json',seed_audit)
            print('Source-model seed projection audit:',json.dumps(seed_audit,sort_keys=True),flush=True)
            kwargs.update(initial_state_seed=seed,
                          initial_state_seed_s=np.asarray(track.s),
                          initial_state_seed_is_model_consistent=True)
        if args.phase1_max_cpu_seconds > 0.0:
            os.environ['SDU_IPOPT_PHASE1_MAX_CPU_SECONDS'] = str(
                args.phase1_max_cpu_seconds)
        else:
            os.environ.pop('SDU_IPOPT_PHASE1_MAX_CPU_SECONDS', None)
        if args.time_max_cpu_seconds > 0.0:
            os.environ['SDU_IPOPT_TIME_MAX_CPU_SECONDS'] = str(
                args.time_max_cpu_seconds)
        else:
            os.environ.pop('SDU_IPOPT_TIME_MAX_CPU_SECONDS', None)
        if args.audit_time_checkpoint is not None:
            if not args.skip_native or args.optimizer_seed_file is None:
                ap.error('NLP audit requires --skip-native and the original --optimizer-seed-file')
            with np.load(args.audit_time_checkpoint, allow_pickle=False) as audit_archive:
                audit_factor = float(audit_archive['trust_factor'])
                audit_geometry = (str(audit_archive['trust_geometry'].item())
                    if 'trust_geometry' in audit_archive else 'aggregate-checkpoint')
            kwargs['time_trust_region_factors'] = (audit_factor, audit_factor*2.)
            # Existing captures used the aggregate (mesh-dependent) box.
            # Reconstruct that graph exactly instead of auditing new bounds.
            audit_prior_trust_geometry = os.environ.get('SDU_IPOPT_TRUST_GEOMETRY')
            os.environ['SDU_IPOPT_TRUST_GEOMETRY'] = audit_geometry
            os.environ['SDU_IPOPT_AUDIT_CHECKPOINT'] = str(args.audit_time_checkpoint.resolve())
            os.environ['SDU_IPOPT_AUDIT_OUTPUT'] = str((args.output/'nlp_audit.json').resolve())
        from ipopt_nlp_audit import NlpAuditComplete
        try:
            candidate,report,phase1_progress=run_phase1_or_diagnose(
                old,track,model,kwargs,output=args.output,phase=phase,
                max_restarts=args.phase1_restarts,
                minimum_improvement=args.phase1_min_defect_improvement,
                diagnose_initial_only=args.diagnose_initial_only)
        except NlpAuditComplete:
            (args.output/'report.json').write_text(json.dumps({
                'status': 'checkpoint_audit_only', 'ipopt_time_solve_invoked': False,
                'unity_validated': False, 'trajectory_promoted': False}, indent=2)+'\n')
            return 0
        finally:
            if args.audit_time_checkpoint is not None:
                os.environ.pop('SDU_IPOPT_AUDIT_CHECKPOINT', None)
                os.environ.pop('SDU_IPOPT_AUDIT_OUTPUT', None)
                if audit_prior_trust_geometry is None:
                    os.environ.pop('SDU_IPOPT_TRUST_GEOMETRY', None)
                else:
                    os.environ['SDU_IPOPT_TRUST_GEOMETRY'] = audit_prior_trust_geometry
        report['initial_guess_source'] = initial_guess_source
        report['reference_speed_seed'] = run_info['reference_speed_seed']
        report['requested_seed_backend'] = args.seed_backend
        report['phase1_progress'] = phase1_progress
        report['phase1_solver_overlay'] = solver_overlay
        annotate_spatial_temporal_semantics(
            report,
            unity_source_fixed_step_s=float(
                solver_overlay['unity_source_fixed_step_s']))
        (args.output/f'ipopt_report_pass_{phase:02d}.json').write_text(
            json.dumps(report,indent=2,default=str)+'\n')
        if args.diagnose_initial_only:
            print_feasibility_diagnostics(report,args.feasibility_trust_region)
            (args.output/'report.json').write_text(json.dumps({
                'status':'initial_guess_diagnostic_only',
                'ipopt_invoked':False,
                'native_replay_invoked':False,
                'phase':phase,
                'initial_guess_source': initial_guess_source,
                'initial_raw_dynamic_max_defect': report.get(
                    'nlp_diagnostics',{}).get(
                        'initial_raw_dynamic_max_defect'),
                'source_feasibility_sample_gap_max_m':
                    max_source_feasibility_sample_gap_m,
                'source_feasibility_sampling_is_continuous_certification': False,
                'native_validated':False,
                'lap_target_s':4.9},indent=2)+'\n')
            print('Initial-state constraint diagnostics written; no IPOPT '
                  'or native PhysX run.',file=sys.stderr)
            return 0
        if not candidate or not report.get('solver',{}).get('feasible_incumbent',False):
            print_feasibility_diagnostics(report,args.feasibility_trust_region)
            (args.output/'report.json').write_text(json.dumps({
                'status':'ipopt_not_feasible','phase':phase,
                'native_validated':False, 'lap_target_s':4.9},indent=2)+'\n')
            print('IPOPT did not find a fully feasible periodic lap. '
                  'Inspect phase-I defect-continuation diagnostics in ipopt_report_pass_XX.json. No MPC raceline exported.',file=sys.stderr)
            return 2
        target_path = args.output / (
            f'optimizer_target_raceline_pass_{phase:02d}_not_mpc_validated.csv')
        export_mpc(candidate, grid, target_path, args.wall_margin)
        report['raceline_target_export'] = {
            'path': str(target_path),
            'semantics': 'optimized geometry, body-forward speed, and signed curvature target; no actuator commands',
            'curvature_export': (
                'optimizer Menger curvature preserved; independently checked '
                'against the exported XY samples'),
            'curvature_geometry_check_passed': True,
            'map_footprint_checked': True,
            'map_check_method': (
                'dense swept footprint samples with conservative distance-map '
                'interval bound'),
            'source_model_feasibility_sampling': (
                'OCP knots and each spatial segment midpoint for wheel speed, '
                'suspension travel/load, steering bounds, and all six '
                'body force/moment balances'),
            'source_feasibility_sample_gap_max_m':
                max_source_feasibility_sample_gap_m,
            'source_feasibility_sampling_is_continuous_certification': False,
            'continuous_dynamic_feasibility_proven': False,
            'mpc_tracking_validated': False,
            'unity_validated': False,
        }
        candidate_dir=args.output/f'ipopt_candidate_pass_{phase:02d}'
        old.write_candidate(candidate,report,track,args.centerline,candidate_dir)
        optimizer_checkpoint_path = candidate_dir / 'optimizer_seed.npz'
        write_optimizer_seed_checkpoint(
            candidate, optimizer_checkpoint_path, track=track,
            map_path=args.map, centerline_path=args.centerline, model=model)
        report['optimizer_seed_checkpoint'] = {
            'path':str(optimizer_checkpoint_path),
            'provenance_bound':True,
            'same_mesh_required':True,
            'source_feasibility_required':True,
        }
        midpoint_witness_path=candidate_dir/'unity_wheel_feasibility_witness_midpoints.csv'
        from command_realizability import validate_exported_candidate
        command_report = validate_exported_candidate(
            candidate, track, args.repo_root, candidate_dir/'command_realizability.json')
        report['source_command_realizability'] = command_report
        write_midpoint_witness(candidate,midpoint_witness_path)
        report['feasibility_witness_midpoints']={
            'path':str(midpoint_witness_path),
            'rows':int(len(candidate['midpoint_s_ref_m'])),
            'sampling':'one source-model evaluation at every OCP segment midpoint',
            'continuous_dynamic_feasibility_proven':False,
        }
        (candidate_dir/'optimizer_report.json').write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str)+'\n')
        (args.output/f'ipopt_report_pass_{phase:02d}.json').write_text(
            json.dumps(report,indent=2,default=str)+'\n')
        print(f'Optimizer target exported (not yet MPC-validated): {target_path}',
              flush=True)
        print('IPOPT predicted full-lap time:',report.get('predicted_lap_time_s',
                                                         report.get('lap_time_s','see report')),flush=True)
        if args.skip_native:
            (args.output/'report.json').write_text(json.dumps({
                'status':'ocp_target_exported_native_witness_skipped','phase':phase,
                'raceline_target':str(target_path),
                'raceline_semantics':'geometry/speed/curvature target, not commands',
                'mpc_tracking_validated':False,'unity_validated':False,
                'lap_target_s':4.9},indent=2)+'\n')
            print('Native feasibility-witness replay intentionally skipped; target remains '
                  'unvalidated in MPC/Unity.',file=sys.stderr)
            return 3
        print('== Replay internal OCP control witness through the source native plant ==',flush=True)
        print('This is a feasibility-witness diagnostic, not MPC tracking or Unity validation.',
              flush=True)
        native_dir=args.output/f'native_pass_{phase:02d}'
        result=replay(args.repo_root,args.physx_binary,candidate,track,grid,init_ref,
                      native_dir,args.wall_margin,args.warmup_laps,
                      args.native_max_seconds,
                      minimum_drive_throttle=min_drive_throttle)
        (args.output/'report.json').write_text(json.dumps({
            'phase':phase,
            'status':'control_witness_replayed' if result['completed'] and result['collision_free']
                     else 'control_witness_replay_failed',
            'optimizer':'restored CasADi/IPOPT Unity-wheel spatial OCP',
            'raceline_target':str(target_path),
            'raceline_semantics':'geometry/speed/curvature target, not commands',
            'control_witness_replay':result,
            'control_witness_replayed':bool(result['completed'] and result['collision_free']),
            'mpc_tracking_validated':False,'unity_validated':False,
            'lap_target_s':4.9},indent=2,default=str)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='source_notes'},indent=2),flush=True)
        if result['completed'] and result['collision_free']:
            print('The source-native control witness completed; this does not establish '
                  'MPC tracking or Unity parity.',flush=True)
            print(f'Witness lap time: {result["native_flying_lap_time_s"]:.3f}s; '
                  f'optimizer target remains {target_path}',flush=True)
            return 0
        print('Control-witness replay failed; the optimizer target is retained for '
              'analysis and has NOT been declared MPC-trackable.',file=sys.stderr)
        return 4
    return 4

if __name__=='__main__':
    try:
        sys.exit(run())
    except (Exception,KeyboardInterrupt) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}',file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
