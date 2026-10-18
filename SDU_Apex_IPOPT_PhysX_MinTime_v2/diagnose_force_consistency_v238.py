#!/usr/bin/env python3
"""Compare native-PhysX seed to the exact scalar tire-force formula used by IPOPT.

No IPOPT solve and no native simulation are performed. This cannot certify a
racing lap. Its purpose is to localize a discrepancy before modifying physics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import casadi as ca
import numpy as np

from recover import load_modules
from source_vehicle import current_model, bind_current_wheel_moment_arms
from seed_projection import load_native_seed, project_algebraics, validate_seed_provenance


def evaluate_scalar_tire_forces(model, wheel, seed, loads):
    """Evaluate the wheel model one node at a time, as IPOPT constructs it."""
    n = len(seed['u_mps'])
    z = ca.MX.sym('node_forces', 12)
    fx, fy, _, fx_w, fy_w, sx, sy = wheel.tire_forces_casadi(
        model, z[0], z[1], z[2], z[3],
        [z[4+i] for i in range(4)], [z[8+i] for i in range(4)])
    f = ca.Function('scalar_physx_tire_audit', [z],
                    [fx, fy, ca.vertcat(*fx_w), ca.vertcat(*fy_w),
                     ca.vertcat(*sx), ca.vertcat(*sy)])
    data = np.vstack((seed['u_mps'], seed['v_mps'], seed['yaw_rate_radps'],
                      seed['steering_rad'], seed['wheel_omega_radps'], loads))
    values = [f(data[:, k]) for k in range(n)]
    fx_body = np.array([float(v[0]) for v in values])
    fy_body = np.array([float(v[1]) for v in values])
    fx_wheels = np.stack([np.asarray(v[2], dtype=float).reshape(4) for v in values])
    fy_wheels = np.stack([np.asarray(v[3], dtype=float).reshape(4) for v in values])
    sx = np.stack([np.asarray(v[4], dtype=float).reshape(4) for v in values])
    sy = np.stack([np.asarray(v[5], dtype=float).reshape(4) for v in values])
    return fx_body, fy_body, fx_wheels, fy_wheels, sx, sy


def analyze_seed(seed, model, wheel):
    """Return summary and node rows; no model constants are tuned."""
    symbolic, projection = project_algebraics(seed, model, wheel)
    measured_fx, measured_fy, _, _, _, _ = evaluate_scalar_tire_forces(
        model, wheel, seed, seed['normal_load_n'])
    projected_fx, projected_fy, fwx, fwy, slips_x, slips_y = evaluate_scalar_tire_forces(
        model, wheel, symbolic, symbolic['normal_load_n'])
    native_ax = np.asarray(seed['ax_mps2'])
    native_ay = np.asarray(seed['ay_mps2'])
    scalar_native_ax = measured_fx / model.mass_kg - model.drag_per_s * seed['u_mps']
    scalar_native_ay = measured_fy / model.mass_kg - model.drag_per_s * seed['v_mps']
    scalar_projected_ax = projected_fx / model.mass_kg - model.drag_per_s * seed['u_mps']
    scalar_projected_ay = projected_fy / model.mass_kg - model.drag_per_s * seed['v_mps']
    residual_u = native_ax - scalar_native_ax
    residual_v = native_ay - scalar_native_ay
    projection_residual = max(
        np.max(np.abs(symbolic['ax_mps2'] - scalar_projected_ax)),
        np.max(np.abs(symbolic['ay_mps2'] - scalar_projected_ay)))
    if projection_residual > 1.e-8:
        raise RuntimeError('The native seed projector does not satisfy the IPOPT scalar force equations! '
                           f'Worst error={projection_residual:.9g} m/s²')
    n = len(native_ax)
    worst = np.argsort(-np.abs(residual_v))[:12]
    summary = {
        'diagnostic': 'scalar_tire_source_vs_native_seed_no_ipopt_no_native_replay',
        'nodes': n,
        'unprojected_lateral_force_balance_max_abs_mps2': float(np.max(np.abs(residual_v))),
        'unprojected_lateral_force_balance_rms_mps2': float(np.sqrt(np.mean(residual_v**2))),
        'unprojected_longitudinal_force_balance_max_abs_mps2': float(np.max(np.abs(residual_u))),
        'symbolic_load_model_max_abs_difference_n': float(np.max(np.abs(
            symbolic['normal_load_n'] - seed['normal_load_n']))),
        'symbolic_load_model_max_scaled_residual': float(np.max(np.abs(
            symbolic['normal_load_n'] - seed['normal_load_n'])))/10.,
        'projected_force_balance_max_residual_mps2': float(projection_residual),
        'projected_ay_vs_native_max_abs_mps2': float(np.max(np.abs(native_ay - scalar_projected_ay))),
        'projected_ay_vs_native_rms_mps2': float(np.sqrt(np.mean(
            (native_ay - scalar_projected_ay)**2))),
        'projected_input_force_vs_native_load_input_max_abs_n': float(np.max(np.abs(
            projected_fy - measured_fy))),
        'original_vs_scalar_vector_projection_max_force_error_n': projection[
            'scalar_vs_vectorized_tire_force_max_error_n'],
        'worst_unprojected_lateral_nodes': [{
            'k': int(k), 'residual_mps2': float(residual_v[k]),
            'native_ay_mps2': float(native_ay[k]),
            'symbolic_ay_with_native_loads_mps2': float(scalar_native_ay[k]),
            'symbolic_ay_with_symbolic_loads_mps2': float(scalar_projected_ay[k])
        } for k in worst],
        'caveat': 'Different PhysX contact/load integration and differential-model discretization are NOT reconciled by algebraic projection.'
    }
    rows = []
    for k in range(n):
        row = {'k': k, 'u_mps': float(seed['u_mps'][k]),
            'v_mps': float(seed['v_mps'][k]),
            'yaw_rate_radps': float(seed['yaw_rate_radps'][k]),
            'native_ax_mps2': float(native_ax[k]),
            'native_ay_mps2': float(native_ay[k]),
            'symbolic_ax_native_load_mps2': float(scalar_native_ax[k]),
            'symbolic_ay_native_load_mps2': float(scalar_native_ay[k]),
            'symbolic_ax_projected_load_mps2': float(scalar_projected_ax[k]),
            'symbolic_ay_projected_load_mps2': float(scalar_projected_ay[k]),
            'unprojected_longitudinal_residual_mps2': float(residual_u[k]),
            'unprojected_lateral_residual_mps2': float(residual_v[k]),
            'total_fx_projected_load_n': float(projected_fx[k]),
            'total_fy_projected_load_n': float(projected_fy[k])}
        for i, tag in enumerate(('fl','fr','rl','rr')):
            row[f'wheel_{tag}_native_fz_n'] = float(seed['normal_load_n'][i,k])
            row[f'wheel_{tag}_symbolic_fz_n'] = float(symbolic['normal_load_n'][i,k])
            row[f'wheel_{tag}_fx_n'] = float(fwx[k,i])
            row[f'wheel_{tag}_fy_n'] = float(fwy[k,i])
            row[f'wheel_{tag}_slip_long'] = float(slips_x[k,i])
            row[f'wheel_{tag}_slip_lat'] = float(slips_y[k,i])
        rows.append(row)
    return summary, rows


def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=Path('..'))
    ap.add_argument('--seed-file', type=Path, default=Path(
        'optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz'))
    ap.add_argument('--output', type=Path, default=Path('force_consistency_v238'))
    ap.add_argument('--allow-history-fetch', action='store_true')
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    pkg = Path(__file__).resolve().parent
    seed_path = args.seed_file.resolve()
    mapfile = repo/'f1tenth_planning/maps/autodrive_practice_20260924_b.pgm'
    center = repo/'f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv'
    checksums = validate_seed_provenance(seed_path, mapfile, center)
    old, wheel, trackmod, _, overlay = load_modules(repo, pkg,
        args.allow_history_fetch, 'limited-memory')
    model, _ = current_model(repo, wheel)
    bind_current_wheel_moment_arms(repo, old, wheel)
    track = trackmod.load_centerline_csv(center, .20)
    seed = load_native_seed(seed_path, track.count)
    summary, rows = analyze_seed(seed, model, wheel)
    summary['seed_path'] = str(seed_path)
    summary['source_hash'] = overlay['working_solver_sha256']
    summary['seed_map_checksums'] = checksums
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'per_node.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k != 'seed_map_checksums'},
                     indent=2))
    print(f'Saved {out}/summary.json and {out}/per_node.csv')
    print('NO IPOPT solve or native replay ran. NO racing lap was generated.')
    return 0

if __name__ == '__main__':
    raise SystemExit(run())
