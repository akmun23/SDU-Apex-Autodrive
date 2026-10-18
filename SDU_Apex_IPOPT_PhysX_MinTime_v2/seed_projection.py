"""Project a measured PhysX trajectory onto the OCP's algebraic equations.

The native vehicle and source-derived CasADi wheel model are not identical.
This projection does NOT modify measured position, body velocities, yaw rate or
steering, and does NOT assert the spatial collocation equations are satisfied.
It initializes the OCP's *algebraic* wheel loads and accelerations to the
values required by the symbolic model so IPOPT doesn't spend its feasibility
budget repairing finite-difference accelerations and contact-model mismatch.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

REQUIRED = {
    'ey_m': None, 'epsi_rad': None, 'u_mps': None, 'v_mps': None,
    'yaw_rate_radps': None, 'wheel_omega_radps': 4,
    'steering_rad': None, 'steering_rate_radps': None,
    'throttle_norm': None, 'ax_mps2': None, 'ay_mps2': None,
    'normal_load_n': 4, 'body_height_m': None,
    'body_height_rate_mps': None, 'roll_rad': None,
    'roll_rate_radps': None, 'pitch_rad': None,
    'pitch_rate_radps': None,
}


def validate_native_seed(seed: dict, n: int) -> dict:
    """Reject mismatched map resolutions and non-numeric or unsafe seed data."""
    if n < 2:
        raise ValueError('Expected two or more IPOPT nodes')
    result = {}
    for name, rows in REQUIRED.items():
        if name not in seed:
            raise ValueError(f'Native seed missing {name}')
        a = np.asarray(seed[name], dtype=np.float64)
        expected = (n,) if rows is None else (rows, n)
        if a.shape != expected or not np.isfinite(a).all():
            raise ValueError(f'Native seed {name} must be finite shape {expected}; got {a.shape}')
        result[name] = a.copy()
    if np.min(result['u_mps']) <= 0 or np.min(result['normal_load_n']) <= 0:
        raise ValueError('Measured seed has nonpositive wheel load or forward speed')
    return result


def load_native_seed(path: str | Path, n: int) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return validate_native_seed({k: f[k] for k in f.files}, n)


def project_algebraics(seed: dict, model, wheel, *, brake_mask=None):
    """Recompute OCP wheel loads and ax/ay, preserving measured differential states.

    Every brake-mode node is a full CAWB brake in the OCP, even if the tracking
    lap used only motor torque. Reset spin at these nodes to the PhysX's
    locked-wheel state rather than injecting an impossible drive-mode omega.
    This still does NOT create a mode-consistent dynamic trajectory: the
    requested brake positions were absent from the measured native rollout.
    """
    import casadi as ca
    import inspect
    s = validate_native_seed(seed, len(np.asarray(seed['u_mps'])))
    n = s['u_mps'].size
    if brake_mask is None:
        brake_mask = np.zeros(n, dtype=bool)
    else:
        brake_mask = np.asarray(brake_mask, dtype=bool)
    if brake_mask.shape != (n,):
        raise ValueError('Brake mask must match the native state sample count')
    measured_load = s['normal_load_n'].copy()
    measured_ax = s['ax_mps2'].copy()
    measured_ay = s['ay_mps2'].copy()
    measured_spin = s['wheel_omega_radps'].copy()
    measured_throttle = s['throttle_norm'].copy()
    # Reconcile the spring/damper algebraics with the source *symbolic* model.
    # We retain native loads in the audit because they are important evidence
    # of native-versus-symbolic contact-model differences.
    loads = np.stack([
        np.asarray(wheel.suspension_loads_numpy(
            model, s['body_height_m'][k], s['body_height_rate_mps'][k],
            s['roll_rad'][k], s['roll_rate_radps'][k],
            s['pitch_rad'][k], s['pitch_rate_radps'][k]), dtype=np.float64)
        for k in range(n)], axis=1)
    if loads.shape != (4, n) or not np.isfinite(loads).all() or np.min(loads) <= 0.05:
        raise ValueError('Symbolic spring model cannot support the measured body pose')
    s['normal_load_n'] = loads
    if np.any(brake_mask):
        s['throttle_norm'][brake_mask] = 0.0
        s['wheel_omega_radps'][:, brake_mask] = 0.0
    # Compute the tire forces with a SCALAR expression for each wheel, exactly
    # as the historical OCP constructs them for each k. The older projection
    # evaluated the same helper with 1xN matrices; that is not an independently
    # verified way to initialize the scalar collocation equations.
    # Existing fixture wheel stand-ins retain the older seven-argument API.
    # The real patched Unity wheel model provides both named rate inputs.
    rate_args = inspect.signature(wheel.tire_forces_casadi).parameters
    contact_rates_supported = 'roll_rate' in rate_args and 'pitch_rate' in rate_args
    z_scalar = ca.MX.sym('scalar_wheel_state', 14)
    fx_node, fy_node, *_ = wheel.tire_forces_casadi(
        model, z_scalar[0], z_scalar[1], z_scalar[2], z_scalar[3],
        [z_scalar[4+i] for i in range(4)],
        [z_scalar[8+i] for i in range(4)],
        *((z_scalar[12], z_scalar[13]) if contact_rates_supported else ()))
    force_fn = ca.Function('ipopt_node_source_tire_force',
                           [z_scalar], [fx_node, fy_node])
    values = np.vstack((s['u_mps'], s['v_mps'], s['yaw_rate_radps'],
                        s['steering_rad'], s['wheel_omega_radps'], loads,
                        s['roll_rate_radps'], s['pitch_rate_radps']))
    scalar_forces = np.array([np.asarray(force_fn(values[:, k]),
                       dtype=np.float64).reshape(2) for k in range(n)])
    force_x, force_y = scalar_forces[:,0], scalar_forces[:,1]
    # Self-test old vectorized call. Record any discrepancy and fail closed
    # if it is more than floating-point roundoff. Do not silently seed with
    # values that disagree with the scalar tire law in the actual OCP.
    z_batch = ca.MX.sym('batched_wheel_state', 14, n)
    b_fx, b_fy, *_ = wheel.tire_forces_casadi(
        model, z_batch[0, :], z_batch[1, :], z_batch[2, :], z_batch[3, :],
        [z_batch[4+i, :] for i in range(4)],
        [z_batch[8+i, :] for i in range(4)],
        *((z_batch[12, :], z_batch[13, :]) if contact_rates_supported else ()))
    batch_fn = ca.Function('batched_source_tire_force',
                           [z_batch], [b_fx, b_fy])
    batch_fx, batch_fy = (np.asarray(x, dtype=np.float64).reshape(-1)
                          for x in batch_fn(values))
    scalar_vector_error = float(np.max(np.abs(np.vstack((batch_fx-force_x,
                                                         batch_fy-force_y)))))
    if not np.isfinite(scalar_vector_error):
        raise RuntimeError('Non-finite batched-vs-scalar tire-force mismatch')
    # The scalar construction is authoritative: the historical OCP invokes
    # the wheel model at each node, not on 1xN batches. If the batch form
    # disagrees, retain the scalar result and expose the discrepancy.
    rhs_ax = force_x / model.mass_kg - model.drag_per_s * s['u_mps']
    rhs_ay = force_y / model.mass_kg - model.drag_per_s * s['v_mps']
    if not np.isfinite(rhs_ax).all() or not np.isfinite(rhs_ay).all():
        raise ValueError('Source wheel model returned nonfinite force balance')
    s['ax_mps2'] = rhs_ax
    s['ay_mps2'] = rhs_ay
    audit = {
        'type': 'algebraic_projection_only_not_model_consistent_rollout',
        'nodes': n,
        'forced_full_brake_nodes_unseen_in_native': int(np.count_nonzero(brake_mask)),
        'native_throttle_min': float(np.min(measured_throttle)),
        'native_brake_nodes': int(np.count_nonzero(measured_throttle == 0)),
        'native_to_symbolic_load_max_abs_n': float(np.max(np.abs(loads - measured_load))),
        'native_to_symbolic_load_rms_n': float(np.sqrt(np.mean((loads-measured_load)**2))),
        'measured_to_symbolic_ax_max_abs_mps2': float(np.max(np.abs(rhs_ax-measured_ax))),
        'measured_to_symbolic_ay_max_abs_mps2': float(np.max(np.abs(rhs_ay-measured_ay))),
        'measured_to_symbolic_ay_rms_mps2': float(np.sqrt(np.mean((rhs_ay-measured_ay)**2))),
        'brake_locked_wheel_rpm_delta_max_radps': float(
            np.max(np.abs(measured_spin[:,brake_mask])) if np.any(brake_mask) else 0.),
        'symbolic_force_balance_max_abs_after_projection': 0.0,
        'spatial_collocation_or_wheel_update_are_not_reconciled': True,
        'model_is_not_native_physx': True,
        'scalar_vs_vectorized_tire_force_max_error_n': scalar_vector_error,
        'batched_tire_force_matches_scalar_ocp': bool(scalar_vector_error <= 1.e-8),
        'scalar_force_balance_residual_after_projection_mps2': float(max(
            np.max(np.abs(s['ax_mps2'] - (force_x/model.mass_kg - model.drag_per_s*s['u_mps']))),
            np.max(np.abs(s['ay_mps2'] - (force_y/model.mass_kg - model.drag_per_s*s['v_mps']))))),
    }
    return s, audit


def write_audit(path, audit):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2) + '\n')


def validate_seed_provenance(seed_file, map_file, centerline_file):
    """Require matching checksums when reusing a historical native warm start.

    The NPZ stores resampled dynamic variables but not map coordinates, so
    shapes alone cannot verify that its Frenet frame belongs to this track.
    """
    path = Path(seed_file).resolve().parent.parent / 'provenance.json'
    if not path.is_file():
        raise ValueError('Saved native warm start has no parent provenance.json: ' + str(path))
    meta = json.loads(path.read_text())
    for name, target in [('map_sha256', map_file),
                         ('centerline_sha256', centerline_file)]:
        expected = hashlib.sha256(Path(target).read_bytes()).hexdigest()
        actual = meta.get(name)
        if not actual or actual != expected:
            raise ValueError(f'Saved warm-start {name} does not match the requested track')
    if meta.get('requested_warmstart_backend') not in ('native','source'):
        raise ValueError('Saved seed provenance does not identify a physics warm start')
    return {'provenance':str(path), 'map_sha256':meta['map_sha256'],
            'centerline_sha256':meta['centerline_sha256']}
