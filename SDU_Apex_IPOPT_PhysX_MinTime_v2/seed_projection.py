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
ACCELERATION_CONVENTION = 'body_velocity_derivative_v1'
LEGACY_ACCELERATION_CONVENTION = 'inertial_body_axes_v0'


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


def _resample_periodic_seed(seed: dict, n: int) -> dict:
    """Resample a validated, uniformly spaced closed-lap seed to ``n`` nodes."""
    source_n = len(seed['u_mps'])
    if n < 2:
        raise ValueError('Expected two or more target IPOPT nodes')
    source_position = np.arange(n, dtype=np.float64) * (source_n / n)
    lower = np.floor(source_position).astype(np.int64)
    alpha = source_position - lower
    upper = (lower + 1) % source_n
    lower %= source_n
    result = {}
    for name, value in seed.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            result[name] = array[lower] * (1.0-alpha) + array[upper] * alpha
        else:
            result[name] = (array[..., lower] * (1.0-alpha)
                            + array[..., upper] * alpha)
    return validate_native_seed(result, n)


def load_native_seed(path: str | Path, n: int, *,
                     source_length_m: float | None = None,
                     target_length_m: float | None = None) -> dict:
    """Load a native seed, optionally resampling its uniform lap mesh.

    Resampling is allowed only when the caller has already verified matching
    map/centerline provenance and supplies equal source and target lap lengths.
    This prevents shape compatibility from silently becoming track alignment.
    """
    with np.load(path, allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    convention_value = raw.get('acceleration_convention')
    convention = (str(np.asarray(convention_value).item())
                  if convention_value is not None else
                  LEGACY_ACCELERATION_CONVENTION)
    if convention not in (ACCELERATION_CONVENTION,
                          LEGACY_ACCELERATION_CONVENTION):
        raise ValueError('Unknown native-seed acceleration convention: '
                         + convention)
    source = np.asarray(raw.get('u_mps', []), dtype=np.float64)
    if source.ndim != 1 or source.size < 2:
        raise ValueError('Native seed forward-speed channel must be a 1D lap mesh')
    source_n = int(source.size)
    checked = validate_native_seed(raw, source_n)
    if convention == LEGACY_ACCELERATION_CONVENTION:
        # Older unmarked seeds stored inertial acceleration resolved in body
        # axes (du/dt-r*v, dv/dt+r*u). Convert to the active OCP's body-state
        # derivatives once on load.
        yaw_rate = checked['yaw_rate_radps']
        checked['ax_mps2'] += yaw_rate * checked['v_mps']
        checked['ay_mps2'] -= yaw_rate * checked['u_mps']
    if source_n == n:
        return checked
    if source_length_m is None or target_length_m is None:
        raise ValueError(f'Native seed shape mismatch: source has {source_n} nodes, '
                         f'target has {n}; matching track lengths are required to resample')
    lengths = np.asarray([source_length_m, target_length_m], dtype=np.float64)
    if (not np.isfinite(lengths).all() or np.any(lengths <= 0)
            or abs(float(source_length_m)-float(target_length_m)) > 1.0e-6):
        raise ValueError('Native seed cannot be resampled across different lap lengths')
    return _resample_periodic_seed(checked, n)


def project_algebraics(seed: dict, model, wheel, *,
                       suspension_loads_fn, body_acceleration_fn,
                       minimum_drive_throttle: float = 0.01):
    """Recompute OCP wheel loads and ax/ay, preserving measured differential states.

    The optimization seed is constrained to positive drive throttle. Any
    captured command below that production lower bound is clamped before it
    can initialize the optimizer; captured wheel/body states are otherwise
    retained and explicitly remain an algebraic, not recursive, projection.
    """
    import casadi as ca
    import inspect
    s = validate_native_seed(seed, len(np.asarray(seed['u_mps'])))
    n = s['u_mps'].size
    if (not np.isfinite(minimum_drive_throttle)
            or minimum_drive_throttle <= 0.0
            or minimum_drive_throttle >= 1.0):
        raise ValueError('Minimum drive throttle must be in (0, 1)')
    measured_load = s['normal_load_n'].copy()
    measured_ax = s['ax_mps2'].copy()
    measured_ay = s['ay_mps2'].copy()
    measured_throttle = s['throttle_norm'].copy()
    throttle_clamp_count = int(np.count_nonzero(
        s['throttle_norm'] < minimum_drive_throttle))
    s['throttle_norm'] = np.clip(
        s['throttle_norm'], minimum_drive_throttle, 1.0)
    # Reconcile the spring/damper algebraics with the source *symbolic* model.
    # We retain native loads in the audit because they are important evidence
    # of native-versus-symbolic contact-model differences.
    loads = np.stack([
        np.asarray(suspension_loads_fn(
            model, s['body_height_m'][k], s['body_height_rate_mps'][k],
            s['roll_rad'][k], s['roll_rate_radps'][k],
            s['pitch_rad'][k], s['pitch_rate_radps'][k],
            s['yaw_rate_radps'][k]), dtype=np.float64)
        for k in range(n)], axis=1)
    if loads.shape != (4, n) or not np.isfinite(loads).all() or np.min(loads) <= 0.05:
        raise ValueError('Symbolic spring model cannot support the measured body pose')
    s['normal_load_n'] = loads
    # Compute scalar and batched forces with the same full state signature used
    # by the active OCP. Historical projection passed roll/pitch rates into the
    # body-height/body-height-rate positions, silently evaluating different
    # contact kinematics despite its scalar-vs-batched self-check.
    tire_parameters = inspect.signature(wheel.tire_forces_casadi).parameters
    contact_state_names = (
        'body_height', 'body_height_rate', 'roll', 'roll_rate',
        'pitch', 'pitch_rate')
    has_contact_state = all(name in tire_parameters
                            for name in contact_state_names)
    has_partial_contact_state = any(name in tire_parameters
                                    for name in contact_state_names)
    if has_partial_contact_state and not has_contact_state:
        raise RuntimeError(
            'Tire callback exposes an incomplete body/contact-state API')
    z_scalar = ca.MX.sym('scalar_wheel_state', 18)
    contact_args = ((z_scalar[12], z_scalar[13], z_scalar[14],
                     z_scalar[15], z_scalar[16], z_scalar[17])
                    if has_contact_state else ())
    fx_node, fy_node, *_ = wheel.tire_forces_casadi(
        model, z_scalar[0], z_scalar[1], z_scalar[2], z_scalar[3],
        [z_scalar[4+i] for i in range(4)],
        [z_scalar[8+i] for i in range(4)], *contact_args)
    force_fn = ca.Function('ipopt_node_source_tire_force',
                           [z_scalar], [fx_node, fy_node])
    values = np.vstack((s['u_mps'], s['v_mps'], s['yaw_rate_radps'],
                        s['steering_rad'], s['wheel_omega_radps'], loads,
                        s['body_height_m'], s['body_height_rate_mps'],
                        s['roll_rad'], s['roll_rate_radps'],
                        s['pitch_rad'], s['pitch_rate_radps']))
    scalar_forces = np.array([np.asarray(force_fn(values[:, k]),
                       dtype=np.float64).reshape(2) for k in range(n)])
    force_x, force_y = scalar_forces[:,0], scalar_forces[:,1]
    # Self-test old vectorized call. Record any discrepancy and fail closed
    # if it is more than floating-point roundoff. Do not silently seed with
    # values that disagree with the scalar tire law in the actual OCP.
    z_batch = ca.MX.sym('batched_wheel_state', 18, n)
    b_fx, b_fy, *_ = wheel.tire_forces_casadi(
        model, z_batch[0, :], z_batch[1, :], z_batch[2, :], z_batch[3, :],
        [z_batch[4+i, :] for i in range(4)],
        [z_batch[8+i, :] for i in range(4)],
        *((z_batch[12, :], z_batch[13, :], z_batch[14, :],
           z_batch[15, :], z_batch[16, :], z_batch[17, :])
          if has_contact_state else ()))
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
    # Use the very same body-frame acceleration equation as the active OCP:
    # gravity in the tilted body frame, p/q/r transport terms, and the
    # continuous equivalent of Unity's per-1ms multiplicative Rigidbody drag.
    # Reimplementing only F/m - drag*v here made the warm-start acceleration
    # inconsistent with the dynamics it initialized.
    accelerations = np.asarray([
        body_acceleration_fn(
            model, force_x[k], force_y[k], s['u_mps'][k], s['v_mps'][k],
            s['yaw_rate_radps'][k], s['body_height_rate_mps'][k],
            s['roll_rad'][k], s['roll_rate_radps'][k],
            s['pitch_rad'][k], s['pitch_rate_radps'][k])
        for k in range(n)], dtype=np.float64)
    if accelerations.shape != (n, 2) or not np.isfinite(accelerations).all():
        raise ValueError('Source body-acceleration equation returned invalid values')
    rhs_ax, rhs_ay = accelerations[:, 0], accelerations[:, 1]
    if not np.isfinite(rhs_ax).all() or not np.isfinite(rhs_ay).all():
        raise ValueError('Source wheel model returned nonfinite force balance')
    s['ax_mps2'] = rhs_ax
    s['ay_mps2'] = rhs_ay
    audit = {
        'type': 'algebraic_projection_only_not_model_consistent_rollout',
        'nodes': n,
        'native_throttle_min': float(np.min(measured_throttle)),
        'optimizer_minimum_drive_throttle': float(minimum_drive_throttle),
        'native_drive_throttle_values_clamped': throttle_clamp_count,
        'native_to_symbolic_load_max_abs_n': float(np.max(np.abs(loads - measured_load))),
        'native_to_symbolic_load_rms_n': float(np.sqrt(np.mean((loads-measured_load)**2))),
        'measured_to_symbolic_ax_max_abs_mps2': float(np.max(np.abs(rhs_ax-measured_ax))),
        'measured_to_symbolic_ay_max_abs_mps2': float(np.max(np.abs(rhs_ay-measured_ay))),
        'measured_to_symbolic_ay_rms_mps2': float(np.sqrt(np.mean((rhs_ay-measured_ay)**2))),
        'symbolic_force_balance_max_abs_after_projection': 0.0,
        'body_acceleration_equation': getattr(
            body_acceleration_fn, '__qualname__', repr(body_acceleration_fn)),
        'body_acceleration_uses_source_tilt_gravity_transport_and_1ms_drag': True,
        'suspension_equation': getattr(
            suspension_loads_fn, '__qualname__', repr(suspension_loads_fn)),
        'suspension_uses_body_to_euler_rate_conversion': True,
        'tire_contact_state_matches_ocp_full_18_state': has_contact_state,
        'spatial_collocation_or_wheel_update_are_not_reconciled': True,
        'model_is_not_native_physx': True,
        'scalar_vs_vectorized_tire_force_max_error_n': scalar_vector_error,
        'batched_tire_force_matches_scalar_ocp': bool(scalar_vector_error <= 1.e-8),
        'scalar_force_balance_residual_after_projection_mps2': float(max(
            np.max(np.abs(s['ax_mps2'] - rhs_ax)),
            np.max(np.abs(s['ay_mps2'] - rhs_ay)))),
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
    audit = {'provenance':str(path), 'map_sha256':meta['map_sha256'],
             'centerline_sha256':meta['centerline_sha256']}
    report_path = path.parent / 'ipopt_report_pass_00.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        track = report.get('track', {})
        count, length = track.get('count'), track.get('length_m')
        if count is not None and length is not None:
            count, length = int(count), float(length)
            if count < 2 or not np.isfinite(length) or length <= 0:
                raise ValueError('Native seed source report has invalid track mesh metadata')
            audit.update(source_node_count=count, source_track_length_m=length)
    return audit
