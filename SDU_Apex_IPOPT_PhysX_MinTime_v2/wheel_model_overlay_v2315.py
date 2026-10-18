"""v2.3.15: PhysX right-positive lateral tire FORCE -> model left-positive.

PhysX tire callback Fy is in its wheel RIGHT axis. The symbolic IPOPT
vehicle state and wheel Ackermann angles use forward/LEFT. The v2.3.14
slip correction converted lateral velocity to PhysX's tire-slip axis, but
forgot the inverse transformation on callback tire force. Convert the raw
callback force in the chassis rotation only; keep returned per-wheel Fy in
native units for wheel-contact diagnostics. Tire coefficients are untouched.

The original archived wheel model is not edited. Cached working wheel files
must match known exact upgrades; unknown changes are rejected.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from wheel_model_overlay_v2312 import _WHEEL_REL, transform_wheel_model as v2312
from wheel_model_overlay_v2314 import transform_wheel_model as v2314

_BODY_FORCES_OLD = '''        fx_body = fx * ca.cos(angle) - fy * ca.sin(angle)
        fy_body = fx * ca.sin(angle) + fy * ca.cos(angle)'''
_BODY_FORCES_NEW = '''        # The Unity/PhysX callback returns lateral force in wheel RIGHT,
        # but IPOPT chassis axes are forward/LEFT/up. Ackermann steer angle
        # is LEFT-positive. Right unit vector=(sin(angle), -cos(angle)).
        # Preserve callback raw fy in fy_wheels for contact parity checks.
        fx_body = fx * ca.cos(angle) + fy * ca.sin(angle)
        fy_body = fx * ca.sin(angle) - fy * ca.cos(angle)'''


def transform_wheel_model(historical_source: str) -> str:
    patched = v2314(historical_source)
    if patched.count(_BODY_FORCES_OLD) == 1:
        return patched.replace(_BODY_FORCES_OLD, _BODY_FORCES_NEW, 1)
    # Legacy isolated fixture has no genuine vehicle wheel equations.
    if ('def tire_forces_casadi(model, suspension_forces, index):' in historical_source
            and '_physx_applied_torque_slips_casadi' not in historical_source):
        return patched
    raise RuntimeError('Historical wheel source mismatch at PhysX chassis lateral-force frame')


def prepare_working_wheel_model(historical: Path, working: Path,
                                legacy_name: str) -> dict:
    historical, working = Path(historical).resolve(), Path(working).resolve()
    if historical == working:
        raise ValueError('Refusing to edit original historical wheel model')
    source_file = historical / legacy_name / _WHEEL_REL
    if not source_file.is_file():
        return {'working_wheel_model_parity_applied': False,
                'wheel_model_fixture_not_present': True}
    old_bytes = source_file.read_bytes()
    old = old_bytes.decode('utf8')
    corrected = transform_wheel_model(old)
    path = working / legacy_name / _WHEEL_REL
    if not path.is_file():
        raise FileNotFoundError('Recovered working wheel model missing: ' + str(path))
    current = path.read_text(encoding='utf8')
    known = (old, v2312(old), v2314(old), corrected)
    if current not in known:
        raise RuntimeError('Working wheel model has unknown local edits; refusing to overwrite ' + str(path))
    if current != corrected:
        path.write_text(corrected, encoding='utf8')
    if source_file.read_bytes() != old_bytes:
        raise RuntimeError('Historical wheel source was modified')
    real = ('_physx_applied_torque_slips_casadi' in old)
    return {
        'working_wheel_model_parity_applied': True,
        'original_wheel_model_sha256': hashlib.sha256(old_bytes).hexdigest(),
        'working_wheel_model_sha256': hashlib.sha256(corrected.encode()).hexdigest(),
        'working_wheel_model_path': str(path),
        'historical_wheel_model_untouched': source_file.read_bytes() == old_bytes,
        'native_callback_load_model': 'tireLoad=PxVehicle.suspSpringForce',
        'native_combined_slip_model': 'normalize raw curve response before stiffness',
        'native_lateral_slip_sign_corrected': real,
        'native_lateral_slip_mapping': 'left-positive wheel velocity -> right-positive PhysX slip',
        'native_lateral_force_frame_corrected': real,
        'native_lateral_force_frame_mapping': 'callback Fy right-positive -> chassis Fy left-positive; preserve raw callback Fy',
    }
