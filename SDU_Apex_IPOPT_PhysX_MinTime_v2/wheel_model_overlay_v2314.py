"""v2.3.14: source-verified lateral slip coordinate sign in GENERATED CasADi model.

The native PhysX lateral slip is positive toward Unity right; the optimizer
uses forward/LEFT/up. Converting left-positive wheel-axis lateral velocity to
PhysX's right-positive slip sign is mandatory before the Unity tire callback.

All upgrades are exact-string guarded and historical source is never modified.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from wheel_model_overlay_v2312 import (
    _WHEEL_REL, transform_wheel_model as transform_wheel_model_v2312,
)

_SLIP_OLD = '''    lateral_slip = ca.atan(
        lateral_speed
        / (long_speed_abs + PHYSX_LATERAL_SLIP_SPEED_OFFSET_MPS
           * PHYSX_TOLERANCE_SCALE_LENGTH_M))'''

_SLIP_NEW = '''    # Optimizer wheel-axis lateral_speed is LEFT-positive. PhysX/Unity
    # reports applied-torque lateral slip in the RIGHT-positive native basis.
    # A native 2.4 m/s 481-sample flight confirms the sign inversion on both
    # steered front wheels; no tire friction coefficients are changed.
    lateral_slip = -ca.atan(
        lateral_speed
        / (long_speed_abs + PHYSX_LATERAL_SLIP_SPEED_OFFSET_MPS
           * PHYSX_TOLERANCE_SCALE_LENGTH_M))'''

_TOY_SIGNATURE = 'def tire_forces_casadi(model, suspension_forces, index):'


def transform_wheel_model(historical_source: str) -> str:
    """Patch callback, load and slip sites; fail closed if slip site differs."""
    output = transform_wheel_model_v2312(historical_source)
    if output.count(_SLIP_OLD) == 1:
        return output.replace(_SLIP_OLD, _SLIP_NEW, 1)
    # Older isolated unit-test fixture has no slip function at all; the real
    # restored source must include it, and must never take this escape hatch.
    if (_TOY_SIGNATURE in historical_source
        and '_physx_applied_torque_slips_casadi' not in historical_source):
        return output
    raise RuntimeError('Historical wheel source mismatch at PhysX lateral slip coordinate sign')


def prepare_working_wheel_model(historical: Path, working: Path,
                                legacy_name: str) -> dict:
    historical = Path(historical).resolve()
    working = Path(working).resolve()
    if historical == working:
        raise ValueError('Refusing to edit original historical wheel model')
    source_file = historical / legacy_name / _WHEEL_REL
    if not source_file.is_file():
        return {'working_wheel_model_parity_applied': False,
                'wheel_model_fixture_not_present': True}
    historical_bytes = source_file.read_bytes()
    old = historical_bytes.decode('utf8')
    previous = transform_wheel_model_v2312(old)
    corrected = transform_wheel_model(old)
    target = working / legacy_name / _WHEEL_REL
    if not target.is_file():
        raise FileNotFoundError('Recovered working wheel model missing: ' + str(target))
    current = target.read_text(encoding='utf8')
    if current not in (old, previous, corrected):
        raise RuntimeError('Working wheel model has unknown local edits; refusing to overwrite ' + str(target))
    if current != corrected:
        target.write_text(corrected, encoding='utf8')
    if source_file.read_bytes() != historical_bytes:
        raise RuntimeError('Historical wheel source was modified')
    is_real = '_physx_applied_torque_slips_casadi' in old
    return {
        'working_wheel_model_parity_applied': True,
        'original_wheel_model_sha256': hashlib.sha256(historical_bytes).hexdigest(),
        'working_wheel_model_sha256': hashlib.sha256(corrected.encode()).hexdigest(),
        'working_wheel_model_path': str(target),
        'historical_wheel_model_untouched': source_file.read_bytes() == historical_bytes,
        'native_callback_load_model': 'tireLoad=PxVehicle.suspSpringForce (no second symbolic load filter)',
        'native_combined_slip_model': 'normalize bare curve response before curve stiffness',
        'native_lateral_slip_sign_corrected': is_real,
        'native_lateral_slip_mapping': 'model left-positive lateral_speed -> PhysX right-positive lateralSlip',
    }
