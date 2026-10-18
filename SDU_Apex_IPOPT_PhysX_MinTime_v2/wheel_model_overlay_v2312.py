"""Fail-closed working-copy restoration of PhysX 4.1/Unity native tire callback.

Historical source remains byte-identical. On the 481-sample 2.4 m/s native
contact replay, callback forces were reproduced to <1e-6 N/component using
RAW suspension forces and the native C++ curve order; the historic CasADi model
introduced a redundant load filter and normalized already-stiffened curves.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

_WHEEL_REL = 'f1tenth_planning/autodrive_mintime/unity_wheel_model.py'

_SHADER_OLD = '''    utilization_sq = (
        (long_response / forward_curve[1])**2
        + (lateral_response / lateral_curve[1])**2)
    combined_slip_scale = 1.0 / ca.sqrt(ca.fmax(1.0, utilization_sq))
    force_long = tire_load * long_response * combined_slip_scale
    force_lateral = -tire_load * lateral_response * combined_slip_scale'''

_SHADER_NEW = '''    # Native unityCurveTireForce: normalize un-stiffened curve responses,
    # THEN apply each WheelFrictionCurve.stiffness to the final force. The
    # helper above already multiplies stiffness, so divide it out here.
    raw_long_response = long_response / forward_curve[4]
    raw_lateral_response = lateral_response / lateral_curve[4]
    utilization_sq = (
        (raw_long_response / forward_curve[1])**2
        + (raw_lateral_response / lateral_curve[1])**2)
    combined_slip_scale = 1.0 / ca.sqrt(ca.fmax(1.0, utilization_sq))
    force_long = tire_load * raw_long_response * combined_slip_scale * forward_curve[4]
    force_lateral = -tire_load * raw_lateral_response * combined_slip_scale * lateral_curve[4]'''

_LOAD_OLD = '''        tire_load = _physx_filtered_tire_load_casadi(
            suspension_forces[index], model.sprung_mass_kg[index],
            WHEEL_MASS_KG, UNITY_PHYSICS_GRAVITY_MPS2)'''

_LOAD_NEW = '''        # Native callback tireLoad equals PxVehicleWheelQueryResult.suspSpringForce
        # in the source-matched 1 ms replay (all 1,924 measured wheel samples).
        # The older symbolic filter inflated that load by ~1.49 N. Do not
        # apply it again: native callback already receives the effective load.
        tire_load = suspension_forces[index]'''


def transform_wheel_model(historical_source: str) -> str:
    """Patch exactly two anchored expression sites; reject unrecognized code."""
    result = historical_source
    for original, replacement, label in (
        (_SHADER_OLD, _SHADER_NEW, 'combined-slip stiffness'),
        (_LOAD_OLD, _LOAD_NEW, 'native callback tireLoad'),
    ):
        if result.count(original) != 1:
            raise RuntimeError('Historical wheel source mismatch at ' + label)
        result = result.replace(original, replacement, 1)
    return result


def prepare_working_wheel_model(historical: Path, working: Path,
                                legacy_name: str) -> dict:
    """Only upgrade exact original/known generated files; never edit history."""
    historical = Path(historical).resolve()
    working = Path(working).resolve()
    if historical == working:
        raise ValueError('Refusing to edit original historical wheel model')
    source_file = historical / legacy_name / _WHEEL_REL
    # Legacy isolated toy regression fixtures mock optimizer_unity_wheels.py
    # alone. Actual Git restorations always have the full original wheel file.
    if not source_file.exists():
        return {'working_wheel_model_parity_applied': False,
                'wheel_model_fixture_not_present': True}
    source_bytes = source_file.read_bytes()
    old = source_bytes.decode('utf8')
    new = transform_wheel_model(old)
    out = working / legacy_name / _WHEEL_REL
    if not out.is_file():
        raise FileNotFoundError('Recovered working wheel model missing: ' + str(out))
    current = out.read_text(encoding='utf8')
    if current not in (old, new):
        raise RuntimeError('Working wheel model has unknown local edits; refusing to overwrite ' + str(out))
    if current != new:
        out.write_text(new, encoding='utf8')
    if source_file.read_bytes() != source_bytes:
        raise RuntimeError('Historical wheel source was modified')
    return {
        'working_wheel_model_parity_applied': True,
        'original_wheel_model_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'working_wheel_model_sha256': hashlib.sha256(new.encode()).hexdigest(),
        'working_wheel_model_path': str(out),
        'historical_wheel_model_untouched': source_file.read_bytes() == source_bytes,
        'native_callback_load_model': 'tireLoad=PxVehicle.suspSpringForce (no second symbolic load filter)',
        'native_combined_slip_model': 'normalize bare curve response before curve stiffness',
    }
