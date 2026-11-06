"""v2.4.1: correct the 25 mm tire-force contact-arm offset.

The v2.3.16 / v2.4.0 generated model confused the WheelCollider ray
reference with the *loaded* wheel centre. The source rigid-body COM to
contact-force offset includes the wheel suspension's target travel. Keep the
historical wheel source pristine and accept only byte-exact known caches.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from wheel_model_overlay_v2312 import _WHEEL_REL, transform_wheel_model as v2312
from wheel_model_overlay_v2314 import transform_wheel_model as v2314
from wheel_model_overlay_v2315 import transform_wheel_model as v2315
from wheel_model_overlay_v2316 import transform_wheel_model as v2316

_CONTACT_ARM_OLD = """        contact_arm_down_m = (RIGIDBODY_CENTER_OF_MASS_BODY_M[1]
                              + model.wheel_radius_m
                              - model.suspension_base_xyz_body_m[index][1])"""
_CONTACT_ARM_NEW = """        # The stored suspension base includes the +25 mm collider ray
        # reference. PhysX's source tire-force application point instead uses
        # the loaded wheel centre at target travel; otherwise roll/pitch slip
        # velocities are computed with a contact arm 25 mm too short.
        contact_arm_down_m = (RIGIDBODY_CENTER_OF_MASS_BODY_M[1]
                              + model.wheel_radius_m
                              + model.suspension_target_travel_m[index]
                              - model.suspension_base_xyz_body_m[index][1])"""


def transform_wheel_model(historical_source: str) -> str:
    patched = v2316(historical_source)
    if '_physx_applied_torque_slips_casadi' not in historical_source:
        return patched  # historic miniature unit-test fixture
    if patched.count(_CONTACT_ARM_OLD) != 1:
        raise RuntimeError('Unknown tire contact-arm site; refusing unsafe edit')
    return patched.replace(_CONTACT_ARM_OLD, _CONTACT_ARM_NEW, 1)


def prepare_working_wheel_model(historical: Path, working: Path,
                                legacy_name: str) -> dict:
    historical, working = Path(historical).resolve(), Path(working).resolve()
    if historical == working:
        raise ValueError('Cannot patch historical wheel source in place')
    src = historical / legacy_name / _WHEEL_REL
    if not src.is_file():
        return {'working_wheel_model_parity_applied': False,
                'wheel_model_fixture_not_present': True}
    src_bytes = src.read_bytes()
    original = src_bytes.decode('utf-8')
    patched = transform_wheel_model(original)
    dest = working / legacy_name / _WHEEL_REL
    if not dest.is_file():
        raise FileNotFoundError('Cached working wheel model missing: ' + str(dest))
    current = dest.read_text(encoding='utf-8')
    known = (original, v2312(original), v2314(original), v2315(original),
             v2316(original), patched)
    if current not in known:
        raise RuntimeError('Working wheel model has unknown local edits; refusing overwrite: '
                           + str(dest))
    if current != patched:
        dest.write_text(patched, encoding='utf-8')
    if src.read_bytes() != src_bytes:
        raise RuntimeError('Historical wheel source modified')
    real = '_physx_applied_torque_slips_casadi' in original
    return {
        'working_wheel_model_parity_applied': True,
        'original_wheel_model_sha256': hashlib.sha256(src_bytes).hexdigest(),
        'working_wheel_model_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'working_wheel_model_path': str(dest),
        'historical_wheel_model_untouched': True,
        'native_callback_load_model': 'tireLoad=PxVehicle.suspSpringForce',
        'native_combined_slip_model': 'normalize raw curve response before stiffness',
        'native_lateral_slip_sign_corrected': real,
        'native_lateral_slip_mapping': 'model left-positive -> PhysX right-positive',
        'native_lateral_force_frame_corrected': real,
        'native_lateral_force_frame_mapping': 'native right-positive Fy -> chassis left-positive Fy',
        'native_full_contact_velocity_corrected': real,
        'native_full_contact_velocity_model': 'COM velocity plus yaw/roll/pitch cross native force application offset',
        'native_contact_vertical_arm_m': 'Rigidbody COM y plus wheel radius plus suspension target travel minus model suspension ray base y',
        'source_contact_arm_25mm_offset_corrected': real,
        'source_contact_arm_correction_m': 0.025 if real else None,
    }
