"""v2.3.16: PhysX full tire-contact velocity including roll/pitch angular rates.

Use COM linear velocity + angular velocity cross source tire-force application
point. Fixed wheel force application offset is wheel centre minus radius along
Unity Y/up: front 0.063435m down, rear 0.073340m down (source parameters).
Prior tire callback, force-frame and slip sign remain source-consistent.
Historical source untouched, and unknown cache edits always rejected.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from wheel_model_overlay_v2312 import _WHEEL_REL, transform_wheel_model as v2312
from wheel_model_overlay_v2314 import transform_wheel_model as v2314
from wheel_model_overlay_v2315 import transform_wheel_model as v2315

_SIGNATURE_OLD = '''def tire_forces_casadi(model: UnityWheelModel, u, v, yaw_rate, steering,
                       wheel_omega, suspension_forces):'''
_SIGNATURE_NEW = '''def tire_forces_casadi(model: UnityWheelModel, u, v, yaw_rate, steering,
                       wheel_omega, suspension_forces, roll_rate=0.0,
                       pitch_rate=0.0):'''
_VELOCITY_OLD = '''        vx_body = u - yaw_rate * WHEEL_Y_M[index]
        vy_body = v + yaw_rate * WHEEL_X_M[index]'''
_VELOCITY_NEW = '''        # Native tire slip uses contact-point velocity = COM velocity +
        # angular_velocity x contact_arm, not the previous planar COM speed.
        # The tire-force application point is one radius below its wheel
        # centre; both mount Y and Rigidbody COM Y are Unity up-positive.
        contact_arm_down_m = (RIGIDBODY_CENTER_OF_MASS_BODY_M[1]
                              + model.wheel_radius_m
                              - model.suspension_base_xyz_body_m[index][1])
        # Body axes are forward, LEFT, up. A downward contact arm yields
        # forward correction -pitch_rate*depth and lateral +roll_rate*depth.
        vx_body = (u - yaw_rate * WHEEL_Y_M[index]
                   - pitch_rate * contact_arm_down_m)
        vy_body = (v + yaw_rate * WHEEL_X_M[index]
                   + roll_rate * contact_arm_down_m)'''
_TOY_SIGNATURE='def tire_forces_casadi(model, suspension_forces, index):'


def transform_wheel_model(historical_source: str) -> str:
    patched=v2315(historical_source)
    if (_TOY_SIGNATURE in historical_source and
            '_physx_applied_torque_slips_casadi' not in historical_source):
        return patched
    for old,new,label in ((_SIGNATURE_OLD,_SIGNATURE_NEW,'tire function signature'),
                          (_VELOCITY_OLD,_VELOCITY_NEW,'tire contact velocity')):
        if patched.count(old)!=1:
            raise RuntimeError(f'Historical wheel source mismatch at {label}: count={patched.count(old)}')
        patched=patched.replace(old,new,1)
    return patched


def prepare_working_wheel_model(historical: Path, working: Path,
                                legacy_name: str) -> dict:
    historical,working=Path(historical).resolve(),Path(working).resolve()
    if historical==working:raise ValueError('Historical and working tree must differ')
    src=historical/legacy_name/_WHEEL_REL
    if not src.is_file():
        return {'working_wheel_model_parity_applied':False,
                'wheel_model_fixture_not_present':True}
    src_bytes=src.read_bytes();original=src_bytes.decode('utf-8')
    patched=transform_wheel_model(original)
    dest=working/legacy_name/_WHEEL_REL
    if not dest.is_file():
        raise FileNotFoundError('Cached working wheel model missing: '+str(dest))
    current=dest.read_text(encoding='utf-8')
    if current not in (original,v2312(original),v2314(original),
                       v2315(original),patched):
        raise RuntimeError('Working wheel model has unknown local edits; refusing overwrite: '+str(dest))
    if current!=patched:dest.write_text(patched,encoding='utf-8')
    if src.read_bytes()!=src_bytes:
        raise RuntimeError('Historical wheel source modified')
    real='_physx_applied_torque_slips_casadi' in original
    return {'working_wheel_model_parity_applied':True,
            'original_wheel_model_sha256':hashlib.sha256(src_bytes).hexdigest(),
            'working_wheel_model_sha256':hashlib.sha256(patched.encode()).hexdigest(),
            'working_wheel_model_path':str(dest),
            'historical_wheel_model_untouched':True,
            'native_callback_load_model':'tireLoad=PxVehicle.suspSpringForce',
            'native_combined_slip_model':'normalize raw curve response before stiffness',
            'native_lateral_slip_sign_corrected':real,
            'native_lateral_slip_mapping':'left-positive wheel velocity -> right-positive PhysX slip',
            'native_lateral_force_frame_corrected':real,
            'native_lateral_force_frame_mapping':'callback Fy right-positive -> chassis Fy left-positive; preserve raw callback Fy',
            'native_full_contact_velocity_corrected':real,
            'native_full_contact_velocity_model':'COM velocity + angular rate cross source tire force-application point; yaw/roll/pitch',
            'native_contact_vertical_arm_m':'Rigidbody COM y minus Unity wheel mount y plus source wheel radius'}
