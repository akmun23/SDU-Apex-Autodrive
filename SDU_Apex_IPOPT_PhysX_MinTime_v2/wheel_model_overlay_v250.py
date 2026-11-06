"""v2.5.0: use PhysX wheel-bottom contact kinematics in IPOPT tire slips.

PhysX computes wheelBottomPos from the current suspension ray/ground-plane
intersection, then uses v_COM + omega x (wheelBottomPos-COM) and tire axes
projected into the hit plane. The prior OCP used a fixed force-application
arm and a planar tire basis for slip. Force application remains at Unity's
separate configured tire-force application point; this patch changes only
contact velocity and projected tire directions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from wheel_model_overlay_v241 import (
    _WHEEL_REL,
    transform_wheel_model as transform_v241,
)


_SIGNATURE_V241 = '''def tire_forces_casadi(model: UnityWheelModel, u, v, yaw_rate, steering,
                       wheel_omega, suspension_forces, roll_rate=0.0,
                       pitch_rate=0.0):'''
_SIGNATURE_V250 = '''def tire_forces_casadi(model: UnityWheelModel, u, v, yaw_rate, steering,
                       wheel_omega, suspension_forces, body_height=None,
                       body_height_rate=0.0, roll=0.0, roll_rate=0.0,
                       pitch=0.0, pitch_rate=0.0):'''
_CONTACT_V241 = '''        # Native tire slip uses contact-point velocity = COM velocity +
        # angular_velocity x contact_arm, not the previous planar COM speed.
        # The tire-force application point is one radius below its wheel
        # centre; both mount Y and Rigidbody COM Y are Unity up-positive.
        # The stored suspension base includes the +25 mm collider ray
        # reference. PhysX's source tire-force application point instead uses
        # the loaded wheel centre at target travel; otherwise roll/pitch slip
        # velocities are computed with a contact arm 25 mm too short.
        contact_arm_down_m = (RIGIDBODY_CENTER_OF_MASS_BODY_M[1]
                              + model.wheel_radius_m
                              + model.suspension_target_travel_m[index]
                              - model.suspension_base_xyz_body_m[index][1])
        # Body axes are forward, LEFT, up. A downward contact arm yields
        # forward correction -pitch_rate*depth and lateral +roll_rate*depth.
        vx_body = (u - yaw_rate * WHEEL_Y_M[index]
                   - pitch_rate * contact_arm_down_m)
        vy_body = (v + yaw_rate * WHEEL_X_M[index]
                   + roll_rate * contact_arm_down_m)
        forward_speed = ca.cos(angle) * vx_body + ca.sin(angle) * vy_body
        lateral_speed = -ca.sin(angle) * vx_body + ca.cos(angle) * vy_body'''

_CONTACT_V250 = '''        forward_speed, lateral_speed = _physx_wheel_bottom_speeds_casadi(
            model, index, angle, u, v, yaw_rate, body_height,
            body_height_rate, roll, roll_rate, pitch, pitch_rate)'''

_CONTACT_HELPER = '''def _physx_wheel_bottom_speeds_casadi(
        model: UnityWheelModel, index, steer_angle, u, v, yaw_rate,
        body_height, body_height_rate, roll, roll_rate, pitch, pitch_rate):
    """PhysX ray-plane wheelBottomPos and projected tire-axis speeds.

    Mounts are stored as (Unity-left, Unity-up, Unity-forward) after the
    source frame conversion. Returned lateral speed is left-positive; the
    existing slip helper performs the verified Unity/PhysX sign conversion.
    """
    if body_height is None:
        body_height = model.suspension_reference_body_height_m
    base_left, base_up, base_forward = model.suspension_base_xyz_body_m[index]
    com_right, com_up, com_forward = RIGIDBODY_CENTER_OF_MASS_BODY_M
    com_forward_body = com_forward
    com_left_body = -com_right
    target_compression = model.suspension_target_travel_m[index]
    max_compression = target_compression
    # Unity WheelCollider::ComputeWheelCentreOfMassOffset places the native
    # wheel centre at max-compression offset; PhysX ray start is radius plus
    # max-compression above that centre. These are distinct from force app point.
    ray_start_forward = base_forward - com_forward_body
    ray_start_left = base_left - com_left_body
    ray_start_up = (base_up - target_compression
                    + model.wheel_radius_m + max_compression - com_up)

    sr, cr = ca.sin(roll), ca.cos(roll)
    sp, cp = ca.sin(pitch), ca.cos(pitch)
    normal_forward = -sp
    normal_left = cp * sr
    normal_up = cp * cr
    com_plane_offset = (normal_forward * com_forward_body
                        + normal_left * com_left_body + normal_up * com_up)
    ray_plane_offset = (normal_forward * ray_start_forward
                        + normal_left * ray_start_left
                        + normal_up * ray_start_up)
    suspension_direction_dot_normal = -normal_up
    ray_travel = -((body_height - model.suspension_ground_y_m)
                   + com_plane_offset + ray_plane_offset) \\
        / suspension_direction_dot_normal
    contact_forward = ray_start_forward
    contact_left = ray_start_left
    contact_up = ray_start_up - ray_travel

    # Unity's body_height_rate is world-vertical velocity. Convert it to
    # chassis vertical speed exactly as the production source plant does.
    body_vertical_speed = (body_height_rate + sp * u - cp * sr * v) \\
        / (cp * cr)
    # Body angular-rate components are roll, pitch and yaw rates, as in the
    # 18-state source-plant vector and PhysX carChassisAngVel.
    vx = u + pitch_rate * contact_up - yaw_rate * contact_left
    vy = v + yaw_rate * contact_forward - roll_rate * contact_up
    vz = (body_vertical_speed + roll_rate * contact_left
          - pitch_rate * contact_forward)

    long_norm = ca.sqrt(cr * cr + sp * sp * sr * sr)
    long0_f = cp * cr / long_norm
    long0_l = 0.0
    long0_u = sp / long_norm
    lat0_f = cp * sp * sr / long_norm
    lat0_l = (sp * sp * sr * sr + cr * cr) / long_norm
    lat0_u = -cp * cp * cr * sr / long_norm
    c, s = ca.cos(steer_angle), ca.sin(steer_angle)
    tire_long_f = long0_f * c + lat0_f * s
    tire_long_l = long0_l * c + lat0_l * s
    tire_long_u = long0_u * c + lat0_u * s
    tire_lat_f = lat0_f * c - long0_f * s
    tire_lat_l = lat0_l * c - long0_l * s
    tire_lat_u = lat0_u * c - long0_u * s
    forward_speed = (vx * tire_long_f + vy * tire_long_l
                     + vz * tire_long_u)
    lateral_speed_left = (vx * tire_lat_f + vy * tire_lat_l
                          + vz * tire_lat_u)
    return forward_speed, lateral_speed_left


'''


def transform_wheel_model(source: str) -> str:
    patched = transform_v241(source)
    # Unit fixtures intentionally omit the real tire model.
    if "_physx_applied_torque_slips_casadi" not in source:
        return patched
    # The symbolic function contains one source-level call inside its
    # four-wheel loop, not four textually unrolled calls.
    if patched.count(_SIGNATURE_V250) == 1 and patched.count(_CONTACT_V250) == 1:
        return patched
    if patched.count(_SIGNATURE_V241) != 1:
        raise RuntimeError("Unknown IPOPT tire function signature; refusing v250 patch")
    if patched.count(_CONTACT_V241) != 1:
        raise RuntimeError("Unknown IPOPT wheel-contact equations; refusing v250 patch")
    patched = patched.replace(_SIGNATURE_V241, _SIGNATURE_V250, 1)
    patched = patched.replace(_CONTACT_V241, _CONTACT_V250, 1)
    marker = "def tire_forces_casadi(model: UnityWheelModel,"
    if patched.count(marker) != 1:
        raise RuntimeError("Unknown IPOPT tire function insertion point")
    patched = patched.replace(marker, _CONTACT_HELPER + marker, 1)
    return patched


def prepare_working_wheel_model(historical: Path, working: Path,
                                legacy_name: str) -> dict:
    historical, working = Path(historical).resolve(), Path(working).resolve()
    if historical == working:
        raise ValueError("Refusing to patch historical wheel source in place")
    source_path = historical / legacy_name / _WHEEL_REL
    target_path = working / legacy_name / _WHEEL_REL
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Historical or generated wheel model is missing")
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    patched = transform_wheel_model(source)
    current = target_path.read_text(encoding="utf-8")
    known = {source, transform_v241(source), patched}
    if current not in known:
        raise RuntimeError("Unknown generated wheel-model cache; refusing overwrite")
    if current != patched:
        target_path.write_text(patched, encoding="utf-8")
    if source_path.read_bytes() != source_bytes:
        raise RuntimeError("Historical wheel model was modified")
    return {
        "native_wheel_bottom_contact_kinematics": True,
        "native_wheel_contact_velocity_formula":
            "wheelBottomVel=vCOM+omega cross (wheelBottomPos-COM); ray-plane wheelBottomPos; hit-plane projected tire axes",
        "native_contact_force_application_point_kept_separate": True,
        "contact_height_roll_pitch_states_used": True,
        "native_flat_ground_contact_model": True,
        "original_wheel_model_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "working_wheel_model_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "working_wheel_model_path": str(target_path),
        "historical_wheel_model_untouched": source_path.read_bytes() == source_bytes,
    }
