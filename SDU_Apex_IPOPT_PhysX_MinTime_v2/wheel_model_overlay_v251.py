"""v2.5.1: source-aligned 3-D WheelCollider wrench for the IPOPT OCP.

This is a code translation of the active production plant's flat-ground
WheelCollider ray, contact velocity, tire basis, filtered tire load, and
force/moment application. It does not reproduce PhysX's native contact
solver or its discrete outer-step integration.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

from wheel_model_overlay_v250 import (
    _WHEEL_REL,
    _SIGNATURE_V250,
    _CONTACT_V250,
    transform_wheel_model as transform_v250,
)
from wheel_model_overlay_v241 import transform_wheel_model as transform_v241


_FRAME_HELPER = '''def _physx_wheel_contact_frame_casadi(
        model: UnityWheelModel, index, steer_angle, u, v, yaw_rate,
        body_height, body_height_rate, roll, roll_rate, pitch, pitch_rate):
    """Translate wheelBottomPos, v_COM+omega x r, and computeTireDirs.

    Body coordinates are forward/left/up. Wheel mount values are stored in
    left/up/forward order from the Unity prefab. The ray starts at the
    source-derived WheelCollider centre plus radius, relative to COM.
    """
    base_left, base_up, base_forward = model.suspension_base_xyz_body_m[index]
    com_right, com_up, com_forward = RIGIDBODY_CENTER_OF_MASS_BODY_M
    com_left = -com_right
    # Unity's target-compression offset and PhysX's max-compression ray
    # extension cancel for targetPosition=0.5; retaining the terms makes the
    # source geometry explicit and guards future config changes.
    target_compression = model.suspension_target_travel_m[index]
    max_compression = target_compression
    ray_forward = base_forward - com_forward
    ray_left = base_left - com_left
    ray_up = (base_up - target_compression + model.wheel_radius_m
              + max_compression - com_up)

    sr, cr = ca.sin(roll), ca.cos(roll)
    sp, cp = ca.sin(pitch), ca.cos(pitch)
    normal_forward = -sp
    normal_left = cp * sr
    normal_up = cp * cr
    com_offset = (normal_forward * com_forward
                  + normal_left * com_left + normal_up * com_up)
    ray_offset = (normal_forward * ray_forward + normal_left * ray_left
                  + normal_up * ray_up)
    ray_direction_dot_normal = -normal_up
    ray_travel = -((body_height - model.suspension_ground_y_m)
                   + com_offset + ray_offset) / ray_direction_dot_normal
    contact_forward = ray_forward
    contact_left = ray_left
    contact_up = ray_up - ray_travel

    body_vertical_speed = (body_height_rate + sp * u - cp * sr * v) / (cp * cr)
    omega_x = roll_rate
    omega_y = pitch_rate
    omega_z = yaw_rate
    vx = u + omega_y * contact_up - omega_z * contact_left
    vy = v + omega_z * contact_forward - omega_x * contact_up
    vz = (body_vertical_speed + omega_x * contact_left
          - omega_y * contact_forward)

    long_norm = ca.sqrt(cr * cr + sp * sp * sr * sr)
    long0_forward = cp * cr / long_norm
    long0_left = 0.0
    long0_up = sp / long_norm
    lat0_forward = cp * sp * sr / long_norm
    lat0_left = (sp * sp * sr * sr + cr * cr) / long_norm
    lat0_up = -cp * cp * cr * sr / long_norm
    c, s = ca.cos(steer_angle), ca.sin(steer_angle)
    tire_long = (long0_forward * c + lat0_forward * s,
                 long0_left * c + lat0_left * s,
                 long0_up * c + lat0_up * s)
    tire_lat = (lat0_forward * c - long0_forward * s,
                lat0_left * c - long0_left * s,
                lat0_up * c - long0_up * s)
    vx_contact = vx * tire_long[0] + vy * tire_long[1] + vz * tire_long[2]
    vy_contact = vx * tire_lat[0] + vy * tire_lat[1] + vz * tire_lat[2]
    normal = (normal_forward, normal_left, normal_up)
    return vx_contact, vy_contact, normal, tire_long, tire_lat


'''

_FRAME_CALL_250 = '''        forward_speed, lateral_speed = _physx_wheel_bottom_speeds_casadi(
            model, index, angle, u, v, yaw_rate, body_height,
            body_height_rate, roll, roll_rate, pitch, pitch_rate)'''
_FRAME_CALL_251 = '''        (forward_speed, lateral_speed, contact_normal,
         tire_long, tire_lat) = _physx_wheel_contact_frame_casadi(
            model, index, angle, u, v, yaw_rate, body_height,
            body_height_rate, roll, roll_rate, pitch, pitch_rate)'''

_LOAD_OLD = '''        # Native callback tireLoad equals PxVehicleWheelQueryResult.suspSpringForce
        # in the source-matched 1 ms replay (all 1,924 measured wheel samples).
        # The older symbolic filter inflated that load by ~1.49 N. Do not
        # apply it again: native callback already receives the effective load.
        tire_load = suspension_forces[index]'''
_LOAD_NEW = '''        # PxVehicleUpdate filters suspensionForceMag before invoking the
        # tire shader. Keep suspension_forces as the raw source spring force;
        # apply the native normalized-load filter only for tire-force input.
        tire_load = _physx_filtered_tire_load_casadi(
            suspension_forces[index], model.sprung_mass_kg[index],
            WHEEL_MASS_KG, UNITY_PHYSICS_GRAVITY_MPS2)'''

_BODY_FORCE_OLD = '''        # The Unity/PhysX callback returns lateral force in wheel RIGHT,
        # but IPOPT chassis axes are forward/LEFT/up. Ackermann steer angle
        # is LEFT-positive. Right unit vector=(sin(angle), -cos(angle)).
        # Preserve callback raw fy in fy_wheels for contact parity checks.
        fx_body = fx * ca.cos(angle) + fy * ca.sin(angle)
        fy_body = fx * ca.sin(angle) - fy * ca.cos(angle)
        fx_wheels.append(fx)
        fy_wheels.append(fy)
        body_fx.append(fx_body)
        body_fy.append(fy_body)
        slip_x_values.append(slip_x)
        slip_y_values.append(slip_y)
    total_fx = sum(body_fx)
    total_fy = sum(body_fy)
    yaw_moment = sum(WHEEL_X_M[i] * body_fy[i]
                     - WHEEL_Y_M[i] * body_fx[i] for i in range(4))
    return (total_fx, total_fy, yaw_moment, tuple(fx_wheels), tuple(fy_wheels),
            tuple(slip_x_values), tuple(slip_y_values))'''
_BODY_FORCE_NEW = '''        # The callback's signed lateral force is right-positive. The
        # source-projected tire basis is left-positive in model coordinates.
        tire_force_body = tuple(
            fx * tire_long[axis] - fy * tire_lat[axis]
            for axis in range(3))
        suspension_force_body = tuple(
            suspension_forces[index] * contact_normal[axis]
            for axis in range(3))
        wheel_force_body = tuple(
            tire_force_body[axis] + suspension_force_body[axis]
            for axis in range(3))
        force_point = (WHEEL_X_M[index], WHEEL_Y_M[index],
                       WHEEL_FORCE_APPLICATION_Z_M[index])
        total_force_body = tuple(
            total_force_body[axis] + wheel_force_body[axis]
            for axis in range(3))
        total_torque_body = (
            total_torque_body[0]
            + force_point[1] * wheel_force_body[2]
            - force_point[2] * wheel_force_body[1],
            total_torque_body[1]
            + force_point[2] * wheel_force_body[0]
            - force_point[0] * wheel_force_body[2],
            total_torque_body[2]
            + force_point[0] * wheel_force_body[1]
            - force_point[1] * wheel_force_body[0])
        fx_wheels.append(fx)
        fy_wheels.append(fy)
        slip_x_values.append(slip_x)
        slip_y_values.append(slip_y)
    return (total_force_body[0], total_force_body[1], total_force_body[2],
            total_torque_body[0], total_torque_body[1], total_torque_body[2],
            tuple(fx_wheels), tuple(fy_wheels),
            tuple(slip_x_values), tuple(slip_y_values))'''

_FUNCTION_INIT_OLD = '''    body_fx = []
    body_fy = []
    slip_x_values = []'''
_FUNCTION_INIT_NEW = '''    slip_x_values = []
    total_force_body = (0.0, 0.0, 0.0)
    total_torque_body = (0.0, 0.0, 0.0)'''
_IMPORT_Z_OLD = '    WHEEL_Y_M,\n'
_IMPORT_Z_NEW = '    WHEEL_Y_M,\n    WHEEL_FORCE_APPLICATION_Z_M,\n'
_TIRE_DOC_OLD = '    """Four-wheel force/slip equations from source config and planar state."""'
_TIRE_DOC_NEW = '    """Four-wheel source-derived force/slip equations with 3-D contact state."""'


def transform_wheel_model(source: str) -> str:
    def remove_optimizer_non_drive_wheel_helper(text: str) -> str:
        """Keep the IPOPT wheel module free of the zero-throttle wheel lock."""
        helper = "def physx_no_drive_wheel_speed_interval_casadi("
        next_function = "def _wheel_surface_speed_from_physx_long_slip("
        start = text.find(helper)
        if start < 0:
            if "physx_no_drive_wheel_speed_interval_casadi" in text:
                raise RuntimeError("Partial zero-throttle wheel helper remains")
            return text
        end = text.find(next_function, start)
        if end < 0 or text.find(helper, start + len(helper)) >= 0:
            raise RuntimeError("Cannot isolate zero-throttle wheel helper")
        result = text[:start] + text[end:]
        if "physx_no_drive_wheel_speed_interval_casadi" in result:
            raise RuntimeError("IPOPT wheel model still exposes zero-throttle lock")
        result = result.replace(
            "brake torque at each optimizer node",
            "positive motor torque at each optimizer node")
        result = result.replace(
            "PhysX 4.1 non-tank slip equations used when torque/brake is active.",
            "PhysX 4.1 non-tank slip equations for positive drive torque.")
        if "brake" in result.lower():
            remaining = [line.strip() for line in result.splitlines()
                         if "brake" in line.lower()]
            raise RuntimeError(
                "IPOPT wheel model documentation still describes braking: "
                + repr(remaining[:8]))
        return result

    if (source.count(_SIGNATURE_V250) == 1
            and source.count(_FRAME_CALL_251) == 1
            and source.count(_LOAD_NEW) == 1
            and source.count(_BODY_FORCE_NEW) == 1
            and source.count(_FUNCTION_INIT_NEW) == 1
            and source.count(_IMPORT_Z_NEW) == 1
            and source.count(_TIRE_DOC_NEW) == 1):
        return remove_optimizer_non_drive_wheel_helper(source)
    patched = transform_v250(source)
    if "_physx_applied_torque_slips_casadi" not in source:
        return remove_optimizer_non_drive_wheel_helper(patched)
    already_v251 = (
        patched.count(_SIGNATURE_V250) == 1
        and patched.count(_FRAME_CALL_251) == 1
        and patched.count(_LOAD_NEW) == 1
        and patched.count(_BODY_FORCE_NEW) == 1
        and patched.count(_FUNCTION_INIT_NEW) == 1
        and patched.count(_IMPORT_Z_NEW) == 1
        and patched.count(_TIRE_DOC_NEW) == 1)
    if already_v251:
        return remove_optimizer_non_drive_wheel_helper(patched)
    if (patched.count(_SIGNATURE_V250) != 1
            or patched.count(_FRAME_CALL_251) != 0
            or patched.count(_BODY_FORCE_NEW) != 0):
        raise RuntimeError("Unknown v250 wheel model; refusing v251 rewrite")
    if patched.count(_FRAME_CALL_250) != 1:
        raise RuntimeError("Unknown v250 contact frame call")
    if patched.count(_LOAD_OLD) != 1:
        raise RuntimeError("Unknown tire callback load input; refusing v251 rewrite")
    if patched.count(_BODY_FORCE_OLD) != 1:
        raise RuntimeError("Unknown planar wheel wrench block; refusing v251 rewrite")
    if patched.count(_FUNCTION_INIT_OLD) != 1:
        raise RuntimeError("Unknown wheel force accumulator initialization")
    if patched.count(_IMPORT_Z_OLD) != 1:
        raise RuntimeError("Unknown mechanics imports; refusing v251 rewrite")
    if patched.count(_TIRE_DOC_OLD) != 1:
        raise RuntimeError("Unknown tire-model docstring; refusing v251 rewrite")
    patched = patched.replace(_FRAME_CALL_250, _FRAME_CALL_251, 1)
    patched = patched.replace(_LOAD_OLD, _LOAD_NEW, 1)
    patched = patched.replace(_FUNCTION_INIT_OLD, _FUNCTION_INIT_NEW, 1)
    patched = patched.replace(_BODY_FORCE_OLD, _BODY_FORCE_NEW, 1)
    patched = patched.replace(_IMPORT_Z_OLD, _IMPORT_Z_NEW, 1)
    patched = patched.replace(_TIRE_DOC_OLD, _TIRE_DOC_NEW, 1)
    marker = "def tire_forces_casadi(model: UnityWheelModel,"
    if patched.count(marker) != 1:
        raise RuntimeError("Unknown tire-function helper insertion point")
    patched = patched.replace(marker, _FRAME_HELPER + marker, 1)
    return remove_optimizer_non_drive_wheel_helper(patched)


def prepare_working_wheel_model(historical: Path, working: Path,
                                legacy_name: str) -> dict:
    historical, working = Path(historical).resolve(), Path(working).resolve()
    if historical == working:
        raise ValueError("Refusing to edit recovered historical source in place")
    source_path = historical / legacy_name / _WHEEL_REL
    target_path = working / legacy_name / _WHEEL_REL
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    patched = transform_wheel_model(source)
    current = target_path.read_text(encoding="utf-8")
    previous_v251 = patched.replace(_TIRE_DOC_NEW, _TIRE_DOC_OLD, 1)
    helper_start = previous_v251.find(
        "def physx_no_drive_wheel_speed_interval_casadi(")
    helper_end = previous_v251.find(
        "def _wheel_surface_speed_from_physx_long_slip(", helper_start)
    previous_drive_only = previous_v251
    if helper_start >= 0 and helper_end >= 0:
        previous_drive_only = (previous_v251[:helper_start]
                               + previous_v251[helper_end:])
    prior_generated_drive_only = patched.replace(
        "positive motor torque at each optimizer node",
        "brake torque at each optimizer node").replace(
            "PhysX 4.1 non-tank slip equations for positive drive torque.",
            "PhysX 4.1 non-tank slip equations used when torque/brake is active.")
    known = {source, transform_v241(source), transform_v250(source),
             previous_v251, previous_drive_only, patched,
             prior_generated_drive_only}
    if current not in known:
        raise RuntimeError("Unknown generated wheel model; refusing overwrite")
    if current != patched:
        target_path.write_text(patched, encoding="utf-8")
    if source_path.read_bytes() != source_bytes:
        raise RuntimeError("Recovered historical wheel source was modified")
    return {
        "native_ray_plane_contact_velocity": True,
        "native_tire_direction_projection": True,
        "native_default_tire_load_filter": True,
        "native_full_3d_tire_and_suspension_wrench": True,
        "callback_lateral_force_sign": "right-positive callback scalar mapped onto negative left-positive tire basis",
        "native_force_application_point_separate_from_wheel_bottom": True,
        "optimizer_uses_positive_drive_only_wheel_model": True,
        "optimizer_zero_throttle_wheel_lock_helper_removed": True,
        "historical_source_untouched": True,
        "wheel_model_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "wheel_model_path": str(target_path),
        "remaining_discretization": (
            "OCP evaluates this algebraic wrench at spatial collocation nodes; "
            "it is not a 1 ms or 25 ms PhysX scene-step map."),
    }
