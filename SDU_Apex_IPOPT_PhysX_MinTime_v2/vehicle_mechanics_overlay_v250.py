"""Pin active F1TENTH collider geometry to the practice Unity prefab.

The restored historical mechanics file documents +/-5 degree front collider
camber but encoded +/-2.5 degrees. This overlay corrects only those serialized
transform values and the derived wheel-center lateral arms. It never edits the
restored archive and fails closed on unknown source/cache content.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


_MECHANICS_REL = Path("tools/vehicle_dynamics_learning/unity_vehicle_mechanics.py")
_ROTATIONS_OLD = '''WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW = (
    (0.0, 0.0, -math.sin(math.radians(2.5)), math.cos(math.radians(2.5))),
    (0.0, 0.0, math.sin(math.radians(2.5)), math.cos(math.radians(2.5))),
    (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))'''
_ROTATIONS_NEW = '''WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW = (
    # Practice F1TENTH.prefab Front Left/Right WheelCollider transforms are
    # serialized at -5/+5 degrees (not +/-2.5 degrees).
    (0.0, 0.0, -math.sin(math.radians(5.0)), math.cos(math.radians(5.0))),
    (0.0, 0.0, math.sin(math.radians(5.0)), math.cos(math.radians(5.0))),
    (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))'''
_ARMS_OLD = 'WHEEL_Y_M = (0.118, -0.118, 0.118, -0.118)'
_ARMS_NEW = '''# Collider-center lateral moment arms including each serialized center
# offset transformed by its actual WheelCollider local rotation, relative to COM.
WHEEL_Y_M = (0.11574111133813858, -0.11590110510587692,
             0.11792000383138657, -0.11807999759912491)'''
_SPRUNG_OLD = '''WHEEL_SPRUNG_MASSES_KG = (0.817168832, 0.8160193,
                           0.9189912, 0.9178206)'''
_SPRUNG_NEW = '''# Production PxVehicleComputeSprungMasses output for the active practice
# prefab and its serialized +/-5 degree front WheelCollider orientations.
WHEEL_SPRUNG_MASSES_KG = (0.817194700124942, 0.8160186332083913,
                           0.9189924292167562, 0.9177942374499102)'''
_FORCE_Z_NEW = '''WHEEL_FORCE_APPLICATION_Z_M = (
    -0.0634351372718811, -0.0634351372718811,
    -0.07333999872207642, -0.07333999872207642)'''


def transform_mechanics(source: str) -> str:
    rotations_new = source.count(_ROTATIONS_NEW)
    arms_new = source.count(_ARMS_NEW)
    sprung_new = source.count(_SPRUNG_NEW)
    force_z_new = source.count(_FORCE_Z_NEW)
    if rotations_new == 1 and arms_new == 1 and sprung_new == 1 and force_z_new == 1:
        return source
    if (source.count(_ROTATIONS_OLD) != 1
            or source.count(_ARMS_OLD) != 1
            or source.count(_SPRUNG_OLD) != 1):
        raise RuntimeError("Unknown Unity wheel geometry source; refusing patch")
    patched = source.replace(_ROTATIONS_OLD, _ROTATIONS_NEW, 1)
    patched = patched.replace(_ARMS_OLD, _ARMS_NEW, 1)
    patched = patched.replace(_SPRUNG_OLD, _SPRUNG_NEW, 1)
    if patched.count(_ARMS_NEW) != 1:
        raise RuntimeError("Unknown force-point insertion site; refusing patch")
    patched = patched.replace(_ARMS_NEW, _ARMS_NEW + "\n" + _FORCE_Z_NEW, 1)
    return patched


def prepare_working_mechanics(recovered_root: Path, working_root: Path) -> dict:
    recovered_root = Path(recovered_root).resolve()
    working_root = Path(working_root).resolve()
    if recovered_root == working_root:
        raise ValueError("Refusing to modify recovered historical mechanics")
    source_path = recovered_root / _MECHANICS_REL
    target_path = working_root / _MECHANICS_REL
    if not source_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Recovered or working Unity mechanics source is missing")
    source_bytes = source_path.read_bytes()
    original = source_bytes.decode("utf-8")
    patched = transform_mechanics(original)
    current = target_path.read_text(encoding="utf-8")
    if current not in (original, patched):
        raise RuntimeError("Unknown generated Unity mechanics cache; refusing overwrite")
    if current != patched:
        target_path.write_text(patched, encoding="utf-8")
    if source_path.read_bytes() != source_bytes:
        raise RuntimeError("Historical mechanics source was modified")
    return {
        "practice_prefab_front_collider_rotation_deg": [-5.0, 5.0],
        "practice_prefab_front_collider_rotation_corrected": True,
        "sprung_mass_kg": [0.817194700124942, 0.8160186332083913,
                           0.9189924292167562, 0.9177942374499102],
        "wheel_force_application_z_m": [
            -0.0634351372718811, -0.0634351372718811,
            -0.07333999872207642, -0.07333999872207642],
        "practice_prefab_wheel_center_lateral_arms_m": [
            0.11574111133813858, -0.11590110510587692,
            0.11792000383138657, -0.11807999759912491],
        "working_mechanics_sha256": hashlib.sha256(
            patched.encode("utf-8")).hexdigest(),
        "working_mechanics_path": str(target_path),
        "historical_mechanics_untouched": source_path.read_bytes() == source_bytes,
    }
