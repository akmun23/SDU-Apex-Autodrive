"""Float32 ports of PhysX 4.1 ``PxVehicleNoDrive`` wheel-state updates."""

from __future__ import annotations

import math

import numpy as np


UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 5.0
UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS = 5
UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS = 1


def no_drive_substep_count(forward_speed_mps: float) -> int:
    """Unity player's configured `setSubStepCount(5, 5, 1)` branch."""
    if not math.isfinite(forward_speed_mps):
        raise ValueError("forward speed must be finite")
    return (UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS
            if abs(forward_speed_mps) < UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS
            else UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS)


def no_drive_wheel_speed_step(
    omega_radps: float,
    drive_torque_nm: float,
    raw_brake_torque_nm: float,
    tire_torque_nm: float,
    moment_of_inertia_kgm2: float,
    damping_rate_nms: float,
    dt_s: float,
) -> float:
    """One source-equivalent `integrateNoDriveWheelSpeeds` substep.

    `raw_brake_torque_nm` is the positive value stored in PxVehicleNoDrive;
    PhysX applies it against the sign of the current wheel speed. The tire
    torque is signed as returned by the vehicle tire-force callback.
    """
    values = (omega_radps, drive_torque_nm, raw_brake_torque_nm,
              tire_torque_nm, moment_of_inertia_kgm2, damping_rate_nms, dt_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wheel state, torques, and parameters must be finite")
    if (raw_brake_torque_nm < 0.0 or moment_of_inertia_kgm2 <= 0.0
            or damping_rate_nms < 0.0 or dt_s <= 0.0):
        raise ValueError("brake, inertia, damping, or timestep is invalid")

    omega = np.float32(omega_radps)
    drive = np.float32(drive_torque_nm)
    raw_brake = np.float32(raw_brake_torque_nm)
    tire = np.float32(tire_torque_nm)
    inverse_inertia = np.float32(1.0 / np.float32(moment_of_inertia_kgm2))
    damping = np.float32(damping_rate_nms)
    dt = np.float32(dt_s)
    brake_applied = raw_brake != np.float32(0.0)
    if omega > 0.0:
        signed_brake = np.float32(-raw_brake)
    elif omega < 0.0:
        signed_brake = raw_brake
    else:
        signed_brake = np.float32(0.0)

    dt_over_inertia = np.float32(dt * inverse_inertia)
    numerator = np.float32(
        omega + dt_over_inertia * np.float32(tire + drive + signed_brake))
    denominator = np.float32(
        np.float32(1.0) + damping * dt_over_inertia)
    result = np.float32(numerator / denominator)
    if brake_applied and np.float32(omega * result) <= np.float32(0.0):
        result = np.float32(0.0)
    return float(result)
