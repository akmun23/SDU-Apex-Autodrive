"""Explicit AutoDRIVE packet-frame conventions for offline identification.

The official bridge copies the simulator vectors into ROS messages without a
coordinate conversion.  The Unity API mapping therefore has to be documented
at the offline boundary instead of being rediscovered by choosing whichever
frame happens to minimize a fit residual.

This module is diagnostics/offline-only.  It is not imported by runtime
odometry, localization, or MPC nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


TWO_PI = 2.0 * math.pi


class PacketFrameError(ValueError):
    """Raised when a raw packet cannot be decoded without inventing data."""


def wrap_angle(angle: float) -> float:
    """Return an angle in [-pi, pi], preserving finite input only."""
    if not math.isfinite(angle):
        raise PacketFrameError("angle is not finite")
    return (angle + math.pi) % TWO_PI - math.pi


def _number(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise PacketFrameError(f"missing or invalid packet field: {key}") from exc
    if not math.isfinite(value):
        raise PacketFrameError(f"packet field is not finite: {key}")
    return value


def yaw_from_api_quaternion(
        qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract API-frame yaw from the raw AutoDRIVE quaternion.

    ``IMU.cs`` publishes the remapped quaternion as x/y/z/w and the bridge
    copies those four values unchanged.  Normalization is performed only for
    this derived diagnostic; the raw quaternion remains in the dataset.
    """
    values = (qx, qy, qz, qw)
    if not all(math.isfinite(value) for value in values):
        raise PacketFrameError("orientation quaternion is not finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        raise PacketFrameError("orientation quaternion has zero norm")
    qx, qy, qz, qw = (value / norm for value in values)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def body_velocity_to_world(
        yaw_rad: float, body_u_mps: float, body_v_mps: float) -> tuple[float, float]:
    """Transform ROS/API body x/y velocity into the map/API x/y plane.

    The bridge publishes linear velocity in the child vehicle frame.  The
    AutoDRIVE API uses x-forward/y-lateral and the quaternion is copied into
    the same ROS frame, so the ordinary planar body-to-world transform applies.
    """
    if not all(math.isfinite(value) for value in
               (yaw_rad, body_u_mps, body_v_mps)):
        raise PacketFrameError("body velocity transform received non-finite input")
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return c * body_u_mps - s * body_v_mps, s * body_u_mps + c * body_v_mps


@dataclass(frozen=True)
class DecodedPacket:
    """Raw packet values plus explicitly derived planar state."""

    time_s: float
    physics_step: int
    position_x_m: float
    position_y_m: float
    position_z_m: float
    yaw_quaternion_rad: float
    yaw_euler_rad: float
    body_u_mps: float
    body_v_mps: float
    body_vertical_velocity_mps: float
    angular_velocity_x_radps: float
    angular_velocity_y_radps: float
    angular_velocity_z_radps: float
    throttle_norm: float
    steering_norm: float
    applied_command_sequence: int | None = None

    @property
    def yaw_rate_radps(self) -> float:
        """Yaw rate is API angular-velocity z, per ``IMU.cs`` mapping."""
        return self.angular_velocity_z_radps


def decode_packet(row: Mapping[str, object]) -> DecodedPacket:
    """Decode one ``simulator_packets.csv`` row without changing raw fields."""
    yaw_quaternion = yaw_from_api_quaternion(
        _number(row, "simulator_orientation_quaternion_x"),
        _number(row, "simulator_orientation_quaternion_y"),
        _number(row, "simulator_orientation_quaternion_z"),
        _number(row, "simulator_orientation_quaternion_w"),
    )
    applied_command_sequence: int | None = None
    raw_sequence = row.get("applied_command_sequence")
    if raw_sequence not in (None, ""):
        try:
            parsed_sequence = int(float(raw_sequence))
        except (TypeError, ValueError) as exc:
            raise PacketFrameError(
                "applied command sequence is not an integer") from exc
        if parsed_sequence < 0:
            raise PacketFrameError("applied command sequence is negative")
        applied_command_sequence = parsed_sequence
    return DecodedPacket(
        time_s=_number(row, "simulation_time_s"),
        physics_step=int(round(_number(row, "simulation_physics_step"))),
        position_x_m=_number(row, "simulator_position_x"),
        position_y_m=_number(row, "simulator_position_y"),
        position_z_m=_number(row, "simulator_position_z"),
        yaw_quaternion_rad=yaw_quaternion,
        yaw_euler_rad=wrap_angle(_number(row, "simulator_orientation_euler_z")),
        body_u_mps=_number(row, "simulator_linear_velocity_x"),
        body_v_mps=_number(row, "simulator_linear_velocity_y"),
        body_vertical_velocity_mps=_number(row, "simulator_linear_velocity_z"),
        angular_velocity_x_radps=_number(row, "simulator_angular_velocity_x"),
        angular_velocity_y_radps=_number(row, "simulator_angular_velocity_y"),
        angular_velocity_z_radps=_number(row, "simulator_angular_velocity_z"),
        throttle_norm=_number(row, "applied_throttle_norm"),
        steering_norm=_number(row, "applied_steering_norm"),
        applied_command_sequence=applied_command_sequence,
    )
