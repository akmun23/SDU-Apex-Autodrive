import math

import pytest

from sdu_apex_autodrive.model_id_frames import (
    PacketFrameError,
    body_velocity_to_world,
    decode_packet,
    yaw_from_api_quaternion,
)


def _packet(**overrides):
    row = {
        "simulation_time_s": "1.0",
        "simulation_physics_step": "40",
        "simulator_position_x": "2.0",
        "simulator_position_y": "3.0",
        "simulator_position_z": "0.1",
        "simulator_orientation_quaternion_x": "0.0",
        "simulator_orientation_quaternion_y": "0.0",
        "simulator_orientation_quaternion_z": str(math.sin(math.pi / 4.0)),
        "simulator_orientation_quaternion_w": str(math.cos(math.pi / 4.0)),
        "simulator_orientation_euler_z": str(math.pi / 2.0),
        "simulator_linear_velocity_x": "2.0",
        "simulator_linear_velocity_y": "0.5",
        "simulator_linear_velocity_z": "0.0",
        "simulator_angular_velocity_x": "0.1",
        "simulator_angular_velocity_y": "0.2",
        "simulator_angular_velocity_z": "-0.7",
        "applied_throttle_norm": "0.3",
        "applied_steering_norm": "-0.1",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def test_api_quaternion_yaw_is_normalized_and_extracted():
    assert yaw_from_api_quaternion(0.0, 0.0, 2.0, 2.0) == pytest.approx(math.pi / 2.0)


def test_body_velocity_transform_uses_planar_api_convention():
    world_x, world_y = body_velocity_to_world(math.pi / 2.0, 2.0, 0.5)
    assert world_x == pytest.approx(-0.5)
    assert world_y == pytest.approx(2.0)


def test_packet_uses_angular_z_as_yaw_rate_and_keeps_raw_velocity_axes():
    packet = decode_packet(_packet())
    assert packet.body_u_mps == pytest.approx(2.0)
    assert packet.body_v_mps == pytest.approx(0.5)
    assert packet.yaw_rate_radps == pytest.approx(-0.7)
    assert packet.yaw_quaternion_rad == pytest.approx(math.pi / 2.0)


def test_zero_quaternion_is_rejected_instead_of_fabricated():
    with pytest.raises(PacketFrameError):
        decode_packet(_packet(
            simulator_orientation_quaternion_x=0.0,
            simulator_orientation_quaternion_y=0.0,
            simulator_orientation_quaternion_z=0.0,
            simulator_orientation_quaternion_w=0.0,
        ))
