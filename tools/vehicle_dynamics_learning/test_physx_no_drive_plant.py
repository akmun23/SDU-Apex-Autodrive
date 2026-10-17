from dataclasses import replace

import pytest

from tools.vehicle_dynamics_learning.physx_no_drive_plant import (
    PhysXNoDriveState,
    PhysXNoDriveWheel,
    step_physx_no_drive_static_planes,
)


def _wheel(center_y=-0.00434, application_y=-0.06334, plane=(0.0, 1.0, 0.0, 0.0)):
    return PhysXNoDriveWheel(
        wheel_center_offset_com_body_m=(0.0, center_y, 0.0),
        suspension_direction_body=(0.0, -1.0, 0.0),
        force_application_offset_com_body_m=(0.0, application_y, 0.0),
        wheel_radius_m=0.059,
        max_compression_m=0.025,
        max_droop_m=0.025,
        wheel_mass_kg=0.109,
        sprung_mass_kg=0.8675,
        spring_rate_npm=500.0,
        damper_rate_nspm=100.0,
        wheel_inertia_kgm2=0.5 * 0.109 * 0.059**2,
        wheel_damping_rate_nms=0.25,
        forward_curve=(0.15, 0.9, 0.25, 0.58, 0.8),
        sideways_curve=(0.01, 1.0, 0.1, 0.5, 1.0),
        steer_angle_rad=0.0,
        drive_torque_nm=0.0,
        brake_torque_nm=428.0,
        cached_hit_plane_world_xyzd=plane,
    )


def test_composed_vehicle_and_free_scene_step_holds_static_source_equilibrium():
    state = PhysXNoDriveState(
        actor_position_world_m=(0.0, 0.0625, 0.0),
        actor_rotation_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_velocity_world_radps=(0.0, 0.0, 0.0),
        wheel_omega_radps=(0.0, 0.0, 0.0, 0.0),
        previous_jounce_m=(None, None, None, None),
    )
    wheel = replace(
        _wheel(),
        wheel_center_offset_com_body_m=(0.0, 0.0, 0.0),
        force_application_offset_com_body_m=(0.0, -0.0625, 0.0),
        wheel_radius_m=0.0625,
        max_compression_m=0.03125,
        max_droop_m=0.03125,
        wheel_mass_kg=0.125,
        sprung_mass_kg=0.5,
        wheel_inertia_kgm2=0.5 * 0.125 * 0.0625**2,
        damper_rate_nspm=0.0,
    )
    result = step_physx_no_drive_static_planes(
        state, (wheel,) * 4, 0.001, 2.0, (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
        gravity_world_mps2=(0.0, -10.0, 0.0))

    assert result.substep_count == 5
    assert result.last_jounce_m == pytest.approx((0.0,) * 4, abs=5e-6)
    assert result.last_suspension_force_n == pytest.approx(
        (5.0,) * 4, abs=0.003)
    assert result.state.actor_position_world_m == pytest.approx(
        state.actor_position_world_m, abs=3e-7)
    assert result.state.linear_velocity_world_mps == pytest.approx(
        (0.0, 0.0, 0.0), abs=2e-6)


def test_composed_step_uses_one_vehicle_substep_at_five_meters_per_second():
    state = PhysXNoDriveState(
        actor_position_world_m=(0.0, 0.01, 0.0),
        actor_rotation_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        linear_velocity_world_mps=(0.0, 0.0, 5.0),
        angular_velocity_world_radps=(0.0, 0.0, 0.0),
        wheel_omega_radps=(0.0, 0.0, 0.0, 0.0),
        previous_jounce_m=(None, None, None, None),
    )
    airborne = _wheel(plane=None)
    result = step_physx_no_drive_static_planes(
        state, (airborne,) * 4, 0.001, 3.47,
        (1 / 0.0935, 1 / 0.1002, 1 / 0.0228),
        (0.0, 0.06434, 0.0), (0.0, 0.0, 0.0, 1.0))

    assert result.substep_count == 1
    assert result.state.actor_position_world_m[2] == pytest.approx(
        5.0 * (1.0 - 0.273 * 0.001) * 0.001, abs=2e-7)
    assert result.last_jounce_m == pytest.approx((-0.025,) * 4)
    assert result.state.previous_jounce_m == pytest.approx((-0.025,) * 4)
