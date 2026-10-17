import math

import numpy as np
import pytest

from tools.vehicle_dynamics_learning.physx_vehicle_source import (
    physx_filtered_tire_load_n,
    physx_jounce_from_wheel_pose_actor_body,
    physx_tire_contact_frame,
    physx_tire_slips,
    physx_wheel_bottom_velocity_mps,
    physx_wheel_pose_position_actor_body,
    physx_wheel_pose_rotation_actor_body,
    physx_wheel_rotation_angle_from_pose_actor_body,
    suspension_force_magnitude_n,
    suspension_jounce_from_cached_plane,
    suspension_jounce_speed_mps,
    unity_tire_force_callback,
    unity_cylinder_collider_plane_clearance_m,
    unity_convex_mesh_plane_clearance_m,
    unity_wheel_collider_offsets_com_body,
    unity_wheel_friction_curve_response,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    RIGIDBODY_CENTER_OF_MASS_BODY_M,
    WHEEL_COLLIDER_CENTER_Y_M,
    WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
    WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW,
    WHEEL_RADIUS_M,
    WHEEL_SUSPENSION_DISTANCE_M,
    WHEEL_SUSPENSION_TARGET_POSITION,
)


FORWARD_CURVE = (0.15, 0.9, 0.25, 0.58, 0.8)
SIDEWAYS_CURVE = (0.01, 1.0, 0.1, 0.5, 1.0)


def test_unity_wheelcollider_mounts_match_serialized_f1tenth_geometry():
    offsets = tuple(
        unity_wheel_collider_offsets_com_body(
            WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M[index],
            WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW[index],
            (0.0, WHEEL_COLLIDER_CENTER_Y_M, 0.0),
            RIGIDBODY_CENTER_OF_MASS_BODY_M,
            WHEEL_SUSPENSION_TARGET_POSITION,
            WHEEL_SUSPENSION_DISTANCE_M,
            WHEEL_RADIUS_M)
        for index in range(4))

    wheel_centres = tuple(offset[0] for offset in offsets)
    force_points = tuple(offset[1] for offset in offsets)
    expected_wheel_centres = (
        (-0.1157411, -0.00443514, 0.17468),
        (0.1159011, -0.00443514, 0.17468),
        (-0.11792, -0.01434, -0.15532),
        (0.11808, -0.01434, -0.15532))
    expected_force_points = (
        (-0.1157411, -0.06343514, 0.17468),
        (0.1159011, -0.06343514, 0.17468),
        (-0.11792, -0.07334, -0.15532),
        (0.11808, -0.07334, -0.15532))
    for actual, expected in zip(wheel_centres, expected_wheel_centres):
        assert actual == pytest.approx(expected, abs=2e-7)
    for actual, expected in zip(force_points, expected_force_points):
        assert actual == pytest.approx(expected, abs=2e-7)


def test_physx_wheel_pose_round_trips_jounce_from_com_relative_geometry():
    center_of_mass = (-0.00008, 0.06434, -0.00468)
    wheel_offset = (-0.1157411, -0.00443514, 0.17468)
    suspension_direction = (0.0, -1.0, 0.0)
    wheel_pose = physx_wheel_pose_position_actor_body(
        center_of_mass, wheel_offset, suspension_direction, 0.0016)

    assert wheel_pose == pytest.approx(
        (-0.1158211, 0.06150486, 0.17), abs=2e-7)
    assert physx_jounce_from_wheel_pose_actor_body(
        wheel_pose, center_of_mass, wheel_offset,
        suspension_direction) == pytest.approx(0.0016, abs=2e-7)


@pytest.mark.parametrize("steer,roll", [
    (-0.6313, 3.5883), (-0.4451, 4.7258), (0.0, -6.1637),
    (0.48, 27.2),
])
def test_wheel_pose_orientation_round_trips_unity_steer_and_physx_roll(
    steer, roll
):
    pose = physx_wheel_pose_rotation_actor_body(steer, roll)
    inferred_roll, residual = physx_wheel_rotation_angle_from_pose_actor_body(
        pose, steer)

    assert math.atan2(math.sin(inferred_roll - roll),
                      math.cos(inferred_roll - roll)) == pytest.approx(
                          0.0, abs=2e-6)
    assert residual < 1e-6


def test_unity_wheel_mesh_cylinder_clearance_uses_axle_and_tire_radius():
    # Built-in cylinder is Y-axis aligned; the prefab's 90-degree Z rotation
    # makes its axle horizontal, so world-up support is the 0.055 m radius.
    clearance = unity_cylinder_collider_plane_clearance_m(
        (0.0, 0.060, 0.0), (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
        0.055, 0.022, (0.0, 1.0, 0.0, 0.0))
    assert clearance == pytest.approx(0.005, abs=2e-7)


def test_unity_wheel_mesh_cylinder_clearance_includes_child_offset():
    clearance = unity_cylinder_collider_plane_clearance_m(
        (0.0, 0.060, 0.0), (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.001, 0.0), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
        0.055, 0.022, (0.0, 1.0, 0.0, 0.0))
    assert clearance == pytest.approx(0.006, abs=2e-7)


def test_unity_wheel_mesh_cylinder_clearance_rejects_nonunit_plane():
    with pytest.raises(ValueError, match="unit length"):
        unity_cylinder_collider_plane_clearance_m(
            (0.0, 0.060, 0.0), (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
            0.055, 0.022, (0.0, 2.0, 0.0, 0.0))


def test_convex_mesh_plane_support_uses_extracted_vertices_and_scale():
    vertices = np.asarray((
        (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5), (0.5, -0.5, 0.5),
        (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)))
    clearances = unity_convex_mesh_plane_clearance_m(
        np.asarray(((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))),
        np.asarray(((0.0, 0.0, 0.0, 1.0),
                    (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))),
        vertices,
        (2.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
    )

    # First: unrotated y half-extent is 0.5. Second: rotation sends the
    # scaled x half-extent (1.0) into world-up.
    assert clearances == pytest.approx((1.5, 0.0), abs=1e-12)


def test_convex_mesh_plane_support_rejects_nonunit_plane():
    with pytest.raises(ValueError, match="unit length"):
        unity_convex_mesh_plane_clearance_m(
            np.asarray(((0.0, 1.0, 0.0),)),
            np.asarray(((0.0, 0.0, 0.0, 1.0),)),
            np.asarray(((0.0, 0.0, 0.0),)),
            (1.0, 1.0, 1.0),
            (0.0, 2.0, 0.0, 0.0),
        )


def test_cached_plane_intersection_matches_physx_ray_jounce_equations():
    result = suspension_jounce_from_cached_plane(
        (0.0, 0.07434, 0.0), (0.0, 0.0, 0.0, 1.0),
        (0.0, -0.00434, 0.0), (0.0, -1.0, 0.0),
        0.059, 0.025, (0.0, 1.0, 0.0, 0.0))
    assert result is not None
    jounce, wheel_bottom = result
    assert jounce == pytest.approx(-0.011, abs=2e-6)
    assert wheel_bottom == pytest.approx((0.0, 0.0, 0.0), abs=2e-7)


def test_cached_hit_plane_is_reused_against_moved_substep_pose():
    arguments = dict(
        chassis_com_rotation_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        wheel_centre_offset_com_body_m=(0.0, -0.00434, 0.0),
        suspension_direction_body=(0.0, -1.0, 0.0),
        wheel_radius_m=0.059,
        max_compression_m=0.025,
        hit_plane_world_xyzw=(0.0, 1.0, 0.0, 0.0),
    )
    first = suspension_jounce_from_cached_plane(
        (0.0, 0.07434, 0.0), **arguments)
    second = suspension_jounce_from_cached_plane(
        (0.0, 0.07534, 0.0), **arguments)
    assert first is not None and second is not None
    assert second[0] == pytest.approx(first[0] - 0.001, abs=2e-6)
    assert second[1][1] == pytest.approx(0.0, abs=2e-7)


def test_cached_plane_rejects_ray_start_inside_hit_plane():
    common = dict(
        chassis_com_position_world_m=(0.0, 0.07434, 0.0),
        chassis_com_rotation_world_xyzw=(0.0, 0.0, 0.0, 1.0),
        wheel_centre_offset_com_body_m=(0.0, -0.00434, 0.0),
        suspension_direction_body=(0.0, -1.0, 0.0),
        wheel_radius_m=0.059,
        max_compression_m=0.025,
    )
    assert suspension_jounce_from_cached_plane(
        **{**common, "chassis_com_position_world_m": (0.0, -0.1, 0.0)},
        hit_plane_world_xyzw=(0.0, 1.0, 0.0, 0.0)) is None


def test_cached_plane_does_not_add_a_normal_direction_rejection():
    result = suspension_jounce_from_cached_plane(
        (0.0, 0.07434, 0.0), (0.0, 0.0, 0.0, 1.0),
        (0.0, -0.00434, 0.0), (0.0, -1.0, 0.0),
        0.059, 0.025, (0.0, -1.0, 0.0, 1.0))
    assert result is not None


def test_uninitialized_previous_jounce_has_zero_source_damping_rate():
    assert suspension_jounce_speed_mps(-0.01, None, 0.0002) == 0.0
    assert suspension_jounce_speed_mps(-0.009, -0.01, 0.0002) == pytest.approx(5.0)


def test_projected_suspension_force_matches_static_sprung_weight():
    force = suspension_force_magnitude_n(
        (0.0, -9.81, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
        0.817, 500.0, 100.0, 0.0, 0.0)
    assert force == pytest.approx(0.817 * 9.81, abs=2e-6)


def test_suspension_force_uses_source_spring_then_damper_order():
    force = suspension_force_magnitude_n(
        (0.0, -9.81, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
        0.817, 500.0, 100.0, 0.01, 0.1)
    assert force == pytest.approx(0.817 * 9.81 + 5.0 + 10.0, abs=3e-6)


def test_projection_disabled_branch_uses_source_unprojected_equation():
    force = suspension_force_magnitude_n(
        (0.0, -9.81, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
        0.817, 500.0, 100.0, 0.0, 0.0,
        disable_force_projection=True)
    assert force == pytest.approx(0.817 * 9.81, abs=2e-6)


def test_contact_frame_matches_physx_flat_ground_steer_rotation():
    steering = 0.3
    longitudinal, lateral = physx_tire_contact_frame(
        (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0), steering)
    assert longitudinal[0] == pytest.approx(math.sin(steering))
    assert longitudinal[1] == pytest.approx(0.0, abs=1e-7)
    assert longitudinal[2] == pytest.approx(math.cos(steering))
    assert lateral == pytest.approx((math.cos(steering), 0.0,
                                     -math.sin(steering)))
    assert sum(a * b for a, b in zip(longitudinal, lateral)) == pytest.approx(
        0.0, abs=1e-7)


def test_contact_frame_is_orthonormal_on_a_tilted_surface():
    angle = 0.2
    normal = (0.0, math.cos(angle), math.sin(angle))
    longitudinal, lateral = physx_tire_contact_frame(
        (0.0, 0.0, 0.0, 1.0), normal, -0.17)
    dot = sum(a * b for a, b in zip(longitudinal, lateral))
    assert sum(value * value for value in longitudinal) == pytest.approx(1.0)
    assert sum(value * value for value in lateral) == pytest.approx(1.0)
    assert dot == pytest.approx(0.0, abs=2e-7)
    assert sum(a * b for a, b in zip(longitudinal, normal)) == pytest.approx(0.0, abs=2e-7)


def test_wheel_point_velocity_is_rigid_body_velocity_at_bottom_contact():
    velocity = physx_wheel_bottom_velocity_mps(
        (0.0, 0.0, 4.0), (0.0, 2.0, 0.0), (0.0, 0.1, 0.0),
        (0.2, 0.0, 0.3), (0.1, 0.0, 0.0))
    # omega x r = (0.6, 0, -0.4), then subtract surface velocity.
    assert velocity == pytest.approx((0.5, 0.0, 3.6), abs=1e-7)


def test_no_drive_tire_slip_uses_active_torque_denominator():
    assert physx_tire_slips(5.0, 0.5, 5.5 / 0.059, 0.059, True) == pytest.approx(
        (0.5 / 5.6, math.atan(0.5 / 6.0)), rel=2e-6)


def test_no_drive_coast_slip_uses_configured_minimum_denominator():
    long_slip, lateral_slip = physx_tire_slips(
        0.2, 0.5, 0.0, 0.059, False, 4.0)
    assert long_slip == pytest.approx(-0.05, abs=1e-7)
    assert lateral_slip == pytest.approx(math.atan(0.5 / 1.2), rel=2e-6)


@pytest.mark.parametrize("slip,expected", [
    (0.0, 0.0),
    (0.075, 0.9 * 0.8 * 0.625),
    (0.15, 0.9 * 0.8),
    (0.25, 0.58 * 0.8),
    (0.5, 0.58 * 0.8),
    (-0.25, -0.58 * 0.8),
])
def test_native_wheel_friction_cubic_and_asymptote(slip, expected):
    assert unity_wheel_friction_curve_response(slip, FORWARD_CURVE) == pytest.approx(
        expected, abs=1e-7)


def test_native_tire_callback_combined_slip_force_and_wheel_reaction():
    torque, force_long, force_lateral, aligning = unity_tire_force_callback(
        0.15, 0.01, 100.0, 0.059, FORWARD_CURVE, SIDEWAYS_CURVE)
    combined_scale = 1.0 / math.sqrt(1.0**2 + 1.0**2)
    assert force_long == pytest.approx(100.0 * 0.9 * 0.8 * combined_scale,
                                       rel=2e-6)
    assert force_lateral == pytest.approx(-100.0 * combined_scale, rel=2e-6)
    assert torque == pytest.approx(-force_long * 0.059, rel=1e-7)
    assert aligning == 0.0


def test_native_tire_callback_applies_stiffness_after_combined_slip_and_surface_friction():
    forward = (0.15, 0.9, 0.25, 0.58, 0.8)
    sideways = (0.01, 1.0, 0.1, 0.5, 0.5)
    torque, force_long, force_lateral, aligning = unity_tire_force_callback(
        0.15, 0.01, 100.0, 0.059, forward, sideways,
        surface_tire_friction=0.4)
    # Native code forms the friction-circle normalization from unscaled curve
    # responses, then applies each curve's stiffness to its final force.
    combined_scale = 1.0 / math.sqrt(1.0**2 + 1.0**2)
    assert force_long == pytest.approx(100.0 * 0.4 * 0.9 * combined_scale * 0.8,
                                       rel=2e-6)
    assert force_lateral == pytest.approx(-100.0 * 0.4 * combined_scale * 0.5,
                                          rel=2e-6)
    assert torque == pytest.approx(-force_long * 0.059, rel=1e-7)
    assert aligning == 0.0


def test_physx_tire_load_filter_and_no_contact_branch():
    sprung = 0.8171947
    rest_load = 9.81 * (sprung + 0.109)
    load = physx_filtered_tire_load_n(
        sprung * 9.81, sprung, 0.109)
    assert load == pytest.approx(
        rest_load * (0.2308 + (sprung / (sprung + 0.109))
                     * (3.0 - 0.2308) / 3.0), rel=2e-6)
    assert physx_filtered_tire_load_n(
        100.0, sprung, 0.109, in_contact=False) == 0.0


@pytest.mark.parametrize("dt", [0.0, -0.001])
def test_jounce_speed_rejects_nonpositive_timestep(dt):
    with pytest.raises(ValueError):
        suspension_jounce_speed_mps(0.0, 0.0, dt)
