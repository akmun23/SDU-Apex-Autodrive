import math

import numpy as np
import pytest

from sdu_apex_autodrive.odometry_analysis.scan_match_benchmark import (
    OccupancyMap,
    Pose,
    Scan,
    align_world_to_map,
    odom_seed_in_map,
    scan_endpoints,
    score_pose,
)


def test_world_pixel_conversion_uses_ros_bottom_left_origin():
    image = np.full((5, 7), 255, dtype=np.uint8)
    map_data = OccupancyMap(image, 0.5, -1.0, -2.0)
    col, row = map_data.world_to_pixel(-0.5, -1.0)
    assert float(col) == 1.0
    assert float(row) == 2.0
    x, y = map_data.pixel_to_world(col, row)
    assert float(x) == -0.5
    assert float(y) == -1.0


def test_scan_endpoint_and_likelihood_score_prefer_wall_aligned_pose():
    image = np.full((21, 21), 255, dtype=np.uint8)
    image[:, 15] = 0
    map_data = OccupancyMap(image, 0.1, -1.0, -1.0)
    distance_field = map_data.distance_field()
    scan = Scan(0.0, np.asarray([0.5]), 0.0, 1.0)
    points = scan_endpoints(scan, laser_offset_x_m=0.0, max_beams=10)
    good = score_pose(map_data, distance_field, points, Pose(0.0, 0.0, 0.0))
    bad = score_pose(map_data, distance_field, points, Pose(-0.2, 0.0, 0.0))
    assert points.shape == (1, 2)
    assert good > bad


def test_alignment_preserves_relative_motion_and_rotates_yaw():
    world_start = Pose(10.0, 20.0, 0.5)
    map_start = Pose(-2.0, 3.0, -1.0)
    world_pose = Pose(11.0, 20.0, 0.5 + math.pi / 2.0)
    mapped = align_world_to_map(world_pose, world_start, map_start)
    assert mapped.x_m == pytest.approx(-2.0 + math.cos(-1.5), abs=1.0e-9)
    assert mapped.y_m == pytest.approx(3.0 + math.sin(-1.5), abs=1.0e-9)
    assert mapped.yaw_rad == pytest.approx(-1.0 + math.pi / 2.0, abs=1.0e-9)


def test_odom_seed_uses_verified_map_start_instead_of_raceline_origin():
    local_start = Pose(0.0, 0.0, -math.pi / 2.0)
    map_start = Pose(0.8006, 3.1583, -math.pi / 2.0)
    local_pose = Pose(0.25, 0.10, -math.pi / 2.0 + 0.2)
    seed = odom_seed_in_map(local_pose, local_start, map_start)
    # The observer publishes x/y in its odom/world-aligned axes; the pose
    # translation is therefore not rotated by the vehicle heading when the
    # two frame headings already agree.
    assert seed.x_m == pytest.approx(1.0506, abs=1.0e-9)
    assert seed.y_m == pytest.approx(3.2583, abs=1.0e-9)
    assert seed.yaw_rad == pytest.approx(-math.pi / 2.0 + 0.2, abs=1.0e-9)
