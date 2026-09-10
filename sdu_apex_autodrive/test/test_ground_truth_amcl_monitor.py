import math

import pytest

from sdu_apex_autodrive.ground_truth_amcl_monitor import align_world_pose_to_map


def test_explicit_map_alignment_uses_relative_start_heading() -> None:
    # A simulator launch along world -Y must remain along map -Y when the
    # recorded map start has the same heading.  Rotating by -world_yaw would
    # incorrectly turn this into map +X.
    result = align_world_pose_to_map(
        (0.8, 2.0, -math.pi / 2.0),
        (0.8, 3.0, -math.pi / 2.0),
        (1.2, 5.0, -math.pi / 2.0),
    )
    assert result[0] == pytest.approx(1.2)
    assert result[1] == pytest.approx(4.0)
    assert result[2] == pytest.approx(-math.pi / 2.0)


def test_explicit_map_alignment_rotates_between_frame_headings() -> None:
    result = align_world_pose_to_map(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (10.0, 20.0, math.pi / 2.0),
    )
    assert result[0] == pytest.approx(10.0)
    assert result[1] == pytest.approx(21.0)
    assert result[2] == pytest.approx(math.pi / 2.0)
