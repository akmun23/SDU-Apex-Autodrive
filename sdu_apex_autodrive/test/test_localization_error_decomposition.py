import math
import json

import pytest

from sdu_apex_autodrive.odometry_analysis.localization_error_decomposition import (
    _angle_diff,
    _bin_label,
    _nearest_stamped_value,
    _source_stamped_health,
)


def test_angle_difference_wraps_at_pi():
    assert _angle_diff(-math.pi + 0.01, math.pi - 0.01) == pytest.approx(0.02)


def test_speed_bin_labels_have_stable_boundaries():
    edges = [0.0, 2.0, 4.0, math.inf]
    assert _bin_label(0.0, edges, prefix="speed_mps") == "speed_mps[0,2)"
    assert _bin_label(2.0, edges, prefix="speed_mps") == "speed_mps[2,4)"
    assert _bin_label(8.0, edges, prefix="speed_mps") == "speed_mps[4,inf)"


def test_source_stamped_health_join_allows_small_ros_timestamp_rounding_error():
    stamps = [1_000_000_000, 1_025_000_000]
    values = {stamps[0]: 0.025, stamps[1]: 0.026}
    assert _nearest_stamped_value(stamps, values, 1_025_000_200) == pytest.approx(0.026)
    assert _nearest_stamped_value(stamps, values, 1_030_000_000) is None


def test_source_stamped_health_uses_backward_compatible_field_fifteen():
    events = {
        "/amcl_localization_health": [{
            "payload_json": json.dumps({"data": [0.025] + [0.0] * 13 + [1.025]}),
        }],
    }
    stamps, ages = _source_stamped_health(events)
    assert stamps == [1_025_000_000]
    assert ages[1_025_000_000] == pytest.approx(0.025)
