import math
import json

import pytest

from sdu_apex_autodrive.odometry_analysis.localization_error_decomposition import (
    _angle_diff,
    _amcl_along_track_gain_counterfactual,
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


def test_amcl_along_track_counterfactual_is_labeled_and_compares_same_scan():
    rows = [
        {
            "topic": "/current_map_pose", "along_track_error_m": 0.10,
            "amcl_applied_along_track_m": -0.02,
            "amcl_correction_accepted": 1.0,
            "amcl_scan_correction_distance_m": 0.20, "speed_mps": 5.0,
        },
        {
            "topic": "/current_map_pose", "along_track_error_m": 0.10,
            "amcl_applied_along_track_m": 0.02,
            "amcl_correction_accepted": 1.0,
            "amcl_scan_correction_distance_m": 0.20, "speed_mps": 5.0,
        },
        {
            "topic": "/current_map_pose", "along_track_error_m": 0.10,
            "amcl_applied_along_track_m": -0.05,
            "amcl_correction_accepted": 0.0,
            "amcl_scan_correction_distance_m": 0.20, "speed_mps": 5.0,
        },
    ]

    report = _amcl_along_track_gain_counterfactual(rows)

    assert report["offline_only"] is True
    assert "No later AMCL state" in report["method"]
    high_speed = report["groups"]["speed_at_least_3_mps"]
    assert high_speed["samples"] == 2
    assert high_speed["actual_along_track_error_m"]["p95"] == pytest.approx(0.10)
    assert high_speed["double_increment_improves_fraction"] == pytest.approx(0.5)
