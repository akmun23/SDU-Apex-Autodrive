import json

import pytest

from sdu_apex_autodrive.odometry_analysis.full_speed_localization_report import (
    _delivery_timing_report,
    _motion_regime_scores,
    _position_error_truth_frame,
)


def _packet(source_s, request_ns, arrival_ns, sequence):
    return {
        "payload_json": json.dumps({
            "simulation_time_s": source_s,
            "request_monotonic_ns": request_ns,
            "bridge_arrival_monotonic_ns": arrival_ns,
            "request_sequence": sequence,
            "simulation_physics_step": round(source_s * 1000),
            "simulation_render_frame": round(source_s * 3000),
        }),
        "arrival_monotonic_ns": str(arrival_ns + 1_000_000),
    }


def test_position_error_is_projected_into_offline_truth_heading_frame():
    rows = [
        {"estimate_x_m": 1.0, "estimate_y_m": 0.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0, "truth_yaw_rad": 0.0},
        {"estimate_x_m": -1.0, "estimate_y_m": 0.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0, "truth_yaw_rad": 0.0},
        {"estimate_x_m": 0.0, "estimate_y_m": 1.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0, "truth_yaw_rad": 0.0},
        {"estimate_x_m": 0.0, "estimate_y_m": -1.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0, "truth_yaw_rad": 0.0},
        {"estimate_x_m": 0.0, "estimate_y_m": 1.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0,
         "truth_yaw_rad": 1.5707963267948966},
        {"estimate_x_m": 0.0, "estimate_y_m": -1.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0,
         "truth_yaw_rad": 1.5707963267948966},
        {"estimate_x_m": 1.0, "estimate_y_m": 0.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0,
         "truth_yaw_rad": 1.5707963267948966},
        {"estimate_x_m": -1.0, "estimate_y_m": 0.0,
         "truth_x_m": 0.0, "truth_y_m": 0.0,
         "truth_yaw_rad": 1.5707963267948966},
    ]

    report = _position_error_truth_frame(rows)

    assert report["offline_only"] is True
    assert report["frame"] == "simulator_truth_yaw"
    assert report["longitudinal_error_abs_m"]["p95"] == pytest.approx(1.0)
    assert report["lateral_error_abs_m"]["p95"] == pytest.approx(1.0)
    assert report["longitudinal_error_signed_m"]["mean"] == pytest.approx(0.0)
    assert report["lateral_error_signed_m"]["mean"] == pytest.approx(0.0)


def test_delivery_report_separates_regular_source_from_bunched_arrivals():
    packets = [
        _packet(1.000, 0, 0, 1),
        _packet(1.025, 25_000_000, 6_500_000, 2),
        _packet(1.050, 50_000_000, 49_800_000, 3),
    ]

    report = _delivery_timing_report(packets, {})

    assert report["source_intervals_outside_nominal_15_35ms"] == 0
    assert report["source_intervals_degraded_over_35ms"] == 0
    assert report["source_intervals_outside_supported_1_250ms"] == 0
    assert report["bridge_arrival_intervals_under_15ms"] == 1
    assert report["bridge_arrival_intervals_over_35ms"] == 1
    assert report["short_then_long_arrival_pairs"] == 1
    assert report["bridge_arrival_late_then_early_pairs"] == 0
    assert report["bridge_arrival_early_then_late_pairs"] == 1
    assert report["near_50ms_early_then_late_pairs_pm2ms"] == 1
    assert report["source_interval_s"]["mean"] == pytest.approx(0.025)


def test_delivery_report_detects_user_suggested_30ms_then_20ms_arrivals():
    packets = [
        _packet(1.000, 0, 0, 1),
        _packet(1.025, 25_000_000, 30_000_000, 2),
        _packet(1.050, 50_000_000, 50_000_000, 3),
    ]

    report = _delivery_timing_report(packets, {})

    assert report["bridge_arrival_late_then_early_pairs"] == 1
    assert report["bridge_arrival_early_then_late_pairs"] == 0
    assert report["near_50ms_late_then_early_pairs_pm2ms"] == 1
    assert report["user_30ms_then_20ms_like_pairs"] == 1
    # Arrival jitter does not alter the independently checked source cadence.
    assert report["source_intervals_outside_nominal_15_35ms"] == 0


def test_variable_source_time_is_degraded_not_rejected_as_nominal_fault():
    packets = [
        _packet(1.000, 0, 0, 1),
        _packet(1.037, 25_000_000, 25_000_000, 2),
        _packet(1.074, 50_000_000, 50_000_000, 3),
    ]

    report = _delivery_timing_report(packets, {})

    assert report["source_intervals_outside_nominal_15_35ms"] == 2
    assert report["source_intervals_degraded_over_35ms"] == 2
    assert report["source_intervals_outside_supported_1_250ms"] == 0


def test_short_source_step_is_valid_but_remains_outside_nominal_40hz_band():
    packets = [
        _packet(1.000, 0, 0, 1),
        _packet(1.013, 25_000_000, 22_849_000, 2),
        _packet(1.038, 50_000_000, 49_000_000, 3),
    ]

    report = _delivery_timing_report(packets, {})

    assert report["source_intervals_outside_nominal_15_35ms"] == 1
    assert report["source_intervals_degraded_over_35ms"] == 0
    assert report["source_intervals_outside_supported_1_250ms"] == 0
    assert report["source_interval_s"]["mean"] == pytest.approx(0.019)


def test_source_gap_above_observer_bound_is_reported_unsupported():
    packets = [
        _packet(1.000, 0, 0, 1),
        _packet(1.037, 25_000_000, 25_000_000, 2),
        _packet(1.288, 50_000_000, 50_000_000, 3),
    ]

    report = _delivery_timing_report(packets, {})

    assert report["source_intervals_degraded_over_35ms"] == 2
    assert report["source_intervals_outside_supported_1_250ms"] == 1


def test_delivery_report_calculates_header_age_when_epoch_arrival_is_present():
    events = {
        "/odom": [
            {
                "header_stamp_ns": "1000000000",
                "arrival_monotonic_ns": "100",
                "arrival_epoch_ns": "1005000000",
            },
            {
                "header_stamp_ns": "1025000000",
                "arrival_monotonic_ns": "25100000",
                "arrival_epoch_ns": "1040000000",
            },
        ],
    }

    report = _delivery_timing_report([], events)

    odom = report["topic_timing"]["/odom"]
    assert odom["header_age_on_callback_s"]["mean"] == pytest.approx(0.010)
    assert odom["header_interval_s"]["mean"] == pytest.approx(0.025)


def test_motion_regime_scores_use_offline_full_brake_and_yaw_labels():
    rows = [
        {
            "topic": "/odom",
            "offline_applied_throttle_norm": 0.0,
            "offline_truth_yaw_rate_radps": 0.02,
            "truth_pose_u_mps": 4.0,
            "truth_pose_v_mps": 0.0,
            "u_error_mps": -0.2,
            "u_error_mps_com": -0.25,
            "v_error_mps": 0.01,
            "position_error_m": 0.5,
            "longitudinal_error_truth_frame_m": 0.3,
            "lateral_error_truth_frame_m": 0.4,
            "odom_observer_wheel_update_used": 0.0,
            "odom_observer_speed_prediction_error_mps": 0.1,
        },
        {
            "topic": "/ekf_odom",
            "offline_applied_throttle_norm": 0.0,
            "offline_truth_yaw_rate_radps": 0.5,
            "truth_pose_u_mps": 4.0,
            "truth_pose_v_mps": 0.0,
            "u_error_mps": 0.1,
            "v_error_mps": -0.03,
            "position_error_m": 0.2,
            "longitudinal_error_truth_frame_m": 0.12,
            "lateral_error_truth_frame_m": -0.16,
        },
        {
            "topic": "/odom",
            "offline_applied_throttle_norm": 0.2,
            "offline_truth_yaw_rate_radps": 0.4,
            "truth_pose_u_mps": 5.0,
            "truth_pose_v_mps": 0.0,
            "u_error_mps": 0.05,
            "v_error_mps": 0.02,
            "position_error_m": 0.1,
            "longitudinal_error_truth_frame_m": 0.06,
            "lateral_error_truth_frame_m": 0.08,
        },
        {
            "topic": "/odom",
            "offline_applied_throttle_norm": 0.2,
            "offline_truth_yaw_rate_radps": 0.0,
            "truth_pose_u_mps": 0.2,
            "truth_pose_v_mps": 0.0,
            "u_error_mps": 0.5,
            "v_error_mps": 0.0,
            "position_error_m": 0.5,
        },
    ]

    report = _motion_regime_scores(rows)

    groups = report["groups"]
    assert set(groups) == {"full_brake_straight", "full_brake_turn", "powered_turn"}
    straight = groups["full_brake_straight"]
    assert straight["samples_by_topic"]["/odom"] == 1
    odom = straight["topics"]["/odom"]
    assert odom["u_error_mps"]["p95"] == pytest.approx(0.2)
    assert odom["u_error_mps_com"]["p95"] == pytest.approx(0.25)
    assert odom["u_signed_error_mps_com"]["bias"] == pytest.approx(-0.25)
    assert odom["u_signed_error_mps"]["bias"] == pytest.approx(-0.2)
    assert odom["longitudinal_position_error_truth_frame_m"][
        "p95"] == pytest.approx(0.3)
    assert odom["lateral_position_error_truth_frame_m"][
        "p95"] == pytest.approx(0.4)
    assert odom["position_error_euclidean_m"]["p95"] == pytest.approx(0.5)
    assert odom["observer_wheel_usage"]["wheel_update_fraction"] == pytest.approx(0.0)
    assert odom["observer_wheel_usage"]["imu_prediction_u_error_mps"][
        "p95"] == pytest.approx(0.1)
    assert groups["full_brake_turn"]["samples_by_topic"]["/ekf_odom"] == 1
    assert groups["powered_turn"]["samples_by_topic"]["/odom"] == 1
    assert report["label_policy"]["offline_only"] is True
