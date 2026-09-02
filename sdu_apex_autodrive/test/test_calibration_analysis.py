from sdu_apex_autodrive.scripts.analyze_calibration import (
    DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR,
    DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M,
    DOCUMENTED_MAX_SPEED_MPS,
    classify_motion_regime,
    deduplicate_source_events,
    ground_truth_step_limit_m,
    motion_regime_metrics,
    valid_ground_truth_step,
)


def _row(**values):
    row = {
        "phase": "throttle_0.500",
        "stamp_s": "100.0",
        "time_s": "0.0",
        "gt_odom_event_count": "1",
        "gt_x_m": "0.0",
        "gt_y_m": "0.0",
        "gt_z_m": "0.0",
        "gt_vx_mps": "0.0",
        "gt_vy_mps": "0.0",
        "gt_speed_mps": "0.0",
        "gt_yaw_rate_radps": "0.0",
        "speed_mps": "0.0",
        "left_encoder_rad": "0.0",
        "right_encoder_rad": "0.0",
    }
    row.update({key: str(value) for key, value in values.items()})
    return row


def test_deduplicate_source_events_keeps_latest_callback_snapshot():
    rows = [
        _row(speed_mps=1.0),
        _row(speed_mps=1.5),
        _row(
            phase="settle",
            gt_odom_event_count=2,
            stamp_s=100.1,
            time_s=0.1,
            speed_mps=2.0,
        ),
        _row(
            phase="throttle_0.500",
            gt_odom_event_count=2,
            stamp_s=100.11,
            time_s=0.11,
            speed_mps=2.5,
        ),
        _row(gt_odom_event_count="", stamp_s=100.2, time_s=0.2, speed_mps=3.0),
    ]

    unique, duplicates = deduplicate_source_events(rows)

    assert duplicates == 2
    assert len(unique) == 3
    assert unique[0]["speed_mps"] == "1.5"
    assert unique[1]["gt_odom_event_count"] == "2"
    assert unique[1]["speed_mps"] == "2.5"
    assert unique[2]["speed_mps"] == "3.0"


def test_timestamp_aware_truth_filter_accepts_legal_high_speed_step():
    before = _row(stamp_s=100.0, gt_x_m=0.0)
    after = _row(stamp_s=100.087, gt_x_m=1.95, gt_odom_event_count=2)

    limit = ground_truth_step_limit_m(
        0.087,
        DOCUMENTED_MAX_SPEED_MPS,
        DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR,
        DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M,
    )

    assert limit is not None and limit > 1.95
    assert valid_ground_truth_step(1.95, before, after)


def test_timestamp_aware_truth_filter_rejects_reset_jump_at_short_dt():
    before = _row(stamp_s=100.0, gt_x_m=12.0)
    after = _row(stamp_s=100.02, gt_x_m=12.88, gt_odom_event_count=2)

    assert not valid_ground_truth_step(0.88, before, after)


def test_motion_regime_metrics_reports_unique_event_speed_errors():
    rows = [
        _row(phase="settle", stamp_s=100.0, gt_odom_event_count=1),
        _row(
            stamp_s=100.1,
            gt_odom_event_count=2,
            gt_vx_mps=1.0,
            gt_speed_mps=1.0,
            left_encoder_rad=2.0,
            right_encoder_rad=2.0,
            speed_mps=0.8,
        ),
        _row(
            stamp_s=100.2,
            gt_odom_event_count=3,
            gt_vx_mps=3.0,
            gt_speed_mps=3.0,
            gt_yaw_rate_radps=0.8,
            left_encoder_rad=7.0,
            right_encoder_rad=7.0,
            speed_mps=2.5,
        ),
    ]

    regimes = motion_regime_metrics(rows)
    by_name = {result["regime"]: result for result in regimes}

    assert classify_motion_regime(rows[0], None) == "stationary"
    assert by_name["high_longitudinal_slip"]["events"] == 1
    assert abs(by_name["high_longitudinal_slip"]["odom_speed_mae_mps"] - 0.2) < 1.0e-12
    assert by_name["high_lateral_acceleration"]["events"] == 1
