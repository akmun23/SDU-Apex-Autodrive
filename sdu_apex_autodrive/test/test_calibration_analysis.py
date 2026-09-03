import math

from sdu_apex_autodrive.scripts.analyze_calibration import (
    DEFAULT_GROUND_TRUTH_STEP_JITTER_FACTOR,
    DEFAULT_GROUND_TRUTH_POSITION_MARGIN_M,
    DOCUMENTED_MAX_SPEED_MPS,
    acceleration_envelope,
    classify_motion_regime,
    deduplicate_source_events,
    frozen_encoder_brake_model,
    ground_truth_position,
    ground_truth_step_limit_m,
    motion_regime_metrics,
    phase_is_diagnostic_reset,
    relative_error_metrics,
    slip_model_metrics,
    slip_model_samples,
    throttle_table,
    truth_stamp,
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


def test_ground_truth_position_prefers_timestamped_odom_over_legacy_fields():
    row = _row(gt_x_m=99.0, gt_y_m=98.0, gt_odom_x_m=1.25, gt_odom_y_m=-0.5)

    assert ground_truth_position(row) == (1.25, -0.5)


def test_truth_derivatives_prefer_ground_truth_source_timestamp():
    row = _row(stamp_s=100.5, gt_odom_stamp_s=200.25)

    assert truth_stamp(row) == 200.25


def test_phase_is_diagnostic_reset_excludes_only_reset_transients():
    assert phase_is_diagnostic_reset("grid_reset_0.20_throttle_0.00")
    assert phase_is_diagnostic_reset("reset")
    assert not phase_is_diagnostic_reset("grid_throttle_0.20_at_0.00")


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


def test_slip_model_uses_signed_body_longitudinal_velocity_and_keeps_braking():
    rows = [
        _row(
            phase="grid_throttle_0.800_at_3.00",
            gt_odom_event_count=1,
            gt_vx_mps=2.0,
            gt_speed_mps=10.0,
            gt_longitudinal_accel_mps2=1.0,
            encoder_wheel_speed_mps=3.0,
        ),
        _row(
            phase="grid_brake_3.00_throttle_0.800",
            gt_odom_event_count=2,
            stamp_s=100.1,
            gt_vx_mps=2.0,
            gt_speed_mps=2.0,
            gt_longitudinal_accel_mps2=-1.0,
            encoder_wheel_speed_mps=0.0,
        ),
    ]

    samples = slip_model_samples(rows)
    assert len(samples) == 2
    assert abs(float(samples[0]["slip_ratio"]) - 0.5) < 1.0e-12
    assert samples[1]["motion_regime"] == "braking"
    assert float(samples[1]["slip_ratio"]) == -1.0
    metrics = slip_model_metrics(rows)
    assert {result["motion_regime"] for result in metrics} == {"traction", "braking"}


def test_frozen_encoder_brake_model_fits_speed_dependent_deceleration():
    rows = []
    for index, (speed, deceleration) in enumerate(
            ((1.0, 6.0), (3.0, 7.0), (5.0, 8.0), (7.0, 9.0)), start=1):
        rows.append(_row(
            phase="grid_brake_8.00_throttle_0.800",
            gt_odom_event_count=index,
            stamp_s=100.0 + index * 0.1,
            gt_vx_mps=speed,
            gt_speed_mps=speed,
            gt_longitudinal_accel_mps2=-deceleration,
            encoder_wheel_speed_mps=0.0,
        ))

    result = frozen_encoder_brake_model(rows)

    assert result is not None
    model, bins = result
    assert model["samples"] == 4
    assert model["bins"] == 4
    assert abs(float(model["intercept_mps2"]) - 5.5) < 1.0e-12
    assert abs(float(model["speed_gain_per_s"]) - 0.5) < 1.0e-12
    assert len(bins) == 4


def test_acceleration_envelope_is_monotonic_and_uses_full_throttle_data():
    trace = [
        {
            "phase": "grid_throttle_1.000_at_0.00",
            "throttle": 1.0,
            "speed_mps": float(speed),
            "acceleration_mps2": float(acceleration),
        }
        for speed, acceleration in ((0.0, 5.5), (2.0, 4.0),
                                    (4.0, 4.5), (6.0, 3.0))
    ]

    result = acceleration_envelope(trace)
    limits = [row[2] for row in result]

    assert result[0][3] == 1
    assert result[1][3] == 1
    assert all(left >= right for left, right in zip(limits, limits[1:]))
    assert limits[1] == limits[2]


def test_throttle_table_does_not_use_moving_coast_as_feedforward():
    rows = []
    for index, speed in enumerate((0.2, 0.8, 1.4, 1.9), start=1):
        rows.append(_row(
            phase="grid_throttle_0.100_at_0.00",
            gt_odom_stamp_s=100.0 + index * 0.1,
            gt_speed_mps=speed,
        ))
    for index, speed in enumerate((15.0, 14.6, 14.0, 13.4), start=10):
        rows.append(_row(
            phase="grid_throttle_0.100_at_20.00",
            gt_odom_stamp_s=100.0 + index * 0.1,
            gt_speed_mps=speed,
        ))

    table = throttle_table(rows, "gt_speed_mps")

    assert len(table) == 1
    assert table[0][1] == 0.1
    assert table[0][0] == 1.9


def test_relative_error_metrics_keeps_low_speed_absolute_error_without_percent():
    rows = [
        _row(
            phase="reset",
            stamp_s=100.0,
            gt_odom_event_count=1,
            gt_odom_x_m=0.0,
            gt_odom_y_m=0.0,
            x_odom_m=0.0,
            y_odom_m=0.0,
        ),
        _row(
            phase="speed_4.00",
            stamp_s=100.1,
            gt_odom_stamp_s=100.1,
            gt_odom_event_count=2,
            gt_odom_x_m=0.2,
            gt_odom_y_m=0.0,
            gt_speed_mps=0.5,
            x_odom_m=0.0,
            y_odom_m=0.0,
        ),
        _row(
            phase="speed_4.00",
            stamp_s=100.2,
            gt_odom_stamp_s=100.2,
            gt_odom_event_count=3,
            gt_odom_x_m=1.2,
            gt_odom_y_m=0.0,
            gt_speed_mps=2.0,
            speed_mps=1.0,
            x_odom_m=1.0,
            y_odom_m=0.0,
        ),
    ]

    metrics = relative_error_metrics(rows)
    by_key = {(row["metric"], row["bin"]): row for row in metrics}

    low_speed = by_key[("odom_speed_vs_truth", "0-1_mps")]
    assert low_speed["samples"] == 1
    assert low_speed["relative_samples"] == 0
    assert math.isnan(float(low_speed["relative_error_p95_pct"]))

    one_to_three = by_key[("odom_speed_vs_truth", "1-3_mps")]
    assert one_to_three["relative_samples"] == 1
    assert abs(float(one_to_three["relative_error_median_pct"]) - 50.0) < 1.0e-12


def test_relative_error_metrics_does_not_leak_lower_speeds_into_high_bin():
    rows = [
        _row(phase="grid_throttle_1.000", stamp_s=300.0,
             gt_odom_event_count=1, gt_speed_mps=5.0, speed_mps=4.0),
        _row(phase="grid_throttle_1.000", stamp_s=300.1,
             gt_odom_stamp_s=300.1, gt_odom_event_count=2,
             gt_speed_mps=21.0, speed_mps=20.0),
    ]

    metrics = relative_error_metrics(rows)
    by_key = {(row["metric"], row["bin"]): row for row in metrics}

    high_speed = by_key[("odom_speed_vs_truth", "20-23_mps")]
    assert high_speed["samples"] == 1
    assert abs(float(high_speed["reference_median"]) - 21.0) < 1.0e-12


def test_relative_error_metrics_uses_stable_controller_tail_and_distance_epoch():
    rows = [_row(
        phase="reset", stamp_s=200.0, gt_odom_stamp_s=200.0,
        gt_odom_event_count=1, gt_odom_x_m=0.0, gt_odom_y_m=0.0,
        x_odom_m=0.0, y_odom_m=0.0)]
    for index, (truth_speed, odom_speed, truth_x) in enumerate(
            ((1.0, 0.0, 1.0), (4.0, 3.0, 4.0), (3.5, 3.5, 7.0),
             (3.5, 3.5, 10.0), (3.5, 3.5, 13.0)), start=2):
        rows.append(_row(
            phase="speed_4.00", stamp_s=200.0 + index * 0.1,
            gt_odom_stamp_s=200.0 + index * 0.1, gt_odom_event_count=index,
            target_speed_mps=4.0, gt_speed_mps=truth_speed,
            gt_odom_x_m=truth_x, gt_odom_y_m=0.0,
            x_odom_m=truth_x - 0.5, y_odom_m=0.0,
        ))

    metrics = relative_error_metrics(rows)
    by_key = {(row["metric"], row["bin"]): row for row in metrics}
    target = by_key[("speed_target_vs_truth", "speed_4.00")]
    assert target["samples"] == 2
    assert abs(float(target["absolute_error_median"]) - 0.5) < 1.0e-12
    assert abs(float(target["relative_error_median_pct"]) - 12.5) < 1.0e-12

    endpoint = by_key[("odom_position_endpoint", "epoch_1")]
    assert abs(float(endpoint["reference_max"]) - 12.0) < 1.0e-12
    assert abs(float(endpoint["relative_error_median_pct"]) - (0.5 / 12.0 * 100.0)) < 1.0e-12
