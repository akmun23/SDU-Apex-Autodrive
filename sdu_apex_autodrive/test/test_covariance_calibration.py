import math
import csv

import pytest

from sdu_apex_autodrive.odometry_analysis.covariance_calibration import (
    CovarianceSample,
    calibrate_process_noise,
    evaluate_process_noise,
    load_covariance_samples,
)


def test_covariance_fit_requires_clean_motion() -> None:
    samples = [
        CovarianceSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        CovarianceSample(0.05, 0.1, 0.0, 0.001, -0.001, 0.001, 0.1, 0.0, 0.05),
    ]
    with pytest.raises(ValueError, match="insufficient"):
        calibrate_process_noise(samples)


def test_covariance_fit_returns_coverage_candidate() -> None:
    samples = []
    for i in range(1, 41):
        distance = i * 0.1
        error = 0.003 * math.sin(i * 0.7)
        samples.append(CovarianceSample(
            stamp_s=i * 0.05,
            distance_m=distance,
            yaw_distance_rad=i * 0.01,
            error_x_m=error,
            error_y_m=-error * 0.5,
            error_yaw_rad=error * 0.1,
            step_distance_m=0.1,
            step_yaw_rad=0.01,
            dt_s=0.05,
        ))
    result = calibrate_process_noise(samples)
    assert result["samples"] == 40
    assert result["distance_m"] == pytest.approx(4.0)
    candidate = result["candidate"]
    assert all(float(value) > 0.0 for value in candidate.values())
    coverage = result["coverage"]
    assert coverage["component_coverage"] >= 0.95
    assert coverage["ellipse_coverage"] >= 0.95
    assert coverage["yaw_coverage"] >= 0.95


def test_covariance_candidate_is_scored_on_a_chronological_holdout() -> None:
    samples = []
    for i in range(1, 81):
        distance = i * 0.1
        # The suffix is deliberately a little noisier than the fit prefix.
        scale = 0.003 if i <= 60 else 0.006
        error = scale * math.sin(i * 0.7)
        samples.append(CovarianceSample(
            stamp_s=i * 0.05,
            distance_m=distance,
            yaw_distance_rad=i * 0.01,
            error_x_m=error,
            error_y_m=-error * 0.5,
            error_yaw_rad=error * 0.1,
            step_distance_m=0.1,
            step_yaw_rad=0.01,
            dt_s=0.05,
        ))
    fit = calibrate_process_noise(samples[:60])
    holdout = evaluate_process_noise(
        samples[60:], fit["candidate"], reference_stamp_s=samples[0].stamp_s)
    assert holdout["component_coverage"] >= 0.95
    assert holdout["ellipse_coverage"] >= 0.95
    assert holdout["yaw_coverage"] >= 0.95


def test_covariance_evaluator_exposes_holdout_failure() -> None:
    samples = []
    for i in range(1, 21):
        samples.append(CovarianceSample(
            stamp_s=i * 0.05,
            distance_m=i * 0.1,
            yaw_distance_rad=i * 0.01,
            error_x_m=0.0 if i < 16 else 1.0,
            error_y_m=0.0,
            error_yaw_rad=0.0,
            step_distance_m=0.1,
            step_yaw_rad=0.01,
            dt_s=0.05,
        ))
    coverage = evaluate_process_noise(
        samples[15:],
        {
            "process_noise_xy_m2_per_m": 1.0e-9,
            "process_noise_xy_m2_per_s": 1.0e-9,
            "process_noise_yaw2_per_m": 1.0e-9,
            "process_noise_yaw2_per_rad": 1.0e-9,
        },
        reference_stamp_s=samples[0].stamp_s,
    )
    assert coverage["component_coverage"] < 0.95
    assert coverage["ellipse_coverage"] < 0.95


def test_covariance_loader_discards_startup_diagnostic_before_truth(tmp_path) -> None:
    path = tmp_path / "startup_skew.csv"
    header = [
        "source_event_name", "source_event_stamp_s", "gt_odom_x_m",
        "gt_odom_y_m", "gt_odom_yaw_rad", "gt_collision_count",
        "odom_observer_source_stamp_s", "odom_observer_x_m",
        "odom_observer_y_m", "odom_observer_yaw_rad",
        "odom_observer_left_angle_rad", "odom_observer_right_angle_rad",
        "odom_observer_imu_yaw_rad", "odom_observer_sensor_outlier",
        "odom_observer_reset_epoch", "odom_observer_timing_degraded",
    ]
    rows = []
    for stamp, event in [(0.0, "odom_diagnostics"), (0.05, "gt_odom"),
                         (0.10, "gt_odom"), (0.15, "gt_odom"),
                         (0.20, "gt_odom")]:
        row = {key: "" for key in header}
        row["source_event_name"] = event
        row["source_event_stamp_s"] = str(stamp)
        if event == "gt_odom":
            row.update({
                "gt_odom_x_m": str((stamp - 0.05) * 2.0),
                "gt_odom_y_m": "0.0", "gt_odom_yaw_rad": "0.0",
                "gt_collision_count": "0.0",
            })
        else:
            row.update({
                "odom_observer_source_stamp_s": "0.0",
                "odom_observer_x_m": "0.0", "odom_observer_y_m": "0.0",
                "odom_observer_yaw_rad": "0.0",
                "odom_observer_left_angle_rad": "0.0",
                "odom_observer_right_angle_rad": "0.0",
                "odom_observer_imu_yaw_rad": "0.0",
                "odom_observer_sensor_outlier": "0.0",
                "odom_observer_reset_epoch": "0.0",
                "odom_observer_timing_degraded": "0.0",
            })
        rows.append(row)
    # Add three source-exact diagnostics after the truth stream begins.
    for i, stamp in enumerate((0.05, 0.10, 0.15, 0.20)):
        row = {key: "" for key in header}
        row.update({
            "source_event_name": "odom_diagnostics",
            "source_event_stamp_s": str(stamp),
            "gt_collision_count": "0.0",
            "odom_observer_source_stamp_s": str(stamp),
            "odom_observer_x_m": str(i * 0.1),
            "odom_observer_y_m": "0.0", "odom_observer_yaw_rad": "0.0",
            "odom_observer_left_angle_rad": str(i),
            "odom_observer_right_angle_rad": str(i),
            "odom_observer_imu_yaw_rad": "0.0",
            "odom_observer_sensor_outlier": "0.0",
            "odom_observer_reset_epoch": "0.0",
            "odom_observer_timing_degraded": "0.0",
        })
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    samples = load_covariance_samples(path)
    assert len(samples) == 4
    assert samples[0].stamp_s == pytest.approx(0.05)
