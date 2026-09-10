import csv

import pytest

from sdu_apex_autodrive.odometry_analysis.source_time_diagnostic_report import (
    build_report,
)


def test_source_time_report_marks_collision_epoch_and_aligns_poses(tmp_path) -> None:
    input_path = tmp_path / "sensor_record.csv"
    output_path = tmp_path / "source_time.csv"
    fields = [
        "source_event_name", "source_event_stamp_s", "gt_x_m", "gt_y_m",
        "gt_yaw_rad", "gt_speed_mps", "gt_collision_count",
        "odom_observer_x_m", "odom_observer_y_m", "odom_observer_yaw_rad",
        "odom_observer_source_stamp_s", "x_ekf_odom_m", "y_ekf_odom_m",
        "yaw_ekf_odom_rad", "x_amcl_m", "y_amcl_m", "yaw_amcl_rad",
        "x_current_map_m", "y_current_map_m", "yaw_current_map_rad",
        "pp_pose_x_m", "pp_pose_y_m", "pp_pose_yaw_rad",
    ]
    rows = []

    def add(name, stamp, **values):
        row = {field: "" for field in fields}
        row.update({"source_event_name": name, "source_event_stamp_s": str(stamp)})
        row.update({key: str(value) for key, value in values.items()})
        rows.append(row)

    add(
        "amcl", 1.0, x_amcl_m=10.0, y_amcl_m=20.0, yaw_amcl_rad=0.5,
    )
    for stamp, x, y, yaw, collision in (
        (1.0, 0.0, 0.0, 0.0, 0),
        (1.05, 0.1, 0.0, 0.0, 0),
        # The simulator reports the reset pose one source packet before its
        # collision counter increments.  The report must invalidate this row
        # from the pose jump itself.
        (1.10, 1.0, 0.8, 0.5, 0),
    ):
        add(
            "gt_odom", stamp, gt_x_m=x, gt_y_m=y, gt_yaw_rad=yaw,
            gt_speed_mps=1.0, gt_collision_count=collision,
        )
        add(
            "odom_diagnostics", stamp,
            odom_observer_x_m=x, odom_observer_y_m=y,
            odom_observer_yaw_rad=yaw,
            odom_observer_source_stamp_s=stamp,
        )
        add(
            "ekf_odom", stamp, x_ekf_odom_m=x, y_ekf_odom_m=y,
            yaw_ekf_odom_rad=yaw,
        )
        add(
            "amcl", stamp, x_amcl_m=10.0 + 0.8775825619 * x,
            y_amcl_m=20.0 + 0.4794255386 * x,
            yaw_amcl_rad=0.5 + yaw,
        )
        add(
            "current_map_pose", stamp,
            x_current_map_m=10.0 + 0.8775825619 * x,
            y_current_map_m=20.0 + 0.4794255386 * x,
            yaw_current_map_rad=0.5 + yaw,
        )

    with input_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_report(input_path, output_path)
    assert summary["rows"] == 3
    assert summary["pre_collision_rows"] == 2
    assert summary["post_collision_rows"] == 1
    assert summary["max_pre_collision_observer_error_m"] == 0.0
    assert summary["max_pre_collision_ekf_odom_error_m"] == 0.0
    assert output_path.exists()

    with output_path.open(newline="") as stream:
        report = list(csv.DictReader(stream))
    assert report[0]["pre_collision_valid"] == "1"
    assert report[-1]["pre_collision_valid"] == "0"
    assert report[-1]["gt_reset_detected"] == "1"
    assert float(report[1]["current_map_xy_error_m"]) == pytest.approx(0.0, abs=1.0e-9)


def test_explicit_map_start_does_not_align_truth_to_wrong_amcl_pose(tmp_path) -> None:
    input_path = tmp_path / "sensor_record.csv"
    output_path = tmp_path / "source_time.csv"
    fields = [
        "source_event_name", "source_event_stamp_s", "gt_x_m", "gt_y_m",
        "gt_yaw_rad", "gt_collision_count", "x_amcl_m", "y_amcl_m",
        "yaw_amcl_rad",
    ]
    rows = []

    def add(name, stamp, **values):
        row = {field: "" for field in fields}
        row.update({"source_event_name": name, "source_event_stamp_s": str(stamp)})
        row.update({key: str(value) for key, value in values.items()})
        rows.append(row)

    add("amcl", 1.0, x_amcl_m=10.0, y_amcl_m=20.0, yaw_amcl_rad=0.5)
    add("amcl", 1.1, x_amcl_m=10.0, y_amcl_m=20.0, yaw_amcl_rad=0.5)
    add("gt_odom", 1.0, gt_x_m=0.0, gt_y_m=0.0, gt_yaw_rad=0.0,
        gt_collision_count=0)
    add("gt_odom", 1.1, gt_x_m=0.1, gt_y_m=0.0, gt_yaw_rad=0.0,
        gt_collision_count=0)

    with input_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = build_report(
        input_path, output_path, map_start_pose=(100.0, 200.0, 1.0))
    assert summary["map_alignment_mode"] == "explicit_map_start"
    with output_path.open(newline="") as stream:
        report = list(csv.DictReader(stream))
    assert float(report[1]["gt_map_x_m"]) == pytest.approx(
        100.0 + 0.1 * 0.5403023059, abs=1.0e-9)
    assert float(report[1]["gt_map_y_m"]) == pytest.approx(
        200.0 + 0.1 * 0.8414709848, abs=1.0e-9)
    assert float(report[1]["amcl_xy_error_m"]) > 100.0
