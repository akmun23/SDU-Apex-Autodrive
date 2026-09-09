import csv
import math

from sdu_apex_autodrive.odometry_analysis.localization_run_score import score_csv


def _write(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(stamp, x, error, collision=0):
    return {
        "stamp_s": stamp,
        "gt_x_m": x,
        "gt_y_m": 0.0,
        "gt_yaw_rad": 0.0,
        "amcl_error_m": error,
        "current_map_error_m": error,
        "ekf_error_m": error,
        "odom_error_m": error,
        "amcl_error_rad": 0.0,
        "current_map_error_rad": 0.0,
        "ekf_error_rad": 0.0,
        "odom_error_rad": 0.0,
        "collision_count": collision,
    }


def test_path_normalized_full_run_passes_and_deduplicates(tmp_path):
    rows = [_row(0.0, 0.0, 0.0), _row(1.0, 10.0, 0.1), _row(1.0, 99.0, 9.0)]
    rows += [_row(2.0, 20.0, 0.2), _row(3.0, 40.0, 0.4)]
    path = tmp_path / "run.csv"
    _write(path, rows)

    result = score_csv(path, min_path_length_m=35.0)

    assert result["overall_pass"] is True
    assert result["sample_count"] == 4
    assert math.isclose(result["path_length_m"], 40.0)
    assert result["estimators"]["amcl"]["error_max_percent_of_path"] == 1.0


def test_collision_and_large_error_fail_the_full_run(tmp_path):
    rows = [_row(0.0, 0.0, 0.0), _row(1.0, 20.0, 0.2), _row(2.0, 40.0, 1.0, collision=1)]
    path = tmp_path / "run.csv"
    _write(path, rows)

    result = score_csv(path, min_path_length_m=35.0)

    assert result["collision_free"] is False
    assert result["full_path"] is False
    assert result["overall_pass"] is False


def test_absolute_error_is_preferred_when_available(tmp_path):
    rows = [_row(0.0, 0.0, 0.0), _row(1.0, 20.0, 0.1)]
    rows[1]["amcl_error_m"] = 0.1
    rows[1]["amcl_absolute_error_m"] = 1.0
    path = tmp_path / "run.csv"
    _write(path, rows)

    result = score_csv(path, min_path_length_m=1.0)

    assert math.isclose(result["estimators"]["amcl"]["error_max_m"], 1.0)
    assert result["scoring_mode"] == "relative_first_pair"
