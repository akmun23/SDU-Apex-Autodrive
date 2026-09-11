import csv
import importlib.util
import json
import math


_SPEC = importlib.util.spec_from_file_location(
    "score_replay", "tools/model_id/score_replay.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


FIELDS = ["time_s", "x_m", "y_m", "yaw_rad", "u_mps", "v_mps", "r_radps", "steering_rad"]


def _write(path, offset=0.0):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(81):
            time_s = index * 0.025
            writer.writerow({
                "time_s": time_s,
                "x_m": time_s + offset,
                "y_m": 0.1 * time_s,
                "yaw_rad": 0.02 * time_s,
                "u_mps": 4.0 + time_s + offset,
                "v_mps": 0.05 * time_s,
                "r_radps": 0.02,
                "steering_rad": 0.01,
            })


def test_replay_score_reports_recursive_horizons_and_finite_values(tmp_path):
    predicted = tmp_path / "predicted.csv"
    truth = tmp_path / "truth.csv"
    _write(predicted, offset=0.01)
    _write(truth)
    report = _MODULE.score(predicted, truth)
    assert report["matched_row_count"] == 81
    assert "2.00s" in report["recursive_horizon_scores"]
    assert report["state_scores"]["u_mps"]["relative_p95"] < 0.01
    assert json.dumps(report, allow_nan=False)


def test_replay_score_rejects_non_monotonic_source_time(tmp_path):
    predicted = tmp_path / "predicted.csv"
    truth = tmp_path / "truth.csv"
    _write(predicted)
    _write(truth)
    lines = predicted.read_text(encoding="utf-8").splitlines()
    second_data_row = lines[2].split(",")
    second_data_row[0] = "0.0"
    lines[2] = ",".join(second_data_row)
    predicted.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        _MODULE.score(predicted, truth)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("non-monotonic replay timestamps were accepted")
