import csv
import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "tools/model_id/score_legal_n30.py"
_SPEC = importlib.util.spec_from_file_location("score_legal_n30", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


FIELDS = (
    "origin_id", "origin_input_mode", "source_dt_s", "horizon_steps",
    "horizon_s", "pred_x_m", "pred_y_m", "pred_yaw_rad", "truth_x_m",
    "truth_y_m", "truth_yaw_rad",
)


def _write(path: Path, mutate=None) -> None:
    rows = []
    for steps in (4, 10, 20, 30):
        row = {
            "origin_id": f"origin-{steps}",
            "origin_input_mode": "sensor_legal",
            "source_dt_s": 0.025,
            "horizon_steps": steps,
            "horizon_s": steps * 0.025,
            "pred_x_m": 1.1,
            "pred_y_m": 2.0,
            "pred_yaw_rad": 0.15,
            "truth_x_m": 1.0,
            "truth_y_m": 2.0,
            "truth_yaw_rad": 0.10,
        }
        rows.append(row if mutate is None else mutate(row, steps))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_legal_n30_score_requires_the_four_controller_horizons(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(path)
    report = _MODULE.score(path)
    assert report["origin_contract"] == "sensor_legal"
    assert report["horizons"]["0.75s"]["steps"] == 30
    assert report["horizons"]["0.10s"]["position_m"]["p95"] == pytest.approx(0.1)


def test_legal_n30_score_rejects_nonlegal_origin(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(path, lambda row, steps: {**row, "origin_input_mode": "simulator_truth"}
           if steps == 4 else row)
    with pytest.raises(ValueError, match="origin_input_mode"):
        _MODULE.score(path)


def test_legal_n30_score_rejects_bad_cadence_and_long_horizon(tmp_path):
    path = tmp_path / "predictions.csv"
    _write(path, lambda row, steps: {**row, "source_dt_s": 0.1}
           if steps == 4 else row)
    with pytest.raises(ValueError, match="source_dt_s"):
        _MODULE.score(path)

    _write(path, lambda row, steps: {**row, "horizon_steps": 80, "horizon_s": 2.0}
           if steps == 30 else row)
    with pytest.raises(ValueError, match="horizon_steps"):
        _MODULE.score(path)
