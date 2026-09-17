import csv
import importlib.util
from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parents[2] / "tools/model_id"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _load(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_raceline(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "# s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps",
            "ax_mps2", "d_left_m", "d_right_m",
        ])
        writer.writerow([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 1.0])
        writer.writerow([1.0, 1.0, 0.0, 0.0, 0.1, 5.0, 0.5, 0.8, 1.2])


def test_frenet_error_signs_are_local_track_frame():
    module = _load("score_raceline_model")
    raceline = [
        {"s_m": 0.0, "x_m": 0.0, "y_m": 0.0, "psi_rad": 0.0,
         "d_left_m": 1.0, "d_right_m": 1.0},
        {"s_m": 1.0, "x_m": 1.0, "y_m": 0.0, "psi_rad": 0.0,
         "d_left_m": 1.0, "d_right_m": 1.0},
    ]
    error = module.frenet_error(0.2, 0.1, 0.05, 0.0, 0.0, 0.0,
                                module._raceline_arrays(raceline))
    assert error["e_along_m"] > 0.0
    assert error["e_cross_m"] > 0.0
    assert error["e_heading_rad"] > 0.0
    assert error["normalized_corridor_error"] == 0.1


def test_envelope_extracts_coupled_bins_and_writes_compact_csv(tmp_path):
    module = _load("build_raceline_operating_envelope")
    raceline = tmp_path / "raceline.csv"
    config = tmp_path / "path_tracking.yaml"
    output_json = tmp_path / "envelope.json"
    output_csv = tmp_path / "envelope.csv"
    _write_raceline(raceline)
    config.write_text(
        "wheelbase: 0.324\nmax_speed: 16.0\nmax_lateral_accel: 6.5\n",
        encoding="utf-8")
    report = module.build(raceline, config, [], output_json=output_json,
                          output_csv=output_csv)
    assert report["track"]["point_count"] == 2
    assert report["controller_limits"]["max_speed_mps"] == 16.0
    assert report["bins"]["core_bin_count"] == 2
    assert "speed_mps[4,5) x kappa_radpm[0,0.025)" in report[
        "occupancy_2d"]["speed_x_abs_curvature"]
    assert output_json.exists()
    assert output_csv.exists()


def test_score_rows_reports_horizon_run_and_segment_groups():
    module = _load("score_raceline_model")
    raceline = [
        {"s_m": 0.0, "x_m": 0.0, "y_m": 0.0, "psi_rad": 0.0,
         "d_left_m": 1.0, "d_right_m": 1.0},
        {"s_m": 1.0, "x_m": 1.0, "y_m": 0.0, "psi_rad": 0.0,
         "d_left_m": 1.0, "d_right_m": 1.0},
    ]
    rows = [{
        "pred_x_m": 0.2, "pred_y_m": 0.1, "pred_yaw_rad": 0.05,
        "true_x_m": 0.0, "true_y_m": 0.0, "true_yaw_rad": 0.0,
        "horizon_s": "0.750", "run": "run_a", "segment_id": "3",
    }]
    report = module.score_rows(rows, raceline)
    assert report["primary_metric"] == "e_cross_m"
    assert report["overall"]["e_cross_m"]["p95"] == 0.1
    assert "0.750s" in report["horizons"]
    assert "run_a" in report["per_run"]
    assert "3" in report["per_segment"]


def test_operating_class_keeps_speed_curvature_coupled():
    module = _load("score_raceline_model")
    rows = [
        {"s_m": 0.0, "x_m": 0.0, "y_m": 0.0, "psi_rad": 0.0,
         "kappa_radpm": 0.0, "d_left_m": 1.0, "d_right_m": 1.0},
        {"s_m": 1.0, "x_m": 1.0, "y_m": 0.0, "psi_rad": 0.0,
         "kappa_radpm": 0.1, "d_left_m": 1.0, "d_right_m": 1.0},
    ]
    raceline = module._raceline_arrays(rows)
    core = {(4, 0)}
    guard = {(3, 0), (5, 0)}
    assert module.operating_class(0.0, 0.0, 4.0, raceline, core, guard) == "core"
    assert module.operating_class(0.0, 0.0, 5.0, raceline, core, guard) == "guard"
    assert module.operating_class(0.0, 0.0, 4.0, raceline, {(4, 4)}, guard) == "stress"
