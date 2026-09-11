import csv
import json

import numpy as np
import pytest

from sdu_apex_autodrive.model_id_analysis import analyze


def _write_run(tmp_path, quality_gate=True):
    run_dir = tmp_path / "run"
    assembled = run_dir / "assembled"
    assembled.mkdir(parents=True)
    (assembled / "timing_quality_report.json").write_text(
        json.dumps({"quality_gate_pass": quality_gate, "status": "test"}),
        encoding="utf-8")
    fields = [
        "simulation_time_k_s", "simulation_time_k1_s",
        "simulation_physics_step_k", "simulation_physics_step_k1",
        "dt_sim_s", "u_k_mps", "u_k1_mps", "steering_k_rad",
        "throttle_k_norm",
    ]
    with (assembled / "model_transition_v3.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        u = 0.0
        for index in range(120):
            throttle = [-0.5, 0.0, 0.5, 1.0][index % 4]
            next_u = u + 0.03 * (0.5 * throttle - 0.05 * u)
            writer.writerow({
                "simulation_time_k_s": index * 0.03,
                "simulation_time_k1_s": (index + 1) * 0.03,
                "simulation_physics_step_k": index,
                "simulation_physics_step_k1": index + 1,
                "dt_sim_s": 0.03,
                "u_k_mps": u,
                "u_k1_mps": next_u,
                "steering_k_rad": 0.0,
                "throttle_k_norm": throttle,
            })
            u = next_u
    return run_dir


def test_analysis_refuses_uncertified_source_data(tmp_path):
    with pytest.raises(ValueError, match="quality gate is false"):
        analyze(_write_run(tmp_path, quality_gate=False))


def test_analysis_reports_finite_recursive_scores_and_horizons(tmp_path):
    report = analyze(_write_run(tmp_path))
    score = report["validation_score"]
    assert score["invalid_rollout_count"] == 0
    assert score["recursive"]["finite_count"] == score["samples"]
    assert score["recursive"]["mae"] is not None
    assert "2.00s" in report["validation_recursive_horizon_scores"]
    serialized = json.dumps(report, allow_nan=False)
    assert np.isfinite(report["validation_score"]["recursive"]["p95"])
    assert serialized
