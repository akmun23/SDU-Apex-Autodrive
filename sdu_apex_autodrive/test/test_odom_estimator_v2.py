from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "sdu_apex_autodrive/sdu_apex_autodrive/scripts/odom_estimator_v2"
VALIDATION = (
    REPO_ROOT
    / "sdu_apex_autodrive/artifacts/calibration/raw/"
    / "identification_grid_fusion_candidate_20260904/"
    / "identification_grid_20260904_123232.csv"
)


def test_v2_replays_the_attached_holdout(tmp_path: Path) -> None:
    output = tmp_path / "validation_predictions.csv"
    subprocess.run(
        [sys.executable, str(V2_ROOT / "apply_v2.py"), str(VALIDATION), "--output", str(output)],
        cwd=V2_ROOT,
        check=True,
    )
    result = pd.read_csv(output)
    summary = json.loads((V2_ROOT / "results/summary_v2.json").read_text())
    truth = result["truth_speed_mps"].to_numpy(dtype=float)
    estimate = result["estimated_speed_mps"].to_numpy(dtype=float)
    moving = np.isfinite(truth) & (truth >= 1.0)
    error = np.abs(estimate - truth)
    relative = np.full_like(error, np.nan)
    relative[moving] = 100.0 * error[moving] / truth[moving]

    assert len(result) == 62800
    assert abs(float(np.percentile(relative[moving], 95)) - summary["v2_p95_relative_pct"]) < 1.0e-10
    assert abs(float(error[moving].mean()) - summary["v2_mae_mps"]) < 1.0e-10
    assert abs(float(np.mean(relative[moving] <= 2.0)) - summary["fraction_validation_samples_le_2pct"]) < 1.0e-12
