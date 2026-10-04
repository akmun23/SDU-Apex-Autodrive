"""Checks for first-divergence and regime aggregation diagnostics."""

import numpy as np
import pytest

from tools.vehicle_dynamics_learning.analyze_truncated_history_divergence import (
    _divergence_row,
    _run_region_metrics,
)


def test_divergence_time_uses_first_sample_after_threshold() -> None:
    truth_state = np.zeros((4, 5), dtype=np.float32)
    pred_state = truth_state.copy()
    pred_state[:, 0] = [0.1, 0.2, 0.3, 0.4]
    truth_pose = np.zeros((4, 3), dtype=np.float32)
    pred_pose = truth_pose.copy()
    pred_pose[:, 0] = [0.0, 0.1, 0.2, 0.3]
    row = _divergence_row(pred_state, pred_pose, truth_state, truth_pose, 4)
    assert row["u_gt_0_25mps_first_s"] == pytest.approx(0.075)
    assert row["position_gt_0_10m_first_s"] == pytest.approx(0.075)


def test_region_rollup_reports_no_crossing_as_missing() -> None:
    state = np.zeros((1, 4, 5), dtype=np.float32)
    pose = np.zeros((1, 4, 3), dtype=np.float32)
    result = _run_region_metrics(state, pose, state, pose,
                                 [{"high_speed_near_straight"}], 4)
    assert result["all"]["position_gt_0_25m_first_s"] is None
    assert result["high_speed_near_straight"]["start_count"] == 1
