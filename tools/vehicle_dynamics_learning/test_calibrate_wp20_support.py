from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.vehicle_dynamics_learning import calibrate_wp20_support as wp20
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    SUPPORT_FEATURE_NAMES,
)


def test_support_features_use_only_current_and_previous_observations() -> None:
    frames = np.zeros((6, 9), dtype=np.float64)
    frames[:, 0] = np.arange(6) + 2.0
    frames[:, 1] = 0.25
    frames[:, 2] = -0.4
    frames[:, 3] = np.arange(6) * 0.01
    frames[:, 4] = np.arange(6) * 0.02
    frames[:, 5] = np.arange(6) * 0.1
    frames[:, 6] = np.arange(6) * 0.1 + 0.2

    features = wp20._feature_rows(frames)
    assert features.shape == (5, len(SUPPORT_FEATURE_NAMES))
    np.testing.assert_allclose(features[0, 0], np.hypot(3.0, 0.25))
    np.testing.assert_allclose(features[0, 2], 0.01 / wp20.DT_S)
    np.testing.assert_allclose(features[0, 4], 0.02 / wp20.DT_S)
    expected_mismatch = abs(np.mean(frames[1, 5:7]) - frames[1, 0])
    np.testing.assert_allclose(features[0, 6], expected_mismatch)

    changed_future = frames.copy()
    changed_future[3:] = 10000.0
    np.testing.assert_array_equal(
        wp20._feature_rows(frames)[0], wp20._feature_rows(changed_future)[0])


def test_training_features_never_compute_slew_across_sequence_boundary() -> None:
    frames = np.zeros((8, 9), dtype=np.float64)
    frames[:, 0] = 4.0
    frames[:, 3] = np.asarray([0, 0, 0, 0, 100, 100, 100, 100])
    capture = SimpleNamespace(
        bounds=np.asarray([[0, 4], [4, 8]], dtype=np.int64), frames=frames)
    refs = [(0, sequence, row) for sequence in (0, 1)
            for row in (1, 2, 3)]
    training_windows = {"run-a": refs, "support-only-run": refs}
    by_run, counts = wp20._training_feature_bank(
        [capture], training_windows)
    assert counts["run-a"]["sequences"] == 2
    # Three within-sequence transitions per sequence; the -200 rad jump from
    # row 3 to row 4 is not included as a cross-boundary slew.
    assert len(by_run["run-a"]) == 6
    assert np.max(np.abs(by_run["run-a"][:, 2])) == 0.0

    selected, selected_counts = wp20._training_feature_bank(
        [capture], training_windows, allowed_run_ids={"run-a"})
    assert set(selected) == {"run-a"}
    assert set(selected_counts) == {"run-a"}


def test_both_support_scores_are_finite_and_order_near_vs_far_queries() -> None:
    rng = np.random.default_rng(7)
    bank = {
        f"train-{index}": rng.normal(index * 0.2, 0.05, size=(24, 7))
        for index in range(12)
    }
    queries = np.vstack((np.zeros(7), np.full(7, 5.0)))
    scores = wp20._support_scores(queries, bank)
    assert scores["run_balanced_knn_distance"][0] < \
        scores["run_balanced_knn_distance"][1]
    assert scores["local_mahalanobis_distance"][0] < \
        scores["local_mahalanobis_distance"][1]
    assert scores["nearest_independent_run_distance"][0] < \
        scores["nearest_independent_run_distance"][1]
    assert scores["nearest_independent_run_id"][0] in bank
    for name in ("run_balanced_knn_distance", "local_mahalanobis_distance",
                 "nearest_independent_run_distance"):
        assert np.isfinite(scores[name]).all()


def test_calibration_gate_requires_monotonic_whole_run_evidence(monkeypatch) -> None:
    monkeypatch.setattr(wp20, "BOOTSTRAP_REPLICATES", 200)
    run_ids = [f"validation-{index}" for index in range(6)]
    rows = []
    for run_index, run_id in enumerate(run_ids):
        for point in np.linspace(0.0, 1.0, 20):
            score = float(point + run_index * 0.001)
            error = float(0.2 + 2.0 * score + run_index * 0.005)
            rows.append({
                "run_id": run_id,
                "support_scores": {"score": score},
                "normalized_body_trajectory_rmse_2s": error,
                "body_trajectory_rmse_2s_by_channel": {
                    "u_rear_mps": error,
                    "v_rear_mps": error * 0.5,
                    "yaw_rate_rps": error * 0.25,
                },
                "wheel_rate_trajectory_rmse_2s_mps": error * 0.8,
                "encoder_angle_increment_rmse_2s_rad": error * 0.1,
            })
    result = wp20._calibrate_estimator(
        rows, "score", "synthetic kNN", run_ids, seed=23)
    assert result["calibration_gate"]["status"] == "calibrated"
    assert result["metrics"]["normalized_body_trajectory_rmse_2s"][
        "pooled_window_spearman_rho"] > 0.99
    assert result["supported_weak_unsupported_candidate_thresholds"] is not None
    assert set(result["candidate_thresholds_by_validation_percentile"][
        "categories"]) == {"supported", "weak_support", "unsupported"}
