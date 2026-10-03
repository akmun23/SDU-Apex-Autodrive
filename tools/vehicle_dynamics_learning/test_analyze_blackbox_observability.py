from __future__ import annotations

import numpy as np
import pytest

from tools.vehicle_dynamics_learning.analyze_blackbox_observability import (
    HISTORY_LENGTHS,
    SequenceData,
    _aggregate_query_metrics,
    _future_responses,
    _history_context,
    _history_length_decision,
    _select_samples,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import DT_S
from tools.vehicle_dynamics_learning.operating_regions import region_masks
from tools.vehicle_dynamics_learning.signal_semantics import WHEEL_RADIUS_M


def _sequence(core: np.ndarray, filtered: np.ndarray | None = None
              ) -> SequenceData:
    count = len(core)
    frames = np.zeros((count, 9), dtype=np.float64)
    return SequenceData(
        run_id="test", split="validation", frames=frames,
        packet=np.arange(count, dtype=np.int64),
        reset=np.zeros(count, dtype=np.int64),
        state=np.zeros((count, 3), dtype=np.float64),
        pose=np.zeros((count, 3), dtype=np.float64),
        raw_wheel=np.zeros((count, 2), dtype=np.float64),
        raw_valid=np.ones(count, dtype=bool),
        core_features=core,
        filtered_features=filtered if filtered is not None else core,
        responses=np.zeros((count, 1), dtype=np.float64),
        response_names=("target",),
        eligible=np.arange(count, dtype=np.int64),
        region_masks={}, source_path="synthetic")


def test_future_response_alignment_and_encoder_label_validity() -> None:
    count = 100
    time = np.arange(count) * DT_S
    truth = np.column_stack((2.0 * time, np.zeros(count), np.zeros(count)))
    frames = np.zeros((count, 9), dtype=np.float64)
    frames[:, 5] = np.arange(count)
    frames[:, 6] = -np.arange(count)
    pose = np.column_stack((np.zeros(count), np.zeros(count), 0.01 * np.arange(count)))
    wheel = np.full((count, 2), 4.0)
    valid = np.ones(count, dtype=bool)
    valid[0] = False

    responses, names = _future_responses(truth, frames, pose, wheel, valid)
    columns = {name: index for index, name in enumerate(names)}

    assert responses[0, columns["effective_ax_mps2_mean_25ms"]] == 2.0
    assert responses[0, columns["effective_ax_mps2_mean_100ms"]] == 2.0
    assert responses[0, columns["delta_u_mps_250ms"]] == 0.5
    assert responses[0, columns["heading_change_rad_250ms"]] == pytest.approx(0.1)
    assert responses[0, columns["filtered_wheel_left_mps_250ms"]] == 10.0
    # The next-period encoder label is valid even when the current encoder
    # interval is invalid; no future sensor appears in the matching inputs.
    assert responses[0, columns["fixed25_encoder_angle_increment_left_rad"]] == (
        4.0 * DT_S / WHEEL_RADIUS_M)
    assert np.isnan(responses[-1, columns["delta_u_mps_250ms"]])


def test_history_context_is_causal_and_slope_uses_chronological_sign() -> None:
    values = np.repeat(np.arange(12, dtype=np.float64)[:, None], 7, axis=1)
    sequence = _sequence(values)
    first = _history_context(sequence, np.asarray([10]), 4, False)
    changed = values.copy()
    changed[11:] = 10000.0
    sequence_after_future_change = _sequence(changed)
    second = _history_context(sequence_after_future_change, np.asarray([10]),
                              4, False)

    np.testing.assert_array_equal(first, second)
    assert first[0, :7].tolist() == [10.0] * 7
    assert first[0, 7:14].tolist() == [8.0] * 7
    assert first[0, 14:].tolist() == [40.0] * 7


def test_neighbor_metrics_exclude_same_run_and_keep_per_target_uncertainty() -> None:
    contexts = {
        ("query", 0): np.asarray([[0.0]], dtype=np.float32),
    }
    responses = {("query", 0): np.asarray([[999.0]])}
    candidate_contexts = {
        run: np.asarray([[0.0]], dtype=np.float32)
        for run in ("query", "run-a", "run-b", "run-c")}
    candidate_responses = {
        "query": np.asarray([[999.0]]),
        "run-a": np.asarray([[0.0]]),
        "run-b": np.asarray([[1.0]]),
        "run-c": np.asarray([[3.0]]),
    }

    result = _aggregate_query_metrics(
        {"query": [("query", 0)]}, contexts, responses,
        candidate_contexts, candidate_responses,
        center=np.asarray([0.0]), scale=np.asarray([1.0]),
        target_variance=np.asarray([1.0]), target_names=("target",), seed=1)
    metrics = result["metrics"]
    count = metrics["effective_independent_run_count"]["per_target"]["target"]
    assert count["macro_run_mean"] == 3.0
    assert count["independent_query_run_count"] == 1
    assert metrics["normalized_conditional_variance"]["per_target"][
        "target"]["per_run"]["query"] > 0.0


def test_candidate_and_query_caps_apply_to_whole_independent_run() -> None:
    sequences = []
    for run_index in range(7):
        for _ in range(2):
            core = np.zeros((20, 7), dtype=np.float64)
            sequence = _sequence(core)
            sequence.run_id = f"run-{run_index}"
            sequence.split = "validation" if run_index == 6 else "train"
            sequence.region_masks = region_masks(
                sequence.frames, sequence.reset, sequence.packet,
                np.ones(len(sequence.frames), dtype=bool))
            sequence.region_masks = {
                name: np.zeros(len(sequence.frames), dtype=bool)
                for name in sequence.region_masks}
            sequence.eligible = np.arange(len(sequence.frames), dtype=np.int64)
            sequences.append(sequence)

    candidates, queries, _ = _select_samples(
        sequences, np.random.default_rng(42), max_query_rows=5,
        candidate_cap=30, candidate_extra_cap=3)
    assert sum(map(len, candidates["run-0"].values())) == 30
    assert sum(map(len, candidates["run-6"].values())) == 30
    assert sum(map(len, queries["all_valid"]["run-6"].values())) == 5


def test_history_rule_selects_shortest_history_meeting_both_gates() -> None:
    response = "delta_u_mps_250ms"
    values = {
        "0ms": 1.0, "100ms": 0.30, "250ms": 0.105,
        "500ms": 0.102, "1s": 0.101, "2s": 0.1005, "4s": 0.100,
    }
    settings = []
    for length in HISTORY_LENGTHS:
        settings.append({
            "feature_variant": "body_actuator",
            "history_length": length,
            "regions": {"all_valid": {
                "normalized_conditional_variance": {
                    "per_target": {response: {
                        "macro_run_mean": values[length],
                        "independent_query_run_count": 5,
                    }}}}},
        })

    decision = _history_length_decision(settings, [response])
    assert decision["chosen_history_length"] == "250ms"
    assert decision["candidate_history_evaluations"][2][
        "less_than_5pct_next_improvement"]
