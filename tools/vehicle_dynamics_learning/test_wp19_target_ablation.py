from __future__ import annotations

import numpy as np
import pytest

from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    BODY_TARGETS,
    HISTORY_STEPS,
    ROLLOUT_STEPS,
    SUPPORT_FEATURE_NAMES,
    WHEEL_TARGETS,
    Capture,
    _advance_body_numpy,
    _future_command_rows,
    _implied_acceleration_numpy,
    _observable_input_features,
    _run_balanced_normalizer,
    _support_features,
    collect_windows,
    target_values,
)
from tools.vehicle_dynamics_learning.signal_semantics import (
    DT_S,
    WHEEL_RADIUS_M,
)


def _capture() -> Capture:
    count = 2 * (HISTORY_STEPS + ROLLOUT_STEPS + 10)
    frames = np.zeros((count, 9), dtype=np.float32)
    rows = np.arange(count, dtype=np.float32)
    frames[:, 0] = 2.0 + rows * 0.01
    frames[:, 1] = -0.2 + rows * 0.001
    frames[:, 2] = 0.3 + rows * 0.001
    frames[:, 3] = rows * 0.002
    frames[:, 4] = rows * 0.003
    frames[:, 5] = rows * 0.01
    frames[:, 6] = -rows * 0.02
    frames[:, 7] = rows * 0.004
    frames[:, 8] = rows * 0.005
    body = np.column_stack((
        3.0 + rows * 0.01,
        0.1 + rows * 0.002,
        0.2 + rows * 0.001,
    )).astype(np.float32)
    accel = np.full((count, 3), np.nan, dtype=np.float32)
    accel[:-1] = np.asarray((0.4, -0.3, 0.2), dtype=np.float32)
    encoder_rate = np.column_stack((rows * 0.1, -rows * 0.05)).astype(np.float32)
    bounds = np.asarray([[0, count // 2], [count // 2, count]], dtype=np.int64)
    return Capture(
        name="synthetic", source_path=__file__, fixed_path=__file__,
        parent_path=__file__, parent_sha256="0" * 64,
        frames=frames, dt_s=np.full(count, DT_S, dtype=np.float32),
        packet=np.concatenate((np.arange(count // 2), np.arange(count // 2))),
        bounds=bounds, sequence_run=np.asarray([0, 0], dtype=np.int32),
        sequence_reset=np.asarray([0, 1], dtype=np.int32),
        run_ids=np.asarray(["synthetic-run"]),
        splits=np.asarray(["train"]), rigid=np.zeros((count, 13), dtype=np.float32),
        encoder_rate=encoder_rate, encoder_valid=np.ones(count, dtype=bool),
        body=body, acceleration=accel,
        input_features=_observable_input_features(frames),
    )


def test_rigid_body_transition_representations_are_equivalent() -> None:
    body = np.asarray([[6.0, 0.45, 1.2], [3.0, -0.7, -0.8]])
    acceleration = np.asarray([[1.4, -0.9, 3.2], [-2.1, 0.6, -1.5]])
    next_body = _advance_body_numpy(body, acceleration, "effective_acceleration")
    increment = next_body - body

    np.testing.assert_allclose(
        _advance_body_numpy(body, increment, "body_state_increment"),
        next_body, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        _advance_body_numpy(body, next_body, "next_body_state"),
        next_body, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        _implied_acceleration_numpy(body, next_body),
        acceleration, rtol=0.0, atol=1e-11)


def test_observable_history_excludes_simulator_truth_and_future_sensor_rows() -> None:
    frames = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)
    features = _observable_input_features(frames)
    np.testing.assert_array_equal(features, frames[:, [0, 1, 2, 3, 4, 7, 8]])
    capture = _capture()
    sequence, source_row = 0, HISTORY_STEPS
    begin = source_row
    original = _support_features(capture, sequence, source_row)
    mismatch_index = SUPPORT_FEATURE_NAMES.index("wheel_body_mismatch_mps")
    expected_mismatch = abs(float(np.mean(
        capture.frames[begin, 5:7]) - capture.frames[begin, 0]))
    assert original[mismatch_index] == pytest.approx(expected_mismatch)
    capture.frames[begin + 1:, 0:7] = -9999.0
    np.testing.assert_array_equal(
        _support_features(capture, sequence, source_row), original)


def test_future_commands_are_aligned_to_the_state_after_transition() -> None:
    capture = _capture()
    sequence_index, source_row = 0, HISTORY_STEPS
    begin = int(capture.bounds[sequence_index, 0]) + source_row
    commands = _future_command_rows(capture, sequence_index, source_row)
    assert commands.shape == (ROLLOUT_STEPS, 2)
    np.testing.assert_array_equal(commands[0], capture.frames[begin + 1, 7:9])
    np.testing.assert_array_equal(
        commands[-1], capture.frames[begin + ROLLOUT_STEPS, 7:9])


def test_wheel_target_definitions_use_the_canonical_25ms_conversion() -> None:
    capture = _capture()
    source_row = HISTORY_STEPS
    begin = int(capture.bounds[0, 0]) + source_row
    now, following = capture.encoder_rate[begin], capture.encoder_rate[begin + 1]
    expected = {
        "wheel_acceleration": (following - now) / DT_S,
        "wheel_rate_increment": following - now,
        "next_wheel_rate": following,
        "encoder_angle_increment": following * DT_S / WHEEL_RADIUS_M,
    }
    for wheel_target in WHEEL_TARGETS:
        _, feedback, value, valid = target_values(
            capture, 0, source_row, "next_body_state", wheel_target)
        assert valid
        np.testing.assert_allclose(value, expected[wheel_target], rtol=1e-6)
        np.testing.assert_array_equal(feedback, capture.frames[begin + 1, 3:5])

    capture.encoder_valid[begin] = False
    for wheel_target in WHEEL_TARGETS:
        *_, valid = target_values(
            capture, 0, source_row, "next_body_state", wheel_target)
        assert not valid


def test_run_balanced_normalization_gives_each_run_equal_weight() -> None:
    groups = {
        "long_run": np.zeros((5000, 1)),
        "short_run": np.full((3, 1), 10.0),
    }
    mean, scale = _run_balanced_normalizer(groups, minimum=1e-4)
    np.testing.assert_allclose(mean, [5.0])
    np.testing.assert_allclose(scale, [5.0])


def test_windows_never_cross_independent_sequence_bounds() -> None:
    capture = _capture()
    by_run = collect_windows([capture], {0}, {"train"})
    refs = by_run["synthetic-run"]
    assert refs
    sequence_lengths = np.diff(capture.bounds, axis=1).reshape(-1)
    for capture_index, sequence_index, source_row in refs:
        assert capture_index == 0
        assert 0 <= source_row - (HISTORY_STEPS - 1)
        assert source_row + ROLLOUT_STEPS < sequence_lengths[sequence_index]
    assert {sequence_index for _, sequence_index, _ in refs} == {0, 1}


@pytest.mark.parametrize("body_target", BODY_TARGETS)
def test_body_transition_rejects_unknown_dimensions_and_modes(body_target: str) -> None:
    result = _advance_body_numpy(
        np.zeros((2, 3)), np.zeros((2, 3)), body_target)
    assert result.shape == (2, 3)
    with pytest.raises(ValueError):
        _advance_body_numpy(np.zeros(3), np.zeros(2), body_target)
