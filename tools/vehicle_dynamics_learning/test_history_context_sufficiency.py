"""Focused checks for fixed-capacity WP28 history masking and advancement."""

import numpy as np
import torch
from types import SimpleNamespace

from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    MAX_CONTEXT_STEPS,
    FEATURES,
    FixedCapacityHistoryTransition,
    _draw_plan,
    _history_window,
    advance_context,
    pack_history,
)


def test_context_pack_keeps_recent_rows_and_marks_only_admitted_history() -> None:
    raw = np.arange(MAX_CONTEXT_STEPS * FEATURES, dtype=np.float32).reshape(
        MAX_CONTEXT_STEPS, FEATURES)
    mean = np.full(FEATURES, 1.5, dtype=np.float32)
    scale = np.full(FEATURES, 2.0, dtype=np.float32)
    context_steps = CONTEXT_STEPS["0.5s"]

    packed, mask = pack_history(raw, context_steps, mean, scale)

    assert packed.shape == (MAX_CONTEXT_STEPS, FEATURES)
    assert mask.shape == (MAX_CONTEXT_STEPS,)
    assert np.count_nonzero(mask) == context_steps
    np.testing.assert_array_equal(packed[:-context_steps], 0.0)
    np.testing.assert_allclose(
        packed[-context_steps:], (raw[-context_steps:] - mean) / scale)


def test_short_context_can_start_before_fixed_capacity_history_is_available() -> None:
    features = np.arange(120 * FEATURES, dtype=np.float32).reshape(120, FEATURES)
    capture = SimpleNamespace(
        bounds=np.asarray([[0, 120]], dtype=np.int64),
        input_features=features)

    history = _history_window(capture, 0, 79, context_steps=80)

    np.testing.assert_array_equal(history[:80], 0.0)
    np.testing.assert_array_equal(history[80:], features[:80])
    with np.testing.assert_raises(ValueError):
        _history_window(capture, 0, 79, context_steps=81)


def test_stage_plan_can_use_horizon_safe_run_pools_without_changing_other_stages() -> None:
    short = {"run-a": {"condition": [(0, 0, 10)]}}
    long = {"run-b": {"condition": [(0, 0, 20)]}}
    stages = (("L0", (1,), 2, 1), ("L2", (80, 200), 2, 1))

    plan, _ = _draw_plan(
        short, seed=11, stages=stages, pools_by_stage={"L2": long})

    assert [draw["stage"] for draw in plan] == ["L0", "L0", "L2", "L2"]
    assert all(draw["refs"][0][2] == 10 for draw in plan[:2])
    assert all(draw["refs"][0][2] == 20 for draw in plan[2:])
    assert all(draw["horizon_steps"] in (80, 200) for draw in plan[2:])


def test_advance_context_retains_exact_window_and_fixed_interface() -> None:
    context_steps = CONTEXT_STEPS["0.25s"]
    history = torch.arange(MAX_CONTEXT_STEPS * FEATURES, dtype=torch.float32)
    history = history.reshape(1, MAX_CONTEXT_STEPS, FEATURES)
    mask = torch.zeros(1, MAX_CONTEXT_STEPS)
    mask[:, -context_steps:] = 1.0
    row = torch.full((1, FEATURES), -7.0)

    next_history, next_mask = advance_context(
        history, mask, row, context_steps)

    assert next_history.shape == history.shape
    assert next_mask.shape == mask.shape
    torch.testing.assert_close(next_history[:, :-context_steps],
                               torch.zeros_like(next_history[:, :-context_steps]))
    torch.testing.assert_close(next_history[0, -context_steps:-1],
                               history[0, -context_steps + 1:])
    torch.testing.assert_close(next_history[0, -1], row[0])
    assert torch.count_nonzero(next_mask) == context_steps


def test_all_context_lengths_have_identical_network_capacity() -> None:
    counts = []
    shapes = []
    for seed, context_steps in enumerate(CONTEXT_STEPS.values()):
        torch.manual_seed(seed)
        model = FixedCapacityHistoryTransition(
            np.zeros(5, dtype=np.float32), np.ones(5, dtype=np.float32))
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        history = torch.zeros(3, MAX_CONTEXT_STEPS, FEATURES)
        mask = torch.zeros(3, MAX_CONTEXT_STEPS)
        mask[:, -context_steps:] = 1.0
        output = model(history, mask, torch.zeros(3, 5), torch.zeros(3, 2))
        shapes.append(tuple(output.shape))
        assert torch.isfinite(output).all()
    assert len(set(counts)) == 1
    assert shapes == [(3, 5)] * len(CONTEXT_STEPS)
