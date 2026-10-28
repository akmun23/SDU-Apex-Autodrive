import math
from pathlib import Path

import numpy as np
import pytest

from tools.search_mpc_source_plant_continuation import (
    PLANT_STATE_COLUMNS,
    STEERING_COMMAND_LIMIT_RAD,
    STATE_COLUMNS,
    THROTTLE_MIN,
    _parse_starts,
    controls_from_knots,
    exact_production_objective,
    load_stage_control_seed,
    read_saved_states,
    row_to_diagnostic,
    seed_decision,
    select_saved_state,
    stage_controls_to_knots,
)


def test_controls_from_knots_interpolates_both_inputs():
    knots = np.asarray([0.0] * 10 + [0.01 + i * 0.01 for i in range(10)])
    controls = controls_from_knots(knots)
    assert len(controls) == 30
    assert controls[0] == pytest.approx([0.0, 0.01])
    assert controls[-1] == pytest.approx([0.0, 0.10])
    assert controls[14][1] == pytest.approx(0.05344827586206897)


def test_controls_from_knots_rejects_wrong_shape_and_nan():
    with pytest.raises(ValueError):
        controls_from_knots(np.zeros(19))
    invalid = np.zeros(20)
    invalid[4] = math.nan
    with pytest.raises(ValueError):
        controls_from_knots(invalid)


def test_stage_control_seed_load_and_offset_preserve_30_stage_sequence(
        tmp_path: Path):
    path = tmp_path / "witness.csv"
    rows = ["stage,command_steering_rad,command_throttle"]
    rows.extend(f"{i + 1},{0.01 * i},{0.1 + 0.001 * i}" for i in range(30))
    path.write_text("\n".join(rows) + "\n")
    controls = load_stage_control_seed(path)
    assert len(controls) == 30
    decision = stage_controls_to_knots(controls, knots=30, throttle_offset=0.05)
    assert decision.shape == (60,)
    expanded = controls_from_knots(decision, knots=30)
    assert expanded[0] == pytest.approx([0.0, 0.15])
    assert expanded[-1] == pytest.approx([0.29, 0.179])


def test_stage_control_seed_accepts_production_float32_command_bounds(
        tmp_path: Path):
    path = tmp_path / "production_float_bounds.csv"
    steering = float(np.float32(0.5235987756))
    throttle = float(np.float32(0.001))
    rows = ["stage,command_steering_rad,command_throttle"]
    rows.extend(f"{i + 1},{steering},{throttle}" for i in range(30))
    path.write_text("\n".join(rows) + "\n")
    controls = load_stage_control_seed(path)
    assert controls[0] == pytest.approx(
        [STEERING_COMMAND_LIMIT_RAD, THROTTLE_MIN])
    assert controls_from_knots(stage_controls_to_knots(
        controls, knots=30), knots=30)[0] == pytest.approx(controls[0])


def test_stage_control_seed_rejects_illegal_throttle_offset():
    controls = [[0.0, 0.99] for _ in range(30)]
    with pytest.raises(ValueError, match="outside legal bounds"):
        stage_controls_to_knots(controls, knots=30, throttle_offset=0.02)


def test_stage_control_seed_rejects_wrong_length():
    with pytest.raises(ValueError, match=r"shape \(30, 2\)"):
        stage_controls_to_knots([[0.0, 0.1] for _ in range(29)], knots=30)


def test_seed_decision_supports_full_stage_controls():
    diag = {"progress_m": 0.0, "state": [0.0, 0.0, 2.0]}
    points = [
        {"s": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0,
         "curvature": 0.0, "speed": 2.0, "acceleration": 0.0,
         "left": 1.0, "right": 1.0},
        {"s": 10.0, "x": 10.0, "y": 0.0, "heading": 0.0,
         "curvature": 0.0, "speed": 2.0, "acceleration": 0.0,
         "left": 1.0, "right": 1.0},
    ]
    decision = seed_decision(diag, points, 10.0, 1.0, 0.2, knots=30)
    assert decision.shape == (60,)
    assert decision[30:] == pytest.approx([0.2] * 30)


def test_saved_state_conversion_preserves_queue_and_hidden_states():
    names = ["stamp_ns", "progress_m", "speed_ceiling_mps", *STATE_COLUMNS,
             "suspension_history_initialized", "yaw_residual_radps",
             "yaw_residual_valid", *PLANT_STATE_COLUMNS]
    values = [1.0e18, 27.0, 16.0, *range(10), 1.0,
              0.0, 0.0, *range(22)]
    diag = row_to_diagnostic(names, values)
    assert diag["progress_m"] == 27.0
    assert diag["state"] == list(map(float, range(10)))
    assert diag["plant_internal_state"] == list(map(float, range(22)))
    assert diag["plant_state_flags"][1] == 1
    assert diag["source_stamp_ns"] == int(1.0e18)


def test_read_saved_states_parses_comment_header(tmp_path: Path):
    path = tmp_path / "states.csv"
    names = ["stamp_ns", "progress_m"]
    path.write_text("# " + ",".join(names) + "\n1,27.5\n")
    header, rows = read_saved_states(path)
    assert header == names
    assert rows == [[1.0, 27.5]]


def test_select_saved_state_can_disambiguate_duplicate_progress():
    header = ["progress_m", "value"]
    rows = [[27.4478, 1.0], [27.4479, 2.0]]
    selected, index, distance = select_saved_state(
        header, rows, 27.4479, 0.001, row_index=1)
    assert selected == rows[1]
    assert index == 1
    assert distance == pytest.approx(0.0)


def test_select_saved_state_rejects_invalid_index_or_progress():
    header = ["progress_m"]
    rows = [[27.0]]
    with pytest.raises(ValueError, match="outside"):
        select_saved_state(header, rows, 27.0, 0.01, row_index=1)
    with pytest.raises(ValueError, match="from requested progress"):
        select_saved_state(header, rows, 28.0, 0.01, row_index=0)


def test_parse_starts_keeps_only_legal_throttle():
    assert _parse_starts("0.7:0.001,1.0:0.05") == [(0.7, 0.001), (1.0, 0.05)]
    with pytest.raises(ValueError):
        _parse_starts("1.0:0")


def test_search_scores_the_exact_production_nonlinear_objective():
    assert exact_production_objective({
        "scalar_nonlinear_objective": {"total": 123.5},
    }) == pytest.approx(123.5)
    assert exact_production_objective({}) == 1.0e12
