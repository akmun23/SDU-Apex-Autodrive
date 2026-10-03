from __future__ import annotations

import numpy as np
import pytest

from tools.vehicle_dynamics_learning.intervene_effective_race_teacher import (
    ORACLE_LABEL,
    validate_oracle_alignment,
)


def test_oracle_rows_must_match_current_state_rows() -> None:
    rows = np.asarray([[100, 101, 102], [205, 206, 207]])
    validate_oracle_alignment(rows, rows.copy(), "I1")


@pytest.mark.parametrize("mode", ("I1", "I2", "I4", "I5"))
def test_one_sample_future_shift_is_rejected(mode: str) -> None:
    expected = np.asarray([[100, 101, 102], [205, 206, 207]])
    with pytest.raises(ValueError, match="shifted/future-offset"):
        validate_oracle_alignment(expected, expected + 1, mode)


def test_normal_rollout_is_not_an_oracle_mode() -> None:
    rows = np.asarray([[0, 1]])
    with pytest.raises(ValueError, match="not an oracle mode"):
        validate_oracle_alignment(rows, rows, "I0")


def test_oracle_result_label_is_explicitly_noncausal() -> None:
    assert ORACLE_LABEL == "diagnostic noncausal future intervention"
