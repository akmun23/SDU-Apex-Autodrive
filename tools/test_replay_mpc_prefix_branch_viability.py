import pytest

from tools.replay_mpc_prefix_branch_viability import (
    REPLAY_ARGUMENTS,
    branch_diagnostic,
    get_prefix_controls,
    saved_state_row,
)
from tools.search_mpc_source_plant_continuation import (
    PLANT_STATE_COLUMNS,
    STATE_COLUMNS,
)


def test_next_epoch_replay_exercises_third_backtrack():
    # argv[29] configures the line-search backtrack limit.
    assert REPLAY_ARGUMENTS[29 - 3] == "3"


def test_candidate_and_prior_prefixes_use_the_recorded_cursor():
    candidate = [[0.1, 0.3]] * 30
    prior = [[float(i), 0.2] for i in range(30)]
    candidate_prefix, prior_prefix = get_prefix_controls({
        "status": "accepted_candidate_prefix",
        "candidate_controls": candidate,
        "certified_plan_snapshot": {
            "valid": True, "cursor": 5, "horizon": 30,
            "controls": prior,
        },
    })
    assert candidate_prefix == candidate[:3]
    assert prior_prefix == prior[5:8]


def test_next_epoch_state_preserves_hidden_state_and_advances_stamp():
    diag = {
        "source_stamp_ns": 1_000_000_000,
        "active_speed_ceiling_mps": 9.0,
    }
    stage = {
        "stage": 3, "progress_m": 12.3,
        "mpc_state": list(map(float, range(10))),
        "plant_internal_state": list(map(float, range(22))),
        "suspension_history_initialized": 1,
    }
    predicted = branch_diagnostic(diag, stage)
    row = saved_state_row(predicted)
    assert len(STATE_COLUMNS) == 10
    assert len(PLANT_STATE_COLUMNS) == 22
    assert predicted["stamp_ns"] == 1_075_000_000
    assert predicted["state"] == list(map(float, range(10)))
    assert row[3:13] == list(map(float, range(10)))
    assert row[-22:] == list(map(float, range(22)))


def test_next_epoch_state_rejects_missing_plant_state():
    with pytest.raises(ValueError):
        branch_diagnostic({"source_stamp_ns": 1}, {
            "stage": 3, "progress_m": 1.0,
            "mpc_state": [0.0] * 10,
            "plant_internal_state": [0.0],
        })
