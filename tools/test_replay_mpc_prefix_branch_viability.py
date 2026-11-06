import pytest

from tools.replay_mpc_prefix_branch_viability import (
    REPLAY_ARGUMENTS,
    branch_diagnostic,
    has_candidate_prior_prefix_comparison,
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


def test_replay_direct_weights_match_production_profile_order():
    assert REPLAY_ARGUMENTS[-12:] == [
        "4000", "1500", "7.5", "60", "25", "5", "7.8125",
        "3", "5", "0.1", "2000", "100",
    ]


def test_candidate_and_prior_prefixes_follow_runtime_cursor_semantics():
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


def test_rejected_cycle_compares_with_next_unpublished_prior_action():
    candidate = [[0.1, 0.3]] * 30
    prior = [[float(i), 0.2] for i in range(30)]
    candidate_prefix, prior_prefix = get_prefix_controls({
        "status": "rejected_nonlinear_rollout",
        "candidate_controls": candidate,
        "certified_plan_snapshot": {
            "valid": True, "cursor": 5, "horizon": 30,
            "controls": prior,
        },
    })
    assert candidate_prefix == candidate[:3]
    assert prior_prefix == prior[6:9]


def test_short_retained_plan_holds_its_last_action_for_prefix_comparison():
    candidate = [[0.1, 0.3]] * 30
    prior = [[float(i), 0.2] for i in range(30)]
    candidate_prefix, prior_prefix = get_prefix_controls({
        "status": "accepted_candidate_prefix",
        "candidate_controls": candidate,
        "certified_plan_snapshot": {
            "valid": True, "cursor": 28, "horizon": 30,
            "controls": prior,
        },
    })
    assert candidate_prefix == candidate[:3]
    assert prior_prefix == [prior[28], prior[29], prior[29]]


def test_quality_comparison_uses_logged_prefixes_after_prior_selection():
    diag = {
        "status": "accepted_prior_plan",
        "quality_aware_prefix_selection": {
            "attempted": True,
            "candidate_prefix_controls": [[0.2, 0.4]] * 30,
            "prior_prefix_controls": [[-0.1, 0.3], [-0.2, 0.3], [-0.3, 0.3]],
        },
    }
    assert has_candidate_prior_prefix_comparison(diag)
    candidate, prior = get_prefix_controls(diag)
    assert candidate == [[0.2, 0.4]] * 3
    assert prior == [[-0.1, 0.3], [-0.2, 0.3], [-0.3, 0.3]]


def test_unattempted_rejection_is_not_a_prefix_branch_event():
    assert not has_candidate_prior_prefix_comparison({
        "status": "rejected_nonlinear_rollout",
        "quality_aware_prefix_selection": {"attempted": False},
    })


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
    assert row[3:13] == [0., 1., 2., 3., 4., 7., 5., 6., 8., 9.]
    assert row[-22:] == list(map(float, range(22)))


def test_next_epoch_state_rejects_missing_plant_state():
    with pytest.raises(ValueError):
        branch_diagnostic({"source_stamp_ns": 1}, {
            "stage": 3, "progress_m": 1.0,
            "mpc_state": [0.0] * 10,
            "plant_internal_state": [0.0],
        })
