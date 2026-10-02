from __future__ import annotations

import numpy as np

from tools.vehicle_dynamics_learning.build_unseen_practice_branch_view import (
    _race_domain_mask,
    _true_intervals,
)


def test_race_domain_rows_resume_only_after_full_post_excursion_cooldown():
    speed = np.asarray([5.0, 5.0, 12.1, *([5.0] * 81)])
    valid = np.ones(len(speed), dtype=bool)

    eligible = _race_domain_mask(speed, valid, cooldown_steps=80)

    assert eligible[:2].all()
    assert not eligible[2:83].any()
    assert eligible[83]


def test_invalid_sample_breaks_an_in_progress_post_excursion_cooldown():
    speed = np.asarray([12.1, *([5.0] * 20), 5.0, *([5.0] * 81)])
    valid = np.ones(len(speed), dtype=bool)
    valid[21] = False

    eligible = _race_domain_mask(speed, valid, cooldown_steps=80)

    assert not eligible[:102].any()
    assert eligible[102]


def test_true_intervals_do_not_bridge_missing_or_rejected_rows():
    assert _true_intervals(np.asarray([False, True, True, False, True])) == [
        (1, 3), (4, 5)]
