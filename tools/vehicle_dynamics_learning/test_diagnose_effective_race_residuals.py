#!/usr/bin/env python3
"""Check residual attribution bins cover the candidate's declared domain."""

import numpy as np

from tools.vehicle_dynamics_learning.diagnose_effective_race_residuals import (
    BUCKETS,
)


def test_full_throttle_and_maximum_slew_are_included_in_factor_bins():
    throttle_edges, throttle_labels = BUCKETS["throttle_command"]
    slew_edges, slew_labels = BUCKETS["abs_throttle_slew_per_s"]

    assert len(throttle_labels) == len(throttle_edges) - 1
    assert len(slew_labels) == len(slew_edges) - 1
    assert throttle_edges[0] <= 0.0 < throttle_edges[1]
    assert throttle_edges[-2] <= 1.0 < throttle_edges[-1]
    assert slew_edges[-2] <= 40.0 < slew_edges[-1]
    np.testing.assert_array_equal(
        throttle_edges,
        np.asarray((0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00001)))
