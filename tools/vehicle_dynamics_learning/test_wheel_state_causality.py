from __future__ import annotations

import numpy as np
import pytest

from tools import analyze_open_plane_dynamics as analysis
from tools.vehicle_dynamics_learning.diagnose_wheel_state_causality import (
    _source_current_index,
    _wheel_views,
)
from tools.vehicle_dynamics_learning.signal_semantics import (
    DT_S,
    WHEEL_RADIUS_M,
)


def test_encoder_alignment_selects_latest_nonfuture_measurement() -> None:
    stamps = np.asarray([1_000_000_000, 1_025_000_000, 1_050_000_000])
    assert _source_current_index(stamps, 1_049_000_000) == 1
    assert _source_current_index(stamps, 1_081_000_001) is None


def test_wheel_views_match_fixed_rate_and_never_cross_reset_or_packet_gap() -> None:
    origin = 10_000_000_000
    count = 6
    times = origin + np.arange(count, dtype=np.int64) * 25_000_000
    rate = 3.0
    angles = np.arange(count, dtype=np.float64) * rate * DT_S / WHEEL_RADIUS_M
    encoders = {
        topic: (times.copy(), angles.copy())
        for topic in (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)
    }
    packets = np.arange(100, 100 + count, dtype=np.int64)
    resets = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int32)
    views, valid = _wheel_views(
        encoders, times, packets, resets,
        np.zeros((count, 2), dtype=np.float64),
        np.ones(count, dtype=bool),
        np.zeros((count, 2), dtype=np.float64))

    assert not valid[0]
    assert valid[1]
    assert np.isnan(views["rate_25ms_fixed_mps"][2]).all()
    assert np.isnan(views["rate_50ms_fixed_mps"][3]).all()
    assert views["rate_50ms_fixed_mps"][4] == pytest.approx((rate, rate))
    assert np.isnan(views["rate_100ms_fixed_mps"][5]).all()
    assert valid[3] and valid[4] and valid[5]
    assert views["rate_25ms_fixed_mps"][1] == pytest.approx((rate, rate))


def test_packet_gap_invalidates_transition_and_longer_angle_windows() -> None:
    origin = 20_000_000_000
    count = 6
    times = origin + np.arange(count, dtype=np.int64) * 25_000_000
    angles = np.arange(count, dtype=np.float64) * 0.5
    encoders = {topic: (times.copy(), angles.copy()) for topic in
                (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)}
    packets = np.asarray([1, 2, 3, 5, 6, 7], dtype=np.int64)
    resets = np.zeros(count, dtype=np.int32)
    views, valid = _wheel_views(encoders, times, packets, resets,
        np.zeros((count, 2)), np.ones(count, dtype=bool),
        np.zeros((count, 2)))
    assert not valid[3]
    assert np.isnan(views["rate_25ms_fixed_mps"][3]).all()
    assert np.isnan(views["rate_50ms_fixed_mps"][4]).all()
