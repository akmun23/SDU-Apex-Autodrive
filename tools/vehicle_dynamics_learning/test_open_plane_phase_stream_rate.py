from __future__ import annotations

import pytest

from tools.analyze_open_plane_dynamics import (
    _is_probe_phase,
    _phase_segmented_rate,
)


def test_reset_pauses_do_not_lower_active_stream_rate() -> None:
    receipts = [
        0, 25_000_000, 50_000_000, 75_000_000,
        1_100_000_000, 1_125_000_000, 1_150_000_000, 1_175_000_000,
    ]
    rate, p95_gap, max_gap, over_limit, excluded, active_s = _phase_segmented_rate(
        receipts,
        [(0, 75_000_000), (1_100_000_000, 1_175_000_000)],
    )

    assert rate == pytest.approx(40.0)
    assert p95_gap == pytest.approx(25.0)
    assert max_gap == pytest.approx(25.0)
    assert over_limit == 0
    assert excluded == 1
    assert active_s == pytest.approx(0.15)


def test_in_phase_stall_remains_visible_to_cadence_gate() -> None:
    receipts = [0, 25_000_000, 50_000_000, 150_000_000]
    rate, p95_gap, max_gap, over_limit, excluded, active_s = _phase_segmented_rate(
        receipts, [(0, 150_000_000)],
    )

    assert rate == pytest.approx(20.0)
    assert p95_gap == pytest.approx(92.5)
    assert max_gap == pytest.approx(100.0)
    assert over_limit == 1
    assert excluded == 0
    assert active_s == pytest.approx(0.15)


@pytest.mark.parametrize("label", [
    "probe_yawerr_crawl_fine_onset_v0.65_a0.350_turn+1_delay0.25_step",
    "probe_yawerr_midwheel_v6.50_a0.420_turn-1_down_d0.080_ramp",
])
def test_yaw_error_probe_phases_are_admitted_by_run_quality_gate(label: str) -> None:
    assert _is_probe_phase(label)


@pytest.mark.parametrize("label", [
    "approach_yawerr_crawl_fine_onset_v0.65_a0.350_turn+1",
    "settle_yawerr_crawl_fine_onset_v0.65_a0.350_turn+1",
])
def test_yaw_error_approach_and_settle_are_not_probe_phases(label: str) -> None:
    assert not _is_probe_phase(label)
