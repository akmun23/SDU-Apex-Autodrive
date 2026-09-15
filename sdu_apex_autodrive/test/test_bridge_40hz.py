from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdu_apex_autodrive.bridge_40hz import (  # noqa: E402
    MAX_SOURCE_INTERVAL_S,
    MIN_SOURCE_INTERVAL_S,
    SOURCE_INTERVAL_COMPARISON_EPSILON_S,
    _source_interval_is_valid,
)
import sdu_apex_autodrive.bridge_40hz as bridge_40hz


def test_source_boundary_rounding_is_accepted_but_real_faults_are_not():
    epsilon = SOURCE_INTERVAL_COMPARISON_EPSILON_S

    assert _source_interval_is_valid(MIN_SOURCE_INTERVAL_S - 0.5 * epsilon)
    assert _source_interval_is_valid(MAX_SOURCE_INTERVAL_S + 0.5 * epsilon)
    assert not _source_interval_is_valid(MIN_SOURCE_INTERVAL_S - 2.0 * epsilon)
    assert not _source_interval_is_valid(MAX_SOURCE_INTERVAL_S + 2.0 * epsilon)


def test_shutdown_does_not_promote_buffered_response_to_timing_fault(monkeypatch):
    monkeypatch.setattr(bridge_40hz, "_shutdown_requested", True)
    monkeypatch.setattr(bridge_40hz, "_timing_fault", False)
    bridge_40hz._trigger_timing_fault(
        "buffered response after orderly shutdown")
    assert bridge_40hz._timing_fault is False
