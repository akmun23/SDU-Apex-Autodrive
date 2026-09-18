from pathlib import Path
import math
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdu_apex_autodrive.bridge_40hz import (  # noqa: E402
    EXPECTED_RATE_HZ,
    MAX_BRIDGE_ARRIVAL_INTERVAL_S,
    MAX_OUTSTANDING_REQUESTS,
    MAX_RESPONSE_WAIT_S,
    MAX_SOURCE_INTERVAL_S,
    MIN_REQUEST_SPACING_S,
    MIN_SOURCE_INTERVAL_S,
    SOURCE_INTERVAL_COMPARISON_EPSILON_S,
    _next_request_deadline,
    _parse_optional_int,
    _format_source_cadence_fault_detail,
    _format_bridge_arrival_fault_detail,
    _source_interval_is_valid,
    _validate_source_packet,
)
import sdu_apex_autodrive.bridge_40hz as bridge_40hz


def test_bounded_pipeline_covers_the_full_response_deadline():
    required_slots = math.ceil(MAX_RESPONSE_WAIT_S / MIN_REQUEST_SPACING_S)

    assert MAX_OUTSTANDING_REQUESTS == required_slots == 7
    assert MAX_RESPONSE_WAIT_S == MAX_BRIDGE_ARRIVAL_INTERVAL_S == 0.150


def test_transport_acceptance_covers_observed_77ms_late_packet():
    assert 0.077032 < MAX_BRIDGE_ARRIVAL_INTERVAL_S


def _seed_source_timing_state(monkeypatch, arrival_ns):
    monkeypatch.setattr(bridge_40hz, "_first_valid_source_packet", True)
    monkeypatch.setattr(bridge_40hz, "_last_packet_arrival_ns", 1_000_000_000)
    monkeypatch.setattr(bridge_40hz, "_last_source_time_s", 1.000)
    monkeypatch.setattr(bridge_40hz, "_last_source_physics_step", 1000)
    monkeypatch.setattr(bridge_40hz, "_last_source_frame", 500)
    monkeypatch.setattr(bridge_40hz, "_last_source_telemetry_sequence", 40)
    monkeypatch.setattr(bridge_40hz, "_valid_source_packet_count", 1)
    monkeypatch.setattr(bridge_40hz, "_reset_level", False)
    monkeypatch.setattr(bridge_40hz, "_timing_fault", False)
    monkeypatch.setattr(bridge_40hz, "_shutdown_requested", False)
    monkeypatch.setattr(bridge_40hz, "_active_request_monotonic_ns", 1_000_000_000)
    monkeypatch.setattr(bridge_40hz, "_active_request_sequence", 41)
    monkeypatch.setattr(bridge_40hz.time, "monotonic_ns", lambda: arrival_ns)


def test_valid_source_packet_after_77ms_transport_gap_is_accepted(monkeypatch):
    faults = []
    _seed_source_timing_state(monkeypatch, 1_077_032_000)
    monkeypatch.setattr(bridge_40hz, "_trigger_timing_fault", faults.append)

    packet = {
        "Simulation Time": "1.025",
        "Physics Step": "1025",
        "Simulation Frame": "501",
        "Telemetry Sequence": "41",
    }

    assert _validate_source_packet(packet)
    assert not faults
    assert bridge_40hz._last_source_time_s == 1.025


def test_transport_gap_over_150ms_fails_with_source_details(monkeypatch):
    faults = []
    _seed_source_timing_state(monkeypatch, 1_151_000_000)
    monkeypatch.setattr(bridge_40hz, "_trigger_timing_fault", faults.append)

    packet = {
        "Simulation Time": "1.025",
        "Physics Step": "1025",
        "Simulation Frame": "501",
        "Telemetry Sequence": "41",
    }

    assert not _validate_source_packet(packet)
    assert len(faults) == 1
    assert "bridge transport interval 0.151000s" in faults[0]
    assert "source_interval=0.025000s" in faults[0]
    assert "physics_step_delta=25" in faults[0]
    assert "telemetry_sequence=40->41" in faults[0]


def test_late_request_cannot_make_the_next_request_burst_early():
    period = 1.0 / EXPECTED_RATE_HZ

    # An on-time emission retains the periodic grid.
    assert _next_request_deadline(10.050, 10.025, period) == 10.050

    # A send that was 6 ms late moves the next deadline to a 24 ms floor,
    # instead of catching up 19 ms later; the 25 ms phase grid resumes after.
    assert math.isclose(
        _next_request_deadline(10.050, 10.031, period),
        10.055,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_collision_count_parser_preserves_optional_offline_label():
    assert _parse_optional_int("8") == 8
    assert _parse_optional_int(0) == 0
    assert _parse_optional_int("") is None
    assert _parse_optional_int("not-a-count") is None


def test_variable_source_intervals_are_preserved_within_observer_bound():
    epsilon = SOURCE_INTERVAL_COMPARISON_EPSILON_S

    assert _source_interval_is_valid(MIN_SOURCE_INTERVAL_S - 0.5 * epsilon)
    assert _source_interval_is_valid(MAX_SOURCE_INTERVAL_S + 0.5 * epsilon)
    # Requests target 40 Hz, but Unity source time advances in 1 ms physics
    # steps and can differ from both the request period and arrival spacing.
    assert _source_interval_is_valid(0.013)
    assert _source_interval_is_valid(0.037)
    assert _source_interval_is_valid(0.050)
    assert _source_interval_is_valid(0.250)
    assert not _source_interval_is_valid(MIN_SOURCE_INTERVAL_S - 2.0 * epsilon)
    assert not _source_interval_is_valid(MAX_SOURCE_INTERVAL_S + 2.0 * epsilon)


def test_shutdown_does_not_promote_buffered_response_to_timing_fault(monkeypatch):
    monkeypatch.setattr(bridge_40hz, "_shutdown_requested", True)
    monkeypatch.setattr(bridge_40hz, "_timing_fault", False)
    bridge_40hz._trigger_timing_fault(
        "buffered response after orderly shutdown")
    assert bridge_40hz._timing_fault is False


def test_source_cadence_fault_keeps_transport_and_unity_step_evidence():
    detail = _format_source_cadence_fault_detail(
        source_dt=0.300,
        arrival_dt=0.025,
        previous_source_time=35.470,
        source_time=35.770,
        previous_source_step=35471,
        source_step=35771,
        previous_source_frame=114259,
        source_frame=114260,
        previous_telemetry_sequence=1420,
        telemetry_sequence=1421,
        request_sequence=1421,
        request_response_age_s=0.029,
    )

    assert "source sample interval 0.300000s" in detail
    assert "supported range [0.001, 0.250]s" in detail
    assert "bridge_arrival_dt=0.025000s" in detail
    assert "request_sequence=1421" in detail
    assert "request_response_age=0.029000s" in detail
    assert "physics_step_delta=300" in detail
    assert "render_frame_delta=1" in detail
    assert "telemetry_sequence_delta=1" in detail


def test_transport_fault_detail_includes_source_and_request_evidence():
    detail = _format_bridge_arrival_fault_detail(
        arrival_dt=0.151,
        source_dt=0.078,
        previous_source_time=8.125,
        source_time=8.203,
        previous_source_step=8126,
        source_step=8204,
        previous_source_frame=414,
        source_frame=416,
        previous_telemetry_sequence=328,
        telemetry_sequence=329,
        request_sequence=330,
        request_response_age_s=0.080,
    )

    assert "bridge transport interval 0.151000s" in detail
    assert "source_interval=0.078000s" in detail
    assert "physics_step_delta=78" in detail
    assert "telemetry_sequence=328->329" in detail
    assert "request_sequence=330" in detail
    assert "request_response_age=0.080000s" in detail
