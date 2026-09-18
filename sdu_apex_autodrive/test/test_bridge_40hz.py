from pathlib import Path
from types import SimpleNamespace
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gevent.lock import Semaphore  # noqa: E402

import sdu_apex_autodrive.bridge_40hz as bridge_40hz  # noqa: E402


class FakeClock:
    def __init__(self):
        self.calls = 0

    def now(self):
        self.calls += 1
        return SimpleNamespace(to_msg=lambda: FakeStamp(100 + self.calls, 7))


class FakeStamp:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


class FakeNode:
    def __init__(self):
        self.clock = FakeClock()

    def get_clock(self):
        return self.clock


class NoUnityClockPacket(dict):
    """Fail the test if runtime code probes simulator-only time metadata."""
    forbidden = {
        "Simulation Time", "Simulation Frame", "Physics Step",
        "V1 Physics Step", "Telemetry Sequence",
        "V1 Bridge Request Sequence",
    }

    def get(self, key, default=None):
        assert key not in self.forbidden
        return super().get(key, default)


def _prepare_pending_packet(monkeypatch, faults):
    bridge_40hz._reset_request_pipeline()
    monkeypatch.setattr(bridge_40hz, "_timing_fault", False)
    monkeypatch.setattr(bridge_40hz, "_shutdown_requested", False)
    monkeypatch.setattr(bridge_40hz, "_reset_level", False)
    monkeypatch.setattr(bridge_40hz, "_trigger_timing_fault", faults.append)
    monkeypatch.setattr(
        bridge_40hz.official_bridge, "autodrive_bridge", FakeNode(), raising=False)
    bridge_40hz._client_connected.set()
    bridge_40hz._bootstrap_packet_seen.set()
    pool = Semaphore(1)
    assert pool.acquire(blocking=False)
    assert bridge_40hz._set_pending_request(
        pool,
        bridge_40hz._connection_generation,
        17,
        {"V1 Throttle": "0.0", "V1 Steering": "0.0", "V1 Reset": "False"},
    )
    return pool


def test_stock_protocol_uses_bounded_fifo_window_without_response_ids():
    assert bridge_40hz.MAX_OUTSTANDING_REQUESTS == 7
    assert bridge_40hz.MAX_RESPONSE_WAIT_S == 0.150
    assert (bridge_40hz.MAX_OUTSTANDING_REQUESTS *
            bridge_40hz.MIN_REQUEST_SPACING_S >=
            bridge_40hz.MAX_RESPONSE_WAIT_S)


def test_multiple_stock_responses_keep_request_fifo_association(monkeypatch):
    faults = []
    bridge_40hz._reset_request_pipeline()
    monkeypatch.setattr(bridge_40hz, "_timing_fault", False)
    monkeypatch.setattr(bridge_40hz, "_shutdown_requested", False)
    monkeypatch.setattr(bridge_40hz, "_reset_level", False)
    monkeypatch.setattr(bridge_40hz, "_trigger_timing_fault", faults.append)
    monkeypatch.setattr(
        bridge_40hz.official_bridge, "autodrive_bridge", FakeNode(), raising=False)
    bridge_40hz._client_connected.set()
    bridge_40hz._bootstrap_packet_seen.set()
    pool = Semaphore(2)
    assert pool.acquire(blocking=False)
    assert pool.acquire(blocking=False)
    generation = bridge_40hz._connection_generation
    assert bridge_40hz._set_pending_request(
        pool, generation, 17, {"V1 Throttle": "0.1"})
    assert bridge_40hz._set_pending_request(
        pool, generation, 18, {"V1 Throttle": "0.2"})

    for expected_sequence in (17, 18):
        assert bridge_40hz._capture_competition_packet({
            "V1 Position": "1 2 3",
        })
        assert bridge_40hz._active_request_sequence == expected_sequence
        assert bridge_40hz._active_command["V1 Throttle"] == (
            "0.1" if expected_sequence == 17 else "0.2")
        bridge_40hz._finish_packet()
        assert pool.acquire(blocking=False)

    assert not bridge_40hz._pending_requests
    assert not faults


def test_connect_bootstrap_opens_request_gate_without_becoming_a_sample(monkeypatch):
    faults = []
    bridge_40hz._reset_request_pipeline()
    monkeypatch.setattr(bridge_40hz, "_timing_fault", False)
    monkeypatch.setattr(bridge_40hz, "_trigger_timing_fault", faults.append)
    bridge_40hz._client_connected.set()

    assert not bridge_40hz._capture_competition_packet({"bootstrap": "only"})
    assert bridge_40hz._bootstrap_packet_seen.is_set()
    assert not bridge_40hz._pending_requests
    assert not faults


def test_stock_packet_needs_no_unity_clock_and_uses_receive_stamp(monkeypatch):
    faults = []
    pool = _prepare_pending_packet(monkeypatch, faults)
    packet = NoUnityClockPacket({
        "V1 Encoder Angles": "1.2 1.3",
        "V1 Position": "4 5 6",
        "V1 Orientation Quaternion": "0 0 0 1",
        "V1 Orientation Euler Angles": "0 0 0",
        "V1 Linear Velocity": "1 0 0",
        "V1 Angular Velocity": "0 0 0.1",
        "V1 Linear Acceleration": "0 0 0",
        "V1 Throttle": "0.0",
        "V1 Steering": "0.0",
        "V1 Collisions": "0",
    })

    assert bridge_40hz._capture_competition_packet(packet)
    assert bridge_40hz._active_request_stamp.sec == 101
    assert bridge_40hz._active_packet_stamp.sec == 102
    assert bridge_40hz._active_packet_arrival_ns is not None
    assert bridge_40hz._active_request_sequence == 17
    assert bridge_40hz._validate_received_packet()
    assert bridge_40hz._valid_packet_count == 1
    assert not faults

    # Exercise the actual message-constructor patch: it must use packet
    # receive time, not request-send time or any simulator-provided clock.
    class Message:
        def __init__(self):
            self.header = SimpleNamespace(stamp=FakeStamp(1, 0))

    monkeypatch.setattr(
        bridge_40hz.official_bridge, "create_odom_msg", lambda: Message(),
        raising=False)
    monkeypatch.setattr(
        bridge_40hz.official_bridge, "publish_encoder_data", lambda *_a, **_k: None,
        raising=False)
    for name in (
            "create_joint_state_msg", "create_imu_msg", "create_laserscan_msg",
            "create_image_msg", "create_tf_msg"):
        if hasattr(bridge_40hz.official_bridge, name):
            monkeypatch.setattr(
                bridge_40hz.official_bridge, name,
                getattr(bridge_40hz.official_bridge, name))
    monkeypatch.setattr(bridge_40hz, "_packet_timestamp_patch_installed", False)
    monkeypatch.setattr(bridge_40hz, "_packet_stamp", None)
    bridge_40hz._install_packet_timestamp_patch()
    published_message = bridge_40hz.official_bridge.create_odom_msg()
    assert published_message.header.stamp.sec == 102
    assert published_message.header.stamp.sec != bridge_40hz._active_request_stamp.sec
    bridge_40hz._finish_packet()
    assert pool.acquire(blocking=False)


def test_unsolicited_packet_after_bootstrap_fails_closed(monkeypatch):
    faults = []
    bridge_40hz._reset_request_pipeline()
    monkeypatch.setattr(bridge_40hz, "_timing_fault", False)
    monkeypatch.setattr(bridge_40hz, "_trigger_timing_fault", faults.append)
    bridge_40hz._client_connected.set()
    bridge_40hz._bootstrap_packet_seen.set()

    assert not bridge_40hz._capture_competition_packet({"unexpected": "packet"})
    assert faults == ["simulator packet arrived without a pending Bridge request"]


def test_delayed_packet_is_measured_not_retimed_to_nominal_25ms(monkeypatch):
    faults = []
    _prepare_pending_packet(monkeypatch, faults)
    monkeypatch.setattr(bridge_40hz, "_last_packet_arrival_ns", 1_000_000_000)
    monkeypatch.setattr(bridge_40hz, "_active_packet_arrival_ns", 1_030_000_000)

    assert bridge_40hz._validate_received_packet()
    assert bridge_40hz._last_packet_arrival_ns == 1_030_000_000
    assert not faults


def test_packet_gap_over_transport_limit_faults(monkeypatch):
    faults = []
    _prepare_pending_packet(monkeypatch, faults)
    monkeypatch.setattr(bridge_40hz, "_last_packet_arrival_ns", 1_000_000_000)
    monkeypatch.setattr(bridge_40hz, "_active_packet_arrival_ns", 1_151_000_000)

    assert not bridge_40hz._validate_received_packet()
    assert len(faults) == 1
    assert "arrival interval exceeded 0.150s" in faults[0]
