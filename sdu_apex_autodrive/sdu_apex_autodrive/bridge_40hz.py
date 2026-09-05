"""Run the official AutoDRIVE bridge with an independent command clock.

The shipped simulator emits one telemetry packet after each ``Bridge``
command.  The official bridge sends that command at the end of its telemetry
callback, so decoding images and publishing ROS messages throttles the whole
simulator feedback loop.  This wrapper keeps the official decoder and topic
contract, but repeats the latest command at a fixed rate from a separate
thread.  Incoming telemetry is still published only when the simulator sends
it; no ROS sample is fabricated here.
"""

from __future__ import annotations

import os
import threading
import time
from copy import copy
from typing import Any

from autodrive_roboracer import autodrive_bridge as official_bridge


DEFAULT_RATE_HZ = 40.0
_command_lock = threading.Lock()
_latest_command: dict[str, str] = {
    "V1 Throttle": "0.0",
    "V1 Steering": "0.0",
    "V1 Reset": "False",
}
_reset_level = False
_stop_sender = threading.Event()
_packet_stamp_lock = threading.Lock()
_packet_stamp: Any = None
_packet_timestamp_patch_installed = False


def _copy_stamp(stamp: Any) -> Any:
    """Copy a ROS time value without sharing a mutable message object."""
    return copy(stamp)


def _install_packet_timestamp_patch() -> None:
    """Give every ROS message from one simulator packet one source stamp.

    The official bridge calls ``get_clock().now()`` independently in every
    message-construction helper.  Since the helpers are invoked sequentially,
    the encoder and IMU headers can differ just enough for a source-time
    synchronizer to treat them as neighboring samples.  The websocket packet
    is the actual common sample boundary, so retain the first sensor header
    stamp and apply it to the rest of that packet.  This changes timestamps
    only; incoming simulator values and the native packet cadence are
    untouched.
    """
    global _packet_timestamp_patch_installed
    if _packet_timestamp_patch_installed:
        return

    def stamped_creator(original: Any) -> Any:
        def create(*args: Any, **kwargs: Any) -> Any:
            global _packet_stamp
            message = original(*args, **kwargs)
            if not hasattr(message, "header"):
                return message
            with _packet_stamp_lock:
                if _packet_stamp is None:
                    _packet_stamp = _copy_stamp(message.header.stamp)
                else:
                    message.header.stamp = _copy_stamp(_packet_stamp)
            return message
        return create

    for name in (
        "create_joint_state_msg", "create_imu_msg", "create_odom_msg",
        "create_laserscan_msg", "create_image_msg", "create_tf_msg",
    ):
        setattr(official_bridge, name, stamped_creator(
            getattr(official_bridge, name)))

    original_encoder_publish = official_bridge.publish_encoder_data

    def begin_packet(*args: Any, **kwargs: Any) -> Any:
        global _packet_stamp
        with _packet_stamp_lock:
            _packet_stamp = None
        return original_encoder_publish(*args, **kwargs)

    original_collision_publish = official_bridge.publish_collision_count_data

    def end_packet(*args: Any, **kwargs: Any) -> Any:
        global _packet_stamp
        result = original_collision_publish(*args, **kwargs)
        with _packet_stamp_lock:
            _packet_stamp = None
        return result

    official_bridge.publish_encoder_data = begin_packet
    official_bridge.publish_collision_count_data = end_packet
    _packet_timestamp_patch_installed = True


def _rate_hz() -> float:
    value = os.environ.get("AUTODRIVE_BRIDGE_RATE_HZ", str(DEFAULT_RATE_HZ))
    try:
        rate = float(value)
    except ValueError as exc:
        raise RuntimeError("AUTODRIVE_BRIDGE_RATE_HZ must be a positive number") from exc
    if rate <= 0.0:
        raise RuntimeError("AUTODRIVE_BRIDGE_RATE_HZ must be a positive number")
    return rate


def _remember_command(data: Any) -> None:
    global _reset_level
    if not isinstance(data, dict) or "V1 Throttle" not in data:
        return
    command = {
        "V1 Throttle": str(data.get("V1 Throttle", "0.0")),
        "V1 Steering": str(data.get("V1 Steering", "0.0")),
        "V1 Reset": str(data.get("V1 Reset", "False")),
    }
    with _command_lock:
        # The current simulator consumes V1 Reset as a level on the Bridge
        # command, not as a one-shot edge.  Keep the level asserted while the
        # diagnostics harness is waiting for a reset acknowledgement.  The
        # harness only enters this state after braking has confirmed a stop,
        # so repeated reset commands cannot move a car that is still driving.
        _reset_level = command["V1 Reset"].strip().lower() == "true"
        _latest_command.update(command)


def _run_command_sender(original_emit: Any, rate_hz: float) -> None:
    period = 1.0 / rate_hz
    next_send = time.monotonic()
    while not _stop_sender.is_set():
        now = time.monotonic()
        wait_time = next_send - now
        if wait_time > 0.0:
            _stop_sender.wait(wait_time)
            continue

        with _command_lock:
            command = dict(_latest_command)
            command["V1 Reset"] = "True" if _reset_level else "False"
        try:
            # Match the official bridge's broadcast semantics.  The saved
            # bound method bypasses the interception wrapper below.
            original_emit("Bridge", data=command)
        except Exception as exc:  # keep the sender alive across reconnects
            print(f"[autodrive_bridge_40hz] command send failed: {exc}")

        next_send += period
        if next_send < now - period:
            # Do not send a burst after a paused or reconnecting process.
            next_send = now + period


def main() -> None:
    rate_hz = _rate_hz()
    _install_packet_timestamp_patch()
    original_emit = official_bridge.sio.emit

    def intercept_emit(event: str, data: Any = None, *args: Any, **kwargs: Any) -> Any:
        if event == "Bridge":
            _remember_command(data)
            return None
        return original_emit(event, data, *args, **kwargs)

    official_bridge.sio.emit = intercept_emit
    _stop_sender.clear()
    sender = threading.Thread(
        target=_run_command_sender,
        args=(original_emit, rate_hz),
        name="autodrive-bridge-command-clock",
        daemon=True,
    )
    sender.start()
    print(f"[autodrive_bridge_40hz] command pacing enabled at {rate_hz:g} Hz")
    try:
        official_bridge.main()
    finally:
        _stop_sender.set()
        sender.join(timeout=max(1.0, 2.0 / rate_hz))
        official_bridge.sio.emit = original_emit


if __name__ == "__main__":
    main()
