"""Reconstruct coherent AutoDRIVE packets from recorder source-event rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_EVENTS = ("left_encoder", "right_encoder", "imu", "gt_odom")
_IMU_YAW_MAX_STEP_RAD = 0.30


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _normalize_imu_yaw(packets: pd.DataFrame) -> None:
    """Mirror sensor_odometry_node's raw-IMU-to-local-yaw conversion.

    The deployed observer starts its odom frame at the first coherent IMU
    sample. Replaying raw ``imu_yaw_rad`` directly would rotate the replay
    pose by the simulator's absolute spawn heading and make pose scoring
    invalid even when the speed replay is correct.
    """
    if packets.empty:
        return
    reference = float(packets.iloc[0].yaw_rad)
    previous_raw_relative = 0.0
    continuous = 0.0
    previous_stamp = float(packets.iloc[0].stamp_s)
    normalized = [continuous]
    for index in range(1, len(packets)):
        stamp = float(packets.iloc[index].stamp_s)
        raw_yaw = float(packets.iloc[index].yaw_rad)
        yaw_rate = float(packets.iloc[index].yaw_rate_radps)
        dt = stamp - previous_stamp
        raw_relative = _wrap(raw_yaw - reference)
        raw_delta = _wrap(raw_relative - previous_raw_relative)
        if 0.0 < dt <= 0.5:
            continuous = _wrap(continuous + yaw_rate * dt)
            if abs(raw_delta) <= _IMU_YAW_MAX_STEP_RAD:
                continuous = _wrap(continuous + _wrap(raw_relative - continuous))
            else:
                reference = _wrap(raw_yaw - continuous)
                previous_raw_relative = continuous
        if abs(raw_delta) <= _IMU_YAW_MAX_STEP_RAD:
            previous_raw_relative = raw_relative
        previous_stamp = stamp
        normalized.append(continuous)
    packets["yaw_rad"] = normalized


def _value(row: pd.Series, name: str) -> float:
    value = pd.to_numeric(row.get(name, np.nan), errors="coerce")
    return float(value) if np.isfinite(value) else float("nan")


def reconstruct_packets(csv_path: str | Path) -> pd.DataFrame:
    """Return one row per exact source timestamp with all required sensors.

    The recorder callback row is only used for the fields belonging to its
    own source event.  No arbitrary callback snapshot is forward-filled into
    another event.  Coherence statistics are available in ``df.attrs``.
    """
    raw = pd.read_csv(csv_path)
    required = {"source_event_name", "source_event_stamp_s"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"missing source-event columns: {sorted(missing)}")
    raw = raw[raw.source_event_name.isin(_EVENTS)].copy()
    raw["source_event_stamp_s"] = pd.to_numeric(
        raw["source_event_stamp_s"], errors="coerce")
    raw = raw[np.isfinite(raw.source_event_stamp_s)]

    duplicate_events = 0
    frames: list[pd.DataFrame] = []
    for event in _EVENTS:
        selected = raw[raw.source_event_name.eq(event)]
        duplicate_events += max(0, len(selected) - selected.source_event_stamp_s.nunique())
        selected = selected.drop_duplicates("source_event_stamp_s", keep="last").set_index(
            "source_event_stamp_s")
        if event == "left_encoder":
            names = {"left_encoder_rad": "left_angle_rad"}
        elif event == "right_encoder":
            names = {"right_encoder_rad": "right_angle_rad"}
        elif event == "imu":
            names = {
                "ax_mps2": "ax_mps2", "ay_mps2": "ay_mps2",
                "yaw_rate_radps": "yaw_rate_radps", "imu_yaw_rad": "yaw_rad",
            }
        else:
            names = {
                "gt_speed_mps": "gt_speed_mps", "gt_x_m": "gt_x_m",
                "gt_y_m": "gt_y_m", "gt_yaw_rad": "gt_yaw_rad",
                "gt_vx_mps": "gt_vx_mps", "gt_vy_mps": "gt_vy_mps",
                "gt_longitudinal_accel_mps2": "gt_longitudinal_accel_mps2",
            }
        available = {source: target for source, target in names.items()
                     if source in selected.columns}
        frame = selected[list(available)].rename(columns=available)
        frames.append(frame)

    all_stamps = pd.Index([], dtype=float)
    for frame in frames:
        all_stamps = all_stamps.union(frame.index)
    packets = pd.DataFrame(index=all_stamps.sort_values())
    for frame in frames:
        packets = packets.join(frame, how="left")
    packets.index.name = "stamp_s"
    packets = packets.reset_index()
    complete_mask = packets[[
        "left_angle_rad", "right_angle_rad", "ax_mps2", "ay_mps2",
        "yaw_rate_radps", "yaw_rad", "gt_speed_mps",
    ]].notna().all(axis=1)
    incomplete_packets = int((~complete_mask).sum())
    packets = packets.loc[complete_mask].reset_index(drop=True)
    for column in packets.columns:
        if column != "stamp_s":
            packets[column] = pd.to_numeric(packets[column], errors="coerce")
    _normalize_imu_yaw(packets)
    packets.attrs["coherence"] = {
        "input_source_rows": int(len(raw)),
        "unique_source_timestamps": int(raw.source_event_stamp_s.nunique()),
        "complete_packets": int(len(packets)),
        "incomplete_packets": int(incomplete_packets),
        "duplicate_event_rows": int(duplicate_events),
    }
    return packets
