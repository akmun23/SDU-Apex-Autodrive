"""Reconstruct coherent AutoDRIVE packets from recorder source-event rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_EVENTS = ("left_encoder", "right_encoder", "imu", "gt_odom")


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
    packets.attrs["coherence"] = {
        "input_source_rows": int(len(raw)),
        "unique_source_timestamps": int(raw.source_event_stamp_s.nunique()),
        "complete_packets": int(len(packets)),
        "incomplete_packets": int(incomplete_packets),
        "duplicate_event_rows": int(duplicate_events),
    }
    return packets
