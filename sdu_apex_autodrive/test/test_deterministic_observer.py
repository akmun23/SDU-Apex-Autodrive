"""Pure-data tests for deterministic observer packet reconstruction and replay."""

from pathlib import Path

import pandas as pd
import pytest

from sdu_apex_autodrive.odometry_analysis.packet_reconstruction import (
    reconstruct_packets,
)
from sdu_apex_autodrive.odometry_analysis.reference_observer import (
    ReferenceObserver,
)


def _event(stamp: float, name: str, **values: float) -> dict[str, object]:
    row: dict[str, object] = {
        "source_event_stamp_s": stamp,
        "source_event_name": name,
    }
    row.update(values)
    return row


def test_reconstruction_requires_exact_complete_packet(tmp_path: Path) -> None:
    rows = [
        # Deliberately permuted callback order: a callback snapshot/forward
        # fill would associate these values by arrival order, not timestamp.
        _event(1.000, "gt_odom", gt_speed_mps=0.0),
        _event(1.000, "imu", ax_mps2=0.0, ay_mps2=0.0,
               yaw_rate_radps=0.0, imu_yaw_rad=0.0),
        _event(1.000, "right_encoder", right_encoder_rad=0.0),
        _event(1.000, "left_encoder", left_encoder_rad=0.0),
        _event(1.025, "right_encoder", right_encoder_rad=0.1),
        _event(1.025, "left_encoder", left_encoder_rad=0.1),
        _event(1.025, "imu", ax_mps2=0.0, ay_mps2=0.0,
               yaw_rate_radps=0.0, imu_yaw_rad=0.0),
        # No ground-truth event at 1.025: this packet must not be completed
        # from the previous timestamp.
    ]
    path = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    packets = reconstruct_packets(path)

    assert packets["stamp_s"].tolist() == [1.0]
    assert packets.attrs["coherence"] == {
        "input_source_rows": 7,
        "unique_source_timestamps": 2,
        "complete_packets": 1,
        "incomplete_packets": 1,
        "duplicate_event_rows": 0,
    }


def test_reference_replay_updates_deterministically() -> None:
    observer = ReferenceObserver()
    first = pd.Series({
        "stamp_s": 1.000, "left_angle_rad": 0.0, "right_angle_rad": 0.0,
        "ax_mps2": 0.0, "ay_mps2": 0.0, "yaw_rate_radps": 0.0,
        "yaw_rad": 0.0,
    })
    second = first.copy()
    second.update({
        "stamp_s": 1.025,
        "left_angle_rad": 0.32,
        "right_angle_rad": 0.32,
        "ax_mps2": 30.0,
    })

    initial = observer.update(first)
    update = observer.update(second)

    assert initial.speed_mps == 0.0
    assert update.dt_s == pytest.approx(0.025)
    assert update.speed_mps > 0.0
    assert update.body_u_mps == update.speed_mps
    assert update.body_v_mps == 0.0
