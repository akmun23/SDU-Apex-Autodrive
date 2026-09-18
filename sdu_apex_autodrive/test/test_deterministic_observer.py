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


def test_reference_observer_loads_deployed_parameter_contract() -> None:
    params = (Path(__file__).resolve().parents[2] /
              "f1tenth_localization/config/sensor_odometry.yaml")
    observer = ReferenceObserver.from_yaml(params)

    assert observer.wheel_radius_m == pytest.approx(0.059)
    assert observer.wheel_speed_scale_speeds_mps[-1] == pytest.approx(14.0)
    assert observer.wheel_speed_scale_values[-1] == pytest.approx(0.9445)
    assert observer.use_coherent_packet_velocity_for_pose
    assert observer.use_kinematic_lateral_slip_model
    assert observer.lateral_velocity_yaw_rate_gain_m == pytest.approx(0.167)
    assert observer.lateral_velocity_speed_yaw_rate_gain_s == pytest.approx(-0.0063)
    assert observer.lateral_velocity_max_mps == pytest.approx(0.35)
    assert observer.lateral_velocity_reference_forward_offset_m == pytest.approx(0.15532)
    assert observer.imu_acceleration_reference_x_m == pytest.approx(0.15532)
    assert observer.turn_speed_bias_constant_mps == pytest.approx(-0.03)
    assert observer.max_integratable_gap_s == pytest.approx(0.250)
    assert observer.wheel_update_ax_abs_max_mps2 == pytest.approx(6.5)
    assert not observer.use_yaw_direction_encoder_side


@pytest.mark.parametrize(
    ("yaw_rate", "left_speed", "right_speed", "expected_speed"),
    ((0.8, 2.0, 3.0, 2.0), (-0.8, 3.0, 2.0, 2.0)),
)
def test_yaw_direction_selects_the_encoder_supported_by_track_holdout(
        yaw_rate: float, left_speed: float, right_speed: float,
        expected_speed: float) -> None:
    observer = ReferenceObserver(
        wheel_radius_m=0.05,
        wheel_speed_window_s=0.0,
        wheel_speed_scale_speeds_mps=(0.0, 10.0),
        wheel_speed_scale_values=(1.0, 1.0),
        use_yaw_direction_encoder_side=True,
    )
    first = pd.Series({
        "stamp_s": 1.0, "left_angle_rad": 0.0, "right_angle_rad": 0.0,
        "ax_mps2": 0.0, "ay_mps2": 0.0, "yaw_rate_radps": yaw_rate,
        "yaw_rad": 0.0,
    })
    dt = 0.025
    second = first.copy()
    second.update({
        "stamp_s": 1.0 + dt,
        "left_angle_rad": left_speed * dt / 0.05,
        "right_angle_rad": right_speed * dt / 0.05,
    })

    observer.update(first)
    result = observer.update(second)

    assert result.wheel_raw_mps == pytest.approx(expected_speed)


def test_kinematic_lateral_velocity_is_translated_to_rear_axle() -> None:
    observer = ReferenceObserver(
        use_kinematic_lateral_slip_model=True,
        lateral_velocity_yaw_rate_gain_m=0.167,
        lateral_velocity_speed_yaw_rate_gain_s=-0.0063,
        lateral_velocity_max_mps=0.35,
        lateral_velocity_reference_forward_offset_m=0.15532,
    )

    actual = observer.kinematic_lateral_velocity(0.8, 3.0)

    assert actual == pytest.approx(0.8 * (0.167 - 0.0063 * 3.0) - 0.8 * 0.15532)


def test_reconstruction_zeroes_absolute_imu_yaw_for_observer_replay(tmp_path: Path) -> None:
    rows = []
    for stamp, yaw in ((1.0, -1.57), (1.05, -1.52)):
        rows.extend([
            _event(stamp, "left_encoder", left_encoder_rad=0.0),
            _event(stamp, "right_encoder", right_encoder_rad=0.0),
            _event(stamp, "imu", ax_mps2=0.0, ay_mps2=0.0,
                   yaw_rate_radps=1.0, imu_yaw_rad=yaw),
            _event(stamp, "gt_odom", gt_speed_mps=0.0, gt_x_m=0.0,
                   gt_y_m=0.0, gt_yaw_rad=yaw),
        ])
    path = tmp_path / "absolute_yaw_events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    packets = reconstruct_packets(path)

    assert packets.yaw_rad.iloc[0] == pytest.approx(0.0)
    assert packets.yaw_rad.iloc[1] == pytest.approx(0.05, abs=1.0e-3)
