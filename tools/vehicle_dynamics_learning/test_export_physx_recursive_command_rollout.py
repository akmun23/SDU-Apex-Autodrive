import numpy as np
import pandas as pd
import pytest

from tools.vehicle_dynamics_learning.export_physx_recursive_command_rollout import (
    _controller_boundary_wheel_commands,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    MOTOR_TORQUE_NM,
)


def _held_command_rows(steering=0.0, throttle=0.0):
    return pd.DataFrame({
        "steering_command": np.full(25, steering),
        "throttle_command": np.full(25, throttle),
        "fl_steer_deg": np.zeros(25),
        "fr_steer_deg": np.zeros(25),
        "rl_steer_deg": np.zeros(25),
        "rr_steer_deg": np.zeros(25),
        "fl_motor_torque_nm": np.zeros(25),
        "fr_motor_torque_nm": np.zeros(25),
        "rl_motor_torque_nm": np.zeros(25),
        "rr_motor_torque_nm": np.zeros(25),
        "fl_brake_torque_nm": np.full(25, MOTOR_TORQUE_NM),
        "fr_brake_torque_nm": np.full(25, MOTOR_TORQUE_NM),
        "rl_brake_torque_nm": np.full(25, MOTOR_TORQUE_NM),
        "rr_brake_torque_nm": np.full(25, MOTOR_TORQUE_NM),
    })


def test_controller_boundary_expands_zero_command_to_exact_source_brake():
    rows = _held_command_rows()
    previous = pd.Series({"fl_steer_deg": 0.0, "fr_steer_deg": 0.0})

    commands, summary = _controller_boundary_wheel_commands(
        rows, previous, ticks_per_command=25)

    assert len(commands) == 25
    assert summary["control_rate_hz"] == 40.0
    assert summary["held_command_intervals"] == 1
    assert summary["max_abs_wheel_steering_reconstruction_error_deg"] == 0.0
    assert summary["max_abs_wheel_drive_torque_reconstruction_error_nm"] == 0.0
    assert summary["max_abs_wheel_brake_torque_reconstruction_error_nm"] == 0.0
    assert all(wheel["steer_angle_rad"] == 0.0 for wheel in commands[0])
    assert all(wheel["drive_torque_nm"] == 0.0 for wheel in commands[0])
    assert all(wheel["brake_torque_nm"] == MOTOR_TORQUE_NM
               for wheel in commands[0])


def test_controller_boundary_rejects_unheld_input_within_control_interval():
    rows = _held_command_rows()
    rows.loc[7, "steering_command"] = 0.1
    previous = pd.Series({"fl_steer_deg": 0.0, "fr_steer_deg": 0.0})

    with pytest.raises(ValueError, match="not held"):
        _controller_boundary_wheel_commands(
            rows, previous, ticks_per_command=25)


def test_controller_boundary_rejects_partial_command_interval():
    rows = _held_command_rows().iloc[:-1]
    previous = pd.Series({"fl_steer_deg": 0.0, "fr_steer_deg": 0.0})

    with pytest.raises(ValueError, match="whole command intervals"):
        _controller_boundary_wheel_commands(
            rows, previous, ticks_per_command=25)
