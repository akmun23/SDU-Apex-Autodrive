import pytest
import casadi as ca

from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.unity_wheel_model import (
    physx_no_drive_wheel_speed_interval_casadi,
)
from tools.vehicle_dynamics_learning.physx_no_drive_source import (
    no_drive_substep_count,
    no_drive_wheel_speed_step,
)


def test_unity_player_substep_threshold_uses_strict_less_than():
    assert no_drive_substep_count(4.999) == 5
    assert no_drive_substep_count(-4.999) == 5
    assert no_drive_substep_count(5.0) == 1
    assert no_drive_substep_count(-5.0) == 1


def test_no_drive_implicit_euler_matches_physx_equation():
    omega, drive, brake, tire, inertia, damping, dt = (
        10.0, 0.25, 0.0, -0.1, 0.0001897145, 0.25, 0.001)
    actual = no_drive_wheel_speed_step(
        omega, drive, brake, tire, inertia, damping, dt)
    dt_over_inertia = dt / inertia
    expected = (omega + dt_over_inertia * (drive + tire)) / (
        1.0 + damping * dt_over_inertia)
    assert actual == pytest.approx(expected, rel=2e-6)


def test_brake_torque_opposes_forward_and_reverse_rotation():
    forward = no_drive_wheel_speed_step(
        10.0, 0.0, 1.0, 0.0, 0.01, 0.0, 0.001)
    reverse = no_drive_wheel_speed_step(
        -10.0, 0.0, 1.0, 0.0, 0.01, 0.0, 0.001)
    assert forward < 10.0
    assert reverse > -10.0


def test_braked_wheel_is_locked_when_one_substep_crosses_zero():
    assert no_drive_wheel_speed_step(
        1.0, 0.0, 100.0, 0.0, 0.01, 0.0, 0.001) == 0.0
    assert no_drive_wheel_speed_step(
        -1.0, 0.0, 100.0, 0.0, 0.01, 0.0, 0.001) == 0.0


def test_unbraked_wheel_can_change_rotation_sign():
    result = no_drive_wheel_speed_step(
        1.0, -100.0, 0.0, 0.0, 0.01, 0.0, 0.001)
    assert result == pytest.approx(-9.0)


@pytest.mark.parametrize("omega,drive,brake,tire,duration,substeps", [
    (75.0, 12.0, 0.0, -3.5, 0.025, 1),
    (120.0, 65.0, 0.0, -28.0, 0.100, 1),
    (80.0, 0.0, 428.0, 6.0, 0.005, 5),
    (30.0, 0.0, 428.0, 0.0, 0.050, 5),
    (0.0, 10.0, 0.0, -2.0, 0.025, 5),
])
def test_casadi_interval_composes_source_implicit_steps(
        omega, drive, brake, tire, duration, substeps):
    inertia, damping, physics_dt = 0.0001897145, 0.25, 0.001
    omega_symbol = ca.MX.sym("omega")
    dt_symbol = ca.MX.sym("dt")
    step = physx_no_drive_wheel_speed_interval_casadi(
        omega_symbol, drive, brake, tire, inertia, damping, dt_symbol,
        physics_dt, substeps)
    function = ca.Function("wheel_step", [omega_symbol, dt_symbol], [step])
    actual = float(function(omega, duration))

    expected = omega
    internal_dt = physics_dt / substeps
    full_steps = int(round(duration / internal_dt))
    for _ in range(full_steps):
        expected = no_drive_wheel_speed_step(
            expected, drive, brake, tire, inertia, damping, internal_dt)
    assert actual == pytest.approx(expected, abs=2.0e-3)


@pytest.mark.parametrize("args", [
    (0.0, 0.0, -1.0, 0.0, 0.1, 0.0, 0.001),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001),
    (0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
])
def test_rejects_invalid_wheel_step_parameters(args):
    with pytest.raises(ValueError):
        no_drive_wheel_speed_step(*args)
