"""Pure watchdog decision logic used by actuator integration tests."""

import math


def watchdog_failure(
    *,
    has_command: bool,
    command_age_sec: float,
    command_timeout_sec: float,
    has_odometry: bool,
    odom_age_sec: float,
    odom_timeout_sec: float,
) -> str | None:
    if not has_command:
        return 'no command'
    if not has_odometry:
        return 'no odometry'
    values = (
        command_age_sec,
        command_timeout_sec,
        odom_age_sec,
        odom_timeout_sec,
    )
    if not all(math.isfinite(value) for value in values):
        return 'invalid watchdog time'
    if command_age_sec > command_timeout_sec:
        return 'command timeout'
    if odom_age_sec > odom_timeout_sec:
        return 'odometry timeout'
    return None
