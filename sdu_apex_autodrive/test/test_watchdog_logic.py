from sdu_apex_autodrive.watchdog import watchdog_failure


def decision(**overrides):
    values = dict(
        has_command=True,
        command_age_sec=0.01,
        command_timeout_sec=0.25,
        has_odometry=True,
        odom_age_sec=0.01,
        odom_timeout_sec=0.25,
    )
    values.update(overrides)
    return watchdog_failure(**values)


def test_fresh_inputs_pass():
    assert decision() is None


def test_missing_command_neutralizes():
    assert decision(has_command=False) == 'no command'


def test_missing_odometry_neutralizes():
    assert decision(has_odometry=False) == 'no odometry'


def test_stale_command_neutralizes():
    assert decision(command_age_sec=0.3) == 'command timeout'


def test_stale_odometry_neutralizes():
    assert decision(odom_age_sec=0.3) == 'odometry timeout'
