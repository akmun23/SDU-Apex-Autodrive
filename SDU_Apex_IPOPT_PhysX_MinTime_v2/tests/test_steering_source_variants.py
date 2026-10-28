"""Verify actuator constants across source-plant revisions without inventing defaults."""
import math
from pathlib import Path

import pytest

from source_vehicle import _cpp_number, _current_steering_parameters


def write_native(tmp_path: Path, limit: str = '30.0f', rate: str = '183.346f'):
    native = tmp_path / 'f1tenth_localization/src/physx_odom_runtime.cpp'
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text(
        f'constexpr float kSteeringLimitDeg = {limit};\n'
        f'constexpr float kSteeringRateDegPerS = {rate};\n')


def test_source_cpp_float_and_arithmetic_literals():
    raw = '''
    // constexpr double kMaxSteeringRad = 5.0;
    constexpr double kMaxSteeringRad = (30.0 * kPi) / 180.0;
    constexpr float kSteeringRateRadps = 3.2f;
    '''
    assert _cpp_number(raw, 'kMaxSteeringRad') == pytest.approx(math.pi / 6)
    assert _cpp_number(raw, 'kSteeringRateRadps') == pytest.approx(3.2)


def test_source_missing_steering_declarations_uses_native_actuator(tmp_path):
    write_native(tmp_path)
    limit, rate, prov = _current_steering_parameters(tmp_path, '// no steering constants')
    assert limit == pytest.approx(math.pi/6)
    assert rate == pytest.approx(math.radians(183.346))
    assert prov['limit_source'].startswith('physx_odom_runtime')
    assert prov['rate_source'].startswith('physx_odom_runtime')


def test_source_literals_must_agree_with_native_actuator(tmp_path):
    write_native(tmp_path)
    raw = '''
    constexpr double kMaxSteeringRad = 0.52359877559829887308;
    constexpr double kSteeringRateRadps = 3.2;
    '''
    limit, rate, prov = _current_steering_parameters(tmp_path, raw)
    assert limit == pytest.approx(math.pi/6)
    assert rate == pytest.approx(3.2)
    assert prov['limit_source'].startswith('unity_source_plant')


def test_source_disagreement_rejected(tmp_path):
    write_native(tmp_path)
    with pytest.raises(RuntimeError, match='disagrees'):
        _current_steering_parameters(tmp_path, 'constexpr double kMaxSteeringRad = 0.3;')


def test_unknown_constant_expression_is_not_guessed(tmp_path):
    write_native(tmp_path)
    with pytest.raises(RuntimeError, match='Cannot resolve'):
        _current_steering_parameters(tmp_path,
            'constexpr double kMaxSteeringRad = steeringAngleFromConfig;')


def test_missing_native_source_is_not_silently_defaulted(tmp_path):
    with pytest.raises(RuntimeError, match='Cannot verify'):
        _current_steering_parameters(tmp_path, '')
