"""Regression: local MPC source refers to UNITY_SOURCE_* named aliases."""
import math
from pathlib import Path
import pytest
from source_vehicle import _cpp_number, _current_steering_parameters


def native_fixture(root):
    p = root / 'f1tenth_localization/src/physx_odom_runtime.cpp'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('constexpr float kSteeringLimitDeg = 30.0f;\n'
                 'constexpr float kSteeringRateDegPerS = 183.346f;\n')


def test_resolved_cpp_named_steering_aliases():
    code = '''
    #define UNITY_SOURCE_MAX_STEERING_RAD (30.0 * kPi / 180.0)
    constexpr double UNITY_SOURCE_STEERING_RATE_RADPS = 3.2;
    constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
    constexpr double kSteeringRateRadps = UNITY_SOURCE_STEERING_RATE_RADPS;
    '''
    assert _cpp_number(code, 'kMaxSteeringRad') == pytest.approx(math.pi/6)
    assert _cpp_number(code, 'kSteeringRateRadps') == pytest.approx(3.2)


def test_both_unresolved_known_aliases_use_explicit_native_with_warning(tmp_path):
    native_fixture(tmp_path)
    code = '''
    constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
    constexpr double kSteeringRateRadps = UNITY_SOURCE_STEERING_RATE_RADPS;
    '''
    with pytest.warns(RuntimeWarning, match='cannot verify plant alias parity') as recorded:
        limit, rate, metadata = _current_steering_parameters(tmp_path, code)
    assert len(recorded) == 2
    assert limit == pytest.approx(math.pi/6)
    assert rate == pytest.approx(math.radians(183.346))
    assert metadata['limit_alias_unverified'] == 'UNITY_SOURCE_MAX_STEERING_RAD'
    assert metadata['rate_alias_unverified'] == 'UNITY_SOURCE_STEERING_RATE_RADPS'


def test_resolve_known_alias_from_included_header(tmp_path):
    native_fixture(tmp_path)
    inc = tmp_path / 'f1tenth_mpc/include/local_actuators.h'
    inc.parent.mkdir(parents=True, exist_ok=True)
    inc.write_text('#define UNITY_SOURCE_MAX_STEERING_RAD (30.0 * kPi / 180.0)\n')
    code = '''
    #include "local_actuators.h"
    constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
    constexpr double kSteeringRateRadps = 3.2;
    '''
    lim, rate, prov = _current_steering_parameters(tmp_path, code)
    assert lim == pytest.approx(math.pi/6)
    assert rate == pytest.approx(3.2)
    assert prov['limit_source'].startswith('unity_source_plant')
    assert 'limit_alias_unverified' not in prov


def test_wrong_known_alias_value_still_rejected(tmp_path):
    native_fixture(tmp_path)
    code = '''
    #define UNITY_SOURCE_MAX_STEERING_RAD 0.2
    constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
    '''
    with pytest.raises(RuntimeError, match='disagrees'):
        _current_steering_parameters(tmp_path, code)


def test_unknown_unresolved_steering_alias_rejected(tmp_path):
    native_fixture(tmp_path)
    code = 'constexpr double kMaxSteeringRad = SOME_UNKNOWN_STEERING_NAME;'
    with pytest.raises(RuntimeError, match='Cannot resolve'):
        _current_steering_parameters(tmp_path, code)


def test_recursive_cpp_alias_cycle_rejected():
    code = '''
    constexpr double AAA = BBB;
    constexpr double BBB = AAA;
    '''
    with pytest.raises(RuntimeError, match='Circular'):
        _cpp_number(code, 'AAA')
