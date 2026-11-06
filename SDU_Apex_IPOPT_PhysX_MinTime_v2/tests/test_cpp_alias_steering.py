"""Regression: local MPC source refers to UNITY_SOURCE_* named aliases."""
import math
from pathlib import Path
import pytest
from source_vehicle import _cpp_number, _current_steering_parameters


def test_resolved_cpp_named_steering_aliases():
    code = '''
    #define UNITY_SOURCE_MAX_STEERING_RAD (30.0 * kPi / 180.0)
    constexpr double UNITY_SOURCE_STEERING_RATE_RADPS = 3.2;
    constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
    constexpr double kSteeringRateRadps = UNITY_SOURCE_STEERING_RATE_RADPS;
    '''
    assert _cpp_number(code, 'kMaxSteeringRad') == pytest.approx(math.pi/6)
    assert _cpp_number(code, 'kSteeringRateRadps') == pytest.approx(3.2)


def test_unresolved_known_aliases_fail_closed(tmp_path):
    code = '''
    constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
    constexpr double kSteeringRateRadps = UNITY_SOURCE_STEERING_RATE_RADPS;
    '''
    with pytest.raises(RuntimeError, match='Cannot resolve'):
        _current_steering_parameters(tmp_path, code)


def test_resolve_known_alias_from_included_header(tmp_path):
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


def test_disagreement_with_implemented_update_rejected(tmp_path):
    code = '''
    #define UNITY_SOURCE_MAX_STEERING_RAD 0.2
    constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
    constexpr double kSteeringRateRadps = 3.2;
    const double threshold = last_internal_deg / 30.0;
    const double target_deg = -command_norm * 30.0;
    '''
    with pytest.raises(RuntimeError, match='disagrees'):
        _current_steering_parameters(tmp_path, code)


def test_unknown_unresolved_steering_alias_rejected(tmp_path):
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
