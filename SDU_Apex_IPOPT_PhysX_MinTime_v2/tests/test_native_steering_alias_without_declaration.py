"""Regression for user checkout with no kSteeringLimitDeg native declaration."""
import math
import pytest
from source_vehicle import _current_steering_parameters, _literal_plant_actuator


def write_native(tmp_path, content):
    path = tmp_path / 'f1tenth_localization/src/physx_odom_runtime.cpp'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


PLANT_WITH_ALIASES = '''
constexpr double kDt = 0.001;
constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
constexpr double kSteeringRateRadps = UNITY_SOURCE_STEERING_RATE_RADPS;
  const double threshold = last_internal_deg / 30.0;
  const double increment = 183.345993 * kDt;
  const double target_deg = -command_norm * 30.0;
'''


def test_real_checkout_missing_native_declaration_with_plant_update(tmp_path):
    # No kSteeringLimitDeg or kSteeringRateDegPerS in this revision.
    write_native(tmp_path, 'float unity_steering_step(float angle, float target);')
    with pytest.warns(RuntimeWarning) as caught:
        limit, rate, prov = _current_steering_parameters(tmp_path, PLANT_WITH_ALIASES)
    assert limit == pytest.approx(math.pi / 6)
    assert rate == pytest.approx(math.radians(183.345993))
    assert prov['limit_native_unverified']
    assert prov['rate_native_unverified']
    assert prov['limit_alias_unverified'] == 'UNITY_SOURCE_MAX_STEERING_RAD'
    assert any('native runtime does not expose' in str(x.message) for x in caught)
    assert prov['limit_source'] == 'unity_source_plant.cpp actuator update'


def test_native_present_crosschecks_against_plant_literals(tmp_path):
    write_native(tmp_path, 'constexpr float kSteeringLimitDeg = 30.0f;\n'
                           'constexpr float kSteeringRateDegPerS = 183.346f;\n')
    with pytest.warns(RuntimeWarning):
        limit, rate, _ = _current_steering_parameters(tmp_path, PLANT_WITH_ALIASES)
    assert limit == pytest.approx(math.pi/6)
    assert rate == pytest.approx(math.radians(183.346), abs=1e-6)


def test_native_mismatch_still_blocks_unsafe_model(tmp_path):
    write_native(tmp_path, 'constexpr float kSteeringLimitDeg = 20.0f;\n'
                           'constexpr float kSteeringRateDegPerS = 183.346f;\n')
    with pytest.raises(RuntimeError, match='disagrees'):
        _current_steering_parameters(tmp_path, PLANT_WITH_ALIASES)


def test_no_numbers_anywhere_refuses_guess(tmp_path):
    write_native(tmp_path, 'float unity_steering_step(float angle, float target);')
    code = '''constexpr double kMaxSteeringRad = UNITY_SOURCE_MAX_STEERING_RAD;
              constexpr double kSteeringRateRadps = UNITY_SOURCE_STEERING_RATE_RADPS;'''
    with pytest.raises(RuntimeError, match='Cannot verify'):
        _current_steering_parameters(tmp_path, code)


def test_plant_update_conflicting_limit_is_rejected(tmp_path):
    write_native(tmp_path, '')
    code = PLANT_WITH_ALIASES.replace('last_internal_deg / 30.0', 'last_internal_deg / 20.0')
    with pytest.raises(RuntimeError, match='different limits'):
        _current_steering_parameters(tmp_path, code)


def test_native_included_header_constant_is_resolved(tmp_path):
    write_native(tmp_path, '#include "actuator_constants.h"\n')
    h = tmp_path / 'f1tenth_localization/include/actuator_constants.h'
    h.parent.mkdir(parents=True, exist_ok=True)
    h.write_text('constexpr float kSteeringLimitDeg = 30.0f;\n'
                 'constexpr float kSteeringRateDegPerS = 183.346f;\n')
    with pytest.warns(RuntimeWarning):
        limit, rate, meta = _current_steering_parameters(tmp_path, PLANT_WITH_ALIASES)
    assert limit == pytest.approx(math.pi/6)
    assert rate == pytest.approx(math.radians(183.346),abs=1e-6)
    assert 'limit_native_unverified' not in meta
