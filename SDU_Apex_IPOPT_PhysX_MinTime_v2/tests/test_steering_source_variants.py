"""Check steering constants against the active source-plant equations."""
import math
from pathlib import Path

import pytest

from source_vehicle import _cpp_number, _current_steering_parameters


def plant_source(limit_deg=30.0, rate_degps=183.346, declarations=True):
    constants = ""
    if declarations:
        constants = (
            f"constexpr double kMaxSteeringRad = ({limit_deg} * kPi) / 180.0;\n"
            f"constexpr double kSteeringRateRadps = ({rate_degps} * kPi) / 180.0;\n"
        )
    return constants + f"""
constexpr double kDt = 0.001;
void update_steering(double last_internal_deg, double command_norm) {{
  const double threshold = last_internal_deg / {limit_deg};
  const double increment = {rate_degps} * kDt;
  const double target_deg = -command_norm * {limit_deg};
}}
"""


def test_source_cpp_float_and_arithmetic_literals():
    raw = """
    // constexpr double kMaxSteeringRad = 5.0;
    constexpr double kMaxSteeringRad = (30.0 * kPi) / 180.0;
    constexpr float kSteeringRateRadps = 3.2f;
    """
    assert _cpp_number(raw, "kMaxSteeringRad") == pytest.approx(math.pi / 6)
    assert _cpp_number(raw, "kSteeringRateRadps") == pytest.approx(3.2)


def test_source_declarations_match_the_actuator_update():
    source = plant_source()
    limit, rate, provenance = _current_steering_parameters(Path("."), source)
    assert limit == pytest.approx(math.pi / 6)
    assert rate == pytest.approx(math.radians(183.346))
    assert provenance["limit_source"] == "unity_source_plant.cpp definition"
    assert provenance["rate_source"] == "unity_source_plant.cpp definition"


def test_source_update_can_supply_values_when_declarations_are_absent():
    limit, rate, provenance = _current_steering_parameters(
        Path("."), plant_source(declarations=False))
    assert limit == pytest.approx(math.pi / 6)
    assert rate == pytest.approx(math.radians(183.346))
    assert provenance["limit_source"] == "unity_source_plant.cpp actuator update"
    assert provenance["rate_source"] == "unity_source_plant.cpp actuator update"


def test_source_disagreement_is_rejected():
    source = plant_source().replace(
        "constexpr double kMaxSteeringRad = (30.0 * kPi) / 180.0;",
        "constexpr double kMaxSteeringRad = 0.3;")
    with pytest.raises(RuntimeError, match="disagrees"):
        _current_steering_parameters(Path("."), source)


def test_unknown_constant_expression_is_not_guessed():
    source = plant_source().replace(
        "constexpr double kMaxSteeringRad = (30.0 * kPi) / 180.0;",
        "constexpr double kMaxSteeringRad = steeringAngleFromConfig;")
    with pytest.raises(RuntimeError, match="Cannot resolve"):
        _current_steering_parameters(Path("."), source)


def test_conflicting_limit_in_update_is_rejected():
    source = plant_source().replace(
        "const double target_deg = -command_norm * 30.0;",
        "const double target_deg = -command_norm * 20.0;")
    with pytest.raises(RuntimeError, match="different limits"):
        _current_steering_parameters(Path("."), source)


def test_missing_source_values_are_not_silently_defaulted():
    with pytest.raises(RuntimeError, match="Cannot (resolve|verify)"):
        _current_steering_parameters(Path("."), "")


def test_constants_in_the_source_owned_header_are_resolved(tmp_path):
    include_dir = tmp_path / "f1tenth_mpc/include"
    include_dir.mkdir(parents=True)
    (include_dir / "actuator_constants.h").write_text(
        "constexpr double kMaxSteeringRad = (30.0 * kPi) / 180.0;\n"
        "constexpr double kSteeringRateRadps = (183.346 * kPi) / 180.0;\n")
    source = '#include "actuator_constants.h"\n' + plant_source(declarations=False)
    limit, rate, _ = _current_steering_parameters(tmp_path, source)
    assert limit == pytest.approx(math.pi / 6)
    assert rate == pytest.approx(math.radians(183.346))
