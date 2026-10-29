"""Track-profile selection is pure configuration; no ROS installation needed."""
import ast
import os
from pathlib import Path


def parameter_files(map_name, overlay=''):
    source = Path(__file__).resolve().parents[1]/'sdu_apex_autodrive/launch/controller.launch.py'
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_mpc_parameter_files')
    scope = dict(os=os, MAP_NAME='autodrive_practice_20260924_b.yaml')
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), scope)
    return scope['_mpc_parameter_files']('/mpc', '/maps/'+map_name, overlay)


def test_iros_does_not_silently_inherit_practice_weights_or_sqp_radii():
    assert parameter_files('autodrive_iros_2026_compete.yaml') == ['/mpc/config/mpc_competition.yaml']


def test_practice_keeps_verified_profile_and_explicit_overlay_wins():
    assert parameter_files('autodrive_practice_20260924_b.yaml', '/tuning.yaml') == [
        '/mpc/config/mpc_competition.yaml', '/mpc/config/mpc_practice_fastline.yaml', '/tuning.yaml']
    assert parameter_files('another_track.yaml', '/tuning.yaml') == [
        '/mpc/config/mpc_competition.yaml', '/tuning.yaml']


def test_default_speed_does_not_override_map_specific_yaml():
    source = Path(__file__).resolve().parents[1]/'sdu_apex_autodrive/launch/controller.launch.py'
    tree = ast.parse(source.read_text())
    declaration = next(
        call for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == 'DeclareLaunchArgument' and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 'controller_max_speed')
    default = next(kw.value.value for kw in declaration.keywords if kw.arg == 'default_value')
    assert default == ''  # The old implicit 16.0 override shadowed YAML speed limits.
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_optional_controller_speed')
    import math
    scope = dict(math=math)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), scope)
    policy = scope['_optional_controller_speed']
    assert policy('') == {}
    assert policy('  ') == {}
    assert policy('5') == {'max_speed_mps': 5.}
    assert policy('16') == {'max_speed_mps': 16.}
    import pytest
    for invalid in ('0', '-1', '17', 'nan', 'inf', 'garbage'):
        with pytest.raises(RuntimeError):
            policy(invalid)
