"""Source-fixed CasADi callback regression using real measured 1ms PhysX values."""
import hashlib
from pathlib import Path
import casadi as ca
import numpy as np
import pytest
from wheel_model_overlay_v2312 import (
    transform_wheel_model, prepare_working_wheel_model, _SHADER_OLD, _LOAD_OLD,
)
from tests.test_phase1_overlay import _sample, LEGACY


FRICTION = '''def _unity_friction_curve_response(slip, curve):
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = curve
    magnitude = ca.fabs(slip)
    first_t = magnitude / extremum_slip
    first = extremum_value * (first_t + first_t**2 - first_t**3)
    decay_t = (magnitude - extremum_slip) / (asymptote_slip - extremum_slip)
    smooth = 3.0 * decay_t**2 - 2.0 * decay_t**3
    sign = ca.if_else(slip >= 0.0, 1.0, -1.0)
    second = extremum_value + (asymptote_value - extremum_value) * smooth
    return stiffness * ca.if_else(
        magnitude < extremum_slip, sign * first,
        sign * ca.if_else(magnitude < asymptote_slip,
                          second, asymptote_value))
'''


def mock_original_wheel_text():
    return (FRICTION + '\ndef _unity_tire_force_shader_casadi(long_slip, lateral_slip, tire_load, wheel_radius, forward_curve, lateral_curve):\n'
        + '    long_response = _unity_friction_curve_response(long_slip, forward_curve)\n'
        + '    lateral_response = _unity_friction_curve_response(lateral_slip,\n'
        + '                                                       lateral_curve)\n'
        + _SHADER_OLD + '\n'
        + '    wheel_torque = -force_long * wheel_radius\n'
        + '    return wheel_torque, force_long, force_lateral, 0.0\n'
        + '\ndef tire_forces_casadi(model, suspension_forces, index):\n'
        + '    for index in range(len(suspension_forces)):\n'
        + _LOAD_OLD + '\n'
        + '        return tire_load\n')


def _casadi_shader(mock):
    scope={'ca':ca, 'WHEEL_MASS_KG':0.109,'UNITY_PHYSICS_GRAVITY_MPS2':9.81,
           '_physx_filtered_tire_load_casadi': lambda fz,*args: fz+1.49}
    exec(mock,scope)
    sx=ca.SX.sym('sx');sy=ca.SX.sym('sy');fz=ca.SX.sym('fz')
    curve_fx=(.15,.9,.25,.58,.8); curve_fy=(.01,1.,.1,.5,1.)
    _,fx,fy,_=scope['_unity_tire_force_shader_casadi'](
        sx,sy,fz,.059,curve_fx,curve_fy)
    return ca.Function('shader', [sx,sy,fz], [fx,fy]),scope


@pytest.mark.parametrize('sx,sy,fz,fx,fy', [
    (-0.04169334098696709,-0.0036509723868221045,10.793487548828125,
     -2.5935990810394287,4.854125499725342),
    (0.06920290738344193,-0.005760534666478634,6.91856575012207,
     2.869276762008667,4.958777904510498),
])
def test_generated_casadi_matches_recorded_native_force_under_native_slip(sx,sy,fz,fx,fy):
    updated=transform_wheel_model(mock_original_wheel_text())
    fn,ns=_casadi_shader(updated)
    got=tuple(float(v) for v in fn(sx,sy,fz))
    np.testing.assert_allclose(got,(fx,fy),atol=5e-6,rtol=0)
    assert ns['tire_forces_casadi'](None,[fz],0)==fz


def test_original_model_overfilters_and_has_wrong_stiffness_order():
    old=mock_original_wheel_text()
    f_old,ns=_casadi_shader(old)
    f_new,_=_casadi_shader(transform_wheel_model(old))
    sx,sy,fz=0.11,-0.01,8.4
    original=np.asarray([float(v) for v in f_old(sx,sy,fz)])
    corrected=np.asarray([float(v) for v in f_new(sx,sy,fz)])
    assert np.max(abs(original-corrected))>0.01
    assert ns['tire_forces_casadi'](type('Model', (), {'sprung_mass_kg':[0.817]})(),[fz],0)==pytest.approx(fz+1.49)


def test_exact_wheel_cache_upgrade_and_historical_protection(tmp_path):
    restored, original_solver, solver_src=_sample(tmp_path)
    path=restored/LEGACY/'f1tenth_planning/autodrive_mintime/unity_wheel_model.py'
    historical=mock_original_wheel_text()
    path.write_text(historical)
    working=tmp_path/'working'
    target=working/LEGACY/'f1tenth_planning/autodrive_mintime/unity_wheel_model.py'
    target.parent.mkdir(parents=True)
    target.write_text(historical)
    meta=prepare_working_wheel_model(restored,working,LEGACY)
    corrected=transform_wheel_model(historical)
    assert meta['working_wheel_model_parity_applied']
    assert target.read_text()==corrected
    assert path.read_text()==historical
    assert meta['working_wheel_model_sha256']==hashlib.sha256(corrected.encode()).hexdigest()
    assert prepare_working_wheel_model(restored,working,LEGACY)==meta
    target.write_text(corrected+'\n# unknown local edit\n')
    with pytest.raises(RuntimeError,match='unknown local edits'):
        prepare_working_wheel_model(restored,working,LEGACY)
    assert path.read_text()==historical


def test_unrecognized_historical_sites_fail_closed():
    source=mock_original_wheel_text()
    with pytest.raises(RuntimeError,match='combined-slip stiffness'):
        transform_wheel_model(source.replace('utilization_sq =', 'usage_sq ='))
    with pytest.raises(RuntimeError,match='native callback tireLoad'):
        transform_wheel_model(source.replace('WHEEL_MASS_KG,', 'WHEEL_MASS_CHANGED,'))


def test_prepare_working_solver_upgrades_both_known_cached_files_atomically(tmp_path):
    from solver_overlay import prepare_working_solver, transform_solver
    restored, historical_solver_file, source_solver=_sample(tmp_path)
    historical_wheel_file=(restored/LEGACY/
        'f1tenth_planning/autodrive_mintime/unity_wheel_model.py')
    orig_wheel=mock_original_wheel_text()
    historical_wheel_file.write_text(orig_wheel)
    out=tmp_path/'generated'
    prepared,meta=prepare_working_solver(restored,out,LEGACY)
    target_wheel=out/LEGACY/'f1tenth_planning/autodrive_mintime/unity_wheel_model.py'
    target_solver=out/LEGACY/'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py'
    assert target_wheel.read_text()==transform_wheel_model(orig_wheel)
    assert target_solver.read_text()==transform_solver(source_solver,'limited-memory')
    assert meta['working_wheel_model_parity_applied']
    assert historical_wheel_file.read_text()==orig_wheel
    assert historical_solver_file.read_text()==source_solver
    # Existing v2.3.11 generated wheel source is EXACT original; upgrade it.
    target_wheel.write_text(orig_wheel)
    _,meta2=prepare_working_solver(restored,out,LEGACY)
    assert meta2==meta
    assert target_wheel.read_text()==transform_wheel_model(orig_wheel)
    # Unknown changes must not mutate the already generated solver either.
    target_wheel.write_text(orig_wheel+'\n# user customized\n')
    solver_before=target_solver.read_bytes()
    with pytest.raises(RuntimeError,match='unknown local edits'):
        prepare_working_solver(restored,out,LEGACY)
    assert target_solver.read_bytes()==solver_before
