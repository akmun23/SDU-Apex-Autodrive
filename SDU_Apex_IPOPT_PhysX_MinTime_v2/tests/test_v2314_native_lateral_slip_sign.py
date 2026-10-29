"""v2.3.14 native lateral-slip coordinate parity and guarded cache upgrade."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import casadi as ca
import pytest
from tests.test_v2312_native_tire_load_parity import mock_original_wheel_text
from wheel_model_overlay_v2312 import transform_wheel_model as transform_v2312
from wheel_model_overlay_v2314 import (
    transform_wheel_model,prepare_working_wheel_model,_SLIP_OLD,_SLIP_NEW,
)
from diagnostic_fixtures.diagnose_slip_parity_v2314 import compare_baseline,reconstruct_records
from tests.test_phase1_overlay import LEGACY,_sample


def wheel_source():
    # Mock original model includes source-identical pre-v2314 slip expression.
    return mock_original_wheel_text()+'''
def _physx_applied_torque_slips_casadi(long_speed,lateral_speed,wheel_surface_speed):
    long_speed_abs = ca.fabs(long_speed)
    long_slip = 0.0
'''+_SLIP_OLD+'''
    return long_slip, lateral_slip
'''


def test_native_signed_lateral_slip_is_reverse_of_left_positive_model():
    hist=wheel_source(); updated=transform_wheel_model(hist)
    assert hist.count(_SLIP_OLD)==1
    assert updated.count(_SLIP_NEW)==1
    assert 'tire_load = suspension_forces[index]' in updated
    context={'ca':ca,'PHYSX_LATERAL_SLIP_SPEED_OFFSET_MPS':1.0,
             'PHYSX_TOLERANCE_SCALE_LENGTH_M':1.0}
    exec(updated,context)
    slips=context['_physx_applied_torque_slips_casadi']
    old_context={**context}
    exec(hist,old_context)
    old=old_context['_physx_applied_torque_slips_casadi']
    speed=2.3
    v=-0.012
    native=-np.arctan(v/(speed+1))
    assert float(slips(speed,v,0)[1])==pytest.approx(native,abs=1e-12)
    assert float(slips(speed,v,0)[1])==-float(old(speed,v,0)[1])


def test_full_known_generated_cache_upgrades_and_unknown_edit_refused(tmp_path):
    restored, original_solver, _=_sample(tmp_path)
    hist=wheel_source()
    rel=Path(LEGACY)/'f1tenth_planning/autodrive_mintime/unity_wheel_model.py'
    src=restored/rel;src.write_text(hist)
    working=tmp_path/'working'
    path=working/rel;path.parent.mkdir(parents=True)
    old_patch=transform_v2312(hist)
    final=transform_wheel_model(hist)
    path.write_text(old_patch)
    meta=prepare_working_wheel_model(restored,working,LEGACY)
    assert meta['native_lateral_slip_sign_corrected']
    assert meta['working_wheel_model_parity_applied']
    assert path.read_text()==final
    assert src.read_text()==hist
    assert prepare_working_wheel_model(restored,working,LEGACY)==meta
    path.write_text(final+'\n# unrecognized modifications\n')
    with pytest.raises(RuntimeError,match='unknown local edits'):
        prepare_working_wheel_model(restored,working,LEGACY)
    assert src.read_text()==hist


def test_fail_closed_on_unexpected_real_source():
    hist=wheel_source()
    with pytest.raises(RuntimeError,match='lateral slip'):
        transform_wheel_model(hist.replace('lateral_slip = ca.atan(', 'lateral_slip = something_else('))
    with pytest.raises(RuntimeError,match='lateral slip'):
        transform_wheel_model(hist.replace('lateral_slip = ca.atan(', 'lateral_slip = ca.atan(\n# injected comment'))


def test_known_legacy_mock_fixture_does_not_break_existing_regressions():
    old=mock_original_wheel_text()
    assert transform_wheel_model(old)==transform_v2312(old)


def test_saved_native_csv_records_reconstruction_checks_wheel_order():
    template={'time_s':'12.025','progress_m':'28.4','native_u_mps':'2.4',
              'native_v_mps':'-.23','native_yaw_rate_radps':'-1.6',
              'native_steering_rad':'-.24','native_omega_radps':'41',
              'native_suspension_force_n':'10','native_long_slip':'-.04',
              'native_lat_slip':'-.003','native_callback_fx_n':'-2',
              'native_callback_fy_n':'4'}
    rows=[{**template,'wheel':w} for w in ('fl','fr','rl','rr')]*481
    records=reconstruct_records(rows)
    assert len(records)==481
    assert records[0]['wheel_omega_radps'].shape==(4,)
    bad=list(rows);bad[0]={**bad[0],'wheel':'fr'}
    with pytest.raises(ValueError,match='unordered'):
        reconstruct_records(bad)


def test_baseline_comparison_requires_fy_and_slip_improvement_and_unchanged_sx():
    old=[{'model_lat_slip':0.008,'native_lat_slip':-0.008,
          'model_long_slip':0.01,'native_long_slip':0.0102,
          'model_reconstructed_fy_n':-4.,'native_callback_fy_n':4.,
          'model_reconstructed_fx_n':1.,'native_callback_fx_n':1.1}]*4
    new=[{**r,'model_lat_slip':-0.0079,'model_reconstructed_fy_n':3.9} for r in old]
    d=compare_baseline(old,new)
    assert d['lateral_tire_force_n']['before']['rms']>7
    assert d['lateral_tire_force_n']['after']['rms']<0.11
    with pytest.raises(RuntimeError,match='tire-force RMS'):
        compare_baseline(old,old)
