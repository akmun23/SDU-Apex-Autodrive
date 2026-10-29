"""v2.3.15 native right-positive tire forces -> left-positive chassis."""
from pathlib import Path
import numpy as np
import pytest
from wheel_model_overlay_v2314 import transform_wheel_model as v2314, _SLIP_OLD
from wheel_model_overlay_v2315 import (
    transform_wheel_model,prepare_working_wheel_model,
    _BODY_FORCES_OLD,_BODY_FORCES_NEW,
)
from tests.test_v2314_native_lateral_slip_sign import wheel_source
from tests.test_phase1_overlay import LEGACY, _sample
from diagnostic_fixtures.diagnose_force_frame_v2315 import rotation_prediction


def genuine_fragment():
    return wheel_source()+'\ndef real_forces():\n'+_BODY_FORCES_OLD+'\n'


def test_generated_casadi_reverses_force_vector_without_touching_callback_raw():
    source=genuine_fragment()
    final=transform_wheel_model(source)
    assert final.count(_BODY_FORCES_NEW)==1
    assert final.count(_BODY_FORCES_OLD)==0
    assert 'lateral_slip = -ca.atan(' in final
    assert 'fy_wheels.append(fy)' not in source or 'fy_wheels.append(fy)' in final
    assert 'fx_body = fx * ca.cos(angle) + fy * ca.sin(angle)' in final
    assert 'fy_body = fx * ca.sin(angle) - fy * ca.cos(angle)' in final
    assert source==genuine_fragment()


def test_right_positive_native_callback_force_becomes_left_negative_chassis_force():
    angle=-0.25;long_force=-2.6;native_right_force=4.85
    model_x=long_force*np.cos(angle)+native_right_force*np.sin(angle)
    model_y=long_force*np.sin(angle)-native_right_force*np.cos(angle)
    assert model_y<0
    assert model_x<0
    old_y=long_force*np.sin(angle)+native_right_force*np.cos(angle)
    assert old_y>0


def test_safe_known_v2314_wheel_cache_upgrade_and_manual_edit_refusal(tmp_path):
    restored, *_=_sample(tmp_path)
    source=genuine_fragment()
    relative=Path(LEGACY)/'f1tenth_planning/autodrive_mintime/unity_wheel_model.py'
    original=restored/relative;original.write_text(source)
    working=tmp_path/'working'
    target=working/relative;target.parent.mkdir(parents=True)
    target.write_text(v2314(source))
    meta=prepare_working_wheel_model(restored,working,LEGACY)
    assert meta['native_lateral_force_frame_corrected']
    assert meta['native_lateral_slip_sign_corrected']
    assert target.read_text()==transform_wheel_model(source)
    assert original.read_text()==source
    assert prepare_working_wheel_model(restored,working,LEGACY)==meta
    target.write_text(target.read_text()+'\n# user change\n')
    with pytest.raises(RuntimeError, match='unknown local edits'):
        prepare_working_wheel_model(restored,working,LEGACY)
    assert original.read_text()==source


def test_unexpected_real_wheel_source_fails_closed():
    raw=genuine_fragment()
    with pytest.raises(RuntimeError,match='chassis lateral-force frame'):
        transform_wheel_model(raw.replace(_BODY_FORCES_OLD,'fx_body = wrong'))
    with pytest.raises(RuntimeError,match='chassis lateral-force frame'):
        transform_wheel_model(raw.replace(_BODY_FORCES_OLD,_BODY_FORCES_OLD+'\n'+_BODY_FORCES_OLD))


def test_lateral_native_acceleration_checks_force_sign_from_complete_capture():
    # 481 periodic samples at the production control cadence. All four
    # native wheels push right; model left acceleration must be negative.
    records=[]
    for i in range(481):
        records.append({'time_s':12+.025*i,'u_mps':2.3,'v_mps':0.,
                        'yaw_rate_radps':0.,'steering_rad':0.,
                        'tire_longitudinal_force_n':np.zeros(4),
                        'tire_lateral_force_n':np.full(4,2.0)})
    _,observed,before,after,_=rotation_prediction(records,3.47,0.)
    assert np.max(np.abs(observed))<1e-9
    assert np.all(before>0)
    assert np.all(after<0)
    assert np.allclose(after,-before)
