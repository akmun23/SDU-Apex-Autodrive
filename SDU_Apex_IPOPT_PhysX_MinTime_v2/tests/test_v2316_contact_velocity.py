"""Physical contact-point velocity, strict generated solver upgrade, and sync."""
from pathlib import Path
import numpy as np
import pytest
from solver_overlay import (transform_solver,transform_solver_v2316, transform_solver_v240,
                            prepare_working_solver,_CONTACT_SOLVER_CALL_OLD,
                            _CONTACT_SOLVER_CALL_NEW)
from wheel_model_overlay_v2316 import (transform_wheel_model,
                                       prepare_working_wheel_model,
                                       _SIGNATURE_OLD,_SIGNATURE_NEW,
                                       _VELOCITY_OLD,_VELOCITY_NEW)
from wheel_model_overlay_v2315 import (transform_wheel_model as v2315,
                                       _BODY_FORCES_OLD)
from tests.test_v2314_native_lateral_slip_sign import wheel_source
from tests.test_phase1_overlay import _sample,LEGACY,SOLVER
from diagnose_contact_v2316 import interpolate_native_body_rates,rms


def realistic_original_wheel_fixture():
    return (wheel_source() + '\n' + _SIGNATURE_OLD + '\n' +
            _VELOCITY_OLD + '\n' + _BODY_FORCES_OLD + '\n')


def test_source_gated_wheel_contact_upgrade_and_unmodified_original():
    original=realistic_original_wheel_fixture()
    updated=transform_wheel_model(original)
    assert _SIGNATURE_NEW in updated
    assert _VELOCITY_NEW in updated
    assert updated.count('pitch_rate * contact_arm_down_m')==1
    assert updated.count('roll_rate * contact_arm_down_m')==1
    assert 'lateral_slip = -ca.atan(' in updated
    assert 'fy_body = fx * ca.sin(angle) - fy * ca.cos(angle)' in updated
    assert _SIGNATURE_OLD in original and _VELOCITY_OLD in original


def test_cache_upgrades_from_exact_v2315_wheel_source_and_fails_unknown(tmp_path):
    recovered,_,_= _sample(tmp_path)
    original=realistic_original_wheel_fixture()
    rel=Path(LEGACY)/'f1tenth_planning/autodrive_mintime/unity_wheel_model.py'
    source=recovered/rel;source.write_text(original)
    target=tmp_path/'working'/rel;target.parent.mkdir(parents=True)
    target.write_text(v2315(original))
    meta=prepare_working_wheel_model(recovered,tmp_path/'working',LEGACY)
    assert meta['native_full_contact_velocity_corrected']
    assert target.read_text()==transform_wheel_model(original)
    assert source.read_text()==original
    assert prepare_working_wheel_model(recovered,tmp_path/'working',LEGACY)==meta
    target.write_text(target.read_text()+'\n# arbitrary change')
    with pytest.raises(RuntimeError,match='unknown local edits'):
        prepare_working_wheel_model(recovered,tmp_path/'working',LEGACY)
    assert source.read_text()==original


def test_modified_source_does_not_silently_skip_contact_velocity():
    original=realistic_original_wheel_fixture()
    with pytest.raises(RuntimeError,match='contact velocity'):
        transform_wheel_model(original.replace(_VELOCITY_OLD,'vx_body=0\nvy_body=0'))
    with pytest.raises(RuntimeError,match='signature'):
        transform_wheel_model(original.replace(_SIGNATURE_OLD,'def unexpected():'))


def test_generated_real_solver_site_accepts_only_source_known_wheel_call(tmp_path):
    recovered,original,stub=_sample(tmp_path)
    realish=stub+'\n    def source_wheel_update():\n        pass\n'+_CONTACT_SOLVER_CALL_OLD+'\n'
    original.write_text(realish)
    prior=transform_solver(realish,'limited-memory')
    latest=transform_solver_v2316(realish,'limited-memory')
    assert _CONTACT_SOLVER_CALL_OLD in prior
    assert _CONTACT_SOLVER_CALL_NEW in latest
    assert _CONTACT_SOLVER_CALL_OLD not in latest
    target=tmp_path/'working'
    dest=target/LEGACY/SOLVER
    dest.parent.mkdir(parents=True)
    dest.write_text(prior)
    root,meta=prepare_working_solver(recovered,target,LEGACY)
    assert root==target
    assert dest.read_text()==transform_solver_v240(realish,'limited-memory')
    assert 'rate' in meta['physical_dynamics_change']
    assert 'u=x[2,k]' in meta['physical_dynamics_change']
    assert original.read_text()==realish
    dest.write_text(latest+'\n# unauthorized local modification')
    with pytest.raises(RuntimeError,match='unknown local edits'):
        prepare_working_solver(recovered,target,LEGACY)
    with pytest.raises(RuntimeError,match='contact site'):
        transform_solver_v2316(realish.replace(_CONTACT_SOLVER_CALL_OLD,'unexpected'),
                               'limited-memory')


def test_source_contact_arm_and_angular_cross_product():
    com_up=0.06434;wheel_radius=0.059
    mount_up=np.array([0.0599048612231,0.0599048612231,0.05,0.05])
    down=com_up+wheel_radius-mount_up
    np.testing.assert_allclose(down,[.0634351387769,.0634351387769,.07334,.07334],atol=1e-10)
    for i in range(4):
        rr=0.066;pr=0.09;omega=np.array([rr,pr,1.5]);point=np.array([.17468,.11574,-down[i]])
        contact_correction=np.cross(omega,point)
        assert contact_correction[0]==pytest.approx(-pr*down[i]-1.5*.11574)
        assert contact_correction[1]==pytest.approx(1.5*.17468+rr*down[i])


def test_saved_seed_phase_alignment_refuses_unmatched_trajectories():
    seed={k:np.ones(143)*v for k,v in [('u_mps',2.3),('v_mps',.1),
          ('yaw_rate_radps',-.2),('steering_rad',-.25),
          ('roll_rate_radps',.05),('pitch_rate_radps',.1)]}
    records=[{'progress_m':28.41+.04*k,'u_mps':2.3,'v_mps':.1,
              'yaw_rate_radps':-.2,'steering_rad':-.25} for k in range(481)]
    roll,pitch,align=interpolate_native_body_rates(seed,records,28.41016469)
    assert align['yaw_rate_radps_rms']==pytest.approx(0,abs=1e-12)
    assert np.allclose(roll,.05) and np.allclose(pitch,.1)
    for record in records:
        record['steering_rad']=.4
    with pytest.raises(RuntimeError,match='does not phase-align'):
        interpolate_native_body_rates(seed,records,28.41016469)


def test_rms_rejects_nan():
    assert rms([1.,-1.])['rms']==pytest.approx(1.)
    with pytest.raises(ValueError):rms([float('nan')])
