"""Test v2.3.13's fail-closed same-trajectory native slip audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import casadi as ca
import pytest
from diagnostic_fixtures.diagnose_slip_v2313 import build_audit, build_casadi_audit, summarize


def _record():
    return {
        'time_s':12.025, 'progress_m':28.5,
        'u_mps':2.4,'v_mps':0.1,'yaw_rate_radps':-0.1,
        'steering_rad':-0.05,
        'wheel_omega_radps':np.array([39.,40.,41.,42.]),
        'normal_load_n':np.array([8.,9.,10.,11.]),
        'longitudinal_slip':np.array([.1,.2,.3,.4]),
        'lateral_slip':np.array([-.04,.02,.01,-.03]),
        'tire_longitudinal_force_n':np.array([1.,2.,3.,4.]),
        'tire_lateral_force_n':np.array([-.4,.5,.6,.7]),
    }


def _fake_native_aware_fun(u,v,yaw,steering,omega,load,sx,sy):
    return tuple(ca.DM(x) for x in (
        np.array(sx)+.01, np.array(sy)-.02, np.ones(4)*2,
        np.ones(4)*-.3, np.ones(4)*1, np.ones(4)*-.4,
        [4],[5]))


def test_audit_preserves_per_wheel_native_slip_and_produces_separate_reconstructed_metrics():
    rows=build_audit([_record() for _ in range(20)],_fake_native_aware_fun)
    assert len(rows)==80
    assert [r['wheel'] for r in rows[:4]]==['fl','fr','rl','rr']
    assert rows[0]['native_omega_radps']==39.
    assert rows[0]['long_slip_difference']==pytest.approx(.01)
    assert rows[0]['lat_slip_difference']==pytest.approx(-.02)
    summary=summarize(rows,{'completed_flying_lap':True})
    assert summary['metrics']['longitudinal_slip']['rms']==pytest.approx(.01)
    assert summary['metrics']['lateral_slip']['rms']==pytest.approx(.02)
    assert summary['sample_timing_caveat'].find('1 ms')>=0
    assert summary['actual_ipopt_solve_ran'] is False
    assert 'reconstructed_fy_n' in summary['per_wheel']['fl']


def test_missing_wheel_omega_is_rejected():
    data=_record();del data['wheel_omega_radps']
    with pytest.raises(KeyError):build_audit([data]*20,_fake_native_aware_fun)


def test_malformed_wheel_vector_rejected():
    data=_record();data['wheel_omega_radps']=np.array([1.,2.,3.])
    with pytest.raises(ValueError,match='shape'):
        build_audit([data]*20,_fake_native_aware_fun)


def test_source_seed_capture_records_true_native_wheel_state():
    source=(Path(__file__).resolve().parents[1]/'native_seed.py').read_text()
    assert "'wheel_omega_radps': raw[3:7].copy()" in source
    assert "if contact_records is not None:" in source
    assert 'body velocities and wheel angular speeds are sampled after' in (
        Path(__file__).resolve().parent/'diagnostic_fixtures'/'diagnose_slip_v2313.py').read_text()


def test_native_audit_casadi_function_preserves_wheel_order_and_native_slip_path():
    class Model:
        wheel_radius_m=.059
        forward_curve=((.15,.9,.25,.58,.8),)*4
        lateral_curve=((.01,1.,.1,.5,1.),)*4
    class Wheel:
        @staticmethod
        def tire_forces_casadi(model,u,v,yaw,steer,omega,loads):
            fx=[o*.05 for o in omega]
            fy=[-o*.1 for o in omega]
            return sum(fx),sum(fy),0.,tuple(fx),tuple(fy),tuple(o*.01 for o in omega),tuple(-o*.02 for o in omega)
        @staticmethod
        def _unity_tire_force_shader_casadi(sx,sy,load,radius,forward,lateral):
            return 0.,sx*load,-sy*load,0.
    fn=build_casadi_audit(Wheel(), Model())
    values=fn(2.4,.1,-.1,-.05,[39.,40.,41.,42.],[8.,9.,10.,11.],[.1,.2,.3,.4],[-.04,.02,.01,-.03])
    assert np.asarray(values[0].full()).flatten().tolist()==pytest.approx([.39,.40,.41,.42])
    assert np.asarray(values[1].full()).flatten().tolist()==pytest.approx([-.78,-.80,-.82,-.84])
    assert np.asarray(values[4].full()).flatten().tolist()==pytest.approx([.8,1.8,3.,4.4])
    assert float(values[6])==pytest.approx(8.1)
