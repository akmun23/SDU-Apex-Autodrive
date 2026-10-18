"""v2.3.9: frame candidate must be a forensic test, never a silent model rewrite."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from diagnose_suspension_frame_v239 import (
    apply_sign_options, lateral_transfer,load_metrics,mirrored_lateral_geometry,compare_candidates)
from test_v238_scalar_force_audit import Tire,seed

@dataclass(frozen=True)
class SuspensionModel:
    mass_kg:float=3.47
    drag_per_s:float=.12
    suspension_base_xyz_body_m:tuple=((-.12,.06,.17),(.12,.06,.17),(-.12,.05,-.16),(.12,.05,-.16))

class SuspensionWheel(Tire):
    @staticmethod
    def suspension_loads_numpy(model,heave,hd,roll,rd,pitch,pd):
        return [8.+10*roll*x-2*pitch*z+rd*x for x,_,z in model.suspension_base_xyz_body_m]


def test_reflected_geometry_matches_reversed_roll_and_roll_rate_for_mock():
    s=seed(15)
    s['roll_rad']=np.linspace(-.05,.03,15)
    s['roll_rate_radps']=np.linspace(.005,.01,15)
    s['pitch_rad']=np.full(15,.02)
    out,rows=compare_candidates(s,SuspensionModel(),SuspensionWheel)
    assert out['geometry_reflection_equals_sign_reversed_native_roll_max_error_n']<1e-12
    assert out['no_physics_equation_changed'] and out['no_ipopt_or_native_replay']
    assert len(rows)==15
    assert len(out['candidate_results'])==5


def test_metrics_detect_native_symbolic_opposite_load_transfer():
    n=np.array([[10,11,10],[7,8,7],[12,13,12],[8,9,8]],dtype=float)
    m=np.array([[7,8,7],[10,11,10],[8,9,8],[12,13,12]],dtype=float)
    metrics=load_metrics(n,m)
    assert metrics['front_left_minus_right']['sign_mismatch_fraction']==1
    assert metrics['rear_left_minus_right']['sign_mismatch_fraction']==1
    assert metrics['global_max_abs_load_error_n']==pytest.approx(4.)
    assert metrics['global_rms_load_error_n']>0


def test_model_reflection_changes_only_suspension_lateral_mount_sign():
    model=SuspensionModel()
    mirror=mirrored_lateral_geometry(model)
    assert model.suspension_base_xyz_body_m[0][0]==-.12
    assert mirror.suspension_base_xyz_body_m[0][0]==.12
    for a,b in zip(model.suspension_base_xyz_body_m,mirror.suspension_base_xyz_body_m):
        assert (-a[0],a[1],a[2])==b
    assert mirror.mass_kg==model.mass_kg


def test_reflect_roll_does_not_mutate_saved_native_seed():
    s=seed(3)
    s['roll_rad']=np.array([.1,-.2,.05]);s['roll_rate_radps']=np.array([.3,.4,-.1])
    r=apply_sign_options(s,flip_roll=True)
    np.testing.assert_allclose(r['roll_rad'],-s['roll_rad'])
    np.testing.assert_allclose(r['roll_rate_radps'],-s['roll_rate_radps'])
    np.testing.assert_allclose(s['roll_rad'],[.1,-.2,.05])
    assert r['normal_load_n'] is s['normal_load_n']
