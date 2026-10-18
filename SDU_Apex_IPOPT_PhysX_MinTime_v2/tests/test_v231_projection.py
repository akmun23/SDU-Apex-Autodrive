"""v2.3.1 does not run the native simulator or a full lap-time NLP."""
from __future__ import annotations
import numpy as np
import casadi as ca
import pytest
from types import SimpleNamespace
from seed_projection import validate_native_seed, load_native_seed, project_algebraics
from solver_overlay import transform_solver, transform_solver_v229, prepare_working_solver
from test_phase1_overlay import _sample, LEGACY, SOLVER

class FakeWheel:
    @staticmethod
    def suspension_loads_numpy(model, heave, hd, roll, rd, pitch, pd):
        return np.array([8., 8.2, 8.4, 8.6])
    @staticmethod
    def tire_forces_casadi(model, u, v, r, steer, omegas, loads):
        # CasADi array ops are the source of truth, as in the real wheel model.
        fx=2*u + .1*omegas[0]; fy=3*r-2*v + steer
        return fx, fy, None, None, None, None, None


def seed(n=8):
    return dict(
        ey_m=np.zeros(n),epsi_rad=np.zeros(n),
        u_mps=np.full(n,2.2),v_mps=np.full(n,-.2),
        yaw_rate_radps=np.full(n,.8),wheel_omega_radps=np.full((4,n),40.),
        steering_rad=np.full(n,.15),steering_rate_radps=np.zeros(n),
        throttle_norm=np.full(n,.09),ax_mps2=np.full(n,99.),ay_mps2=np.full(n,99.),
        normal_load_n=np.full((4,n),9.),body_height_m=np.zeros(n),
        body_height_rate_mps=np.zeros(n),roll_rad=np.zeros(n),
        roll_rate_radps=np.zeros(n),pitch_rad=np.zeros(n),pitch_rate_radps=np.zeros(n))


def test_algebraic_projection_is_exact_and_preserves_measured_pose_wheel_motion():
    x=seed();original={k:v.copy() for k,v in x.items()}
    model=SimpleNamespace(mass_kg=3.47,drag_per_s=.273)
    out,audit=project_algebraics(x,model,FakeWheel)
    expected_ax=(2*2.2+.1*40)/3.47-.273*2.2
    expected_ay=(3*.8-2*(-.2)+.15)/3.47-.273*(-.2)
    np.testing.assert_allclose(out['ax_mps2'],expected_ax,atol=1e-12)
    np.testing.assert_allclose(out['ay_mps2'],expected_ay,atol=1e-12)
    for k in ('ey_m','epsi_rad','u_mps','v_mps','yaw_rate_radps','wheel_omega_radps','steering_rad','throttle_norm'):
        np.testing.assert_array_equal(out[k],original[k])
    np.testing.assert_array_equal(x['ax_mps2'],original['ax_mps2'])
    assert audit['native_to_symbolic_load_max_abs_n'] > .4
    assert audit['forced_full_brake_nodes_unseen_in_native']==0
    assert audit['spatial_collocation_or_wheel_update_are_not_reconciled']


def test_full_brake_seed_mode_cannot_claim_measured_consistency():
    x=seed();mask=np.zeros(8,dtype=bool);mask[2:4]=True
    out,audit=project_algebraics(x,SimpleNamespace(mass_kg=3.47,drag_per_s=.273),FakeWheel,brake_mask=mask)
    assert audit['forced_full_brake_nodes_unseen_in_native']==2
    assert audit['native_brake_nodes']==0
    assert audit['brake_locked_wheel_rpm_delta_max_radps']==40.
    assert (out['wheel_omega_radps'][:,mask]==0).all()
    assert (out['throttle_norm'][mask]==0).all()
    assert (x['wheel_omega_radps'][:,mask]==40).all()


def test_native_npz_reuse_validates_content(tmp_path):
    good=seed()
    path=tmp_path/'seed.npz';np.savez_compressed(path,**good)
    assert len(load_native_seed(path,8))==len(good)
    with pytest.raises(ValueError,match='shape'):
        load_native_seed(path,9)
    bad=seed();bad['yaw_rate_radps'][4]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        validate_native_seed(bad,8)


def test_existing_v230_solver_cache_is_upgradeable(tmp_path):
    source_root,original,source=_sample(tmp_path)
    dest=tmp_path/'working'/LEGACY/SOLVER
    dest.parent.mkdir(parents=True)
    cached=transform_solver_v229(source,'limited-memory')
    dest.write_text(cached)
    prepare_working_solver(source_root,tmp_path/'working',LEGACY)
    updated=dest.read_text()
    assert updated != cached
    assert updated==transform_solver(source,'limited-memory')
    assert original.read_text()==source
    assert 'ipopt_projected_iteration' in updated
    assert 'opti.set_value(elastic_cap, 0.0)' in updated


def test_projected_elastic_equalities_must_respect_hard_track_limit():
    # The elasticity equation can be repaired without altering physical x.
    # Projection is not permission to cross a hard lane boundary.
    op=ca.Opti();x=op.variable();e=op.variable()
    op.subject_to(x**2-e==0);op.subject_to(x<=.5)
    op.set_initial(x,.4);op.set_initial(e,1.)
    projected=.4**2
    op.set_initial(e,projected)
    g=np.asarray(op.value(op.g,op.initial())).reshape(-1)
    low=np.asarray(op.value(op.lbg,op.initial())).reshape(-1)
    high=np.asarray(op.value(op.ubg,op.initial())).reshape(-1)
    assert max(float(np.max(low-g)),float(np.max(g-high)),0.)<1e-8
    op.set_initial(x,.6);op.set_initial(e,.6**2)
    g=np.asarray(op.value(op.g,op.initial())).reshape(-1)
    assert max(float(np.max(low-g)),float(np.max(g-high)),0.)>.09


def test_saved_seed_requires_matching_map_and_centerline_checksum(tmp_path):
    import json, hashlib
    from seed_projection import validate_seed_provenance
    output=tmp_path/'output';run=output/'warmstart_physx';run.mkdir(parents=True)
    np.savez_compressed(run/'native_warmstart.npz',**seed())
    mp=tmp_path/'practice.pgm';mp.write_bytes(b'practice map')
    ref=tmp_path/'track.csv';ref.write_bytes(b'practice centerline')
    provenance={'map_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
                'centerline_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),
                'requested_warmstart_backend':'native'}
    (output/'provenance.json').write_text(json.dumps(provenance))
    check=validate_seed_provenance(run/'native_warmstart.npz',mp,ref)
    assert check['map_sha256']==provenance['map_sha256']
    ref.write_bytes(b'different track')
    with pytest.raises(ValueError,match='does not match'):
        validate_seed_provenance(run/'native_warmstart.npz',mp,ref)
    ref.write_bytes(b'practice centerline')
    (output/'provenance.json').unlink()
    with pytest.raises(ValueError,match='no parent provenance'):
        validate_seed_provenance(run/'native_warmstart.npz',mp,ref)
