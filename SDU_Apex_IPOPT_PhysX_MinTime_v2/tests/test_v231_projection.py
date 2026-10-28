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


def flat_source_suspension_loads(model, height, height_rate, roll,
                                 roll_rate, pitch, pitch_rate, yaw_rate):
    return FakeWheel.suspension_loads_numpy(
        model, height, height_rate, roll, 0.0, pitch, 0.0)


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


def flat_source_acceleration(model, fx, fy, u, v, yaw_rate,
                             body_height_rate, roll, roll_rate,
                             pitch, pitch_rate):
    damping = -np.log1p(-model.drag_per_s * 0.001) / 0.001
    return (fx/model.mass_kg - damping*u + yaw_rate*v,
            fy/model.mass_kg - damping*v - yaw_rate*u)


def test_algebraic_projection_is_exact_and_preserves_measured_pose_wheel_motion():
    x=seed();original={k:v.copy() for k,v in x.items()}
    model=SimpleNamespace(mass_kg=3.47,drag_per_s=.273)
    out,audit=project_algebraics(
        x,model,FakeWheel,
        suspension_loads_fn=flat_source_suspension_loads,
        body_acceleration_fn=flat_source_acceleration)
    damping = -np.log1p(-.273 * .001) / .001
    expected_ax=(2*2.2+.1*40)/3.47-damping*2.2+.8*(-.2)
    expected_ay=(3*.8-2*(-.2)+.15)/3.47-damping*(-.2)-.8*2.2
    np.testing.assert_allclose(out['ax_mps2'],expected_ax,atol=1e-12)
    np.testing.assert_allclose(out['ay_mps2'],expected_ay,atol=1e-12)
    for k in ('ey_m','epsi_rad','u_mps','v_mps','yaw_rate_radps','wheel_omega_radps','steering_rad','throttle_norm'):
        np.testing.assert_array_equal(out[k],original[k])
    np.testing.assert_array_equal(x['ax_mps2'],original['ax_mps2'])
    assert audit['native_to_symbolic_load_max_abs_n'] > .4
    assert not any('brake' in name.lower() or 'zero_throttle' in name.lower()
                   for name in audit)
    assert audit['spatial_collocation_or_wheel_update_are_not_reconciled']


def test_drive_seed_throttle_is_projected_to_runtime_mpc_envelope():
    x = seed()
    x['throttle_norm'][0] = 0.0
    x['throttle_norm'][1] = 0.005
    out, audit = project_algebraics(
        x, SimpleNamespace(mass_kg=3.47, drag_per_s=.273), FakeWheel,
        suspension_loads_fn=flat_source_suspension_loads,
        body_acceleration_fn=flat_source_acceleration,
        minimum_drive_throttle=0.01)
    assert out['throttle_norm'][0] == pytest.approx(0.01)
    assert out['throttle_norm'][1] == pytest.approx(0.01)
    assert audit['optimizer_minimum_drive_throttle'] == pytest.approx(0.01)
    assert audit['native_drive_throttle_values_clamped'] == 2


def test_projection_has_no_brake_mode_api():
    import inspect
    assert 'brake_mask' not in inspect.signature(project_algebraics).parameters
    assert not any('brake' in name.lower() or 'zero_throttle' in name.lower()
                   for name in project_algebraics.__code__.co_names)


def test_native_npz_reuse_validates_content(tmp_path):
    good=seed()
    path=tmp_path/'seed.npz';np.savez_compressed(path,**good)
    assert len(load_native_seed(path,8))==len(good)
    with pytest.raises(ValueError,match='shape'):
        load_native_seed(path,9)
    bad=seed();bad['yaw_rate_radps'][4]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        validate_native_seed(bad,8)


def test_legacy_seed_acceleration_is_converted_to_active_ocp_derivatives(tmp_path):
    source = seed(8)
    source['u_mps'][:] = 2.0
    source['v_mps'][:] = 0.25
    source['yaw_rate_radps'][:] = 0.6
    expected_ax = np.full(8, 0.3)
    expected_ay = np.full(8, -0.4)
    source['ax_mps2'][:] = expected_ax - source['yaw_rate_radps']*source['v_mps']
    source['ay_mps2'][:] = expected_ay + source['yaw_rate_radps']*source['u_mps']
    path = tmp_path/'legacy_acceleration_seed.npz'
    np.savez_compressed(path, **source)
    loaded = load_native_seed(path, 8)
    np.testing.assert_allclose(loaded['ax_mps2'], expected_ax, atol=1e-12)
    np.testing.assert_allclose(loaded['ay_mps2'], expected_ay, atol=1e-12)


def test_marked_derivative_seed_is_not_converted_twice(tmp_path):
    source = seed(8)
    source['ax_mps2'][:] = 0.3
    source['ay_mps2'][:] = -0.4
    path = tmp_path/'marked_acceleration_seed.npz'
    np.savez_compressed(path, **source,
                        acceleration_convention=np.asarray(
                            'body_velocity_derivative_v1'))
    loaded = load_native_seed(path, 8)
    np.testing.assert_allclose(loaded['ax_mps2'], 0.3, atol=1e-12)
    np.testing.assert_allclose(loaded['ay_mps2'], -0.4, atol=1e-12)


def test_native_seed_mesh_resample_requires_same_verified_lap_length(tmp_path):
    source = seed(8)
    source['u_mps'] = 2.2 + .2*np.sin(2*np.pi*np.arange(8)/8)
    source['wheel_omega_radps'] = np.tile(source['u_mps'], (4, 1))*10
    path = tmp_path/'seed_resample.npz'
    np.savez_compressed(path, **source)
    resampled = load_native_seed(path, 16, source_length_m=28.41,
                                 target_length_m=28.41)
    assert resampled['u_mps'].shape == (16,)
    assert resampled['wheel_omega_radps'].shape == (4, 16)
    np.testing.assert_allclose(resampled['u_mps'][::2], source['u_mps'], atol=1e-12)
    assert resampled['u_mps'][1] == pytest.approx(
        .5*(source['u_mps'][0]+source['u_mps'][1]))
    with pytest.raises(ValueError, match='different lap lengths'):
        load_native_seed(path, 16, source_length_m=28.41,
                         target_length_m=28.40)
    with pytest.raises(ValueError, match='matching track lengths'):
        load_native_seed(path, 16)


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
