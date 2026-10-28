"""v2.3.8 source-wheel scalar projection checks; no IPOPT or native physics."""
import numpy as np
import casadi as ca
import pytest
from types import SimpleNamespace

from seed_projection import project_algebraics
from diagnose_force_consistency_v238 import analyze_seed, evaluate_scalar_tire_forces


class Tire:
    @staticmethod
    def suspension_loads_numpy(model, heave, hd, roll, rd, pitch, pd):
        return [8.+2*roll, 8.2-2*roll, 8.4+pitch, 8.6-pitch]

    @staticmethod
    def tire_forces_casadi(model, u, v, r, steer, omegas, loads):
        fxw = [loads[i] * (omegas[i]*.059-u)*.1 for i in range(4)]
        fyw = [loads[i] * (r*.3-v+steer)*.4 for i in range(4)]
        sx = [omegas[i]*.059-u for i in range(4)]
        sy = [r*.3-v+steer for i in range(4)]
        return sum(fxw), sum(fyw), 0, tuple(fxw), tuple(fyw), tuple(sx), tuple(sy)


class SourceShapeTire(Tire):
    @staticmethod
    def tire_forces_casadi(model, u, v, r, steer, omegas, loads):
        fx, fy, _, fxw, fyw, sx, sy = Tire.tire_forces_casadi(
            model, u, v, r, steer, omegas, loads)
        return fx, fy, 0.0, 0.0, 0.0, 0.0, fxw, fyw, sx, sy


def flat_source_suspension_loads(model, height, height_rate, roll,
                                 roll_rate, pitch, pitch_rate, yaw_rate):
    return Tire.suspension_loads_numpy(
        model, height, height_rate, roll, 0.0, pitch, 0.0)


class BatchIncompatibleTire(Tire):
    @staticmethod
    def tire_forces_casadi(model, u, v, r, steer, omegas, loads):
        fx, fy, moment, fxw, fyw, sx, sy = Tire.tire_forces_casadi(
            model,u,v,r,steer,omegas,loads)
        # A deliberate regression: batched tire forces do NOT match per-node.
        return (fx + (1. if int(u.numel())>1 else 0.), fy, moment,
                fxw,fyw,sx,sy)


def seed(n=8):
    x=np.arange(n)
    return dict(ey_m=np.zeros(n),epsi_rad=np.zeros(n),u_mps=2.+x*.02,
        v_mps=np.linspace(-.3,.2,n),yaw_rate_radps=np.linspace(-.5,.7,n),
        wheel_omega_radps=np.full((4,n),40.),steering_rad=np.linspace(-.1,.2,n),
        steering_rate_radps=np.zeros(n),throttle_norm=np.full(n,.09),
        ax_mps2=np.full(n,0.),ay_mps2=np.linspace(-1.,1.,n),
        normal_load_n=np.full((4,n),9.),body_height_m=np.zeros(n),
        body_height_rate_mps=np.zeros(n),roll_rad=np.zeros(n),
        roll_rate_radps=np.zeros(n),pitch_rad=np.zeros(n),
        pitch_rate_radps=np.zeros(n))


def model():
    return SimpleNamespace(mass_kg=3.47,drag_per_s=.12)


def flat_source_acceleration(model, fx, fy, u, v, yaw_rate,
                             body_height_rate, roll, roll_rate,
                             pitch, pitch_rate):
    damping = -np.log1p(-model.drag_per_s * 0.001) / 0.001
    return (fx/model.mass_kg - damping*u + yaw_rate*v,
            fy/model.mass_kg - damping*v - yaw_rate*u)


def test_projected_scalar_tire_force_matches_every_node_exactly():
    s=seed()
    result, audit=project_algebraics(
        s,model(),Tire,suspension_loads_fn=flat_source_suspension_loads,
        body_acceleration_fn=flat_source_acceleration)
    fx,fy,*_=evaluate_scalar_tire_forces(model(),Tire,result,result['normal_load_n'])
    expected = np.asarray([
        flat_source_acceleration(model(), fx[k], fy[k], result['u_mps'][k],
            result['v_mps'][k], result['yaw_rate_radps'][k], 0., 0., 0., 0., 0.)
        for k in range(len(fx))])
    np.testing.assert_allclose(result['ax_mps2'],expected[:,0],atol=1e-12)
    np.testing.assert_allclose(result['ay_mps2'],expected[:,1],atol=1e-12)
    assert audit['scalar_vs_vectorized_tire_force_max_error_n'] < 1e-12
    assert audit['scalar_force_balance_residual_after_projection_mps2'] < 1e-12
    np.testing.assert_array_equal(result['wheel_omega_radps'],s['wheel_omega_radps'])


def test_scalar_vector_disagreement_detected_instead_of_labeled_consistent():
    output, audit = project_algebraics(
        seed(),model(),BatchIncompatibleTire,
        suspension_loads_fn=flat_source_suspension_loads,
        body_acceleration_fn=flat_source_acceleration)
    assert not audit['batched_tire_force_matches_scalar_ocp']
    assert audit['scalar_vs_vectorized_tire_force_max_error_n'] == pytest.approx(1.)
    f_x,_,*_=evaluate_scalar_tire_forces(model(),BatchIncompatibleTire,output,output['normal_load_n'])
    expected = np.asarray([
        flat_source_acceleration(model(), f_x[k], 0., output['u_mps'][k],
            output['v_mps'][k], output['yaw_rate_radps'][k], 0., 0., 0., 0., 0.)[0]
        for k in range(len(f_x))])
    np.testing.assert_allclose(output['ax_mps2'],expected,atol=1e-12)


def test_forensic_report_residuals_are_truly_computed_from_scalar_model():
    summary, rows=analyze_seed(
        seed(),model(),Tire,flat_source_suspension_loads,
        flat_source_acceleration)
    assert summary['nodes']==len(rows)==8
    assert summary['projected_force_balance_max_residual_mps2'] < 1e-12
    assert summary['unprojected_lateral_force_balance_max_abs_mps2'] > .01
    residuals=np.array([r['unprojected_lateral_residual_mps2'] for r in rows])
    assert summary['unprojected_lateral_force_balance_max_abs_mps2']==pytest.approx(np.max(np.abs(residuals)))
    assert len(summary['worst_unprojected_lateral_nodes'])==8
    assert {'wheel_fl_fy_n','wheel_rr_native_fz_n','wheel_fr_slip_lat'} <= rows[0].keys()
    assert summary['symbolic_load_model_max_scaled_residual'] > 0


def test_forensic_report_decodes_active_ten_value_tire_callback():
    summary, rows = analyze_seed(
        seed(), model(), SourceShapeTire, flat_source_suspension_loads,
        flat_source_acceleration)
    assert summary['nodes'] == len(rows) == 8
    assert summary['projected_force_balance_max_residual_mps2'] < 1e-12
    assert np.isfinite([row['total_fy_projected_load_n'] for row in rows]).all()
    assert 'wheel_fl_slip_lat' in rows[0]


def test_physical_per_node_input_consistency_without_projection():
    s=seed()
    # Report must preserve the seed's original loads for unprojected balance.
    summary, rows=analyze_seed(s,model(),Tire,
                               flat_source_suspension_loads,
                               flat_source_acceleration)
    assert rows[0]['wheel_fl_native_fz_n']==pytest.approx(9.)
    assert rows[0]['wheel_fl_symbolic_fz_n']==pytest.approx(8.)
    for key in ('normal_load_n','ax_mps2','ay_mps2'):
        assert np.isfinite(s[key]).all()
