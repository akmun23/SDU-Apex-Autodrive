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


def test_projected_scalar_tire_force_matches_every_node_exactly():
    s=seed()
    result, audit=project_algebraics(s,model(),Tire)
    fx,fy,*_=evaluate_scalar_tire_forces(model(),Tire,result,result['normal_load_n'])
    np.testing.assert_allclose(result['ax_mps2'],fx/3.47-.12*result['u_mps'],atol=1e-12)
    np.testing.assert_allclose(result['ay_mps2'],fy/3.47-.12*result['v_mps'],atol=1e-12)
    assert audit['scalar_vs_vectorized_tire_force_max_error_n'] < 1e-12
    assert audit['scalar_force_balance_residual_after_projection_mps2'] < 1e-12
    np.testing.assert_array_equal(result['wheel_omega_radps'],s['wheel_omega_radps'])


def test_scalar_vector_disagreement_detected_instead_of_labeled_consistent():
    output, audit = project_algebraics(seed(),model(),BatchIncompatibleTire)
    assert not audit['batched_tire_force_matches_scalar_ocp']
    assert audit['scalar_vs_vectorized_tire_force_max_error_n'] == pytest.approx(1.)
    f_x,_,*_=evaluate_scalar_tire_forces(model(),BatchIncompatibleTire,output,output['normal_load_n'])
    np.testing.assert_allclose(output['ax_mps2'],f_x/3.47-.12*output['u_mps'],atol=1e-12)


def test_forensic_report_residuals_are_truly_computed_from_scalar_model():
    summary, rows=analyze_seed(seed(),model(),Tire)
    assert summary['nodes']==len(rows)==8
    assert summary['projected_force_balance_max_residual_mps2'] < 1e-12
    assert summary['unprojected_lateral_force_balance_max_abs_mps2'] > .01
    residuals=np.array([r['unprojected_lateral_residual_mps2'] for r in rows])
    assert summary['unprojected_lateral_force_balance_max_abs_mps2']==pytest.approx(np.max(np.abs(residuals)))
    assert len(summary['worst_unprojected_lateral_nodes'])==8
    assert {'wheel_fl_fy_n','wheel_rr_native_fz_n','wheel_fr_slip_lat'} <= rows[0].keys()
    assert summary['symbolic_load_model_max_scaled_residual'] > 0


def test_physical_per_node_input_consistency_without_projection():
    s=seed()
    # Report must preserve the seed's original loads for unprojected balance.
    summary, rows=analyze_seed(s,model(),Tire)
    assert rows[0]['wheel_fl_native_fz_n']==pytest.approx(9.)
    assert rows[0]['wheel_fl_symbolic_fz_n']==pytest.approx(8.)
    for key in ('normal_load_n','ax_mps2','ay_mps2'):
        assert np.isfinite(s[key]).all()
