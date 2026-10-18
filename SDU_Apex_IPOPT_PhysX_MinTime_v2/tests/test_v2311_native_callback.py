import math
import numpy as np
from diagnose_native_tire_v2311 import (
    curve_response_without_stiffness, filtered_tire_load,
    predicted_native_callback, calculate_contact_audit)


def test_cpp_curve_preserves_stiffness_until_after_combined_norm():
    forward=(.15,.9,.25,.58,.8)
    sideways=(.01,1.,.1,.5,1.)
    slip_x=.16
    slip_y=.05
    raw_x=curve_response_without_stiffness(slip_x,forward)
    raw_y=curve_response_without_stiffness(slip_y,sideways)
    scale=1/math.sqrt(max(1.,(raw_x/.9)**2+(raw_y/1.)**2))
    fx,fy=predicted_native_callback(slip_x,slip_y,10.,forward,sideways)
    assert math.isclose(fx,10*raw_x*scale*.8,rel_tol=1e-12)
    assert math.isclose(fy,-10*raw_y*scale,rel_tol=1e-12)
    fx_old,fy_old=predicted_native_callback(
        slip_x,slip_y,10.,forward,sideways,True)
    assert abs(fx-fx_old)>0.01
    assert abs(fy-fy_old)>0.01


def test_callback_exact_curve_when_utilization_not_saturated():
    long=(.15,.9,.25,.58,.8)
    lat=(.01,1.,.1,.5,1.)
    assert predicted_native_callback(.01,0.,10.,long,lat) == (
        predicted_native_callback(.01,0.,10.,long,lat,True))


def test_tire_load_filter_and_native_observation_audit():
    forward=[.15,.9,.25,.58,.8]
    sideways=[.01,1.,.1,.5,1.]
    wheel=dict(sprung_mass_kg=.8172174,mass_kg=.109,
               forward_curve=forward,sideways_curve=sideways)
    load=filtered_tire_load(8.,wheel['sprung_mass_kg'],wheel['mass_kg'])
    assert 0 < load < 20
    predicted=predicted_native_callback(.15,.05,load,forward,sideways)
    record=dict(time_s=.05,progress_m=30.,u_mps=2.4,v_mps=.1,
                yaw_rate_radps=.4,steering_rad=.1,throttle_norm=.08,
                longitudinal_slip=np.full(4,.15),
                lateral_slip=np.full(4,.05),
                normal_load_n=np.full(4,8.),
                tire_longitudinal_force_n=np.full(4,predicted[0]),
                tire_lateral_force_n=np.full(4,predicted[1]))
    result,rows=calculate_contact_audit([record],[wheel]*4)
    assert len(rows)==4
    assert result['source_stiffness_outside_norm_rms_force_error_n']<1e-12
    assert result['historical_stiffness_inside_norm_rms_force_error_n']>.001
    assert result['inferred_callback_load_coverage_fraction']==1.


def test_empty_native_audit_fails_closed():
    import pytest
    with pytest.raises(ValueError,match='No native'):
        calculate_contact_audit([],[])
