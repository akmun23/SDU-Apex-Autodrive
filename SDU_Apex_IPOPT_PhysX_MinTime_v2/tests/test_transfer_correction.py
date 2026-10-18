import csv
import numpy as np
from types import SimpleNamespace
from transfer_correction import native_lateral_bias,candidate_warm_state


def test_native_lateral_correction_uses_observed_sector_only(tmp_path):
    s=np.linspace(0,30,150,endpoint=False)
    radius=30/(2*np.pi)
    theta=s/radius
    xy=np.column_stack([radius*np.cos(theta),radius*np.sin(theta)])
    psi=theta+np.pi/2
    track=SimpleNamespace(s=s,length=30.,psi=psi)
    candidate=dict(s_ref_m=s,x_m=xy[:,0],y_m=xy[:,1],
                   speed_mps=np.ones(len(s)),lateral_speed_mps=np.zeros(len(s)))
    f=tmp_path/'native_trace.csv'
    with f.open('w',newline='') as stream:
        wr=csv.writer(stream);wr.writerow(['progress_m','x_m','y_m'])
        # The first 8m has native drift +10cm along the left normal.
        for q in np.linspace(0,8,80):
            aa=q/radius
            p=np.array([radius*np.cos(aa),radius*np.sin(aa)])
            n=np.array([-np.sin(aa+np.pi/2),np.cos(aa+np.pi/2)])
            px=p+.10*n
            wr.writerow([q,*px])
    bias,info=native_lateral_bias(f,candidate,track)
    assert info['status']=='computed_from_native_open_loop_tracking_difference'
    assert np.median(bias[(s>=1)&(s<7)]) > .07
    assert not np.any(bias[(s>12)&(s<25)])


def test_ipopt_candidate_warm_state_maps_wheels_and_throttle():
    import transfer_correction as t
    src={v:np.arange(5.) for v in dict(
        u_mps='speed_mps',v_mps='lateral_speed_mps',
        ey_m='ey_m',epsi_rad='epsi_rad',yaw_rate_radps='yaw_rate_radps',
        wheel_omega_radps='wheel_omega_radps',steering_rad='steering_rad',
        steering_rate_radps='steering_rate_radps',throttle_norm='throttle_norm',
        ax_mps2='ax_mps2',ay_mps2='ay_mps2',normal_load_n='normal_load_n',
        body_height_m='body_height_m',body_height_rate_mps='body_height_rate_mps',
        roll_rad='roll_rad',roll_rate_radps='roll_rate_radps',
        pitch_rad='pitch_rad',pitch_rate_radps='pitch_rate_radps').values()}
    out=t.candidate_warm_state(src)
    assert set(out).issuperset({'u_mps','wheel_omega_radps','throttle_norm'})
    assert np.array_equal(out['u_mps'],src['speed_mps'])
