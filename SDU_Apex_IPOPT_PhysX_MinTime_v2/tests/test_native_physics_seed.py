"""Native-warm-start mechanics: no native simulator or full IPOPT run launched."""
import math
import numpy as np
import pytest
from types import SimpleNamespace
from native_seed import build_seed,command,generate


def circle(length=20.0,n=100):
    ss=np.linspace(0,length,n,endpoint=False)
    radius=length/(2*math.pi)
    angle=2*math.pi*ss/length
    xy=np.column_stack([radius*np.cos(angle), radius*np.sin(angle)])
    return SimpleNamespace(s=ss,length=length,x=xy[:,0],y=xy[:,1],
                           psi=angle+math.pi/2, kappa=np.ones(n)/radius)


def trace_circle(track,count=125):
    result=[]
    radius=track.length/(2*math.pi)
    for j in range(count):
        s=track.length*(1+(j+.2)/count)
        t=s/2
        angle=2*math.pi*s/track.length
        state=np.zeros(18)
        state[0]=2.0;state[2]=2./radius
        state[3:7]=2/.059
        state[7]=.00464;state[13]=.10
        result.append((s,t,radius*math.cos(angle),radius*math.sin(angle),
                       angle+math.pi/2,.080,state,np.ones(4)*8))
    return result


def test_native_state_seed_shapes_and_values():
    track=circle()
    seed=build_seed(trace_circle(track),track,track.length)
    assert seed['ey_m'].shape==(len(track.s),)
    assert max(abs(seed['ey_m']))<0.002
    assert max(abs(seed['epsi_rad']))<1e-4
    assert seed['wheel_omega_radps'].shape==(4,len(track.s))
    assert seed['normal_load_n'].shape==(4,len(track.s))
    assert abs(seed['wheel_omega_radps'].mean()-2/.059)<1e-5
    assert np.all(seed['u_mps']>=.36)
    assert np.all(seed['throttle_norm']>=.015)
    assert abs(seed['body_height_m'].mean()-.00464)<1e-6
    assert np.isfinite(seed['ax_mps2']).all()
    assert np.isfinite(seed['ay_mps2']).all()


def test_incomplete_or_ambiguous_lap_rejected():
    track=circle()
    with pytest.raises(ValueError,match='complete lap'):
        build_seed(trace_circle(track)[:20],track,track.length)
    data=trace_circle(track)
    with pytest.raises(ValueError,match='strictly increasing'):
        build_seed(data[:2]+[data[1]]+data[3:],track,track.length)


def test_feedback_turns_back_toward_raceline():
    class Ref:
        def at(self,station):return 0.,0.,0.,0.
    l=command(Ref(),0.,.12,0.,0.,2.,2.)[0]
    r=command(Ref(),0.,-.12,0.,0.,2.,2.)[0]
    assert l<0<r
    assert command(Ref(),0.,0.,0.,0.,2.,2.)[1]>0


def test_warmstart_failure_keeps_ipopt_unlaunched(monkeypatch,tmp_path):
    import native_seed
    def fail(*args,**kwargs):return None, {'failure':'footprint_clearance','completed_flying_lap':False}
    monkeypatch.setattr(native_seed,'one_trial',fail)
    s,r=native_seed.generate(None,None,'native',None,None,None,tmp_path,.15,speeds=(2.4,1.8))
    assert s is None and len(r)==2
    assert 'footprint_clearance' in (tmp_path/'native_seed_report.json').read_text()


def test_completed_measured_flying_lap_is_passed_to_optimizer(monkeypatch,tmp_path):
    import native_seed
    ref_track=circle(n=120)
    radius=ref_track.length/(2*math.pi)
    class Ref:
        length=ref_track.length
        def at(self,s):
            angle=2*math.pi*s/self.length
            return radius*math.cos(angle),radius*math.sin(angle),angle+math.pi/2,1/radius
        def project(self,x,y,previous_s=None):
            angle=math.atan2(y,x)
            base=(angle%(2*math.pi))*radius
            if previous_s is not None:
                base+=self.length*round((previous_s-base)/self.length)
            return base,0.,angle+math.pi/2
    class Grid:
        resolution=.025
        def footprint_clearance(self,*args,**kwargs):return 1.0
    class FakeNative:
        def __init__(self,*args,**kwargs):self.last_source_state=None
        def reset(self,x,y,yaw,speed):
            self.spd=speed;self.angle=0.
            return SimpleNamespace(x=x,y=y,yaw=yaw,u=speed,v=0.,r=speed/radius)
        def step(self,steer,throttle):
            self.angle+=self.spd/native_seed.L * 0 + self.spd*native_seed.DT/radius
            raw=np.zeros(18);raw[0]=self.spd;raw[2]=self.spd/radius
            raw[3:7]=self.spd/.059;raw[7]=.004648;raw[13]=steer
            self.last_source_state=raw
            return SimpleNamespace(x=radius*math.cos(self.angle),y=radius*math.sin(self.angle),
                yaw=self.angle+math.pi/2,u=self.spd,v=0.,r=self.spd/radius,
                normal_load=[8.5]*4)
        def close(self):pass
    monkeypatch.setattr(native_seed,'NativePhysX',FakeNative)
    seed, report=native_seed.one_trial(tmp_path/'repo',None,'native',Ref(),Grid(),
                                      ref_track,tmp_path,2.4,.15,max_seconds=25.)
    assert report['completed_flying_lap'] and report['failure'] is None
    assert seed['wheel_omega_radps'].shape==(4,len(ref_track.s))
    assert max(abs(seed['epsi_rad']))<1e-3
    assert (tmp_path/'native_warmstart.npz').exists()
