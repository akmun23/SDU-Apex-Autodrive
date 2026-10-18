import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from recover import restore
import recover
from run_ipopt import brake_seed
from native_validate import _time_schedule
from source_vehicle import _cpp_number,_quat_rotation


def test_historical_git_archive_restoration(tmp_path,monkeypatch):
    repo=tmp_path/'repo';repo.mkdir()
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'config','user.name','Test'],check=True)
    subprocess.run(['git','-C',str(repo),'config','user.email','test@example.org'],check=True)
    original=repo/'removed/optimizer_unity_wheels.py'
    original.parent.mkdir()
    original.write_text('full_ipopt=True\n')
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'commit','-m','old optimizer'],check=True,capture_output=True)
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    original.unlink()
    monkeypatch.setattr(recover,'REF',sha)
    monkeypatch.setattr(recover,'PATHS',['removed/optimizer_unity_wheels.py'])
    monkeypatch.setattr(recover,'LEGACY','removed')
    dest=tmp_path/'out'
    assert restore(repo,dest)==dest
    assert (dest/'removed/optimizer_unity_wheels.py').read_text()=='full_ipopt=True\n'


def test_brake_seed_is_periodic_and_has_resolved_sections():
    n=120
    xx=np.linspace(0,2*math.pi,n,endpoint=False)
    kappa=1+np.sin(2*xx)**2
    track=SimpleNamespace(count=n,kappa=kappa)
    b=brake_seed(track,3)
    assert b.shape==(n,)
    assert not b[0] and not b[-1]
    assert b.sum()>0
    changes=np.flatnonzero(b[1:]!=b[:-1])+1
    for left,right in zip(np.r_[0,changes],np.r_[changes,n]):
        assert right-left>=6
    assert not brake_seed(track,0).any()


def test_time_integral_uses_actual_spatial_speed():
    n=100
    s=np.linspace(0,20,n,endpoint=False)
    track=SimpleNamespace(s=s,length=20.,kappa=np.zeros(n))
    candidate=dict(s_ref_m=s,speed_mps=np.full(n,5.),
                   lateral_speed_mps=np.zeros(n),epsi_rad=np.zeros(n),
                   ey_m=np.zeros(n))
    time,lap=_time_schedule(candidate,track)
    assert len(time)==n+1
    assert lap==pytest.approx(4.)
    assert np.all(np.diff(time)>0.)


def test_sim_specific_cpp_numbers_are_strict():
    code='constexpr double kMotorTorqueNm = 428.0;'
    assert _cpp_number(code,'kMotorTorqueNm')==428.
    with pytest.raises(RuntimeError,match='Cannot resolve'):
        _cpp_number(code,'arbitraryTorque')
    assert np.allclose(_quat_rotation((0,0,0,1)),np.eye(3))


def test_checked_in_wheel_geometry_overrides_rounded_legacy(tmp_path):
    from source_vehicle import bind_current_wheel_moment_arms
    repo=tmp_path/'repo';src=repo/'f1tenth_mpc/src';src.mkdir(parents=True)
    (src/'unity_source_plant.cpp').write_text('''
    std::array<double,4> kWheelX{.17468,.17468,-.15532,-.15532};
    std::array<double,4> kWheelY{.115741,-.115901,.11792,-.11808};
    ''')
    a=SimpleNamespace();b=SimpleNamespace()
    meta=bind_current_wheel_moment_arms(repo,a,b)
    assert meta['y_m'][0]==pytest.approx(.115741)
    assert a.WHEEL_Y_M==b.WHEEL_Y_M
