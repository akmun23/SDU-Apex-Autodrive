import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from recover import restore
import recover
from native_validate import _time_schedule
from source_vehicle import (_cpp_number, _quat_rotation,
                            _unity_scene_vehicle_parameters)


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


def test_time_integral_uses_actual_spatial_speed():
    n=100
    s=np.linspace(0,20,n,endpoint=False)
    angle=np.arange(n)*2*np.pi/n
    radius=20./(2*n*np.sin(np.pi/n))
    x,y=radius*np.cos(angle),radius*np.sin(angle)
    heading=np.arctan2(np.roll(y,-1)-y,np.roll(x,-1)-x)
    track=SimpleNamespace(s=s,length=20.,count=n,kappa=np.zeros(n),
                          x=x,y=y,psi=heading)
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


def test_practice_scene_torque_override_matches_cawd_brake_semantics():
    repo=Path(__file__).resolve().parents[2]
    config=_unity_scene_vehicle_parameters(repo)
    plant=(repo/'f1tenth_mpc/src/unity_source_plant.cpp').read_text()
    assert config['prefab_default_motor_torque_nm']==pytest.approx(85.6)
    assert config['scene_motor_torque_override_nm']==pytest.approx(428.0)
    assert _cpp_number(plant,'kMotorTorqueNm')==pytest.approx(
        config['scene_motor_torque_override_nm'])
    assert config['drive_type']=='CAWD'
    assert config['brake_type']=='CAWB'
    assert config['drive_torque_per_wheel_at_full_command_nm']==pytest.approx(107.0)
    assert config['brake_torque_per_wheel_at_zero_command_nm']==pytest.approx(428.0)


def test_checked_in_wheel_geometry_overrides_rounded_legacy(tmp_path):
    from source_vehicle import bind_current_wheel_moment_arms
    repo=tmp_path/'repo';src=repo/'f1tenth_mpc/src';src.mkdir(parents=True)
    (src/'unity_source_plant.cpp').write_text('''
    std::array<double,4> kWheelX{.17468,.17468,-.15532,-.15532};
    std::array<double,4> kWheelY{.115741,-.115901,.11792,-.11808};
    std::array<double,4> kWheelColliderTransformZCom{.020,.020,.020,.020};
    constexpr double kWheelRadiusM = .059;
    ''')
    a=SimpleNamespace();b=SimpleNamespace()
    meta=bind_current_wheel_moment_arms(repo,a,b)
    assert meta['y_m'][0]==pytest.approx(.115741)
    assert meta['force_application_z_m']==pytest.approx((-.039,)*4)
    assert a.WHEEL_Y_M==b.WHEEL_Y_M
