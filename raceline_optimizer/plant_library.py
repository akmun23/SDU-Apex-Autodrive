"""Build and load the production MPC vehicle plant through its C ABI."""
from __future__ import annotations

import ctypes as C
import hashlib
import math
from pathlib import Path
import subprocess
from dataclasses import dataclass


SOURCE_PLANT_STATE_SIZE = 26  # Must match UNITY_SOURCE_PLANT_STATE_SIZE in unity_source_plant_c.h.
SOURCE_PLANT_CORE_STATE_SIZE = 18  # IPOPT seed stores the 18 optimized physical states.
STEER_MAX=math.pi/6


def source_hash(paths):
    h=hashlib.sha256()
    for p in paths:
        h.update(str(p).encode());h.update(Path(p).read_bytes())
    return h.hexdigest()


@dataclass
class State:
    x: float
    y: float
    yaw: float
    u: float
    v: float
    r: float
    steering: float
    longitudinal_slip: list[float]
    lateral_slip: list[float]
    normal_load: list[float]


class _CInput(C.Structure):
    # Exact field order and dimensions from UnitySourcePlantInputC in
    # unity_source_plant_c.h. Truncating state shifts every following member.
    _fields_=[('state',C.c_double*SOURCE_PLANT_STATE_SIZE),
              ('wheel_rotation_angle_rad',C.c_double*4),
              ('steering_target_rad',C.c_double),
              ('throttle_normalized',C.c_double),
              ('suspension_history_initialized',C.c_int)]


class _COutput(C.Structure):
    # Exact UnitySourcePlantOutputC layout. The C function writes the complete
    # structure even when a warm-start caller consumes only some fields.
    _fields_=[('state',C.c_double*SOURCE_PLANT_STATE_SIZE),
              ('integrated_body_forward_displacement_m',C.c_double),
              ('integrated_body_lateral_displacement_m',C.c_double),
              ('integrated_yaw_delta_rad',C.c_double),
              ('wheel_rotation_angle_rad',C.c_double*4),
              ('suspension_history_initialized',C.c_int),
              ('longitudinal_slip',C.c_double*4),
              ('maximum_positive_longitudinal_slip',C.c_double*4),
              ('maximum_grip_loss',C.c_double*4),
              ('lateral_slip',C.c_double*4),
              ('normal_load_n',C.c_double*4),
              ('suspension_force_n',C.c_double*4),
              ('suspension_jounce_rate_mps',C.c_double*4),
              ('tire_longitudinal_force_n',C.c_double*4),
              ('tire_lateral_force_n',C.c_double*4),
              ('longitudinal_acceleration_mps2',C.c_double),
              ('lateral_acceleration_mps2',C.c_double),
              ('yaw_acceleration_radps2',C.c_double)]


class SourcePlant:
    """Compile the production MPC plant and advance its state through the C ABI."""
    kind='source-derived'
    def __init__(self, repo: Path, build: Path):
        self.repo=Path(repo)
        code=[self.repo/'f1tenth_mpc/src/unity_source_plant.cpp',
              self.repo/'f1tenth_mpc/src/unity_source_plant_c.cpp',
              self.repo/'f1tenth_mpc/include/unity_source_plant.hpp',
              self.repo/'f1tenth_mpc/include/unity_source_plant_c.h']
        if not all(p.is_file() for p in code):
            raise FileNotFoundError('MPC plant source unavailable in --repo-root: '+str(code[0]))
        self.code_hash=source_hash(code)
        build.mkdir(parents=True,exist_ok=True)
        so=build/('lib_apex_source_'+self.code_hash[:16]+'.so')
        if not so.is_file():
            cmd=['g++','-std=c++17','-O3','-DNDEBUG','-ffp-contract=off',
                 '-shared','-fPIC','-I'+str(self.repo/'f1tenth_mpc/include'),
                 str(code[0]),str(code[1]),'-o',str(so)]
            subprocess.run(cmd,check=True)
        self.lib=C.CDLL(str(so))
        self.lib.unity_source_plant_step_c.argtypes=[C.POINTER(_CInput),C.c_double,C.POINTER(_COutput)]
        self.lib.unity_source_plant_step_c.restype=C.c_int
        self.input=_CInput()
        self.output=_COutput()
        self.state=None

    def reset(self,x,y,yaw,speed):
        arr=[0.]*SOURCE_PLANT_STATE_SIZE
        arr[0]=float(speed)
        for i in range(3,7): arr[i]=speed/.059
        arr[7]=.00464804098
        self.input=_CInput()
        self.input.state[:]=arr
        self.input.wheel_rotation_angle_rad[:]=[0.0]*4
        self.input.suspension_history_initialized=0
        self.state=State(float(x),float(y),float(yaw),float(speed),0.,0.,0.,[0.]*4,[0.]*4,[0.]*4)
        return self.state

    def step(self,steer,throttle,*,duration_s=.025):
        inp=self.input;out=self.output
        inp.steering_target_rad=max(-STEER_MAX,min(STEER_MAX,float(steer)))
        inp.throttle_normalized=max(0.,min(1.,float(throttle)))
        if not self.lib.unity_source_plant_step_c(C.byref(inp),C.c_double(duration_s),C.byref(out)):
            raise RuntimeError('production source plant rejected vehicle transition')
        old=self.state
        # The production plant already integrates rear-axle pose at every
        # source substep. Re-integrating endpoint COM velocities here both
        # discards that accuracy and uses the wrong lateral velocity origin.
        dx=out.integrated_body_forward_displacement_m
        dy=out.integrated_body_lateral_displacement_m
        c,s=math.cos(old.yaw),math.sin(old.yaw)
        xx=old.x+c*dx-s*dy
        yy=old.y+s*dx+c*dy
        self.state=State(xx,yy,old.yaw+out.integrated_yaw_delta_rad,out.state[0],out.state[1],out.state[2],
                         out.state[13],list(out.longitudinal_slip),list(out.lateral_slip),
                         list(out.normal_load_n))
        inp.state[:]=out.state[:]
        inp.wheel_rotation_angle_rad[:]=out.wheel_rotation_angle_rad[:]
        inp.suspension_history_initialized=out.suspension_history_initialized
        return self.state

    def close(self): pass
