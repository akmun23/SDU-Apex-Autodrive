"""Directly execute the project's two existing source-specific vehicle plants.

`native`: public PhysX 4.1 PxVehicleNoDrive + Unity source configuration,
collision hulls and tire callback; stronger simulator fidelity, not identical
Unity player (documented source-hull/contact parity limitations).
`source`: fast project MPC C ABI; intended for warm-start only.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

REPO_DEFAULT='akmun23/SDU-Apex-Autodrive'
STEER_MAX=math.pi/6
MOTOR_TORQUE=428.0
WHEELBASE=.324
TRACK=.236
STEER_RATE=183.346


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
    _fields_=[('state',C.c_double*18),('steering_target_rad',C.c_double),
              ('throttle_normalized',C.c_double),('suspension_history_initialized',C.c_int)]


class _COutput(C.Structure):
    _fields_=[('state',C.c_double*18),('suspension_history_initialized',C.c_int),
              ('longitudinal_slip',C.c_double*4),('lateral_slip',C.c_double*4),
              ('normal_load_n',C.c_double*4),('suspension_force_n',C.c_double*4),
              ('suspension_jounce_rate_mps',C.c_double*4),
              ('tire_longitudinal_force_n',C.c_double*4),
              ('tire_lateral_force_n',C.c_double*4),
              ('longitudinal_acceleration_mps2',C.c_double),
              ('lateral_acceleration_mps2',C.c_double),
              ('yaw_acceleration_radps2',C.c_double)]


class SourcePlant:
    """Exactly calls the checked-out production MPC source-derived plant C ABI.

    For calibration / warm-start ONLY; does not implement native scene contacts.
    """
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
        arr=[0.]*18
        arr[0]=float(speed)
        for i in range(3,7): arr[i]=speed/.059
        arr[7]=.00464804098
        self.input=_CInput()
        self.input.state[:]=arr
        self.input.suspension_history_initialized=0
        self.state=State(float(x),float(y),float(yaw),float(speed),0.,0.,0.,[0.]*4,[0.]*4,[0.]*4)
        return self.state

    def step(self,steer,throttle):
        inp=self.input;out=self.output
        inp.steering_target_rad=max(-STEER_MAX,min(STEER_MAX,float(steer)))
        inp.throttle_normalized=max(0.,min(1.,float(throttle)))
        if not self.lib.unity_source_plant_step_c(C.byref(inp),C.c_double(.025),C.byref(out)):
            raise RuntimeError('production source plant rejected vehicle transition')
        old=self.state
        u=.5*(old.u+out.state[0]);v=.5*(old.v+out.state[1]);r=.5*(old.r+out.state[2])
        yaw_mid=old.yaw+.0125*r
        xx=old.x+.025*(u*math.cos(yaw_mid)-v*math.sin(yaw_mid))
        yy=old.y+.025*(u*math.sin(yaw_mid)+v*math.cos(yaw_mid))
        self.state=State(xx,yy,old.yaw+.025*r,out.state[0],out.state[1],out.state[2],
                         out.state[13],list(out.longitudinal_slip),list(out.lateral_slip),
                         list(out.normal_load_n))
        inp.state[:]=out.state[:]
        inp.suspension_history_initialized=out.suspension_history_initialized
        return self.state

    def close(self): pass


class NativePhysX:
    """Drive original project's persistent native 1 kHz PhysX subprocess.

    Uses its existing native `rollout` request (one stage, 25 tick profile).
    There are no fitted force coefficients, interpolated tire limits, or proxy
    dynamics in this adapter. Native ground is flat; map walls are *checked*
    separately; contact with obstacles is not simulated in the native model.
    """
    kind='native-physx-4.1'
    def __init__(self,repo:Path, binary:Path|None=None):
        self.repo=Path(repo)
        loc=self.repo/'f1tenth_localization'
        self.init_path=loc/'config/physx_odom_reset_initial.json'
        self.shapes=loc/'config/physx_odom_vehicle_shapes.json'
        options=[self.repo/'install/f1tenth_localization/lib/f1tenth_localization/physx_native_vehicle_step',
                 self.repo/'install-current/f1tenth_localization/lib/f1tenth_localization/physx_native_vehicle_step',
                 self.repo/'build/f1tenth_localization/physx_native_vehicle_step',
                 self.repo/'build-current/f1tenth_localization/physx_native_vehicle_step']
        if binary: options.insert(0,Path(binary))
        self.binary=next((v.resolve() for v in options if v.is_file()),None)
        if self.binary is None:
            raise FileNotFoundError('Native PhysX binary not found. Build f1tenth_localization or pass --physx-binary.')
        if not self.init_path.is_file() or not self.shapes.is_file():
            raise FileNotFoundError('PhysX reset config or recovered Unity collider shapes missing in repo')
        self.config=json.loads(self.init_path.read_text())['initial']
        # The verified compiled native binary is executable even if its C++ source
        # has been deleted from the worktree. Fingerprint the *actual backend*.
        source_cpp=loc/'src/physx_native_vehicle_step.cpp'
        self.code_hash=source_hash(([source_cpp] if source_cpp.is_file() else [])
                                   +[self.init_path,self.shapes,self.binary])
        self.proc=subprocess.Popen([str(self.binary),'--server',str(self.init_path),str(self.shapes)],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,text=True,bufsize=1)
        self.step_index=0
        self.steering_internal_deg=0.0
        self.state=None
        self.initial=None
        # Physical 18-state vector returned by native PhysX for warm starts.
        self.last_source_state=None
        # Observations from the native wheel tire-force callback.
        self.last_native_contact=None
        # Diagnostic-only observables. Native binary already emits these;
        # do not change the simulation, integrator or control sequence.
        self.last_native_math=None

    def reset(self,x,y,yaw,speed):
        # Map x/y -> Unity x/z: model forward = world +z, model left = world -x.
        config=json.loads(json.dumps(self.config))
        body=config['body']
        body['position_world_m']=[-float(y),.01,float(x)]
        body['rotation_world_xyzw']=[0.,math.sin(-yaw/2),0.,math.cos(-yaw/2)]
        body['linear_velocity_world_mps']=[-float(speed)*math.sin(yaw),0.,float(speed)*math.cos(yaw)]
        body['angular_velocity_world_radps']=[0.,0.,0.]
        for wheel in config['wheel_inputs']:
            wheel['wheel_omega_radps']=float(speed)/wheel['radius_m']
            wheel['wheel_rotation_angle_rad']=0.
        self.initial=config
        self.steering_internal_deg=0.
        self.step_index=0
        self.state=State(float(x),float(y),float(yaw),float(speed),0.,0.,0.,[0.]*4,[0.]*4,[0.]*4)
        self.last_source_state=None
        self.last_full_source_state=None
        self.last_native_contact=None
        self.last_native_math=None
        return self.state

    def step(self,steer,throttle):
        # `initial` is a one-shot reset payload. It becomes None after the
        # first successful stage while the persistent PhysX process remains
        # initialized. `state` is the true local initialization sentinel.
        if self.state is None: raise ValueError('Call reset before step')
        steer=max(-STEER_MAX,min(STEER_MAX,float(steer)))
        throttle=max(0.,min(1.,float(throttle)))
        normalized=steer/STEER_MAX
        torque=(MOTOR_TORQUE*throttle/4) if throttle != 0. else 0.
        brake=MOTOR_TORQUE if throttle == 0. else 0.
        profile=[]
        for tick in range(25):
            old=self.steering_internal_deg
            inc=STEER_RATE*.001
            target=-normalized*30.
            if normalized>old/30.:
                self.steering_internal_deg=max(target,old-inc)
            else:
                self.steering_internal_deg=min(target,old+inc)
            a=math.radians(self.steering_internal_deg)
            tang=math.tan(a)
            fl=math.atan(2*WHEELBASE*tang/(2*WHEELBASE+TRACK*tang))
            fr=math.atan(2*WHEELBASE*tang/(2*WHEELBASE-TRACK*tang))
            profile.append([
                {'steer_angle_rad':fl,'drive_torque_nm':torque,'brake_torque_nm':brake},
                {'steer_angle_rad':fr,'drive_torque_nm':torque,'brake_torque_nm':brake},
                {'steer_angle_rad':0.,'drive_torque_nm':torque,'brake_torque_nm':brake},
                {'steer_angle_rad':0.,'drive_torque_nm':torque,'brake_torque_nm':brake}])
        stage={'capture_row':self.step_index*25,
               'time_s':self.step_index*.025,'wheel_commands_by_tick':profile}
        message={'op':'rollout','stages':[stage]}
        if self.initial is not None:
            message['initial']=self.initial
            self.initial=None
        if self.proc.poll() is not None:
            raise RuntimeError('PhysX subprocess exited: '+self.proc.stderr.read()[-3000:])
        self.proc.stdin.write(json.dumps(message,separators=(',',':'))+'\n')
        self.proc.stdin.flush()
        response=self.proc.stdout.readline()
        if not response:
            raise RuntimeError('PhysX process failed: '+self.proc.stderr.read()[-3000:])
        data=json.loads(response)
        world=data['position_world_m']
        q=data['rotation_world_xyzw']
        # Native actor quaternion rotates world +z forward into world (x,z).
        # Its direction mapped to x-forward/y-left is (forward_z,-forward_x).
        fx=2*(q[0]*q[2]+q[3]*q[1])
        fz=1-2*(q[0]*q[0]+q[1]*q[1])
        yaw=math.atan2(-fx,fz)
        if self.state is not None:
            yaw+=2*math.pi*round((self.state.yaw-yaw)/(2*math.pi))
        raw_native_state=data['unity_source_state']
        # C++ revisions expose the same 18 historical observable channels,
        # sometimes followed by four low-forward/side-speed timers each (26).
        # Do NOT confuse additional internal state with an incompatible prefix.
        if len(raw_native_state) not in (18,26):
            raise RuntimeError(f'Native PhysX returned unsupported vehicle state length: {len(raw_native_state)}')
        full=tuple(float(v) for v in raw_native_state)
        if not np.isfinite(full).all():
            raise RuntimeError('Native PhysX returned nonfinite vehicle state')
        self.last_full_source_state=full
        self.last_source_state=full[:18]
        observed_fields = ('tire_longitudinal_force_n',
                           'tire_lateral_force_n',
                           'longitudinal_slip',
                           'lateral_slip',
                           'normal_load_n')
        if all(field in data for field in observed_fields):
            self.last_native_contact = {
                key: np.asarray(data[key], dtype=float).reshape(4).copy()
                for key in observed_fields
            }
            if not all(np.isfinite(v).all()
                       for v in self.last_native_contact.values()):
                raise RuntimeError('Native PhysX returned nonfinite wheel contact values')
        else:
            self.last_native_contact = None
        # Preserve the precise 1 ms IMU and opt-in vehicle/scene solver
        # stage split, which a 25 ms finite-difference cannot reconstruct.
        vector_keys=('suspension_force_n','suspension_jounce_rate_mps',
                     'linear_velocity_world_mps','angular_velocity_world_radps',
                     'rotation_world_xyzw')
        scalar_keys=('longitudinal_acceleration_mps2',
                     'lateral_acceleration_mps2','yaw_acceleration_radps2')
        self.last_native_math={
            **{k:np.asarray(data[k],dtype=float).copy()
               for k in vector_keys if k in data},
            **{k:float(data[k]) for k in scalar_keys if k in data},
            'solver_diagnostics':data.get('solver_diagnostics'),
        }
        if any(not np.isfinite(val).all() for val in self.last_native_math.values()
               if isinstance(val,np.ndarray)):
            raise RuntimeError('Nonfinite native IMU/solver-stage diagnostic')
        s=State(float(world[2]),-float(world[0]),yaw,
                float(data['body_forward_velocity_mps']),
                float(data['body_lateral_velocity_mps']),
                float(data['unity_source_state'][2]),
                -math.radians(self.steering_internal_deg),
                list(data['longitudinal_slip']),list(data['lateral_slip']),list(data['normal_load_n']))
        self.state=s
        self.step_index+=1
        return s

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('{"op":"shutdown"}\n'); self.proc.stdin.flush()
                self.proc.stdin.close();self.proc.wait(timeout=4)
            except (OSError,subprocess.TimeoutExpired):
                self.proc.kill();self.proc.wait()
        self.proc.stdout.close();self.proc.stderr.close()
