"""Exercise the subprocess protocol against a fake server, NOT PhysX."""
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physics import NativePhysX

FAKE_SERVER='''#!/usr/bin/env python3
import json,sys
request_count = 0
for line in sys.stdin:
    q=json.loads(line)
    if q['op']=='shutdown': break
    assert q['op']=='rollout'
    # A reset payload is sent only on the first stage following reset(),
    # never on subsequent 25ms steps of the same recursive vehicle rollout.
    assert ('initial' in q) == (request_count in (0, 2)), q
    request_count += 1
    assert len(q['stages'])==1
    profile=q['stages'][0]['wheel_commands_by_tick']
    assert len(profile)==25
    assert all(len(x)==4 for x in profile)
    for arr in profile:
        assert all(abs(w['brake_torque_nm']-428.0)<1e-9 for w in arr) or all(w['brake_torque_nm']==0 for w in arr)
    payload=dict(position_world_m=[0,.01,0],rotation_world_xyzw=[0,0,0,1],
        body_forward_velocity_mps=1.,body_lateral_velocity_mps=0.,
        unity_source_state=[1,0,0]+[0]*15,
        longitudinal_slip=[0]*4,lateral_slip=[0]*4,normal_load_n=[5]*4)
    print(json.dumps(payload),flush=True)
'''


def test_native_protocol_runs_stages(tmp_path,monkeypatch):
    loc=tmp_path/'f1tenth_localization'
    (loc/'src').mkdir(parents=True)
    (loc/'config').mkdir(parents=True)
    (loc/'src/physx_native_vehicle_step.cpp').write_text('// fake repo source hash\n')
    body=dict(position_world_m=[0,.01,0],rotation_world_xyzw=[0,0,0,1],
        linear_velocity_world_mps=[0,0,0],angular_velocity_world_radps=[0,0,0])
    wheels=[dict(radius_m=.059) for i in range(4)]
    (loc/'config/physx_odom_reset_initial.json').write_text(json.dumps(dict(initial=dict(body=body,wheel_inputs=wheels))))
    (loc/'config/physx_odom_vehicle_shapes.json').write_text('{}')
    fake=tmp_path/'fake_physics.py';fake.write_text(FAKE_SERVER);os.chmod(fake,0o755)
    monkeypatch.chdir(tmp_path)
    backend=NativePhysX(tmp_path,binary=Path("fake_physics.py"))
    try:
        backend.reset(2,3,1.,3.)
        s=backend.step(.12,.2)
        assert s.u==1. and s.r==0.
        assert math.isfinite(s.yaw)
        assert len(s.longitudinal_slip)==4
        # Regression: stage 2 must keep using the established native process.
        # Previously this raised 'Call reset before step' every time.
        s2=backend.step(.13,.18)
        assert s2.u==1. and len(s2.lateral_slip)==4
        assert backend.step_index==2
        backend.reset(2,3,0.,0.)
        backend.step(0.,0.)
    finally:
        backend.close()
