"""Re-evaluate v2.3.11 measured tire callback using corrected v2.3.12 CasADi.

No IPOPT or native replay. This checks the actual recovered/generated tire shader,
not a Python reimplementation, against all 1,924 native measurements.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import casadi as ca
from recover import load_modules
from source_vehicle import current_model


def measured_parity(rows: list[dict], wheel_module, model):
    if not rows:
        raise RuntimeError('Missing v2.3.11 native contact measurements')
    slip_long=ca.SX.sym('longitudinal_slip')
    slip_lat=ca.SX.sym('lateral_slip')
    load=ca.SX.sym('native_suspension_force')
    funcs=[]
    for i in range(4):
        _,fx,fy,_=wheel_module._unity_tire_force_shader_casadi(
            slip_long, slip_lat, load, model.wheel_radius_m,
            model.forward_curve[i], model.lateral_curve[i])
        funcs.append(ca.Function('v2312_native_shader_'+str(i),
                                 [slip_long,slip_lat,load],[fx,fy]))
    wheels=('fl','fr','rl','rr')
    errs=[]
    rows_per_wheel={w:0 for w in wheels}
    load_deltas=[]
    for row in rows:
        i=wheels.index(row['wheel'])
        sx=float(row['native_long_slip'])
        sy=float(row['native_lat_slip'])
        fz=float(row['native_suspension_force_n'])
        fx,fy=funcs[i](sx,sy,fz)
        diff=[float(fx)-float(row['native_callback_fx_n']),
              float(fy)-float(row['native_callback_fy_n'])]
        if not np.isfinite(diff).all() or fz<=0:
            raise RuntimeError('Non-finite native tire-force test or noncontact wheel')
        errs.append(diff)
        load_deltas.append(float(row['symbolic_filtered_tire_load_n'])-fz)
        rows_per_wheel[row['wheel']]+=1
    errs=np.asarray(errs,dtype=np.float64)
    deltas=np.asarray(load_deltas,dtype=np.float64)
    result={
        'comparison':'actual native PxVehicle tire force vs generated CasADi native tire callback',
        'samples':len(rows),
        'wheel_samples_by_wheel':rows_per_wheel,
        'corrected_callback_rms_force_error_n':float(np.sqrt(np.mean(errs**2))),
        'corrected_callback_max_force_error_n':float(np.max(np.abs(errs))),
        'previous_extra_filter_minus_native_load_mean_n':float(np.mean(deltas)),
        'previous_extra_filter_minus_native_load_rms_n':float(np.sqrt(np.mean(deltas**2))),
        'contact_slip_parity_tested':False,
        'force_balance_feasible':False,
        'lap_racing_result':False,
        'note':('Tests tire-force calculation GIVEN native-measured wheel slips and native suspension loads; '
                'optimizer planar contact reconstruction, full 143-node periodic feasibility, and Unity parity remain unverified.'),
    }
    if result['corrected_callback_rms_force_error_n']>1e-4 or result['corrected_callback_max_force_error_n']>1e-3:
        raise RuntimeError('Corrected CasADi tire callback still differs from native output: '+str(result))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,default=Path('..'))
    ap.add_argument('--input',type=Path,default=Path('native_contact_v2311/per_wheel.csv'))
    ap.add_argument('--output',type=Path,default=Path('native_contact_v2312/summary.json'))
    args=ap.parse_args()
    root=args.repo_root.resolve()
    if not args.input.is_file():
        raise SystemExit('v2.3.11 native measurement missing: '+str(args.input))
    src=(root/'f1tenth_localization/src/physx_native_vehicle_step.cpp').read_text()
    # Fail closed on unknown native shader changes.
    anchors=('frictionPairs->setTypePairFriction(0, 0, 1.0f)',
             'longForce = tireLoad * tireFriction * fxCurve * combinedScale',
             'latForce = -tireLoad * tireFriction * fyCurve * combinedScale')
    if any(a not in src for a in anchors):
        raise RuntimeError('Native PhysX shader source changed; parity test invalid')
    _,wheel,_,_,meta=load_modules(root,Path(__file__).resolve().parent)
    if not meta.get('working_wheel_model_parity_applied'):
        raise RuntimeError('Generated working wheel model not patched')
    model,_=current_model(root,wheel)
    import inspect
    tire_code=inspect.getsource(wheel.tire_forces_casadi)
    if 'tire_load = suspension_forces[index]' not in tire_code:
        raise RuntimeError('OCP tire_forces_casadi is not using native callback load')
    with args.input.open(newline='') as f:
        rows=list(csv.DictReader(f))
    result=measured_parity(rows,wheel,model)
    result['original_wheel_model_sha256']=meta['original_wheel_model_sha256']
    result['working_wheel_model_sha256']=meta['working_wheel_model_sha256']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('No IPOPT optimization or native/Unity replay executed.')


if __name__=='__main__':
    main()
