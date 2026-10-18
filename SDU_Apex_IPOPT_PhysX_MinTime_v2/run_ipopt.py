#!/usr/bin/env python3
"""Restore original full-lap IPOPT OCP and validate against native PhysX.

This is NOT the stochastic `optimize.py` from the previous archive.

- Source model: historical CasADi/Ipopt 16-state Unity-wheel min-time OCP,
  initialized with *current* checked-in PhysX/Unity vehicle constants.
- Acceptance model: exact repo native 4.1 PhysX engine on 1ms steps.
- Validity: candidate is never exported unless native completes a safe lap.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import numpy as np
from scipy.signal import find_peaks

from recover import load_modules, REF
from source_vehicle import current_model,bind_current_wheel_moment_arms
from map_track import MapGrid, map_to_track
from native_validate import replay,export_mpc
from transfer_correction import native_lateral_bias,candidate_warm_state
from native_seed import generate as generate_native_seed
from phase1_campaign import continue_phase1, validate_restart_checkpoint
from seed_projection import (load_native_seed, project_algebraics,
                             write_audit, validate_seed_provenance)


def brake_seed(track, number:int):
    """Initialize *locations only*; IPOPT moves all brake phase boundaries."""
    n=track.count
    mask=np.zeros(n,dtype=bool)
    if number==0: return mask
    # Track curvature maxima provide an initial discrete drive/brake topology.
    # No generic lateral acceleration cap or real-car tire model is used.
    curvature=np.abs(np.asarray(track.kappa,float))
    gap=max(9,n//(number+2))
    peaks,_=find_peaks(curvature,distance=gap,prominence=max(.08,float(np.max(curvature))*.08))
    ranked=sorted(peaks,key=lambda ix:curvature[ix],reverse=True)
    selected=[]
    for p in ranked:
        if p<8 or p>n-9: continue  # IPOPT's periodic gate is in drive mode.
        if any(min(abs(p-old),n-abs(p-old))<gap for old in selected):continue
        selected.append(p)
        if len(selected)>=number:break
    for peak in selected:
        # The braking zone precedes a turn; six nodes remain for resolution.
        a=max(6,peak-7);b=min(n-6,a+6)
        mask[a:b]=True
    # Avoid brake spans closer than six nodes to each other.
    changes=np.flatnonzero(mask[1:]!=mask[:-1])+1
    bounds=np.r_[0,changes,n]
    if any(end-start<6 for start,end in zip(bounds[:-1],bounds[1:])):
        mask[:]=False
    return mask


def validate_geometry(track,grid,margin):
    xy=np.column_stack((track.x,track.y))
    # Geometry seed must live in known free space; unlike native rollout we
    # only check centerline here (IPOPT constrains body footprint separately).
    probe=xy[:,None,:]+np.linspace(0,1,6,endpoint=False)[None,:,None]*(
        np.roll(xy,-1,axis=0)-xy)[:,None,:]
    min_center=float(np.min(grid.distance(probe.reshape(-1,2))))
    if min_center < margin+.1365:
        raise ValueError('Original reference/resampled centerline unsafe for IPOPT: '
                         f'min wall distance={min_center:.3f}m; need >= {margin+.1365:.3f}m')
    return min_center


def print_feasibility_diagnostics(report:dict, trust_limit:float)->None:
    """Print *solver iterate* constraints, not just the default Ipopt summary."""
    diag=report.get('nlp_diagnostics',{})
    print('IPOPT Phase-I diagnostics:',flush=True)
    print('  IPOPT status:',report.get('solver',{}).get('return_status','unknown'),flush=True)
    print('  Max NLP violation:',diag.get('max_constraint_violation'),flush=True)
    worst=diag.get('worst_constraints',[])
    if not worst:
        print('  No labeled constraint diagnostics available; inspect JSON report.',flush=True)
        return
    for item in worst[:8]:
        print(f"  {item.get('label', 'unknown')}: violation={item.get('violation')} ",flush=True)
    if worst[0].get('label')=='captured_motion_trust_region':
        print('  Unexpected captured-motion trust-bound violation during Phase I. '
              'The v2.2.3 numerical overlay disables this artificial bound; '
              'verify your working solver overlay and provenance.',flush=True)


def run():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description='CasADi/IPOPT Unity-wheel min-time + exact native PhysX 1ms validation')
    ap.add_argument('--repo-root',type=Path,default=Path('..'))
    ap.add_argument('--map',type=Path,default=None,help='PGM with matching ROS .yaml; default Porto practice map')
    ap.add_argument('--centerline',type=Path,default=None,help='CSV reference only; optimizer changes geometry')
    ap.add_argument('--output',type=Path,default=Path('optimized_practice_ipopt'))
    ap.add_argument('--physx-binary',type=Path,default=None)
    ap.add_argument('--mesh-spacing',type=float,default=.20,help='IPOPT OCP discretization (m); must be <= .24 due original IPOPT phase mesh constraint')
    ap.add_argument('--max-speed',type=float,default=16.,help='current MPC command envelope, not real-car limit')
    ap.add_argument('--iterations',type=int,default=1200,help='IPOPT minimum-time stage iterations')
    ap.add_argument('--feasibility-iterations',type=int,default=500)
    ap.add_argument('--phase1-restarts',type=int,default=0,help='controlled Phase-I restarts from best hard-feasible but nonzero-defect checkpoint')
    ap.add_argument('--phase1-min-defect-improvement',type=float,default=1.e-5,help='minimum max-defect progress required to restart Phase I')
    ap.add_argument('--phase1-hessian', choices=('limited-memory','exact'), default='limited-memory', help='Use IPOPT L-BFGS for stiff/non-smooth Phase-I feasibility; original exact Hessian remains available')
    ap.add_argument('--phase1-formulation', choices=('elastic','condensed','eliminated'), default='elastic', help='Choose historical elastic NLP or v2.4.2 reduced residual least-squares diagnostic; condensed NEVER certifies a lap')
    ap.add_argument('--feasibility-trust-region',type=float,default=200.,help='Elastic Phase-I disables the seed-distance cap; condensed Phase-I uses four times this value as its hard numeric trust radius')
    ap.add_argument('--diagnose-initial-only',action='store_true',help='print initial constraint violations without running IPOPT or PhysX')
    ap.add_argument('--brake-zones',type=int,default=3,help='initial zero-throttle brake phase count; phase lengths optimized by IPOPT')
    ap.add_argument('--phase1-warmstart-report', type=Path, default=None,
                    help='Use a verified v2.4.2 Phase-I checkpoint to initialize eliminated Phase I')
    ap.add_argument('--seed-file',type=Path,default=None, help='Reuse a native_warmstart.npz for the same OCP mesh and map; skips driving warm-start')
    ap.add_argument('--no-seed-force-projection',action='store_true',help='Diagnostic: keep measured finite-difference acceleration and PhysX loads')
    ap.add_argument('--seed-backend',choices=('native','source','synthetic'),default='native',
                    help='native PhysX flying-lap seed (default), C++ source-derived seed, or legacy synthetic initialization')
    ap.add_argument('--native-correction-passes',type=int,default=1,help='number of IPOPT corridor re-solves using observed native-PhysX lateral deviations')
    ap.add_argument('--wall-margin',type=float,default=.15)
    ap.add_argument('--native-max-seconds',type=float,default=18.)
    ap.add_argument('--warmup-laps',type=int,default=1)
    ap.add_argument('--print-level',type=int,default=4)
    ap.add_argument('--allow-history-fetch',action='store_true')
    ap.add_argument('--restore-only',action='store_true',help='restore exact historical OCP source to package cache and exit')
    ap.add_argument('--setup-check',action='store_true',help='restore and check map/model imports but skip IPOPT')
    ap.add_argument('--skip-native',action='store_true',help='diagnostics only; NEVER exports a controller raceline')
    args=ap.parse_args()
    args.repo_root=args.repo_root.resolve()
    if args.map is None:
        args.map=args.repo_root/'f1tenth_planning/maps/autodrive_practice_20260924_b.pgm'
    args.map=args.map.resolve()
    if args.centerline is None:
        if args.map.name=='autodrive_practice_20260924_b.pgm':
            args.centerline=args.repo_root/'f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv'
        elif args.map.name=='autodrive_track_ftg_commit_20260909_025m.pgm':
            args.centerline=args.repo_root/'f1tenth_planning/trajectories/autodrive_mintime_sim_5p0_dense/autodrive_mintime_raceline.csv'
        else:
            ap.error('For a custom PGM, supply a centerline CSV (--centerline) with correct map correspondences')
    args.centerline=args.centerline.resolve()
    args.output=args.output.resolve()
    if args.mesh_spacing<.12 or args.mesh_spacing>.24 or args.brake_zones<0 or args.brake_zones>8 or args.native_correction_passes<0 or args.native_correction_passes>3 or args.wall_margin<0 or args.max_speed<.4 or args.iterations<1 or args.feasibility_iterations<1 or not math.isfinite(args.feasibility_trust_region) or args.feasibility_trust_region<=0 or args.native_max_seconds<.5 or args.warmup_laps<0:
        ap.error('invalid optimization bounds (mesh-spacing .12-.24, brake-zones 0-8, positive speed/time/iterations)')
    print('== Recover verified original IPOPT spatial min-time source ==',flush=True)
    old, wheel, track_mod, vendor, solver_overlay = load_modules(
        args.repo_root, root, args.allow_history_fetch, args.phase1_hessian, args.phase1_formulation)
    print(f'Original commit: {REF}',flush=True)
    print(f'Working CasADi solver: {Path(old.__file__)}',flush=True)
    print(f'Phase-I Hessian: {args.phase1_hessian}; historical source preserved: '
          f'{solver_overlay["source_untouched"]}',flush=True)
    if args.restore_only:
        print('Restored successfully; original files are unchanged.');return 0
    print('== Resolve current PhysX/Unity source parameters ==',flush=True)
    model, model_meta=current_model(args.repo_root,wheel)
    model_meta['current_wheel_moment_arms']=bind_current_wheel_moment_arms(args.repo_root,old,wheel)
    grid=MapGrid.load(args.map)
    # Seed verification also detects mismatched map/raceline pairs.
    init_ref=map_to_track(grid,args.centerline,spacing=.12)
    track=track_mod.load_centerline_csv(args.centerline,args.mesh_spacing)
    min_center=validate_geometry(track,grid,args.wall_margin)
    print(f'Map {args.map.name} | OCP length {track.length:.3f} m | nodes {track.count} '
          f'| reference min clearance {min_center:.3f} m',flush=True)
    print(f'Motor {model.motor_torque_nm} Nm; wheel radius {model.wheel_radius_m} m; '
          f'brake at exactly zero throttle; steering ±{model.steering_limit_rad:.4f} rad',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    run_info={
        'optimizer':'CasADi/IPOPT restored from original wheel-level minimum-time collocation',
        'repo_root':str(args.repo_root),'map':str(args.map),
        'map_sha256':hashlib.sha256(args.map.read_bytes()).hexdigest(),
        'centerline':str(args.centerline),
        'centerline_sha256':hashlib.sha256(args.centerline.read_bytes()).hexdigest(),
        'restore_commit':REF,'phase1_solver_overlay':solver_overlay,'model':model_meta,
        'requested_warmstart_backend':args.seed_backend,
        'native_preflight_replay_does_not_certify_Unity':True,
        'stage':'setup_complete',
        'limitations':'IPOPT historical CasADi wheel+heave+roll+pitch contact approximation, not exact 1ms PhysX scene integration. Native 1ms validation is mandatory and is not Unity.',
    }
    (args.output/'provenance.json').write_text(json.dumps(run_info,indent=2)+'\n')
    if args.setup_check:
        print('SETUP CHECK PASS (no IPOPT or PhysX evaluation launched)');return 0
    if (args.output/'ipopt_candidate_pass_00').exists():
        ap.error('Refusing to overwrite previous candidate; choose a fresh --output directory')
    if args.seed_file is not None and args.seed_backend == 'synthetic':
        ap.error('--seed-file requires --seed-backend native or source')
    if args.brake_zones == 0:
        print('WARNING: --brake-zones 0 prohibits ALL zero-throttle CAWB braking. '
              'With 428 Nm total motor torque, a drive-only periodic lap may be '
              'infeasible. For the Porto lap-time target use --brake-zones 3.',
              flush=True)
    mask=brake_seed(track,args.brake_zones)
    # Native measured state initialization replaces the repeatedly stalled
    # curvature-derived seed. Never pass an incomplete or colliding lap to IPOPT.
    first_seed=None
    if args.seed_file is not None:
        provenance_audit=validate_seed_provenance(args.seed_file,args.map,args.centerline)
        first_seed=load_native_seed(args.seed_file.resolve(), track.count)
        seed_reports=[{'backend':args.seed_backend,'completed_flying_lap':True,
                       'source':'reused_native_warmstart_file',
                       'path':str(args.seed_file.resolve()),
                       'provenance_check':provenance_audit,
                       'note':'matching map/centerline checksums and seed shape verified'}]
        print('== Reusing physics warm-start file: '+str(args.seed_file.resolve())+' ==',flush=True)
    elif args.seed_backend != 'synthetic' and not args.diagnose_initial_only:
        first_seed,seed_reports=generate_native_seed(
            args.repo_root,args.physx_binary,args.seed_backend,init_ref,grid,
            track,args.output/'warmstart_physx',args.wall_margin)
        if first_seed is None:
            (args.output/'report.json').write_text(json.dumps({
                'status':'no_complete_safe_native_seed',
                'backend':args.seed_backend,'trials':seed_reports,
                'native_validated':False,'lap_target_s':4.9},indent=2)+'\n')
            print('No complete, wall-clear physics warm-start lap. '
                  'IPOPT was NOT launched; inspect warmstart_physx/native_seed_report.json '
                  'and seed_trace_*.csv. Fix the path follower or use '
                  '--seed-backend synthetic for the old behavior.',file=sys.stderr)
            return 5
        print('== Complete PhysX motion seed loaded; optimizing from observed '
              'wheel/body states instead of synthetic curvature dynamics ==',flush=True)
        run_info['warmstart_preflight']=seed_reports
        run_info['warmstart_state_path']=str(args.output/'warmstart_physx/native_warmstart.npz')
        (args.output/'provenance.json').write_text(json.dumps(run_info,indent=2)+'\n')
    phase1_checkpoint_seed = None
    if args.phase1_warmstart_report is not None:
        if args.phase1_formulation != 'eliminated' or args.seed_file is None:
            ap.error('--phase1-warmstart-report requires eliminated formulation and a matching --seed-file')
        old_report = json.loads(args.phase1_warmstart_report.read_text())
        if (old_report.get('phase1_strict_feasible') is not False
                or old_report.get('solver', {}).get('feasible_incumbent') is not False
                or old_report.get('phase1_restart_checkpoint') is None):
            ap.error('Checkpoint source is not a verified failed Phase-I report')
        if old_report.get('unity_vehicle_model') != json.loads(json.dumps(model.metadata())):
            ap.error('Checkpoint source vehicle model differs from current source parameters')
        source_track = old_report.get('track', {})
        if (int(source_track.get('count', -1)) != track.count
                or abs(float(source_track.get('length_m', -1)) - track.length) > 1.e-6):
            ap.error('Checkpoint source track differs from current OCP')
        expected_phases = int(np.count_nonzero(mask[1:] != mask[:-1])) + 1
        phase1_checkpoint_seed = validate_restart_checkpoint(
            old_report['phase1_restart_checkpoint'], track.count,
            track.length, expected_phases)
        print('== Reusing verified dynamics-infeasible checkpoint: max defect '
              f'{phase1_checkpoint_seed[3]:.9g} (not a racing lap) ==', flush=True)

    print(f'== IPOPT OCP: {track.count} nodes, {int(mask.sum())} initial brake nodes, '
          f'{args.feasibility_iterations} feasibility + {args.iterations} time iterations, '
          f'Phase-I seed-distance cap=disabled, seed={args.seed_backend}, '
          f'Phase-I {args.phase1_formulation} dynamics + safe spatial evaluations + simple bounds, '
          f'Phase-I Hessian={args.phase1_hessian} ==',flush=True)
    previous_candidate=None;previous_report=None
    tracking_bias=np.zeros(track.count)
    for phase in range(args.native_correction_passes+1):
        print(f'== IPOPT feasibility -> minimum-time solve {phase+1}/{args.native_correction_passes+1} ==',flush=True)
        kwargs=dict(
            wall_clearance_m=args.wall_margin,
            feasibility_trust_region=args.feasibility_trust_region,
            diagnose_initial_only=args.diagnose_initial_only,
            max_speed_mps=args.max_speed,
            brake_mode_seed=mask,
            feasibility_max_iterations=args.feasibility_iterations,
            time_optimization_max_iterations=args.iterations,
            print_level=args.print_level)
        if previous_candidate is not None:
            kwargs.update(
                initial_state_seed=candidate_warm_state(previous_candidate),
                initial_state_seed_s=np.asarray(previous_candidate['s_ref_m']),
                initial_state_seed_is_model_consistent=True,
                phase_lengths_initial=np.asarray(previous_report['optimized_phase_lengths_m']),
                tracking_bias_m=tracking_bias)
        elif phase1_checkpoint_seed is not None:
            cp_seed, cp_station, cp_lengths, cp_defect, _ = phase1_checkpoint_seed
            kwargs.update(initial_state_seed=cp_seed,
                          initial_state_seed_s=cp_station,
                          phase_lengths_initial=cp_lengths,
                          initial_state_seed_is_model_consistent=True,
                          brake_mode_seed_source=(
                              'v2.4.2 validated hard-safe checkpoint, '
                              'strict dynamics infeasible (algebraic elimination)'))
        elif first_seed is not None:
            # The native measured differential state is not algebraically
            # consistent with the source-translated symbolic force model.
            # Compute the *symbolic* suspension loads and tire-force-derived
            # accelerations before seeding IPOPT, but retain native u/v/yaw.
            if not args.no_seed_force_projection:
                seed,seed_audit=project_algebraics(first_seed,model,wheel,
                                                   brake_mask=mask)
            else:
                seed={k:np.asarray(v).copy() for k,v in first_seed.items()}
                seed['throttle_norm'][mask]=0.0
                seed_audit={'type':'unprojected_native_state',
                    'forced_full_brake_nodes_unseen_in_native':int(np.sum(mask))}
            write_audit(args.output/'warmstart_physx/seed_projection_audit.json',seed_audit)
            print('Source-model seed projection audit:',json.dumps(seed_audit,sort_keys=True),flush=True)
            if seed_audit.get('forced_full_brake_nodes_unseen_in_native',0):
                print('WARNING: The native lap used no full-brake throttle mode, but IPOPT '
                      'contains forced CAWB braking nodes. The brake-mode trajectory '
                      'is still NOT dynamically consistent. Run with --brake-zones 0 '
                      'first to isolate symbolic dynamics consistency.',flush=True)
            kwargs.update(initial_state_seed=seed,
                          initial_state_seed_s=np.asarray(track.s),
                          initial_state_seed_is_model_consistent=True,
                          brake_mode_seed_source=('native-PhysX motion (unprojected)'
                              if args.no_seed_force_projection else
                              'native-PhysX motion with symbolic algebraic projection'))
        candidate,report,phase1_progress=continue_phase1(
            old,track,model,kwargs,output=args.output,phase=phase,
            max_restarts=args.phase1_restarts,
            minimum_improvement=args.phase1_min_defect_improvement)
        report['phase1_progress'] = phase1_progress
        report['phase1_solver_overlay'] = solver_overlay
        (args.output/f'ipopt_report_pass_{phase:02d}.json').write_text(
            json.dumps(report,indent=2,default=str)+'\n')
        if not candidate or not report.get('solver',{}).get('feasible_incumbent',False):
            print_feasibility_diagnostics(report,args.feasibility_trust_region)
            (args.output/'report.json').write_text(json.dumps({
                'status':'ipopt_not_feasible','phase':phase,
                'native_validated':False, 'lap_target_s':4.9},indent=2)+'\n')
            if args.diagnose_initial_only:
                print('Initial-state constraint diagnostics written; no solver or native PhysX run.',file=sys.stderr)
                return 0
            print('IPOPT did not find a fully feasible periodic lap. '
                  'Inspect phase-I defect-continuation diagnostics in ipopt_report_pass_XX.json. No MPC raceline exported.',file=sys.stderr)
            return 2
        candidate_dir=args.output/f'ipopt_candidate_pass_{phase:02d}'
        old.write_candidate(candidate,report,track,args.centerline,candidate_dir)
        print('IPOPT predicted full-lap time:',report.get('predicted_lap_time_s',
                                                         report.get('lap_time_s','see report')),flush=True)
        if args.skip_native:
            (args.output/'report.json').write_text(json.dumps({
                'status':'ipopt_feasible_native_skipped','phase':phase,
                'native_validated':False, 'lap_target_s':4.9},indent=2)+'\n')
            print('Native validation intentionally skipped: diagnostic candidate only.',file=sys.stderr)
            return 3
        print('== Replay IPOPT spatial controls through native PhysX (1ms steps) ==',flush=True)
        native_dir=args.output/f'native_pass_{phase:02d}'
        result=replay(args.repo_root,args.physx_binary,candidate,track,grid,init_ref,
                      native_dir,args.wall_margin,args.warmup_laps,args.native_max_seconds)
        (args.output/'report.json').write_text(json.dumps({
            'phase':phase,'status':'native_validated' if result['completed'] else 'native_rejected',
            'optimizer':'restored CasADi/IPOPT Unity-wheel spatial OCP',
            'native':result,'native_validated':bool(result['completed'] and result['collision_free']),
            'lap_target_s':4.9},indent=2,default=str)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='source_notes'},indent=2),flush=True)
        if result['completed'] and result['collision_free']:
            # Only the native accepted *complete* lap reaches this branch.
            target=args.output/'autodrive_mintime_raceline.csv'
            export_mpc(candidate,grid,target,args.wall_margin)
            print(f'Native-validated MPC path: {target}',flush=True)
            print(f'Native flying lap: {result["native_flying_lap_time_s"]:.3f}s, target <4.900s',flush=True)
            print('Unity simulator parity is not certified. Validate in Unity before deploying.',flush=True)
            return 0
        if phase>=args.native_correction_passes or result['failure']!='footprint_wall_clearance':
            print('Native PhysX rejected the candidate. No controller raceline exported. '
                  'Inspect native_pass_XX/native_trace.csv for the first divergence.',file=sys.stderr)
            return 4
        bias,diagnostics=native_lateral_bias(native_dir/'native_trace.csv',candidate,track)
        print('Native tracking-residual correction:',json.dumps(diagnostics,indent=2),flush=True)
        if diagnostics['status']!='computed_from_native_open_loop_tracking_difference':
            print('Native correction has insufficient valid samples. No raceline exported.',file=sys.stderr)
            return 4
        (args.output/f'native_correction_pass_{phase:02d}.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        tracking_bias=bias
        previous_candidate=candidate;previous_report=report
        mask=np.asarray(candidate['brake_mode'],float)>.5
    return 4

if __name__=='__main__':
    try:
        sys.exit(run())
    except (Exception,KeyboardInterrupt) as exc:
        print(f'ERROR {type(exc).__name__}: {exc}',file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
