"""v2.4.0: multiple improvements in one batch, never counterfeit feasibility."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from phase1_campaign import (validate_restart_checkpoint, continue_phase1,
                             load_phase1_resume_report)
from solver_overlay import (transform_solver_v240, SOLVE_ELASTIC,
                            _PHASE1_FAILURE_MARKER, _PHASE1_REPORT_MARKER)


def fixture_checkpoint(n=5):
    names = ['ey_m','epsi_rad','u_mps','v_mps','yaw_rate_radps',
             'steering_rad','body_height_m','body_height_rate_mps',
             'roll_rad','roll_rate_radps','pitch_rad','pitch_rate_radps',
             'steering_rate_radps','throttle_norm','ax_mps2','ay_mps2']
    s={key:[0.0]*n for key in names}
    s['u_mps']=[2.4]*n
    s['normal_load_n']=[[8.]*n for _ in range(4)]
    s['wheel_omega_radps']=[[40.]*n for _ in range(4)]
    return {'seed':s, 'station_nodes_m':list(np.arange(n, dtype=float)),
            'phase_lengths_m':[float(n)],'raw_max_dynamic_defect':0.8,
            'elastic_merit':10.0, 'hard_constraints_satisfied':True,
            'strict_original_dynamics_feasible':False, 'origin':'checkpoint'}


def test_validate_checkpoint_with_original_physical_state_mesh():
    cp=fixture_checkpoint()
    s, stations, lengths, defect, energy=validate_restart_checkpoint(cp,5,5.,1)
    assert s['wheel_omega_radps'].shape==(4,5)
    assert np.array_equal(stations,np.arange(5))
    assert list(lengths)==[5.]
    assert defect==0.8 and energy==10


def _write_resume_fixture(tmp_path):
    import hashlib
    track=SimpleNamespace(count=5,length=5.,s=np.arange(5,dtype=float))
    map_path=tmp_path/'track.pgm';map_path.write_bytes(b'map')
    centerline=tmp_path/'line.csv';centerline.write_bytes(b'centerline')
    mesh=tmp_path/'mesh.csv';mesh.write_bytes(b'mesh')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    model={'mass_kg':3.47}
    source_sha='source-sha-256'
    overlay={'phase1_formulation':'elastic',
             'optimizer_source_sha256':source_sha}
    checkpoint=fixture_checkpoint()
    report={
        'phase1_strict_feasible':False,
        'solver':{'feasible_incumbent':False},
        'track':{'count':track.count,'length_m':track.length},
        'unity_vehicle_model':model,
        'phase1_solver_overlay':overlay,
        'phase1_restart_checkpoint':checkpoint,
    }
    report_path=tmp_path/'ipopt_report_pass_00.json'
    report_path.write_text(json.dumps(report))
    provenance={
        'map_sha256':digest(map_path),
        'centerline_sha256':digest(centerline),
        'optimizer_reference_track_mesh':{
            'sha256':digest(mesh),'nodes':track.count,'length_m':track.length},
        'phase1_solver_overlay':overlay,
    }
    (tmp_path/'provenance.json').write_text(json.dumps(provenance))
    return report_path,map_path,centerline,mesh,track,model,source_sha


def test_resume_report_requires_matching_source_map_mesh_model_and_formulation(tmp_path):
    args=_write_resume_fixture(tmp_path)
    checkpoint=load_phase1_resume_report(
        args[0],map_path=args[1],centerline_path=args[2],mesh_path=args[3],
        track=args[4],model_metadata=args[5],
        optimizer_source_sha256=args[6],phase1_formulation='elastic')
    assert checkpoint[3]==0.8
    assert np.array_equal(checkpoint[1],args[4].s)


@pytest.mark.parametrize('kwargs,match',[
    ({'optimizer_source_sha256':'different'},'optimizer source'),
    ({'phase1_formulation':'eliminated'},'formulation'),
    ({'model_metadata':{'mass_kg':9.0}},'vehicle model'),
])
def test_resume_report_rejects_stale_source_metadata(tmp_path,kwargs,match):
    args=_write_resume_fixture(tmp_path)
    call={'map_path':args[1],'centerline_path':args[2],
          'mesh_path':args[3],'track':args[4],
          'model_metadata':args[5],'optimizer_source_sha256':args[6],
          'phase1_formulation':'elastic'}
    call.update(kwargs)
    with pytest.raises(ValueError,match=match):
        load_phase1_resume_report(args[0],**call)


def test_resume_report_rejects_map_or_mesh_mismatch(tmp_path):
    args=_write_resume_fixture(tmp_path)
    args[1].write_bytes(b'changed map')
    with pytest.raises(ValueError,match='map_sha256'):
        load_phase1_resume_report(
            args[0],map_path=args[1],centerline_path=args[2],mesh_path=args[3],
            track=args[4],model_metadata=args[5],
            optimizer_source_sha256=args[6],phase1_formulation='elastic')


@pytest.mark.parametrize('mutation',[
    lambda cp:cp.__setitem__('hard_constraints_satisfied',False),
    lambda cp:cp.__setitem__('strict_original_dynamics_feasible',True),
    lambda cp:cp.__setitem__('station_nodes_m',[0,1,2,2,4]),
    lambda cp:cp.__setitem__('phase_lengths_m',[4.]),
    lambda cp:cp['seed'].__setitem__('u_mps',[-1]*5),
    lambda cp:cp['seed'].__setitem__('normal_load_n',[[0]*5 for _ in range(4)]),
])
def test_reject_invalid_physics_restart(mutation):
    cp=fixture_checkpoint();mutation(cp)
    with pytest.raises(ValueError):validate_restart_checkpoint(cp,5,5.,1)


class FakeOld:
    def __init__(self, values):
        self.values=list(values);self.kwargs=[]
    def solve_bottom_up(self,track,model,**kwargs):
        self.kwargs.append(kwargs)
        val=self.values.pop(0)
        cp=fixture_checkpoint()
        cp['raw_max_dynamic_defect']=val
        return {}, {
            'solver': {'return_status':'Maximum_Iterations_Exceeded', 'feasible_incumbent':False},
            'solve_stages':[{'initial_raw_max_defect':1.0 if len(self.kwargs)==1 else self.kwargs[-1]['initial_state_seed']['u_mps'][0]*0+float(self.kwargs[-2]['mock_defect']) if 'mock_defect' in self.kwargs[-2] else 0.8,
                             'best_raw_max_defect':val,'best_elastic_merit':10.0,'ipopt_iterations':5}],
            'phase1_restart_checkpoint':cp}


def test_phase1_restarts_reuse_best_hard_feasible_state(tmp_path):
    solver=FakeOld([0.8,0.7,0.6])
    track=SimpleNamespace(count=5,length=5.)
    kwargs={}
    candidate, report, progress=continue_phase1(solver,track,None,kwargs,
        output=tmp_path,phase=0,max_restarts=2,minimum_improvement=0.00001)
    assert not candidate and not report['solver']['feasible_incumbent']
    assert len(progress)==3
    assert len(solver.kwargs)==3
    assert 'initial_state_seed' not in solver.kwargs[0]
    assert solver.kwargs[1]['initial_state_seed']['u_mps'].shape==(5,)
    assert np.array_equal(solver.kwargs[1]['initial_state_seed_s'],np.arange(5))
    assert len(list(tmp_path.glob('ipopt_report_pass_00_restart_*.json')))==3
    assert json.loads((tmp_path/'phase1_progress_pass_00.json').read_text())['final_strict_feasible'] is False


def test_stop_repeating_when_no_new_feasibility_progress(tmp_path):
    solver=FakeOld([0.8,0.79999999,0.799999])
    _,_,progress=continue_phase1(solver,SimpleNamespace(count=5,length=5.),None,
        {},output=tmp_path,
        phase=0,max_restarts=8,minimum_improvement=1e-5)
    assert len(progress)==2


def test_guarded_solver_overlay_preserves_strict_failure_gate():
    assert SOLVE_ELASTIC.count(_PHASE1_FAILURE_MARKER)==1
    assert SOLVE_ELASTIC.count(_PHASE1_REPORT_MARKER)==1
    assert 'opti.set_value(elastic_cap, 0.0)' in SOLVE_ELASTIC
    from solver_overlay import _PHASE1_RESTART_EXPORT, _PHASE1_REPORT_EXPORT
    assert "'strict_original_dynamics_feasible': False" in _PHASE1_RESTART_EXPORT
    assert "'phase1_restart_checkpoint'" in _PHASE1_REPORT_EXPORT


def test_bash_script_v2316_does_not_overwrite_v2315():
    s=(Path(__file__).parents[1]/'diagnose_phase1_v2316.sh').read_text()
    assert 'optimized_practice_v2_3_16_contact_debug' in s
    assert 'optimized_practice_v2_3_15_native_slip_debug' not in s


def test_cached_solver_can_switch_hessian_modes_without_erasing_source(tmp_path):
    from tests.test_phase1_overlay import _sample,LEGACY,SOLVER
    from solver_overlay import prepare_working_solver, transform_solver_v240
    recovered,source,stub=_sample(tmp_path)
    realish=stub+'\n    def source_wheel_update():\n        pass\n'+__import__('solver_overlay')._CONTACT_SOLVER_CALL_OLD+'\n'
    source.write_text(realish)
    dest=tmp_path/'work'/LEGACY/SOLVER
    dest.parent.mkdir(parents=True)
    dest.write_text(transform_solver_v240(realish,'limited-memory'))
    prepare_working_solver(recovered,tmp_path/'work',LEGACY,mode='exact')
    assert dest.read_text()==transform_solver_v240(realish,'exact')
    assert source.read_text()==realish
    prepare_working_solver(recovered,tmp_path/'work',LEGACY,mode='limited-memory')
    assert dest.read_text()==transform_solver_v240(realish,'limited-memory')


def test_generated_full_solver_site_is_syntactically_valid(tmp_path):
    import ast
    from tests.test_phase1_overlay import _sample
    from solver_overlay import _CONTACT_SOLVER_CALL_OLD
    _,_,stub=_sample(tmp_path)
    realish=stub+'\n    def source_wheel_update():\n        pass\n'+_CONTACT_SOLVER_CALL_OLD+'\n'
    for mode in ('limited-memory','exact'):
        ast.parse(transform_solver_v240(realish,mode))
