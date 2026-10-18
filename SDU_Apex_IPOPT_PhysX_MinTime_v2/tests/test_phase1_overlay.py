"""Regression tests for numerical IPOPT overlay; no simulator is launched."""
from pathlib import Path
import hashlib
import textwrap
import pytest
import casadi as ca
import numpy as np
from solver_overlay_v226 import (
    HESSIAN_SITE, HESSIAN_ADDITION, OPTIMIZER_DECLARATION,
    CONSTRAINT_FN, CONSTRAINT_REPLACEMENT, TRUST_SITE, INIT_SITE,
    FEASIBILITY_SITE, SOLVE_SITE, REPORT_SITE, STRICT_BOUND_DIAGNOSTICS_SITE,
    transform_solver, prepare_working_solver, STRICT_GATE_SITE,
)

LEGACY = 'SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3'
SOLVER = 'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py'


def mock_historical_source():
    # A minimal site-compatibility fixture, not a stand-in for actual solver.
    # The real overlay compares exact fragments of the restored historical file.
    return ('def solve_bottom_up():\n'
            + OPTIMIZER_DECLARATION
            + '    x = opti.variable(2)\n'
            + '    constraint_labels = []\n'
            + CONSTRAINT_FN
            + '    trust_region_bound = opti.parameter()\n'
            + TRUST_SITE
            + INIT_SITE.replace('    try:\n        initial_constraints',
                                '    try:\n        initial_constraints = None')
            + '\n    except Exception:\n        pass\n'
            + '    opti.solver("ipopt", {"ipopt":{\n'
            + '         ' + HESSIAN_SITE + '    }})\n'
            + '    solver_error = None\n'
            + '    stage_reports = []\n'
            + '    feasibility_max_iterations = 50\n'
            + FEASIBILITY_SITE + '\n'
            + '    if initial_is_feasible:\n        pass\n'
            + SOLVE_SITE.replace('    def stage_values',
                                 '    def stage_values')
            + '(solution, stats, expression):\n        return expression\n'
            + STRICT_BOUND_DIAGNOSTICS_SITE + '\n            return {}\n'
            + '        except Exception:\n            return {}\n'
            + REPORT_SITE
            + STRICT_GATE_SITE + '\n        pass\n')


def _sample(tmp_path):
    restored = tmp_path / 'original'
    original = restored / LEGACY / SOLVER
    original.parent.mkdir(parents=True)
    text = mock_historical_source()
    original.write_text(text)
    return restored, original, text


def test_transition_to_zero_defect_and_old_cache_upgrade(tmp_path):
    restored, original, text = _sample(tmp_path)
    # Mimic a real v2.2.2 user directory with a previous L-BFGS working file.
    target = tmp_path/'working'
    previous=target/LEGACY/SOLVER
    previous.parent.mkdir(parents=True)
    previous.write_text(text.replace(HESSIAN_SITE, HESSIAN_ADDITION, 1))
    working, meta = prepare_working_solver(restored,target,LEGACY)
    updated = (working/LEGACY/SOLVER).read_text()
    assert 'phase1_target_schedule = (0.5, 0.35, 0.20, 0.10, 0.05, 0.02, 0.005, 0.0)' in updated
    assert 'opti.set_value(phase1_defect_tolerance, 0.0)' in updated
    assert 'if True:' in updated  # limited-memory branch
    assert 'if (phase1_reached_strict' in updated
    assert 'if not accepted:' in updated
    assert updated.count('opti.bounded(') == 1
    assert 'all(any(name.startswith(prefix)' in updated
    assert '1.0e12' in updated
    assert 'never' in updated.lower() or 'previous OptiSol' in updated
    assert original.read_text() == text
    assert meta['source_untouched']
    assert meta['original_solver_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    again,meta2=prepare_working_solver(restored,target,LEGACY)
    assert again==working and meta2==meta


def test_exact_hessian_mode_keeps_physical_inequalities(tmp_path):
    restored,original,text=_sample(tmp_path)
    working,meta=prepare_working_solver(restored,tmp_path/'exact',LEGACY,'exact')
    patch=(working/LEGACY/SOLVER).read_text()
    assert 'if True:' not in patch
    assert 'if False:' in patch
    assert 'dynamic_equality_tolerance' in patch
    assert 'if False:' in patch and HESSIAN_ADDITION not in patch
    assert original.read_text()==text


def test_source_mismatch_fails_closed(tmp_path):
    restored,original,text=_sample(tmp_path)
    original.write_text(text.replace('initial_is_feasible = (', 'initial_is_feasible = True #'))
    with pytest.raises(RuntimeError,match='source mismatch'):
        prepare_working_solver(restored,tmp_path/'working',LEGACY)


def test_unknown_existing_cache_edits_are_not_overwritten(tmp_path):
    restored,original,text=_sample(tmp_path)
    existing=tmp_path/'working'/LEGACY/SOLVER
    existing.parent.mkdir(parents=True)
    existing.write_text('USER MODIFIED')
    with pytest.raises(RuntimeError,match='unknown edits'):
        prepare_working_solver(restored,tmp_path/'working',LEGACY)
    assert existing.read_text()=='USER MODIFIED'


def test_casadi_dynamic_homotopy_only_relaxes_labeled_equalities():
    # Compile the actual injected helper body and check the Opti constraint
    # expressions in a toy CasADi problem (no wheel model approximation).
    opti=ca.Opti()
    x=opti.variable()
    phase1_defect_tolerance=opti.parameter()
    opti.set_value(phase1_defect_tolerance,0.5)
    constraint_labels=[]
    phase1_defect_residuals=[]
    scope={'ca':ca,'opti':opti,'phase1_defect_tolerance':phase1_defect_tolerance,
           'constraint_labels':constraint_labels,
           'phase1_defect_residuals':phase1_defect_residuals}
    exec(textwrap.dedent(CONSTRAINT_REPLACEMENT),scope)
    add=scope['add_constraint']
    add(x==1.,'lateral_force_balance_k0')
    add(x>=0.0,'physical_left_track_boundary_k0')
    add(x<=0.8,'physical_right_track_boundary_k0')
    assert constraint_labels==['lateral_force_balance_k0',
                               'physical_left_track_boundary_k0',
                               'physical_right_track_boundary_k0']
    opti.set_initial(x,0.8)
    g=np.asarray(opti.value(opti.g,opti.initial())).flatten()
    l=np.asarray(opti.value(opti.lbg,opti.initial())).flatten()
    u=np.asarray(opti.value(opti.ubg,opti.initial())).flatten()
    assert np.all(g >= l-1.e-8) and np.all(g <= u+1.e-8)
    opti.set_value(phase1_defect_tolerance,0.0)
    l=np.asarray(opti.value(opti.lbg,opti.initial())).flatten()
    u=np.asarray(opti.value(opti.ubg,opti.initial())).flatten()
    assert any(g < l-1e-8) or any(g > u+1e-8)
    assert np.isclose(u[0],0.0) and np.isclose(l[0],0.0)


def test_nondynamics_equality_not_relaxed():
    opti=ca.Opti();x=opti.variable(); p=opti.parameter();opti.set_value(p,0.5)
    labels=[];scope={'ca':ca,'opti':opti,'phase1_defect_tolerance':p,'constraint_labels':labels,'phase1_defect_residuals':[]}
    exec(textwrap.dedent(CONSTRAINT_REPLACEMENT),scope)
    scope['add_constraint'](x==0,'periodic_phase_lengths_sum')
    op=opti.g
    assert str(opti.lbg)=='0' and str(opti.ubg)=='0'



def test_previous_opti_solution_retains_relaxed_bounds_so_strict_audit_required():
    # Regression for a safety-critical CasADi behavior: OptiSol.value(ubg)
    # snapshots parameter values at solve time. Testing just that snapshot
    # would mistakenly accept a relaxed Phase-I point as a physical solution.
    opti=ca.Opti()
    x=opti.variable()
    defect=opti.parameter()
    opti.set_value(defect,0.5)
    opti.subject_to(opti.bounded(-defect,x-1,defect))
    opti.minimize((x-0.7)**2)
    opti.solver('ipopt',{'print_time':False},{'print_level':0,'max_iter':50})
    result=opti.solve_limited()
    assert float(result.value(opti.ubg))==pytest.approx(0.5)
    opti.set_initial(x, result.value(x))
    opti.set_value(defect,0.)
    strict_upper=float(opti.value(opti.ubg,opti.initial()))
    assert strict_upper==pytest.approx(0.)
    assert abs(float(result.value(opti.g))) > 1.e-3


def test_upgrade_existing_v223_working_solver_cache(tmp_path):
    # User is applying ZIP over v2.2.3; its prior working solver must be
    # accepted without deleting the restored historical source.
    from solver_overlay_v223 import transform_solver as v223_transform
    restored, original, text = _sample(tmp_path)
    working = tmp_path / 'existing_v223'
    cache = working / LEGACY / SOLVER
    cache.parent.mkdir(parents=True)
    cache.write_text(v223_transform(text, 'limited-memory'))
    result, meta = prepare_working_solver(restored, working, LEGACY)
    upgraded = cache.read_text()
    assert result == working.resolve()
    assert upgraded == transform_solver(text, 'limited-memory')
    assert 'if (phase1_reached_strict' in upgraded
    assert original.read_text() == text
    assert meta['source_untouched']


def test_homotopy_does_not_advance_failed_stage():
    from solver_overlay_v226 import SOLVE_REPLACEMENT
    assert 'if not accepted:' in SOLVE_REPLACEMENT
    assert 'if defect_tol == 0.0:' in SOLVE_REPLACEMENT
    assert 'phase1_reached_strict = True' in SOLVE_REPLACEMENT
    assert 'phase1_diagnostics()' in SOLVE_REPLACEMENT
    assert 'opti.set_value(phase1_defect_tolerance, 0.0)' in SOLVE_REPLACEMENT


def test_upgrades_actual_v224_working_cache(tmp_path):
    from solver_overlay_v224 import transform_solver as previous_transform
    restored, original, historical = _sample(tmp_path)
    cache_root = tmp_path / 'working_v224'
    working_path = cache_root / LEGACY / SOLVER
    working_path.parent.mkdir(parents=True)
    working_path.write_text(previous_transform(historical, 'limited-memory'))
    _, meta = prepare_working_solver(restored, cache_root, LEGACY)
    assert working_path.read_text() == transform_solver(historical, 'limited-memory')
    assert meta['source_untouched']
    assert original.read_text() == historical
    assert 'defect-merit' in meta['phase1_method']


def test_defect_merit_drives_a_relaxed_feasible_seed_toward_physics():
    # Regression: v2.2.4 skipped an already feasible relaxed seed, even if it
    # retained large violations of the true equations. This small scalar
    # model proves the replacement objective has a nonzero restoring drive.
    from solver_overlay_v226 import INIT_REPLACEMENT, SOLVE_REPLACEMENT
    assert 'phase1_defect_energy + 1.0e-3 * feasibility_objective' in INIT_REPLACEMENT
    assert 'winner_energy' in SOLVE_REPLACEMENT
    assert 'normalized_physics_defect_energy' in SOLVE_REPLACEMENT
    assert 'energy < winner_energy' in SOLVE_REPLACEMENT
    op = ca.Opti()
    x = op.variable()
    tol = op.parameter()
    op.set_value(tol, 0.5)
    # Starting x=0.6 satisfies a relaxed 0.5 defect but not the zero-defect
    # physical equation x=1. This stage must MOVE toward x=1.
    op.subject_to(op.bounded(-tol, x - 1.0, tol))
    op.subject_to(op.bounded(0., x, 1.2))
    op.set_initial(x, 0.6)
    energy = (x-1)**2
    op.minimize(energy + 1.e-3*(x-0.6)**2)
    op.solver('ipopt', {'print_time': False},
              {'print_level': 0, 'max_iter': 90, 'tol': 1.e-9})
    sol = op.solve()
    assert abs(float(sol.value(x))-1) < 1.e-3
    op.set_value(tol, 0.)
    op.set_initial(x,sol.value(x))
    assert abs(float(op.value(op.g, op.initial())[0])) <= 1.e-3


def test_hard_nonphysics_limits_not_added_to_defect_merit():
    op=ca.Opti()
    x=op.variable()
    p=op.parameter()
    op.set_value(p, 0.5)
    residuals=[]
    labels=[]
    scope={'ca':ca,'opti':op,'phase1_defect_tolerance':p,
           'phase1_defect_residuals':residuals,'constraint_labels':labels}
    exec(textwrap.dedent(CONSTRAINT_REPLACEMENT),scope)
    add=scope['add_constraint']
    add(x==1.,'lateral_force_balance_k0')
    add(x>=0.0,'footprint_left_limit')
    add(x<=0.75,'track_boundary_right')
    assert len(residuals)==1
    op.set_initial(x,0.5)
    # The force-balance residual is divided by 10 only in the objective,
    # never in the physical constraint itself.
    weighted=float(op.value(residuals[0], op.initial()))
    assert weighted==pytest.approx(0.05,abs=1.e-10)
    op.set_initial(x,1.0)
    values=np.asarray(op.value(op.g,op.initial())).reshape(-1)
    ub=np.asarray(op.value(op.ubg,op.initial())).reshape(-1)
    assert np.any(values > ub+0.01)
