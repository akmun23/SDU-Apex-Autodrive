import io
from contextlib import redirect_stdout
from pathlib import Path
from diagnostic_fixtures.diagnose_report import summary
from run_ipopt import (print_feasibility_diagnostics,
                       competition_wall_margin,
                       run_phase1_or_diagnose,
                       validate_initial_diagnostic_seed_mode,
                       validate_native_diagnostic_capture_mode)
from map_track import DEFAULT_WALL_MARGIN_M


def test_diagnoses_observed_25_unit_trust_limit():
    report={'solver':{'return_status':'Maximum_Iterations_Exceeded','feasible_incumbent':False},
            'nlp_diagnostics':{'max_constraint_violation':6.7362661,
                               'worst_constraints':[{'label':'captured_motion_trust_region','violation':6.7362661}]}}
    out=summary(report)
    assert 'captured_motion_trust_region' in out
    assert 'not a vehicle physics constraint' in out
    buf=io.StringIO()
    with redirect_stdout(buf):print_feasibility_diagnostics(report,25.)
    assert 'artificial bound' in buf.getvalue()


def test_does_not_mislabel_dynamics_violation_as_trust():
    report={'nlp_diagnostics':{'max_constraint_violation':.5,
            'worst_constraints':[{'label':'periodic_collocation_state_u_mps','violation':.5}]}}
    buf=io.StringIO()
    with redirect_stdout(buf):print_feasibility_diagnostics(report,200.)
    assert 'periodic_collocation' in buf.getvalue()
    assert 'artificial Phase-I' not in buf.getvalue()


def test_initial_only_diagnostic_rejects_implicit_native_to_synthetic_seed():
    try:
        validate_initial_diagnostic_seed_mode(
            diagnose_initial_only=True, seed_file=None, seed_backend='native')
    except ValueError as error:
        assert 'does not generate a native/source warm-start' in str(error)
    else:
        raise AssertionError('native diagnostic silently substituted synthetic seed')


def test_initial_only_diagnostic_accepts_explicit_synthetic_or_saved_seed():
    validate_initial_diagnostic_seed_mode(
        diagnose_initial_only=True, seed_file=None, seed_backend='synthetic')
    validate_initial_diagnostic_seed_mode(
        diagnose_initial_only=True, seed_file=Path('seed.npz'),
        seed_backend='native')


def test_native_diagnostic_capture_requires_fresh_native_trial():
    validate_native_diagnostic_capture_mode(
        capture=False, seed_file=None, seed_backend='native',
        skip_native=False, setup_check=False, diagnose_initial_only=False)
    validate_native_diagnostic_capture_mode(
        capture=True, seed_file=None, seed_backend='native',
        skip_native=False, setup_check=False, diagnose_initial_only=False)
    invalid_modes = (
        {'seed_backend':'source'},
        {'seed_file':Path('seed.npz')},
        {'skip_native':True},
        {'setup_check':True},
        {'diagnose_initial_only':True},
    )
    for invalid in invalid_modes:
        options = dict(capture=True, seed_file=None, seed_backend='native',
                       skip_native=False, setup_check=False,
                       diagnose_initial_only=False)
        options.update(invalid)
        try:
            validate_native_diagnostic_capture_mode(**options)
        except ValueError:
            pass
        else:
            raise AssertionError(f'capture accepted incompatible mode: {invalid}')


def test_initial_diagnostic_prints_equation_label_and_seed_source():
    report = {
        'solver': {'return_status': 'initial_guess_diagnostic_only'},
        'initial_guess_source': {'kind': 'synthetic'},
        'nlp_diagnostics': {
            'initial_guess': {'max_constraint_violation': 0.0,
                              'nonfinite_constraint_count': 0},
            'initial_raw_dynamic_max_defect': 16.1,
            'initial_raw_dynamic_defects': {
                'unity_spring_damper_load': {
                    'max_abs': 16.1, 'rmse': 1.9,
                    'worst_constraint_label':
                        'unity_spring_damper_load_midpoint_w2_k37'}}}}
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_feasibility_diagnostics(report, 200.)
    output = buf.getvalue()
    assert 'mix physical units and scaled collocation residuals' in output
    assert 'unity_spring_damper_load_midpoint_w2_k37' in output
    assert 'Actual seed source: synthetic' in output


def test_initial_only_diagnostic_bypasses_phase1_continuation():
    expected_report = {'solver': {'return_status': 'initial_guess_diagnostic_only'}}

    class DiagnosticSolver:
        def __init__(self):
            self.calls = 0

        def solve_bottom_up(self, track, model, **kwargs):
            self.calls += 1
            assert kwargs['diagnose_initial_only'] is True
            return {}, expected_report

    def forbidden_continuation(*args, **kwargs):
        raise AssertionError('diagnostic entered the Phase-I continuation path')

    solver = DiagnosticSolver()
    candidate, report, progress = run_phase1_or_diagnose(
        solver, object(), object(), {'diagnose_initial_only': True},
        output=Path('unused'), phase=0, max_restarts=0,
        minimum_improvement=1.0e-5, diagnose_initial_only=True,
        continuation=forbidden_continuation)
    assert candidate == {}
    assert report is expected_report
    assert progress == []
    assert solver.calls == 1


def test_competition_wall_margin_is_loaded_from_runtime_profile(tmp_path):
    config_path = tmp_path / 'f1tenth_mpc/config/mpc_competition.yaml'
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '/**:\n  ros__parameters:\n    corridor_margin_m: 0.05\n',
        encoding='utf-8')
    assert competition_wall_margin(tmp_path) == 0.05


def test_current_runtime_and_offline_map_defaults_share_five_cm_margin():
    repo_root = Path(__file__).resolve().parents[2]
    assert competition_wall_margin(repo_root) == DEFAULT_WALL_MARGIN_M == 0.05
