import io
from contextlib import redirect_stdout
from diagnose_report import summary
from run_ipopt import print_feasibility_diagnostics


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
