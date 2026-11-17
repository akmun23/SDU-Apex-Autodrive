"""Reserve regression tests. Mock replays test gates, not vehicle accuracy."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import casadi as ca
import numpy as np
import pytest

import command_ocp as ocp
from plant_library import SourcePlant


@pytest.fixture
def certificate(tmp_path, monkeypatch):
    # A one-packet symbolic circuit isolates certification policy from plant_library.
    state = np.zeros((21, 2))
    state[2, 1] = 2*np.pi
    state[3] = 1.
    controls = np.array([[.1], [.05]])
    seed, map_path = tmp_path/'seed.npz', tmp_path/'map.pgm'
    ocp.save_command_checkpoint(seed, state, controls, 'fixture', 0., 0)
    map_path.write_bytes(b'fixture only')
    p = ca.SX.sym('p', 23)
    next_state = p[:21]+ca.DM([0., 0., 2*np.pi, *([0.]*18)])
    function = ca.Function('fixture_step', [p], [next_state])
    monkeypatch.setattr(ocp, 'production_command_map', lambda *a: (function, 'fixture'))
    values = dict(replay_slip=.006, shooting_slip=.006, excursion_tick=7)
    margins = []
    class Grid:
        resolution = .05
        def footprint_clearance(self, *pose, margin):
            margins.append(margin)
            return .1-margin
    monkeypatch.setattr(ocp.MapGrid, 'load', lambda *a: Grid())
    class Plant:
        def __init__(self, *a):
            self.input = SimpleNamespace(state=np.zeros(26))
            self.tick = 0
        def reset(self, x, y, yaw, speed):
            self.state = SimpleNamespace(x=x, y=y, yaw=yaw)
        def step(self, *control, duration_s):
            assert duration_s == .001
            self.tick += 1
            slip = values['replay_slip'] if self.tick == values['excursion_tick'] else .006
            self.state = SimpleNamespace(x=0., y=0., yaw=self.tick*2*np.pi/25,
                                         lateral_slip=[1., -1., slip, -slip])
            return self.state
    monkeypatch.setattr(ocp, 'SourcePlant', Plant)
    def rear_slip(*a):
        slip = np.full(50, .006)
        slip[2*(values['excursion_tick']-1)+1] = values['shooting_slip']
        return ca.Function('fixture_slip', [p], [ca.DM(slip)])
    monkeypatch.setattr(ocp, 'production_rear_slip', rear_slip)
    monkeypatch.setattr(ocp, 'export_reference', lambda *a: dict(fixture_only=True))
    return dict(repo=tmp_path, path=seed, map_path=map_path,
                output=tmp_path/'certified'), values, margins


@pytest.mark.parametrize('slip', [.009, -.009, float('nan')])
def test_certifier_rejects_an_interior_tick_rear_slip_violation(certificate, slip):
    kwargs, values, _ = certificate
    values['replay_slip'] = slip
    report = ocp.certify_witness(**kwargs, rear_slip_limit=.008)
    assert report['status'] == 'no_validated_witness'
    assert report['rear_slip_limit'] == .008
    assert not report['rollout']['source_tick_rear_slip_validated']
    assert 'reference_export' not in report


def test_certifier_also_checks_slip_from_each_shooting_state(certificate):
    kwargs, values, _ = certificate
    values['shooting_slip'] = -.009
    report = ocp.certify_witness(**kwargs, rear_slip_limit=.008)
    assert report['status'] == 'no_validated_witness'
    assert report['rollout']['maximum_1ms_rear_lateral_slip'] == .006
    assert report['rollout']['maximum_shooting_rear_lateral_slip'] == .009


def test_certifier_accepts_bound_and_uses_requested_margin(certificate):
    kwargs, values, margins = certificate
    values.update(replay_slip=.008, shooting_slip=-.008)
    report = ocp.certify_witness(**kwargs, margin=.092, rear_slip_limit=.008)
    assert report['status'] == 'source_witness_passed'
    assert report['required_margin_m'] == .092
    assert len(margins) == 26 and set(margins) == {.092}
    assert report['rollout']['minimum_1ms_footprint_slack_m'] == pytest.approx(.008)
    assert report['rollout']['maximum_1ms_rear_lateral_slip'] == .008


def test_larger_margin_is_not_silently_certified_at_default(certificate):
    kwargs, _, margins = certificate
    report = ocp.certify_witness(**kwargs, margin=.101)
    assert report['status'] == 'no_validated_witness'
    assert set(margins) == {.101}


def test_slip_option_off_preserves_certification_and_default_margin(certificate, monkeypatch):
    kwargs, values, margins = certificate
    values['replay_slip'] = .2
    monkeypatch.setattr(ocp, 'production_rear_slip', lambda *a: pytest.fail('option off'))
    report = ocp.certify_witness(**kwargs)
    assert report['status'] == 'source_witness_passed'
    assert report['rear_slip_limit'] is None
    assert set(margins) == {.050}
    assert 'source_tick_rear_slip_validated' not in report['rollout']


@pytest.mark.parametrize('limit', [0., -.001, float('inf'), float('nan')])
def test_invalid_slip_limit_rejected_before_writing(certificate, limit):
    kwargs, _, _ = certificate
    with pytest.raises(ValueError, match='finite and positive'):
        ocp.certify_witness(**kwargs, rear_slip_limit=limit)
    assert not kwargs['output'].exists()


@pytest.mark.parametrize('validate_only', [False, True])
def test_command_cli_passes_same_reserves_to_solve_and_validate(tmp_path, monkeypatch, validate_only):
    calls = []
    def solve(**kwargs):
        calls.append(kwargs)
        return dict(status='source_witness_passed')
    def certify(*args, **kwargs):
        calls.append(kwargs)
    monkeypatch.setattr(ocp, 'solve_witness', solve)
    monkeypatch.setattr(ocp, 'certify_witness', certify)
    args = ['command_ocp', '--command-seed', str(tmp_path/'seed.npz'),
            '--map', str(tmp_path/'map.pgm'), '--output', str(tmp_path/'out'),
            '--margin', '.092', '--rear-slip-limit', '.008']
    monkeypatch.setattr(sys, 'argv', args+(['--validate-only'] if validate_only else []))
    ocp.main()
    assert len(calls) == 1
    assert calls[0]['margin'] == .092 and calls[0]['rear_slip_limit'] == .008


def test_shell_launcher_forwards_reserve_options(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    tools = tmp_path/'tools/raceline'
    tools.mkdir(parents=True)
    launcher = tools/'run_raceline_campaign.sh'
    launcher.write_bytes((repo/'tools/raceline/run_raceline_campaign.sh').read_bytes())
    python = tmp_path/'raceline_optimizer/.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    args = ['map.yaml', '--status', '--margin', '.092', '--rear-slip-limit', '.008']
    result = subprocess.run(['bash', str(launcher), *args], check=True,
                            text=True, capture_output=True)
    assert result.stdout.splitlines()[1:] == args


@pytest.mark.parametrize('enabled', [False, True])
def test_command_problem_adds_only_enabled_signed_tick_bounds(tmp_path, monkeypatch, enabled):
    p = ca.SX.sym('p', 23)
    step = ca.Function('state_fixture', [p], [p[:21]])
    slip = ca.Function('slip_fixture', [p], [ca.repmat(p[4], 50, 1)])
    point = ca.SX.sym('point', 2)
    wall = ca.Function('wall_fixture', [point], [point[0]+1.])
    monkeypatch.setattr(ocp, 'clearance_interpolant', lambda grid: wall)
    monkeypatch.setattr(ocp, 'production_fixed_prefix', lambda a,b,c,packed,*args,**kw: packed[:21, :])
    monkeypatch.setattr(ocp, 'production_rear_slip',
                        lambda *a: slip if enabled else pytest.fail('option off'))
    state, controls = np.zeros((21, 3)), np.full((2, 2), .05)
    state[3] = 1.
    grid = SimpleNamespace(yaw=0., origin=np.zeros(2))
    baseline, *_ = ocp.command_problem(tmp_path, step, 'fixture', state, controls, 1,
                                      grid, .05, 1)
    problem, *_ = ocp.command_problem(tmp_path, step, 'fixture', state, controls, 1,
        grid, .092, 1, rear_slip_limit=.008 if enabled else None)
    assert problem.ng == baseline.ng+(100 if enabled else 0)
    lower = np.asarray(problem.value(problem.lbg, problem.initial())).ravel()
    old_lower = np.asarray(baseline.value(baseline.lbg, baseline.initial())).ravel()
    offset = 100 if enabled else 0
    wall_rows = len(ocp._FOOTPRINT_LOCAL)*11  # two packets: 3 nodes + 4 prefixes each
    np.testing.assert_allclose(lower[-wall_rows:]-old_lower[-wall_rows:], .092-.05)
    if enabled:
        np.testing.assert_array_equal(lower[:offset], -np.ones(offset))
        evaluate = ca.Function('reserve_rows', [problem.x], [problem.g[:offset]])
        flat = np.asarray(problem.value(problem.x, problem.initial())).ravel()
        # The first decision is the column-major shooting state; v_COM at 4.
        flat[4] = -.009
        assert np.min(np.asarray(evaluate(flat))) == pytest.approx(-.009/.008)


def test_real_rear_slip_trace_matches_independent_ticks_and_forward_ad(tmp_path):
    # Compare rear-slip values and derivatives against independent plant ticks.
    repo = Path(__file__).resolve().parents[2]
    _, digest = ocp.production_command_map(repo, tmp_path/'kernel')
    slip = ocp.production_rear_slip(tmp_path/'kernel', digest)
    plant = SourcePlant(repo, tmp_path/'plant')
    plant.reset(0., 0., 0., 4.)
    for _ in range(4):
        plant.step(.08, .08)
    packed = np.r_[plant.state.x, plant.state.y, plant.state.yaw,
                   plant.input.state[:18], .08, .08]
    expected = []
    for _ in range(25):
        expected.extend(plant.step(.08, .08, duration_s=.001).lateral_slip[2:4])
    np.testing.assert_allclose(np.asarray(slip(packed)).ravel(), expected, atol=1.e-13, rtol=0.)
    symbolic = ca.MX.sym('packed', 23)
    derivative = ca.Function('slip_ad', [symbolic], [ca.jacobian(slip(symbolic), symbolic)])
    actual = np.asarray(derivative(packed))
    np.testing.assert_array_equal(actual[:, :3], np.zeros((50, 3)))
    for column in range(3, 23):
        h = 1.e-6*(ocp.STATE_SCALES[column] if column < 21 else 1.)
        plus, minus = packed.copy(), packed.copy()
        plus[column] += h
        minus[column] -= h
        finite_difference = (np.asarray(slip(plus))-np.asarray(slip(minus))).ravel()/(2*h)
        np.testing.assert_allclose(actual[:, column], finite_difference, rtol=2.e-3, atol=2.e-5)
