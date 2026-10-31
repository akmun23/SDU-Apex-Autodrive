from pathlib import Path
import json

import casadi as ca
import numpy as np
import pytest

from command_ocp import (production_command_map, production_fixed_prefix,
                         mapped_transition, load_command_seed, export_reference)
from command_ocp import shorten_seed_by_packet
from command_ocp import time_prediction_seed, certify_witness, save_npz_checkpoint
from command_ocp import recondition_command_seed
from command_ocp import command_problem, STATE_SCALES
from command_ocp import source_clearance_refinement, reported_clearance_cuts, clearance_interpolant
import command_ocp
from physics import SourcePlant


@pytest.fixture(scope='module')
def maps(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    build = tmp_path_factory.mktemp('command-kernel')
    function, digest = production_command_map(root, build)
    z = ca.MX.sym('command_test', 23)
    derivative = ca.Function('command_derivative', [z], [ca.jacobian(function(z), z)])
    return function, derivative, SourcePlant(root, build), digest


def test_command_kernel_matches_production_map_pose_and_all_physical_states(maps):
    function, derivative, plant, _ = maps
    for speed, rack, target, q in ((3., .2, -.3, .08), (5., -.1, .25, .12),
                                  (8., -.4, -.35, .003), (12., .1, -.05, .3)):
        plant.reset(1., -2., .6, speed)
        plant.step(rack, q)  # establish genuine suspension/jounce history
        z = np.r_[plant.state.x, plant.state.y, plant.state.yaw,
                  np.array(plant.input.state[:18]), target, q]
        actual = plant.step(target, q)
        expected = np.r_[actual.x, actual.y, actual.yaw, plant.input.state[:18]]
        np.testing.assert_allclose(np.asarray(function(z)).reshape(-1), expected,
                                   atol=1.e-12, rtol=1.e-12)
        jac = np.asarray(derivative(z))
        # Check physical state, pose rotation, both controls, suspension memory.
        for column in (0, 1, 2, 3, 4, 5, 6, 12, 16, 17, 21, 22):
            dz = np.zeros(23)
            dz[column] = 1.e-6
            finite = np.asarray((function(z+dz)-function(z-dz))/(2.e-6)).reshape(-1)
            np.testing.assert_allclose(jac[:, column], finite, rtol=.005, atol=.003)


def test_kernel_rejects_zero_brake_mode(maps):
    function, _, plant, _ = maps
    plant.reset(0., 0., 0., 5.)
    plant.step(.1, .1)
    z = np.r_[0., 0., 0., plant.input.state[:18], .1, 0.]
    with pytest.raises(RuntimeError): function(z)


def test_kernel_build_is_content_addressed(maps, tmp_path):
    function, _, _, digest = maps
    root = Path(__file__).resolve().parents[2]
    again, digest2 = production_command_map(root, root/'build_task/command_ocp')
    assert digest == digest2
    assert function.sparsity_in(0).shape == (23, 1)
    assert again.sparsity_out(0).shape == (21, 1)


def test_fixed_prefix_matches_composed_production_ticks_and_derivatives():
    root = Path(__file__).resolve().parents[2]
    build = root/'build_task/command_ocp'
    function, digest = production_command_map(root, build)
    plant = SourcePlant(root, build)
    plant.reset(1., -2., .7, 7.)
    plant.step(.2, .1)
    value = np.r_[plant.state.x, plant.state.y, plant.state.yaw,
                  plant.input.state[:18], -.15, .08]
    z = ca.MX.sym('prefix_test', 23)
    first = production_fixed_prefix(root, build, digest, z, 12)
    second = production_fixed_prefix(root, build, digest, ca.vertcat(first, z[21:]), 13)
    compose = ca.Function('compose_test', [z], [second, ca.jacobian(second, z)])
    expected = ca.Function('whole_test', [z], [function(z), ca.jacobian(function(z), z)])
    for actual, target in zip(compose(value), expected(value)):
        np.testing.assert_allclose(actual, target, rtol=2.e-4, atol=.001)
    for bad in (0, 26, 12.5, ca.MX.sym('not_a_decision')):
        with pytest.raises(ValueError):
            production_fixed_prefix(root, build, digest, z, bad)


def test_all_literal_prefixes_match_independent_one_ms_production_steps(maps):
    _, _, plant, digest = maps
    root=Path(__file__).resolve().parents[2]
    for speed,rack,q in ((3.,-.25,.08),(8.,.35,.12)):
        plant.reset(1.,-2.,.7,speed)
        plant.step(rack,q)
        packed=np.r_[plant.state.x,plant.state.y,plant.state.yaw,
                     np.asarray(plant.input.state[:18]),-.1,q]
        for tick in range(1,26):
            actual=plant.step(-.1,q,duration_s=.001)
            expected=np.r_[actual.x,actual.y,actual.yaw,plant.input.state[:18]]
            prefix=production_fixed_prefix(root,root/'build_task/command_ocp',digest,
                                            ca.DM(packed),tick)
            np.testing.assert_allclose(np.asarray(prefix).reshape(-1),expected,
                                       atol=1.e-10,rtol=1.e-10)


def test_refinement_selects_worst_unconstrained_tick_not_duplicate_base_prefix():
    report=dict(full_lap_xy_max_m=.0001,violating_source_samples=[
        dict(interval=1,tick=12,slack_m=-.003),
        dict(interval=1,tick=11,slack_m=-.002),
        dict(interval=1,tick=10,slack_m=-.001),
        dict(interval=2,tick=7,slack_m=-.001),
        dict(interval=2,tick=8,slack_m=-.002)])
    assert source_clearance_refinement(report,[(1,11)],3)==[(1,10),(2,8)]
    report['full_lap_xy_max_m']=.051
    assert source_clearance_refinement(report,[],3)==[]
    report['full_lap_xy_max_m']=float('nan')
    assert source_clearance_refinement(report,[],3)==[]


def test_reported_missing_ticks_take_precedence_over_already_constrained_worst():
    report=dict(requested_source_clearance_cuts=[[2,11]],
        rollout=dict(violating_source_ticks=[dict(interval=2,tick=12)]))
    assert reported_clearance_cuts(report)=={(2,11)}
    del report['requested_source_clearance_cuts']
    assert reported_clearance_cuts(report)=={(2,12)}


@pytest.mark.parametrize('yaw',[0.,.37])
def test_optimizer_map_interpolation_matches_independent_footprint_checker(yaw):
    from map_track import MapGrid, _FOOTPRINT_LOCAL, _FOOTPRINT_UNSAMPLED_CORRECTION
    rows,cols=90,100
    rr,cc=np.indices((rows,cols))
    grid=MapGrid(np.zeros((rows,cols),np.uint8),np.ones((rows,cols),bool),
                 .025,np.array([1.2,-3.]),yaw,.2+.001*cc+.002*rr+.00001*rr*cc)
    wall=clearance_interpolant(grid)
    c,s=np.cos(yaw),np.sin(yaw)
    for heading in (-.7,.4,1.2):
        x,y=grid.origin+np.array([c*1.2-s*1.1,s*1.2+c*1.1])
        ch,sh=np.cos(heading),np.sin(heading)
        along,side=_FOOTPRINT_LOCAL.T
        wx=x+ch*along-sh*side-grid.origin[0]
        wy=y+sh*along+ch*side-grid.origin[1]
        local=np.vstack([c*wx+s*wy,-s*wx+c*wy])
        nlp=float(np.min(np.asarray(wall.map(len(along),'serial')(local))))- \
            _FOOTPRINT_UNSAMPLED_CORRECTION-.05
        assert nlp==pytest.approx(grid.footprint_clearance(x,y,heading,margin=.05),abs=1.e-12)


@pytest.mark.parametrize('objective',['repair','smooth'])
def test_real_ipopt_callback_stops_on_new_tick_and_recovers_durable_cut(
        tmp_path,monkeypatch,capsys,objective):
    """Tiny orchestration NLP, not a synthetic claim about vehicle dynamics."""
    count=3
    states=np.zeros((21,count+1))
    states[2]=np.linspace(0.,2*np.pi,count+1)
    controls=np.full((2,count),.1)
    seed=tmp_path/'seed.npz'
    save_npz_checkpoint(seed,schema='production-command-ocp/v1',state=states,
                        controls=controls,dt_s=.025)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'fixture map never used for physics')
    calls=[]
    def problem(_repo,_function,_digest,initial_x,initial_u,_winding,_grid,_margin,
                _workers,cuts,**_kwargs):
        calls.append(list(cuts))
        opti=ca.Opti()
        x=opti.variable(*initial_x.shape)
        u=opti.variable(*initial_u.shape)
        opti.set_initial(x,initial_x)
        opti.set_initial(u,initial_u)
        opti.subject_to(x==initial_x)
        opti.subject_to(u==initial_u)
        opti.minimize(ca.sumsqr(x-initial_x)+ca.sumsqr(u-initial_u))
        return opti,x,u,None
    replays=[]
    def replay(_function,x,u,_grid,_margin,**_kwargs):
        replays.append((x.copy(),u.copy()))
        return dict(full_lap_xy_max_m=0.,minimum_1ms_footprint_slack_m=-.001,
                    violating_source_ticks=[dict(interval=1,tick=7,slack_m=-.001)])
    monkeypatch.setattr(command_ocp,'production_command_map',lambda *_:(object(),'fixture'))
    monkeypatch.setattr(command_ocp.MapGrid,'load',lambda *_:object())
    monkeypatch.setattr(command_ocp,'command_problem',problem)
    monkeypatch.setattr(command_ocp,'check_full_witness',replay)
    first=command_ocp.solve_witness(repo=tmp_path,checkpoint=None,mesh=None,audit=None,
        map_path=grid,output=tmp_path/'first',count=count,command_seed=seed,
        workers=1,iterations=20,objective_mode=objective)
    assert first['termination_reason']=='source_clearance_refinement'
    assert first['requested_source_clearance_cuts']==[[1,7]]
    assert first['ipopt_iterations']<20
    with np.load(tmp_path/'first/best_iterate.npz') as checkpoint:
        np.testing.assert_array_equal(checkpoint['state'],replays[0][0])
        np.testing.assert_array_equal(checkpoint['controls'],replays[0][1])
    command_ocp.solve_witness(repo=tmp_path,checkpoint=None,mesh=None,audit=None,
        map_path=grid,output=tmp_path/'second',count=count,
        command_seed=tmp_path/'first/best_iterate.npz',workers=1,iterations=20,
        objective_mode=objective)
    assert calls==[[],[(1,7)]]
    headers=[json.loads(line) for line in capsys.readouterr().out.splitlines()
             if line.startswith('{"stage": "production-command witness"')]
    assert [h['output'] for h in headers]==[
        str((tmp_path/'first').resolve()), str((tmp_path/'second').resolve())]


@pytest.mark.parametrize('terminal_fraction',[-.1,1.1])
def test_predictor_skips_invalid_trial_and_terminal_timing_without_losing_checkpoint(
        tmp_path,monkeypatch,capsys,terminal_fraction):
    """Real callback/terminal path on a tiny NLP, not vehicle performance data."""
    count,sectors=40,4
    states=np.zeros((21,count+1))
    states[2]=np.linspace(0.,2*np.pi,count+1)
    controls=np.full((2,count),.1)
    seed=tmp_path/'seed.npz'
    save_npz_checkpoint(seed,schema='production-command-ocp/v1',state=states,
                        controls=controls,dt_s=.025)
    grid=tmp_path/'map.pgm'
    grid.write_bytes(b'fixture, never used as a physical map')
    def problem(_repo,_function,_digest,initial_x,initial_u,*_args,**_kwargs):
        opti=ca.Opti()
        x=opti.variable(*initial_x.shape)
        u=opti.variable(*initial_u.shape)
        fraction=opti.variable(sectors)
        opti.set_initial(x,initial_x)
        opti.set_initial(u,initial_u)
        opti.set_initial(fraction,1.)
        opti.subject_to(x==initial_x)
        opti.subject_to(u==initial_u)
        # Deliberately force an out-of-domain timing iterate. It must not be
        # clipped into a valid proposal or terminate the surrounding campaign.
        opti.subject_to(fraction==terminal_fraction)
        opti.minimize(ca.sumsqr(fraction-terminal_fraction))
        return opti,x,u,fraction
    monkeypatch.setattr(command_ocp,'production_command_map',lambda *_:(object(),'fixture'))
    monkeypatch.setattr(command_ocp.MapGrid,'load',lambda *_:object())
    monkeypatch.setattr(command_ocp,'command_problem',problem)
    output=tmp_path/'predictor'
    report=command_ocp.solve_time_predictor(repo=tmp_path,command_seed=seed,
        map_path=grid,output=output,time_sectors=sectors,workers=1,iterations=20)
    assert report['status']=='no_time_proposal'
    assert report['source_certified'] is False
    assert report['retained_shorter_proposals']==0
    assert report['invalid_timing_iterates']>=1
    np.testing.assert_allclose(report['last_invalid_timing']['sector_fractions'],terminal_fraction)
    with np.load(output/'best_predictor.npz') as saved:
        np.testing.assert_array_equal(saved['fraction'],np.ones(sectors))
    assert (output/'report.json').exists()
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()
            if line.startswith('{"event": "time_predictor_iterate_skipped"')]
    assert any(e['terminal'] for e in events)
    assert not (output/'proposed_seed.npz').exists()


def test_threaded_maps_preserve_values_and_derivatives_bitwise(maps):
    function, derivative, plant, _ = maps
    plant.reset(0., 0., 0., 6.)
    plant.step(.1, .1)
    value = np.r_[0., 0., 0., plant.input.state[:18], .15, .1]
    batch = np.repeat(value[:, None], 8, axis=1)
    for f in (function, derivative):
        np.testing.assert_array_equal(mapped_transition(f, 8, 1)(batch),
                                      mapped_transition(f, 8, 2)(batch))


def test_dense_seed_keeps_fast_internal_history_missing_from_25ms_nodes(tmp_path):
    # A coordinate/sampling fixture, not a claim this synthetic curve is a car.
    count = 40
    t = np.linspace(0., 1., count*25+1)
    dense = np.zeros((21, len(t)))
    dense[0], dense[1], dense[2] = np.cos(2*np.pi*t), np.sin(2*np.pi*t), 2*np.pi*t
    dense[3] = 2*np.pi
    dense[11] = .1*np.sin(2*np.pi*40*t)
    states = dense[:, ::25]
    controls = np.vstack([np.full(count, .1), np.full(count, .08)])
    path = tmp_path/'witness.npz'
    np.savez(path, schema='production-command-ocp/v1', state=states, controls=controls, dt_s=.025)
    np.savez(tmp_path/'source_trace.npz', state=dense, commands=controls, dt_s=.001)
    sampled, command, winding, duration = load_command_seed(path, 60)
    assert sampled.shape == (21, 61) and command.shape == (2, 60)
    assert np.max(np.abs(sampled[11])) > .08
    assert np.max(np.abs(states[11])) < 1.e-12
    assert winding == 1 and duration == 1.
    np.savez(tmp_path/'source_trace.npz', state=dense, commands=controls*2, dt_s=.001)
    with pytest.raises(ValueError, match='does not match'):
        load_command_seed(path, 60)


@pytest.mark.parametrize('count', [192, 195])
def test_clearance_restart_preserves_primal_and_every_held_packet(tmp_path, count):
    states = np.arange(21*(count+1), dtype=float).reshape(21, count+1)*1.e-5
    states[2, -1] = states[2, 0]+2*np.pi
    controls = np.vstack([np.linspace(-.4, .4, count), np.linspace(.02, .3, count)])
    path = tmp_path/'witness.npz'
    np.savez(path, schema='production-command-ocp/v1', state=states, controls=controls, dt_s=.025)
    # Even a genuine dense replay need not equal all shooting nodes. A restart
    # preserves the primal instead of replacing it by that recursive history.
    dense = np.repeat(states[:, :1], count*25+1, axis=1)
    np.savez(tmp_path/'source_trace.npz', state=dense, commands=controls, dt_s=.001)
    actual, held, winding, lap = load_command_seed(path, count)
    np.testing.assert_array_equal(actual, states)
    np.testing.assert_array_equal(held, controls)
    assert winding == 1 and lap == count*.025


def test_resampling_assigns_shared_packet_edges_using_exact_integers(tmp_path):
    count, new_count = 195, 390
    states = np.zeros((21, count+1))
    states[2] = np.linspace(0., 2*np.pi, count+1)
    controls = np.vstack([np.linspace(-.4, .4, count), np.linspace(.02, .3, count)])
    path = tmp_path/'witness.npz'
    np.savez(path, schema='production-command-ocp/v1', state=states, controls=controls, dt_s=.025)
    _, held, _, _ = load_command_seed(path, new_count)
    np.testing.assert_array_equal(held, np.repeat(controls, 2, axis=1))


def test_packet_shortening_preserves_all_other_states_and_held_controls(tmp_path, maps):
    function, _, plant, _ = maps
    count = 40
    plant.reset(0., 0., 0., 5.)
    plant.step(.1, .1)
    states = np.repeat(np.r_[0., 0., 0., plant.input.state[:18]][:, None], count+1, axis=1)
    states[0] = np.arange(count+1)*.1
    states[2, -1] = 2*np.pi
    controls = np.vstack([np.linspace(.1, .3, count), np.linspace(.05, .2, count)])
    path = tmp_path/'witness.npz'
    np.savez(path, schema='production-command-ocp/v1', state=states, controls=controls, dt_s=.025)
    actual, held, winding, lap = shorten_seed_by_packet(path, count-1, function, 1)
    missing = next(k for k in range(count) if not np.any(held[0] == controls[0, k]))
    np.testing.assert_array_equal(actual, np.delete(states, missing+1, axis=1))
    np.testing.assert_array_equal(held, np.delete(controls, missing, axis=1))
    np.testing.assert_array_equal(actual[:, -1], states[:, -1])
    assert winding == 1 and lap == 1.
    with pytest.raises(ValueError, match='exactly one fewer'):
        shorten_seed_by_packet(path, count-2, function, 1)


def test_export_keeps_exact_source_positions_and_separates_time_semantics(tmp_path):
    from map_track import MapGrid
    t = np.linspace(0., 1., 1001)
    state = np.zeros((21, len(t)))
    state[0], state[1] = 3.+np.cos(2*np.pi*t), 3.+np.sin(2*np.pi*t)
    state[2], state[3] = 2*np.pi*t+np.pi/2, 2*np.pi*np.ones(len(t))
    state[5], state[16] = 3., .2
    path = tmp_path/'trace.npz'
    commands=np.vstack([.1+np.arange(40)*.001, np.full(40, .2)])
    np.savez(path, state=state, dt_s=.001, commands=commands)
    grid = MapGrid(np.full((120, 120), 255, dtype=np.uint8),
        np.ones((120, 120), dtype=bool), .05, np.zeros(2), 0., np.full((120, 120), 3.))
    report = export_reference(path, grid, tmp_path)
    table = np.loadtxt(report['path'], delimiter=',')
    assert table.shape == (1000, 9)
    np.testing.assert_allclose(table[:, 1:3], state[:2, :-1].T, atol=1.e-10, rtol=0.)
    assert report['exact_command_lap_s'] == 1.
    assert report['body_forward_reference_integral_s'] == pytest.approx(1., abs=2.e-6)
    assert not report['post_export_geometry_smoothing']
    witness=np.loadtxt(report['source_witness_csv'], delimiter=',')
    assert witness.shape == (1000, 15)
    assert Path(report['source_witness_csv']).read_text().startswith('# sdu_apex_source_witness/v1\n')
    np.testing.assert_allclose(witness[:, 10], -.15532*state[5, :-1])
    np.testing.assert_allclose(witness[:, 12], .2)
    np.testing.assert_allclose(witness[:, 13], np.repeat(commands[0], 25))
    np.testing.assert_allclose(witness[:, 14], .2)


def test_time_proposal_rounds_up_and_preserves_phase_and_packet_ownership():
    states = np.arange(21*101, dtype=float).reshape(21, 101)
    held = np.vstack([np.linspace(-.4, .4, 100), np.linspace(.01, .4, 100)])
    guessed, commands, time = time_prediction_seed(states, held, .5)
    assert time == pytest.approx(2.45)
    assert commands.shape == (2, 98)
    assert commands.shape[1]*.025 >= time
    np.testing.assert_array_equal(guessed[:, [0, -1]], states[:, [0, -1]])
    np.testing.assert_array_equal(commands, held[:, np.arange(98)*100//98])
    assert time_prediction_seed(states, held, 1.) is None
    with pytest.raises(ValueError):
        time_prediction_seed(states, held, float('nan'))


def test_raw_time_relaxation_is_never_a_source_witness(tmp_path):
    path = tmp_path/'predictor.npz'
    save_npz_checkpoint(path, schema='source-time-predictor/v1', state=np.zeros((21, 41)),
                        controls=np.zeros((2, 40)), fraction=.5, source_certified=False)
    with pytest.raises(ValueError, match='not a production command initializer'):
        load_command_seed(path, 40)
    # Rejected before the compiler/model/map is accessed, even if a caller
    # mistakenly sends the predictor straight to the certification function.
    with pytest.raises(ValueError, match='unknown command schema'):
        certify_witness(tmp_path, path, tmp_path/'unused.pgm', tmp_path/'certified')


@pytest.mark.parametrize('slack,expected',[(0.,'source_witness_passed'),
    (-5.e-7,'source_witness_passed'),(-2.e-6,'no_validated_witness'),
    (-.000172,'no_validated_witness'),(-.00109,'no_validated_witness')])
def test_certification_uses_and_reports_explicit_required_margin(tmp_path, monkeypatch,slack,expected):
    count = 3
    witness = tmp_path/'witness.npz'
    states = np.zeros((21, count+1))
    states[2] = np.linspace(0., 2*np.pi, count+1)
    controls = np.zeros((2, count))
    save_npz_checkpoint(witness, schema='production-command-ocp/v1',
                        state=states, controls=controls, dt_s=.025)

    class Mapped:
        def map(self, columns, _mode):
            assert columns == count
            return lambda _packed: states[:, 1:]

    class Grid:
        @staticmethod
        def load(_path):
            return object()

    checked_margins = []
    def check(_function, _states, _controls, _grid, margin, **_kwargs):
        checked_margins.append(margin)
        return dict(full_lap_xy_max_m=0., minimum_1ms_footprint_slack_m=slack)

    monkeypatch.setattr(command_ocp, 'production_command_map', lambda *_: (Mapped(), 'source'))
    monkeypatch.setattr(command_ocp, 'MapGrid', Grid)
    monkeypatch.setattr(command_ocp, 'check_full_witness', check)
    monkeypatch.setattr(command_ocp, 'export_reference', lambda *_: dict(path='fixture'))

    report = certify_witness(tmp_path, witness, tmp_path/'map.pgm',
                             tmp_path/'certification', margin=.125)

    assert report['status'] == expected
    assert report['required_margin_m'] == .125
    assert checked_margins == [.125]
    assert report['clearance_numerical_tolerance_m']==1.e-6
    assert report['rollout']['minimum_1ms_footprint_slack_m']==slack


def test_clearance_tolerance_is_absolute_and_never_hides_real_misses():
    from map_track import clearance_passes, CLEARANCE_NUMERICAL_TOLERANCE_M
    assert CLEARANCE_NUMERICAL_TOLERANCE_M==1.e-6
    for value in (0.,-1.3e-10,-5.e-7,-1.e-6):
        assert clearance_passes(value)
    for value in (-1.00001e-6,-.000172,-.00109,float('nan'),float('inf')):
        assert not clearance_passes(value)
    assert not clearance_passes(-5.e-7,tolerance_m=0.)
    assert not clearance_passes(0.,tolerance_m=.01)
    assert clearance_passes(.002-5.e-7,.002)
    assert not clearance_passes(.002-2.e-6,.002)


def test_refinement_ignores_only_tolerance_size_clearance_deficits():
    report=dict(full_lap_xy_max_m=0.,violating_source_samples=[
        dict(interval=0,tick=7,slack_m=-5.e-7),
        dict(interval=1,tick=7,slack_m=-2.e-6)])
    assert source_clearance_refinement(report,[],2)==[(1,7)]


@pytest.mark.parametrize('margin', [-.01, float('nan'), float('inf')])
def test_certification_rejects_invalid_margin_before_writing(tmp_path, margin):
    with pytest.raises(ValueError, match='margin'):
        certify_witness(tmp_path, tmp_path/'missing.npz', tmp_path/'map.pgm',
                        tmp_path/'certification', margin=margin)
    assert not (tmp_path/'certification').exists()


def test_source_history_initialization_preserves_body_guess_and_propagates_hidden_states(maps):
    function, _, plant, _ = maps
    plant.reset(0., 0., 0., 3.)
    plant.step(0., .1)
    count = 40
    states = np.repeat(np.r_[0., 0., 0., plant.input.state[:18]][:, None], count+1, axis=1)
    phase = np.linspace(0., 2*np.pi, count+1)
    states[:3] = np.vstack([np.cos(phase), np.sin(phase), phase])
    states[6:10] = 1000.  # deliberately inconsistent numerical initialization
    states[16] = .5
    controls = np.vstack([np.zeros(count), np.full(count, .1)])
    original_controls = controls.copy()
    fixed, report = recondition_command_seed(function, states, controls, 1)
    np.testing.assert_array_equal(fixed[:6], states[:6])
    np.testing.assert_array_equal(controls, original_controls)
    assert report['scaled_defect_after'] < report['scaled_defect_before']
    propagated = np.asarray(mapped_transition(function, count, 1)(
        np.vstack([fixed[:, :-1], controls])))
    np.testing.assert_allclose(propagated[6:], fixed[6:, 1:], atol=1.e-12, rtol=0.)
    # The prescribed body/yaw path above is NOT a motion-prediction fixture.
    assert report['prescribed_body_initializer'] and not report['source_certified']


def test_cost_rescaling_preserves_every_relative_penalty(maps):
    from map_track import MapGrid
    root = Path(__file__).resolve().parents[2]
    function, _, _, digest = maps
    # Literal prefix library used by command_problem lives in its content cache.
    production_command_map(root, root/'build_task/command_ocp')
    count = 4
    initial_x = np.zeros((21, count+1))
    initial_u = np.vstack([np.linspace(-.2, .2, count), np.linspace(.1, .2, count)])
    image = np.full((80, 80), 255, dtype=np.uint8)
    grid = MapGrid(image, image > 0, .05, np.zeros(2), 0., np.full((80, 80), 3.))
    opti, x, u, _ = command_problem(root, function, digest, initial_x, initial_u, 1, grid, .05, 1)
    trial_x = initial_x+STATE_SCALES[:, None]*.03
    trial_u = initial_u+.01
    opti.set_initial(x, trial_x)
    opti.set_initial(u, trial_u)
    old_cost = (.001*np.sum(((trial_x-initial_x)/STATE_SCALES[:, None])**2)/count
                +.001*np.sum((trial_u-initial_u)**2)/count
                +.0001*np.sum((trial_u-np.roll(trial_u,1,axis=1))**2)/count)
    assert float(opti.value(opti.f, opti.initial())) == pytest.approx(old_cost*count/.001)


def test_feasible_checkpoint_retains_quality_instead_of_lowest_residual():
    from command_ocp import iterate_rank, quality_proposal_merit
    assert iterate_rank(5.e-7, .1) < iterate_rank(1.e-12, 1.)
    assert iterate_rank(1.e-12, 1.) < iterate_rank(.001, -.5)
    assert iterate_rank(.002, 10.) < iterate_rank(.01, -100.)
    assert iterate_rank(float('nan'), 0.) > iterate_rank(.1, 100.)
    # A good feasible checkpoint remains the selected witness while a distinct
    # exploratory proposal records a materially better, slightly infeasible point.
    assert iterate_rank(5.e-7, .1) < iterate_rank(.0036, -.31)
    assert quality_proposal_merit(.0036, -.31, .1) == pytest.approx(-.274)
    assert quality_proposal_merit(.01001, -.5, .1) is None
    assert quality_proposal_merit(.001, .099999, .1) is None


def test_equal_time_smoothing_can_leave_old_command_sequence(maps):
    from map_track import MapGrid
    root=Path(__file__).resolve().parents[2]
    function,_,_,digest=maps
    production_command_map(root,root/'build_task/command_ocp')
    initial_x=np.zeros((21,5))
    initial_u=np.array([[.15,-.15,.15,-.15],[.1,.1,.1,.1]])
    image=np.full((80,80),255,dtype=np.uint8)
    grid=MapGrid(image,image>0,.05,np.zeros(2),0.,np.full((80,80),3.))
    opti,x,u,_=command_problem(root,function,digest,initial_x,initial_u,1,grid,
                              .05,1,objective_mode='smooth')
    seed_cost=float(opti.value(opti.f,opti.initial()))
    opti.set_initial(u,np.array([[0.,0.,0.,0.],[.1,.1,.1,.1]]))
    assert float(opti.value(opti.f,opti.initial())) < seed_cost*.01


def test_sector_time_predictor_has_independent_cyclic_timing_decisions(maps):
    from map_track import MapGrid
    root=Path(__file__).resolve().parents[2]
    function,_,_,digest=maps
    production_command_map(root,root/'build_task/command_ocp')
    count=6
    initial_x=np.zeros((21,count+1))
    initial_x[2]=np.linspace(0,2*np.pi,count+1)
    initial_x[3]=4.
    initial_u=np.vstack([np.zeros(count),np.full(count,.1)])
    image=np.full((90,90),255,dtype=np.uint8)
    grid=MapGrid(image,image>0,.05,np.zeros(2),0.,np.full((90,90),4.))
    opti,x,u,theta=command_problem(root,function,digest,initial_x,initial_u,
        1,grid,.05,1,time_predictor=True,time_sectors=3)
    assert theta.shape==(3,1)
    before=float(opti.value(opti.f,opti.initial()))
    assert before==pytest.approx(.15)
    opti.set_initial(theta,[0.,1.,1.])
    after=float(opti.value(opti.f,opti.initial()))
    assert 0 < before-after < .003
    assert opti.ng > 0
    with pytest.raises(ValueError,match='sector timing only'):
        command_problem(root,function,digest,initial_x,initial_u,
            1,grid,.05,1,time_predictor=False,time_sectors=3)



@pytest.mark.parametrize('target,objective',[
    (-.001,'repair'),(float('nan'),'repair'),(.003,'clearance'),(.2,'repair')])
def test_early_checkpoint_requires_valid_repair_clearance_target(
        tmp_path,target,objective):
    from command_ocp import solve_witness
    with pytest.raises(ValueError,match='early certification target'):
        solve_witness(repo=tmp_path,checkpoint=None,mesh=None,audit=None,
            map_path=tmp_path/'not_available.pgm',output=tmp_path/'must_not_write',
            early_clearance_target_m=target,objective_mode=objective)
    assert not (tmp_path/'must_not_write').exists()
