from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from command_ocp import (production_command_map, production_fixed_prefix,
                         mapped_transition, load_command_seed, export_reference)
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


def test_export_keeps_exact_source_positions_and_separates_time_semantics(tmp_path):
    from map_track import MapGrid
    t = np.linspace(0., 1., 1001)
    state = np.zeros((21, len(t)))
    state[0], state[1] = 3.+np.cos(2*np.pi*t), 3.+np.sin(2*np.pi*t)
    state[2], state[3] = 2*np.pi*t+np.pi/2, 2*np.pi*np.ones(len(t))
    path = tmp_path/'trace.npz'
    np.savez(path, state=state, dt_s=.001)
    grid = MapGrid(np.full((120, 120), 255, dtype=np.uint8),
        np.ones((120, 120), dtype=bool), .05, np.zeros(2), 0., np.full((120, 120), 3.))
    report = export_reference(path, grid, tmp_path)
    table = np.loadtxt(report['path'], delimiter=',')
    assert table.shape == (1000, 9)
    np.testing.assert_allclose(table[:, 1:3], state[:2, :-1].T, atol=1.e-10, rtol=0.)
    assert report['exact_command_lap_s'] == 1.
    assert report['body_forward_reference_integral_s'] == pytest.approx(1., abs=2.e-6)
    assert not report['post_export_geometry_smoothing']
