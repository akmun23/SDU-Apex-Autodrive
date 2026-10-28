import ctypes as C
import math
from types import SimpleNamespace

import numpy as np
import pytest

from command_realizability import (periodic, time_grid, seed_plant,
                                   source_rack_history, inverse_rack_mean)
from physics import SourcePlant, State, _CInput, _COutput


def test_inverse_uses_actual_source_rack_branch_without_future_state_resets(tmp_path):
    from pathlib import Path
    from source_vehicle import _cpp_number
    repo = Path(__file__).resolve().parents[2]
    plant = SourcePlant(repo, tmp_path)
    rate = _cpp_number((repo/'f1tenth_mpc/src/unity_source_plant.cpp').read_text(),
                       'kSteeringRateDegps')
    for current in (-.45, -.1, 0., .1, .45):
        for target in (-.45, -.1, 0., .1, .45):
            plant.reset(0., 0., 0., 4.)
            plant.input.state[13] = current
            actual = plant.step(target, .1)
            assert source_rack_history(current, target, rate)[-1] == pytest.approx(actual.steering, abs=1.e-14)
    assert inverse_rack_mean(.1, .1, rate) == pytest.approx(.1, abs=1.e-8)


def test_adapter_uses_integrated_rear_axle_motion_not_endpoint_com_twist():
    plant = SourcePlant.__new__(SourcePlant)
    plant.input, plant.output = _CInput(), _COutput()
    plant.state = State(1., 2., math.pi/2, 100., 50., 9., 0., [], [], [])
    def step(inp, duration, out):
        assert float(duration.value) == .025
        plant.output.integrated_body_forward_displacement_m = .3
        plant.output.integrated_body_lateral_displacement_m = -.1
        plant.output.integrated_yaw_delta_rad = .02
        plant.output.state[0] = 99.
        return 1
    plant.lib = SimpleNamespace(unity_source_plant_step_c=step)
    result = plant.step(.2, .1)
    assert result.x == pytest.approx(1.1)
    assert result.y == pytest.approx(2.3)
    assert result.yaw == pytest.approx(math.pi/2+.02)


def test_periodic_command_never_uses_future_observations():
    times = np.array([0., .1, .2])
    assert periodic(times, [.2, .4], .05) == pytest.approx(.3)
    assert periodic(times, [.2, .4], .25) == pytest.approx(.3)


def test_time_grid_reproduces_known_straight_progress_and_checks_lap():
    track = SimpleNamespace(s=np.array([0., 1., 2.]), length=3., count=3,
                            psi=np.zeros(3), x=np.array([0., 1., 2.]), y=np.zeros(3))
    # A closed track needs positive progress on the closing segment too;
    # use a regular polygon with its actual chord tangent instead.
    angle = np.arange(8)*2*math.pi/8
    xy = np.column_stack([np.cos(angle), np.sin(angle)])
    ds = np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1)
    direction = np.roll(xy, -1, axis=0)-xy
    track = SimpleNamespace(count=8, s=np.r_[0., np.cumsum(ds)[:-1]],
        length=sum(ds), x=xy[:, 0], y=xy[:, 1],
        psi=np.arctan2(direction[:, 1], direction[:, 0]))
    states = np.zeros((16, 8))
    states[2] = 4.
    times = time_grid(track, states, track.length/4.)
    assert times[-1] == pytest.approx(track.length/4.)
    with pytest.raises(ValueError, match='saved'):
        time_grid(track, states, 8.)


def test_seed_mapping_keeps_com_motion_rack_and_suspension_separate():
    plant = SourcePlant.__new__(SourcePlant)
    plant.input, plant.output = _CInput(), _COutput()
    state = np.arange(16.)/100.
    seed_plant(plant, state, [1., 2., .5])
    np.testing.assert_allclose(plant.input.state[:3], state[2:5])
    np.testing.assert_allclose(plant.input.state[3:7], state[5:9])
    np.testing.assert_allclose(plant.input.state[7:13], state[10:16])
    assert plant.input.state[13] == state[9]
    assert not plant.input.suspension_history_initialized
