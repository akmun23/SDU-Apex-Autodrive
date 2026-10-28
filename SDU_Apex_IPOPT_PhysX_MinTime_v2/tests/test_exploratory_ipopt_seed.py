"""Exercise the saved raw-stage → Phase-I initializer contract without IPOPT."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from make_exploratory_ipopt_seed import (  # noqa: E402
    make_seed, convert_com_path_to_rear_axle)
from run_ipopt import (load_optimizer_seed_checkpoint,  # noqa: E402
                       preserve_optimizer_seed_inputs,
                       write_optimizer_seed_checkpoint)


def test_exact_raw_restart_keeps_commands_without_claiming_feasibility():
    modification = {
        'full_differential_state_transferred': True, 'fraction': 1.0,
        'com_path_converted_to_rear_axle': False, 'clipped_speed_samples': 0,
        'source_feasible': False, 'requires_strict_phase1_recertification': True,
    }
    def preserves(value, **kwargs):
        return preserve_optimizer_seed_inputs(
            value, speed_replaced=kwargs.get('speed', False),
            reproject_requested=kwargs.get('reproject', False))
    assert preserves(modification)
    assert preserves({})
    assert not preserves(modification, speed=True)
    assert not preserves(modification, reproject=True)
    for field, changed in (
            ('fraction', .5), ('full_differential_state_transferred', False),
            ('com_path_converted_to_rear_axle', True),
            ('clipped_speed_samples', 1)):
        assert not preserves({**modification, field: changed})
    assert not preserves({'fraction': 1.0})
    assert modification['source_feasible'] is False
    assert modification['requires_strict_phase1_recertification'] is True


def test_raw_wide_iterate_becomes_explicitly_uncertified_phase1_seed(tmp_path):
    station = np.array([0.0, 3.0, 6.0])
    track = SimpleNamespace(count=3, length=9.0, s=station)
    model = SimpleNamespace(metadata=lambda: {"vehicle": "practice"})
    map_path = tmp_path / "map.pgm"
    line_path = tmp_path / "reference.csv"
    map_path.write_text("test map")
    line_path.write_text("test line")
    zeros = np.zeros(3)
    candidate = {
        "ey_m": zeros.copy(), "epsi_rad": zeros.copy(),
        "speed_mps": np.full(3, 4.0), "lateral_speed_mps": zeros.copy(),
        "yaw_rate_radps": zeros.copy(),
        "wheel_omega_radps": np.ones((4, 3)),
        "steering_rad": zeros.copy(), "steering_rate_radps": zeros.copy(),
        "throttle_norm": np.full(3, 0.1),
        "ax_mps2": zeros.copy(), "ay_mps2": zeros.copy(),
        "normal_load_n": np.ones((4, 3)),
        "body_height_m": zeros.copy(), "body_height_rate_mps": zeros.copy(),
        "roll_rad": zeros.copy(), "roll_rate_radps": zeros.copy(),
        "pitch_rad": zeros.copy(), "pitch_rate_radps": zeros.copy(),
    }
    base = tmp_path / "certified.npz"
    write_optimizer_seed_checkpoint(candidate, base, track=track,
                                    map_path=map_path,
                                    centerline_path=line_path, model=model)
    with np.load(base, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    explore = tmp_path / "explore"
    raw_dir = explore / "time_stage_iterates"
    raw_dir.mkdir(parents=True)
    (explore / "provenance.json").write_text(json.dumps({
        "optimizer_checkpoint_seed": {"path": str(base)},
        "map_sha256": metadata["map_sha256"],
        "centerline_sha256": metadata["centerline_sha256"],
        "model": {"model_metadata": model.metadata()},
        "optimizer_reference_track_mesh": {"nodes": 3},
    }))
    state = np.zeros((16, 3))
    state[0] = [0.4, 0.0, -0.2]
    state[1] = [0.0, 0.2, 0.0]
    state[2] = [5.0, 5.5, 6.0]
    raw = raw_dir / "factor_1e+12_raw_iterate.npz"
    np.savez_compressed(raw, schema=1, sample_s=station, state_x=state,
                        trust_factor=1e12, predicted_lap_time_s=4.95,
                        max_constraint_violation=0.04,
                        steering_rate_radps=np.full(3,.2),
                        throttle_norm=np.full(3,.15))
    output = tmp_path / "exploratory_seed.npz"
    modification = make_seed(raw_stage=raw, base_seed=base, output=output,
                             fraction=0.25)
    loaded, loaded_station = load_optimizer_seed_checkpoint(
        output, track=track, map_path=map_path,
        centerline_path=line_path, model=model)
    np.testing.assert_array_equal(loaded_station, station)
    np.testing.assert_allclose(loaded["ey_m"], [0.1, 0.0, -0.05])
    np.testing.assert_allclose(loaded["epsi_rad"], [0.0, 0.05, 0.0])
    np.testing.assert_allclose(loaded["u_mps"], [4.25, 4.375, 4.5])
    assert loaded["_seed_modification"]["source_feasible"] is False
    assert modification["requires_strict_phase1_recertification"] is True
    full = tmp_path / 'full_state.npz'
    make_seed(raw_stage=raw, base_seed=base, output=full,
              fraction=1.0, full_state=True)
    with np.load(full) as archive:
        np.testing.assert_array_equal(archive['wheel_omega_radps'], state[5:9])
        np.testing.assert_array_equal(archive['steering_rad'], state[9])
        np.testing.assert_array_equal(archive['body_height_rate_mps'], state[11])
        np.testing.assert_array_equal(archive['pitch_rate_radps'], state[15])
        np.testing.assert_array_equal(archive['throttle_norm'], np.full(3,.15))
        np.testing.assert_array_equal(archive['normal_load_n'], candidate['normal_load_n'])
    with pytest.raises(FileExistsError):
        make_seed(raw_stage=raw, base_seed=base, output=output, fraction=0.25)
    provenance_path = explore / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["model"]["model_metadata"] = {"vehicle": "different"}
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="model differs"):
        make_seed(raw_stage=raw, base_seed=base,
                  output=tmp_path / "wrong_model.npz", fraction=0.25)


def test_raw_stage_refuses_unrelated_certified_seed(tmp_path):
    raw_dir = tmp_path / "explore" / "time_stage_iterates"
    raw_dir.mkdir(parents=True)
    (raw_dir.parent / "provenance.json").write_text(json.dumps({
        "optimizer_checkpoint_seed": {"path": str(tmp_path / "other.npz")},
    }))
    raw = raw_dir / "factor_1e+12_raw_iterate.npz"
    raw.write_bytes(b"placeholder")
    base = tmp_path / "certified.npz"
    base.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="matching provenance"):
        make_seed(raw_stage=raw, base_seed=base,
                  output=tmp_path / "seed.npz", fraction=0.25)


def test_com_to_rear_seed_conversion_preserves_rigid_point_geometry_at_seam():
    # Independent circular geometry oracle. The converter uses the polygonal
    # frame, hence a bounded chord error, not exact circle equality.
    count, radius = 512, 3.0
    angle = np.arange(count)*2*np.pi/count
    chord = 2*radius*np.sin(np.pi/count)
    station = np.arange(count)*chord
    heading = np.arctan2(np.sin(angle+np.pi/2), np.cos(angle+np.pi/2))
    mesh = np.column_stack((station, radius*np.cos(angle),
                            radius*np.sin(angle), heading))
    offset, epsi, pitch = .1, .04, .02
    fields = {'ey_m': np.full(count, offset),
              'epsi_rad': np.full(count, epsi),
              'pitch_rad': np.full(count, pitch),
              'u_mps': np.full(count, 5.),
              'wheel_omega_radps': np.full((4,count), 83.)}
    converted = convert_com_path_to_rear_axle(fields, station, mesh)
    lever = .15532*np.cos(pitch)
    radial, tangent = radius-offset+lever*np.sin(epsi), -lever*np.cos(epsi)
    expected_offset = radius-np.hypot(radial,tangent)
    expected_epsi = epsi-np.arctan2(tangent,radial)
    np.testing.assert_allclose(converted['ey_m'], expected_offset, atol=6.e-5)
    np.testing.assert_allclose(converted['epsi_rad'], expected_epsi, atol=2.e-6)
    # No wheel/body dynamics are changed by this reference-point transport.
    np.testing.assert_array_equal(converted['u_mps'], fields['u_mps'])
    np.testing.assert_array_equal(converted['wheel_omega_radps'],
                                  fields['wheel_omega_radps'])
    assert abs(converted['epsi_rad'][0]-converted['epsi_rad'][-1]) < 1.e-10
