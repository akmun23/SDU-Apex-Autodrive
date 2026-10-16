from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/racing"))

from build_hybrid_unity_open_loop_profile import merge_profiles  # noqa: E402


def profiles():
    base = {
        "fixed_delta_time_s": 0.001,
        "physics_ticks_per_command": 25,
        "warmup_command_count": 2,
        "scored_horizon_commands": 4,
        "commands": [
            {"steering_norm": 0.0, "throttle_norm": 0.0},
            {"steering_norm": 0.0, "throttle_norm": 0.0},
            {"steering_norm": 0.1, "throttle_norm": 0.2,
             "has_reference_position": True, "reference_x_m": 0.0,
             "reference_z_m": 0.0},
            {"steering_norm": 0.2, "throttle_norm": 0.3,
             "has_reference_position": True, "reference_x_m": 1.0,
             "reference_z_m": 0.0},
            {"steering_norm": 0.3, "throttle_norm": 0.4,
             "has_reference_position": True, "reference_x_m": 1.0,
             "reference_z_m": 1.0},
            {"steering_norm": 0.4, "throttle_norm": 0.5,
             "has_reference_position": True, "reference_x_m": 0.0,
             "reference_z_m": 1.0},
        ],
    }
    candidate = {
        "fixed_delta_time_s": 0.001,
        "physics_ticks_per_command": 25,
        "warmup_command_count": 2,
        "raceline_path_length_m": 4.0,
        "bottom_up_optimizer_predicted_lap_time_s": 5.6,
        "commands": [
            {"steering_norm": 0.0, "throttle_norm": 0.0},
            {"steering_norm": 0.0, "throttle_norm": 0.0},
            {"route_arc_m": 0.0, "steering_norm": -0.2,
             "throttle_norm": 0.6},
            {"route_arc_m": 1.0, "steering_norm": -0.3,
             "throttle_norm": 0.7},
            {"route_arc_m": 2.0, "steering_norm": -0.4,
             "throttle_norm": 0.8},
            {"route_arc_m": 3.0, "steering_norm": -0.5,
             "throttle_norm": 0.9},
            {"route_arc_m": 3.5, "steering_norm": -0.6,
             "throttle_norm": 0.95},
        ],
    }
    return base, candidate


def test_splice_preserves_warmup_route_and_controls_outside_patch():
    base, candidate = profiles()
    original = copy.deepcopy(base)
    merged = merge_profiles(base, candidate, 0.35, 0.65, 0.10)
    assert merged["commands"][:2] == original["commands"][:2]
    assert [row["reference_x_m"] for row in merged["commands"][2:]] == [
        row["reference_x_m"] for row in original["commands"][2:]]
    assert merged["commands"][2]["steering_norm"] == pytest.approx(0.1)
    assert merged["commands"][5]["throttle_norm"] == pytest.approx(0.5)
    assert base == original


def test_full_patch_uses_candidate_controls_and_records_prediction_as_unmeasured():
    base, candidate = profiles()
    merged = merge_profiles(base, candidate, 0.25, 0.75, 0.125)
    # The two middle base stations lie fully inside the splice interval.
    assert merged["commands"][3]["steering_norm"] == pytest.approx(-0.3)
    assert merged["commands"][4]["throttle_norm"] == pytest.approx(0.8)
    metadata = merged["hybrid_control_splice"]
    assert metadata["candidate_predicted_lap_time_s_is_not_measured"] == 5.6
    assert metadata["candidate_controls_only_inside_patch"] is True


def test_component_mix_can_isolate_throttle_from_steering():
    base, candidate = profiles()
    merged = merge_profiles(base, candidate, 0.25, 0.75, 0.125,
                            steering_mix=0.0, throttle_mix=1.0)
    assert merged["commands"][4]["steering_norm"] == pytest.approx(0.3)
    assert merged["commands"][4]["throttle_norm"] == pytest.approx(0.8)
    metadata = merged["hybrid_control_splice"]
    assert metadata["candidate_steering_mix"] == 0.0
    assert metadata["candidate_throttle_mix"] == 1.0


def test_invalid_progress_and_sampling_are_rejected():
    base, candidate = profiles()
    with pytest.raises(ValueError, match="progress"):
        merge_profiles(base, candidate, 0.0, 0.75, 0.1)
    candidate["commands"][4]["route_arc_m"] = 0.5
    with pytest.raises(ValueError, match="increasing route_arc"):
        merge_profiles(base, candidate, 0.25, 0.75, 0.125)
    with pytest.raises(ValueError, match=r"mixes must be in \[0, 1\]"):
        merge_profiles(base, candidate, 0.25, 0.75, 0.125,
                       steering_mix=1.1)
