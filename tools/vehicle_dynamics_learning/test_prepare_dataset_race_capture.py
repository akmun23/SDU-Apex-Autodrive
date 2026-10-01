from __future__ import annotations

import unittest

from tools.race_domain_experiment_plan import (
    RACE_DOMAIN_BOUNDARY_SPEED_MPS,
    RACE_DOMAIN_GOVERNOR_MPS,
    RACE_DOMAIN_HARD_LIMIT_MPS,
    RACE_DOMAIN_THROTTLE_SPEED_ANCHORS,
    build_race_domain_boundary_plan,
    build_race_domain_moderate_braking_plan,
    build_race_domain_plan,
)
from tools.vehicle_dynamics_learning.prepare_dataset import (
    _validate_race_domain_capture_events,
)


def _complete_events(seed: int = 17002,
                     boundary_speed_mps: float = 11.0,
                     version: int | None = None) -> list[tuple[int, dict]]:
    plan = build_race_domain_plan(seed, boundary_speed_mps=boundary_speed_mps)
    serialized = [
        {
            "label": block.label,
            "duration_s": block.duration_s,
            "target_speed_mps": block.target_speed_mps,
            "steering_rad": block.steering_rad,
        }
        for block in plan
    ]
    start = {
        "event": "phase_start",
        "profile": "race_domain_continuous",
        "label": "race_domain_continuous",
        "phase_index": 0,
        "phase_count": 1,
        "seed": seed,
        "race_domain_command_plan": serialized,
        "race_domain_speed_governor_mps": RACE_DOMAIN_GOVERNOR_MPS,
        "race_domain_hard_limit_mps": RACE_DOMAIN_HARD_LIMIT_MPS,
        "race_domain_throttle_speed_anchors": [
            {"speed_mps": speed, "throttle_norm": throttle}
            for speed, throttle in RACE_DOMAIN_THROTTLE_SPEED_ANCHORS
        ],
    }
    if version is not None:
        start["race_domain_plan_version"] = version
        start["race_domain_boundary_target_mps"] = boundary_speed_mps
    end = {
        "event": "phase_end",
        "profile": "race_domain_continuous",
        "phase_index": 0,
        "status": "complete",
        "valid": None,
        "quality_failures": [],
    }
    finish = {
        "event": "experiment_end",
        "profile": "race_domain_continuous",
        "phase_count": 1,
        "aborted": False,
        "reason": "schedule complete",
        "quality_failures": [],
    }
    return [(1_000_000_000, start),
            (261_000_000_000, end),
            (261_010_000_000, finish)]


def _complete_boundary_events(seed: int = 17020) -> list[tuple[int, dict]]:
    plan = build_race_domain_boundary_plan(seed)
    serialized = [
        {
            "label": block.label,
            "duration_s": block.duration_s,
            "target_speed_mps": block.target_speed_mps,
            "steering_rad": block.steering_rad,
        }
        for block in plan
    ]
    start = {
        "event": "phase_start",
        "profile": "race_domain_brake_boundary",
        "label": "race_domain_brake_boundary",
        "phase_index": 0,
        "phase_count": 1,
        "seed": seed,
        "race_domain_command_plan": serialized,
        "race_domain_speed_governor_mps": RACE_DOMAIN_GOVERNOR_MPS,
        "race_domain_hard_limit_mps": RACE_DOMAIN_HARD_LIMIT_MPS,
        "race_domain_throttle_speed_anchors": [
            {"speed_mps": speed, "throttle_norm": throttle}
            for speed, throttle in RACE_DOMAIN_THROTTLE_SPEED_ANCHORS
        ],
        "race_domain_plan_version": 3,
        "race_domain_boundary_target_mps": RACE_DOMAIN_BOUNDARY_SPEED_MPS,
    }
    duration_ns = int(sum(block.duration_s for block in plan) * 1e9)
    end = {
        "event": "phase_end",
        "profile": "race_domain_brake_boundary",
        "phase_index": 0,
        "status": "complete",
        "valid": None,
        "quality_failures": [],
    }
    finish = {
        "event": "experiment_end",
        "profile": "race_domain_brake_boundary",
        "phase_count": 1,
        "aborted": False,
        "reason": "schedule complete",
        "quality_failures": [],
    }
    events = [(1_000_000_000, start)]
    events.extend((1_000_000_000 + int(index * block.duration_s * 1e9), {
        "event": "race_domain_block_start",
        "profile": "race_domain_brake_boundary",
        "seed": seed,
        "phase_index": 0,
        "block_index": index,
        "label": block.label,
        "target_speed_mps": block.target_speed_mps,
        "steering_command_rad": block.steering_rad,
        "measured_speed_mps": block.target_speed_mps,
    }) for index, block in enumerate(plan))
    events.extend(((1_000_000_000 + duration_ns, end),
                   (1_010_000_000 + duration_ns, finish)))
    return events


def _complete_moderate_braking_events(seed: int = 17022) -> list[tuple[int, dict]]:
    plan = build_race_domain_moderate_braking_plan(seed)
    serialized = [
        {
            "label": block.label,
            "duration_s": block.duration_s,
            "target_speed_mps": block.target_speed_mps,
            "steering_rad": block.steering_rad,
        }
        for block in plan
    ]
    start = {
        "event": "phase_start",
        "profile": "race_domain_moderate_braking",
        "label": "race_domain_moderate_braking",
        "phase_index": 0,
        "phase_count": 1,
        "seed": seed,
        "race_domain_command_plan": serialized,
        "race_domain_speed_governor_mps": RACE_DOMAIN_GOVERNOR_MPS,
        "race_domain_hard_limit_mps": RACE_DOMAIN_HARD_LIMIT_MPS,
        "race_domain_throttle_speed_anchors": [
            {"speed_mps": speed, "throttle_norm": throttle}
            for speed, throttle in RACE_DOMAIN_THROTTLE_SPEED_ANCHORS
        ],
        "race_domain_plan_version": 4,
        "race_domain_boundary_target_mps": RACE_DOMAIN_BOUNDARY_SPEED_MPS,
    }
    duration_ns = int(sum(block.duration_s for block in plan) * 1e9)
    end = {
        "event": "phase_end",
        "profile": "race_domain_moderate_braking",
        "phase_index": 0,
        "status": "complete",
        "valid": None,
        "quality_failures": [],
    }
    finish = {
        "event": "experiment_end",
        "profile": "race_domain_moderate_braking",
        "phase_count": 1,
        "aborted": False,
        "reason": "schedule complete",
        "quality_failures": [],
    }
    events = [(1_000_000_000, start)]
    events.extend((1_000_000_000 + int(index * block.duration_s * 1e9), {
        "event": "race_domain_block_start",
        "profile": "race_domain_moderate_braking",
        "seed": seed,
        "phase_index": 0,
        "block_index": index,
        "label": block.label,
        "target_speed_mps": block.target_speed_mps,
        "steering_command_rad": block.steering_rad,
        "measured_speed_mps": block.target_speed_mps,
    }) for index, block in enumerate(plan))
    events.extend(((1_000_000_000 + duration_ns, end),
                   (1_010_000_000 + duration_ns, finish)))
    return events


class PrepareDatasetRaceCaptureTest(unittest.TestCase):
    def test_admits_only_exact_complete_seeded_profile(self):
        result = _validate_race_domain_capture_events(_complete_events())
        self.assertTrue(result["admitted"], result)
        self.assertEqual(result["planned_blocks"], 65)
        self.assertEqual(result["planned_duration_s"], 260.0)

    def test_admits_new_boundary_plan_without_reinterpreting_legacy_capture(self):
        events = _complete_events(
            boundary_speed_mps=RACE_DOMAIN_BOUNDARY_SPEED_MPS, version=2)
        result = _validate_race_domain_capture_events(events)
        self.assertTrue(result["admitted"], result)
        self.assertEqual(result["plan_version"], 2)
        self.assertEqual(result["boundary_target_mps"], 11.1)

    def test_rejects_truncated_plan(self):
        events = _complete_events()
        events[0][1]["race_domain_command_plan"].pop()
        result = _validate_race_domain_capture_events(events)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "command_plan_mismatch_or_wrong_length")

    def test_rejects_aborted_run_even_with_full_plan(self):
        events = _complete_events()
        events[-1][1]["aborted"] = True
        result = _validate_race_domain_capture_events(events)
        self.assertFalse(result["admitted"])
        self.assertEqual(
            result["reason"], "phase_or_experiment_did_not_complete_cleanly")

    def test_rejects_plan_that_does_not_match_seed(self):
        events = _complete_events()
        events[0][1]["race_domain_command_plan"][0]["target_speed_mps"] = 11.0
        result = _validate_race_domain_capture_events(events)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "command_plan_mismatch_or_wrong_length")

    def test_admits_exact_combined_braking_boundary_profile(self):
        result = _validate_race_domain_capture_events(_complete_boundary_events())
        self.assertTrue(result["admitted"], result)
        self.assertEqual(result["profile"], "race_domain_brake_boundary")
        self.assertEqual(result["plan_version"], 3)
        self.assertEqual(result["planned_blocks"], 51)
        self.assertEqual(result["planned_duration_s"], 204.0)

    def test_rejects_combined_profile_with_missing_block_feedback_event(self):
        events = _complete_boundary_events()
        events.pop(2)
        result = _validate_race_domain_capture_events(events)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "boundary_block_event_count_mismatch")

    def test_admits_moderate_steering_braking_and_release_plan_exactly(self):
        result = _validate_race_domain_capture_events(
            _complete_moderate_braking_events())
        self.assertTrue(result["admitted"], result)
        self.assertEqual(result["profile"], "race_domain_moderate_braking")
        self.assertEqual(result["plan_version"], 4)
        self.assertEqual(result["planned_blocks"], 64)
        self.assertEqual(result["planned_duration_s"], 384.0)

    def test_rejects_moderate_plan_if_any_block_is_missing(self):
        events = _complete_moderate_braking_events()
        events.pop(3)
        result = _validate_race_domain_capture_events(events)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reason"], "boundary_block_event_count_mismatch")


if __name__ == "__main__":
    unittest.main()
