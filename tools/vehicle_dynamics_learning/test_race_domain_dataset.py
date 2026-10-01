from __future__ import annotations

import unittest

import numpy as np

from tools.vehicle_dynamics_learning.build_race_domain_dataset import (
    _family_for_run,
    _practice_active_interval_passed,
    _quality_run_ids,
    _safe_domain_mask,
    _view_arrays,
)


class RaceDomainDatasetTest(unittest.TestCase):
    def test_speed_cap_and_post_excursion_cooldown_are_sample_exact(self):
        speed = np.asarray([1.0, 11.9, 12.0, 12.01, 12.2,
                            12.0, 11.0, 10.0, 9.0])
        valid = np.ones(len(speed), dtype=bool)
        actual = _safe_domain_mask(speed, valid, cooldown_steps=2)
        expected = np.asarray([True, True, True, False, False,
                               False, False, True, True])
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_labels_break_sequence_without_consuming_speed_cooldown(self):
        speed = np.asarray([8.0, 12.5, 10.0, 10.0, 10.0, 10.0])
        valid = np.asarray([True, True, False, True, True, True])
        actual = _safe_domain_mask(speed, valid, cooldown_steps=2)
        expected = np.asarray([True, False, False, False, False, True])
        np.testing.assert_array_equal(actual, expected)

    def test_run_family_uses_race_and_acquisition_protocol(self):
        self.assertEqual(_family_for_run("practice_mpc_run", "practice_track"),
                         "practice_race")
        self.assertEqual(_family_for_run("openplane_throttle_slew_r01", "open_plane"),
                         "steering_transition_slew")
        self.assertEqual(_family_for_run("openplane_rootless_40_surface", "open_plane"),
                         "throttle_surface")
        self.assertEqual(_family_for_run("openplane_full_input_train1", "open_plane"),
                         "full_input_openplane")

    def test_practice_post_lap12_disconnect_uses_explicit_active_interval_gate(self):
        row = {
            "quality_gate_scope": "complete_lap_0_to_12_active_interval",
            "practice_active_interval_validation": {
                "lap_count_transitions": list(range(13)),
                "collision_min_max": [0, 0],
                "timing_faults_during_active_interval": 0,
                "stream_cadence": {"/odom": {"pass": True}},
            },
        }
        self.assertTrue(_practice_active_interval_passed(row))
        row["practice_active_interval_validation"]["stream_cadence"][
            "/odom"]["pass"] = False
        self.assertFalse(_practice_active_interval_passed(row))

    def test_unscored_phases_require_explicit_completed_capture_admission(self):
        row = {
            "run_id": "openplane_race_domain_train",
            "effective_split": "train",
            "clean_stream_and_collision_gate": True,
            "aborted": False,
            "timing_faults": 0,
            "collisions": [0, 0],
            "whole_bag_quality_failures": [],
            "quality_failures": [],
            "unscored_phases": 1,
        }
        accepted, rejected = _quality_run_ids({"runs": [row]})
        self.assertFalse(accepted)
        self.assertIn("unscored_phase_not_admitted_by_capture_protocol",
                      rejected[row["run_id"]])
        row["unscored_race_domain_capture_admission"] = {"admitted": True}
        accepted, rejected = _quality_run_ids({"runs": [row]})
        self.assertEqual(accepted, {row["run_id"]})
        self.assertFalse(rejected)

    def test_valid_phases_survive_unscored_tail_phases(self):
        row = {
            "run_id": "openplane_valid_transition_with_unscored_tail",
            "effective_split": "train",
            "clean_stream_and_collision_gate": True,
            "aborted": False,
            "timing_faults": 0,
            "collisions": [0, 0],
            "whole_bag_quality_failures": [],
            "quality_failures": [],
            "valid_phases": 126,
            "unscored_phases": 8,
        }
        accepted, rejected = _quality_run_ids({"runs": [row]})
        self.assertEqual(accepted, {row["run_id"]})
        self.assertFalse(rejected)

    def test_view_rebuilds_run_local_condition_catalog_without_source_mapping(self):
        source = {
            "frames": np.arange(8, dtype=np.float32).reshape(4, 2),
            "sequence_bounds": np.asarray([[0, 2], [2, 4]], dtype=np.int64),
            "sequence_run_index": np.asarray([0, 1], dtype=np.int32),
            "sequence_labels": np.asarray(["phase", "phase"]),
            "sequence_condition_id": np.asarray([0, 0], dtype=np.int32),
            "sequence_reset_index": np.asarray([0, 0], dtype=np.int32),
            "sequence_replicate_index": np.asarray([0, 0], dtype=np.int32),
            "condition_labels": np.asarray(["shared-condition"]),
        }
        view = _view_arrays(source, [(0, 0, 2), (1, 2, 4)], 4,
                            role="race_domain")
        np.testing.assert_array_equal(view["sequence_condition_id"], [0, 1])
        np.testing.assert_array_equal(view["condition_run_index"], [0, 1])
        np.testing.assert_array_equal(view["sequence_source_condition_id"],
                                      [0, 0])
        np.testing.assert_array_equal(view["condition_labels"],
                                      ["shared-condition", "shared-condition"])


if __name__ == "__main__":
    unittest.main()
