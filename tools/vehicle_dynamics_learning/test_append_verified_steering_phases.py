from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.vehicle_dynamics_learning.append_verified_steering_phases import append


def _base() -> dict[str, np.ndarray]:
    frames = np.zeros((4, 9), dtype=np.float32)
    rigid = np.zeros((4, 13), dtype=np.float32)
    rigid[:, 7] = 4.0
    data = {
        "schema_version": np.asarray([8], dtype=np.int32),
        "source_schema_version": np.asarray([7], dtype=np.int32),
        "dataset_role": np.asarray(["race_domain"], dtype="U32"),
        "domain_speed_cap_mps": np.asarray([12.0], dtype=np.float32),
        "domain_cooldown_steps": np.asarray([80], dtype=np.int32),
        "feature_names": np.asarray([f"f{i}" for i in range(9)], dtype="U64"),
        "sensor_feature_names": np.asarray([f"s{i}" for i in range(10)], dtype="U64"),
        "attitude_feature_names": np.asarray([f"a{i}" for i in range(4)], dtype="U64"),
        "predicted_state_names": np.asarray([f"p{i}" for i in range(7)], dtype="U64"),
        "frames": frames,
        "sensor_frames": np.zeros((4, 10), dtype=np.float32),
        "sensor_valid": np.ones(4, dtype=bool),
        "imu_attitude_frames": np.zeros((4, 4), dtype=np.float32),
        "imu_attitude_valid": np.ones(4, dtype=bool),
        "dt_s": np.full(4, 0.025, dtype=np.float32),
        "packet_sequence": np.arange(4, dtype=np.int64),
        "sample_time_ns": np.arange(4, dtype=np.int64) * 25_000_000,
        "odom_pose_xyyaw": np.zeros((4, 3), dtype=np.float32),
        "simulator_pose_xyyaw": np.zeros((4, 3), dtype=np.float32),
        "lap_count": np.zeros(4, dtype=np.int32),
        "simulator_rigid_state": rigid,
        "simulator_linear_acceleration": np.zeros((4, 3), dtype=np.float32),
        "frame_run_index": np.zeros(4, dtype=np.int32),
        "frame_reset_index": np.zeros(4, dtype=np.int32),
        "frame_source_index": np.arange(4, dtype=np.int64),
        "frame_source_sequence_index": np.zeros(4, dtype=np.int32),
        "frame_domain_speed_mps": np.full(4, 4.0, dtype=np.float32),
        "sequence_bounds": np.asarray([[0, 4]], dtype=np.int64),
        "sequence_run_index": np.asarray([0], dtype=np.int32),
        "sequence_labels": np.asarray(["base_phase"], dtype="U256"),
        "sequence_condition_id": np.asarray([0], dtype=np.int32),
        "sequence_reset_index": np.asarray([0], dtype=np.int32),
        "sequence_replicate_index": np.asarray([-1], dtype=np.int32),
        "sequence_source_condition_id": np.asarray([0], dtype=np.int32),
        "sequence_source_index": np.asarray([0], dtype=np.int32),
        "run_ids": np.asarray(["base_run"], dtype="U128"),
        "run_families": np.asarray(["open_plane"], dtype="U32"),
        "run_splits": np.asarray(["train"], dtype="U32"),
        "training_families": np.asarray(["full_input_openplane"], dtype="U40"),
        "condition_labels": np.asarray(["base_phase"], dtype="U256"),
        "condition_run_index": np.asarray([0], dtype=np.int32),
        "training_family_names": np.asarray(["steering_transition_slew"], dtype="U40"),
        "training_family_probabilities": np.asarray([1.0], dtype=np.float32),
    }
    return data


def _salvage() -> dict[str, np.ndarray]:
    frames = np.zeros((12, 9), dtype=np.float32)
    frames[:, 0] = 6.0
    frames[:, 3] = 0.3
    rigid = np.zeros((12, 13), dtype=np.float32)
    rigid[:, 7] = 6.0
    return {
        "schema_version": np.asarray([7], dtype=np.int32),
        "feature_names": np.asarray([f"f{i}" for i in range(9)], dtype="U64"),
        "sensor_feature_names": np.asarray([f"s{i}" for i in range(10)], dtype="U64"),
        "attitude_feature_names": np.asarray([f"a{i}" for i in range(4)], dtype="U64"),
        "predicted_state_names": np.asarray([f"p{i}" for i in range(7)], dtype="U64"),
        "frames": frames,
        "sensor_frames": np.zeros((12, 10), dtype=np.float32),
        "sensor_valid": np.ones(12, dtype=bool),
        "imu_attitude_frames": np.zeros((12, 4), dtype=np.float32),
        "imu_attitude_valid": np.ones(12, dtype=bool),
        "dt_s": np.full(12, 0.025, dtype=np.float32),
        "packet_sequence": np.arange(10, 22, dtype=np.int64),
        "sample_time_ns": np.arange(12, dtype=np.int64) * 25_000_000,
        "odom_pose_xyyaw": np.zeros((12, 3), dtype=np.float32),
        "simulator_pose_xyyaw": np.zeros((12, 3), dtype=np.float32),
        "lap_count": np.zeros(12, dtype=np.int32),
        "simulator_rigid_state": rigid,
        "simulator_linear_acceleration": np.zeros((12, 3), dtype=np.float32),
        "sequence_bounds": np.asarray([[0, 6], [6, 12]], dtype=np.int64),
        "sequence_run_index": np.asarray([0, 0], dtype=np.int32),
        "sequence_labels": np.asarray([
            "isolated_r2_steer_+0.3_hold",
            "isolated_r3_steer_+0.4_hold"], dtype="U256"),
        "sequence_splits": np.asarray(["train", "train"], dtype="U16"),
        "run_ids": np.asarray(["openplane_isolated_highspeed_surface_20260927"], dtype="U128"),
        "run_splits": np.asarray(["train"], dtype="U16"),
    }


class AppendVerifiedSteeringPhasesTest(unittest.TestCase):
    def test_append_preserves_prefix_and_assigns_train_condition_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, salvage = _base(), _salvage()
            base_path, salvage_path = root / "base.npz", root / "verified.npz"
            np.savez_compressed(base_path, **base)
            np.savez_compressed(salvage_path, **salvage)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({
                "export_by_quality_tier": {
                    "verified": {"file": "verified.npz"}},
                "runs": [{
                    "run_id": str(salvage["run_ids"][0]),
                    "source_split": "train",
                    "phases": [{
                        "label": str(salvage["sequence_labels"][0]),
                        "status": "salvaged",
                        "quality_tier": "verified",
                        "reasons": [],
                    }, {
                        "label": str(salvage["sequence_labels"][1]),
                        "status": "salvaged",
                        "quality_tier": "verified",
                        "reasons": [],
                    }],
                }],
            }))
            output = root / "out"
            result = append(base_path, salvage_path, audit_path, output,
                            [str(salvage["run_ids"][0])])
            with np.load(output / "openplane_dynamics.npz", allow_pickle=False) as z:
                for key, value in base.items():
                    if key in ("schema_version", "source_schema_version",
                               "dataset_role", "domain_speed_cap_mps",
                               "domain_cooldown_steps", "feature_names",
                               "sensor_feature_names", "attitude_feature_names",
                               "predicted_state_names", "training_family_names",
                               "training_family_probabilities"):
                        np.testing.assert_array_equal(z[key], value)
                    elif key in {"sequence_bounds", "sequence_run_index",
                                 "sequence_labels", "sequence_condition_id",
                                 "sequence_reset_index", "sequence_replicate_index",
                                 "sequence_source_condition_id", "sequence_source_index"}:
                        np.testing.assert_array_equal(z[key][:len(value)], value)
                    elif key in {"run_ids", "run_families", "run_splits",
                                 "training_families"}:
                        np.testing.assert_array_equal(z[key][:len(value)], value)
                    elif key in {"condition_labels", "condition_run_index"}:
                        np.testing.assert_array_equal(z[key][:len(value)], value)
                    elif value.ndim and value.shape[0] == len(base["frames"]):
                        np.testing.assert_array_equal(z[key][:len(value)], value)
                np.testing.assert_array_equal(z["run_splits"], ["train", "train"])
                np.testing.assert_array_equal(z["training_families"],
                                              ["full_input_openplane",
                                               "steering_transition_slew"])
                np.testing.assert_array_equal(z["frame_run_index"][4:], 1)
                np.testing.assert_array_equal(z["sequence_bounds"][-2:],
                                              [[4, 10], [10, 16]])
                np.testing.assert_array_equal(z["sequence_condition_id"], [0, 1, 2])
                np.testing.assert_array_equal(z["sequence_reset_index"], [0, 0, 1])
                np.testing.assert_array_equal(z["sequence_replicate_index"], [-1, 2, 3])
                np.testing.assert_array_equal(z["condition_run_index"], [0, 1, 1])
            self.assertEqual(result["added_sequences"], 2)
            self.assertEqual(result["added_samples"], 12)

    def test_rejects_unverified_or_non_train_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, salvage = _base(), _salvage()
            base_path, salvage_path = root / "base.npz", root / "verified.npz"
            np.savez_compressed(base_path, **base)
            np.savez_compressed(salvage_path, **salvage)
            audit_path = root / "audit.json"
            report = {
                "export_by_quality_tier": {"verified": {"file": "verified.npz"}},
                "runs": [{
                    "run_id": str(salvage["run_ids"][0]), "source_split": "train",
                    "phases": [{
                        "label": str(salvage["sequence_labels"][0]),
                        "status": "salvaged", "quality_tier": "unverified",
                        "reasons": [],
                    }],
                }],
            }
            audit_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "not an audited verified"):
                append(base_path, salvage_path, audit_path, root / "out",
                       [str(salvage["run_ids"][0])])

    def test_rejects_a_packet_gap_inside_a_salvaged_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, salvage = _base(), _salvage()
            salvage["packet_sequence"][3:] += 1
            base_path, salvage_path = root / "base.npz", root / "verified.npz"
            np.savez_compressed(base_path, **base)
            np.savez_compressed(salvage_path, **salvage)
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({
                "export_by_quality_tier": {
                    "verified": {"file": "verified.npz"}},
                "runs": [{
                    "run_id": str(salvage["run_ids"][0]),
                    "source_split": "train",
                    "phases": [{
                        "label": str(salvage["sequence_labels"][0]),
                        "status": "salvaged",
                        "quality_tier": "verified",
                        "reasons": [],
                    }, {
                        "label": str(salvage["sequence_labels"][1]),
                        "status": "salvaged",
                        "quality_tier": "verified",
                        "reasons": [],
                    }],
                }],
            }))
            with self.assertRaisesRegex(ValueError, "packet gap"):
                append(base_path, salvage_path, audit_path, root / "out",
                       [str(salvage["run_ids"][0])])


if __name__ == "__main__":
    unittest.main()
