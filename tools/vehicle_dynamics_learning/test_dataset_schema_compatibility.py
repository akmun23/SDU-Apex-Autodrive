from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


class DatasetSchemaCompatibilityTest(unittest.TestCase):
    def test_native_schema7_export_does_not_require_schema8_condition_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema7.npz"
            frames = np.zeros((4, 9), dtype=np.float32)
            rigid = np.zeros((4, 13), dtype=np.float32)
            rigid[:, 3] = 1.0
            np.savez_compressed(
                path,
                schema_version=np.asarray([7], dtype=np.int32),
                feature_names=np.asarray((
                    "u_rear_mps", "v_rear_mps", "yaw_rate_rps",
                    "steering_feedback_rad", "throttle_feedback_norm",
                    "rear_left_surface_mps", "rear_right_surface_mps",
                    "steering_command_rad", "throttle_command_norm")),
                sensor_feature_names=np.asarray((
                    "steering_feedback_rad", "throttle_feedback_norm",
                    "rear_left_surface_mps", "rear_right_surface_mps",
                    "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps",
                    "steering_command_rad", "throttle_command_norm",
                    "sample_dt_s")),
                attitude_feature_names=np.asarray((
                    "imu_roll_rad", "imu_pitch_rad", "imu_roll_rate_rps",
                    "imu_pitch_rate_rps")),
                predicted_state_names=np.asarray((
                    "u_rear_mps", "v_rear_mps", "yaw_rate_rps",
                    "steering_feedback_rad", "throttle_feedback_norm",
                    "rear_left_surface_mps", "rear_right_surface_mps")),
                frames=frames,
                sensor_frames=np.zeros((4, 10), dtype=np.float32),
                sensor_valid=np.ones(4, dtype=bool),
                imu_attitude_frames=np.zeros((4, 4), dtype=np.float32),
                imu_attitude_valid=np.ones(4, dtype=bool),
                dt_s=np.full(4, 0.025, dtype=np.float32),
                packet_sequence=np.arange(4, dtype=np.int64),
                sample_time_ns=np.arange(4, dtype=np.int64) * 25_000_000,
                odom_pose_xyyaw=np.zeros((4, 3), dtype=np.float32),
                simulator_pose_xyyaw=np.zeros((4, 3), dtype=np.float32),
                lap_count=np.zeros(4, dtype=np.int32),
                simulator_rigid_state=rigid,
                simulator_linear_acceleration=np.zeros((4, 3), dtype=np.float32),
                sequence_bounds=np.asarray([[0, 4]], dtype=np.int64),
                sequence_run_index=np.asarray([0], dtype=np.int32),
                sequence_labels=np.asarray(["same-condition"]),
                sequence_condition_id=np.asarray([0], dtype=np.int32),
                sequence_reset_index=np.asarray([0], dtype=np.int32),
                sequence_replicate_index=np.asarray([-1], dtype=np.int32),
                frame_run_index=np.zeros(4, dtype=np.int32),
                frame_reset_index=np.zeros(4, dtype=np.int32),
                run_ids=np.asarray(["validation-run"]),
                run_families=np.asarray(["open_plane"]),
                run_splits=np.asarray(["validation"]),
                condition_labels=np.asarray(["same-condition"]),
            )

            data = _load_dataset(path)

        self.assertEqual(data["schema_version"], 7)
        self.assertEqual(data["frames"].shape, (4, 9))
        np.testing.assert_array_equal(data["sequence_condition_id"], [0])


if __name__ == "__main__":
    unittest.main()
