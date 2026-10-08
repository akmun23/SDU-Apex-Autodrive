import unittest

import numpy as np

from tools.racing.specialists.train_yaw_multihorizon_teacher import (
    CaptureArrays,
    _build_examples,
    _split_safe_mixed_train_validation,
)


class DirectYawTeacherFeatureTests(unittest.TestCase):
    def test_mixed_archive_gate_never_admits_test_partitions(self):
        self.assertTrue(_split_safe_mixed_train_validation({"train", "validation"}))
        self.assertFalse(_split_safe_mixed_train_validation({"train", "test"}))
        self.assertFalse(_split_safe_mixed_train_validation(
            {"train", "validation", "final_test"}))
        self.assertFalse(_split_safe_mixed_train_validation({"train"}))

    def make_capture(self):
        count = 160
        frames = np.zeros((count, 9), dtype=np.float32)
        frames[:, 0] = np.arange(count) * 0.01
        frames[:, 2] = np.arange(count) * 0.01
        frames[:, 3] = np.arange(count) * 0.001
        frames[:, 4] = 0.2
        frames[:, 5] = 2.0
        frames[:, 6] = 2.1
        frames[:, 7] = np.arange(count) * 0.002
        frames[:, 8] = 0.3
        sensors = np.zeros((count, 10), dtype=np.float32)
        sensors[:, 0] = frames[:, 3]
        sensors[:, 1] = frames[:, 4]
        sensors[:, 2:4] = frames[:, 5:7]
        sensors[:, 6] = np.arange(count) * 0.005
        sensors[:, 7:9] = frames[:, 7:9]
        attitude = np.zeros((count, 4), dtype=np.float32)
        attitude[:, 0] = np.arange(count) * 0.0001
        attitude[:, 2] = np.arange(count) * 0.0002
        rigid = np.zeros((count, 13), dtype=np.float32)
        rigid[:, 12] = np.arange(count) * 0.01
        return CaptureArrays(
            "synthetic", "train", sensors,
            np.ones(count, dtype=bool), attitude,
            np.ones(count, dtype=bool), rigid,
            np.asarray([[0, 80], [80, count]], dtype=np.int64), "synthetic.npz",
        )

    def test_features_use_only_planned_commands_through_horizon(self):
        capture = self.make_capture()
        x_before, y_before, _ = _build_examples(capture, 4, (0, 1, 2))
        changed = self.make_capture()
        changed.sensors[12:, 7:9] += 5.0
        changed.sensors[9:12, :7] += 500.0
        x_after, _, _ = _build_examples(changed, 4, (0, 1, 2))
        # Row 6 starts at global frame 8 and predicts frame 12. Commands after
        # frame 11 are outside that prediction horizon and must be invisible.
        row_at_8 = 6
        np.testing.assert_array_equal(x_before[row_at_8], x_after[row_at_8])
        self.assertAlmostEqual(float(y_before[row_at_8]), 0.12, places=6)

    def test_future_truth_is_target_only_and_sequences_do_not_join(self):
        capture = self.make_capture()
        x_before, y_before, _ = _build_examples(capture, 4, (0, 1, 2))
        changed = self.make_capture()
        changed.rigid[6, 12] = 123.0
        x_after, y_after, _ = _build_examples(changed, 4, (0, 1, 2))
        # First returned row is k=2, so its target is frame 6. Its GT value is
        # a label only and cannot alter any feature.
        np.testing.assert_array_equal(x_before[0], x_after[0])
        self.assertNotEqual(y_before[0], y_after[0])
        # First run contributes rows k=2..75; second starts at k=82. Its
        # history must start at 82, not borrow samples from the prior sequence.
        self.assertEqual(len(x_before), 74 + 74)
        self.assertAlmostEqual(float(x_before[74, 0]), 0.082, places=5)


if __name__ == "__main__":
    unittest.main()
