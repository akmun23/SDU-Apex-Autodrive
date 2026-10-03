"""Hash-lineage checks for the frozen reflection-projection benchmark."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.vehicle_dynamics_learning.score_turn_reflection_projection import (
    _high_steering_windows,
    _validate_practice_lineage,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PracticeLineageTest(unittest.TestCase):
    def test_accepts_exact_frozen_dataset_without_sidecar_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "practice.npz"
            dataset.write_bytes(b"frozen practice fixture")
            result = _validate_practice_lineage(
                {"practice_dataset_sha256": _sha256(dataset)},
                dataset, Path(directory) / "absent.json")
        self.assertEqual(result["mode"], "exact_dataset")

    def test_accepts_only_hash_linked_sidecar_derivative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_hash = "source-hash"
            dataset = root / "practice-sidecar.npz"
            dataset.write_bytes(b"derived practice fixture")
            manifest = root / "lineage.json"
            manifest.write_text(json.dumps({
                "source_dataset_sha256": source_hash,
                "output_dataset_sha256": _sha256(dataset),
                "source_dataset": "frozen-original.npz",
            }), encoding="utf-8")
            result = _validate_practice_lineage(
                {"practice_dataset_sha256": source_hash}, dataset, manifest)
            self.assertEqual(result["mode"], "hash_verified_sidecar_derivative")
            self.assertEqual(result["source_sha256"], source_hash)

            manifest.write_text(json.dumps({
                "source_dataset_sha256": "wrong-source",
                "output_dataset_sha256": _sha256(dataset),
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_practice_lineage(
                    {"practice_dataset_sha256": source_hash}, dataset, manifest)

    def test_high_steering_subset_uses_only_future_command_prefix(self):
        frames = [[0.0] * 9 for _ in range(12)]
        frames[5][7] = 0.31
        windows = [{"start": 1}, {"start": 8}]
        selected = _high_steering_windows(
            {"frames": frames}, windows, horizon_steps=4)
        self.assertEqual(selected, [windows[0]])


if __name__ == "__main__":
    unittest.main()
