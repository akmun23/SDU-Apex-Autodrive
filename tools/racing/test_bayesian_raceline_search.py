#!/usr/bin/env python3
"""Focused checks for measured-run ingestion and Bayesian trial selection."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.racing.bayesian_raceline_search import (
    FIELDS, propose_scale, record_observation,
)


def _row(scale: float, objective: float, feasible: bool = True) -> dict[str, str]:
    return {"scale": str(scale), "objective_s": str(objective),
            "feasible": str(int(feasible))}


class BayesianRacelineSearchTest(unittest.TestCase):
    def test_initial_design_and_expected_improvement_are_new_trials(self):
        first = propose_scale([_row(1.0, 5.76)])
        self.assertEqual(first["method"], "initial_space_filling")
        self.assertAlmostEqual(first["scale"], 1.075)

        boundary = propose_scale([_row(1.0, 5.76), _row(1.075, 10.0, False)])
        self.assertEqual(boundary["method"], "feasibility_boundary_refinement")
        self.assertAlmostEqual(boundary["scale"], 1.0375)

        rows = [_row(1.0, 5.76), _row(1.15, 5.9),
                _row(1.075, 5.70), _row(1.035, 5.72)]
        proposal = propose_scale(rows)
        self.assertEqual(proposal["method"], "expected_improvement")
        self.assertGreaterEqual(proposal["scale"], 0.98)
        self.assertLessEqual(proposal["scale"], 1.15)
        self.assertTrue(all(abs(proposal["scale"] - float(r["scale"])) >= 0.004
                            for r in rows))

    def test_records_only_complete_collision_free_screens_as_feasible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run-a"
            analysis = run / "analysis/race_run"
            analysis.mkdir(parents=True)
            signature = {
                "run_id": "run-a", "controller_image_id": "controller",
                "simulator_image_digest": "sim", "mpc_config_sha256": "mpc",
                "mpc_overlay_sha256": None, "optimizer_config_sha256": "optimizer",
                "centerline_sha256": "centerline",
                "resolved_optimizer_config_sha256": None,
            }
            resolved = {
                "config": {"lateral_envelope": {"scale": 1.0},
                           "yaw_phase_response": {"enabled": False}},
                "vehicle_model": {"yaw_tau_s": 0.1},
                "lateral_envelope": {"scale": 1.0, "ay_max_mps2": [[8.0]]},
            }
            resolved_path = run / "resolved_config.yaml"
            resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
            resolved_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            signature["resolved_optimizer_config_sha256"] = resolved_hash
            (run / "run_manifest.json").write_text(
                json.dumps(signature), encoding="utf-8")
            (run / "vehicle_contract.json").write_text(json.dumps({
                "resolved_optimizer_config": {
                    "path": "resolved_config.yaml", "sha256": resolved_hash,
                },
            }), encoding="utf-8")
            (analysis / "summary.json").write_text(json.dumps({
                "collision_delta": 0,
                "scored_lap_metrics": {
                    "absolute_tracking_error_m": {"p95": 0.08},
                    "minimum_wall_clearance_m": 0.22,
                },
            }), encoding="utf-8")
            with (analysis / "lap_times.csv").open("w", encoding="utf-8",
                                                   newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "lap_role", "complete", "lap_time_s"))
                writer.writeheader()
                writer.writerows([
                    {"lap_role": "warmup", "complete": "True", "lap_time_s": "26"},
                    {"lap_role": "scored", "complete": "True", "lap_time_s": "5.7"},
                    {"lap_role": "scored", "complete": "True", "lap_time_s": "5.8"},
                ])
            history = root / "observations.csv"
            row = record_observation(history, run, 1.0, 0.05)
            self.assertEqual(tuple(_ for _ in row), FIELDS)
            self.assertEqual(row["feasible"], 1)
            self.assertAlmostEqual(float(row["objective_s"]), 5.75)
            self.assertEqual(len(row["model_signature_sha256"]), 64)

    def test_rejects_campaign_with_different_effective_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "observations.csv"
            first = self._make_run(root, "run-a", phase_enabled=False)
            second = self._make_run(root, "run-b", phase_enabled=True)
            record_observation(history, first, 1.0, 0.05)
            with self.assertRaisesRegex(ValueError, "lineage differs"):
                record_observation(history, second, 1.01, 0.05)

    @staticmethod
    def _make_run(root: Path, run_id: str, phase_enabled: bool) -> Path:
        run = root / run_id
        analysis = run / "analysis/race_run"
        analysis.mkdir(parents=True)
        resolved = {
            "config": {"lateral_envelope": {"scale": 1.0},
                       "yaw_phase_response": {"enabled": phase_enabled}},
            "vehicle_model": {"yaw_tau_s": 0.1},
            "lateral_envelope": {"scale": 1.0, "ay_max_mps2": [[8.0]]},
        }
        resolved_path = run / "resolved_config.yaml"
        resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
        resolved_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
        manifest = {
            "run_id": run_id, "controller_image_id": "controller",
            "simulator_image_digest": "sim", "mpc_config_sha256": "mpc",
            "mpc_overlay_sha256": None, "optimizer_config_sha256": "optimizer",
            "centerline_sha256": "centerline",
            "resolved_optimizer_config_sha256": resolved_hash,
        }
        contract = {"resolved_optimizer_config": {
            "path": resolved_path.name, "sha256": resolved_hash,
        }}
        (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "vehicle_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        (analysis / "summary.json").write_text(json.dumps({
            "collision_delta": 0,
            "scored_lap_metrics": {
                "absolute_tracking_error_m": {"p95": 0.08},
                "minimum_wall_clearance_m": 0.22,
            },
        }), encoding="utf-8")
        with (analysis / "lap_times.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("lap_role", "complete", "lap_time_s"))
            writer.writeheader()
            writer.writerows([
                {"lap_role": "scored", "complete": "True", "lap_time_s": "5.7"},
                {"lap_role": "scored", "complete": "True", "lap_time_s": "5.8"},
            ])
        return run


if __name__ == "__main__":
    unittest.main()
