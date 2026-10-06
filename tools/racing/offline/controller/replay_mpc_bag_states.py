#!/usr/bin/env python3
"""Counterfactual production-MPC replay over recorded, pre-impact controller states."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from collections import Counter
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tools.racing.offline.controller.production_mpc import ProductionMpc


ROOT = Path(__file__).resolve().parents[4]
MPC_TOPIC = "/mpc/diagnostics"
COLLISION_TOPIC = "/autodrive/roboracer_1/collision_count"
STATUS_NAMES = {
    0: "accepted_optimal", 1: "accepted_degraded", 2: "rejected_input",
    3: "rejected_solver", 4: "rejected_residual",
    5: "rejected_regularization", 6: "rejected_nonlinear_rollout",
}
FAILURE_NAMES = {
    0: "none", 1: "invalid_input", 2: "invalid_model",
    3: "command_limit", 4: "state_limit", 5: "corridor",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_inputs(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int | None]:
    rows: list[tuple[int, dict[str, Any]]] = []
    first_collision_ns = None
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        topics = {name: (int(topic_id), kind) for topic_id, name, kind in
                  db.execute("SELECT id,name,type FROM topics")}
        if MPC_TOPIC not in topics or COLLISION_TOPIC not in topics:
            raise ValueError("bag must contain MPC diagnostics and collision count")
        topic_id, type_name = topics[MPC_TOPIC]
        message_type = get_message(type_name)
        for stamp_ns, raw in db.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp,id", (topic_id,)):
            message = deserialize_message(bytes(raw), message_type)
            try:
                payload = json.loads(message.data)
            except (AttributeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and len(payload.get("state", [])) == 12:
                rows.append((int(stamp_ns), payload))

        collision_id, collision_type = topics[COLLISION_TOPIC]
        collision_message_type = get_message(collision_type)
        previous = 0
        for stamp_ns, raw in db.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp,id", (collision_id,)):
            message = deserialize_message(bytes(raw), collision_message_type)
            count = int(message.data)
            if count > previous:
                first_collision_ns = int(stamp_ns)
                break
            previous = count
    if not rows:
        raise ValueError("bag contains no MPC diagnostic rows with a 12-channel state")
    if first_collision_ns is not None:
        rows = [(stamp, row) for stamp, row in rows
                if stamp < first_collision_ns]
    return rows, first_collision_ns


def _replay(name: str, library: Path, config: Path, trajectory: Path,
            rows: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    first_rejection = None
    rejection_samples = []
    region = []
    action_deltas = []
    projection_distances = []
    previous_segment = 2**64 - 1
    previous_wrapped_progress: float | None = None
    progress_lap_offset = 0.0
    with ProductionMpc(library, config, trajectory) as controller:
        for stamp_ns, record in rows:
            state = np.asarray(record["state"], dtype=np.float64)
            # The logged Frenet errors/progress belong to the trajectory that
            # was active during recording. Reusing them against a different
            # candidate silently evaluates the wrong counterfactual. Reproject
            # the captured control-time map pose onto this replay trajectory.
            prediction = record.get("control_time_prediction", {})
            pose = np.asarray(
                prediction.get("predicted_map_pose", record.get("predicted_map_pose", [])),
                dtype=np.float64)
            if (not np.isfinite(state).all() or pose.shape != (3,) or
                    not np.isfinite(pose).all()):
                continue
            projected_s, e_y, e_psi, distance, segment = controller.project(
                pose, previous_segment=previous_segment,
                local_search_radius=160)
            previous_segment = segment
            if previous_wrapped_progress is not None:
                progress_delta = projected_s - previous_wrapped_progress
                if progress_delta < -0.5 * controller.lap_length_m:
                    progress_lap_offset += controller.lap_length_m
                elif progress_delta > 0.5 * controller.lap_length_m:
                    progress_lap_offset -= controller.lap_length_m
            previous_wrapped_progress = projected_s
            progress = projected_s + progress_lap_offset
            state[0] = e_y
            state[1] = e_psi
            projection_distances.append(distance)
            cycle = controller.step(state, progress, 16.0)
            status = STATUS_NAMES.get(cycle.status, f"unknown_{cycle.status}")
            failure = FAILURE_NAMES.get(
                cycle.nonlinear_failure_reason,
                f"unknown_{cycle.nonlinear_failure_reason}")
            statuses[status] += 1
            if failure != "none":
                failures[failure] += 1
            if cycle.status >= 2 and first_rejection is None:
                first_rejection = {
                    "bag_receipt_ns": stamp_ns,
                    "progress_m": progress,
                    "projection_distance_m": distance,
                    "reprojected_e_y_m": e_y,
                    "reprojected_e_psi_rad": e_psi,
                    "source_status": record.get("status"),
                    "replayed_status": status,
                    "reason": failure,
                    "failure_stage": cycle.nonlinear_failure_stage,
                    "speed_mps": float(state[2]),
                    "steering_rad": float(state[9]),
                    "q_speed_tan_steer": float(
                        max(state[2], 0.0) * abs(np.tan(state[9]))),
                    "r1_failure": asdict(cycle.r1_failure),
                    "r2_failure": asdict(cycle.r2_failure),
                }
            if cycle.status >= 2 and len(rejection_samples) < 32:
                rejection_samples.append({
                    "progress_m": progress,
                    "speed_mps": float(state[2]),
                    "steering_rad": float(state[9]),
                    "status": status,
                    "reason": failure,
                    "r1_failure": asdict(cycle.r1_failure),
                    "r2_failure": asdict(cycle.r2_failure),
                })
            if 210.0 <= progress <= 225.0:
                region.append({
                    "bag_receipt_ns": stamp_ns,
                    "progress_m": progress,
                    "projection_distance_m": distance,
                    "speed_mps": float(state[2]),
                    "steering_rad": float(state[9]),
                    "q_speed_tan_steer": float(
                        max(state[2], 0.0) * abs(np.tan(state[9]))),
                    "status": status,
                    "reason": failure,
                    "failure_stage": cycle.nonlinear_failure_stage,
                    "first_steering_command_rad": cycle.steering_command_rad,
                    "first_target_speed_mps": cycle.target_speed_mps,
                })
    return {
        "name": name,
        "config": str(config.relative_to(ROOT)),
        "config_sha256": _sha256(config),
        "cycle_count": sum(statuses.values()),
        "status_counts": dict(statuses),
        "nonlinear_failure_reason_counts": dict(failures),
        "first_rejection": first_rejection,
        "initial_rejection_failure_details": rejection_samples,
        "map_pose_projection_distance_m": {
            "samples": len(projection_distances),
            "p95": float(np.quantile(projection_distances, 0.95))
            if projection_distances else None,
            "max": float(max(projection_distances))
            if projection_distances else None,
        },
        "replay_rows_near_recorded_collision_corner": region,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("ungated_config", type=Path)
    parser.add_argument("gated_config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bag = args.bag.resolve()
    library = args.library.resolve()
    trajectory = args.trajectory.resolve()
    rows, collision_ns = _read_inputs(bag)
    variants = [
        _replay("recorded_surface_no_support_gate", library,
                args.ungated_config.resolve(), trajectory, rows),
        _replay("evidence_supported_surface", library,
                args.gated_config.resolve(), trajectory, rows),
    ]
    report = {
        "purpose": "frozen control-time map poses reprojected onto the replay trajectory before same-state MPC comparison; diagnostic only, not closed-loop validation",
        "bag": str(bag.relative_to(ROOT)),
        "bag_sha256": _sha256(bag),
        "trajectory": str(trajectory.relative_to(ROOT)),
        "trajectory_sha256": _sha256(trajectory),
        "first_collision_receipt_ns": collision_ns,
        "preimpact_input_rows": len(rows),
        "variants": variants,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(output.relative_to(ROOT)),
        "preimpact_input_rows": len(rows),
        "first_collision_receipt_ns": collision_ns,
        "variants": [{
            "name": row["name"],
            "status_counts": row["status_counts"],
            "failure_counts": row["nonlinear_failure_reason_counts"],
            "first_rejection": row["first_rejection"],
        } for row in variants],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
