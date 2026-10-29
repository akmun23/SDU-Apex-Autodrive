#!/usr/bin/env python3
"""Export saved MPC state/queue rows for deterministic offline solver replay.

This reconstructs solver inputs from bag diagnostics; it cannot restore the
live ADMM iterates or certified-plan memory that were not recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from tools import evaluate_mpc_bag as evaluator
except ModuleNotFoundError:
    import evaluate_mpc_bag as evaluator


HEADER = (
    "stamp_ns,progress_m,speed_ceiling_mps,e_y,e_psi,u,v,r,"
    "actual_steering_rad,delayed_steer_1_rad,delayed_steer_2_rad,"
    "delayed_throttle_1,delayed_throttle_2,suspension_history_initialized,"
    "yaw_residual_radps,yaw_residual_valid,wheel_omega_fl,wheel_omega_fr,"
    "wheel_omega_rl,wheel_omega_rr,body_height,body_height_rate,roll,roll_rate,"
    "pitch,pitch_rate,jounce_fl,jounce_fr,jounce_rl,jounce_rr,"
    "low_forward_fl,low_forward_fr,low_forward_rl,low_forward_rr,"
    "low_side_fl,low_side_fr,low_side_rl,low_side_rr"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_solver_row(diag: dict[str, Any]) -> list[float]:
    state = diag.get("state")
    hidden = diag.get("plant_internal_state")
    flags = diag.get("plant_state_flags")
    required = (
        isinstance(state, list) and len(state) == 10 and
        isinstance(hidden, list) and len(hidden) == 22 and
        isinstance(flags, list) and len(flags) == 3
    )
    if not required:
        raise ValueError("diagnostic lacks the complete MPC and hidden-plant state")
    stamp_ns = diag.get("source_stamp_ns")
    progress = diag.get("progress_m")
    speed_ceiling = diag.get("active_speed_ceiling_mps")
    # Diagnostic order is ey,epsi,u,v,r,queued steer 1,queued steer 2,rack,q1,q2.
    # The C replay's numeric schema puts the rack BEFORE the two queued targets.
    # Copying state verbatim silently exchanged physical rack and command history.
    ordered = [*state[:5], state[7], state[5], state[6], *state[8:10]]
    values = [stamp_ns, progress, speed_ceiling, *ordered, flags[1],
              0.0, 0.0, *hidden]
    row = [float(value) for value in values]
    if len(row) != 38:
        raise ValueError(f"internal export row has {len(row)} values, expected 38")
    if not all(math.isfinite(value) for value in row):
        raise ValueError('saved MPC state contains non-finite values')
    return row


def export_rows(rows: list[dict[str, Any]], output: Path,
                first_index: int, last_index: int) -> dict[str, Any]:
    if first_index < 0 or last_index < first_index or last_index >= len(rows):
        raise ValueError("diagnostic row range is outside the decoded bag")
    selected = rows[first_index:last_index + 1]
    exported: list[tuple[int, list[float]]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=first_index):
        try:
            exported.append((index, to_solver_row(row)))
        except (TypeError, ValueError) as error:
            skipped.append({"diagnostic_index": index, "reason": str(error)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        stream.write("# " + HEADER + "\n")
        writer = csv.writer(stream)
        writer.writerows(values for _, values in exported)
    return {
        "schema": "sdu_apex_mpc_solver_saved_state_export/v1",
        "source_kind": "saved_state_reconstruction_not_live_solver_memory",
        "diagnostic_index_first": first_index,
        "diagnostic_index_last": last_index,
        "diagnostic_rows_requested": len(selected),
        "solver_rows_exported": len(exported),
        "solver_rows_skipped": skipped,
        "output_path": str(output),
        "output_sha256": sha256(output),
        "yaw_residual_note": (
            "columns retained for the established input schema; current replay "
            "ignores them and they are exported as zero"),
    }


def controls_to_rows(diag: dict[str, Any], source: str) -> list[list[float]]:
    """Return one saved 30-stage MPC control sequence without reconstruction."""
    if source not in ("candidate", "nominal"):
        raise ValueError("control source must be candidate or nominal")
    name = f"{source}_controls"
    controls = diag.get(name)
    if not isinstance(controls, list) or len(controls) != 30:
        raise ValueError(f"diagnostic does not contain 30 {name}")
    rows = []
    for stage, control in enumerate(controls, start=1):
        if not isinstance(control, list) or len(control) != 2:
            raise ValueError(f"{name} stage {stage} is not a steering/throttle pair")
        values = [float(value) for value in control]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} stage {stage} contains a non-finite value")
        rows.append(values)
    return rows


def export_control_sequence(rows: list[dict[str, Any]], output: Path,
                            diagnostic_index: int, source: str) -> dict[str, Any]:
    """Export the exact controls in one diagnostic row as a source-search seed."""
    if not 0 <= diagnostic_index < len(rows):
        raise ValueError("diagnostic index is outside the decoded bag")
    diag = rows[diagnostic_index]
    controls = controls_to_rows(diag, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("stage", "command_steering_rad", "command_throttle"))
        writer.writerows((stage, *control)
                         for stage, control in enumerate(controls, start=1))
    return {
        "schema": "sdu_apex_mpc_control_sequence_export/v1",
        "source_kind": "exact_saved_mpc_diagnostic_sequence",
        "diagnostic_index": diagnostic_index,
        "source": source,
        "status": diag.get("status"),
        "progress_m": diag.get("progress_m"),
        "source_stamp_ns": diag.get("source_stamp_ns"),
        "issued_command_stamp_ns": diag.get("issued_command_stamp_ns"),
        "command_published": diag.get("command_published"),
        "control_count": len(controls),
        "output_path": str(output),
        "output_sha256": sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--controls-output", type=Path,
                        help="optional CSV destination for one exact saved control sequence")
    parser.add_argument("--controls-index", type=int,
                        help="zero-based diagnostic index for --controls-output")
    parser.add_argument("--controls-source", choices=("candidate", "nominal"),
                        default="candidate")
    args = parser.parse_args()
    if not args.bag_run.is_dir():
        parser.error("bag run directory does not exist")
    try:
        bag = evaluator.read_bag(args.bag_run)
        rows = [dict(record.get("json", {}), bag_time=record.get("bag_time"))
                for record in bag.get(evaluator.DIAGNOSTICS, [])]
        last_index = len(rows) - 1 if args.end_index is None else args.end_index
        report = export_rows(rows, args.output, args.start_index, last_index)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    db3 = next(args.bag_run.glob("*.db3"), None)
    report["source_bag_path"] = str(args.bag_run)
    report["source_bag_sha256"] = sha256(db3) if db3 else None
    if args.controls_output is not None:
        if args.controls_index is None:
            parser.error("--controls-output requires --controls-index")
        controls_report = export_control_sequence(
            rows, args.controls_output, args.controls_index,
            args.controls_source)
        controls_report["source_bag_path"] = str(args.bag_run)
        controls_report["source_bag_sha256"] = report["source_bag_sha256"]
        controls_report_path = args.controls_output.with_suffix(
            args.controls_output.suffix + ".json")
        controls_report_path.write_text(
            json.dumps(controls_report, indent=2, sort_keys=True) + "\n")
        report["control_sequence_export"] = controls_report
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
