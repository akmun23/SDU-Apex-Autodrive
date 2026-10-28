#!/usr/bin/env python3
"""Capture, seal, validate, and replay exact frozen MPC QP fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time

MAGIC = b"SDUFQP2\n"
SCHEMA_VERSION = 2
NX = 23
NU = 2
HORIZON = 30
HEADER_SIZE = 8 + 7 * 4


class FixtureError(ValueError):
    pass


class Cursor:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0
        self.float_count = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise FixtureError("truncated QP fixture")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def word(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def signed(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def f32_array(self, count: int) -> tuple[float, ...]:
        values = struct.unpack(f"<{count}f", self.take(4 * count))
        self.float_count += count
        if not all(math.isfinite(value) for value in values):
            raise FixtureError("fixture contains a non-finite float")
        return values

    def f64(self) -> float:
        value = struct.unpack("<d", self.take(8))[0]
        if not math.isfinite(value):
            raise FixtureError("fixture contains a non-finite double")
        return value

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]


def parse_payload(data: bytes) -> dict:
    cursor = Cursor(data)
    if cursor.take(8) != MAGIC:
        raise FixtureError("bad fixture magic")
    version, endian, nx, nu, horizon, passes = (cursor.word() for _ in range(6))
    raw_status = cursor.signed()
    if version != SCHEMA_VERSION:
        raise FixtureError(f"unsupported schema version {version}")
    if endian != 0x01020304:
        raise FixtureError("unsupported endianness marker")
    if (nx, nu, horizon) != (NX, NU, HORIZON):
        raise FixtureError(
            f"dimension mismatch: got nx={nx}, nu={nu}, N={horizon}; "
            f"expected {NX}, {NU}, {HORIZON}")
    if passes < 1:
        raise FixtureError("fixture has no solver pass")

    per_stage = NX * NX + 2 * NX * NU + 5 * NX + 4 * NU
    lower_state_bounds = []
    upper_state_bounds = []
    lower_input_bounds = []
    upper_input_bounds = []
    for _ in range(HORIZON):
        cursor.f32_array(NX * NX)  # A
        cursor.f32_array(NX * NU)  # B
        cursor.f32_array(NX)  # d
        cursor.f32_array(NX)  # Q diagonal
        cursor.f32_array(NX)  # q
        cursor.f32_array(NX * NU)  # N cross term
        cursor.f32_array(NU)  # R diagonal
        cursor.f32_array(NU)  # r
        lower_state_bounds.append(cursor.f32_array(NX))
        upper_state_bounds.append(cursor.f32_array(NX))
        lower_input_bounds.append(cursor.f32_array(NU))
        upper_input_bounds.append(cursor.f32_array(NU))
    terminal_q = cursor.f32_array(NX)
    terminal_linear = cursor.f32_array(NX)
    terminal_lb = cursor.f32_array(NX)
    terminal_ub = cursor.f32_array(NX)
    cursor.f32_array(NX)  # x0
    for lower, upper in zip(lower_state_bounds + [terminal_lb],
                            upper_state_bounds + [terminal_ub]):
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise FixtureError("state lower bound exceeds upper bound")
    for lower, upper in zip(lower_input_bounds, upper_input_bounds):
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise FixtureError("input lower bound exceeds upper bound")

    rho, rho_u = cursor.f32_array(2)
    rho_state_by_coordinate = cursor.f32_array(NX)
    rho_input_by_coordinate = cursor.f32_array(NU)
    rho_max, solver_tolerance = cursor.f32_array(2)
    max_iterations = cursor.signed()
    adaptive_rho = cursor.signed()
    cold_start = cursor.signed()
    adaptive_update_period = cursor.signed()
    shared_rho = cursor.signed()
    over_relaxation = cursor.f32_array(1)[0]
    max_solve_time_us = cursor.f64()
    use_scaling = cursor.signed()
    state_scale = cursor.f32_array(NX)
    input_scale = cursor.f32_array(NU)

    cursor.f32_array((HORIZON + 1) * NX)  # initial z_x
    cursor.f32_array(HORIZON * NU)  # initial z_u
    cursor.f32_array((HORIZON + 1) * NX)  # initial y_x
    cursor.f32_array(HORIZON * NU)  # initial y_u
    cursor.f32_array(2)  # initial rho and rho_u
    cursor.f32_array((HORIZON + 1) * NX)  # rho by state
    cursor.f32_array(HORIZON * NU)  # rho by input
    initial_initialized = cursor.signed()
    initial_scaling_enabled = cursor.signed()
    cursor.f32_array(NX)  # state scale in state
    cursor.f32_array(NU)  # input scale in state

    cursor.f32_array((HORIZON + 1) * NX)  # final z_x
    cursor.f32_array(HORIZON * NU)  # final z_u
    cursor.f32_array((HORIZON + 1) * NX)  # final y_x
    cursor.f32_array(HORIZON * NU)  # final y_u
    cursor.f32_array(2)  # final rho and rho_u
    cursor.f32_array((HORIZON + 1) * NX)  # final rho by state
    cursor.f32_array(HORIZON * NU)  # final rho by input
    final_initialized = cursor.signed()
    final_scaling_enabled = cursor.signed()
    cursor.f32_array(NX)  # final state scale
    cursor.f32_array(NU)  # final input scale

    cursor.f32_array(HORIZON * NU)  # raw Riccati primal controls

    cursor.f32_array((HORIZON + 1) * NX)  # reference raw solution x
    cursor.f32_array(HORIZON * NU)  # reference raw solution u
    iterations = cursor.signed()
    time_budget_exhausted = cursor.signed()
    residuals = cursor.f32_array(6)
    locations = [cursor.signed() for _ in range(8)]
    solution_status = cursor.signed()
    debug_floats_0 = cursor.f32_array(4)  # initial/final rho values
    debug_rho_changes = cursor.signed()
    debug_floats_1 = cursor.f32_array(5)  # residuals and 2x2 inverse details
    debug_fallbacks = cursor.signed()
    debug_regularization = cursor.f32_array(1)
    debug_counts = [cursor.signed(), cursor.signed()]
    debug_times_ns = [cursor.u64() for _ in range(7)]
    if cursor.offset != len(data):
        raise FixtureError("fixture has trailing or unrecognized bytes")
    if not all(math.isfinite(value) for value in terminal_q + terminal_linear):
        raise FixtureError("invalid terminal objective")
    if max_solve_time_us < 0.0 or iterations < 0 or time_budget_exhausted not in (0, 1):
        raise FixtureError("invalid solver metadata")
    if (rho <= 0 or rho_u <= 0 or rho_max < 1 or rho_max > 127 or
            solver_tolerance <= 0 or max_iterations < 1 or max_iterations > 200 or
            adaptive_rho not in (0, 1) or cold_start not in (0, 1) or
            shared_rho not in (0, 1) or adaptive_update_period < 1 or
            not 1 <= over_relaxation <= 2 or use_scaling not in (0, 1) or
            any(value < 0 or value > 127 or 0 < value < 1
                for value in rho_state_by_coordinate) or
            any(value < 0 or value > 127 for value in rho_input_by_coordinate) or
            any(value <= 0 for value in state_scale + input_scale) or
            initial_initialized not in (0, 1) or
            initial_scaling_enabled not in (0, 1) or
            final_initialized not in (0, 1) or
            final_scaling_enabled not in (0, 1)):
        raise FixtureError("invalid solver configuration or ADMM state")
    if raw_status not in (0, 1, 2) or solution_status not in (0, 1, 2):
        raise FixtureError("invalid solver status")

    return {
        "schema_version": version,
        "nx": nx,
        "nu": nu,
        "qp_horizon": horizon,
        "capture_passes": passes,
        "raw_status": raw_status,
        "raw_solution_status": solution_status,
        "raw_iterations": iterations,
        "raw_time_budget_exhausted": time_budget_exhausted,
        "raw_residuals": residuals,
        "raw_residual_locations": locations,
        "float_count": cursor.float_count,
        "per_stage_float_count": per_stage,
        "initial_max_solve_time_us": max_solve_time_us,
        "debug_info": {
            "rho_values": debug_floats_0,
            "rho_change_count": debug_rho_changes,
            "residual_and_inverse_values": debug_floats_1,
            "inverse_fallback_count": debug_fallbacks,
            "regularization": debug_regularization[0],
            "regularization_count": debug_counts[0],
            "factorization_count": debug_counts[1],
            "phase_times_ns": debug_times_ns,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FixtureError(f"source bag path does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def source_set_sha256(root: Path, paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for rel in paths:
        path = root / rel
        if not path.is_file():
            raise FixtureError(f"ABI source file is missing: {rel}")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def saved_state_context(path: Path | None, row_index: int | None) -> dict | None:
    if path is None or row_index is None:
        return None
    with path.open(newline="") as stream:
        first = stream.readline().strip()
        if first.startswith("#"):
            names = next(csv.reader([first[1:].strip()]))
        else:
            stream.seek(0)
            names = next(csv.reader(stream))
        for index, row in enumerate(csv.reader(stream)):
            if not row or row[0].startswith("#"):
                continue
            if index == row_index:
                if len(row) != len(names):
                    raise FixtureError("saved-state row/header width mismatch")
                selected = dict(zip(names, row))
                return {
                    "row_index": row_index,
                    "input_row_sha256": hashlib.sha256(
                        ",".join(row).encode("utf-8")).hexdigest(),
                    "column_count": len(names),
                    "values": selected,
                }
    raise FixtureError(f"saved-state row {row_index} not found in {path}")


def git_metadata(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    status = git("status", "--porcelain")
    paths = set(git("diff", "--name-only", "HEAD").splitlines())
    paths.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    source_hashes = {}
    for rel in sorted(paths):
        path = root / rel
        if path.is_file():
            source_hashes[rel] = sha256_file(path)
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_clean": not bool(status),
        "working_tree_status": status,
        "changed_source_sha256": source_hashes,
    }


def seal(payload: Path, trajectory: Path | None, input_csv: Path | None,
         executable: Path | None, row: int | None, root: Path,
         output: Path | None = None,
         replay_command: list[str] | None = None,
         source_bag: Path | None = None,
         case_label: str | None = None,
         outcome: str | None = None,
         input_row_context: dict | None = None,
         case_context: dict | None = None) -> Path:
    data = payload.read_bytes()
    summary = parse_payload(data)
    digest = hashlib.sha256(data).hexdigest()
    metadata = {
        "schema": "sdu-apex-frozen-qp-fixture-metadata-v1",
        "payload": payload.name,
        "payload_sha256": digest,
        "canonical_qp_fixture_sha256": digest,
        "fixture_id": f"row-{row if row is not None else 'unknown'}-sha256-{digest}",
        "experiment_regime": "identical_qp",
        "source_regime": "saved_state_reconstruction",
        "live_solver_memory_restored": False,
        "dimensions": {key: summary[key] for key in
                       ("nx", "nu", "qp_horizon")},
        "capture_passes": summary["capture_passes"],
        "solver_recorded": {
            "status": summary["raw_status"],
            "solution_status": summary["raw_solution_status"],
            "iterations": summary["raw_iterations"],
            "time_budget_exhausted": summary["raw_time_budget_exhausted"],
            "residuals": summary["raw_residuals"],
            "residual_locations": summary["raw_residual_locations"],
            "max_solve_time_us": summary["initial_max_solve_time_us"],
            "debug_info": summary["debug_info"],
        },
        "source": git_metadata(root),
        "source_abi": {
            "model": {
                "files": [
                    "f1tenth_mpc/include/mpc_types.h",
                    "f1tenth_mpc/include/mpc_rti.h",
                    "f1tenth_mpc/src/mpc_rti.c",
                    "f1tenth_mpc/src/vehicle_model.c",
                    "f1tenth_mpc/src/unity_source_plant.cpp",
                ],
                "sha256": source_set_sha256(root, (
                    "f1tenth_mpc/include/mpc_types.h",
                    "f1tenth_mpc/include/mpc_rti.h",
                    "f1tenth_mpc/src/mpc_rti.c",
                    "f1tenth_mpc/src/vehicle_model.c",
                    "f1tenth_mpc/src/unity_source_plant.cpp")),
            },
            "solver_and_scaling": {
                "files": [
                    "f1tenth_mpc/include/riccati_solver.h",
                    "f1tenth_mpc/src/riccati_solver.c",
                ],
                "sha256": source_set_sha256(root, (
                    "f1tenth_mpc/include/riccati_solver.h",
                    "f1tenth_mpc/src/riccati_solver.c")),
            },
        },
        "machine": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "row_index": row,
        "trajectory": ({"path": str(trajectory.resolve()),
                        "sha256": sha256_file(trajectory)}
                       if trajectory else None),
        "saved_state_input": ({"path": str(input_csv.resolve()),
                                "sha256": sha256_file(input_csv)}
                               if input_csv else None),
        "replay_configuration": replay_command,
        "case_label": case_label,
        "source_outcome": outcome,
        "source_bag": ({"path": str(source_bag.resolve()),
                         "sha256": sha256_path(source_bag)}
                        if source_bag else None),
        "source_saved_state_row": input_row_context,
        "source_case_context": case_context,
        "solver_seed_provenance": (
            "ADMM memory is captured exactly from this saved-state replay at "
            "the frozen-QP boundary; the originating bag did not record the "
            "live ADMM z/y/rho state"),
        "replay_executable": ({"path": str(executable.resolve()),
                               "sha256": sha256_file(executable)}
                              if executable else None),
        "fixture_validation": summary,
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sidecar = output or payload.with_suffix(payload.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return sidecar


def validate_sidecar(payload: Path, sidecar: Path) -> dict:
    metadata = json.loads(sidecar.read_text())
    summary = parse_payload(payload.read_bytes())
    actual = sha256_file(payload)
    if metadata.get("payload_sha256") != actual:
        raise FixtureError("payload SHA-256 differs from sealed metadata")
    if metadata.get("canonical_qp_fixture_sha256") != actual:
        raise FixtureError("canonical fixture hash differs from sealed metadata")
    if metadata.get("dimensions") != {
            "nx": NX, "nu": NU, "qp_horizon": HORIZON}:
        raise FixtureError("sidecar dimensions do not match payload schema")
    source_abi = metadata.get("source_abi")
    if not isinstance(source_abi, dict) or set(source_abi) != {
            "model", "solver_and_scaling"}:
        raise FixtureError("sidecar source ABI manifest is missing or malformed")
    for component in source_abi.values():
        if (not isinstance(component, dict) or
                not isinstance(component.get("files"), list) or
                not component["files"] or
                any(not isinstance(path, str) or not path
                    for path in component["files"]) or
                not isinstance(component.get("sha256"), str) or
                re.fullmatch(r"[0-9a-f]{64}", component["sha256"]) is None):
            raise FixtureError("sidecar source ABI entry is malformed")
    return {"payload_sha256": actual, "summary": summary}


def capture_command(args: argparse.Namespace) -> int:
    if args.row is not None and args.rows:
        raise FixtureError("use either --row or --rows, not both")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(args.replay_binary.resolve()), str(args.trajectory.resolve()),
               str(args.input_csv.resolve()), *args.replay_args]
    environment = os.environ.copy()
    environment["MPC_REPLAY_QP_CAPTURE_DIR"] = str(output_dir)
    if args.row is not None:
        environment["MPC_REPLAY_QP_CAPTURE_ROW"] = str(args.row)
    else:
        environment.pop("MPC_REPLAY_QP_CAPTURE_ROW", None)
    if args.rows:
        selected_rows = [item.strip() for item in args.rows.split(",")]
        if not selected_rows or any(not item.isdigit() for item in selected_rows):
            raise FixtureError("--rows must be comma-separated nonnegative integers")
        environment["MPC_REPLAY_QP_CAPTURE_ROWS"] = ",".join(selected_rows)
    else:
        environment.pop("MPC_REPLAY_QP_CAPTURE_ROWS", None)
    completed = subprocess.run(command, env=environment, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True, check=False)
    (output_dir / "replay.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        print(f"replay failed: {completed.returncode}; see "
              f"{output_dir / 'replay.stderr.log'}", file=sys.stderr)
        return completed.returncode
    fixtures = sorted(output_dir.glob("qp_*.fqp"))
    if not fixtures:
        print("replay completed but captured no QP (check selected row/status)",
              file=sys.stderr)
        return 1
    for fixture in fixtures:
        stem = fixture.stem
        row = int(stem.split("_")[-1])
        sidecar = seal(fixture, args.trajectory, args.input_csv,
                       args.replay_binary, row, args.root,
                       replay_command=command,
                       source_bag=args.source_bag,
                       case_label=args.case_label,
                       outcome=args.outcome,
                       input_row_context=saved_state_context(
                           args.input_csv, row),
                       case_context=(json.loads(args.case_context_json.read_text())
                                     if args.case_context_json else None))
        print(f"sealed {fixture} -> {sidecar}")
    (output_dir / "capture_manifest.json").write_text(json.dumps({
        "schema": "sdu-apex-frozen-qp-capture-run-v1",
        "command": command,
        "return_code": completed.returncode,
        "requested_row": args.row,
        "requested_rows": args.rows,
        "case_label": args.case_label,
        "source_outcome": args.outcome,
        "source_bag_sha256": (sha256_path(args.source_bag)
                              if args.source_bag else None),
        "captured_fixtures": [str(path.name) for path in fixtures],
        "trajectory_sha256": sha256_file(args.trajectory),
        "input_csv_sha256": sha256_file(args.input_csv),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("payload", type=Path)
    validate_parser.add_argument("--sidecar", type=Path)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("payload", type=Path)
    seal_parser.add_argument("--trajectory", type=Path)
    seal_parser.add_argument("--input-csv", type=Path)
    seal_parser.add_argument("--executable", type=Path)
    seal_parser.add_argument("--row", type=int)
    seal_parser.add_argument("--source-bag", type=Path)
    seal_parser.add_argument("--case-label")
    seal_parser.add_argument("--outcome", choices=("success", "failure"))
    seal_parser.add_argument("--root", type=Path, default=root)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--replay-binary", type=Path, required=True)
    capture_parser.add_argument("--trajectory", type=Path, required=True)
    capture_parser.add_argument("--input-csv", type=Path, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--row", type=int)
    capture_parser.add_argument("--rows")
    capture_parser.add_argument("--source-bag", type=Path)
    capture_parser.add_argument("--case-label")
    capture_parser.add_argument("--case-context-json", type=Path)
    capture_parser.add_argument("--outcome", choices=("success", "failure"))
    capture_parser.add_argument("--root", type=Path, default=root)
    capture_parser.add_argument("replay_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.verb == "validate":
            if args.sidecar:
                result = validate_sidecar(args.payload, args.sidecar)
            else:
                result = {"payload_sha256": sha256_file(args.payload),
                          "summary": parse_payload(args.payload.read_bytes())}
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.verb == "seal":
            print(seal(args.payload, args.trajectory, args.input_csv,
                       args.executable, args.row, args.root.resolve()))
            return 0
        if args.verb == "capture":
            if args.replay_args and args.replay_args[0] == "--":
                args.replay_args = args.replay_args[1:]
            return capture_command(args)
    except (OSError, json.JSONDecodeError, FixtureError,
            subprocess.CalledProcessError) as error:
        print(f"fixture error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
