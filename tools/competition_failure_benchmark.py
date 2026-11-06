#!/usr/bin/env python3
"""Freeze and replay a Unity competition episode using the production source NLP.

Extraction needs the recorded run and ROS message types. Replay needs only the
checked-in bundle and native development tools; Unity/ROS are not required.
Historical SQP iterates reconstructed offline are explicitly distinguished from
recorded primals. An unsuccessful witness search never proves infeasibility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "f1tenth_mpc/test/data/competition_corner_20261105"
DEFAULT_TOOLS = ROOT / "build/f1tenth_mpc"


def read_json(path: Path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data)


def write_json(path: Path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(diagnostic: dict) -> list[str]:
    """Independent observations, never an inferred physical infeasibility proof."""
    solver = diagnostic["solver"]
    result = []
    if solver["source_exact_nlp_deadline_exhausted"]:
        result.append("deadline_exceeded")
    if not solver["source_exact_nlp_final_source_feasible"]:
        result.append("recorded_iterate_primal_infeasible")
    if (solver["source_exact_nlp_full_step_stalled"] and
            solver["source_exact_nlp_source_trials"] > solver["sqp_iterations_accepted"]):
        result.append("nonlinear_step_rejected")
    if not solver["source_exact_nlp_kkt_converged"]:
        result.append("KKT_not_converged")
    if solver["source_exact_nlp_final_source_feasible"]:
        result.append("feasible_solution_available")
    return result


def parity(actual: dict, diagnostic: dict) -> dict:
    """Compare the same primal only. Rejected final primals were not recorded."""
    s = diagnostic["solver"]
    fields = {
        "initial_cost": (actual["initial_cost"], s["source_exact_nlp_original_cost_initial"], 1e-7),
    }
    final = actual.get("recorded_final_score")
    if final is not None:
        fields.update({
            "final_cost": (final["cost"], s["source_exact_nlp_original_cost_final"], 1e-7),
            "final_clearance": (final["minimum_clearance_m"], s["source_exact_nlp_original_minimum_corner_margin_m"], 1e-6),
            "final_violation": (final["elastic_violation"], s["source_exact_nlp_original_constraint_violation"], 1e-7),
        })
    deltas = {key: a - b for key, (a, b, _) in fields.items()}
    return {"passed": all(math.isfinite(a) and math.isfinite(b) and
                           abs(a-b) <= tol * max(1., abs(b))
                           for a, b, tol in fields.values()),
            "deltas": deltas, "recorded_final_primal_available": final is not None}


def stable_result(value):
    """Numerical determinism excludes paths and wall-clock measurements."""
    if isinstance(value, dict):
        return {k: stable_result(v) for k, v in value.items()
                if k != "case" and not k.endswith("_ms")}
    if isinstance(value, list):
        return [stable_result(v) for v in value]
    return value


def bag_evidence(bag: Path, stamps: set[int]) -> tuple[dict, dict]:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    flags, commands = {}, {}
    for dbfile in sorted(bag.glob("*.db3")):
        with sqlite3.connect(dbfile) as db:
            topics = {name: (tid, kind) for tid, name, kind in
                      db.execute("SELECT id,name,type FROM topics")}
            for name in ("/odom/unity_plant_state", "/mpc/drive_command",
                         "/actuators/drive_command", "/odom/bridge_request_command"):
                tid, kind = topics[name]
                cls = get_message(kind)
                stream = []
                for receipt, data in db.execute(
                        "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (tid,)):
                    msg = deserialize_message(data, cls)
                    stamp = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
                    if name == "/odom/unity_plant_state":
                        if stamp in stamps:
                            flags[stamp] = bool(msg.suspension_history_initialized)
                    else:
                        stream.append({"stamp_ns": stamp, "bag_receipt_ns": receipt,
                                       "steering_rad": float(msg.steering_angle_rad),
                                       "throttle": float(msg.throttle_normalized)})
                if name != "/odom/unity_plant_state":
                    commands.setdefault(name, []).extend(stream)
    if set(flags) != stamps:
        raise ValueError("missing exact-stamp source suspension flags in bag")
    return flags, commands


def extract(run: Path, output: Path, start: float, stop: float):
    import yaml

    params = {}
    for name in ("mpc_base.yaml", "mpc_overlay.yaml"):
        params.update(yaml.safe_load((run/name).read_text())["/**"]["ros__parameters"])
    all_rows = [json.loads(line)["payload"] for line in
                (run/"cycles_diagnostics.jsonl").read_text().splitlines() if line.strip()]
    selected = [(i, row) for i, row in enumerate(all_rows)
                if start <= row.get("progress_m", -1) <= stop and "solver" in row]
    if not selected:
        raise ValueError("no source solver epochs in selected progress interval")
    stamps = {r["source_stamp_ns"] for _, r in selected}
    flags, commands = bag_evidence(run/"run", stamps)
    epochs = []
    last_issued = 0
    no_publication = 0
    maximum_streak = 0
    selected_indices = {i for i, _ in selected}
    for i, row in enumerate(all_rows):
        if "solver" not in row:
            continue
        if row.get("command_published"):
            last_issued = row["issued_command_stamp_ns"]
            no_publication = 0
        else:
            no_publication += 1
        if i not in selected_indices:
            continue
        maximum_streak = max(maximum_streak, no_publication)
        source = row["source_stamp_ns"]
        # The controller sets source/IMU initialized after reading this packet.
        # An integrated command-time prediction initializes suspension history.
        propagated = row["candidate_command_stamp_ns"] > source
        row["plant_state_flags"] = [1, int(flags[source] or propagated), 1]
        epochs.append({"record_index": i, "diagnostic": row,
                       "flags_provenance": "exact-stamp bag suspension flag and controller command-time propagation contract",
                       "observed_causes": classify(row),
                       "consecutive_no_publication_epochs": no_publication,
                       "last_published_command_epoch_ns": last_issued,
                       "last_published_command_age_at_candidate_ms":
                           (row["candidate_command_stamp_ns"]-last_issued)/1e6 if last_issued else None})
    feasible = [e for e in epochs if e["diagnostic"]["command_published"]]
    infeasible = [e for e in epochs if not e["diagnostic"]["solver"]["source_exact_nlp_final_source_feasible"]]
    failures = [e for e in epochs if e["diagnostic"]["status"] == "rejected_nonlinear_rollout"]
    roles = {"last_feasible": feasible[-1]["record_index"],
             "first_infeasible": infeasible[0]["record_index"],
             "terminal_failure": failures[0]["record_index"],
             "last_recorded": epochs[-1]["record_index"]}
    lo, hi = min(stamps), max(e["diagnostic"]["candidate_command_stamp_ns"] for e in epochs)
    # Retain the command preceding this interval as well as every command in it.
    for topic, stream in commands.items():
        predecessors = [r for r in stream if r["stamp_ns"] < lo]
        commands[topic] = predecessors[-1:] + [r for r in stream if lo <= r["stamp_ns"] <= hi]
    warnings = []
    for a, b in re.findall(r"Ignoring older MPC command epoch \((\d+) < (\d+)\)", (run/"controller.log").read_text()):
        warnings.append({"ignored_epoch_ns": int(a), "latest_epoch_ns": int(b),
                         "backward_delta_ms": (int(b)-int(a))/1e6})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"sequence.json.gz", {"schema": "competition-source-sequence/v1",
        "run": run.name, "parameters": params, "epochs": epochs, "roles": roles,
        "commands": commands, "older_epoch_warnings": warnings,
        "maximum_no_publication_streak": maximum_streak,
        "historical_capture_gaps": ["Individual live SQP primals were not logged.",
            "Rejected cycles omit the final primal; their final scores cannot be independently certified.",
            "Frozen schedules are reconstructed from captured seeds and checked by source cost parity.",
            "Host command association does not establish Unity FixedUpdate application time."]})
    (output/"raceline.csv.gz").write_bytes(gzip.compress((run/"raceline.csv").read_bytes(), mtime=0))
    pinned = [ROOT/"f1tenth_mpc/src/unity_source_plant.cpp", ROOT/"f1tenth_mpc/src/vehicle_model.cpp",
              ROOT/"f1tenth_mpc/include/mpc_types.h", ROOT/"f1tenth_mpc/include/vehicle_model.h",
              ROOT/"f1tenth_mpc/include/unity_source_plant.hpp", ROOT/"f1tenth_mpc/include/unity_source_plant_c.h"]
    write_json(output/"physical_model.json.gz", {str(p.relative_to(ROOT)): p.read_text() for p in pinned})
    write_json(output/"manifest.json", {"schema": "competition-source-benchmark/v1", "roles": roles,
        "input_sha256": {name: sha(run/name) for name in
                         ("cycles_diagnostics.jsonl", "mpc_base.yaml", "mpc_overlay.yaml", "raceline.csv", "controller_image_id.txt")},
        "bundle_sha256": {name: sha(output/name) for name in ("sequence.json.gz", "raceline.csv.gz", "physical_model.json.gz")},
        "source_model_sha256": {str(p.relative_to(ROOT)): sha(p) for p in pinned},
        "epoch_count": len(epochs), "progress_interval_m": [start, stop]})
    return {"epochs": len(epochs), "roles": roles, "output": str(output)}


def solver_environment(parameters: dict, controls_only_trace: bool = False) -> dict:
    """Frozen numerical settings, with all optional research modes disabled."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MPC_SOURCE_NLP_")}
    env.update(MPC_SOURCE_NLP_QP_REGULARIZATION=str(parameters["source_exact_nlp_qp_regularization"]),
               MPC_SOURCE_NLP_QP_SOFT_WEIGHT=str(parameters["source_exact_nlp_qp_soft_weight"]),
               MPC_SOURCE_NLP_MIN_CLEARANCE_M=str(float32(parameters["source_exact_nlp_extra_clearance_m"])),
               MPC_SOURCE_NLP_RECORD_PRIMAL_TRACE="1")
    if controls_only_trace:
        env["MPC_SOURCE_NLP_TRACE_CONTROLS_ONLY"] = "1"
    return env


def replay(bundle: Path, output: Path, tools: Path, iterations: str, repeat: int,
           solver_timeout: float | None = 60, controls_only_trace: bool = False, jobs: int = 1):
    if jobs < 1:
        raise ValueError("replay jobs must be positive")
    manifest = read_json(bundle/"manifest.json")
    for name, expected in manifest["bundle_sha256"].items():
        if sha(bundle/name) != expected:
            raise ValueError(f"corrupt benchmark input: {name}")
    sequence = read_json(bundle/"sequence.json.gz")
    baseline = read_json(bundle/"oracle_baseline.json.gz") if (bundle/"oracle_baseline.json.gz").is_file() else None
    for name in ("mpc_source_live_fixture", "mpc_source_exact_offline_optimize",
                 "mpc_source_feasibility_witness_probe"):
        if not (tools/name).is_file():
            raise ValueError(f"build required development tool: {tools/name}")
    output.mkdir(parents=True, exist_ok=True)
    (output/"raceline.csv").write_bytes(gzip.decompress((bundle/"raceline.csv.gz").read_bytes()))
    write_json(output/"params.json", sequence["parameters"])
    params = sequence["parameters"]
    env = solver_environment(params, controls_only_trace)
    def replay_epoch(epoch):
        d = dict(epoch["diagnostic"])
        if baseline:
            # Optimizer comparisons must use the frozen original problem,
            # including its schedule, rather than rebuilding new targets.
            d["frozen_source_problem"] = baseline["epochs"][str(epoch["record_index"])]["frozen_problem"]["frozen_source_problem"]
        case = output/f"epoch_{epoch['record_index']}"
        case.mkdir(exist_ok=True)
        write_json(case/"recorded.json", d)
        subprocess.run([str(tools/"mpc_source_live_fixture"), str(case/"recorded.json"),
                        str(output/"params.json"), str(output/"raceline.csv"), str(case)],
                       check=True, capture_output=True, text=True, timeout=30)
        frozen = read_json(case/"frozen_problem.json")
        check = parity(frozen, d)
        attempts = max(1, d["solver"]["sqp_iterations_attempted"]) if iterations == "recorded" else int(iterations)
        samples = []
        for _ in range(repeat):
            proc = subprocess.run([str(tools/"mpc_source_exact_offline_optimize"), str(case), "-", str(attempts),
                                   str(case/"final.controls")], env=env, check=True,
                                  capture_output=True, text=True, timeout=solver_timeout)
            samples.append(json.loads(proc.stdout))
        deterministic = all(stable_result(s) == stable_result(samples[0]) for s in samples[1:])
        result = samples[0]
        write_json(case/"replay.json.gz", result)
        # Trust neither the changed optimizer's feasibility flag nor its
        # summary score. Independently integrate its returned float controls
        # with the original frozen problem, including on rejected cycles.
        controls = [[float(v) for v in line.split()] for line in
                    (case/"final.controls").read_text().splitlines()]
        checked = dict(d)
        checked["source_optimized_controls"] = controls
        checked["solver"] = dict(d["solver"],
            source_exact_nlp_original_cost_final=result["final_cost"],
            source_exact_nlp_original_minimum_corner_margin_m=result["final_corridor_margin_m"],
            source_exact_nlp_original_constraint_violation=result["final_constraint_violation"])
        write_json(case/"optimizer_primal.json", checked)
        validation = case/"optimizer_validation"
        subprocess.run([str(tools/"mpc_source_live_fixture"), str(case/"optimizer_primal.json"),
                        str(output/"params.json"), str(output/"raceline.csv"), str(validation)],
                       check=True, capture_output=True, text=True, timeout=30)
        independently_scored = read_json(validation/"frozen_problem.json")
        final_parity = parity(independently_scored, checked)
        final_score = independently_scored["recorded_final_score"]
        independently_feasible = final_score["minimum_clearance_m"] >= 0. and \
            final_score["rate_margin"] >= -1e-7 and final_score["box_margin"] >= -1e-7
        solver = d["solver"]
        historical_deltas = {"cost": result["final_cost"]-solver["source_exact_nlp_original_cost_final"],
            "clearance_m": result["final_corridor_margin_m"]-solver["source_exact_nlp_original_minimum_corner_margin_m"],
            "elastic_violation": result["final_constraint_violation"]-solver["source_exact_nlp_original_constraint_violation"]}
        return {"record_index": epoch["record_index"], "progress_m": d["progress_m"],
            "recorded_status": d["status"], "observed_causes": epoch["observed_causes"],
            "oracle_parity": check, "deterministic": deterministic,
            "optimizer_primal_parity": final_parity,
            "optimizer_feasibility_flag_correct": independently_feasible == result["source_feasible_final"],
            "source_feasible_final": result["source_feasible_final"],
            "original_nlp_kkt_converged": result["original_nlp_kkt_converged"],
            "final_clearance_m": result["final_corridor_margin_m"],
            "historical_reconstruction_deltas": historical_deltas,
            "regressed_feasibility": bool(solver["source_exact_nlp_final_source_feasible"] and
                                         not result["source_feasible_final"]),
            "timing_qualification": "offline iterations; no integrated deadline qualification"}
    # Cases are independent native processes. Preserve corpus order and all
    # deterministic-repeat/original-source checks when running concurrently.
    with ThreadPoolExecutor(max_workers=jobs) as workers:
        reports = list(workers.map(replay_epoch, sequence["epochs"]))
    report = {"schema": "competition-source-benchmark-result/v1", "epochs": reports,
        "roles": sequence["roles"], "iterations": iterations, "repeat": repeat,
        "passed": all(r["oracle_parity"]["passed"] and r["optimizer_primal_parity"]["passed"] and
                      r["optimizer_feasibility_flag_correct"] and r["deterministic"] and
                      not r["regressed_feasibility"] for r in reports),
        "historical_capture_gaps": sequence["historical_capture_gaps"],
        "physical_unrecoverability_proven": False}
    report["recorded_witness_regressions"] = []
    if (bundle/"investigation.json.gz").is_file():
        for known in read_json(bundle/"investigation.json.gz")["admissible_reachability"]["searches"]:
            if known["best_minimum_clearance_m"] < 0. or not known["original_nlp_prefix_source_feasible"]:
                # A failed finite search is evidence to retain, not a
                # known feasible witness that future solvers must preserve.
                continue
            case = output/f"epoch_{known['record_index']}"
            controls = case/f"known_witness_{known['steps']}.controls"
            controls.write_text("".join(f"{u[0]:.10g} {u[1]:.10g}\n" for u in known["controls"]))
            proc = subprocess.run([str(tools/"mpc_source_feasibility_witness_probe"), str(case),
                                   str(controls), str(known["steps"]), "--evaluate-only"],
                                  check=True, capture_output=True, text=True, timeout=30)
            check = json.loads(proc.stdout)
            ok = check["best_minimum_clearance_m"] >= 0. and check["best_original_nlp_admissible"]
            report["recorded_witness_regressions"].append({"role": known["role"], "steps": known["steps"],
                "record_index": known["record_index"],
                "minimum_clearance_m": check["best_minimum_clearance_m"], "passed": ok})
            report["passed"] &= ok
    report["unresolved_reachability_epochs"] = [e["record_index"] for e in sequence["epochs"]
        if not any(r["passed"] and r["record_index"] == e["record_index"] for r in
                   report["recorded_witness_regressions"]) and not any(
                       r["record_index"] == e["record_index"] and r["source_feasible_final"] for r in reports)]
    write_json(output/"report.json", report)
    return report


def freeze_baseline(bundle: Path, replay_dir: Path):
    sequence = read_json(bundle/"sequence.json.gz")
    report = read_json(replay_dir/"report.json")
    if not report["passed"] or report["iterations"] != "recorded" or report["repeat"] < 2:
        raise ValueError("baseline needs a passing, twice-repeated replay with recorded iteration counts")
    records = {}
    for epoch in sequence["epochs"]:
        index = epoch["record_index"]
        case = replay_dir/f"epoch_{index}"
        records[str(index)] = {"frozen_problem": read_json(case/"frozen_problem.json"),
                               "reconstructed_sqp": stable_result(read_json(case/"replay.json.gz"))}
    write_json(bundle/"oracle_baseline.json.gz", {
        "schema": "competition-source-oracle-baseline/v1",
        "primal_trace_origin": "offline_reconstruction_not_historical_live_iterates",
        "epochs": records})
    manifest = read_json(bundle/"manifest.json")
    manifest["bundle_sha256"]["oracle_baseline.json.gz"] = sha(bundle/"oracle_baseline.json.gz")
    write_json(bundle/"manifest.json", manifest)
    return {"frozen_epochs": len(records), "baseline": str(bundle/"oracle_baseline.json.gz")}


def float32(value: float) -> float:
    import struct
    return struct.unpack("f", struct.pack("f", value))[0]


def witness(bundle: Path, replay_dir: Path, tools: Path, roles: list[str], steps: list[int]):
    sequence = read_json(bundle/"sequence.json.gz")
    reports = []
    selected = [("recorded_epoch", e["record_index"]) for e in sequence["epochs"]] if roles == ["all"] else [
        (role, sequence["roles"][role]) for role in roles]
    for role, index in selected:
        case = replay_dir/f"epoch_{index}"
        if not (case/"source_context.mctx").is_file():
            raise ValueError("run replay into --replay-dir before witness search")
        for horizon in steps:
            target = case/f"witness_{horizon}"
            target.mkdir(exist_ok=True)
            shutil.copyfile(case/"source_context.mctx", target/"source_context.mctx")
            proc = subprocess.run([str(tools/"mpc_source_feasibility_witness_probe"), str(target),
                                   str(case/"initial.controls"), str(horizon)],
                                  check=True, capture_output=True, text=True, timeout=60)
            report = json.loads(proc.stdout)
            controls = [[float(v) for v in line.split()] for line in
                        (target/"witness.controls").read_text().splitlines()]
            if len(controls) != horizon or any(len(u) != 2 or
                    not all(math.isfinite(v) for v in u) or not .001 <= u[1] <= 1. for u in controls):
                raise ValueError("witness violates the positive-throttle controller contract")
            # The original oracle independently validates the 30-stage prefix
            # of every witness. Extended stages use the identical source stage
            # and footprint, not a substituted continuation model.
            env = {k: v for k, v in os.environ.items() if not k.startswith("MPC_SOURCE_NLP_")}
            p = sequence["parameters"]
            env.update(MPC_SOURCE_NLP_QP_REGULARIZATION=str(p["source_exact_nlp_qp_regularization"]),
                       MPC_SOURCE_NLP_MIN_CLEARANCE_M=str(float32(p["source_exact_nlp_extra_clearance_m"])))
            checked = subprocess.run([str(tools/"mpc_source_exact_offline_optimize"), str(case),
                                      str(target/"witness.controls"), "1"], env=env,
                                     check=True, capture_output=True, text=True, timeout=60)
            oracle = json.loads(checked.stdout)
            report.update(role=role, record_index=index,
                          original_nlp_prefix_source_feasible=oracle["source_feasible_initial"],
                          original_nlp_prefix_clearance_m=oracle["initial_corridor_margin_m"],
                          controls=controls,
                          scope="offline source existence witness; no optimality, Unity, or deadline qualification")
            write_json(target/"report.json", report)
            reports.append(report)
    result = {"schema": "competition-source-reachability/v1", "searches": reports,
              "infeasibility_proven": False, "zero_throttle_permitted": False}
    write_json(replay_dir/"reachability.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=float, default=142.68)
    p.add_argument("--stop", type=float, default=144.16, help="includes the recorded 144.154m epoch")
    p = sub.add_parser("replay")
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    p.add_argument("--iterations", default="recorded", help="recorded iteration count or 1..100")
    p.add_argument("--repeat", type=int, default=2)
    p = sub.add_parser("witness")
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    p.add_argument("--roles", nargs="+", default=["first_infeasible"],
                   choices=["first_infeasible", "last_feasible", "terminal_failure", "last_recorded", "all"])
    p.add_argument("--steps", nargs="+", type=int, default=[30, 60])
    p = sub.add_parser("freeze-baseline")
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "extract":
            result = extract(args.run.resolve(), args.output.resolve(), args.start, args.stop)
        elif args.command == "replay":
            if args.repeat < 1 or (args.iterations != "recorded" and not 1 <= int(args.iterations) <= 100):
                parser.error("repeat must be positive; iterations must be recorded or 1..100")
            result = replay(args.bundle.resolve(), args.output.resolve(), args.tools.resolve(), args.iterations, args.repeat)
        elif args.command == "witness":
            if any(not 30 <= n <= 120 for n in args.steps):
                parser.error("witness steps must be in 30..120")
            if "all" in args.roles and args.roles != ["all"]:
                parser.error("use --roles all without other roles")
            result = witness(args.bundle.resolve(), args.replay_dir.resolve(), args.tools.resolve(), args.roles, args.steps)
        else:
            result = freeze_baseline(args.bundle.resolve(), args.replay_dir.resolve())
        print(json.dumps(result, indent=2))
        return 0 if result.get("passed", True) else 1
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"passed": False, "error": str(error),
                          "stderr": getattr(error, "stderr", None)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
