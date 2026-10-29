#!/usr/bin/env python3
"""Independent serialized Unity/MPC test queue for source-certified IROS racelines.

Watches immutable source-certified optimizer exports, NOT unvalidated IPOPT guesses.
Keeps the optimizer's own checkpoint and best.json untouched. Never runs more
than one trial at once. A trial is one warmup plus ten scored official laps.

Commands:
 python3 tools/optimizer_mpc_validation.py --workspace live_runs/optimizer_... --once
 python3 tools/optimizer_mpc_validation.py --workspace live_runs/optimizer_... --watch
 python3 tools/optimizer_mpc_validation.py --workspace live_runs/optimizer_... --retry WITNESS_SHA_PREFIX
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
HEX = re.compile(r"^[a-f0-9]{16,64}$")
TARGET_LAPS = 11
SCORING_LAPS = 10


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {} if default is None else default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_file(path: str, workspace: Path) -> Path:
    try:
        target = Path(path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("source-certified artifact unavailable") from exc
    if not target.is_file() or not target.is_relative_to(workspace.resolve()):
        raise ValueError("source-certified artifact outside optimizer workspace")
    return target


def certified_exports(workspace: Path) -> list[dict]:
    result = []
    for manifest_path in workspace.glob("exports/*/manifest.json"):
        try:
            item = load_json(manifest_path)
            witness_hash = item.get("witness_sha256", "")
            if (not HEX.fullmatch(witness_hash) or not item.get("source_only") or
                    item.get("unity_validated") or item.get("runtime_promoted") or
                    item.get("direction") not in ("clockwise", "unity_verified")):
                continue
            if item.get("direction") == "unity_verified":
                # Verify this candidate was built against the known
                # independently completed IROS Unity racing orientation.
                reference = REPO / "f1tenth_planning/trajectories/autodrive_iros_2026_compete/initial_raceline.csv"
                if (not reference.is_file() or
                        item.get("direction_reference_sha256") != sha256(reference)):
                    continue
            report = safe_file(item["files"]["validation"], workspace)
            witness = safe_file(item["files"]["witness"], workspace)
            trajectory = safe_file(item["files"]["raceline_with_witness"], workspace)
            if (manifest_path.parent.resolve() != trajectory.parent or
                    sha256(witness) != witness_hash or
                    load_json(report).get("status") != "source_witness_passed"):
                continue
            with trajectory.open() as stream:
                schema = stream.readline().strip()
            if schema != "# sdu_apex_source_witness/v1":
                continue
            result.append(dict(id=witness_hash[:16], witness_sha256=witness_hash,
                               lap_s=float(item["lap_s"]), source_sha256=item["source_plant_sha256"],
                               csv=str(trajectory), validation=str(report),
                               certified_epoch_s=manifest_path.stat().st_mtime))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sorted(result, key=lambda item: (item["certified_epoch_s"], item["id"]))


def initialize(workspace: Path, state_path: Path) -> dict:
    exports = certified_exports(workspace)
    state = load_json(state_path, None)
    if state:
        return state
    # Avoid an automatic storm of historic runs: queue the latest valid best
    # and watch *future* exports; historic exports are marked seen.
    best = load_json(workspace / "best.json")
    latest_id = str(best.get("witness_sha256", ""))[:16]
    state = dict(schema="sdu-apex-mpc-validation/v1", created_epoch_s=time.time(),
                 seen=[item["id"] for item in exports],
                 pending=[latest_id] if any(item["id"] == latest_id for item in exports) else [],
                 attempts=[], worker=dict(status="idle"), policy=dict(
                     warmup_laps=1, scored_laps=SCORING_LAPS,
                     track="iros", stop_on_mpc_rejection=True, collision_abort=True))
    atomic_json(state_path, state)
    return state


def discover(workspace: Path, state: dict) -> None:
    seen = set(state["seen"])
    for item in certified_exports(workspace):
        if item["id"] not in seen:
            state["seen"].append(item["id"])
            state["pending"].append(item["id"])
            seen.add(item["id"])


def start_orientation_conflict(trajectory: Path, baseline: Path) -> str | None:
    """Compare candidate tangent near the previously driven IROS spawn path.

    This is a launch-preflight safety gate, NOT a Unity physics or source-model
    certificate. The previously completed IROS initial trial establishes a
    practical spawn-facing reference for this fixed simulator.
    """
    import csv
    import math

    def poses(path):
        points = []
        with path.open(newline="") as stream:
            for row in csv.reader(line for line in stream if not line.startswith("#")):
                if len(row) < 4:
                    continue
                try:
                    x, y, yaw = map(float, row[1:4])
                except ValueError:
                    continue
                if all(math.isfinite(v) for v in (x, y, yaw)):
                    points.append((x, y, yaw))
        return points

    if not baseline.is_file():
        return None
    old, new = poses(baseline), poses(trajectory)
    if not old or not new:
        return "Cannot verify IROS start heading: reference CSV lacks finite poses"
    sx, sy, heading = old[0]
    near = min(new, key=lambda p: (p[0]-sx)**2+(p[1]-sy)**2)
    distance = math.hypot(near[0]-sx, near[1]-sy)
    delta = math.atan2(math.sin(near[2]-heading), math.cos(near[2]-heading))
    if distance < .75 and abs(delta) > 2.0:
        return (f"Known simulator spawn orientation incompatible with certified line: "
                f"nearest point {distance:.3f} m away, heading offset "
                f"{delta:.3f} rad ({math.degrees(delta):.1f} deg). "
                "Previously completed IROS baseline faces the opposite direction. "
                "Reject this optimized direction as incompatible with the "
                "official Unity-driven start; do not weaken AMCL heading rejection.")
    return None


def classification(run_dir: Path, returncode: int) -> tuple[str, str]:
    watcher = (run_dir / "watcher.log").read_text(errors="replace") if (run_dir / "watcher.log").exists() else ""
    if "SCREEN_COMPLETE: lap limit" in watcher and returncode == 0:
        return "completed", "Eleven official laps completed; bag analysis pending"
    for token, kind in (("SCREEN_ABORT: MPC startup timeout", "mpc_startup_timeout"),
                        ("SCREEN_ABORT: MPC fault", "mpc_rejected"),
                        ("SCREEN_ABORT: collision", "collision"),
                        ("SCREEN_ABORT: no odometry progress", "motion_stalled"),
                        ("SCREEN_ABORT: timeout", "watch_timeout"),
                        ("LAP_COUNT_AHEAD_OF_DISTANCE", "lap_counter_inconsistent")):
        if token in watcher:
            lines = [line.strip() for line in watcher.splitlines() if token in line]
            return kind, lines[-1][:1400]
    controller_log = run_dir / "controller.log"
    if controller_log.exists():
        controller = controller_log.read_text(errors="replace")
        if ("Global AMCL candidate rejected: scan heading error" in controller
                and "MPC waiting for its first localized map pose" in controller):
            evidence = [line.strip() for line in controller.splitlines()
                        if "Global AMCL candidate rejected: scan heading error" in line]
            return "localization_start_heading_mismatch", evidence[-1][:1200]
    for name, signatures in (("controller", ["FATAL", "Traceback", "Error while loading", "Failed to load"]),
                             ("recorder", ["ERROR", "Traceback"]),
                             ("simulator", ["ERROR", "failed"])):
        log = run_dir / (name + ".log")
        if log.exists():
            found = [line.strip() for line in log.read_text(errors="replace").splitlines()
                     if any(token in line for token in signatures)]
            if found:
                return name + "_error", found[-1][:1400]
    return "trial_launch_failed", f"Trial exit status {returncode}; inspect controller.log, watcher.log, simulator.log and recorder.log"


def summary_from_report(report: dict) -> dict:
    data = report.get("summary", {})
    all_laps = data.get("recorded_lap_times_s") or []
    laps = [float(v) for v in all_laps if isinstance(v, (float, int)) and v > 0]
    scored = laps[1:TARGET_LAPS]
    controller = data.get("controller_tracking", {}) or {}
    result = dict(lap_times_all_s=laps, warmup_lap_s=laps[0] if laps else None,
                  scored_lap_times_s=scored, completed_scored_laps=len(scored),
                  expected_scored_laps=SCORING_LAPS,
                  tracking=dict(controller),
                  absolute_error_distributions=data.get("absolute_error_distributions", {}),
                  status_counts=data.get("status_counts", {}),
                  cycle_reason_counts=data.get("cycle_reason_counts", {}),
                  nonlinear_failure_reason_counts=data.get("nonlinear_failure_reason_counts", {}),
                  controller_cycles=data.get("cycles_scored"),
                  score_cutoff_reason=data.get("score_cutoff_reason"),
                  per_official_lap_tracking=report.get("per_official_lap_tracking", []),
                  progress_bins=report.get("per_progress_bin", []),
                  gate_timing_ms=data.get("gate_ms"),
                  jacobian_profile=data.get("jacobian_profile"),
                  solver_profile=data.get("solver_profile"),
                  diagnostic_profile=data.get("diagnostic_profile"),
                  solver_termination_counts=data.get("solver_termination_counts"),
                  solver_stop_reason_counts=data.get("solver_stop_reason_counts"),
                  report_schema=report.get("schema"))
    if scored:
        result.update(total_scored_s=sum(scored), average_scored_s=statistics.mean(scored),
                      median_scored_s=statistics.median(scored),
                      best_scored_s=min(scored), worst_scored_s=max(scored),
                      stddev_scored_s=statistics.pstdev(scored))
    return result


def analysis_command(run_dir: Path, reference: Path) -> list[str]:
    """Decode bags with the same ROS messages/runtime used to collect them."""
    def mounted(path):
        return '/workspace/src/'+str(path.resolve().relative_to(REPO.resolve()))
    image_file = run_dir/'controller_image_id.txt'
    image = image_file.read_text().strip() if image_file.exists() else os.environ.get(
        'SDU_APEX_MAPPING_IMAGE', 'sdu-apex-autodrive:iros-mapping')
    if image_file.exists() and not re.fullmatch(r'sha256:[a-f0-9]{64}', image):
        raise ValueError('Invalid recorded controller image ID')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_dir.name):
        raise ValueError('Invalid analysis run name')
    return ['docker', 'run', '--rm', '--name', 'report_'+run_dir.name,
        '-v', str(REPO.resolve())+':/workspace/src:rw', '--entrypoint', '/bin/bash', image,
        '-lc', 'set -e; source /opt/ros/humble/setup.bash; '
        'source /home/autodrive_devkit/install/setup.bash; '
        'source /workspace/install/setup.bash; cd /workspace/src; '
        'exec python3 tools/export_mpc_run_report.py "$@"', 'mpc-report',
        mounted(run_dir/'run'), '--root', '/workspace/src', '--trajectory', mounted(reference),
        '--experiment-regime', 'live_closed_loop',
        '--output-json', mounted(run_dir/'mpc_run_report.json'),
        '--output-csv', mounted(run_dir/'mpc_cycle_detail.csv')]


def run_analysis(run_dir: Path, reference: Path, attempt: dict) -> None:
    bag = run_dir / "run"
    if not (bag / "metadata.yaml").exists():
        attempt["analysis_error"] = "ROS bag metadata.yaml missing; see recorder.log"
        return
    report_path = run_dir / "mpc_run_report.json"
    try:
        command = analysis_command(run_dir, reference)
    except (OSError, ValueError) as exc:
        attempt['analysis_error'] = str(exc)
        return
    with (run_dir / "analysis.log").open("w") as log:
        try:
            finished = subprocess.run(command, cwd=REPO, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=240)
        except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit) as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                subprocess.run(['docker', 'stop', '--time', '0', 'report_'+run_dir.name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                raise
            if isinstance(exc, subprocess.TimeoutExpired):
                try:
                    subprocess.run(['docker', 'stop', '--time', '0', 'report_'+run_dir.name],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            attempt["analysis_error"] = str(exc)
            return
    if finished.returncode:
        attempt["analysis_error"] = f"Report generator exited {finished.returncode}; inspect analysis.log"
        return
    try:
        attempt["metrics"] = summary_from_report(load_json(report_path))
        attempt.pop('analysis_error', None)
    except (ValueError, KeyError, TypeError) as exc:
        attempt["analysis_error"] = str(exc)


def simulation_busy() -> bool:
    result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                            capture_output=True, text=True, timeout=8)
    if result.returncode:
        raise RuntimeError("Docker unavailable: " + result.stderr.strip()[:250])
    names = result.stdout.splitlines()
    return any(("_sim" in name or name.endswith("_controller") or name.startswith("rec_"))
               for name in names)


def optimizer_unit(workspace: Path) -> str | None:
    """Resolve only this workspace's live optimizer; never freeze a login scope."""
    pid = load_json(workspace / 'optimizer.json').get('pid')
    if not isinstance(pid, int) or pid <= 0:
        return None
    proc = Path('/proc') / str(pid)
    try:
        args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        if not any(Path(arg).name == 'track_optimizer.py' for arg in args):
            raise RuntimeError('Optimizer PID no longer identifies track_optimizer.py')
        output = args[args.index('--output') + 1]
        base = proc.joinpath('cwd').resolve(strict=True)
        if (base / output).resolve() != workspace.resolve():
            raise RuntimeError('Optimizer PID belongs to another workspace')
        cgroup = proc.joinpath('cgroup').read_text().strip().split('/')[-1]
    except FileNotFoundError:
        return None
    except (ValueError, IndexError) as exc:
        raise RuntimeError('Cannot verify optimizer process identity') from exc
    if not re.fullmatch(r'sdu-track-optimizer-[a-f0-9]+\.service', cgroup):
        raise RuntimeError('Optimizer must run in its own systemd service before automatic trials')
    result = subprocess.run(['systemctl', '--user', 'show', cgroup, '-p', 'MainPID', '--value'],
                            capture_output=True, text=True, timeout=10, check=True)
    if result.stdout.strip() != str(pid):
        raise RuntimeError('Optimizer systemd identity changed')
    return cgroup


def recover_optimizer(workspace: Path) -> None:
    """Release a freeze left by a crashed supervisor, not an arbitrary unit."""
    lease = workspace / 'mpc_validation/optimizer_pause.json'
    saved = load_json(lease)
    if not saved:
        return
    unit = optimizer_unit(workspace)
    if unit and unit == saved.get('unit'):
        subprocess.run(['systemctl', '--user', 'thaw', unit], check=True, timeout=15)
    lease.unlink(missing_ok=True)


@contextmanager
def yield_optimizer(workspace: Path):
    """One heavy workload: freeze IPOPT during build, Unity, and bag decoding.

    Freezing the dedicated cgroup includes any solver workers and preserves the
    exact running iterate. The durable lease lets a restarted supervisor thaw
    it after a crash. A normal service stop raises KeyboardInterrupt and thaws
    in finally; stopping a frozen optimizer service also thaws it in systemd.
    """
    unit = optimizer_unit(workspace)
    lease = workspace / 'mpc_validation/optimizer_pause.json'
    if unit:
        current = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'FreezerState', '--value'],
                                 capture_output=True, text=True, check=True, timeout=10)
        if current.stdout.strip() != 'running':
            raise RuntimeError('Optimizer is already paused by another owner; deferring trial')
        atomic_json(lease, dict(unit=unit, owner_pid=os.getpid(),
                               reason='Unity driving test', started_epoch_s=time.time()))
    try:
        if unit:
            subprocess.run(['systemctl', '--user', 'freeze', unit], check=True, timeout=15)
        yield
    finally:
        if unit:
            subprocess.run(['systemctl', '--user', 'thaw', unit], check=True, timeout=15)
            lease.unlink(missing_ok=True)


def execute(workspace: Path, state: dict, item: dict, state_path: Path, *, dry_run=False) -> dict:
    ident = item["id"]
    iteration = 1 + sum(a["id"] == ident for a in state["attempts"])
    run_id = f"apex_mpc_{ident}_a{iteration:02d}"
    run_dir = REPO / "live_runs" / run_id
    input_dir = workspace / "mpc_validation" / "inputs" / ident
    reference = input_dir / "source_command_raceline_with_witness.csv"
    attempt = dict(id=ident, witness_sha256=item["witness_sha256"],
                   model_lap_s=item["lap_s"], run_id=run_id,
                   run_dir=str(run_dir), status="queued",
                   attempt=iteration, trajectory_sha256=None,
                   queued_epoch_s=time.time(), source_only=True,
                   unity_validated=False, mpc_validated=False)
    if dry_run:
        print(json.dumps(attempt, indent=2))
        return attempt
    input_dir.mkdir(parents=True, exist_ok=True)
    if reference.exists():
        if sha256(reference) != sha256(Path(item["csv"])):
            raise RuntimeError("snapshot differs from immutable source-certified CSV")
    else:
        temporary = reference.with_suffix(".tmp")
        shutil.copyfile(item["csv"], temporary)
        temporary.replace(reference)
    attempt["trajectory_sha256"] = sha256(reference)
    state["pending"].remove(ident)
    state["attempts"].append(attempt)
    state["worker"] = dict(status="running", id=ident, run_id=run_id)
    attempt["status"] = "running"
    attempt["started_epoch_s"] = time.time()
    atomic_json(state_path, state)
    env = os.environ.copy()
    env.update(SDU_APEX_IROS_TRAJECTORY_FILE=str(reference),
               SDU_APEX_IROS_TOTAL_LAPS=str(TARGET_LAPS),
               SDU_APEX_IROS_WALL_TIMEOUT_S=env.get("SDU_APEX_AUTO_TRIAL_TIMEOUT_S", "900"))
    # The existing launcher builds the production controller image so that a
    # stale prebuilt image is never silently treated as a fresh trial.
    command = ["bash", str(REPO / "tools/run_iros_trial.sh"), run_id]
    status = 1
    try:
        with (input_dir / f"attempt_{iteration:02d}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=1800)
            except (KeyboardInterrupt, SystemExit):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                attempt.update(status='interrupted', reason='Validation supervisor stopped',
                               finished_epoch_s=time.time())
                state['worker'] = dict(status='idle')
                atomic_json(state_path, state)
                raise
            except subprocess.TimeoutExpired:
                # Signal the entire trial process group so shell EXIT traps can
                # stop *its own* simulator/recorder/controller containers.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                attempt["exception"] = "Trial exceeded outer 1800-second watchdog"
                status = 124
    except OSError as exc:
        attempt["exception"] = str(exc)
    kind, reason = classification(run_dir, status)
    attempt.update(status=kind, reason=reason, launcher_exit_code=status,
                   finished_epoch_s=time.time())
    if run_dir.exists():
        run_analysis(run_dir, reference, attempt)
    if kind == "completed":
        count = (attempt.get("metrics") or {}).get("completed_scored_laps")
        if count != SCORING_LAPS:
            attempt["status"] = "incomplete_evidence"
            attempt["reason"] = (f"Watcher completed, but bag yields {count} scored laps; "
                                 "check recorder and lap counter")
        else:
            attempt["mpc_validated"] = True
            attempt["unity_trial_completed"] = True
            # Do NOT claim source-model/native Unity physics parity from a
            # successful MPC run. Independent plant parity remains unverified.
            attempt["reason"] = "Ten scored MPC laps completed without watcher abort"
    state["worker"] = dict(status="idle")
    atomic_json(state_path, state)
    return attempt


def request_retry(workspace: Path, prefix: str) -> Path:
    if not HEX.fullmatch(prefix):
        raise ValueError("Retry requires a 16-64 character witness hash prefix")
    file = workspace / "mpc_validation" / "retry_requests" / (uuid.uuid4().hex + ".json")
    file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(file, dict(witness_prefix=prefix, created_epoch_s=time.time()))
    return file


def consume_retries(workspace: Path, state: dict, exports: list[dict]) -> None:
    directory = workspace / "mpc_validation" / "retry_requests"
    for path in sorted(directory.glob("*.json")):
        request = load_json(path)
        prefix = request.get("witness_prefix", "")
        found = [x for x in exports if x["witness_sha256"].startswith(prefix)]
        recent = next((a for a in reversed(state["attempts"])
                       if len(found) == 1 and a["id"] == found[0]["id"]), None)
        if (len(found) == 1 and recent is not None and
                recent["status"] not in {"completed", "running"} and
                found[0]["id"] not in state["pending"]):
            state["pending"].append(found[0]["id"])
        path.unlink()


def loop(workspace: Path, *, once=False, dry_run=False) -> None:
    workspace = workspace.resolve(strict=True)
    if not (workspace / "optimizer.json").is_file():
        raise ValueError("Missing optimizer.json")
    data_dir = workspace / "mpc_validation"
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_file = data_dir / ".lock"
    with lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_optimizer(workspace)
        state_path = data_dir / "state.json"
        state = initialize(workspace, state_path)
        # Reacquiring the exclusive worker lock proves no old worker remains.
        # Preserve partial logs but never silently mark an interrupted run OK.
        for old in state["attempts"]:
            if old.get("status") == "running":
                run_dir = Path(old.get("run_dir") or "/nonexistent")
                kind, reason = classification(run_dir, 143)
                if kind == "trial_launch_failed":
                    kind = "interrupted"
                old.update(status=kind, finished_epoch_s=time.time(),
                           reason=reason)
        state["worker"] = dict(status="idle")
        atomic_json(state_path, state)
        print("Validation supervisor workspace:", workspace, flush=True)
        while True:
            state['supervisor'] = dict(pid=os.getpid(), heartbeat_epoch_s=time.time())
            exports = certified_exports(workspace)
            discover(workspace, state)
            consume_retries(workspace, state, exports)
            atomic_json(state_path, state)
            candidate = next((item for item in exports if item["id"] in state["pending"]), None)
            if candidate:
                try:
                    # Cross-workspace lock is held across the complete trial,
                    # including log/bag analysis. A second supervisor cannot
                    # race the Docker preflight between inspecting and launching.
                    global_file = REPO / "live_runs" / ".apex_unity_validation_global.lock"
                    with global_file.open("a") as global_lock:
                        try:
                            fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            print("Another source-validated Unity trial holds the global lock; deferring", flush=True)
                        else:
                            if simulation_busy():
                                print("Another Unity/controller/recorder container is active; deferring", flush=True)
                            else:
                                print("Trial queued:", candidate["lap_s"], candidate["id"], flush=True)
                                baseline = REPO / "f1tenth_planning/trajectories/autodrive_iros_2026_compete/initial_raceline.csv"
                                conflict = start_orientation_conflict(Path(candidate["csv"]), baseline)
                                if conflict and os.environ.get("SDU_APEX_ALLOW_OPPOSITE_START") != "1":
                                    if not dry_run:
                                        state["pending"].remove(candidate["id"])
                                        state["attempts"].append(dict(
                                            id=candidate["id"], model_lap_s=candidate["lap_s"],
                                            status="blocked_start_orientation", reason=conflict,
                                            attempt=1 + sum(a["id"] == candidate["id"] for a in state["attempts"]),
                                            witness_sha256=candidate["witness_sha256"],
                                            finished_epoch_s=time.time(),
                                            source_only=True, mpc_validated=False))
                                        atomic_json(state_path, state)
                                    print("Preflight blocked:", conflict, flush=True)
                                else:
                                    if dry_run:
                                        outcome = execute(workspace, state, candidate, state_path, dry_run=True)
                                    else:
                                        with yield_optimizer(workspace):
                                            outcome = execute(workspace, state, candidate, state_path)
                                    print("Trial outcome:", outcome["status"], outcome.get("reason"), flush=True)
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    print("Supervisor preflight or trial error:", exc, flush=True)
                    state["worker"] = dict(status="blocked", reason=str(exc))
                    atomic_json(state_path, state)
            if once:
                break
            time.sleep(5)


def main() -> None:
    def stopped(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stopped)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--watch", action="store_true")
    group.add_argument("--once", action="store_true")
    group.add_argument("--retry", metavar="WITNESS_HASH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.retry:
        print("Retry requested:", request_retry(args.workspace.resolve(), args.retry))
    else:
        try:
            loop(args.workspace, once=not args.watch, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print('Validation supervisor stopped; optimizer released', flush=True)


if __name__ == "__main__":
    main()
