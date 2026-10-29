"""Tests use synthetic artifacts; no Docker/Unity tests are implied."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.optimizer_mpc_validation import (
    certified_exports, initialize, discover, classification, summary_from_report,
    request_retry, consume_retries, safe_file, start_orientation_conflict,
    analysis_command,
    yield_optimizer, recover_optimizer,
)


def fixture(tmp_path):
    workspace = tmp_path / "optimizer_test"
    folder = workspace / "exports" / "candidate"
    folder.mkdir(parents=True)
    witness = folder / "witness.npz"
    witness.write_bytes(b"synthetic witness for contract test")
    digest = hashlib.sha256(witness.read_bytes()).hexdigest()
    (folder / "source_command_raceline_with_witness.csv").write_text(
        "# sdu_apex_source_witness/v1\n# synthetic fixture\n0,0\n")
    (folder / "report.json").write_text(json.dumps(dict(status="source_witness_passed")))
    (folder / "manifest.json").write_text(json.dumps(dict(
        lap_s=7.35, direction="clockwise", source_only=True,
        source_plant_sha256="synthetic-source",
        witness_sha256=digest,
        files=dict(
            witness=str(witness),
            validation=str(folder / "report.json"),
            raceline_with_witness=str(folder / "source_command_raceline_with_witness.csv")))))
    (workspace / "optimizer.json").write_text("{}")
    (workspace / "best.json").write_text(json.dumps(dict(witness_sha256=digest)))
    return workspace, folder, digest


def test_source_validation_and_first_queue_only_contains_current(tmp_path):
    workspace, _, digest = fixture(tmp_path)
    items = certified_exports(workspace)
    assert len(items) == 1 and items[0]["id"] == digest[:16]
    state = initialize(workspace, workspace / "mpc_validation/state.json")
    assert state["pending"] == [digest[:16]]
    discover(workspace, state)
    assert state["pending"] == [digest[:16]]
    assert state["policy"]["scored_laps"] == 10


@pytest.mark.parametrize("edit", [
    lambda x: x.update(source_only=False),
    lambda x: x.update(witness_sha256="0"*64),
    lambda x: x.update(direction="counterclockwise"),
])
def test_never_queue_unvalidated_or_wrong_hash(tmp_path, edit):
    workspace, folder, _ = fixture(tmp_path)
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))
    assert certified_exports(workspace) == []


def test_refuse_external_paths_and_symlink(tmp_path):
    workspace, folder, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        safe_file("/etc/passwd", workspace)
    external = workspace / "escape"
    external.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_file(str(external / "other"), workspace)


def test_retry_failed_only_and_no_duplicate(tmp_path):
    workspace, _, digest = fixture(tmp_path)
    item = certified_exports(workspace)[0]
    state = initialize(workspace, workspace / "mpc_validation/state.json")
    state["pending"].clear()
    state["attempts"].append(dict(id=item["id"], status="mpc_rejected"))
    request_retry(workspace, item["id"])
    consume_retries(workspace, state, [item])
    assert state["pending"] == [item["id"]]
    request_retry(workspace, item["id"])
    consume_retries(workspace, state, [item])
    assert state["pending"] == [item["id"]]
    state["pending"].clear()
    state["attempts"].append(dict(id=item["id"], status="completed"))
    # Last result is successful; retries should be rejected.
    request_retry(workspace, item["id"])
    consume_retries(workspace, state, [item])
    assert state["pending"] == []


def test_failure_reason_and_success_marker(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "watcher.log").write_text(
        "WATCHER_READY\nMPC_CYCLE_REJECTED\n"
        "SCREEN_ABORT: MPC fault (nonlinear_failure@3)\n")
    assert classification(run, 5)[0] == "mpc_rejected"
    (run / "watcher.log").write_text("SCREEN_COMPLETE: lap limit\n")
    assert classification(run, 0)[0] == "completed"
    (run / "watcher.log").write_text("SCREEN_ABORT: collision\n")
    assert classification(run, 3)[0] == "collision"


def test_ten_scored_laps_exclude_warmup_and_report_tracking():
    laps = [19] + [8 + index / 10 for index in range(10)]
    report = dict(schema="v4", summary=dict(
        recorded_lap_times_s=laps, cycles_scored=3000,
        controller_tracking=dict(cross_track_rmse_m=.045, heading_rmse_rad=.03)))
    metrics = summary_from_report(report)
    assert metrics["warmup_lap_s"] == 19
    assert metrics["completed_scored_laps"] == 10
    assert metrics["total_scored_s"] == pytest.approx(84.5)
    assert metrics["average_scored_s"] == pytest.approx(8.45)
    assert metrics["best_scored_s"] == 8
    assert metrics["tracking"]["cross_track_rmse_m"] == .045


def test_analysis_uses_recorded_ros_image_and_repository_paths(tmp_path, monkeypatch):
    import tools.optimizer_mpc_validation as validation
    monkeypatch.setattr(validation, 'REPO', tmp_path)
    run = tmp_path/'live_runs/apex_mpc_fixture_a01'
    run.mkdir(parents=True)
    image = 'sha256:'+'a'*64
    (run/'controller_image_id.txt').write_text(image+'\n')
    command = analysis_command(run, run/'raceline.csv')
    assert image in command and 'docker' == command[0]
    assert '/workspace/src/live_runs/apex_mpc_fixture_a01/run' in command
    assert '/workspace/src/live_runs/apex_mpc_fixture_a01/raceline.csv' in command
    assert any('source /opt/ros/humble/setup.bash' in part for part in command)
    (run/'controller_image_id.txt').write_text('invalid')
    with pytest.raises(ValueError, match='image ID'):
        analysis_command(run, run/'raceline.csv')


def test_known_iros_clockwise_reference_is_opposite_initial_spawn_line():
    root = Path(__file__).resolve().parents[1]
    candidate = root / ("live_runs/optimizer_autodrive_iros_2026_compete_dc3c93097d/"
                        "exports/8c2ef06394b9af69/source_command_raceline.csv")
    baseline = root / "f1tenth_planning/trajectories/autodrive_iros_2026_compete/initial_raceline.csv"
    if candidate.is_file() and baseline.is_file():
        reason = start_orientation_conflict(candidate, baseline)
        assert reason and "opposite direction" in reason


def test_optimizer_is_released_even_if_trial_fails(tmp_path, monkeypatch):
    import tools.optimizer_mpc_validation as validation
    calls = []
    monkeypatch.setattr(validation, 'optimizer_unit', lambda _: 'sdu-track-optimizer-abc.service')
    def run(command, **_):
        calls.append(command)
        return SimpleNamespace(stdout='running\n')
    monkeypatch.setattr(validation.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='trial failed'):
        with yield_optimizer(tmp_path):
            assert (tmp_path/'mpc_validation/optimizer_pause.json').is_file()
            raise RuntimeError('trial failed')
    assert [command[2] for command in calls] == ['show', 'freeze', 'thaw']
    assert not (tmp_path/'mpc_validation/optimizer_pause.json').exists()


def test_restart_recovers_a_crashed_supervisors_optimizer_lease(tmp_path, monkeypatch):
    import tools.optimizer_mpc_validation as validation
    unit = 'sdu-track-optimizer-abc.service'
    calls = []
    monkeypatch.setattr(validation, 'optimizer_unit', lambda _: unit)
    monkeypatch.setattr(validation.subprocess, 'run', lambda command, **_: calls.append(command))
    validation.atomic_json(tmp_path/'mpc_validation/optimizer_pause.json', dict(unit=unit))
    recover_optimizer(tmp_path)
    assert calls == [['systemctl', '--user', 'thaw', unit]]
    assert not (tmp_path/'mpc_validation/optimizer_pause.json').exists()


def test_no_optimizer_means_no_freeze_or_thaw(tmp_path, monkeypatch):
    import tools.optimizer_mpc_validation as validation
    monkeypatch.setattr(validation, 'optimizer_unit', lambda _: None)
    monkeypatch.setattr(validation.subprocess, 'run', lambda *_, **__: pytest.fail('unexpected process control'))
    with yield_optimizer(tmp_path):
        pass


def test_never_steal_an_existing_pause(tmp_path, monkeypatch):
    import tools.optimizer_mpc_validation as validation
    monkeypatch.setattr(validation, 'optimizer_unit', lambda _: 'sdu-track-optimizer-abc.service')
    monkeypatch.setattr(validation.subprocess, 'run', lambda *_, **__: SimpleNamespace(stdout='frozen\n'))
    with pytest.raises(RuntimeError, match='another owner'):
        with yield_optimizer(tmp_path):
            pytest.fail('must not launch a competing trial')
