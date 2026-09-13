import csv
import importlib.util

import pytest


_ASSEMBLER_PATH = "tools/model_id/assemble_transitions.py"
_SPEC = importlib.util.spec_from_file_location("assemble_transitions", _ASSEMBLER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ASSEMBLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ASSEMBLER)


def _write_packets(path, count=12, reverse_index=None, time_gap_index=None,
                   time_step_s=0.025, reset_index=None, position_jump_index=None):
    fields = [
        "simulation_time_s", "simulation_physics_step", "simulation_render_frame",
        "simulator_position_x", "simulator_position_y", "simulator_position_z",
        "simulator_orientation_euler_z",
        "simulator_orientation_quaternion_x", "simulator_orientation_quaternion_y",
        "simulator_orientation_quaternion_z", "simulator_orientation_quaternion_w",
        "simulator_linear_velocity_x", "simulator_linear_velocity_y",
        "simulator_linear_velocity_z", "simulator_angular_velocity_x",
        "simulator_angular_velocity_y", "simulator_angular_velocity_z",
        "applied_command_sequence",
        "applied_throttle_norm", "applied_steering_norm",
        "commanded_throttle_norm", "commanded_steering_norm",
        "simulator_feedback_steering_norm", "sent_reset",
        "simulator_encoder_angles_left", "simulator_encoder_angles_right",
    ]
    rows = []
    for index in range(count):
        if reverse_index == index:
            step = index + 1
        elif reverse_index is not None and index == reverse_index + 1:
            step = index - 1
        else:
            step = index
        source_time = index * time_step_s
        if time_gap_index is not None and index > time_gap_index:
            source_time += 1.0
        steering_command = -0.1 + 0.01 * index
        position_x = index * 0.05
        if position_jump_index is not None and index >= position_jump_index:
            position_x += 2.0
        rows.append({
            "simulation_time_s": source_time,
            "simulation_physics_step": step,
            "simulation_render_frame": index,
            "simulator_position_x": position_x,
            "simulator_position_y": 0.0,
            "simulator_position_z": 0.0,
            "simulator_orientation_euler_z": 0.0,
            "simulator_orientation_quaternion_x": 0.0,
            "simulator_orientation_quaternion_y": 0.0,
            "simulator_orientation_quaternion_z": 0.0,
            "simulator_orientation_quaternion_w": 1.0,
            "simulator_linear_velocity_x": 2.0,
            "simulator_linear_velocity_y": 0.0,
            "simulator_linear_velocity_z": 0.0,
            "simulator_angular_velocity_x": 0.0,
            "simulator_angular_velocity_y": 0.0,
            "simulator_angular_velocity_z": 0.0,
            "applied_command_sequence": index,
            "applied_throttle_norm": 0.2 + 0.01 * index,
            # Applied and feedback steering are physical radians in the
            # source packet despite the historical ``_norm`` field name.
            "applied_steering_norm": steering_command * _ASSEMBLER.MAX_STEERING_RAD,
            "commanded_throttle_norm": 0.2 + 0.01 * index,
            "commanded_steering_norm": steering_command,
            "simulator_feedback_steering_norm": steering_command * _ASSEMBLER.MAX_STEERING_RAD,
            "sent_reset": index == reset_index,
            "simulator_encoder_angles_left": index * 0.1,
            "simulator_encoder_angles_right": index * 0.1,
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_assembler_uses_declared_body_frame_and_passes_known_fixture(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_packets(run_dir / "simulator_packets.csv")
    report = _ASSEMBLER.assemble(run_dir, "auto", False, False, 0.75)
    assert report["twist_frame_selected"] == "body"
    assert report["quality_gate_pass"] is True
    assert report["kinematic_consistency"]["yaw_rate_source"] == \
        "simulator_angular_velocity_z"
    assert report["kinematic_consistency"]["diagnostics"][
        "position_vs_body_velocity_error_mps"]["median"] == pytest.approx(0.0)
    assert report["kinematic_consistency"]["yaw_rate_pass"] is True
    with (run_dir / "assembled" / "model_transition_v4.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[0]["applied_throttle_norm_k1"]) == pytest.approx(0.21)
    assert float(rows[0]["commanded_steering_norm_k1"]) == pytest.approx(-0.09)
    assert float(rows[0]["applied_steering_rad_k1"]) == pytest.approx(
        -0.09 * _ASSEMBLER.MAX_STEERING_RAD)
    assert float(rows[0]["simulator_feedback_steering_rad_k1"]) == pytest.approx(
        -0.09 * _ASSEMBLER.MAX_STEERING_RAD)
    assert int(float(rows[0]["applied_command_sequence_k1"])) == 1
    assert int(float(rows[0]["segment_id"])) == 0
    assert float(rows[0]["wheel_speed_mps_k1"]) == pytest.approx(
        _ASSEMBLER.ENCODER_WHEEL_RADIUS_M * 0.1 / 0.025)


def test_assembler_rejects_source_order_violation(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_packets(run_dir / "simulator_packets.csv", reverse_index=6)
    report = _ASSEMBLER.assemble(run_dir, "body", False, False, 0.75)
    assert report["source_order"]["pass"] is False
    assert report["status"] == "rejected_source_order"
    assert report["quality_gate_pass"] is False
    assert "source_order" in report["quality_gate_failures"]


def test_assembler_rejects_long_source_time_gap(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_packets(run_dir / "simulator_packets.csv", time_gap_index=6)
    report = _ASSEMBLER.assemble(run_dir, "body", False, False, 0.75)
    assert report["status"] == "rejected_non_target_source_rate"
    assert report["quality_gate_pass"] is False
    assert report["source_dt_gap_count"] == 1
    assert report["source_dt_s_max"] == pytest.approx(1.025)
    assert report["transition_schema"]["discarded_transition_count"] == 1
    assert "source_time_gap" in report["quality_gate_failures"]


def test_assembler_rejects_ten_hz_source_cadence(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_packets(run_dir / "simulator_packets.csv", time_step_s=0.1)
    report = _ASSEMBLER.assemble(run_dir, "body", False, False, 0.75)
    assert report["quality_gate_pass"] is False
    assert report["source_dt_cadence_violation_count"] == 11
    assert "source_cadence" in report["quality_gate_failures"]


def test_assembler_marks_reset_as_hard_rollout_boundary(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_packets(run_dir / "simulator_packets.csv", reset_index=6)
    report = _ASSEMBLER.assemble(run_dir, "body", False, False, 0.75)
    assert report["quality_gate_pass"] is True
    assert report["segments"]["boundary_count"] == 1
    assert report["segments"]["boundaries"][0]["reason"] == \
        "reset_command_rising_edge"
    with (run_dir / "assembled" / "model_transition_v4.csv").open(
            newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {int(float(row["segment_id"])) for row in rows} == {0, 1}
    assert all(
        int(float(row["segment_id"])) == int(float(row["reset_epoch"]))
        for row in rows)


def test_assembler_marks_pose_discontinuity_as_hard_rollout_boundary(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_packets(run_dir / "simulator_packets.csv", position_jump_index=6)
    report = _ASSEMBLER.assemble(run_dir, "body", False, False, 0.75)
    assert report["segments"]["boundary_count"] == 1
    assert report["segments"]["boundaries"][0]["reason"] == \
        "position_discontinuity"
