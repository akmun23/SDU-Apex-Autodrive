import csv
import importlib.util

import pytest


_ASSEMBLER_PATH = "tools/model_id/assemble_transitions.py"
_SPEC = importlib.util.spec_from_file_location("assemble_transitions", _ASSEMBLER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ASSEMBLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ASSEMBLER)


def _write_packets(path, count=12, reverse_index=None):
    fields = [
        "simulation_time_s", "simulation_physics_step", "simulation_render_frame",
        "simulator_position_x", "simulator_position_y", "simulator_position_z",
        "simulator_orientation_euler_z",
        "simulator_orientation_quaternion_x", "simulator_orientation_quaternion_y",
        "simulator_orientation_quaternion_z", "simulator_orientation_quaternion_w",
        "simulator_linear_velocity_x", "simulator_linear_velocity_y",
        "simulator_linear_velocity_z", "simulator_angular_velocity_x",
        "simulator_angular_velocity_y", "simulator_angular_velocity_z",
        "applied_throttle_norm", "applied_steering_norm",
    ]
    rows = []
    for index in range(count):
        if reverse_index == index:
            step = index + 1
        elif reverse_index is not None and index == reverse_index + 1:
            step = index - 1
        else:
            step = index
        rows.append({
            "simulation_time_s": index * 0.025,
            "simulation_physics_step": step,
            "simulation_render_frame": index,
            "simulator_position_x": index * 0.05,
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
            "applied_throttle_norm": 0.2,
            "applied_steering_norm": 0.0,
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


def test_assembler_rejects_source_order_violation(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_packets(run_dir / "simulator_packets.csv", reverse_index=6)
    report = _ASSEMBLER.assemble(run_dir, "body", False, False, 0.75)
    assert report["source_order"]["pass"] is False
    assert report["status"] == "rejected_source_order"
    assert report["quality_gate_pass"] is False
    assert "source_order" in report["quality_gate_failures"]
