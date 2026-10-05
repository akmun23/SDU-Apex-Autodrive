from __future__ import annotations

import numpy as np
import pytest

from tools.vehicle_dynamics_learning.build_body_sysid_dataset import (
    DT_S,
    REAR_AXLE_TO_COM_X_M,
    build,
)


def _write_source(path, *, invalid_cadence: bool = False) -> None:
    names = np.asarray((
        "throttle_command_norm", "u_rear_mps", "steering_feedback_rad",
        "yaw_rate_rps", "throttle_feedback_norm", "steering_command_rad",
        "v_rear_mps", "rear_left_surface_mps", "rear_right_surface_mps",
    ))
    n = 9
    frames = np.zeros((n, len(names)), dtype=np.float32)
    index = {name: i for i, name in enumerate(names)}
    frames[:, index["steering_feedback_rad"]] = np.arange(n) * 0.01
    frames[:, index["throttle_feedback_norm"]] = 0.1 + np.arange(n) * 0.02
    frames[:, index["steering_command_rad"]] = -np.arange(n) * 0.02
    frames[:, index["throttle_command_norm"]] = 0.8 - np.arange(n) * 0.03
    rigid = np.zeros((n, 13), dtype=np.float32)
    rigid[:, 7] = 1.0 + np.arange(n) * 0.1
    rigid[:, 8] = -0.2 + np.arange(n) * 0.02
    rigid[:, 12] = 0.5 + np.arange(n) * 0.01
    dt = np.full(n, DT_S, dtype=np.float32)
    if invalid_cadence:
        dt[4] = 0.05
    np.savez_compressed(
        path,
        feature_names=names,
        run_ids=np.asarray(("train_run", "validation_run", "test_run",
                            "final_run")),
        run_splits=np.asarray(("train", "validation", "test", "final_test")),
        frames=frames,
        dt_s=dt,
        simulator_rigid_state=rigid,
        sequence_bounds=np.asarray(((0, 3), (3, 5), (5, 7), (7, 9))),
        sequence_run_index=np.asarray((0, 1, 2, 3), dtype=np.int32),
        sequence_condition_id=np.asarray((4, 5, 6, 7), dtype=np.int32),
        sequence_reset_index=np.asarray((1, 1, 1, 1), dtype=np.int32),
        sequence_labels=np.asarray(("a", "b", "c", "d")),
        frame_run_index=np.repeat(np.arange(4, dtype=np.int32), (3, 2, 2, 2)),
        frame_reset_index=np.ones(n, dtype=np.int32),
        sample_time_ns=np.arange(n, dtype=np.int64) * 25_000_000,
    )


def test_body_dataset_uses_only_feedback_inputs_and_rear_axle_truth(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "body.npz"
    _write_source(source)

    report = build(source, output)
    with np.load(source, allow_pickle=False) as raw, np.load(
            output, allow_pickle=False) as result:
        names = raw["feature_names"].astype(str).tolist()
        feature = {name: i for i, name in enumerate(names)}
        np.testing.assert_array_equal(
            result["inputs"], raw["frames"][:5, [
                feature["steering_feedback_rad"],
                feature["throttle_feedback_norm"]]])
        np.testing.assert_array_equal(
            result["stored_commands"], raw["frames"][:5, [
                feature["steering_command_rad"],
                feature["throttle_command_norm"]]])
        rigid = raw["simulator_rigid_state"][:5]
        expected = np.column_stack((
            rigid[:, 7], rigid[:, 8] - REAR_AXLE_TO_COM_X_M * rigid[:, 12],
            rigid[:, 12]))
        np.testing.assert_allclose(result["outputs"], expected, rtol=0, atol=1e-7)
        assert set(result["split"].astype(str)) == {"train", "validation"}
        assert result["sequence_bounds"].tolist() == [[0, 3], [3, 5]]
        assert result["sequence_id_table"].astype(str).tolist() == [
            "train_run/sequence_00000", "validation_run/sequence_00001"]
        np.testing.assert_allclose(
            result["sample_time_s"], [0.0, 0.025, 0.05, 0.0, 0.025],
            rtol=0, atol=1e-8)
        np.testing.assert_allclose(result["dt_s"], DT_S, rtol=0, atol=1e-8)

    assert report["run_counts"] == {"train": 1, "validation": 1}
    assert report["excluded_runs_by_split"] == {
        "final_test": ["final_run"], "test": ["test_run"]}
    assert report["future_measurements_or_truth_used_as_input"] is False


def test_body_dataset_rejects_non_25ms_cadence(tmp_path):
    source = tmp_path / "bad_source.npz"
    output = tmp_path / "body.npz"
    _write_source(source, invalid_cadence=True)
    with pytest.raises(ValueError, match="25 ms"):
        build(source, output)
    assert not output.exists()


def test_body_dataset_refuses_overwrite(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "body.npz"
    _write_source(source)
    output.write_bytes(b"preserve-me")
    with pytest.raises(FileExistsError):
        build(source, output)
    assert output.read_bytes() == b"preserve-me"
