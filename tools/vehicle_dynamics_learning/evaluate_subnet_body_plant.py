#!/usr/bin/env python3
"""Verify the repository-native SUBNET against the frozen deepSI reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from subnet_body_plant import SubnetBodyPlant


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
REFERENCE_DIR = RESET_ROOT / "deepsi_subnet_reference_v1"
DATASET = RESET_ROOT / "body_sysid_v1.npz"
STARTS = RESET_ROOT / "evaluation_starts.json"
REFERENCE_CHECKPOINT = REFERENCE_DIR / (
    "SS_encoder_sdu_apex_subnet_reference_20261004_best.pth")
NATIVE_CHECKPOINT = REFERENCE_DIR / "subnet_body_plant_native_reference_v1.npz"
REPORT = REFERENCE_DIR / "subnet_native_reproduction_report_v1.json"
HISTORY = 12
HORIZONS = (40, 150)
DT_S = 0.025


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_native_from_reference(path: Path) -> tuple[Any, SubnetBodyPlant]:
    import deepSI

    reference = deepSI.fit_systems.SS_encoder(nx=9, na=HISTORY, nb=HISTORY)
    reference.__dict__ = torch.load(path, map_location="cpu", weights_only=False)
    norm = reference.norm
    native = SubnetBodyPlant(norm.u0, norm.ustd, norm.y0, norm.ystd,
                             history_steps=HISTORY, state_order=9)
    for name in ("encoder", "fn", "hn"):
        getattr(native, name).load_state_dict(getattr(reference, name).state_dict())
    native.eval()
    return reference, native


def _fixed_start_arrays(data: Any, starts_doc: dict[str, Any]
                        ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
    sequence_runs = data["sequence_run_id"].astype(str)
    sequence_conditions = np.asarray(data["sequence_condition_id"], dtype=np.int64)
    sequence_splits = data["sequence_split"].astype(str)
    inputs = np.asarray(data["inputs"], dtype=np.float32)
    outputs = np.asarray(data["outputs"], dtype=np.float32)
    row_runs = data["run_id"].astype(str)
    row_splits = data["split"].astype(str)
    references: list[dict[str, Any]] = []
    for run_id, items in sorted(
            starts_doc["split_roles"]["development_validation"].items()):
        if len(items) != 64:
            raise ValueError(f"frozen start count changed for {run_id}")
        for item in items:
            if (item["run_id"] != run_id
                    or item["split_role"] != "development_validation"):
                raise ValueError("frozen start metadata changed")
            source_row = int(item["source_row"])
            condition = int(item["condition_id"])
            sequence_matches = np.flatnonzero(
                (sequence_runs == run_id) & (sequence_conditions == condition)
                & (sequence_splits == "validation"))
            if len(sequence_matches) != 1:
                raise ValueError("frozen condition is not in exactly one body sequence")
            sequence_index = int(sequence_matches[0])
            begin, end = map(int, bounds[sequence_index])
            anchor = begin + source_row
            horizon = 400
            if (anchor - HISTORY + 1 < begin
                    or anchor >= end
                    or row_runs[anchor] != run_id
                    or row_splits[anchor] != "validation"
                    or anchor + horizon >= end):
                raise ValueError("frozen start lacks its isolated history/continuation")
            references.append({
                "run_id": run_id, "anchor": anchor,
                "sequence_start": begin, "sequence_end": end,
            })
    if len(references) != 384:
        raise ValueError(f"expected 384 frozen development starts, got {len(references)}")
    history_u = np.stack([
        inputs[row["anchor"] - HISTORY + 1:row["anchor"] + 1]
        for row in references])
    history_y = np.stack([
        outputs[row["anchor"] - HISTORY + 1:row["anchor"] + 1]
        for row in references])
    future_u = np.stack([
        inputs[row["anchor"] + 1:row["anchor"] + 401]
        for row in references])
    truth = np.stack([
        outputs[row["anchor"] + 1:row["anchor"] + 401]
        for row in references])
    return (np.concatenate((history_u, history_y), axis=-1).astype(np.float32),
            np.concatenate((future_u, truth), axis=-1).astype(np.float32),
            references)


def _rollout_reference(model: Any, histories: np.ndarray,
                       future_inputs: np.ndarray) -> np.ndarray:
    norm = model.norm
    u_history = histories[..., :2]
    y_history = histories[..., 2:]
    u_norm = (u_history - np.asarray(norm.u0)) / np.asarray(norm.ustd)
    y_norm = (y_history - np.asarray(norm.y0)) / np.asarray(norm.ystd)
    hist = np.concatenate((u_norm.reshape(len(histories), -1),
                           y_norm.reshape(len(histories), -1)), axis=1)
    future_norm = ((future_inputs - np.asarray(norm.u0))
                   / np.asarray(norm.ustd))
    with torch.no_grad():
        state = model.encoder(torch.as_tensor(hist, dtype=torch.float32))
        outputs = []
        for index in range(future_norm.shape[1]):
            outputs.append(model.hn(state))
            state = model.fn(torch.cat((state, torch.as_tensor(
                future_norm[:, index], dtype=torch.float32)), dim=1))
    prediction = torch.stack(outputs, dim=1).cpu().numpy()
    return (prediction * np.asarray(norm.ystd)[None, None, :]
            + np.asarray(norm.y0)[None, None, :]).astype(np.float64)


def _run_macro_nrms(prediction: np.ndarray, truth: np.ndarray,
                    references: list[dict[str, Any]], train_std: np.ndarray,
                    horizon: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for run_id in sorted({row["run_id"] for row in references}):
        indices = [i for i, row in enumerate(references)
                   if row["run_id"] == run_id]
        error = prediction[indices, :horizon] - truth[indices, :horizon]
        rmse = np.sqrt(np.mean(error ** 2, axis=(0, 1)))
        result[run_id] = {
            "rmse_by_channel": rmse.tolist(),
            "nrms_by_channel": (rmse / train_std).tolist(),
            "macro_channel_nrms": float(np.mean(rmse / train_std)),
        }
    macro = float(np.mean([row["macro_channel_nrms"]
                           for row in result.values()]))
    return {"run_macro_mean": macro, "per_run": result}


def evaluate(reference_path: Path = REFERENCE_CHECKPOINT,
             native_path: Path = NATIVE_CHECKPOINT,
             report_path: Path = REPORT) -> dict[str, Any]:
    for path in (reference_path, DATASET, STARTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if report_path.exists():
        raise FileExistsError("refusing to replace a previous native SUBNET report")

    reference, native = _load_native_from_reference(reference_path)
    native_path.parent.mkdir(parents=True, exist_ok=True)
    if native_path.exists():
        loaded = SubnetBodyPlant.load_npz(native_path)
    else:
        native.save_npz(native_path)
        loaded = SubnetBodyPlant.load_npz(native_path)
    with np.load(DATASET, allow_pickle=False) as data:
        histories, future, references = _fixed_start_arrays(
            data, json.loads(STARTS.read_text(encoding="utf-8")))
        outputs = np.asarray(data["outputs"], dtype=np.float32)
        splits = data["split"].astype(str)
        train_std = outputs[splits == "train"].std(axis=0, ddof=0).astype(np.float64)
        if not np.isfinite(train_std).all() or np.any(train_std <= 0):
            raise ValueError("invalid training-only output scales")

    history_u = torch.as_tensor(histories[..., :2])
    history_y = torch.as_tensor(histories[..., 2:])
    future_u = torch.as_tensor(future[..., :2])
    z_ref = loaded.encode_history(history_u, history_y)
    native_prediction = loaded.rollout(z_ref, future_u).detach().cpu().numpy()
    reference_prediction = _rollout_reference(reference, histories, future[..., :2])
    truth = future[..., 2:].astype(np.float64)
    if not np.isfinite(native_prediction).all():
        raise FloatingPointError("native SUBNET generated non-finite output")

    # Check the public native API against deepSI's public simulation API too.
    import deepSI

    api_differences = []
    for index in (0, 64, 192, 320):
        row = references[index]
        probe_u = np.concatenate((histories[index, :, :2], future[index, :40, :2]))
        probe_y = np.concatenate((histories[index, :, 2:], truth[index, :40]))
        system_data = deepSI.System_data(
            u=probe_u, y=probe_y, dt=DT_S)
        with torch.no_grad():
            official = reference.apply_experiment(system_data).y[HISTORY:]
        api_differences.append(float(np.max(np.abs(
            official - reference_prediction[index, :40]))))

    horizons = {}
    for horizon in HORIZONS:
        official_stats = _run_macro_nrms(
            reference_prediction, truth, references, train_std, horizon)
        native_stats = _run_macro_nrms(
            native_prediction, truth, references, train_std, horizon)
        relative_error = abs(native_stats["run_macro_mean"]
                             - official_stats["run_macro_mean"])
        relative_error /= max(official_stats["run_macro_mean"], 1e-12)
        allowed = 0.05 if horizon == 40 else 0.10
        horizons[str(horizon)] = {
            "duration_s": horizon * DT_S,
            "reference": official_stats,
            "native": native_stats,
            "relative_nrms_difference": relative_error,
            "allowed_relative_difference": allowed,
            "passed": bool(relative_error <= allowed),
        }
    report = {
        "study": "repository-native SUBNET reproduction of frozen deepSI reference",
        "status": "passed" if all(v["passed"] for v in horizons.values())
            and max(api_differences) <= 1e-5 else "failed",
        "reference_checkpoint_sha256": _sha256(reference_path),
        "native_checkpoint_sha256": _sha256(native_path),
        "body_dataset_sha256": _sha256(DATASET),
        "evaluation_start_sha256": _sha256(STARTS),
        "frozen_start_count": len(references),
        "independent_validation_runs": len({row["run_id"] for row in references}),
        "native_vs_manual_reference_max_abs": float(np.max(np.abs(
            native_prediction - reference_prediction))),
        "official_deepSI_API_max_abs_on_probes": max(api_differences),
        "horizons": horizons,
        "test_or_practice_data_used": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--native", type=Path, default=NATIVE_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    result = evaluate(args.reference, args.native, args.report)
    print(json.dumps({
        "status": result["status"],
        "native_vs_reference_max_abs": result["native_vs_manual_reference_max_abs"],
        "official_api_probe_max_abs": result["official_deepSI_API_max_abs_on_probes"],
        "horizons": {key: {
            "reference_nrms": value["reference"]["run_macro_mean"],
            "native_nrms": value["native"]["run_macro_mean"],
            "relative_difference": value["relative_nrms_difference"],
            "passed": value["passed"],
        } for key, value in result["horizons"].items()},
    }, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
