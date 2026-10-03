#!/usr/bin/env python3
"""Test whether a causal state-conditioned throttle response beats fixed alpha.

The actuator equation and one-packet command delay are held fixed. A small
linear surface changes only alpha using current/past observable state. The
surface is fit on training runs, evaluated with leave-one-whole-run-out
predictions within that training split, then frozen for validation and unseen
practice runs. It is a diagnostic; it does not modify a plant checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928/"
             "full_modeling_reset_20261001/"
             "replacement_offline_sim_raceline_20261002")
DEFAULT_DATASET = TASK_ROOT / "replacement_teacher_dataset_v1/openplane_dynamics.npz"
DEFAULT_PRACTICE_BENCHMARK = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/practice_transfer_benchmark_v1.json")
DEFAULT_OUTPUT = TASK_ROOT / "actuator_surface_diagnostic_v1/report.json"
RIDGE_FRACTION = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _transitions(data: dict[str, Any], state: np.ndarray
                 ) -> dict[str, np.ndarray]:
    """Extract only same-sequence transitions respecting the fitted delay=1."""
    frames = np.asarray(data["frames"], dtype=np.float64)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    family_source = data.get("training_families")
    if family_source is None:
        family_source = data.get("run_families")
    if family_source is None:
        family_source = np.asarray(data["run_ids"]).astype(str)
    families = np.asarray(family_source).astype(str)
    rows: dict[str, list[np.ndarray]] = {
        key: [] for key in ("run", "sequence", "split", "family", "current",
                            "target", "command", "features")}
    for sequence_id, ((start_raw, end_raw), run_raw) in enumerate(
            zip(data["bounds"], data["seq_run"])):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if end - start < 4:
            continue
        # k is the current feedback/state sample, k+1 is its target, and the
        # registered one-sample delay uses command[k-1]. The slew feature uses
        # command[k-2], never a future command or sensor value.
        k = np.arange(start + 2, end - 1, dtype=np.int64)
        command_index = k - 1
        current = frames[k, 4]
        target = frames[k + 1, 4]
        command = frames[command_index, 8]
        previous_command = frames[command_index - 1, 8]
        speed = np.hypot(state[k, 0], state[k, 1])
        steer = np.abs(state[k, 3])
        gap = command - current
        slew = command - previous_command
        mismatch = 0.5 * (state[k, 5] + state[k, 6]) - state[k, 0]
        speed_feature = (speed - 6.0) / 6.0
        steer_feature = steer / 0.5236
        command_feature = command / 0.5
        gap_feature = gap / 0.5
        slew_feature = slew / 0.05
        mismatch_feature = mismatch / 5.0
        features = np.column_stack((
            speed_feature, steer_feature, command_feature, gap_feature,
            slew_feature, mismatch_feature,
            speed_feature * gap_feature,
            steer_feature * gap_feature,
        ))
        count = len(k)
        rows["run"].append(np.full(count, run, dtype=np.int32))
        rows["sequence"].append(np.full(count, sequence_id, dtype=np.int32))
        rows["split"].append(np.full(count, splits[run]))
        rows["family"].append(np.full(count, families[run]))
        rows["current"].append(current)
        rows["target"].append(target)
        rows["command"].append(command)
        rows["features"].append(features)
    if not rows["current"]:
        raise ValueError("dataset has no usable throttle transitions")
    result = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
    result["run_name"] = run_ids[result["run"]]
    return result


def _weights(rows: dict[str, np.ndarray], selected: np.ndarray,
             data: dict[str, Any]) -> np.ndarray:
    """Equalize sequences then runs, preserving registered family weights."""
    run_values = rows["run"][selected]
    sequence_values = rows["sequence"][selected]
    family_values = rows["family"][selected].astype(str)
    available_runs = np.unique(run_values)
    family_names = np.asarray(data.get("training_family_names", []))
    family_prob = np.asarray(data.get("training_family_probabilities", []),
                             dtype=np.float64)
    registered = dict(zip(family_names.astype(str), family_prob.tolist()))
    families = sorted(set(family_values.tolist()))
    family_mass = {family: registered.get(family, 0.0) for family in families}
    if sum(family_mass.values()) <= 0.0:
        family_mass = {family: 1.0 for family in families}
    total_mass = sum(family_mass.values())
    family_mass = {family: value / total_mass
                   for family, value in family_mass.items()}
    run_family = {int(run): str(rows["family"][np.flatnonzero(
        selected & (rows["run"] == run))[0]]) for run in available_runs}
    runs_per_family = {
        family: sum(value == family for value in run_family.values())
        for family in families
    }
    weights = np.zeros(int(selected.sum()), dtype=np.float64)
    for run in available_runs:
        local = run_values == run
        sequence_ids, sequence_counts = np.unique(sequence_values[local],
                                                  return_counts=True)
        run_share = (family_mass[run_family[int(run)]]
                     / runs_per_family[run_family[int(run)]])
        for sequence_id, count in zip(sequence_ids, sequence_counts):
            weights[local & (sequence_values == sequence_id)] = (
                run_share / len(sequence_ids) / int(count))
    weights /= weights.sum()
    return weights


def _fit(rows: dict[str, np.ndarray], selected: np.ndarray,
         data: dict[str, Any]) -> dict[str, Any]:
    if int(selected.sum()) < 100:
        raise ValueError("too few transitions to fit throttle response")
    current, target = rows["current"][selected], rows["target"][selected]
    command = rows["command"][selected]
    gap = command - current
    change = target - current
    weights = _weights(rows, selected, data)
    denominator = float(np.sum(weights * gap * gap))
    alpha = float(np.clip(np.sum(weights * gap * change) / denominator,
                          0.0, 1.0))
    features = rows["features"][selected]
    mean = np.sum(features * weights[:, None], axis=0)
    variance = np.sum((features - mean) ** 2 * weights[:, None], axis=0)
    scale = np.maximum(np.sqrt(variance), 1e-6)
    normalized = (features - mean) / scale
    design = normalized * gap[:, None]
    residual = change - alpha * gap
    gram = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * residual)
    ridge = RIDGE_FRACTION * float(np.trace(gram)) / gram.shape[0]
    beta = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    return {"alpha": alpha, "feature_mean": mean, "feature_scale": scale,
            "beta": beta, "ridge": ridge}


def _predict(rows: dict[str, np.ndarray], selected: np.ndarray,
             fit: dict[str, Any], surface: bool) -> np.ndarray:
    current = rows["current"][selected]
    gap = rows["command"][selected] - current
    alpha = np.full(len(gap), fit["alpha"], dtype=np.float64)
    if surface:
        features = rows["features"][selected]
        alpha += (((features - fit["feature_mean"])
                   / fit["feature_scale"]) @ fit["beta"])
    alpha = np.clip(alpha, 0.0, 1.0)
    return current + alpha * gap


def _metrics(rows: dict[str, np.ndarray], selected: np.ndarray,
             fit: dict[str, Any]) -> dict[str, Any]:
    target = rows["target"][selected]
    names = rows["run_name"][selected]
    truth_fit = _predict(rows, selected, fit, surface=False)
    surface_fit = _predict(rows, selected, fit, surface=True)
    by_run: dict[str, Any] = {}
    for name in sorted(set(names.tolist())):
        mask = names == name
        pair = {}
        for model_name, prediction in (("fixed_alpha", truth_fit),
                                       ("state_surface", surface_fit)):
            error = prediction[mask] - target[mask]
            pair[model_name] = {
                "transitions": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        pair["rmse_delta_surface_minus_fixed"] = (
            pair["state_surface"]["rmse"] - pair["fixed_alpha"]["rmse"])
        by_run[name] = pair
    run_names = list(by_run)
    fixed = np.asarray([by_run[name]["fixed_alpha"]["rmse"]
                        for name in run_names])
    surface = np.asarray([by_run[name]["state_surface"]["rmse"]
                          for name in run_names])
    return {
        "independent_whole_runs": len(run_names),
        "macro_run_rmse_fixed_alpha": float(np.mean(fixed)),
        "macro_run_rmse_state_surface": float(np.mean(surface)),
        "macro_run_rmse_delta": float(np.mean(surface - fixed)),
        "runs_improved": int(np.count_nonzero(surface < fixed)),
        "per_run": by_run,
    }


def _bootstrap_ci(rows: dict[str, np.ndarray], selected: np.ndarray,
                  fit: dict[str, Any], seed: int) -> list[float] | None:
    names = sorted(set(rows["run_name"][selected].tolist()))
    if len(names) < 2:
        return None
    metric = _metrics(rows, selected, fit)["per_run"]
    deltas = np.asarray([metric[name]["rmse_delta_surface_minus_fixed"]
                         for name in names])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(10000, len(deltas)))
    samples = deltas[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _train_oof(rows: dict[str, np.ndarray], data: dict[str, Any]
               ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    train_mask = rows["split"] == "train"
    runs = sorted(set(rows["run"][train_mask].tolist()))
    if len(runs) < 3:
        raise ValueError("whole-run training diagnostic requires >=3 runs")
    fixed, surface, oof_fits = [], [], {}
    for held_run in runs:
        selected = train_mask & (rows["run"] != held_run)
        fit = _fit(rows, selected, data)
        held = train_mask & (rows["run"] == held_run)
        truth = rows["target"][held]
        fixed_error = _predict(rows, held, fit, surface=False) - truth
        surface_error = _predict(rows, held, fit, surface=True) - truth
        fixed.append(fixed_error)
        surface.append(surface_error)
        oof_fits[held_run] = {"fit": fit, "held_mask": held}
    report_by_run = {}
    for held_run, fit_entry in oof_fits.items():
        held = fit_entry["held_mask"]
        truth = rows["target"][held]
        run_name = str(rows["run_name"][held][0])
        pair = {}
        for name, pred in (("fixed_alpha", _predict(
                rows, held, fit_entry["fit"], surface=False)),
                           ("state_surface", _predict(
                rows, held, fit_entry["fit"], surface=True))):
            error = pred - truth
            pair[name] = {"transitions": int(held.sum()),
                          "rmse": float(np.sqrt(np.mean(error ** 2))),
                          "mae": float(np.mean(np.abs(error))),
                          "bias": float(np.mean(error))}
        pair["rmse_delta_surface_minus_fixed"] = (
            pair["state_surface"]["rmse"] - pair["fixed_alpha"]["rmse"])
        report_by_run[run_name] = pair
    delta = np.asarray([row["rmse_delta_surface_minus_fixed"]
                        for row in report_by_run.values()])
    rng = np.random.default_rng(730020)
    samples = delta[rng.integers(0, len(delta),
                                 size=(10000, len(delta)))].mean(axis=1)
    return ({
        "method": "leave-one-whole-training-run-out; every fit excludes scored run",
        "independent_whole_runs": len(runs),
        "macro_run_rmse_fixed_alpha": float(np.mean([
            row["fixed_alpha"]["rmse"] for row in report_by_run.values()])),
        "macro_run_rmse_state_surface": float(np.mean([
            row["state_surface"]["rmse"] for row in report_by_run.values()])),
        "macro_run_rmse_delta": float(np.mean(delta)),
        "runs_improved": int(np.count_nonzero(delta < 0.0)),
        "per_run": report_by_run,
        "feature_names": ["speed", "abs_steering", "delayed_command",
                          "command_minus_feedback", "command_slew",
                          "signed_wheel_body_mismatch",
                          "speed_x_command_gap", "steering_x_command_gap"],
        "ridge_fraction_of_mean_gram_diagonal": RIDGE_FRACTION,
        "run_bootstrap_95pct_ci_macro_rmse_delta": [
            float(value) for value in np.quantile(samples, (0.025, 0.975))],
    }, oof_fits)


def diagnose(dataset_path: Path, practice_dataset_path: Path,
             practice_benchmark_path: Path) -> dict[str, Any]:
    data = _load_dataset(dataset_path)
    if int(data["schema_version"]) != 9:
        raise ValueError("expected frozen schema-9 replacement teacher data")
    state = physical_state_from_dataset(data).astype(np.float64)
    train_rows = _transitions(data, state)
    oof, _ = _train_oof(train_rows, data)
    all_train = train_rows["split"] == "train"
    frozen_fit = _fit(train_rows, all_train, data)
    dynamic_reports = {}
    for split in ("validation",):
        selected = train_rows["split"] == split
        if np.any(selected):
            dynamic_reports[split] = _metrics(train_rows, selected, frozen_fit)
            dynamic_reports[split]["run_bootstrap_95pct_ci_macro_rmse_delta"] = (
                _bootstrap_ci(train_rows, selected, frozen_fit, 730021))

    practice_benchmark = json.loads(practice_benchmark_path.read_text(
        encoding="utf-8"))
    if (practice_benchmark.get("benchmark_id")
            != "practice_transfer_benchmark_v1"
            or practice_benchmark.get("frozen") is not True
            or practice_benchmark.get("training_or_checkpoint_selection_use")
            is not False):
        raise ValueError("practice source is not the frozen unseen benchmark")
    if _sha256(practice_dataset_path) != practice_benchmark["dataset_sha256"]:
        raise ValueError("practice dataset hash differs from frozen benchmark")
    practice_data = _load_dataset(practice_dataset_path)
    practice_state = physical_state_from_dataset(practice_data).astype(np.float64)
    practice_rows = _transitions(practice_data, practice_state)
    practice_mask = practice_rows["split"] == "unseen_practice"
    if not np.any(practice_mask):
        raise ValueError("frozen practice dataset has no unseen practice rows")
    practice_report = _metrics(practice_rows, practice_mask, frozen_fit)
    practice_report["run_bootstrap_95pct_ci_macro_rmse_delta"] = _bootstrap_ci(
        practice_rows, practice_mask, frozen_fit, 730022)

    # Diagnose support where recursive replay showed its largest u bias.
    subset_reports = {}
    # _transitions stores run/sequence-synchronous feature covariates; recover
    # speed from the explicitly scaled first feature for support accounting.
    speed = train_rows["features"][:, 0] * 6.0 + 6.0
    command = train_rows["command"]
    train_rows_mask = train_rows["split"] == "train"
    low_moving = (train_rows_mask & (speed >= 3.0) & (speed <= 8.0)
                  & (command <= 0.05))
    subset_reports["train_moving_low_throttle"] = {
        "transitions": int(low_moving.sum()),
        "independent_runs": int(len(set(train_rows["run_name"][low_moving]))),
        "run_ids": sorted(set(train_rows["run_name"][low_moving].tolist())),
    }
    return {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "practice_dataset": str(practice_dataset_path.resolve()),
        "practice_dataset_sha256": _sha256(practice_dataset_path),
        "practice_benchmark": str(practice_benchmark_path.resolve()),
        "practice_benchmark_sha256": _sha256(practice_benchmark_path),
        "equation": "throttle[k+1]=throttle[k]+alpha(x[k])*(command[k-1]-throttle[k])",
        "causal_features": ["current body speed", "current absolute steering",
                            "delayed throttle command", "command-feedback gap",
                            "past command slew", "signed wheel/body mismatch"],
        "feature_description": "A low-capacity linear alpha surface with bounded alpha; all scaling and coefficients fit on training runs only. No future sensor/truth inputs.",
        "baseline_alpha_selection": "same weighted training-only scalar alpha as baseline; delay fixed at one 25 ms packet",
        "targeted_revision_is_checkpoint": False,
        "training_whole_run_oof": oof,
        "dynamic_validation": dynamic_reports,
        "unseen_practice": practice_report,
        "training_support": subset_reports,
        "frozen_surface_fit": {
            "alpha_at_center": float(frozen_fit["alpha"]),
            "feature_mean": frozen_fit["feature_mean"].tolist(),
            "feature_scale": frozen_fit["feature_scale"].tolist(),
            "beta": frozen_fit["beta"].tolist(),
            "ridge": float(frozen_fit["ridge"]),
        },
        "interpretation_limit": "This tests only one-step actuator feedback. It does not establish better recursive body, wheel, pose, or full-lap accuracy; any accepted actuator revision must still be used in the complete plant and scored on frozen whole runs.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-benchmark", type=Path,
                        default=DEFAULT_PRACTICE_BENCHMARK)
    parser.add_argument("--practice-dataset", type=Path,
                        help="defaults to the frozen benchmark's dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark = json.loads(args.practice_benchmark.read_text(encoding="utf-8"))
    practice_dataset = args.practice_dataset or Path(str(
        benchmark["dataset"]).replace("/workspace", str(ROOT)))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    report = diagnose(args.dataset.resolve(), practice_dataset.resolve(),
                      args.practice_benchmark.resolve())
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "training_oof": report["training_whole_run_oof"],
        "dynamic_validation": report["dynamic_validation"],
        "unseen_practice": report["unseen_practice"],
        "training_support": report["training_support"],
        "frozen_surface_fit": report["frozen_surface_fit"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
