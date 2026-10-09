#!/usr/bin/env python3
"""Join held-out unwind/reversal errors to exact-cell train-run support.

Ground-truth yaw is the supervised target only. The cell key uses causal
rear-wheel speed, measured steering feedback, and the command-intent event.
Test/final-test runs are rejected by the shared admission gate.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import audit_sensor_yaw_large_errors as audit
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import audit_sensor_yaw_large_errors as audit
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/unwind_reversal_train_support.json")
MIN_SAMPLES = atlas.MIN_EXPERT_SAMPLES
MIN_RUNS = atlas.MIN_EXPERT_RUNS
MIN_PER_RUN = atlas.MIN_SAMPLES_PER_RUN


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _key(event: str, speed: int, steering: int) -> tuple[str, int, int]:
    return event, int(speed), int(steering)


def run() -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    train_series = [row for row in admitted if row.split == "train"]
    counts: Counter[tuple[str, int, int]] = Counter()
    runs_by_key: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    for series in train_series:
        sensor_run = atlas._read_sensor_run(series)
        rows = atlas._rows(sensor_run, "command_intent")
        speed_cells = atlas._speed_cell(rows["wheel_speed"])
        steering_cells = atlas._steer_cell(rows["steering"])
        for event in ("unwind", "reversal"):
            mask = rows["event"] == event
            if not np.any(mask):
                continue
            pairs, pair_counts = np.unique(np.stack((
                speed_cells[mask].astype(np.int64),
                steering_cells[mask].astype(np.int64)), axis=1),
                axis=0, return_counts=True)
            for (speed, steering), amount in zip(pairs, pair_counts):
                key = _key(event, speed, steering)
                counts[key] += int(amount)
                runs_by_key[key][series.run_id] += int(amount)

    atlas_report = json.loads(audit.REPORT_PATH.read_text(encoding="utf-8"))
    error_report = json.loads((audit.OUTPUT_DIR / "sensor_yaw_large_error_audit.json")
                              .read_text(encoding="utf-8"))
    risk_cells = [row for row in error_report[
        "by_observable_speed_steering_event_cell"]
                  if row["event"] in ("unwind", "reversal")
                  and row["errors_over_0p1"] > 0]
    details = []
    by_event: dict[str, dict[str, int]] = {
        name: {"risk_cells": 0, "validation_rows": 0,
               "errors_over_0p1": 0, "train_unsupported_cells": 0,
               "train_supported_cells": 0}
        for name in ("unwind", "reversal")
    }
    for row in risk_cells:
        speed = int(atlas._speed_cell(np.asarray([
            row["wheel_speed_cell_center_mps"]], dtype=np.float32))[0])
        steering = int(atlas._steer_cell(np.asarray([
            row["steering_cell_center_rad"]], dtype=np.float32))[0])
        key = _key(row["event"], speed, steering)
        run_counts = runs_by_key.get(key, Counter())
        eligible_runs = {run_id: n for run_id, n in run_counts.items()
                         if n >= MIN_PER_RUN}
        eligible_samples = sum(eligible_runs.values())
        supported = (eligible_samples >= MIN_SAMPLES
                     and len(eligible_runs) >= MIN_RUNS)
        item = {
            **row,
            "training_samples_exact_sensor_cell_event_raw": int(counts[key]),
            "training_samples_after_per_run_minimum": int(eligible_samples),
            "training_run_count_exact_sensor_cell_event": len(run_counts),
            "training_runs_with_at_least_8_rows": len(eligible_runs),
            "exact_cell_meets_atlas_minimum": bool(supported),
            "support_gap_reason": (
                "supported_but_model_error" if supported else
                "no_training_rows" if counts[key] == 0 else
                "too_few_eligible_training_rows" if eligible_samples < MIN_SAMPLES else
                "insufficient_independent_training_runs"),
        }
        details.append(item)
        target = by_event[row["event"]]
        target["risk_cells"] += 1
        target["validation_rows"] += row["validation_rows"]
        target["errors_over_0p1"] += row["errors_over_0p1"]
        target["train_supported_cells"] += int(supported)
        target["train_unsupported_cells"] += int(not supported)

    summary_by_reason: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cells": 0, "validation_rows": 0, "errors_over_0p1": 0})
    for row in details:
        group = summary_by_reason[row["support_gap_reason"]]
        group["cells"] += 1
        group["validation_rows"] += row["validation_rows"]
        group["errors_over_0p1"] += row["errors_over_0p1"]
    details.sort(key=lambda row: (
        row["errors_over_0p1"] / max(row["validation_rows"], 1),
        row["errors_over_0p1"]), reverse=True)
    result = {
        "title": "Exact training support for held-out unwind/reversal error cells",
        "target_policy": "simulator-GT next yaw is label; never a feature or selector",
        "selector": "fixed-grid rear-wheel mean speed, measured steering feedback, command-intent event",
        "validation_split": "whole-run validation only; test/final-test sealed",
        "minimum_local_expert_support": {
            "samples": MIN_SAMPLES, "independent_runs": MIN_RUNS,
            "minimum_samples_per_run": MIN_PER_RUN,
        },
        "training_runs": len(train_series),
        "training_exact_cell_event_counts": by_event,
        "risk_cells_by_support_reason": dict(summary_by_reason),
        "highest_error_rate_cells": details,
        "source_audit": source_audit,
        "atlas_training_summary": atlas_report["training"],
        "interpretation": [
            "Unsupported exact cells indicate a candidate data-coverage gap, not by themselves a causal model explanation.",
            "Supported exact cells with high validation error are model/history failures; do not answer them by collecting the same stationary points again.",
            "Steering sign symmetry is not assumed; the table retains signed feedback cells.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                  default=_json_default) + "\n",
                      encoding="utf-8")
    for event, row in by_event.items():
        print(event, row)
    print("support categories", dict(summary_by_reason))
    print("wrote", OUTPUT)
    return result


if __name__ == "__main__":
    run()
