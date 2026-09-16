#!/usr/bin/env python3
"""Score model endpoints in the production raceline/Frenet frame.

The prediction CSV is deliberately simple and simulator-independent.  Each
row must contain predicted and ground-truth endpoint pose fields:

    pred_x_m,pred_y_m,pred_yaw_rad,true_x_m,true_y_m,true_yaw_rad

Optional fields ``horizon_s``, ``run`` and ``segment_id`` enable the same
per-horizon/per-run/per-segment reports used by the model fitter.  Ground
truth is used only offline to select the local raceline reference; the model
prediction itself is never replaced with ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_raceline_operating_envelope import (  # noqa: E402
    DEFAULT_RACELINE,
    _read_raceline,
    _stats,
)


DEFAULT_OUTPUT = (Path(__file__).resolve().parents[2] /
                  "sdu_apex_autodrive/artifacts/model_id_work/model_score.json")
SCORE_FIELDS = (
    "position_m", "e_cross_m", "e_along_m", "e_heading_rad",
    "corridor_margin_m", "normalized_corridor_error",
)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _raceline_arrays(rows: Sequence[dict[str, float]]) -> dict[str, np.ndarray]:
    return {
        field: np.asarray([
            row.get(field, 0.0) if field == "kappa_radpm" else row[field]
            for row in rows
        ], dtype=float)
        for field in ("s_m", "x_m", "y_m", "psi_rad", "kappa_radpm",
                      "d_left_m", "d_right_m")
    }


def _nearest_reference(x_m: float, y_m: float,
                       raceline: dict[str, np.ndarray]) -> int:
    distances = ((raceline["x_m"] - x_m) ** 2 +
                 (raceline["y_m"] - y_m) ** 2)
    return int(np.argmin(distances))


def frenet_error(pred_x_m: float, pred_y_m: float, pred_yaw_rad: float,
                 true_x_m: float, true_y_m: float, true_yaw_rad: float,
                 raceline: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Return signed local errors using the true endpoint's track reference."""
    reference_index = _nearest_reference(true_x_m, true_y_m, raceline)
    psi = float(raceline["psi_rad"][reference_index])
    dx = pred_x_m - true_x_m
    dy = pred_y_m - true_y_m
    cross = -math.sin(psi) * dx + math.cos(psi) * dy
    along = math.cos(psi) * dx + math.sin(psi) * dy
    margin = min(float(raceline["d_left_m"][reference_index]),
                 float(raceline["d_right_m"][reference_index]))
    margin = max(margin, 1.0e-6)
    return {
        "position_m": math.hypot(dx, dy),
        "e_cross_m": cross,
        "e_along_m": along,
        "e_heading_rad": _wrap(pred_yaw_rad - true_yaw_rad),
        "corridor_margin_m": margin,
        "normalized_corridor_error": abs(cross) / margin,
        "reference_index": reference_index,
        "reference_s_m": float(raceline["s_m"][reference_index]),
    }


def operating_class(x_m: float, y_m: float, speed_mps: float,
                    raceline: dict[str, np.ndarray],
                    core_bins: set[tuple[int, int]],
                    guard_bins: set[tuple[int, int]],
                    speed_bin_width_mps: float = 1.0,
                    curvature_bin_width_radpm: float = 0.025,
                    speed_ceiling_mps: float = 16.0,
                    corridor_tolerance_m: float = 0.20) -> str:
    """Classify a measured state against coupled raceline occupancy.

    The reference curvature and corridor are taken at the nearest raceline
    point.  Core/guard are occupied coupled bins, not independent maxima.
    """
    if not math.isfinite(speed_mps) or speed_mps < 0.0 or speed_mps > speed_ceiling_mps:
        return "stress"
    index = _nearest_reference(x_m, y_m, raceline)
    psi = float(raceline["psi_rad"][index])
    dx = x_m - float(raceline["x_m"][index])
    dy = y_m - float(raceline["y_m"][index])
    cross = -math.sin(psi) * dx + math.cos(psi) * dy
    left = float(raceline["d_left_m"][index])
    right = float(raceline["d_right_m"][index])
    if (cross > left + corridor_tolerance_m or
            cross < -right - corridor_tolerance_m):
        return "stress"
    speed_bin = math.floor(speed_mps / speed_bin_width_mps)
    curvature_bin = math.floor(
        abs(float(raceline["kappa_radpm"][index])) /
        curvature_bin_width_radpm)
    pair = (speed_bin, curvature_bin)
    if pair in core_bins:
        return "core"
    if pair in guard_bins:
        return "guard"
    return "stress"


def _group_stats(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        group = str(row.get(key, "unknown"))
        groups.setdefault(group, {field: [] for field in SCORE_FIELDS})
        for field in SCORE_FIELDS:
            groups[group][field].append(float(row[field]))
    return {
        group: {field: _stats(values) for field, values in fields.items()}
        for group, fields in sorted(groups.items())
    }


def score_rows(rows: Sequence[dict[str, Any]],
               raceline_rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    raceline = _raceline_arrays(raceline_rows)
    scored: list[dict[str, Any]] = []
    for row in rows:
        errors = frenet_error(
            float(row["pred_x_m"]), float(row["pred_y_m"]),
            float(row["pred_yaw_rad"]), float(row["true_x_m"]),
            float(row["true_y_m"]), float(row["true_yaw_rad"]), raceline)
        scored.append({**row, **errors})
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "offline_raceline_frenet_model_score",
        "primary_metric": "e_cross_m",
        "diagnostic_metric": "position_m",
        "corridor_normalization": (
            "abs(e_cross_m) / min(local raceline d_left_m, d_right_m); "
            "vehicle footprint and wall margin are not subtracted"),
        "sample_count": len(scored),
        "overall": {
            field: _stats([float(row[field]) for row in scored])
            for field in SCORE_FIELDS
        },
        "per_run": _group_stats(scored, "run"),
        "per_segment": _group_stats(scored, "segment_id"),
    }
    horizons = sorted({float(row["horizon_s"]) for row in scored
                       if row.get("horizon_s") not in (None, "")})
    result["horizons"] = {
        f"{horizon:.3f}s": {
            field: _stats([
                float(row[field]) for row in scored
                if abs(float(row.get("horizon_s")) - horizon) <= 1.0e-9
            ]) for field in SCORE_FIELDS
        }
        for horizon in horizons
    }
    result["promotion_horizon_s"] = 0.75
    result["promotion_rule"] = (
        "Use cross-track, heading, twist, and corridor-normalized metrics on "
        "CORE/GUARD; global Euclidean position remains diagnostic.")
    return result


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    required = {
        "pred_x_m", "pred_y_m", "pred_yaw_rad", "true_x_m", "true_y_m",
        "true_yaw_rad",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing prediction fields: {missing}")
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, start=2):
            row: dict[str, Any] = {}
            for field in required:
                try:
                    value = float(raw[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid {field}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{line_number}: non-finite {field}")
                row[field] = value
            for field in ("horizon_s", "run", "segment_id"):
                if raw.get(field, "") != "":
                    row[field] = raw[field]
            rows.append(row)
    return rows


def score(prediction_csv: Path, raceline_csv: Path = DEFAULT_RACELINE,
          output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    result = score_rows(_read_predictions(prediction_csv),
                        _read_raceline(raceline_csv))
    result["prediction_csv"] = str(prediction_csv)
    result["raceline_csv"] = str(raceline_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--raceline", type=Path, default=DEFAULT_RACELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = score(args.prediction_csv, args.raceline, args.output)
    print(json.dumps({
        "output": str(args.output),
        "sample_count": result["sample_count"],
        "cross_track_p95_m": result["overall"]["e_cross_m"]["p95"],
        "heading_p95_rad": result["overall"]["e_heading_rad"]["p95"],
        "position_diagnostic_p95_m": result["overall"]["position_m"]["p95"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
