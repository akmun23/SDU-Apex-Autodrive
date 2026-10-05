#!/usr/bin/env python3
"""Build a provenance-aware racing ledger from saved simulator bags."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_practice_bag import analyze_structured  # noqa: E402


RACE_PREFIXES = ("practice_", "final_", "competition_")
MANIFEST_KEYS = (
    "git_sha",
    "map_sha256",
    "trajectory_sha256",
    "mpc_config_sha256",
    "odom_config_sha256",
    "amcl_config_sha256",
    "simulator_image_digest",
    "controller_image_id",
)


def load_run_manifest(run_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return None, list(MANIFEST_KEYS)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"manifest_error": str(exc)}, list(MANIFEST_KEYS)
    missing = [key for key in MANIFEST_KEYS if not manifest.get(key)]
    return manifest, missing


def bag_path_for(run_dir: Path) -> Path | None:
    for relative in (Path("run/run_0.db3"), Path("run/run/run_0.db3")):
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate
    return None


def collect_run(run_dir: Path) -> dict[str, Any] | None:
    bag = bag_path_for(run_dir)
    if bag is None:
        return None
    manifest, missing_manifest_keys = load_run_manifest(run_dir)
    record: dict[str, Any] = {
        "run_id": run_dir.name,
        "track": "practice" if run_dir.name.startswith("practice_") else "final",
        "bag": bag.relative_to(ROOT).as_posix(),
        "provenance_status": "complete" if not missing_manifest_keys else "incomplete",
        "missing_provenance_fields": missing_manifest_keys,
        "run_manifest": manifest,
    }
    try:
        metrics = analyze_structured(bag)
    except Exception as exc:
        record.update({"analysis_status": "failed", "analysis_error": str(exc)})
        return record

    scored = metrics["scored_laps_s"]
    metrics["scored_laps_s"] = [round(value, 6) for value in scored]
    record["analysis_status"] = "ok"
    record["metrics"] = metrics
    record["run_quality"] = {
        "ten_scored_laps": metrics["scored_lap_count"] == 10,
        "warmup_observed": metrics["warmup_lap_s"] is not None,
        "extra_lap_observed": metrics["extra_lap_s"] is not None,
        "zero_collisions": metrics["collision_delta"] == 0,
        "forty_hz_lidar": (
            metrics["lidar_header_timestamp"]["rate_hz"] is not None
            and abs(metrics["lidar_header_timestamp"]["rate_hz"] - 40.0) <= 1.0
        ),
    }
    record["comparison_eligible"] = (
        record["provenance_status"] == "complete"
        and all(record["run_quality"].values())
    )
    return record


def markdown_table(records: list[dict[str, Any]], track: str) -> list[str]:
    rows = [
        "| Run | Laps | Best (s) | Median (s) | Mean (s) | Collisions | LiDAR Hz | Provenance |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    selected = [
        item for item in records
        if item["track"] == track and item.get("analysis_status") == "ok"
    ]
    selected.sort(
        key=lambda item: (
            item["metrics"]["scored_mean_s"] is None,
            item["metrics"]["scored_mean_s"] or float("inf"),
        )
    )
    for item in selected:
        metrics = item["metrics"]
        rows.append(
            "| {run} | {count}/10 | {best} | {median} | {mean} | {collisions} | {hz} | {prov} |".format(
                run=item["run_id"],
                count=metrics["scored_lap_count"],
                best=_fmt(metrics["scored_best_s"]),
                median=_fmt(metrics["scored_median_s"]),
                mean=_fmt(metrics["scored_mean_s"]),
                collisions=metrics["collision_delta"],
                hz=_fmt(metrics["lidar_header_timestamp"]["rate_hz"], 2),
                prov=item["provenance_status"],
            )
        )
    return rows


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(ledger: dict[str, Any]) -> str:
    records = ledger["runs"]
    lines = [
        "# Racing performance ledger — 2026-10-05",
        "",
        f"Frozen source revision: `{ledger['source_git_sha']}`.",
        f"Saved run bags scanned: {ledger['summary']['run_count']}; "
        f"scored successfully: {ledger['summary']['scored_run_count']}; "
        f"complete provenance: {ledger['summary']['complete_provenance_count']}.",
        "",
        "A bag result without matching map, trajectory, MPC, odometry, "
        "localization, simulator-image, and controller-image hashes is historical "
        "evidence only; it is not an A/B promotion result.",
        "",
        "## External references captured from the handoff",
        "",
        "| Track | Reference best lap | Collisions |",
        "|---|---:|---:|",
        "| Short qualification track | 4.94 s | 0 |",
        "| Longer final track | 7.19 s | 0 |",
        "",
        "These are external benchmarks, not SDU results.",
        "",
        "## Practice-track runs",
        "",
        *markdown_table(records, "practice"),
        "",
        "## Final-track runs",
        "",
        *markdown_table(records, "final"),
        "",
        "## Interpretation",
        "",
        "The historical safe practice baseline is about 6.02 s/lap. Recent 2026-10-05 "
        "candidate bags include incomplete, collision-terminated, and unprovenanced "
        "runs; their apparent best laps must not be treated as repeatable improvements. "
        "The ledger preserves partial/collision runs instead of hiding them.",
        "",
    ]
    return "\n".join(lines)


def build_ledger(live_runs: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for run_dir in sorted(live_runs.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(RACE_PREFIXES):
            continue
        record = collect_run(run_dir)
        if record is not None:
            records.append(record)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "official_references": {
            "short_qualification_best_lap_s": 4.94,
            "short_qualification_collisions": 0,
            "long_final_best_lap_s": 7.19,
            "long_final_collisions": 0,
            "source_note": "Values quoted by the 2026-10-05 racing handoff.",
        },
        "summary": {
            "run_count": len(records),
            "scored_run_count": sum(
                item.get("analysis_status") == "ok" for item in records
            ),
            "complete_provenance_count": sum(
                item["provenance_status"] == "complete" for item in records
            ),
            "comparison_eligible_count": sum(
                item.get("comparison_eligible", False) for item in records
            ),
        },
        "runs": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-runs", type=Path, default=ROOT / "live_runs",
        help="directory containing top-level practice/final run folders",
    )
    parser.add_argument(
        "--json-output", type=Path,
        default=ROOT / "live_runs/racing_performance_ledger.json",
    )
    parser.add_argument(
        "--markdown-output", type=Path,
        default=ROOT / "docs/development/RACING_PERFORMANCE_LEDGER_20261005.md",
    )
    args = parser.parse_args()
    ledger = build_ledger(args.live_runs.resolve())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(ledger), encoding="utf-8")
    print(
        f"Wrote {args.json_output} and {args.markdown_output}: "
        f"{ledger['summary']['scored_run_count']}/{ledger['summary']['run_count']} "
        "bags scored; "
        f"{ledger['summary']['complete_provenance_count']} provenance-complete; "
        f"{ledger['summary']['comparison_eligible_count']} comparison-eligible."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
