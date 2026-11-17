#!/usr/bin/env python3
"""Export captured controller cycles as frozen cases for `mpc_sqp_bench`.

Inputs: TRACK:DIAG.jsonl:RACELINE.csv
Output directory: outside f1tenth_mpc/, e.g. live_runs/cases/
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "cases": [], "sources": []}
    track_sources = {}
    for source in args.source:
        track, diagnostics, trajectory = source.split(":", 2)
        if track not in ("practice", "competition"):
            raise ValueError("track must be practice or competition")
        track_sources[track] = track_sources.get(track, 0) + 1
        prefix = track if track_sources[track] == 1 else f"{track}_{track_sources[track]}"
        records = []
        for index, line in enumerate(Path(diagnostics).open()):
            record = json.loads(line)
            p = record.get("payload", record)
            if p.get("frozen_source_problem") and p.get("nominal_controls"):
                records.append((index, record, p))
        if len(records) < args.per_track:
            raise ValueError(f"{track}: only {len(records)} captured records")
        points = np.loadtxt(trajectory, delimiter=",")
        if np.linalg.norm(points[0, 1:3] - points[-1, 1:3]) < 1.e-4 and len(points) > 3:
            points = points[:-1]
        lap_length = points[-1, 0] - points[0, 0] + np.linalg.norm(points[0, 1:3] - points[-1, 1:3])
        # Preserve the standing start, infeasible starts, both practice
        # hairpins, and distribute the rest across all observed progress bins.
        bins = {}
        for item in records:
            p = item[2]
            s = p["progress_m"]
            solver = p.get("solver", {})
            category = "corner"
            if p["state"][2] < 1:
                category = "standing_start"
            elif solver.get("source_exact_nlp_initial_source_feasible") is False:
                category = "infeasible"
            elif track == "practice" and (s % lap_length < 3 or 15.5 <= s % lap_length <= 18.5):
                category = "hairpin"
            bins.setdefault((category, int(s % lap_length)), []).append(item)
        chosen = []
        ordered = sorted(bins, key=lambda key: ({"infeasible": 0, "standing_start": 1,
                                                "hairpin": 2, "corner": 3}[key[0]], key[1]))
        while len(chosen) < args.per_track:
            for key in ordered:
                if bins[key] and len(chosen) < args.per_track:
                    chosen.append((key[0], bins[key].pop(0)))
        for category, (index, record, p) in chosen:
            name = f"{prefix}_{index:06d}"
            input_file = args.output / f"{name}.json"
            case_file = args.output / f"{name}.case"
            input_file.write_text(json.dumps(record))
            subprocess.run([str(args.exporter.resolve()), str(input_file), trajectory,
                            str(case_file)], check=True, capture_output=True)
            its = p.get("solver", {}).get("source_iterations", [])
            manifest["cases"].append({"file": case_file.name, "track": track, "source": prefix,
                "record": index, "category": category, "progress_m": p["progress_m"],
                "radius": its[0].get("radius", .5) if its else .5,
                "radius_source": "last attempted radius in first logged iteration; initial radius was not captured",
                "sha256": sha(case_file)})
        manifest["sources"].append({"track": track, "source": prefix, "diagnostics": str(Path(diagnostics).resolve()),
            "diagnostics_sha256": sha(diagnostics), "trajectory": str(Path(trajectory).resolve()),
            "trajectory_sha256": sha(trajectory)})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {len(manifest['cases'])} cases to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-track", type=int, default=30)
    return export(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
