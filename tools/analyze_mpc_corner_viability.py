#!/usr/bin/env python3
"""Attribute MPC scored-lap outliers to physical track sections and control events.

Offline only: reads reports and diagnostic JSONL, never runs in the control
callback and never publishes commands. Based on logged progress modulo a
specified lap length, not a guessed sector/time alignment.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median

DEFAULT_SECTORS = (0.0, 1.5, 3.5, 6.0, 10.0, 15.0, 21.0)


def summarize(run: Path, length: float, sectors=DEFAULT_SECTORS):
    report = json.loads((run / "mpc_run_report.json").read_text())
    intervals = report["per_official_lap_tracking"][1:]
    # Extend the final sector to precisely the supplied raceline length.
    edges = (*sectors, length)
    if not length > edges[-2] or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("track length must exceed and include ascending sector bounds")
    laps = []
    for idx, lap in enumerate(intervals, 1):
        laps.append({
            "scored_lap": idx,
            "time_s": lap["bag_receipt_end_s"] - lap["bag_receipt_start_s"],
            "start_s": lap["bag_receipt_start_s"],
            "end_s": lap["bag_receipt_end_s"],
            "sectors": [{"station_from_m": a, "station_to_m": b, "samples": 0,
                         "speed_sum": 0.0, "reference_sum": 0.0, "positive_underspeed_sum": 0.0,
                         "ey_sq_sum": 0.0, "epsi_sq_sum": 0.0,
                         "accepted_recoveries": 0, "line_search_four_trials": 0,
                         "command_steer_jumps_gt_0p08": 0}
                        for a, b in zip(edges, edges[1:])],
            "recoveries": []})
    if not laps:
        raise ValueError(f"No scored laps found in {run}")
    current_idx = 0
    previous_command = None
    with (run / "mpc_cycle_detail_diagnostics.jsonl").open() as f:
        for raw in f:
            data = json.loads(raw)
            t = data.get("bag_receipt_epoch_s")
            if not isinstance(t, (int, float)):
                continue
            while current_idx < len(laps) and t >= laps[current_idx]["end_s"]:
                current_idx += 1
                previous_command = None
            if current_idx >= len(laps):
                break
            lap = laps[current_idx]
            if t < lap["start_s"]:
                continue
            payload = data.get("payload", {})
            progress = payload.get("progress_m")
            state = payload.get("state")
            reference = payload.get("reference", {})
            if not (isinstance(progress, (int, float)) and math.isfinite(progress)
                    and isinstance(state, list) and len(state) >= 3
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in state[:3])
                    and isinstance(reference, dict)
                    and isinstance(reference.get("speed_reference_mps"), (int, float))):
                continue
            station = progress % length
            sector = next((s for s in lap["sectors"] if
                           s["station_from_m"] <= station < s["station_to_m"]), None)
            if sector is None:
                continue
            speed = float(state[2])
            target = float(reference["speed_reference_mps"])
            if not math.isfinite(target):
                continue
            sector["samples"] += 1
            sector["speed_sum"] += speed
            sector["reference_sum"] += target
            sector["positive_underspeed_sum"] += max(0.0, target - speed)
            sector["ey_sq_sum"] += float(state[0]) ** 2
            sector["epsi_sq_sum"] += float(state[1]) ** 2
            solver = payload.get("solver") or {}
            if solver.get("line_search_trials") == 4:
                sector["line_search_four_trials"] += 1
            status = payload.get("status") or ""
            if status == "accepted_viability_recovery":
                sector["accepted_recoveries"] += 1
                lap["recoveries"].append({
                    "station_m": round(station, 4),
                    "source_failure_stage": payload.get("nonlinear_failure_stage"),
                    "source_failure_reason": payload.get("nonlinear_failure_reason")})
            cmd = payload.get("issued_action")
            if status.startswith("accepted") and isinstance(cmd, dict) and all(
                    isinstance(cmd.get(key), (float, int)) for key in ("steering_rad", "throttle")):
                if previous_command and abs(float(cmd["steering_rad"]) - previous_command[0]) > 0.08:
                    sector["command_steer_jumps_gt_0p08"] += 1
                previous_command = (float(cmd["steering_rad"]), float(cmd["throttle"]))
    for lap in laps:
        for sec in lap["sectors"]:
            n = sec.pop("samples")
            sec["n"] = n
            for suffix, src in (("mean_actual_speed_mps", "speed_sum"),
                                ("mean_reference_speed_mps", "reference_sum"),
                                ("mean_positive_underspeed_mps", "positive_underspeed_sum")):
                sec[suffix] = round(sec.pop(src) / n, 5) if n else None
            for suffix, src in (("lateral_rmse_m", "ey_sq_sum"),
                                ("heading_rmse_rad", "epsi_sq_sum")):
                sec[suffix] = round(math.sqrt(sec.pop(src) / n), 5) if n else None
        lap["time_s"] = round(lap["time_s"], 6)
        lap.pop("start_s")
        lap.pop("end_s")
    times = [x["time_s"] for x in laps]
    return {"run": str(run), "track_length_m": length,
            "num_scored_laps": len(laps), "total_scored_s": round(sum(times), 6),
            "median_lap_s": round(median(times), 6), "worst_lap_s": max(times),
            "num_laps_at_least_5s": sum(t >= 5 for t in times), "laps": laps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--track-length-m", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir, args.track_length_m)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
