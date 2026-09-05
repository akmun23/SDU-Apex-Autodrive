#!/usr/bin/env python3
"""Replay the deterministic observer on exact source-timestamp packets."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sdu_apex_autodrive.odometry_analysis.packet_reconstruction import (
    reconstruct_packets,
)
from sdu_apex_autodrive.odometry_analysis.reference_observer import (
    ReferenceObserver,
)


def replay_python(packets: pd.DataFrame) -> pd.DataFrame:
    observer = ReferenceObserver()
    rows = []
    for _, packet in packets.iterrows():
        estimate = observer.update(packet)
        rows.append(estimate.__dict__)
    return pd.DataFrame(rows)


def write_cpp_input(packets: pd.DataFrame, path: Path) -> None:
    packets[[
        "stamp_s", "left_angle_rad", "right_angle_rad", "ax_mps2",
        "ay_mps2", "yaw_rate_radps", "yaw_rad"]].to_csv(path, index=False)


def score(predictions: pd.DataFrame, packets: pd.DataFrame) -> dict[str, float | int]:
    result = predictions.join(packets.drop(columns=["stamp_s"]).reset_index(drop=True))
    moving = np.isfinite(result.gt_speed_mps) & (result.gt_speed_mps >= 1.0)
    error = np.abs(result.loc[moving, "speed_mps"] - result.loc[moving, "gt_speed_mps"])
    relative = 100.0 * error / result.loc[moving, "gt_speed_mps"].to_numpy()
    output: dict[str, float | int] = {
        "samples": int(moving.sum()),
        "mae_mps": float(error.mean()) if len(error) else float("nan"),
        "p95_relative_error_pct": float(np.percentile(relative, 95)) if len(relative) else float("nan"),
        "p99_relative_error_pct": float(np.percentile(relative, 99)) if len(relative) else float("nan"),
        "fraction_le_2pct": float(np.mean(relative <= 2.0)) if len(relative) else float("nan"),
    }
    accel = result.loc[moving, "gt_longitudinal_accel_mps2"].to_numpy()
    for name, mask in {
        "acceleration": accel > 0.5,
        "steady": np.abs(accel) <= 0.5,
        "deceleration": accel < -0.5,
    }.items():
        if mask.any():
            output[f"{name}_p95_relative_error_pct"] = float(np.percentile(relative[mask], 95))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--cpp-replay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packets = reconstruct_packets(args.csv)
    python_result = replay_python(packets)
    report = {"csv": str(args.csv), "coherence": packets.attrs["coherence"]}
    report["metrics"] = score(python_result, packets)

    if args.cpp_replay:
        with tempfile.TemporaryDirectory(prefix="odometry_replay_") as temp:
            temp_path = Path(temp)
            input_path = temp_path / "packets.csv"
            cpp_path = temp_path / "cpp.csv"
            write_cpp_input(packets, input_path)
            subprocess.run(
                [str(args.cpp_replay), str(input_path), str(cpp_path)], check=True)
            cpp = pd.read_csv(cpp_path)
        compare = [
            ("speed_mps", "speed_mps"),
            ("body_u_mps", "body_u_mps"),
            ("body_v_mps", "body_v_mps"),
            ("x_m", "x_m"),
            ("y_m", "y_m"),
        ]
        report["parity"] = {
            left: float(np.max(np.abs(python_result[left] - cpp[right])))
            for left, right in compare
        }
        report["parity"]["wheel_update_used_mismatches"] = int(
            np.count_nonzero(
                python_result.wheel_update_used.to_numpy(dtype=bool) !=
                cpp.wheel_update_used.to_numpy(dtype=bool)))
        report["parity"]["turn_mode_mismatches"] = int(
            np.count_nonzero(
                python_result.turn_mode.to_numpy(dtype=bool) !=
                cpp.turn_mode.to_numpy(dtype=bool)))
        report["parity"]["reset_epoch_mismatches"] = int(
            np.count_nonzero(
                python_result.reset_epoch.to_numpy(dtype=bool) !=
                cpp.reset_epoch.to_numpy(dtype=bool)))
        report["parity"]["timing_degraded_mismatches"] = int(
            np.count_nonzero(
                python_result.timing_degraded.to_numpy(dtype=bool) !=
                cpp.timing_degraded.to_numpy(dtype=bool)))
        if any(value > 1.0e-9 for key, value in report["parity"].items()
               if key.endswith("mps") or key in {"x_m", "y_m"}):
            raise RuntimeError(f"Python/C++ replay mismatch: {report['parity']}")
        if any(report["parity"][key] for key in report["parity"]
               if key.endswith("mismatches")):
            raise RuntimeError(f"Python/C++ flag mismatch: {report['parity']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        python_result.join(
            packets.drop(columns=["stamp_s"]).reset_index(drop=True)).to_csv(
                args.output, index=False)
        report_path = args.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
