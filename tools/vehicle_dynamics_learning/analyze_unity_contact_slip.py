#!/usr/bin/env python3
"""Check rigid-body contact kinematics against pinned-player WheelHit labels.

The input is development-only 1 kHz telemetry, not a competition topic.  The
reported slip relationship is a fitted/measured surrogate, not a claim to have
recovered Unity's private native WheelCollider solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    WHEEL_X_M,
    WHEEL_Y_M,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURES = (
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv",
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv",
)
WHEELS = ("fl", "fr", "rl", "rr")
SPEED_BINS = ((1.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.1))


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, value = line[2:].strip().split("=", 1)
            result[key] = value
    return result


def _rmse(error: np.ndarray) -> float | None:
    if not len(error):
        return None
    return float(np.sqrt(np.mean(np.square(error))))


def _score(error: np.ndarray, speed: np.ndarray,
           steering_abs: np.ndarray | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": int(error.size),
        "bias": float(np.mean(error)) if error.size else None,
        "rmse": _rmse(error),
        "p95_abs": float(np.quantile(np.abs(error), 0.95)) if error.size else None,
    }
    bins: dict[str, Any] = {}
    for low, high in SPEED_BINS:
        keep = (speed >= low) & (speed < high)
        bins[f"{low:g}-{high:g}_mps"] = {
            "n": int(np.count_nonzero(keep)),
            "rmse": _rmse(error[keep]),
        }
    result["speed_bins"] = bins
    if steering_abs is not None:
        steering_bins: dict[str, Any] = {}
        for low, high in ((0.0, 0.1), (0.1, 0.25),
                          (0.25, 0.4), (0.4, 0.55)):
            keep = (steering_abs >= low) & (steering_abs < high)
            steering_bins[f"{low:g}-{high:g}_rad"] = {
                "n": int(np.count_nonzero(keep)),
                "rmse": _rmse(error[keep]),
            }
        result["steering_bins"] = steering_bins
    return result


def _capture_report(path: Path, chunksize: int) -> dict[str, Any]:
    metadata = _metadata(path)
    com = np.asarray([float(value) for value in
                      metadata["center_of_mass_body"].split(",")])
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        wheel: {key: [] for key in (
            "speed", "lateral_angle", "lateral_ratio", "lateral_yaw_only",
            "lateral_planar", "longitudinal_error", "longitudinal_planar",
            "lateral_steer_body_sign", "lateral_steer_unity_sign",
            "lateral_steer_body_sign_flipped", "lateral_steer_unity_sign_flipped",
            "lateral_guide_ratio", "longitudinal_guide_ratio",
            "lateral_steer_guide", "longitudinal_steer_guide",
            "greybox_lateral_slip", "greybox_longitudinal_slip",
            "source_longitudinal_slip", "roll_pitch", "steering_abs")}
        for wheel in WHEELS
    }
    rows = 0
    for frame in pd.read_csv(path, comment="#", chunksize=chunksize):
        rows += len(frame)
        velocity = frame[["vel_body_x_mps", "vel_body_y_mps",
                          "vel_body_z_mps"]].to_numpy(dtype=np.float64)
        omega = frame[["omega_body_x_rps", "omega_body_y_rps",
                       "omega_body_z_rps"]].to_numpy(dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        # Unity local coordinates are (right, up, forward); yaw is about +Y.
        yaw_only = omega.copy()
        yaw_only[:, 0] = 0.0
        yaw_only[:, 2] = 0.0
        roll_pitch_rate = np.hypot(omega[:, 0], omega[:, 2])

        for wheel in WHEELS:
            grounded = frame[f"{wheel}_grounded"].to_numpy() > 0.5
            contact = frame[[f"{wheel}_contact_body_x",
                             f"{wheel}_contact_body_y",
                             f"{wheel}_contact_body_z"]].to_numpy(dtype=np.float64)
            relative = contact - com
            forward = frame[[f"{wheel}_contact_forward_body_x",
                             f"{wheel}_contact_forward_body_y",
                             f"{wheel}_contact_forward_body_z"]].to_numpy(dtype=np.float64)
            sideways = frame[[f"{wheel}_contact_sideways_body_x",
                              f"{wheel}_contact_sideways_body_y",
                              f"{wheel}_contact_sideways_body_z"]].to_numpy(dtype=np.float64)
            velocity_at_contact = velocity + np.cross(omega, relative)
            velocity_yaw_only = velocity + np.cross(yaw_only, relative)
            longitudinal = np.einsum("ij,ij->i", velocity_at_contact, forward)
            lateral = np.einsum("ij,ij->i", velocity_at_contact, sideways)
            lateral_yaw_only = np.einsum("ij,ij->i", velocity_yaw_only, sideways)
            longitudinal_yaw_only = np.einsum(
                "ij,ij->i", velocity_yaw_only, forward)
            # A planar model omits body heave and roll/pitch angular motion.
            # Keep the measured tire basis here so this ablation isolates the
            # missing 3-D contact-point velocity terms rather than conflating
            # them with steering-sign conventions. It is deliberately
            # favorable (GT state and contact frame), not a runtime estimator.
            planar_velocity = np.zeros_like(velocity)
            planar_velocity[:, 0] = velocity[:, 0]
            planar_velocity[:, 2] = velocity[:, 2]
            planar_omega = np.zeros_like(omega)
            planar_omega[:, 1] = omega[:, 1]
            planar_contact = planar_velocity + np.cross(planar_omega, relative)
            planar_forward_axis = forward.copy()
            planar_sideways_axis = sideways.copy()
            planar_forward_axis[:, 1] = 0.0
            planar_sideways_axis[:, 1] = 0.0
            planar_forward_axis /= np.maximum(
                np.linalg.norm(planar_forward_axis, axis=1)[:, None], 1e-12)
            planar_sideways_axis /= np.maximum(
                np.linalg.norm(planar_sideways_axis, axis=1)[:, None], 1e-12)
            planar_longitudinal = np.einsum(
                "ij,ij->i", planar_contact, planar_forward_axis)
            planar_lateral = np.einsum(
                "ij,ij->i", planar_contact, planar_sideways_axis)
            # Production-available frame hypothesis: use only measured wheel
            # steer angle (zero at the rear), body u/v/yaw, and the serialized
            # wheel lever arm. The two sign conventions are retained as
            # explicit ablations until checked against the Unity labels.
            u_body = velocity[:, 2]
            v_body = -velocity[:, 0]
            yaw_body = -omega[:, 1]
            wheel_x_forward = relative[:, 2]
            wheel_y_left = -relative[:, 0]
            point_forward = u_body - yaw_body * wheel_y_left
            point_left = v_body + yaw_body * wheel_x_forward
            wheel_steer = np.deg2rad(
                frame[f"{wheel}_steer_deg"].to_numpy(dtype=np.float64))
            steer_body = -wheel_steer
            steer_unity = wheel_steer
            steer_vel_body = (-np.sin(steer_body) * point_forward
                              + np.cos(steer_body) * point_left)
            steer_vel_unity = (-np.sin(steer_unity) * point_forward
                               + np.cos(steer_unity) * point_left)
            steer_forward_velocity = (
                np.cos(steer_body) * point_forward
                + np.sin(steer_body) * point_left)
            forward_slip = frame[f"{wheel}_forward_slip"].to_numpy(dtype=np.float64)
            sideways_slip = frame[f"{wheel}_sideways_slip"].to_numpy(dtype=np.float64)
            radius = float(metadata[f"{ {'fl':'front_left','fr':'front_right','rl':'rear_left','rr':'rear_right'}[wheel] }_radius_m"])
            wheel_speed = (frame[f"{wheel}_rpm"].to_numpy(dtype=np.float64) *
                           (2.0 * np.pi / 60.0) * radius)
            # Reproduce the prior planar greybox tire inputs using the exact
            # source wheel-center geometry and measured wheel steer. This
            # isolates its slip-coordinate mismatch from parameter fitting.
            wheel_index = WHEELS.index(wheel)
            point_forward_model = (u_body - yaw_body * WHEEL_Y_M[wheel_index])
            point_left_model = (v_body + yaw_body * WHEEL_X_M[wheel_index])
            model_forward = (np.cos(steer_body) * point_forward_model
                             + np.sin(steer_body) * point_left_model)
            model_lateral = (-np.sin(steer_body) * point_forward_model
                             + np.cos(steer_body) * point_left_model)
            model_denominator = np.maximum(np.abs(model_forward), 0.5)
            greybox_longitudinal = (
                (wheel_speed - model_forward) / model_denominator)
            greybox_lateral = model_lateral / model_denominator
            valid = (grounded & (speed > 1.0) & np.isfinite(longitudinal) &
                     np.isfinite(lateral) & np.isfinite(forward_slip) &
                     np.isfinite(sideways_slip) & np.isfinite(wheel_speed))
            denominator = np.abs(longitudinal) + 1.0
            # Candidate two-component slip map. Lateral slip is angular, hence
            # atan2 rather than a small-angle linear approximation.
            angle_prediction = np.arctan2(lateral, denominator)
            ratio_prediction = lateral / denominator
            longitudinal_prediction = (wheel_speed - longitudinal) / denominator
            source_longitudinal_prediction = (
                (wheel_speed - longitudinal)
                / (np.maximum(np.abs(wheel_speed), np.abs(longitudinal))
                   + 0.10054783962080052))
            guide_lateral_prediction = lateral / np.maximum(
                np.abs(longitudinal), 1.0e-12)
            guide_longitudinal_prediction = (
                (wheel_speed - longitudinal)
                / np.where(np.abs(longitudinal) > 1.0e-12,
                           longitudinal, np.nan))
            angle_error = angle_prediction - sideways_slip
            parts[wheel]["speed"].append(speed[valid])
            parts[wheel]["lateral_angle"].append(angle_error[valid])
            parts[wheel]["lateral_ratio"].append(
                (ratio_prediction - sideways_slip)[valid])
            parts[wheel]["lateral_yaw_only"].append(
                (np.arctan2(lateral_yaw_only, np.abs(longitudinal_yaw_only) + 1.0)
                 - sideways_slip)[valid])
            parts[wheel]["lateral_planar"].append(
                (np.arctan2(planar_lateral,
                            np.abs(planar_longitudinal) + 1.0)
                 - sideways_slip)[valid])
            parts[wheel]["longitudinal_error"].append(
                (longitudinal_prediction - forward_slip)[valid])
            parts[wheel]["longitudinal_planar"].append(
                ((wheel_speed - planar_longitudinal)
                 / (np.abs(planar_longitudinal) + 1.0) - forward_slip)[valid])
            parts[wheel]["lateral_guide_ratio"].append(
                (guide_lateral_prediction - sideways_slip)[valid])
            parts[wheel]["longitudinal_guide_ratio"].append(
                (guide_longitudinal_prediction - forward_slip)[valid])
            parts[wheel]["lateral_steer_body_sign"].append(
                (np.arctan2(steer_vel_body,
                            np.abs(point_forward) + 1.0)
                 - sideways_slip)[valid])
            parts[wheel]["lateral_steer_unity_sign"].append(
                (np.arctan2(steer_vel_unity,
                            np.abs(point_forward) + 1.0)
                 - sideways_slip)[valid])
            parts[wheel]["lateral_steer_body_sign_flipped"].append(
                (-np.arctan2(steer_vel_body,
                             np.abs(point_forward) + 1.0)
                 - sideways_slip)[valid])
            parts[wheel]["lateral_steer_unity_sign_flipped"].append(
                (-np.arctan2(steer_vel_unity,
                             np.abs(point_forward) + 1.0)
                 - sideways_slip)[valid])
            parts[wheel]["lateral_steer_guide"].append(
                (-steer_vel_body / np.maximum(np.abs(point_forward), 1.0e-12)
                 - sideways_slip)[valid])
            parts[wheel]["longitudinal_steer_guide"].append(
                ((wheel_speed - steer_forward_velocity)
                 / np.where(np.abs(steer_forward_velocity) > 1.0e-12,
                            steer_forward_velocity, np.nan)
                 - forward_slip)[valid])
            parts[wheel]["greybox_lateral_slip"].append(
                (greybox_lateral - sideways_slip)[valid])
            parts[wheel]["greybox_longitudinal_slip"].append(
                (greybox_longitudinal - forward_slip)[valid])
            parts[wheel]["source_longitudinal_slip"].append(
                (source_longitudinal_prediction - forward_slip)[valid])
            parts[wheel]["roll_pitch"].append(roll_pitch_rate[valid])
            parts[wheel]["steering_abs"].append(np.abs(np.deg2rad(
                frame[f"{wheel}_steer_deg"].to_numpy(dtype=np.float64)[valid])))

    wheel_results: dict[str, Any] = {}
    for wheel in WHEELS:
        values = {key: np.concatenate(chunks) for key, chunks in parts[wheel].items()}
        roll_threshold = float(np.quantile(values["roll_pitch"], 0.90))
        roll_top = values["roll_pitch"] >= roll_threshold
        wheel_results[wheel] = {
            "samples_above_1mps": int(len(values["speed"])),
            "lateral_slip_atan2_regularized": _score(
                values["lateral_angle"], values["speed"]),
            "lateral_slip_linear_ratio_comparator": _score(
                values["lateral_ratio"], values["speed"]),
            "lateral_slip_yaw_only_comparator": _score(
                values["lateral_yaw_only"], values["speed"]),
            "lateral_slip_planar_contact_velocity_candidate": _score(
                values["lateral_planar"], values["speed"]),
            "lateral_slip_guide_ratio_full_contact": _score(
                values["lateral_guide_ratio"], values["speed"]),
            "longitudinal_slip_guide_ratio_full_contact": _score(
                values["longitudinal_guide_ratio"], values["speed"]),
            "lateral_slip_steering_derived_candidates": {
                label: _score(values[key], values["speed"])
                for label, key in (
                    ("body_angle_left_positive", "lateral_steer_body_sign"),
                    ("unity_wheel_angle_positive", "lateral_steer_unity_sign"),
                    ("body_angle_left_positive_flipped", "lateral_steer_body_sign_flipped"),
                    ("unity_wheel_angle_positive_flipped", "lateral_steer_unity_sign_flipped"),
                )
            },
            "longitudinal_slip_candidate": _score(
                values["longitudinal_error"], values["speed"]),
            "greybox_planar_slip_candidate": {
                "longitudinal": _score(
                    values["greybox_longitudinal_slip"], values["speed"],
                    values["steering_abs"]),
                "lateral": _score(
                    values["greybox_lateral_slip"], values["speed"],
                    values["steering_abs"]),
            },
            "source_calibrated_longitudinal_slip_candidate": _score(
                values["source_longitudinal_slip"], values["speed"],
                values["steering_abs"]),
            "longitudinal_slip_planar_contact_velocity_candidate": _score(
                values["longitudinal_planar"], values["speed"]),
            "lateral_slip_guide_ratio_steering_reconstruction": _score(
                values["lateral_steer_guide"], values["speed"]),
            "longitudinal_slip_guide_ratio_steering_reconstruction": _score(
                values["longitudinal_steer_guide"], values["speed"]),
            "lateral_planar_error_by_abs_steering": {
                label: {
                    "n": int(np.count_nonzero(
                        (values["steering_abs"] >= low)
                        & (values["steering_abs"] < high))),
                    "rmse": _rmse(values["lateral_planar"][
                        (values["steering_abs"] >= low)
                        & (values["steering_abs"] < high)]),
                }
                for label, low, high in (
                    ("0_0.1rad", 0.0, 0.1),
                    ("0.1_0.25rad", 0.1, 0.25),
                    ("0.25_0.4rad", 0.25, 0.4),
                    ("0.4_0.55rad", 0.4, 0.55),
                )
            },
            "top_10pct_roll_pitch_rate_radps": {
                "threshold": roll_threshold,
                "n": int(np.count_nonzero(roll_top)),
                "full_3d_atan2_rmse": _rmse(values["lateral_angle"][roll_top]),
                "yaw_only_atan2_rmse": _rmse(values["lateral_yaw_only"][roll_top]),
            },
        }

    return {
        "capture": path.name,
        "file_bytes": path.stat().st_size,
        "unity_version": metadata.get("unity_version"),
        "fixed_delta_time_s": float(metadata["fixed_delta_time_s"]),
        "captured_rows": int(metadata["captured_samples"]),
        "overwritten_rows": int(metadata["overwritten_samples"]),
        "parsed_rows": rows,
        "center_of_mass_unity_xyz_m": [float(value) for value in com],
        "scoring_domain": "grounded WheelHit samples with Rigidbody COM speed > 1 m/s",
        "method": (
            "v_contact = v_COM + omega x (WheelHit.point - COM); project onto "
            "captured WheelHit.forwardDir/sidewaysDir. Candidate sideways slip "
            "is atan2(v_side, abs(v_forward)+1 m/s). Candidate longitudinal "
            "slip is (wheel_RPM*2*pi*radius/60-v_forward)/" 
            "(abs(v_forward)+1 m/s)."
        ),
        "interpretation": (
            "This comparison reconstructs the simulator-reported WheelHit "
            "telemetry only. It does not identify the slip coordinates used "
            "as inputs to the AutoDRIVE tire-force curves. The vehicle guide "
            "documents those separately as Sx=(R*omega-vx)/vx and "
            "Sy=vy/abs(vx); do not substitute these telemetry fits into the "
            "force model without independent force/rollout evidence."
        ),
        "per_wheel": wheel_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path,
                        default=DEFAULT_CAPTURES)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "live_runs/unity_exact_binary_audit_20261012/"
                        "contact_slip_mechanics_analysis_20261012.json")
    parser.add_argument("--chunksize", type=int, default=25000)
    args = parser.parse_args()
    reports = [_capture_report(path, args.chunksize) for path in args.captures]
    payload = {
        "title": "Pinned Unity player contact-slip mechanics check",
        "status": "offline development analysis; no runtime odometry change",
        "captures": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    for report in reports:
        print(report["capture"])
        for wheel, score in report["per_wheel"].items():
            lateral = score["lateral_slip_atan2_regularized"]
            linear = score["lateral_slip_linear_ratio_comparator"]
            yaw_only = score["lateral_slip_yaw_only_comparator"]
            longitudinal = score["longitudinal_slip_candidate"]
            print(f"  {wheel}: lateral atan RMSE={lateral['rmse']:.6g}, "
                  f"linear={linear['rmse']:.6g}, yaw-only={yaw_only['rmse']:.6g}; "
                  f"longitudinal RMSE={longitudinal['rmse']:.6g}")


if __name__ == "__main__":
    main()
