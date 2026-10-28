#!/usr/bin/env python3
"""Blend a raw IPOPT time-stage path into a certified seed for Phase I.

The output is an intentionally modified initializer. It must pass the normal
strict Phase-I gate before any minimum-time candidate can be exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from path_frame import path_frame_numpy, SOURCE_REAR_AXLE_FROM_COM_M


def convert_com_path_to_rear_axle(fields, station, mesh):
    """Transport the old COM trajectory to the GPS point, then re-station it.

    Forces, angular motion and COM velocities retain their physical meaning.
    This is a coordinate conversion of an initializer, never a fitted model.
    """
    length = mesh[-1,0] + np.linalg.norm(mesh[0,1:3] - mesh[-1,1:3])
    track = SimpleNamespace(s=mesh[:,0], x=mesh[:,1], y=mesh[:,2],
                            psi=mesh[:,3], count=len(mesh), length=length)
    grid = np.r_[track.s, length]
    interp = lambda values: np.interp(station, grid, np.r_[values, values[0]])
    old_sin, old_cos = interp(np.sin(track.psi)), interp(np.cos(track.psi))
    yaw = np.arctan2(old_sin, old_cos) + fields['epsi_rad']
    lever = SOURCE_REAR_AXLE_FROM_COM_M * np.cos(fields['pitch_rad'])
    px = interp(track.x) - old_sin*fields['ey_m'] - lever*np.cos(yaw)
    py = interp(track.y) + old_cos*fields['ey_m'] - lever*np.sin(yaw)
    projected_s = station - lever*np.cos(fields['epsi_rad'])
    headings = np.unwrap(np.r_[track.psi, track.psi[0]])
    for _ in range(16):
        cx, cy, heading = path_frame_numpy(track, projected_s)
        c, s = np.cos(heading), np.sin(heading)
        ex, ey = px-cx, py-cy
        lateral = -s*ex + c*ey
        residual = c*ex + s*ey
        if np.max(np.abs(residual)) < 1.e-9:
            break
        index = np.searchsorted(grid, np.mod(projected_s,length), side='right')-1
        ds = np.diff(grid)[index]
        dx = np.diff(np.r_[track.x,track.x[0]])[index]/ds
        dy = np.diff(np.r_[track.y,track.y[0]])[index]/ds
        dh = np.diff(headings)[index]/ds
        metric = c*dx + s*dy - dh*lateral
        if np.any(metric < .25):
            raise ValueError('COM-to-rear conversion crosses a singular frame')
        projected_s += np.clip(residual/metric,-.5,.5)
    else:
        raise ValueError('COM-to-rear coordinate projection did not converge')
    if np.any(np.diff(projected_s) <= 0):
        raise ValueError('rear-axle initializer reverses along reference')
    values = dict(fields)
    values['ey_m'] = lateral
    values['epsi_rad'] = np.arctan2(np.sin(yaw-heading),np.cos(yaw-heading))
    q = np.r_[projected_s[-1]-length, projected_s, projected_s[0]+length]
    if np.any(np.diff(q) <= 0):
        raise ValueError('rear-axle initializer does not cover one periodic lap')
    def sample(values):
        if values.ndim == 1:
            return np.interp(station, q, np.r_[values[-1],values,values[0]])
        return np.vstack([sample(row) for row in values])
    return {name: sample(value) for name,value in values.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_seed(*, raw_stage: Path, base_seed: Path, output: Path,
              fraction: float, maximum_speed_mps: float = 16.0,
              full_state: bool = False, com_path_to_rear_axle: bool = False) -> dict:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if com_path_to_rear_axle and (not full_state or fraction != 1.0):
        raise ValueError('COM path conversion requires the complete raw state')
    if not np.isfinite(maximum_speed_mps) or maximum_speed_mps <= 0.35:
        raise ValueError("maximum speed must exceed 0.35 m/s")
    raw_stage = raw_stage.resolve(strict=True)
    base_seed = base_seed.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite seed: {output}")
    exploration_dir = raw_stage.parent.parent
    provenance = json.loads((exploration_dir / "provenance.json").read_text())
    recorded_seed = provenance.get("optimizer_checkpoint_seed", {}).get("path")
    if recorded_seed is None or Path(recorded_seed).resolve() != base_seed:
        raise ValueError("raw stage and certified seed lack matching provenance")
    with np.load(base_seed, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files
                   if name != "seed_modification_json"}
    metadata = json.loads(str(payload["metadata_json"].item()))
    if (metadata.get("map_sha256") != provenance.get("map_sha256")
            or metadata.get("centerline_sha256") !=
            provenance.get("centerline_sha256")
            or metadata.get("model_sha256") != _sha256_json(
                provenance.get("model", {}).get("model_metadata"))):
        raise ValueError("map, centerline, or model differs from certified seed")
    with np.load(raw_stage, allow_pickle=False) as archive:
        if int(archive["schema"]) != 1:
            raise ValueError("unsupported raw-stage schema")
        station = np.asarray(archive["sample_s"], dtype=np.float64)
        raw_x = np.asarray(archive["state_x"], dtype=np.float64)
        factor = float(archive["trust_factor"])
        violation = float(archive["max_constraint_violation"])
        lap_time = float(archive["predicted_lap_time_s"])
        raw_controls = ({name: np.asarray(archive[name], dtype=float)
                         for name in ('steering_rate_radps', 'throttle_norm')}
                        if full_state else {})
    if (station.shape != payload["seed_sample_s"].shape
            or not np.allclose(station, payload["seed_sample_s"],
                               rtol=0.0, atol=1.0e-9)
            or raw_x.shape != (16, station.size)
            or not np.isfinite(raw_x).all()
            or not np.isfinite([factor, violation, lap_time]).all()):
        raise ValueError("raw stage is nonfinite or not on the certified mesh")
    if (metadata.get("track_count") != station.size
            or provenance.get("optimizer_reference_track_mesh", {}).get(
                "nodes") != station.size):
        raise ValueError("raw stage has a different optimizer track")

    fields = {"ey_m": raw_x[0], "epsi_rad": raw_x[1],
              "u_mps": raw_x[2]}
    if full_state:
        # Exact archived 16-state layout; algebraic forces/loads are rebuilt
        # by the normal source projection before Phase I. No old suspension
        # or wheel state is silently spliced into the new fast trajectory.
        fields.update({
            'v_mps': raw_x[3], 'yaw_rate_radps': raw_x[4],
            'wheel_omega_radps': raw_x[5:9], 'steering_rad': raw_x[9],
            'body_height_m': raw_x[10], 'body_height_rate_mps': raw_x[11],
            'roll_rad': raw_x[12], 'roll_rate_radps': raw_x[13],
            'pitch_rad': raw_x[14], 'pitch_rate_radps': raw_x[15],
            **raw_controls,
        })
        if any(values.shape != payload[name].shape
               or not np.isfinite(values).all() for name, values in fields.items()):
            raise ValueError('raw full-state initializer has malformed states/controls')
    clipped_speed_samples = 0
    if com_path_to_rear_axle:
        mesh = np.loadtxt(exploration_dir/'optimizer_reference_track_mesh.csv',delimiter=',')
        fields = convert_com_path_to_rear_axle(fields, station, mesh)
    for name, raw_values in fields.items():
        original = np.asarray(payload[name], dtype=np.float64)
        blended = original + fraction * (raw_values - original)
        if name == "u_mps":
            bounded = np.clip(blended, 0.35, maximum_speed_mps)
            clipped_speed_samples = int(np.count_nonzero(bounded != blended))
            blended = bounded
        payload[name] = blended
    modification = {
        "kind": "exploratory_raw_geometry_and_speed_initializer",
        "modified_fields": list(fields),
        "full_differential_state_transferred": full_state,
        "com_path_converted_to_rear_axle": com_path_to_rear_axle,
        "fraction": fraction,
        "raw_stage_path": str(raw_stage),
        "raw_stage_sha256": _sha256(raw_stage),
        "raw_stage_trust_factor": factor,
        "raw_stage_predicted_lap_time_s": lap_time,
        "raw_stage_max_constraint_violation": violation,
        "clipped_speed_samples": clipped_speed_samples,
        "source_feasible": False,
        "requires_strict_phase1_recertification": True,
    }
    payload["seed_modification_json"] = np.asarray(json.dumps(
        modification, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return modification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-stage", required=True, type=Path)
    parser.add_argument("--base-seed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--maximum-speed-mps", type=float, default=16.0)
    parser.add_argument("--full-state", action="store_true",
                        help="transfer all archived differential states and controls")
    parser.add_argument("--com-path-to-rear-axle", action="store_true",
                        help="transport an old COM path to Unity GPS rear-axle coordinates")
    args = parser.parse_args()
    result = make_seed(raw_stage=args.raw_stage, base_seed=args.base_seed,
                       output=args.output, fraction=args.fraction,
                       maximum_speed_mps=args.maximum_speed_mps,
                       full_state=args.full_state,
                       com_path_to_rear_axle=args.com_path_to_rear_axle)
    print(json.dumps({"output": str(args.output.resolve()), **result},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
