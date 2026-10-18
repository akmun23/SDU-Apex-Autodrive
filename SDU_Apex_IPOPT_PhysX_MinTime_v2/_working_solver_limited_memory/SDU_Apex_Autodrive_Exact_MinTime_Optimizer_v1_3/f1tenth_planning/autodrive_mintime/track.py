from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import os
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class Track:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    left: np.ndarray
    right: np.ndarray
    ds: np.ndarray
    length: float

    @property
    def count(self) -> int:
        return len(self.s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "length_m": float(self.length),
            "spacing_m_mean": float(np.mean(self.ds)),
            "spacing_m_min": float(np.min(self.ds)),
            "spacing_m_max": float(np.max(self.ds)),
            "curvature_abs_max_m_inv": float(np.max(np.abs(self.kappa))),
            "left_min_m": float(np.min(self.left)),
            "right_min_m": float(np.min(self.right)),
        }


def _remove_duplicate_endpoint(xy: np.ndarray, *arrays: np.ndarray):
    if len(xy) > 1 and np.linalg.norm(xy[-1] - xy[0]) < 1e-5:
        xy = xy[:-1]
        arrays = tuple(a[:-1] for a in arrays)
    return (xy, *arrays)


def _closed_arclength(xy: np.ndarray):
    closed = np.vstack([xy, xy[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = np.ones(len(xy), dtype=bool)
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[i-1]) < 1e-7:
            keep[i] = False
    xy = xy[keep]
    closed = np.vstack([xy, xy[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(seg <= 1e-9):
        raise ValueError("Track contains zero-length segments")
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return xy, closed, s


def _periodic_interp(s_query, s_source, values, length):
    s0 = np.asarray(s_source, dtype=float)
    v0 = np.asarray(values, dtype=float)
    s_ext = np.concatenate([s0 - length, s0, s0 + length])
    v_ext = np.concatenate([v0, v0, v0])
    q = np.mod(np.asarray(s_query, dtype=float), length)
    return np.interp(q, s_ext, v_ext)


def build_track_from_xy_widths(
    xy: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    spacing_m: float,
) -> Track:
    xy = np.asarray(xy, dtype=float)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 8:
        raise ValueError("Need at least 8 Nx2 track points")
    if len(left) != len(xy) or len(right) != len(xy):
        raise ValueError("Width vectors must match track points")
    xy, left, right = _remove_duplicate_endpoint(xy, left, right)

    xy_clean, closed, s_closed = _closed_arclength(xy)
    if len(xy_clean) != len(xy):
        # nearest source widths after duplicate cleanup
        src = np.linspace(0.0, 1.0, len(left), endpoint=False)
        dst = np.linspace(0.0, 1.0, len(xy_clean), endpoint=False)
        left = np.interp(dst, src, left)
        right = np.interp(dst, src, right)
        xy = xy_clean
        closed = np.vstack([xy, xy[0]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s_closed = np.concatenate([[0.0], np.cumsum(seg)])

    length = float(s_closed[-1])
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    n = max(40, int(math.ceil(length / spacing_m)))
    s = np.linspace(0.0, length, n, endpoint=False)

    sx = CubicSpline(s_closed, closed[:, 0], bc_type="periodic")
    sy = CubicSpline(s_closed, closed[:, 1], bc_type="periodic")
    x = sx(s)
    y = sy(s)
    dx = sx(s, 1)
    dy = sy(s, 1)
    ddx = sx(s, 2)
    ddy = sy(s, 2)
    psi = np.unwrap(np.arctan2(dy, dx))
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12)
    kappa = (dx * ddy - dy * ddx) / denom

    src_s = s_closed[:-1]
    left_i = _periodic_interp(s, src_s, left, length)
    right_i = _periodic_interp(s, src_s, right, length)
    ds = np.diff(np.r_[s, length])
    return Track(s=s, x=x, y=y, psi=psi, kappa=kappa,
                 left=left_i, right=right_i, ds=ds, length=length)


def load_centerline_csv(path: str | Path, spacing_m: float) -> Track:
    path = Path(path)
    rows: list[list[float]] = []
    header = ""
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                header += line.lower()
                continue
            try:
                vals = [float(v.strip()) for v in line.split(",")]
            except ValueError:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    a = np.asarray(rows, dtype=float)

    if a.shape[1] >= 9:
        xy = a[:, 1:3]
        left = a[:, 7]
        right = a[:, 8]
    elif a.shape[1] >= 4:
        # TUM input convention is x,y,w_right,w_left.
        xy = a[:, 0:2]
        right = a[:, 2]
        left = a[:, 3]
    else:
        raise ValueError(f"Unsupported centerline CSV with {a.shape[1]} columns: {path}")

    return build_track_from_xy_widths(xy, left, right, spacing_m)


def export_track_csv(track: Track, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# s_m,x_m,y_m,psi_rad,kappa_radpm,d_left_m,d_right_m\n")
        for i in range(track.count):
            f.write(
                f"{track.s[i]:.8f},{track.x[i]:.8f},{track.y[i]:.8f},"
                f"{track.psi[i]:.9f},{track.kappa[i]:.9f},"
                f"{track.left[i]:.7f},{track.right[i]:.7f}\n"
            )


def ensure_centerline_from_repo(
    repo_root: str | Path,
    map_path: str | Path,
    output_dir: str | Path,
    env_overrides: dict[str, str] | None = None,
) -> Path:
    """Run the repository's tested map->centerline pipeline only, if needed."""
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    centerline = output_dir / "SmoothCenterline.csv"
    if centerline.exists():
        return centerline

    script = repo_root / "f1tenth_planning/scripts/optimize_trajectory.py"
    if not script.exists():
        raise FileNotFoundError(script)
    env = os.environ.copy()
    env.update({
        "MINTIME_MAP": str(Path(map_path).resolve()),
        "MINTIME_OUTPUT": str(output_dir),
        "MINTIME_CENTERLINE_ONLY": "1",
    })
    if env_overrides:
        env.update(env_overrides)
    subprocess.run([sys.executable, str(script)], cwd=str(repo_root), env=env, check=True)
    if not centerline.exists():
        raise RuntimeError(f"Centerline generator completed but did not create {centerline}")
    return centerline


def make_circle_track(radius_m: float = 5.0, half_width_m: float = 1.0,
                      spacing_m: float = 0.25) -> Track:
    length = 2.0 * math.pi * radius_m
    n = max(40, int(math.ceil(length / spacing_m)))
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xy = np.column_stack([radius_m * np.cos(th), radius_m * np.sin(th)])
    w = np.full(n, half_width_m)
    return build_track_from_xy_widths(xy, w, w, spacing_m)
