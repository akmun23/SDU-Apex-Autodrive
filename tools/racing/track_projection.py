"""Polyline projection utilities for race-track/Frenet analysis."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class Projection:
    s_m: float
    lateral_offset_m: float
    heading_error_rad: float
    segment_index: int
    curvature_inv_m: float
    left_width_m: float
    right_width_m: float
    projected_x_m: float
    projected_y_m: float
    distance_m: float
    segment_fraction: float


class TrackProjection:
    """Project poses onto a sampled open or closed track polyline.

    Positive lateral offset is to the left of the path direction.  If a
    previous segment is supplied, ``local_search_radius`` limits projection
    to its neighbourhood and preserves the correct branch at nearby track
    crossings.  Closed-path progress is reported in ``[0, total_length)``.
    """

    def __init__(
        self,
        x_m: Sequence[float],
        y_m: Sequence[float],
        *,
        s_m: Sequence[float] | None = None,
        heading_rad: Sequence[float] | None = None,
        curvature_inv_m: Sequence[float] | None = None,
        left_width_m: Sequence[float] | None = None,
        right_width_m: Sequence[float] | None = None,
        closed: bool = False,
        total_length_m: float | None = None,
    ) -> None:
        if len(x_m) != len(y_m) or len(x_m) < 2:
            raise ValueError("track needs matching x/y arrays with at least two points")
        self.x = [float(value) for value in x_m]
        self.y = [float(value) for value in y_m]
        if not all(math.isfinite(value) for value in self.x + self.y):
            raise ValueError("track coordinates must be finite")
        self.closed = bool(closed)

        supplied_s = None if s_m is None else [float(value) for value in s_m]
        self.heading = self._optional_values(heading_rad, "heading")
        self.curvature = self._optional_values(curvature_inv_m, "curvature", 0.0)
        self.left_width = self._optional_values(left_width_m, "left width", math.nan)
        self.right_width = self._optional_values(right_width_m, "right width", math.nan)

        # Some centerline files explicitly repeat the first vertex at the end.
        # Drop that duplicate while retaining its final s as the loop length.
        repeated_endpoint_length = None
        if self.closed and math.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]) < 1e-10:
            if supplied_s is not None:
                repeated_endpoint_length = supplied_s[-1]
            self.x.pop()
            self.y.pop()
            for values in (self.heading, self.curvature, self.left_width, self.right_width):
                values.pop()
            if supplied_s is not None:
                supplied_s.pop()

        if supplied_s is None:
            self.s = [0.0]
            for index in range(len(self.x) - 1):
                self.s.append(self.s[-1] + self._distance(index, index + 1))
        else:
            if len(supplied_s) != len(self.x):
                raise ValueError("s array length must match track points")
            if not all(math.isfinite(value) for value in supplied_s):
                raise ValueError("s values must be finite")
            if any(b <= a for a, b in zip(supplied_s, supplied_s[1:])):
                raise ValueError("s values must be strictly increasing")
            self.s = supplied_s

        if len(self.x) < 2:
            raise ValueError("track needs at least two distinct points")
        if self.closed:
            inferred_length = repeated_endpoint_length
            if inferred_length is None:
                inferred_length = self.s[-1] + self._distance(len(self.x) - 1, 0)
            self.total_length = float(total_length_m or inferred_length)
            if not math.isfinite(self.total_length) or self.total_length <= self.s[-1]:
                raise ValueError("closed track length must exceed its final s value")
        else:
            self.total_length = self.s[-1]
            if total_length_m is not None and not math.isclose(
                float(total_length_m), self.total_length, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("total_length_m only applies to a closed track")

        self.segment_count = len(self.x) if self.closed else len(self.x) - 1
        self.segment_lengths = [
            self._distance(i, (i + 1) % len(self.x))
            for i in range(self.segment_count)
        ]
        if any(length <= 0.0 for length in self.segment_lengths):
            raise ValueError("track contains a zero-length segment")

    def _optional_values(
        self, values: Sequence[float] | None, label: str, default: float = math.nan
    ) -> list[float]:
        if values is None:
            return [default] * len(self.x)
        if len(values) != len(self.x):
            raise ValueError(f"{label} array length must match track points")
        result = [float(value) for value in values]
        if any(not math.isfinite(value) for value in result):
            raise ValueError(f"{label} values must be finite")
        return result

    def _distance(self, first: int, second: int) -> float:
        return math.hypot(self.x[second] - self.x[first], self.y[second] - self.y[first])

    @classmethod
    def from_csv(cls, path: str | Path, *, closed: bool = True) -> "TrackProjection":
        """Load the repository's comment-header trajectory/centerline CSV."""
        source = Path(path)
        lines = source.read_text(encoding="utf-8").splitlines()
        header_line = next((line.lstrip()[1:].strip() for line in lines if line.lstrip().startswith("#")), None)
        data_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if not data_lines:
            raise ValueError(f"no track rows in {source}")
        if header_line:
            names = [name.strip() for name in header_line.split(",")]
        else:
            names = [name.strip() for name in data_lines[0].split(",")]
            data_lines = data_lines[1:]
        rows = csv.DictReader(data_lines, fieldnames=names)

        def column(*names: str) -> list[float] | None:
            key = next((name for name in names if name in rows.fieldnames), None)
            if key is None:
                return None
            try:
                return [float(row[key]) for row in rows_copy]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric {key} column in {source}") from exc

        rows_copy = list(rows)
        x = column("x_m", "x")
        y = column("y_m", "y")
        if x is None or y is None:
            raise ValueError(f"{source} must contain x_m/y_m or x/y columns")
        return cls(
            x,
            y,
            s_m=column("s_m", "s"),
            heading_rad=column("psi_rad", "heading_rad", "yaw_rad"),
            curvature_inv_m=column("kappa_radpm", "curvature_inv_m", "curvature"),
            left_width_m=column(
                "d_left_m", "left_width_m", "left_width", "w_tr_left_m"
            ),
            right_width_m=column(
                "d_right_m", "right_width_m", "right_width", "w_tr_right_m"
            ),
            closed=closed,
        )

    def project(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        *,
        previous_segment: int | None = None,
        local_search_radius: int | None = None,
    ) -> Projection:
        if not all(math.isfinite(value) for value in (x_m, y_m, yaw_rad)):
            raise ValueError("query pose must be finite")
        if local_search_radius is not None and local_search_radius < 0:
            raise ValueError("local_search_radius must be nonnegative")

        if previous_segment is None or local_search_radius is None:
            candidates: Iterable[int] = range(self.segment_count)
        else:
            if not 0 <= previous_segment < self.segment_count:
                raise ValueError("previous_segment is outside the track")
            candidates = sorted({
                (previous_segment + offset) % self.segment_count
                for offset in range(-local_search_radius, local_search_radius + 1)
            })

        best: tuple[float, int, float, float, float] | None = None
        for index in candidates:
            following = (index + 1) % len(self.x)
            dx = self.x[following] - self.x[index]
            dy = self.y[following] - self.y[index]
            length_sq = dx * dx + dy * dy
            fraction = min(1.0, max(0.0, ((x_m - self.x[index]) * dx + (y_m - self.y[index]) * dy) / length_sq))
            px = self.x[index] + fraction * dx
            py = self.y[index] + fraction * dy
            distance_sq = (x_m - px) ** 2 + (y_m - py) ** 2
            candidate = (distance_sq, index, fraction, px, py)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        assert best is not None
        distance_sq, index, fraction, projected_x, projected_y = best
        following = (index + 1) % len(self.x)
        dx = self.x[following] - self.x[index]
        dy = self.y[following] - self.y[index]
        length = self.segment_lengths[index]
        tangent_x, tangent_y = dx / length, dy / length
        lateral = tangent_x * (y_m - projected_y) - tangent_y * (x_m - projected_x)
        geometric_heading = math.atan2(dy, dx)
        if math.isfinite(self.heading[index]) and math.isfinite(self.heading[following]):
            heading_delta = wrap_angle(self.heading[following] - self.heading[index])
            path_heading = wrap_angle(self.heading[index] + fraction * heading_delta)
        else:
            path_heading = geometric_heading
        s_start = self.s[index]
        s_end = self.s[following] if following > index else self.total_length
        progress = (s_start + fraction * (s_end - s_start)) % self.total_length if self.closed else s_start + fraction * (s_end - s_start)
        return Projection(
            s_m=progress,
            lateral_offset_m=lateral,
            heading_error_rad=wrap_angle(yaw_rad - path_heading),
            segment_index=index,
            curvature_inv_m=self._interpolate(self.curvature[index], self.curvature[following], fraction),
            left_width_m=self._interpolate(self.left_width[index], self.left_width[following], fraction),
            right_width_m=self._interpolate(self.right_width[index], self.right_width[following], fraction),
            projected_x_m=projected_x,
            projected_y_m=projected_y,
            distance_m=math.sqrt(distance_sq),
            segment_fraction=fraction,
        )

    @staticmethod
    def _interpolate(first: float, second: float, fraction: float) -> float:
        if not math.isfinite(first) or not math.isfinite(second):
            return math.nan
        return first + fraction * (second - first)
