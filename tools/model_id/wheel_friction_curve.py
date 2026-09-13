"""AutoDRIVE/Unity-shaped two-piece tire friction surrogate.

The public simulator guide exposes the two breakpoints but not Unity's exact
internal interpolation.  This module therefore implements the documented
points with a cubic Hermite surrogate.  It is intentionally explicit about
that distinction so a diagnostic dump can replace the surrogate later.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class TireCurveParameters:
    extremum_slip: float
    extremum_value: float
    asymptote_slip: float
    asymptote_value: float
    stiffness: float = 1.0
    initial_slope: float | None = None
    source: str = "official_2026_guide_surrogate"

    def __post_init__(self) -> None:
        if not (0.0 < self.extremum_slip < self.asymptote_slip):
            raise ValueError("tire slip breakpoints must be positive and ordered")
        if self.extremum_value < 0.0 or self.asymptote_value < 0.0:
            raise ValueError("tire curve ordinates must be non-negative")
        if self.stiffness < 0.0:
            raise ValueError("tire stiffness must be non-negative")
        if self.initial_slope is not None and self.initial_slope < 0.0:
            raise ValueError("initial slope must be non-negative")

    @property
    def low_slip_slope(self) -> float:
        """Return the source/fitted low-slip slope in curve-value per slip."""
        if self.initial_slope is not None:
            return self.initial_slope
        # A conservative default that reaches the documented peak with zero
        # slope at the extremum.  The fitter may replace this with data.
        return 3.0 * self.extremum_value / self.extremum_slip

    def with_stiffness(self, stiffness: float, source: str | None = None) -> "TireCurveParameters":
        return replace(
            self,
            stiffness=stiffness,
            source=self.source if source is None else source,
        )


GUIDE_LONGITUDINAL_CURVE = TireCurveParameters(
    extremum_slip=0.15,
    extremum_value=0.72,
    asymptote_slip=0.25,
    asymptote_value=0.464,
)

GUIDE_LATERAL_CURVE = TireCurveParameters(
    extremum_slip=0.01,
    extremum_value=1.00,
    asymptote_slip=0.10,
    asymptote_value=0.500,
)


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _first_segment(slip: float, curve: TireCurveParameters) -> float:
    se = curve.extremum_slip
    t = min(1.0, max(0.0, slip / se))
    # Cubic Hermite with y(0)=0, y'(0)=K0, y(Se)=Fe, y'(Se)=0.
    h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0
    h10 = t * t * t - 2.0 * t * t + t
    h01 = -2.0 * t * t * t + 3.0 * t * t
    return h10 * se * curve.low_slip_slope + h01 * curve.extremum_value


def friction_value(slip: float, curve: TireCurveParameters) -> float:
    """Evaluate an odd, bounded, piecewise cubic friction curve."""
    if not math.isfinite(slip):
        raise ValueError("slip must be finite")
    magnitude = abs(slip)
    if magnitude <= curve.extremum_slip:
        value = _first_segment(magnitude, curve)
    elif magnitude < curve.asymptote_slip:
        t = ((magnitude - curve.extremum_slip) /
             (curve.asymptote_slip - curve.extremum_slip))
        value = curve.extremum_value + (
            curve.asymptote_value - curve.extremum_value) * _smoothstep(t)
    else:
        value = curve.asymptote_value
    return math.copysign(curve.stiffness * value, slip) if slip else 0.0


def friction_derivative(slip: float, curve: TireCurveParameters) -> float:
    """Evaluate the analytic derivative used by offline linearization checks."""
    if not math.isfinite(slip):
        raise ValueError("slip must be finite")
    magnitude = abs(slip)
    if magnitude < curve.extremum_slip:
        se = curve.extremum_slip
        t = magnitude / se
        # derivative of h10*Se*K0 + h01*Fe with respect to slip
        d_h10 = 3.0 * t * t - 4.0 * t + 1.0
        d_h01 = -6.0 * t * t + 6.0 * t
        slope = d_h10 * curve.low_slip_slope + d_h01 * curve.extremum_value / se
    elif magnitude < curve.asymptote_slip:
        width = curve.asymptote_slip - curve.extremum_slip
        t = (magnitude - curve.extremum_slip) / width
        slope = ((curve.asymptote_value - curve.extremum_value) /
                 width * 6.0 * t * (1.0 - t))
    else:
        slope = 0.0
    return curve.stiffness * slope
