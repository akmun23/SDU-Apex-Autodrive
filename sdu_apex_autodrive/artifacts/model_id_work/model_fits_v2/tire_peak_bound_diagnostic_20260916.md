# Tire-peak bound diagnostic — 2026-09-16

This is an offline identification diagnostic. It did not modify the Unity
simulator, the production MPC, or any runtime parameter.

The same causal steering-input contract, training data, validation data,
raceline CORE/GUARD filter, 30-step/40 Hz recursive scorer, and 0.75 s primary
horizon were used for every bounded fit.

| Low-regime peak upper bound | Fitted `d_r` | CORE `e_cross` p95 | CORE `r` p95 | GUARD `v` p95 | GUARD `r` p95 | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 60 N | 60.0 N (active bound) | 0.032719 m | 0.096568 rad/s | 0.032380 m/s | 0.247839 rad/s | Conservative reference |
| 120 N | 114.3 N | 0.032691 m | 0.096790 rad/s | 0.038821 m/s | 0.238187 rad/s | Small mixed change |
| 240 N | 93.2 N | 0.032697 m | 0.096533 rad/s | 0.035199 m/s | 0.241333 rad/s | Optimizer did not converge |
| 1000 N diagnostic | 179.8 N | 0.032686 m | 0.096824 rad/s | 0.045910 m/s | 0.278536 rad/s | Holdout degradation |

The 120 N and 240 N full reports are
`speed_regime_vehicle_candidate_delayed_steering_raceline_peak120_075_20260916.json`
and
`speed_regime_vehicle_candidate_delayed_steering_raceline_peak240_075_20260916.json`.
The 1000 N run was a low-regime in-memory diagnostic using the other regimes
from the current candidate; it was not retained as a candidate artifact.

## Interpretation

The old 60 N value was active, so it was correct to question whether it was
artificially restrictive. Raising it does not, however, produce a stable
identified parameter: the fitted rear peak moves between 93 and 180 N as the
search range changes, with negligible CORE benefit and worse behavior in the
unbounded holdout diagnostic. This is a weak-identifiability/model-structure
signal, not evidence that the physical tire peak is 114 N or 180 N.

The equivalent no-downforce capacity ratios are only sanity diagnostics. They
are approximately 3.90 at the 120 N fit and 5.83 at the 1000 N diagnostic.
The ratios cannot identify the simulator's normal-load or tire law, but their
growth reinforces that the optimizer is using peak force to absorb another
error source.

## What the current data says it is hiding

Using the current 60 N raceline-filtered candidate on the validation split,
the low-regime rear tire is not near saturation. For CORE rows, the rear
`|C_r alpha_r| / D_r` ratio has p95 `0.193`, p99 `0.346`, and maximum `0.537`;
only `0.22%` of rows exceed `0.5`, and none exceed `0.75`. For GUARD rows the
p95 is `0.121`, the maximum is `0.220`, and no row exceeds `0.25`. The front
tire is more observable: `22.0%` of low-regime CORE rows exceed `0.5` of its
peak. Therefore `D_r` cannot currently be identified as a physical rear tire
capacity. Its increase is mainly a way to alter a weakly observed rear-force
curve while the optimizer trades against recursive yaw error.

The remaining one-step yaw residual is also not uniform tire saturation:
CORE `r` p95 is `0.0447 rad/s` at 2--4 m/s, `0.0237` at 4--6 m/s, `0.1044` at
6--8 m/s, and `0.1181` at 8--10 m/s. The largest track-local residuals are
concentrated around raceline `s=23--30 m` and `s=50--52 m`. In GUARD, yaw
error increases strongly with positive wheel slip, large steering, and
throttle-times-steering. This points to a combination of low-speed/turn
transients, load or force distribution, and possibly run/segment state
alignment—not a single rear peak value.

The next identification pass should therefore:

1. compute a profile likelihood with `D_r` fixed over a grid while refitting
   stiffness and the other peak, and inspect the parameter-Jacobian condition
   number/correlations;
2. fit one competing mechanism at a time—rear/front load distribution, yaw
   damping, geometry/inertia sensitivity, combined-slip law, and steering
   transient—then require the same held-out improvement; and
3. collect raceline-plausible low-speed turn data that actually excites rear
   slip, with independent speed holds and progressive steering, rather than
   inferring `D_r` from the current low-slip track data.

## Decision

Keep 60 N as the current conservative offline regularizer and keep the bound
configurable through `--lateral-peak-upper-n` for future experiments. Do not
promote the 120/240/1000 N results and do not migrate any of them into MPC.
The next useful experiment is new raceline-relevant low-speed transient and
steady-corner data that independently excites tire capacity and separates it
from stiffness, steering timing, localization, and unmodelled load effects.
