# Open-plane per-wheel dynamics findings

Updated: 2026-09-27

> Audit correction (2026-09-26): the raw simulator `/odom` twist is COM-referenced,
> although its pose/child frame is at the rear axle. Earlier wheel-slip and
> lateral-velocity figures in this file that treated raw `twist.linear.y` as
> rear-axle velocity, and the previous 3/5 m/s force-fit values, are superseded
> by the correction section at the end. The Ackermann left/right assignment is
> also now aligned with the official bridge's wheel transforms. Keep old
> numbers below only as an experiment history; do not use them for model
> selection.

## Decision

The vehicle-dynamics guide is useful for structuring an identification model,
but the current open-plane telemetry is insufficient to identify all four
wheel forces or suspension loads independently. Six candidate model forms
were tested against matched-start simulator data:

1. A static-load, lateral-only implementation of the guide's tire curve was
   **rejected** for MPC use.
2. A constant lateral-acceleration hard cap passed its predeclared held-out
   output-error screen, but was **not promoted**: its steering derivative is
   zero at 15 of 16 held-out operating points. That is a poor local model for
   an optimizer that needs useful control sensitivities, and the existing MPC
   history records a separate saturated yaw model reversing the steering
   solution at the first high-curvature transition.
3. A condition-matched rear-wheel-slip correction **passed at 2.2 and
   3.0 m/s**, but failed as a universal feature at 5.0 m/s. It describes a
   low/mid-speed response branch, not a global tire law.
4. At 4.0 and 5.0 m/s, shape-preserving cubic steering-to-yaw-gain maps
   passed leave-one-repetition-out intermediate-angle tests with useful
   positive steering sensitivity.
5. A steering PCHIP per measured speed, with lateral acceleration
   interpolated between 3.0 and 5.0 m/s, predicted the entirely held-out
   4.0 m/s capture to 0.00779 1/m yaw-gain RMSE with positive sensitivities.
   This is accepted only for the steady-state 3–5 m/s band.
6. Stretching that same surface from 2.2 to 5.0 m/s failed at 3.0 m/s. The
   below-3 m/s region has a distinct rear-slip-dependent response; do not use
   one global interpolation or the high-speed map there.
7. Three repeated steering-grid captures now extend the empirical response
   through 7.5 m/s. A 4.5/7.5-to-6.5 m/s holdout produced good output RMSE
   but failed the local-Jacobian sign gate (6/8 intervals); do not promote
   that interpolator to MPC.

No simulator physics, competition runtime, odometry, or MPC production
parameters were changed in this work package. The findings are diagnostics,
not evidence that the guide's tire curve is wrong.

## What the guide gives us

The [AutoDRIVE vehicle dynamics guide, §1.3.2](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics)
describes the body and sprung/unsprung masses, suspension spring/damper
forces, and tire forces driven by longitudinal and lateral slip. It gives a
two-piece cubic tire-response curve and per-wheel steering/velocity
relationships. It also documents the simulated vehicle's relevant geometry
and setup: total mass 3.906 kg, sprung mass 3.470 kg, four unsprung masses
totalling 0.436 kg, wheelbase 0.324 m, track 0.236 m, wheel radius 0.059 m,
and CG longitudinal coordinate 0.15532 m. It specifies AWD, so longitudinal
slip cannot safely be ignored while interpreting lateral response.

For wheel `i`, using body axes x-forward/y-left, rigid-body kinematics give

```text
u_i = u - r*y_i
v_i = v + r*x_i
Vx_i = cos(delta_i)*u_i + sin(delta_i)*v_i
Vy_i = -sin(delta_i)*u_i + cos(delta_i)*v_i
```

The guide's slip coordinates then feed each wheel's nonlinear force curve.
Wheel forces must be rotated back to body coordinates before summing them;
the body equations include both force and yaw moment:

```text
m*(du/dt - r*v) = sum(Fx_i)
m*(dv/dt + r*u) = sum(Fy_i)
Iz*dr/dt = sum((x_i-x_CG)*Fy_i - (y_i-y_CG)*Fx_i)
```

The suspension model determines each changing normal load. Therefore a
static-load lateral-only fit omits exactly the terms most likely to matter in
hard cornering: load transfer, AWD longitudinal tire force, and combined
slip. Those omissions are hypotheses to test, not assumed explanations.

## Simulator evidence

Accepted bags:

| Run | Conditions | Packet rate | Phases | Matched starts | Collisions |
| --- | --- | ---: | ---: | ---: | ---: |
| `openplane_isolated_boundary_window_20260926_172626` | 2.2 m/s, signed 0.30/0.42/0.46/0.50 rad probes | 39.963 Hz | 24/24 valid | 24/24 | 0 |
| `openplane_isolated_force_3mps_holdout_20260926` | 3.0 m/s, same steering levels, three randomized repetitions | 39.972 Hz | 24/24 valid | 24/24 | 0 |
| `openplane_isolated_force_4mps_20260926` | 4.0 m/s, same steering levels, three randomized repetitions | 39.976 Hz | 24/24 valid | 24/24 | 0 |
| `openplane_isolated_force_5mps_20260926` | 5.0 m/s, same steering levels, three randomized repetitions | 39.977 Hz | 24/24 valid | 24/24 | 0 |

The 3.0 m/s run held repetition 3 out. A fixed `a_y,max` model
`r_ss = sign(delta)*min(u*tan(delta)*2.95, a_y,max/u)` fitted to repetitions
1–2 selected `a_y,max = 4.460 m/s^2` (0.455 g). Held-out yaw-rate RMSE was
0.3413 rad/s versus 0.9881 rad/s for the configured steering-only gain
(65.5% lower); the reductions were 38.6% at 2.2 m/s and 96.2% at 3.0 m/s.
This passes the evaluator's predeclared *output prediction* screen.

However, differentiating that hard minimum reveals a control problem: the
candidate's `d(r_ss)/d(delta)` is exactly zero at 15/16 held-out points. It
matches output by flattening the steering response, not by identifying how
incremental steering changes yaw. The fit is therefore not an MPC model yet.
The diagnostic and its Jacobian checks are in
[`evaluate_open_plane_lateral_envelope.py`](../../tools/evaluate_open_plane_lateral_envelope.py).

## Held-out rear-slip result

The two rear encoders are measured inputs, not simulator-truth/debug signals:
the official guide classifies `/autodrive/roboracer_1/left_encoder` and
`right_encoder` as `Input` topics. This makes a rear-wheel-slip feature
available to the controller without consuming a restricted topic. The
high-angle response can be written as a compact local yaw-gain model:

```text
K_yaw = K0(speed, |steer|, turn_sign)
       + beta(speed) * (mean(|Sx_rear|) - Sx_reference(condition))
r_ss = speed * tan(steer) * K_yaw
```

Here `K0` and `Sx_reference` are training-repetition condition means;
`beta` is fitted only from within-condition variation in repetitions 1 and 2.
The untouched third repetition tests whether measured rear longitudinal slip
predicts the response branch beyond speed, steering magnitude, and turn
direction. The acceptance rule was set before evaluation: at least 10% lower
held-out yaw-gain RMSE at each speed, with the same coefficient sign.

| Target speed | Training / holdout blocks | Slip coefficient | Condition-only RMSE | With rear-slip RMSE | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.2 m/s | 16 / 8 | +13.806 per slip ratio | 0.6276 1/m | 0.1620 1/m | -74.2% |
| 3.0 m/s | 16 / 8 | +4.560 per slip ratio | 0.0543 1/m | 0.0208 1/m | -61.7% |

Both fitted coefficients are positive and both speed-specific holdouts pass.
The strongest individual change is the 2.2 m/s `+0.46 rad` block: the
condition-only prediction is 1.537 1/m, while adding the held-out rear-slip
state predicts 2.988 1/m versus 2.927 1/m measured. At 3.0 m/s, the
`+0.42 rad` holdout moves from 1.096 to 1.198 1/m versus 1.240 measured.
This predicts the previously unexplained high-angle branch using a sensor
state the competition guide explicitly permits.

This is explicitly **not** a speed-independent correction. On the new 5.0 m/s
capture, the rear-slip coefficient was -0.039, and it reduced held-out RMSE
by only 1.1% (0.0002 to 0.0002 1/m); it fails the predeclared 10% gate and
does not retain the low/mid-speed coefficient sign. Rear longitudinal slip
is nearly constant across the 5.0 m/s steering conditions, so the response
there is instead repeatable from steering alone. Keep the slip correction
local to the 2.2–3.0 m/s regime until a continuous speed-dependent coefficient
is independently validated.

Interpretation is still bounded. This is an association, not proof that rear
wheel spin causes the yaw change: wheel slip is not independently randomized,
its estimate depends on the documented 0.059 m radius, and the test holds
throttle near one value. It does establish that an angle/speed-only state map
is missing predictive state. A plausible physical route is AWD tire-force
redistribution: each rear wheel's longitudinal force and lateral force act
at different positions, so unequal wheel slip changes both total force and yaw
moment. The guide gives separate longitudinal/lateral slip-force curves but
does not specify a combined-slip coupling law; do not assume a friction-circle
formula without evidence.

The script is
[`evaluate_open_plane_rear_slip_effect.py`](../../tools/evaluate_open_plane_rear_slip_effect.py).
It uses phase medians and holds out whole repetitions, not individual 40 Hz
samples. The lookup baseline means this establishes a local response feature
at the tested low/mid speeds/steering levels; interpolation, transient
prediction, and a 750 ms MPC rollout remain unvalidated.

## High-speed steering curve

At 5.0 m/s, the measured yaw gain is repeatable across repetitions and turn
signs: 0.679, 0.509, 0.471, and 0.448 1/m at steering magnitudes 0.30, 0.42,
0.46, and 0.50 rad. Yaw gain falls as steering increases, but actual steady
yaw rate still increases; a constant lateral-acceleration cap therefore
erases useful steering sensitivity.

[`evaluate_open_plane_yaw_spline.py`](../../tools/evaluate_open_plane_yaw_spline.py)
fits a shape-preserving cubic Hermite yaw-gain map in `|steering|`, separately
for turn direction. It withholds an intermediate steering level and performs
leave-one-repetition-out validation. On the 5.0 m/s bag (12 intermediate
angle predictions):

- yaw-gain RMSE: 0.00269 1/m; maximum absolute error: 0.00302 1/m;
- corresponding yaw-rate RMSE: 0.00633 rad/s;
- configured steering-gain RMSE on the same points: 1.55484 1/m;
- predicted `d(r_ss)/d(steer)` remained positive, 0.765–1.024 s^-1.

It passes the declared high-speed screening gate (yaw-gain RMSE ≤0.03 1/m
and finite positive local steering sensitivity). This is an accurate local
steady-state steering law at 5.0 m/s, not yet a transient model or a speed
interpolator. Frame-corrected slip proxies show substantial axle asymmetry
there: front `|Sy|=0.2436–0.4723`, rear `|Sy|=0.0034–0.0041` at 0.30–0.50
rad. These replace earlier COM-unshifted estimates. The asymmetry is consistent
with different front/rear contributions to yaw, but does not establish a
causal force decomposition without per-wheel loads and forces.

## Validated speed-dependent response, 3–5 m/s

Run `openplane_isolated_force_4mps_20260926` completed 24/24 matched probes;
all measured streams were 39.976 Hz, collision count stayed zero, and no
bridge timing fault occurred. Its yaw gains were 0.967, 0.720, 0.659, and
0.618 1/m at 0.30, 0.42, 0.46, and 0.50 rad, respectively. The corrected
front `|Sy|` proxy spans about 0.212–0.435, past the guide's 0.10 asymptote
landmark, while rear `|Sy|` stays near 0.003. The response was stable between
early and late probe halves.

The predeclared cross-speed test used only the 3.0 and 5.0 m/s captures to
predict the independent 4.0 m/s run. Each endpoint has its own
shape-preserving `K_yaw(|steer|)` curve; interpolate signed lateral
acceleration `a_y = v² tan(steer) K_yaw` across speed, then recover
`r_ss = a_y/v`. With 0.42 and 0.46 rad withheld from both steering curves,
the 12 held-out blocks gave 0.00779 1/m yaw-gain RMSE, 0.01431 rad/s
yaw-rate RMSE, and positive finite local sensitivities of 0.404–0.803 s⁻¹.
The configured gain's yaw-gain RMSE on those same blocks was 1.37372 1/m.
An independent per-speed 4.0 m/s PCHIP check gave 0.00376 1/m yaw-gain RMSE,
0.00708 rad/s yaw-rate RMSE, and 0.608–0.767 s⁻¹ positive sensitivity.
This supports a compact empirical steady-state response map inside 3–5 m/s.
It is not yet integrated into `vehicle_model.c`, and this test does not
establish transient or full-horizon rollout accuracy.

Do not stretch this surface down to 2.2 m/s. Direct yaw-gain interpolation
between 2.2 and 5.0 m/s missed the held-out 3.0 m/s response by 0.625 1/m.
Interpolating lateral acceleration across that same wide interval reduced
the error to 0.189 1/m but created negative local steering sensitivities.
Both wide-interval candidates are rejected. At 3.0 m/s, the direct steering
PCHIP predicts output to 0.0487 1/m RMSE but has a sensitivity sign reversal
at some high-angle points; the rear-slip state feature remains useful in the
2.2–3.0 m/s branch. At 2.2 m/s, the direct steering PCHIP is inaccurate
(0.496 1/m).

The rear-slip feature also fails its predeclared 10% gate at 4.0 m/s: held-out
RMSE decreases only 5.8% (0.0008 to 0.0007 1/m). Do not add it to the 3–5
m/s response surface.

An additional 5.0 m/s initial-speed throttle contrast was captured in
`live_runs/openplane_combined_slip_5mps_20260926/run/run_0.db3`. It completed
without collisions at 39.977 Hz, but is not a matched high-speed slip test:
the 0.08 throttle phases fell to 1.93 m/s, while 0.38 phases stayed near
5.3 m/s. Rear `|Sx|` changed from about 0.07 to 0.62–0.73 as speed changed.
The speed-conditioned slip model improved held-out yaw-gain RMSE only 4.1%,
with the expected effect direction in 2/4 held-out conditions. Reject the
high-speed slip correction and do not repeat that fixed-throttle schedule.
The capture does show a strongly nonlinear throttle/speed transition, but
cannot identify a causal combined-slip effect.

The separate per-wheel force hypothesis used guide tire-curve landmarks,
assumed static axle loads, fitted one common force scale and spline tangent,
and omitted drive force. At 3.0 m/s it required a 4.271x force scale; net
lateral-force RMSE was 15.342 N on held-out repetition 3, and inferred
yaw-acceleration RMSE was 233.595 rad/s^2. Its recursive one-step rollout
diverged badly. At 5.0 m/s it fit net lateral force much better (force scale
2.395; held-out RMSE 1.724 N), but inferred non-positive yaw inertia (-4.276
kg m^2 with the guide's wheel labels, -4.411 kg m^2 with coordinate-consistent
labels). The force-only/yaw-moment implementation is **rejected**. The fitter
uses settled samples, which barely excite yaw acceleration, so that inertia
estimate is weakly identified and is not evidence against the published tire
curve. Missing dynamic loads, AWD longitudinal forces, and per-wheel force
observations still prevent identifying the full wheel model. See
[`fit_open_plane_tire_model.py`](../../tools/fit_open_plane_tire_model.py).

The capture includes body odometry, steering, throttle, IMU, and the two legal
rear-wheel encoders. It does not include an independently observed normal
load, per-wheel contact force, or front wheel speeds. Consequently the
body equations constrain aggregate forces and moments, while the unknowns
include four wheels' longitudinal/lateral forces and dynamic loads. Many
wheel-force allocations can explain the same body motion. The rear-axle
encoder/body-speed ratio supplies a useful longitudinal-slip proxy, but it
does not resolve front-wheel slip or contact loads. The official topic table
marks both rear encoders as allowed inputs; IPS and raw simulator odometry are
marked restricted. See [AutoDRIVE guide §2.3](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#23-data-streams).

In the existing raw-packet/odometry joins, odometry twist exactly relayed the
simulator packet's body velocities. That makes an odometry integration error
an unlikely cause of the open-plane high-angle yaw mismatch. It does not
validate localization or prove the MPC's state estimates on the track.

## MPC implications and next gate

The current production predictor is an identified yaw-response map, not a
four-wheel force model (`vehicle_model.c`). The evidence supports a genuine
high-steering/high-speed loss of yaw authority that a steering-only gain
doesn't represent. But neither the hard cap nor the rejected static-load
force fit is safe to insert directly. In addition, the committed source
comments record that a previous `tanh`/speed-squared saturation caused an
N=30 steering solution to reverse near path distance 34.6 m. The exact failed
candidate implementation is not present in Git history, so this is a safety
warning rather than a claim that the new cap is identical to it.

The next MPC candidate should use the validated lateral-acceleration response
surface inside 3–5 m/s, retain a separately validated rear-slip branch below
3 m/s, and avoid extrapolating either map outside its measured domain. It
must not freeze rear-wheel slip through the prediction horizon if that state
feature is used. The force/suspension data gap prevents recovering four
independent tire laws, but does not prevent using this effective response
model. Before a production MPC change, it must demonstrate all of the
following on held-out captures and then on a single gated practice run:

- predict both yaw rate and lateral velocity across steering sign, angle,
  speed, and throttle condition, with held-out error reported by condition;
- have finite local sensitivities that agree in sign and scale with measured
  finite differences where the data supports a derivative;
- keep the N=30 optimizer from reversing steering at the documented
  high-curvature transition and produce repeatable feasible solutions;
- retain the now-passed 4 m/s cross-speed holdout, generalize to a separate
  speed above 5 m/s, and validate the low-speed slip branch;
- keep the established instant-abort-on-collision policy and pass the
  competition topic-policy check.

The fixed-throttle contrast was recorded and rejected for causal combined-slip
identification because it drove speed from 5.0 to 1.93 m/s in the low-input
condition. Do not repeat it. Any future drive-level experiment must keep
speed matched while creating a measurable encoder-slip difference. If the
simulator exposes wheel loads, wheel torques, front wheel speeds, or tire
forces in development mode, record them; otherwise treat a full per-wheel
force model as non-identifiable from this sensor set and use the validated
compact response surface for MPC.

## Repeated transient sweep at 4 m/s — 2026-09-26

The earlier matched-start probes isolated steady response but did not separate
current-steering response from steering-history effects. A development-only
`transient_4mps` profile now runs four independent sequences at each turn sign:
0.30 -> 0.42 -> 0.46 -> 0.50 rad, then 0.46 -> 0.42 -> 0.30 rad. The fourth
repetition is reserved for held-out rollout validation. This does not alter
simulator physics or the competition controller.

Accepted run:
`live_runs/openplane_transient_4mps_holdout_20260926/run/run_0.db3`

Before scoring, the capture gate required all 56 step phases valid, all eight
sequence starts matched, 38 Hz minimum stream rate, p95 gap <=35 ms, max gap
<=60 ms, zero collisions, and zero bridge timing faults. It passed: all 56
steps and eight starts were valid; every recorded stream ran at 39.977 Hz,
p95 gaps were 25.55–25.67 ms, maximum gaps 48.94–49.24 ms, and collisions and
timing faults remained zero. The closed bag is about 14 MB. The simulator was
launched with the repository's pinned explore image in Xvfb batchmode (not
headless/no-graphics and no GUI window).

The measured response isolates the steering nonlinearity:

| Steering | Yaw gain (1/m) | sign-corrected yaw rate (rad/s) | front \|Sy\| | rear \|Sy\| |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 rad | 0.968 | 1.190 | 0.2125 | 0.0028 |
| 0.42 rad | 0.721 | 1.281 | 0.3405 | 0.0029 |
| 0.46 rad | 0.659 | 1.301 | 0.3878 | 0.0029 |
| 0.50 rad | 0.618 | 1.344 | 0.4345 | 0.0031 |

From 0.30 to 0.50 rad, yaw gain falls 36.1% while actual yaw rate still rises
12.9%. The measured yaw-rate finite-difference sensitivities stay positive,
0.51–1.07 s^-1 across the tested intervals and both turn signs. Thus the
observed nonlinearity is a strong, smooth loss in incremental steering
authority, not a steering-response sign reversal over this 4 m/s range.

The guide places the lateral tire-curve extremum at |Sy|=0.01 and asymptote
landmark at |Sy|=0.10. The corrected front-wheel kinematic slip proxy is
already 0.2125 at 0.30 rad and rises to 0.4345; rear |Sy| stays near 0.003.
This is consistent with the front tires operating beyond the guide's
peak/asymptote landmarks and losing marginal force as steering grows. It is
evidence for a front-tire saturation hypothesis, not a force measurement or
causal proof: these slips are computed from body odometry and Ackermann
geometry, while individual wheel loads and contact forces are not published
in the bag.

The paired-history test found no steady-state hysteresis above its predeclared
0.05 1/m threshold. Across 0.30, 0.42 and 0.46 rad, training up/down yaw-gain
differences were <=0.004 1/m; the untouched fourth repetition reproduced the
near-zero result. Front and rear slip proxies were also essentially unchanged
between directions at a matched angle. Do not add an extra steady hysteresis
state based on these data.

`tools/evaluate_open_plane_transient_rollout.py` then trained only on
repetitions 1–3 and scored repetition 4 from the measured starting odometry
state using recorded speed and steering as conditional inputs. The fitted
local response used the measured yaw-gain knots, a first-order yaw state
(effective tau 0.010 s, zero extra sample delay), and a fitted low-angle
transition. It reduced held-out yaw-rate RMSE by 99.4–99.9% at all 25–750 ms
horizons versus the configured steering gain. This is strong local evidence
that the static nonlinear response plus a compact yaw state predicts the
tested high-angle 4 m/s transients; it is not a closed-loop MPC or cross-speed
result. The fitted time constant is an effective sampled parameter, not a
measured suspension or tire relaxation constant.

An analogous empirical body-lateral-velocity state reduced held-out RMSE by
90–98% from 125–750 ms, but failed the predeclared no-regression gate at
25 ms (0.00030 m/s candidate versus 0.00009 m/s held-state baseline). Reject
that v_y rollout for MPC promotion; the longer-horizon improvement does not
erase the first-step failure. No production MPC/odometry parameters or
competition topic use were changed.

Reproducible evaluators:
[`evaluate_open_plane_transient_sweep.py`](../../tools/evaluate_open_plane_transient_sweep.py)
and
[`evaluate_open_plane_transient_rollout.py`](../../tools/evaluate_open_plane_transient_rollout.py).
Next, validate the yaw surface with the already-measured 3-to-5 m/s
interpolation and its local Jacobian in an MPC-stage-equivalent rollout; keep
the lateral-velocity state unchanged until a fresh held-out capture passes the
first-step gate. Do not fit a per-wheel force law from the current streams.

## Full-angle 4 m/s surface and the 0.20–0.25 rad transition — 2026-09-26

The earlier transient sweep started at 0.30 rad, leaving the controller's
low-to-high steering transition unmeasured. The new `transient_fullsteer_4mps`
profile directly sampled 0.05–0.50 rad, both signs, with matched up/down
sweeps in three repetitions. Repetitions 1–2 trained the response; repetition
3 was held out.

Accepted run:
`live_runs/openplane_fullsteer_4mps_20260926/run/run_0.db3`

The capture gate passed: 114/114 steering steps and 6/6 matched starts, zero
collisions and bridge timing faults. Odom, steering, both rear encoders, IMU,
and packet timing each ran at 39.972 Hz; p95 gaps were 25.61–25.72 ms and
maximum gaps 48.70–49.08 ms. The excitation command stream ran at 39.20 Hz.
The run used the pinned explore simulator in the established Xvfb batchmode
launcher and produced a 26.4 MB bag.

The measured response is sharply non-monotonic around 0.20–0.25 rad:

| |steer| (rad) | `K_yaw = r/(vx tan(delta))` (1/m) | sign-corrected yaw rate (rad/s) | mean front `|Sy|` | mean rear `|Sy|` |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 | 2.947 | 1.758 | 0.0107 | 0.0039 |
| 0.20 | 2.909 | 2.312 | 0.0174 | 0.0064 |
| 0.25 | 1.152 | 1.168 | 0.1605 | 0.0026 |
| 0.30 | 0.968 | 1.189 | 0.2124 | 0.0027 |
| 0.42 | 0.721 | 1.279 | 0.3405 | 0.0028 |
| 0.50 | 0.618 | 1.344 | 0.4344 | 0.0031 |

From 0.20 to 0.25 rad, yaw rate falls about 49.5% even though steering
increases; the corrected front lateral-slip proxy jumps from about 0.017 to
0.161. Above 0.25 rad, yaw rate recovers only gradually while front slip keeps
growing. Rear lateral-slip proxies remain around 0.0025–0.0065 in this band.
The up/down response at 0.20 rad differs by about 0.04 1/m, below the
predeclared 0.05 1/m history-effect threshold and not reproduced above that
threshold in the holdout. This supports a repeatable front-axle saturation
transition, not demonstrated hysteresis. Its coincidence with the guide's
lateral-tire asymptote landmark (`|Sy|=0.10`, normalized force 0.50) is
consistent with that explanation, but `Sy` is calculated from rigid-body
motion and Ackermann geometry, not read from the tire contact solver.

An effective yaw map `r_ss = sign(delta) * vx * g(|delta|)` was fitted using
training knots from 0 through 0.50 rad with shape-preserving cubic
interpolation, plus a first-order yaw state. The untouched repetition gave
0.03852 rad/s steady-yaw RMSE; local sensitivity signs matched in every
measured steering interval, including the negative 0.20–0.25 rad interval.
The fitted effective time constant was 0.145 s with no extra sample delay.
Conditional MPC-midpoint rollout yaw RMSE improved by 88.1–98.9% at 125 ms
and 90.1–97.9% at 250–750 ms against the configured yaw law. At 750 ms,
position RMSE fell from 1.654 to 0.225 m and heading RMSE from 1.318 to
0.079 rad. These are local open-loop conditional-input scores at 4 m/s, not
closed-loop MPC or lap results.

The guide specifies per-tire longitudinal/lateral slip, a two-piece cubic
tire curve, and suspension forces. Its published ROS/data-recorder fields
include the two rear wheel encoders, steering, throttle, IMU, and restricted
IPS/odometry for debugging; they do not include individual contact forces,
suspension displacement/load, or front-wheel angular speeds. Therefore this
bag supports wheel-specific *kinematic slip proxies* and an effective yaw
response, but cannot identify four separate tire-force curves or dynamic
normal loads. Keep the force explanation as a hypothesis; do not call these
proxies measured tire forces. See the [official vehicle dynamics and data
streams](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics).

Package decision: **IMPLEMENTED** for a bounded 4 m/s yaw surface. Do not
insert this 4 m/s-only map into the controller yet. The next targeted capture
should refine 0.18–0.28 rad at 3, 4, and 5 m/s to test whether the collapse
tracks steering angle or front-tire slip; then validate the speed-conditioned
surface and its Jacobian in the MPC stage rollout.

## Reference-frame audit and corrected wheel-model results — 2026-09-26

The bridge's pose/TF origin is explicitly the rear axle, but the ROS odometry
builder copies the simulator's `V1 Linear Velocity` into `twist.linear` without
a rigid-body point shift. On the valid 4 m/s transition capture, central
differentiation of the published rear-axle pose and rotation into body axes
gave 0.2610 m/s combined velocity residual if the twist was treated as rear-
axle velocity, versus 0.0595 m/s after adding the guide's COM lever-arm term
(ω × r_COM). Both turn directions were present; the COM hypothesis reduced
the residual by 4.38×, with 0.0265 rad/s yaw-rate/pose-derivative RMSE.
Accordingly, offline model tools now convert
`v_y,rear = v_y,COM - r*x_COM` before wheel kinematics. The competition
odometry observer does not subscribe to this raw odometry stream; it publishes
its own rear-axle state from the permitted encoder and IMU inputs, so this audit
does not itself imply a runtime topic-policy or estimator change.

The development analysis also had the front Ackermann labels reversed. For
the bridge's x-forward/y-left frame and positive left turn, the inside left
wheel has the larger angle; this now matches the official bridge's static
wheel transforms. The focused spline/geometry self-check verifies both steering
signs and the COM-to-rear-axle transform.

With both corrections, the held-out 4 m/s transition data show a strong
association between front slip and yaw authority: at 0.21 rad, front mean
`|Sy|` is about 0.027 on the high-yaw branch and 0.113 on the low-yaw branch.
Those values cross the guide's 0.10 lateral asymptote landmark while yaw
authority drops. This is physically consistent with the guide's declining
post-peak tire curve, but the slip is reconstructed from body motion, not read
from the contact solver. The same-sample fit is descriptive only. A causal
model using the prior predicted state improved yaw error 69% at 25 ms, then
regressed 2.6%, 17.3%, 23.1%, and 46.1% at 125, 250, 500, and 750 ms. It is
**rejected** as an MPC rollout model.

The corrected guide-informed four-wheel force fit on the transition capture
uses the published peak/asymptote slip landmarks and static COM-derived loads.
It fits the small-slip Hermite tangent to 2.175, force scale 0.982, and predicts
held-out net lateral force with 1.181 N RMSE (train 1.175 N). However, the
force/moment fit implies negative yaw inertia (-0.0249 kg m²), so it cannot
support a forward dynamic rollout. On the full-angle 4 m/s bag, force scale is
1.063 and held-out net-force RMSE 2.866 N; yaw-rate rollout improves versus
the current yaw-only model (0.736 vs 1.808 rad/s RMSE), but rear lateral
velocity diverges (1.279 vs 0.0029 m/s), and inferred inertia 0.1008 kg m² is
outside the fitter's plausible band. The four-wheel model is **not promoted**.
Its fitted small-slip tangent also varies by speed (0.725 at 3 m/s, 2.175 on
the 4 m/s transition, 1.000 on the full-angle 4 m/s capture, and 1.950 at
5 m/s); the current static-load/single-curve assumptions are not statistically
consistent enough for a general tire model.

The earlier 3 and 5 m/s force-fit values quoted above used the unshifted twist
and are superseded: corrected fits give force scales 1.237 and 1.044,
respectively, with held-out aggregate lateral-force RMSE 2.920 N and 0.790 N.
Both still imply non-positive yaw inertia, so those force fits remain
diagnostic and cannot produce a validated yaw-moment rollout.

The same frame-corrected replay tests the deployed odometry lateral-velocity
law using optimistic true speed and yaw inputs: held-out rear-axle `v_y` RMSE
is 0.0061 m/s. Integrating IMU lateral acceleration in turns gives 0.0680 m/s
RMSE, so that alternative is **rejected**. A separate lever-arm fit explains
why the observer's 0.15532 m acceleration reference is appropriate even though
the IMU TF is at x=0.08 m: the IMU acceleration payload follows COM acceleration
(fitted x=0.165 m; held-out RMSE 0.0330 m/s² at COM versus 0.0469 at the IMU
frame). No runtime odometry parameter was changed.

Current conclusion: the guide's individual-wheel force equations are valuable
for identifying the nonlinear region and constructing falsifiable models.
The data support front-tire saturation as a plausible source of the sharp
response loss, but do not yet support a safe per-wheel dynamic MPC rollout.
Missing dynamic normal loads, yaw inertia identification, front wheel speeds,
and contact-force observations remain the limiting measurements. Keep the
validated bounded yaw surface as the current offline candidate and do not
change competition physics or controller behavior from the rejected force fit.

## Four-second branch dwell at 4 m/s — 2026-09-26

To test whether the 0.21 rad sequence effect was just a short response lag,
`openplane_transition_dwell_20260926` held fixed throttle command 0.164 and
steering 0.20–0.23 rad for four seconds in both turn directions and three
repetitions. Training used repetitions 1–2; repetition 3 was held out. The
experiment ran in the pinned explore simulator via the established Xvfb
batchmode launcher.

Capture passed 42/42 scored phases, 6/6 matched starts, 39.971 Hz on all
measured dynamics streams, zero collisions, zero timing faults, and zero
quality failures. The actuator command rate was 39.76 Hz. Bag:
`live_runs/openplane_transition_dwell_20260926/run/run_0.db3`.

| Steering | Sequence | Early yaw gain | Final yaw gain | Front `|Sy|` | Rear `|Sy|` | Rear `|Sx|` | Speed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 rad | up / down | 2.9141 / 2.9141 | 2.9141 / 2.9141 | 0.0166 / 0.0166 | 0.0058 / 0.0058 | 0.0703 / 0.0707 | 3.879 / 3.879 m/s |
| 0.21 rad | up / down | 2.8629 / 1.4276 | 2.8646 / 1.4279 | 0.0208 / 0.1161 | 0.0061 / 0.0028 | 0.0738 / 0.0338 | 3.863 / 4.007 m/s |
| 0.22 rad | up / down | 2.5429 / 1.2986 | 1.2986 / 1.2986 | 0.1308 / 0.1308 | 0.0027 / 0.0027 | 0.0306 / 0.0306 | 4.011 / 4.011 m/s |
| 0.23 rad | up only | 1.2279 | 1.2278 | 0.1421 | 0.0027 | 0.0300 | 4.012 m/s |

Yaw gain is `r/(vx*tan(delta))`; it is not raw yaw rate. At 0.21 rad, the
high and low branches remained stable from the first measured window to the
end of the four-second hold, and the full 1.4367 1/m branch difference
repeated in held-out repetition 3. At 0.22 rad, the up sequence started on
the high branch but moved to the low branch by the final window. Therefore a
brief transient alone does not explain the 0.21 difference. The data support
a long-lived history-conditioned state and a sharp transition near
0.21–0.22 rad; they do not prove two steady equilibria or identify whether
the memory is tire relaxation, combined slip, load transfer, drivetrain
state, or another hidden simulator state.

Comparison with the deployed competition MPC model exposes a concrete gap.
Its steering-gain taper starts at 0.41 rad, so it predicts 2.95 1/m at both
0.25 and 0.30 rad; the measured 4 m/s values are 1.152 and 0.968 1/m. At
0.50 rad it predicts 1.17 1/m versus 0.618 measured. The current model
therefore overpredicts high-angle yaw authority by roughly 1.9–3.0x across
these points, and it has no state to select between the 0.21-rad branches.
This is a demonstrated model mismatch and a plausible explanation for poor
MPC high-angle rollouts, but it has not yet been linked to a particular
optimizer rejection log. The model definition is in
[`vehicle_model.c`](../../f1tenth_mpc/src/vehicle_model.c) and its active
competition parameters are in
[`mpc_competition.yaml`](../../f1tenth_mpc/config/mpc_competition.yaml).

The command was fixed, not the resulting speed: the 0.21-rad branch speeds
differed by 3.6%, and their rear `|Sx|` proxies were 0.0738 and 0.0338. The
front `|Sy|` proxies were 0.0208 and 0.1161. That supports rear drive slip
and front lateral saturation as candidate state variables, but their
correlation is not causal identification. In particular, speed and
longitudinal slip still covary, so this is not yet the experiment that can
attribute the branch to combined slip.

A guide-curve four-wheel fit with an added linear yaw-drag term still failed
physically: unconstrained fitted yaw inertia was -0.15575 kg m^2; the bounded
fit stuck at the 0.005 kg m^2 lower limit. On repetition 3 its yaw-acceleration
RMSE was 29.788 rad/s^2 versus 0.402 for a zero-acceleration baseline. Reject
this model for MPC use. The guide's equations remain useful structure, but
published chassis ground truth plus rear encoders does not expose the
per-wheel forces, front wheel angular speeds, or dynamic normal loads needed
to uniquely identify every tire force and suspension state. See the [official
vehicle-dynamics guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics).

Package decision: **REJECTED for MPC promotion**. The measured response is
real and repeatable, but the candidate force model fails and the available
signals do not yet identify the hidden state causally. The next discriminating
capture should match actual body speed and steering while independently
varying rear longitudinal slip, with entire repeated sequences held out. Fit
only compact causal candidates using competition-permitted inputs; use
ground-truth pose/velocity as offline scoring targets only. If independent
slip excitation cannot be achieved, document the model as non-identifiable
from the available topics instead of adding unvalidated model coefficients.
No production MPC, odometry, physics, or topic-policy changes were made.

## Coupled lateral/yaw body-state model — 2026-09-27 — REJECTED

The existing `openplane_transition_4mps_20260926` bag was used for a new
recursive test, not a new simulator run. Repetitions 1–2 trained a smooth
radial-basis steering surface interacted with current predicted lateral
velocity and yaw rate; repetition 3 was untouched. This is a body-response
model, not an attempt to uniquely allocate forces to individual tires.

The baseline was the measured steering-to-yaw surface with a first-order yaw
state and held lateral velocity. During evaluation, both models received the
held-out measured longitudinal speed and actual steering (conditional inputs);
the candidate propagated lateral velocity and yaw rate recursively. This is a
best-case lateral/yaw subsystem test, not a runtime-legal complete plant or
offline simulator, because future speed was supplied from restricted ground
truth rather than predicted from allowed inputs.

| Horizon | Baseline/candidate `v_y` RMSE (m/s) | Baseline/candidate yaw RMSE (rad/s) |
| ---: | ---: | ---: |
| 25 ms | 0.00091 / 0.00176 | 0.02930 / 0.05337 |
| 125 ms | 0.00563 / 0.00386 | 0.18997 / 0.14155 |
| 250 ms | 0.00836 / 0.00649 | 0.28639 / 0.15653 |
| 500 ms | 0.00602 / 0.03977 | 0.26926 / 0.73259 |
| 750 ms | Diverged | Diverged |

The candidate improved yaw only at 125/250 ms, worsened the first step and
500 ms, and diverged before 750 ms. Lateral velocity also regressed at the
first step and after 250 ms. It fails the predeclared requirement that both
states improve by at least 20% at three horizons with no horizon more than
10% worse. **Reject this model and do not tune it against repetition 3.** The
focused evaluator is
[`evaluate_open_plane_coupled_lateral_model.py`](../../tools/evaluate_open_plane_coupled_lateral_model.py).

The paired early-sensor check on the same transition capture also failed to
add held-out predictive value. It used only early IMU yaw/lateral acceleration,
measured steering/throttle and rear encoder rates to predict later yaw gain;
whole repetitions were held out. Against a steering/sign/sweep-direction
condition mean, pooled error changed from 0.0117 to 0.0119 1/m and only one of
three repetition folds improved. This rejects that simple linear correction,
not every possible history model. The controlled profile's up/down label is
not a runtime state; future models must derive memory from measured/predicted
state and actual input history. Specifically, features were median IMU yaw
rate, lateral acceleration, actual steering, throttle feedback and each rear
wheel's 100 ms angle slope from phase start +50 to +250 ms; the target was
ground-truth yaw gain from +700 to +1,000 ms. A ridge correction (penalty 0.5)
was trained on two repetitions and evaluated on the third, rotating the held
out repetition. The baseline was a train-only mean for steering sign,
up/down sweep direction and commanded steering magnitude. This is recorded as
an exploratory diagnostic, not a reusable general-purpose test or a model
promotion result.

## Rear wheel-rate measurement audit — 2026-09-27

The 4 s dwell bag contains 7,989 messages on each rear encoder at 39.971 Hz.
`JointState.position` is populated; `velocity` and `effort` are empty. A
100 ms difference of cumulative angle yields median wheel rates 69.95 and
69.93 rad/s (left/right); a 25 ms difference is noisier. This confirms rear
angular rate is measurable from permitted encoders, but not directly supplied
as a rate. No front wheel-rate signal exists in the ROS bag. Full-window slip
statistics include transitions and must not be fitted as steady tire
parameters. Values, serialized collider settings and hidden-state inventory
are in [`UNITY_SIMULATOR_ASSET_AUDIT.md`](UNITY_SIMULATOR_ASSET_AUDIT.md).

These results supersede the proposed repeated 4 m/s sweep and slip-only fit
above. Do not repeat the same dwell/sweep or continue adding unconstrained
state terms.

## Why the model fits have stalled — 2026-09-27

The failure is now better localized: the available GT labels are accurate
body-level outputs, but the repeated fits did not observe the tire-level state
that selects between the measured high- and low-yaw branches. The guide's
two-piece tire spline and the loaded WheelCollider's friction landmarks and
stiffness are already known. More spline fitting against pose alone cannot
identify each wheel's time-varying local slip, contact/load state and force
allocation; several per-wheel force combinations produce the same aggregate
body acceleration and yaw moment.

The public Unity source branch uses the same Unity version and a RoboRacer
vehicle prefab whose wheel, suspension, friction, steering and drive settings
match the pinned player. Its `WheelEncoder.RPM` reads `WheelCollider.rpm`, but
the bridge sends only quantized rear encoder ticks/angles. A development-only
source build can read all four wheel RPMs, actual Ackermann angles,
`GetGroundHit` slips/contact geometry/force magnitude, and motor/brake torques
without writing to physics. The Unity public API does not expose separate
per-wheel longitudinal/lateral tire-force components. Also, the public source
Explore scene and the pinned binary's `Copy` scene are not the same named
scene, so a source-built image is only an instrumentation candidate until
same-command public-motion equivalence passes. See
[`UNITY_SIMULATOR_ASSET_AUDIT.md`](UNITY_SIMULATOR_ASSET_AUDIT.md).

For each wheel at `(x_i,y_i)` from the COM, compute body-frame contact
velocity `v_xi = u - r*y_i`, `v_yi = v + r*x_i`, rotate into that wheel's
Ackermann frame, then evaluate longitudinal/lateral slip. Feeding COM `v_y`
directly to all tires omits yaw-rate velocity at each contact point and the
wheel's steer angle. This matters most in the high-angle regime we are trying
to model.

Decision: **the blind state-term/curve-fit route is REJECTED**. Next is a
development-only telemetry patch and pinned-vs-source A/B replay; only if the
public trajectory stays within the pre-existing repeated-run envelope and
40 Hz stream quality is preserved should the internal signals become offline
labels. Fit/validate on complete excitation sequences, use GT only as offline
targets, and keep all added signals out of the competition controller. If A/B
fails, do not use that build's measurements; request the exact scene/project
or fall back to a statistically bounded aggregate plant.

## Instrumented source-player capture — 2026-09-27 — telemetry accepted, model use rejected

Run: `openplane_source_player_equivalence_retry2_20260927`

Bag: `live_runs/openplane_source_player_equivalence_retry2_20260927/run/run_0.db3`

1 kHz wheel-state capture:
`live_runs/openplane_source_player_equivalence_retry2_20260927/wheel_dynamics_source_equivalence_retry2_20260927.csv`

The player was built from the public simulator source with the development-only
read-only recorder in [`../../tools/unity/OpenPlaneWheelDynamicsCapture.cs`](../../tools/unity/OpenPlaneWheelDynamicsCapture.cs).
Copy it into the simulator source project's `Assets/Scripts` and call
`Capture(this)` from `VehicleController.FixedUpdate` after actuator and
wheel-pose updates. The recorder samples `WheelCollider` state once per 1 ms
physics step into a bounded 60,000-row ring buffer, then writes on orderly
shutdown. It records all four wheel RPMs, Ackermann angles, motor/brake torque,
`forwardSlip`, `sidewaysSlip`, contact-force magnitude/normal/point, and body
pose/velocity. It does not apply forces or alter vehicle, actuator, suspension,
friction, or sensor settings. The player ran through the explore-image overlay
path in Xvfb batchmode; no GUI or no-graphics mode was used.

Public bridge acceptance passed: 26/26 phases completed, 24/24 steering probes
valid with matched starts, command rate 39.66 Hz, all recorded sensor/odom/
encoder streams 39.981–39.982 Hz, p95 gaps 26.18–26.25 ms, maximum gaps below
31 ms, zero collisions, and zero in-run bridge timing faults. The capture did
not obstruct the measured 40 Hz public stream. However, this does **not** prove
that the source-built player reproduces the pinned player's high-angle
dynamics. Against the pinned same-profile repeats, its measured low-yaw
response at 0.42–0.50 rad is below the previously observed low-response values;
the 0.30 rad response is close. Consequently, its internal data locate
candidate mechanisms but are rejected as quantitative labels for the pinned
player.

Within the source build, steady 2.27–2.28 m/s samples show this association:

| Commanded angle | Median yaw gain | Mean front `|Sy|` | Mean rear `|Sy|` | Mean rear `|Sx|` |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 rad | 2.91 1/m | 0.0065 | 0.0017 | 0.0787 |
| 0.42 rad, low-yaw branch | 1.49 1/m | 0.1468 | 0.0009 | 0.0560 |
| 0.42 rad, high-yaw branch | 2.88 1/m | 0.0097 | 0.0034 | 0.1256 |
| 0.46 rad | 1.37 1/m | 0.1730 | 0.0010 | 0.0572 |
| 0.50 rad | 1.26 1/m | 0.2003 | 0.0010 | 0.0581 |

The source-built player's yaw gain falls about 57% from 0.30 to 0.50 rad,
while the front-slip proxy rises about 31x; rear lateral-slip proxies remain
near zero. This is direct evidence that front-wheel lateral slip tracks the
high-angle yaw-authority loss in this build, and that the 0.42 rad response
can occupy distinct branches. It is not proof of causation or transferability
to the pinned binary. The branch pair also rules out fitting a
steering-angle-only curve. Per-wheel longitudinal slip and contact/load
geometry must be included when identifying combined-slip and load-transfer
effects. In the two retained 0.42 rad bouts, rear mean `|Sx|` was 0.056 on the
low-yaw branch and 0.126 on the high-yaw branch. Contact-force magnitudes
`(FL, FR, RL, RR)` were `(5.36, 9.20, 6.17, 10.08) N` and
`(11.37, 4.06, 11.43, 3.98) N`, respectively: their left/right distribution
reversed with turn direction, with a larger transfer in the high-lateral-
acceleration bout. Each entry above is a time-sample median from one stable
bout; the two-branch contrast is one bout per branch, not independent
statistical replication. The ROS bag has three repeats but does not expose
these internal wheel labels.

The pinned image is an IL2CPP player, not a managed-assembly player. Its
`GameAssembly.so` is 99,868,608 bytes and its global IL2CPP metadata is
18,458,992 bytes. The source-built development player's library is
569,883,160 bytes and its metadata is 13,887,812 bytes; both the metadata hash
and `ScriptingAssemblies.json` differ. Therefore, swapping only the
source-built `GameAssembly.so` into the pinned image would pair incompatible
generated code and metadata. Do not attempt that shortcut. Exact-build source
or a source revision/toolchain that recreates the pinned metadata is required.

Unity's `WheelHit.force` is a **magnitude**, not a signed tire-force vector.
The public [`WheelHit` API](https://docs.unity3d.com/cn/6000.0/ScriptReference/WheelHit.html)
provides slip, rolling/sideways directions, contact normal, and force magnitude,
but not per-wheel `Fx`/`Fy`. The [`WheelFrictionCurve` API](https://docs.unity3d.com/kr/current/ScriptReference/WheelFrictionCurve.html)
describes the slip-to-force curve shape. Thus this capture reveals the missing
wheel-local state but cannot directly label that force spline. Aggregate GT
acceleration and yaw supply at most three planar force/moment constraints per
instant for eight tire-force components. Shared-curve constraints and
independent excitation are needed to make a fit identifiable; contact-force
magnitude and normal may constrain load effects but do not recover the
tangential force split by themselves.

Decision: **telemetry capture IMPLEMENTED; source/pinned equivalence REJECTED
for model promotion; no MPC or physics change.** Do not fit deployed model
parameters from this CSV. The next identifying step requires the exact
project/scene build used for the pinned player, or a source/player pair whose
high-angle command-response traces pass the existing repeat-envelope gate.
Then capture balanced signs/repetitions over independently varied speed,
steering, throttle/braking, and transient inputs; estimate shared tire curves
subject to wheel-load and planar force/moment constraints; and score whole
held-out sequences with recursive horizon rollouts. If exact-build telemetry
is unavailable, retain the current statistically bounded body-response model
and label per-wheel force parameters non-identifiable from current data.

## Rear encoder observability for longitudinal odometry — 2026-09-27 — REJECTED as a standalone speed fix

An offline analysis reused the existing matched-start high-steering bags at
2.2, 3, 4 and 5 m/s; no new simulator run or runtime change was made. Rear
wheel angular rates were estimated from encoder-angle differences over 100 ms,
then converted using the guide's 0.059 m radius. The target was GT wheel-local
longitudinal contact speed, computed from COM speed, yaw rate and rear track
offset. The current YAML speed-scale table was applied offline. This is a
sensor-model screen, not a replay of the complete C++ odometry observer
(including its gates and IMU propagation).

Raw rear surface speed was not a sufficiently accurate body-speed substitute
in the high-steering blocks. Across the 24 phase medians at each speed, median
/ 95th-percentile absolute error was 0.187/0.312 m/s at 2.2 m/s,
0.150/0.186 at 3 m/s, 0.130/0.138 at 4 m/s and 0.123/0.137 at 5 m/s. This
residual is consistent with combined tire slip and dynamic effects; it is not
evidence that the published wheel radius is wrong. Sparse settled straight
windows yielded apparent effective radii of 0.05745, 0.05426 and 0.05801 m at
2.2, 3 and 5 m/s, respectively, with no qualified 4 m/s window. Those single
phase estimates are contaminated by slip and do not identify physical radius.
They do show that 0.0325 m is not supported as the encoder's effective radius.

Applying the existing speed-scale table changed phase-median error only
slightly: p50/p95 became 0.181/0.289, 0.150/0.167, 0.132/0.137 and
0.118/0.132 m/s at 2.2/3/4/5 m/s. It marginally regressed the 4 m/s median
and leaves every high-steering median above 0.1 m/s. The table therefore does
not explain or remove high-angle common-mode slip. A left/right differential
feature (`v_right-v_left-yaw_rate*track`) also failed leave-one-repetition-out
prediction of the remaining common-mode speed error: it did not achieve the
predeclared 10% RMSE improvement in every speed group, and fitted coefficient
signs varied across repetitions/speeds. Do not add this correction to odometry.

Decision: **rear encoder rate as the missing high-angle body-speed signal is
REJECTED; no odometry configuration or controller input changed.** For a
useful compliant speed estimate, the next candidate must predict common-mode
slip from causal permitted history (encoder rates, IMU acceleration/yaw,
measured throttle/steering, and observer state), then pass whole-run held-out
state replay without future GT inputs. Independently, exact-build per-wheel
signals are still needed to attribute that error to tire forces rather than
motor/encoder/actuator dynamics.

## High-speed steering transition capture — 2026-09-27 — partial, not promoted

Run: `openplane_isolated_highspeed_surface_20260927`

Bag: `live_runs/openplane_isolated_highspeed_surface_20260927/run/run_0.db3`

The pinned-player capture completed all 60 matched probes at 4.5 m/s (three
repetitions over ten steering magnitudes and both signs). It then completed
49 usable matched blocks at 6.5 m/s before an emergency speed cutoff during
recovery from repetition 3 at +0.35 rad; the 7.5 m/s block was never run.
The complete run is rejected for the predeclared cross-speed model gate. The
finished 4.5 m/s subset is retained as exploratory per-speed evidence, not
promoted as a complete high-speed model. Across the bag, odometry, IMU,
steering and both rear encoders remained at 39.977 Hz (p95 gap 25.6 ms,
maximum 48.5 ms), with zero collision-count changes and zero bridge timing
faults.

At 4.5 m/s, the held-out third repetition reproduced a sharp high-angle
transition: yaw gain fell from 2.925 1/m at 0.15 rad to 1.199 1/m at 0.20
rad. Net lateral acceleration fell from 0.890 g to 0.496 g. The reconstructed
front-wheel lateral-slip proxy rose from 0.0135 to 0.126; the rear proxy stayed
near 0.003. The measured lateral-force ratio is 0.557, close to the guide's
normalized tire-curve drop from its 1.0 peak to 0.5 asymptote. The effect was
bilaterally symmetric and repeated across the three runs. This is strong,
guide-consistent evidence for front-tire post-peak saturation as a mechanism
behind this response cliff, but the slip is reconstructed from chassis
kinematics, not read from Unity's tire contact solver; it is not causal
proof.

At 6.5 m/s, the response was different: the front-slip proxy was already
0.113 at 0.15 rad and 0.165 at 0.20 rad, while lateral acceleration remained
about 0.55 g at both settings. Yaw gain decreased from 0.845 to 0.630 1/m,
but this is consistent with an approximately saturated lateral-acceleration
plateau, not a drop in lateral force. Thus the observed high-angle model is
speed- and drive-state-conditioned; one steering-only taper cannot represent
both regimes. Throttle feedback was about 0.185 at 4.5 m/s and 0.268 at 6.5
m/s, so speed and longitudinal drive demand were not independently varied.

The 4.5 m/s shape-preserving curve, trained on repetitions 1--2, predicted
the 16 held-out interior-angle/sign combinations in repetition 3 with
0.0238 1/m yaw-gain RMSE and 0.0568 1/m maximum error (configured-law RMSE
1.9714 1/m), using the measured longitudinal speed component rather than the
target speed.

A separate cross-speed holdout now tests whether the accepted 3-to-5 m/s
lateral-acceleration interpolation predicts that 4.5 m/s response. The
complete 4.5 m/s group was extracted from this bag even though the later
6.5 m/s recovery aborted; the run itself remains incomplete and is not
promoted. Training used only the independent 3.0 and 5.0 m/s bags. The
4.5 m/s group spans 4.446--4.486 m/s measured forward speed. At the
withheld 0.35 and 0.42 rad steering magnitudes (both signs, three repetitions;
12 points), acceleration-space speed interpolation achieved 0.00889 1/m
yaw-gain RMSE, 0.01235 1/m maximum error, and 0.01507 rad/s yaw-rate RMSE.
This is a 99.6% reduction from the configured yaw law (2.12289 1/m RMSE);
all local yaw-rate sensitivities were positive and finite (0.622--0.702
1/s). Repeated-condition yaw-gain standard deviation was at most
0.00055 1/m. Direct linear interpolation of yaw gain failed (0.08583 1/m
RMSE), supporting interpolation in lateral acceleration instead. This
independent-speed result passes the predeclared steady-state gate, but it
does not validate the 0.15--0.20 rad response cliff, transient recovery,
per-wheel force decomposition, or MPC closed-loop rollouts. The evaluator
accepts a complete quality-gated speed subset from an otherwise aborted bag
via `--partial-middle-speed`; it never treats the full capture as successful.

However, the measured yaw-rate-versus-steering slope is negative
across the 0.15--0.20 rad transition. The existing cross-speed acceptance
gate only passed for the withheld 0.35/0.42 rad points; that does not remove
the negative-sensitivity branch at 0.15--0.20 rad. The 6.5/7.5 m/s cross-speed
holdout is incomplete. Do not treat the fit as an MPC-ready model or retune
weights from it. Preserve the negative finite difference as a physical
observation rather than forcing a monotone curve.

The recovery event also exposed a probe-design defect: after +0.35 rad at
6.5 m/s, commanding zero steering while continuing the speed-hold loop was
followed by yaw reversal to about 3.2 rad/s, lateral body speed near 9.3 m/s,
and rear encoder rates rising from about 114 to 215 rad/s. The safety cutoff
then stopped the experiment. There was no collision or transport fault. This
is consistent with a real spin/wheel-spin transient, though body-level streams
alone cannot identify its force cause. Do not repeat the same high-throttle
straight-reset transition; model/record the recovery as a transient condition
or use a measured low-risk recovery protocol, and retain the emergency cutoff.

Conclusion: **partial response data accepted as a mechanism clue; full
cross-speed capture and MPC promotion rejected.** A precise physical model
still requires exact-pinned-build read-only wheel telemetry (front and rear
wheel rates, actual Ackermann angles, native forward/sideways slip and dynamic
wheel loads) or a source/player pair that passes high-angle behavior
equivalence. The guide and Unity APIs establish the tire-curve form and
reported contact quantities, but not a unique decomposition of net chassis
force into four wheel forces. See the [AutoDRIVE vehicle dynamics guide,
§1.3.2](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics),
Unity's [`WheelFrictionCurve`](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/WheelFrictionCurve.html),
and [`WheelHit`](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/WheelHit.html)
API descriptions.

## Why the high-angle model is still not MPC-ready — 2026-09-27

Three held-out results now distinguish the missing pieces:

| Candidate | Held-out result | Decision |
| --- | --- | --- |
| Interpolate the 4.0 and 6.5 m/s response curves across speed through 4.5 m/s, specifically 0.15–0.25 rad | Yaw-gain RMSE 0.655 1/m (gate 0.10); sensitivity has wrong signs in multiple intervals | Reject this cross-speed surface in the transition band |
| Predict lateral acceleration from reconstructed front `|Sy|` | RMSE 0.1366 1/m; 92% better than configured law, but wrong local sensitivity signs | Reject front-slip-proxy-only model |
| Fit an exact 4.5 m/s steering PCHIP from repetitions 1–2, withhold 0.21/0.22/0.23 rad, score repetition 3 | Gain RMSE 0.0131 1/m, max error 0.0300, 99.3% better than configured law; local derivative signs still fail | Useful steady output fit; reject for MPC promotion |

The last result is the important distinction: the model predicts the *values*
of yaw gain through the cliff, but does not reliably predict its local
steering derivative. Its held-out derivative is wrong near 0.215 rad for both
turn directions and near 0.24 rad for one direction. One repetition-3 +0.23
rad phase also had much larger speed variation than its peers (p95 error
0.299 m/s), so this discrepancy may mix local tire curvature with speed and
transient-state variation. The evidence does not justify treating that
asymmetry as a tire law. Detailed repeatable scores are produced by
[`evaluate_open_plane_transition_speed_surface.py`](../../tools/evaluate_open_plane_transition_speed_surface.py),
[`evaluate_open_plane_front_slip_response.py`](../../tools/evaluate_open_plane_front_slip_response.py),
and [`evaluate_open_plane_transition_45_holdout.py`](../../tools/evaluate_open_plane_transition_45_holdout.py).

### What it takes to close the identification gap

The formulas alone are not enough. They define how states and forces relate;
they do not supply the instantaneous four-wheel state needed to evaluate those
equations, nor do the chassis outputs uniquely determine that hidden state.
With only GT body motion we can estimate aggregate planar force and yaw
moment. Many different combinations of four `Fx`, four `Fy`, and four dynamic
normal loads can produce the same aggregate motion. A chassis-derived slip
proxy is therefore correlated evidence, not an observed tire input. The
published WheelHit force magnitude also does not separate longitudinal from
lateral force.

The next experiment should be narrow and discriminating, not another wide
steering sweep:

1. First test the already-captured 4.5 m/s groups for phase-level speed,
   lateral velocity, yaw rate, steering history, throttle, rear `Sx`, and
   repeat spread. Use GT only as offline labels/diagnostics; fit candidate
   features from signals the compliant runtime estimator can actually supply.
2. If that replay confirms the transition ambiguity, repeat only the
   0.205–0.240 rad region with finer steering spacing, both signs, identical
   matched initial conditions and tighter measured-speed/throttle stability.
   Randomize phase order and reserve entire repetitions for validation.
3. For a physical per-wheel tire model, instrument the exact pinned simulator
   without writing to physics: per-wheel angular rate, actual steer angle,
   native forward/sideways slip, dynamic load/suspension travel, drive torque,
   contact state and synchronized body state at the physics update rate. The
   source-built player must first pass a same-command public-motion A/B gate
   against the pinned player, especially through this high-angle transient.
   If equivalence fails, its internal labels cannot be used to train the
   pinned car. If equivalent, WheelHit still supplies only force magnitude;
   obtain signed `Fx/Fy` from read-only solver instrumentation or retain
   aggregate-force identification rather than inventing a per-wheel split.
4. Fit a speed- and combined-slip-conditioned tire/vehicle model with dynamic
   loads and identified yaw inertia; quantify parameter uncertainty and
   identifiability. Validate on complete unseen command sequences and compare
   body acceleration, yaw, lateral velocity, and model Jacobian—not only
   steady output RMSE. Promote only when held-out output *and derivative*
   gates pass without extrapolation, then test in an MPC-stage rollout before
   any competition-stack change.

Until those measurements exist, the safest useful model is an empirical
body-response surface indexed by legal estimated states, with explicit
uncertainty and a bounded operating domain. Do not claim it is a recovered
four-tire force model or feed GT into the competition controller. The new
exact-speed test package is **REJECTED for MPC promotion**; its quantitative
result is retained as evidence that output interpolation alone is inadequate.

## Capture-pairing correction — BLOCKED pending Unity integration — 2026-09-27

An audit of `openplane_source_player_equivalence_retry2_20260927` found that
the wheel CSV cannot be verified as synchronized to its accompanying bag. It
has 60,000 rows but reports 50,716 overwritten samples, begins at simulation
time 50.716 s, and has neither a shared-clock timestamp nor a run identifier
in each row. More importantly, the CSV's steering sequence diverges from the
bag's sequence after the first matching +0.50 rad command. That first match is
not evidence of a paired run. Do not use this CSV and bag together for
per-wheel-versus-ROS encoder error, phase attribution, or model fitting. The
CSV remains a source-player internal observation, but its training use is
also limited by the previously failed pinned-player behavior-equivalence gate.

The development-only recorder now retains 180,000 1 ms physics samples,
covering the excitation schedule's 153 s maximum, and writes Unity real-time
plus a Unix-UTC-mapped nanosecond timestamp in every row. It also records
`Rigidbody.inertiaTensorRotation` alongside the principal `inertiaTensor`
values. Unity stores the diagonal inertia in its principal frame; without its
rotation, body-axis yaw inertia cannot be read from a component of the vector.
The paired encoder evaluator now rejects old or overwritten captures,
insufficient clock overlap, and runs whose body-speed, steering-command or
throttle-command timelines do not agree. Commands are compared directly so
actuator response delay is not mistaken for a different experiment. This
remains offline analysis; no runtime topic policy changed.

These instrumentation edits have not yet been compiled or exercised in Unity:
the exact simulator source project is not present in this workspace. The old
data stays **unpaired / not fit for the planned comparison**, and no new
per-wheel physical parameters are claimed. The next valid run must capture
wheel state from the same open-plane player session as the ROS bag, with the
recorder enabled from process start. Once pairing passes, use each wheel's
measured RPM, actual steer angle, native longitudinal/lateral slip, contact
state and contact-force/load proxy together with same-tick GT body state to
locate the high-steer branch. `WheelHit.force` still does not provide signed
`Fx` and `Fy`; do not invent their individual values from that magnitude.
Work-package status: **BLOCKED** until the exact Explore simulator source
project is available to integrate and run the read-only recorder. The offline
evaluator and bag reader are implemented; Python syntax, bag decoding, and
fail-fast rejection of the stale capture pass.

## Cross-speed steering evidence and command-integrity correction — 2026-09-27

The completed `grid` capture is at
`live_runs/openplane_grid_combo_20260927_01/run/run_0.db3`. Its 40 steering
probes covered both signs at 2.5, 4.5, 6.5 and 7.5 m/s; all passed their
speed/steering phase gates, with zero collisions and zero bridge timing faults.
This is one ordered grid, not repeated independent trials, so it supplies
descriptive evidence only. The 7.5 m/s group closes the prior speed-coverage
gap, but is not enough to promote a model.

Representative sign-paired medians from the grid:

| Target speed | Steering | Yaw gain `r/(vx tan(delta))` | Front `|Sy|` proxy |
| ---: | ---: | ---: | ---: |
| 4.5 m/s | 0.15 rad | 2.925 1/m | 0.014 |
| 4.5 m/s | 0.25 rad | 0.956 1/m | 0.179 |
| 6.5 m/s | 0.15 rad | 0.845 1/m | 0.113 |
| 6.5 m/s | 0.42 rad | 0.330 1/m | 0.408 |
| 7.5 m/s | 0.15 rad | 0.647 1/m | 0.123 |
| 7.5 m/s | 0.42 rad | 0.261 1/m | 0.420 |

The sharp authority loss between 0.15 and 0.25 rad at 4.5 m/s, and the
already-low response at 0.15 rad at 6.5–7.5 m/s, are consistent with the
guide's front lateral-force peak/asymptote landmarks (`|Sy|=0.01/0.10`). This
is still an inference: the `Sy` values are reconstructed from body motion and
steering, not native per-wheel slip. At 7.5 m/s the next useful steering
resolution must be below 0.15 rad; adding more samples only at 0.3–0.5 rad
would refine the saturated branch without locating its onset. Preserve
sign-specific repeats and hold out complete repetitions before fitting.

The same bag exposed a test-harness flaw. In the former `grid`, fixed throttle phases
were silently overwritten with zero throttle whenever speed exceeded the
phase target by 0.50 m/s. The logged 4.5 m/s coast phase began at 4.94 m/s and
then fell to 2.36 m/s; the 7.5 m/s coast phase likewise fell from 7.84 to
2.51 m/s. The guide says idle torque is simulated as braking torque, so those
are not valid fixed-throttle coast observations. The run's zero quality
failures only certify its explicitly validated steering probes; fixed-throttle
phases were not quality-checked. Future fixed-input phases now retain their
requested command, while speed-regulated phases keep their target guard and
the global 9 m/s emergency cutoff remains in force. Phase metadata records the
requested fixed command, and the bag analyzer fails if recorded commands do
not match it (p95 error limit 0.02, minimum 5 samples).

The command-integrity run repeated all 40 steering probes and passed 19/19
fixed-input command checks exactly (p95 error 0.000). It then hit the global
9.002 m/s cutoff in `combined_accel_7.5_-1`, after the steering probes had
finished. The run is therefore **aborted as a whole**, while its fully valid
steering phases remain usable. The combined phases were already non-identifying
because of strong speed change and active idle braking; they have now been
removed from the `grid` schedule rather than weakening the emergency cutoff.
`grid` is now a 48-phase steering-only surface experiment. Fixed-input checks
remain available to profiles that explicitly use fixed throttle.

Work-package status: **IMPLEMENTED** for development capture integrity; no
simulator physics, odometry, localization, MPC, or competition topic policy
changed. Final acceptance for the corrected steering grid is all 40 steering
probes valid, required streams at least 38 Hz, zero collisions and timing
faults, full 48/48 schedule completion, and no emergency cutoff. No old
coast/brake data is reused for fitting.

The corrected steering-only run passed those gates:
`live_runs/openplane_grid_steeringonly_20260927_01/run/run_0.db3`. It was run
with the pinned Explore image through the existing Xvfb batchmode launcher.
All 48/48 phases completed, all 40 steering probes were valid, and there were
zero collisions, zero bridge timing faults, and no emergency cutoff. Odom,
steering, both rear encoders, IMU and packet timing were 39.981–39.982 Hz;
p95 gaps were 25.56–25.68 ms and maximum gaps 47.91–47.99 ms. The bag is 12 MB.

The new 7.5 m/s curve is especially informative:

| `|steer|` | yaw gain (1/m) | front `|Sy|` proxy | lateral acceleration |
| ---: | ---: | ---: | ---: |
| 0.15 rad | 0.647 | 0.123 | 0.560 g |
| 0.25 rad | 0.402 | 0.226 | 0.590 g |
| 0.30 rad | 0.342 | 0.281 | 0.612 g |
| 0.35 rad | 0.303 | 0.336 | 0.638 g |
| 0.42 rad | 0.261 | 0.419 | 0.673 g |

At the shared 0.15/0.25/0.42-rad points, all three captures agree within
0.0012 1/m at 4.5, 6.5 and 7.5 m/s. At 7.5 m/s, IMU lateral acceleration
independently agrees with the odometry-derived value within 0.013 m/s² in the
reported phases, so this response is not an odometry-only artifact. However,
the speed-hold throttle also changes with target speed (about 0.18 at 4.5 m/s
and 0.31 at 7.5 m/s): this is the reproducible response of that operating
condition, not a causal speed-only tire curve. The front-slip values remain
kinematic proxies.

## Grid cross-speed model test — REJECTED for MPC promotion — 2026-09-27

The reusable grid mode in
[`evaluate_open_plane_speed_steering_surface.py`](../../tools/evaluate_open_plane_speed_steering_surface.py)
trained on the 4.5 m/s surface from `openplane_grid_combo_20260927_01` and the
7.5 m/s surface from `openplane_grid_commandfix_20260927_01`, then held out the
6.5 m/s surface from the completed steering-only run. The later-aborted source
bag is accepted only for its complete, valid 7.5 m/s steering group; its abort
occurred afterward during the fixed-acceleration segment.

Reproduce with:

```bash
python3 tools/evaluate_open_plane_speed_steering_surface.py \
  --grid-speed-holdout \
  live_runs/openplane_grid_combo_20260927_01/run/run_0.db3 \
  live_runs/openplane_grid_steeringonly_20260927_01/run/run_0.db3 \
  live_runs/openplane_grid_commandfix_20260927_01/run/run_0.db3
```

The simple yaw-gain blend scored 0.2505 1/m RMSE. Blending lateral
acceleration instead reduced held-out yaw-gain RMSE to 0.0845 1/m against the
configured law's 2.3834 1/m (96.5% reduction), meeting the predeclared output
gate of 0.10 1/m. But the optimizer-relevant derivative failed: the candidate
produced finite `d(yaw rate)/d(steering)` from -0.743 to +0.792 1/s, while the
held-out measured finite differences were positive from +0.224 to +0.738 1/s;
only 6/8 interval signs matched. This is a concrete example of why matching
state outputs alone does not make a safe nonlinear MPC model. Reject this
interpolator; do not change MPC weights or parameters from its RMSE.

The production model in `vehicle_model.c` still uses a speed-independent
steering gain of 2.95 1/m below 0.41 rad, with taper only from 0.41 to 0.46
rad. At 7.5 m/s the measured gain is 0.647 1/m already at 0.15 rad, about
78% below nominal; at 0.42 rad it is 0.261 1/m while the configured taper is
still about 2.59 1/m. This is the model mismatch capable of making the MPC
believe it has much more yaw authority than the car actually delivers.
No MPC change was made because the cross-speed candidate's Jacobian failed.

The steering grid does not isolate speed from throttle: the speed controller
uses progressively higher throttle as its target rises, and the two rear
encoders provide only a partial view of AWD tire slip. To close the causal
gap, the next model-data package must cross steering with short throttle
perturbations at matched measured speed, record actual actuator feedback and
rear encoder slip, and hold out complete runs. Use sub-0.15-rad points at
6.5–7.5 m/s to locate saturation onset, not another sweep concentrated at
0.3–0.5 rad. A full physical four-wheel fit remains blocked on synchronized
native wheel slip, per-wheel dynamic load and signed tire forces from a
behavior-equivalent pinned simulator build. Without those, fit only a bounded
legal-input body-response model, then gate both output error and local
Jacobian sign before any MPC integration.

## Low-angle high-speed onset and actuator cross-factor — 2026-09-27

Two targeted captures narrowed the response failure to a steering reversal
around 0.10–0.175 rad.

`openplane_highspeed_crossfactor_20260927_01` was stopped after 35/184 phases
when ±0.025 fixed-throttle probes separated speed by 1.20–1.22 m/s at the
6.5 m/s setpoint. The commands were delivered correctly, but that contrast
cannot identify a same-speed throttle effect. Keep the bag as negative
experimental-design evidence, not a model fit.

`openplane_highspeed_crossfactor_20260927_02` used ±0.003 pedal offsets,
matched starts, 0.05–0.25 rad signed steering levels, three repetitions, and
the existing 9 m/s cutoff. It completed all 90 probes at 6.5 m/s and one
repetition at 7.5 m/s, then aborted during the 7.5 m/s return-to-zero recovery
at 9.38 m/s. There were zero collisions and zero bridge timing faults;
streams were 39.978 Hz (p95 gaps 25.5–25.7 ms). The final recovery developed
about 9.38 m/s lateral velocity and -3.3 rad/s yaw rate. Do not repeat this
7.5 m/s zero-steer reset; that group has no repeated holdout.

At 6.5 m/s, speed-hold steering response was symmetric and stable across all
three repetitions. Repetitions 1–2 gave yaw gains 2.959, 2.931, 2.374, 1.343,
0.844, 0.715, 0.630, 0.564 and 0.513 1/m at steering magnitudes 0.05, 0.075,
0.10, 0.125, 0.15, 0.175, 0.20, 0.225 and 0.25 rad. The response drop starts
far before the model's 0.41 rad taper: gain is already about 20% below 2.95
at 0.10 rad, falls 43% by 0.125 rad, and falls another 37% by 0.15 rad. The
front-slip proxy rises from about 0.032 at 0.10 rad to 0.076 at 0.125 and
0.113 at 0.15, crossing the guide's tire-curve peak/asymptote region. This is
strong saturation evidence, but the slip is reconstructed from body motion,
not read from the tire solver.

A steering PCHIP trained on repetitions 1–2 predicted all 18 repetition-3
signed yaw-gain points with 0.0213 1/m RMSE and 0.0140 rad/s yaw-rate RMSE.
The production gain scored 1.8080 1/m and 2.3901 rad/s. The PCHIP matched
14/16 measurable local yaw-rate derivative signs; the production monotone
model matched 10/16. Both turn directions reproduce the remaining error at
0.15–0.175 rad: measured `d|r|/d|steer|` is -0.262/-0.270 s^-1, while PCHIP
predicts +0.126/+0.118 s^-1. This accurate steady-output curve is **rejected
for MPC promotion** because it still has the wrong optimizer-relevant
Jacobian at the local minimum.

Reproduce the partial-bag group analysis (the evaluator returns failure for
whole-run acceptance because the 7.5 m/s recovery was aborted):

```bash
python3 tools/evaluate_open_plane_highspeed_crossfactor.py \
  live_runs/openplane_highspeed_crossfactor_20260927_02/run/run_0.db3
```

The ±0.003 pedal test met the 6.5 m/s speed-overlap threshold in 18/18 pairs
(median speed gaps 0.143–0.148 m/s; feedback throttle separation about 0.006).
A throttle feature reduced repetition-3 yaw-gain RMSE from 0.0562 to 0.0483
1/m (14.1%), with consistent coefficient sign in both turn directions. But
using measured speed instead scored 0.0488 1/m. That 0.0005 difference is
negligible: throttle and speed remain confounded, so **no independent
throttle/tire-force coefficient is accepted**. The evidence says the
response is sensitive to the speed state near the steering cliff; any
remaining throttle/slip contribution must be separately identified.

At 7.5 m/s, the single completed repetition measured yaw gain 1.555, 0.849,
0.647, 0.492 and 0.402 1/m at 0.10, 0.125, 0.15, 0.20 and 0.25 rad. The
shared 0.15/0.25 points agree with independent grid captures, but this one
repetition does not validate the local Jacobian or a throttle coefficient.

### What closes the gap

The production stage in `f1tenth_mpc/src/vehicle_model.c` uses a first-order
yaw response with gain 2.95 1/m until 0.41 rad, and copies lateral velocity
unchanged into the next state. It cannot represent the observed early front
tire-force peak, negative steering-response branch, shallow recovery, or
transient lateral-velocity evolution. Weight sweeps cannot repair those
missing dynamics.

The next identification package is limited to 6.5 m/s until a safe high-speed
recovery is designed. Refine 0.145–0.180 rad in 0.005 rad steps, both signs,
and hold out a complete repetition. Then fit a differentiable body-level
transition using legal state/input variables (`u`, `v`, `r`, measured steering,
throttle feedback, rear encoder rates), with GT only as offline labels. Score
one-step and recursive 0.25–0.75 s rollouts and the measured Jacobian signs
before any MPC integration. Do not fit four individual tire-force curves
from aggregate GT; native per-wheel labels still need a behavior-equivalent
build of the pinned simulator. Status: **6.5 m/s steady output curve
implemented and quantified; model promotion rejected; 7.5 m/s capture
rejected as an unsafe reset protocol; exact-build per-wheel instrumentation
remains blocked**.

## Fine high-speed steering minimum — 2026-09-27

The 6.5 m/s follow-up `openplane_transition_65mps_20260927_01` resolved the
0.145–0.180 rad region at 0.005 rad spacing, both steering signs, three
randomized repetitions, with repetition 3 held out. It completed all 50 phases
(48/48 probes valid), with 39.977 Hz odometry/steering/encoder/IMU/timing
streams, p95 receipt gaps of 25.53–25.62 ms, maximum gaps below 48.9 ms, no
collision-count increase, no bridge timing faults, and no abort. The bag is
`live_runs/openplane_transition_65mps_20260927_01/run/run_0.db3` (about 18 MB).

The held-out yaw-rate magnitude forms a narrow, repeatable trough: about
0.844 rad/s at 0.145 rad steering, 0.830 at 0.150, 0.819 at 0.160, then
recovers to 0.825 at 0.180. This explains why the old 0.15-to-0.175 average
slope looked only slightly negative: it hid a sharp negative branch followed
by a positive branch. Chassis lateral acceleration stays near 0.55 g through
the interval, so yaw response alone does not identify individual tire forces
or the front/rear force split. The reconstructed slip proxy rises from about
0.108 at the front at 0.145 rad to 0.145 at 0.180, while the rear proxy stays
near 0.0038. That is consistent with the front tires operating beyond the
guide's 0.10 lateral-slip asymptote while the rears remain lightly slipped,
but these are kinematic proxies, not simulator tire-solver measurements.

The predeclared gain-surface test trained a PCHIP on repetitions 1–2 and
predicted the held-out yaw gain with 0.0001 1/m RMSE, but it got only 12/14
resolved local yaw-rate Jacobian signs right. The sign misses occur near the
trough and are sufficient to **reject this gain parameterization for MPC**.
An offline diagnostic then parameterized the response directly as
`|yaw_rate|(abs(steer))`, rather than interpolating `yaw_rate/(v*tan(steer))`
and reconstructing yaw rate. On the same holdout this direct PCHIP had
0.00016 rad/s output RMSE and matched 14/14 resolved derivative signs. That
is a promising formulation because the MPC predicts yaw rate itself, but this
alternative was considered after examining the original holdout; it is
therefore only a hypothesis at this point in the analysis. It was then frozen
from repetitions 1–2 and checked on a separate randomized simulator run; that
independent confirmation is recorded below. It still must be tested as part
of a state transition including `u`, `v`, `r`, actual steering, throttle
feedback, and rear wheel rates before any controller integration.

The reproducible capture gate is `tools/evaluate_open_plane_transition_65mps.py`.
Its result is intentionally nonzero/rejected because the predeclared gain
surface fails the Jacobian criterion. The primary technical conclusion is
that interpolating a derived steering gain can erase the optimizer-relevant
local curvature even when its held-out pointwise error is tiny; fit and score
the predicted state derivative directly. Per-wheel tire-force identification
still requires synchronized simulator contact slip/load/force labels or
carefully designed independent wheel-force excitation, not merely the public
body-state formulas and GT pose.

## Independent confirmation of the direct yaw surface — 2026-09-27

The direct `|yaw_rate|(abs(steer))` PCHIP was frozen from repetitions 1–2 of
`openplane_transition_65mps_20260927_01` and evaluated without refitting on a
new simulator start, a new seed/order, and all three repetitions of
`openplane_transition_65mps_confirm_20260927_01`. The second run completed
50/50 phases and 48/48 valid, matched probes; odometry, steering, encoders,
IMU, and packet timing were 39.978 Hz (p95 gaps 25.53–25.67 ms, max below
48.4 ms), with zero collisions, zero timing faults, and no abort.

The frozen response predicted all 48 signed test probes with 0.00022 rad/s
RMSE and matched 42/42 local Jacobian signs whose measured magnitude was at
least 0.10 s^-1. The narrow minimum and subsequent recovery repeated. This
passes the predeclared independent **steady response** gate and accepts that
surface only as the yaw-rate component/candidate of the next transition
model. It is not an MPC-ready plant: it is validated at one speed from
near-straight starts and has not modeled `v`/`r` history, transient buildup,
throttle/encoder dependence, or out-of-band steering/speed. Next acceptance
must use an untouched full-sequence holdout and beat the production model on
one-step plus recursive `u,v,r` predictions at 25–750 ms, with significant
input-Jacobian signs correct. No controller source or simulation physics has
been changed.

Reproduce the cross-run gate with:

```bash
python3 tools/evaluate_open_plane_transition_65mps_external_holdout.py \
  live_runs/openplane_transition_65mps_20260927_01/run/run_0.db3 \
  live_runs/openplane_transition_65mps_confirm_20260927_01/run/run_0.db3
```

## What the published per-wheel equations can identify

The guide's equations are useful as a model structure, but not as enough data
to recover each tire curve. It specifies a per-tire slip-to-force curve,
suspension loads, and Ackermann steering; its lateral slip equation only
becomes wheel-specific once velocity is expressed in that wheel's frame. With
body velocity `(u,v)` at the CG, yaw rate `r`, and wheel location `(x_i,y_i)`
relative to the CG, planar rigid-body kinematics give

```text
v_x,i(body) = u - r*y_i
v_y,i(body) = v + r*x_i
```

Rotate the front-wheel velocities by their actual Ackermann angle `delta_i`
to get `(v_x,i(tire), v_y,i(tire))`; then compute each tire's slip angle from
those tire-frame components, and longitudinal slip from that wheel's surface
speed `R*omega_i`. Applying the guide's simpler `v_y/|v_x|` formula to the
single chassis twist without the yaw-at-wheel and steering transforms would
give all wheels the same slip, which cannot represent the guide's own
per-wheel model. These transforms are a rigid-body derivation, not a claim
about undocumented simulator implementation details.

Even with GT position and orientation, the public run gives aggregate body
motion rather than the four tire-force vectors. In planar form the measured
acceleration constrains only the net forces and yaw moment:

```text
m*(u_dot - r*v) = sum(F_x,i in body frame) + other longitudinal forces
m*(v_dot + r*u) = sum(F_y,i in body frame) + other lateral forces
I_z*r_dot       = sum((x_i-x_CG)*F_y,i - (y_i-y_CG)*F_x,i) + other moments
```

That is three aggregate equations for up to eight unknown tire-force
components, before solving for individual normal loads, suspension state, or
front-wheel spin. GT helps estimate the left sides; it does not add the
missing per-wheel equations. The published vehicle table gives useful
geometry, total/sprung/unsprung mass, spring/damper constants, and lateral
curve landmarks (peak at slip 0.01, force 1.0; asymptote at slip 0.10, force
0.5), but does not publish the exact simulator's per-wheel time histories or
all tire-stiffness/spline coefficients. The [AutoDRIVE dynamics guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics)
therefore constrains plausible model structure and limits, not a unique
identified tire law.

Our open-plane bags contain body odometry/IMU, GT IPS, actual steering and
throttle feedback, and only the two rear encoder measurements. They do not
contain four synchronized `(Sx_i,Sy_i,Fz_i,Fx_i,Fy_i)` records. The Unity
[`WheelHit` API](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/WheelHit.html)
documents wheel contact, force magnitude, direction, and forward/sideways
slip, but that is only actionable if the pinned AutoDRIVE vehicle actually
uses that API and exposes the values; the public guide does not promise those
values as ROS topics. Do not treat Unity's API as proof of the AutoDRIVE
implementation.

There are two valid routes forward:

1. For a physical per-wheel fit, add read-only instrumentation to a
   behavior-equivalent build of the pinned simulator and record per-wheel
   contact state, local slip, spin, normal load, and signed tire-frame force at
   the physics tick. Check aggregate force/moment balance against GT/IMU, and
   identify load transfer and each curve only after that balance closes. This
   changes observability, not the physics; any non-equivalent player build is
   not valid evidence.
2. If source-equivalent instrumentation is unavailable, fit a lower-order
   differentiable transition from legal runtime inputs
   `(u,v,r, steering_feedback, throttle_feedback, rear_encoder_rates)` to the
   next `(u,v,r)`, with GT/IMU used only for offline labels. The fine 6.5 m/s
   test shows why the output and its steering derivative must be scored
   directly; a low pointwise yaw-gain error does not guarantee the correct
   optimizer gradient.

For either route, hold out a complete randomized repetition/run, compare
one-step and recursive 25–750 ms state predictions, and check all significant
local input-Jacobian signs before changing the MPC. This is the decision gate;
more weight sweeps cannot compensate for an unobserved or wrongly shaped
plant response.

## Rear-wheel slip check at the 6.5 m/s yaw trough — rejected

To see whether a competition-available wheel-speed signal explains the fine
6.5 m/s trough, the existing condition-centered rear-slip evaluator was run
on both the original and independent-confirmation bags. It derives rear
angular rates from the two encoder-angle streams and compares their
longitudinal-slip proxy with yaw gain within each matched speed/steering/sign
condition. Each run fits repetitions 1–2 and holds out repetition 3.

| Bag | Held-out blocks | Condition-only RMSE | With rear-slip | Improvement | Fitted coefficient |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 16 | 0.0136 1/m | 0.0135 1/m | 0.3% | +5.394 |
| Independent confirm | 16 | 0.0136 1/m | 0.0134 1/m | 1.0% | +2.061 |

The existing predeclared criterion was at least 10% held-out RMSE reduction in
each capture with a stable coefficient sign. The improvement is far below the
criterion, despite a positive sign in both fits, and coefficient magnitude is
not stable. **Reject mean rear longitudinal slip as an explanatory state for
the 6.5 m/s narrow yaw trough.** This is not a rejection of rear encoders for
odometry or of all wheel-slip effects; it says this feature does not explain
this response once speed and steering condition are accounted for. The test
uses simulator odometry as an offline speed label when constructing slip, so
it is an optimistic diagnostic, not a competition-time predictor.

Reproduction:

```bash
source /opt/ros/jazzy/setup.bash
python3 tools/evaluate_open_plane_rear_slip_effect.py \
  live_runs/openplane_transition_65mps_20260927_01/run/run_0.db3 \
  live_runs/openplane_transition_65mps_confirm_20260927_01/run/run_0.db3
```

## Rootless simulator recovery and independent 3 m/s repeat — 2026-09-27

The host's rootful Docker/containerd stores are corrupt (containerd bbolt
snapshotter free-page/ancestor mismatch; dockerd libnetwork invalid bbolt
page). They were left untouched. A user-level rootless Docker service is now
installed, enabled, and active, with the CLI `rootless` context. The existing
`tools/docker_env.sh` prefers `/run/user/1000/docker.sock`, so the established
launch scripts continue to work without changing their semantics. Only the
pinned official Explore and API images were pulled into the separate
`~/.local/share/docker` store.

The official Explore image was launched twice with
`SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode
./tools/start_simulator.sh`; both launches initialized Vulkan on the GTX 1080
and PhysX without segfaulting. No GUI or `-no-graphics` path was used. The
player reports missing optional lidar/search plugins and partial convex hulls
for its wheel meshes, but these are non-fatal warnings: telemetry, actuation,
and the experiment completed. The first high-angle smoke bag had an overall
39.699 Hz rate because its measured interval includes a 548 ms boundary gap;
the repeat had 39.977 Hz, p95 gaps 25.51–25.60 ms, and no in-run gap above
47.6 ms. Both report zero collisions, zero recorded bridge faults, and a
completed schedule.

Those `high_angle_boundary` bags are **not steady-map training data**. That
profile changes steering without settling; phase validation checks speed and
steering but not initial yaw. For example, a +0.46 rad phase began at about
2.85 rad/s after a +0.42 rad phase, yielding an artificially high yaw-gain
estimate. Use the matched-start `isolated_force_*` profiles for steady
response identification instead.

The fresh 3 m/s matched-start capture is
`live_runs/openplane_rootless_isolated3mps_20260927_01/run/run_0.db3` (seed
20260927). It completed 26/26 scheduled phases, including 24/24 scored
steering probes, with no quality failures or abort. Odom, steering, both rear
encoders, IMU, and packet timing each ran at 39.975 Hz; p95 receipt gaps were
25.59–25.73 ms and maximum gaps 48.9–49.0 ms. There were zero collisions and
zero timing-fault messages. A shape-preserving cubic trained on two randomized
repetitions and scored on the remaining repetition at the 0.42/0.46 rad
intermediate points produced 0.02549 1/m yaw-gain RMSE (max error 0.04637),
versus 1.06342 1/m for the current fixed steering taper. This is a useful
within-run holdout result, but the evaluator's 5 m/s acceptance gate does not
apply at 3 m/s, and this test does not establish cross-run or recursive
`u,v,r` prediction.

The mean measured 3 m/s yaw gain is 1.567 1/m at ±0.30 rad, 1.10 at ±0.42,
0.997 at ±0.46, and 0.949 at ±0.50. The calculated front slip-angle proxy
rises from about 0.15 to 0.37 across those levels, while the rear lateral
proxy remains about 0.0015–0.0020. That pattern is consistent with front-tire
lateral authority loss; it is only a kinematic proxy, not a per-wheel force
measurement. It argues for a measured differentiable steering/speed response
surface, not an assumed sine. Keep it offline until a frozen model passes an
independent complete-run holdout and recursive state/Jacobian gates.

The previous runner sent SIGTERM while a final simulator callback could still
be active, producing a `publisher's context is invalid` message labelled as a
fatal timing fault only after bag recording had stopped. The bridge now marks
shutdown first, stops its sender/listener, and unwinds through normal cleanup.
The repeat's bridge log contains no fatal line, while its bag still reports
zero timing faults. This changes process teardown only; simulator physics and
driving behavior are unchanged.

### Independent-run test: model yaw curvature, not only yaw gain

The prior 2026-09-26 3 m/s capture was used as training data only at ±0.30 and
±0.50 rad. The fresh 2026-09-27 run was held out at ±0.42 and ±0.46 rad
(12/12 blocks, three repetitions per cell). A PCHIP of
`K_yaw = r/(u*tan(delta))` reduced yaw-gain RMSE to 0.07881 1/m and yaw-rate
RMSE to 0.10827 rad/s, versus 1.06342 1/m and 1.41134 rad/s for the current
production taper. However, the gain PCHIP predicted negative local yaw-rate
derivatives on both signs (-0.542/-0.600 s^-1) while the held-out finite
differences were positive (+0.490/+0.126 s^-1). **Reject the gain PCHIP for
optimizer use despite its good pointwise fit.**

Using the same training and holdout, directly interpolate
`q = |r|/u = |tan(delta)|*K_yaw` instead. Its held-out yaw-rate RMSE was
0.03195 rad/s; its local derivatives were +0.223 and +0.176 s^-1, with the
correct positive sign on both turn directions. This is encouraging evidence
that the optimizer should fit the response coordinate it actually uses,
including its derivative—not a claim of a universal sine or tire law. The
test is still confined to a 3 m/s steady response and the 0.30–0.50 rad
interval. It does not validate speed interpolation, transients, recursive
`u,v,r` rollout, or MPC behavior; keep it offline until those holdouts pass.

The direct-curvature result was computed from the two bags using the existing
phase medians and PCHIP helper. Recreate the source blocks with
`tools/evaluate_open_plane_rear_slip_effect.py`'s `_load_blocks`; the independent
test bag has its stream, collision, and timing gates summarized above. Do not
use `high_angle_boundary` bags as steady-state validation because their
per-phase gate omits residual-yaw matching.

## Independent 3–5 m/s captures and recursive cross-speed test — 2026-09-27

New pinned-Explore captures:

| Run | Target | Valid probes | Odom rate | Collisions / timing faults | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `openplane_rootless_isolated3mps_20260927_01` | 3 m/s | 24/24 | 39.975 Hz | 0 / 0 | Complete |
| `openplane_rootless_isolated4mps_20260927_01` | 4 m/s | 24/24 | 39.974 Hz | 0 / 0 | Complete |
| `openplane_rootless_isolated5mps_20260927_01` | 5 m/s | 24/24 | 39.969 Hz | 0 / 0 | One 100.14 ms response gap; exclude from strict cadence holdouts |
| `openplane_rootless_isolated5mps_repeat_20260927_02` | 5 m/s | 24/24 | 39.968 Hz | 0 / 0 | Complete; maximum gap 48.45 ms |

Each bag records the matched-start experiment's minimal dynamics topics. The
simulator used the repo's pinned Explore image through the established
`tools/start_simulator.sh` Xvfb `-batchmode` path; no GUI, `-no-graphics`, or
simulator physics changes. Every run completed 26/26 scheduled phases with
zero collision, bridge-timing, or experiment-quality failures.

The first 5 m/s run's single long interval was a real source-response delay,
not a database artifact: request intervals stayed near 25 ms and request
sequence 724 followed 723, but three responses in one `-0.46 rad` block had
53.9–76.8 ms request-to-arrival delay. It did not recur in a randomized repeat
after restarting the same pinned image. The repeat's six recorded streams all
passed the existing cadence limits (rate ≥38 Hz, p95 gap ≤35 ms, maximum gap
≤60 ms). The transient-model evaluator now enforces those same limits for
odometry, steering, both rear encoders, IMU, and packet timing; the 100 ms
capture is excluded from model validation. This points to an intermittent
simulator/host response stall, not a repeatable high-steer fault; its source
remains unresolved.

Across the repeated high-angle blocks, the median `K_yaw` values are:

| Speed | 0.30 rad | 0.42 rad | 0.46 rad | 0.50 rad |
| ---: | ---: | ---: | ---: | ---: |
| 3 m/s | 1.584 | 1.096 | 1.007 | 0.924 |
| 4 m/s | 0.967 | 0.721 | 0.659 | 0.618 |
| 5 m/s | 0.679 | 0.509 | 0.471 | 0.448 |

The production steering gain is 2.95 1/m below its 0.41 rad taper and 1.17
1/m at/above 0.46 rad. Thus it overpredicts yaw response throughout much of
the measured 0.30–0.50 rad band; at 4–5 m/s, the error is roughly 3.0–4.3x at
0.30 rad and 1.9–2.6x at 0.50 rad. This is a concrete optimizer mismatch,
not evidence that a specific wheel's tire force has been identified.

The per-wheel *kinematic* slip proxies add a plausible mechanism. At 4 m/s,
front `|tan(alpha)|` rises from 0.213 at 0.30 rad to 0.435 at 0.50 rad while
the rear proxy stays near 0.0027–0.0031. At 5 m/s it rises 0.244→0.472 while
the rear stays 0.0034–0.0041. Over the same steering change, measured net
lateral acceleration rises only 0.482→0.546 g (4 m/s) and 0.534→0.624 g
(5 m/s). If these kinematic proxies correspond closely to the guide's tire
slip coordinate, the fronts are already beyond its stated 0.01 peak and 0.10
asymptote landmarks; the weak force increase is consistent with tire-force
saturation. That is an inference, not proof: no per-wheel normal loads or
contact-force vectors are in the bag, so load transfer, combined slip, and
steering geometry remain confounded. The observed nonlinearity is therefore
not best summarized as “a sine”; the supported model is a saturated,
speed-conditioned aggregate lateral-acceleration surface.

### Steady-state speed interpolation

Training used the fresh 3 m/s capture and the clean, randomized 5 m/s repeat;
the full fresh 4 m/s run was held out. The 12 ±0.42/±0.46 rad response blocks
were withheld from source steering interpolation. Linear blending of
`K_yaw` gave 0.07827 1/m gain RMSE and 0.14828 rad/s yaw-rate RMSE. Blending
lateral acceleration `a_y = u*r` and deriving `r_ss=a_y/u` scored 0.01584
1/m and 0.02872 rad/s, versus 1.37362 1/m for the configured law. The
predicted local steering derivative had the measured positive sign in both
turn directions, ranging 0.512–1.258 1/s. This accepts the
lateral-acceleration coordinate for steady interpolation only within the
measured 3–5 m/s and 0.30–0.50 rad support. Do not extrapolate this result to
the lower-angle 0.20–0.25 rad response cliff or speeds above 5 m/s.

Reproduce the steady test:

```bash
python3 tools/evaluate_open_plane_speed_steering_surface.py \
  live_runs/openplane_rootless_isolated3mps_20260927_01/run/run_0.db3 \
  live_runs/openplane_rootless_isolated4mps_20260927_01/run/run_0.db3 \
  live_runs/openplane_rootless_isolated5mps_repeat_20260927_02/run/run_0.db3
```

### Recursive yaw-only rollout

A second test trained a lateral-acceleration surface on the new independent 3
m/s run and clean 5 m/s repeat, then recursively predicted all 24 probes from
the fresh 4 m/s capture using recorded speed and steering as conditional
inputs. It evaluates both the source-fitted time constant and the existing
15 ms production prior; the latter was fixed before scoring, not fitted to
the 4 m/s holdout.

| Horizon | Production yaw law | Surface, fitted tau | Surface, fixed 15 ms | Fixed-tau gain |
| ---: | ---: | ---: | ---: | ---: |
| 25 ms | 0.000056 rad/s | 0.000070 | 0.000056 | 0.0% |
| 125 ms | 1.947706 rad/s | 0.143517 | 0.191674 | 90.2% |
| 250 ms | 2.254034 rad/s | 0.049716 | 0.048932 | 97.8% |
| 500 ms | 2.215832 rad/s | 0.018306 | 0.017831 | 99.2% |
| 750 ms | 2.208990 rad/s | 0.025339 | 0.024972 | 98.9% |

The source-fitted tau hit the 5 ms lower search bound, so its true transient
constant is unresolved at 40 Hz; this fitted-tau variant fails the predeclared
no->10%-regression rule at 25 ms. Keeping the existing 15 ms time constant
instead gives no measurable 25 ms regression and >90% improvement at all
later horizons, so the **yaw-only surface passes the unchanged recursive
gate**. This is valid evidence for that measured response component, not a
complete vehicle model: longitudinal speed and lateral velocity are not
predicted, steering below 0.30 rad and speeds outside 3–5 m/s are not
validated, and the time constant is not reidentified. The surface is not yet
integrated into the production MPC. The separate same-speed direct-curvature
tests also improved 125–750 ms yaw error at 3, 4, and 5 m/s but do not by
themselves validate full `u,v,r` dynamics.

Reproduce the recursive test:

```bash
python3 tools/evaluate_open_plane_transient_model.py \
  live_runs/openplane_rootless_isolated3mps_20260927_01/run/run_0.db3 \
  --speed-surface-test \
  live_runs/openplane_rootless_isolated5mps_repeat_20260927_02/run/run_0.db3 \
  live_runs/openplane_rootless_isolated4mps_20260927_01/run/run_0.db3
```

The result identifies the nonlinearity more concretely than “a sine”: at fixed
steering, `K_yaw` falls strongly with speed while measured lateral
acceleration rises moderately; within the tested high-angle interval,
increasing steering raises that acceleration only modestly. Tire-force
saturation, combined slip, load transfer, and steering geometry are plausible
contributors, but their individual roles cannot be separated from aggregate
body motion. Per-wheel force/load instrumentation or a validated legal-input
state-transition fit is still needed before claiming the underlying tire
curve.

### Fixed-speed gate and controller check (2026-09-27)

For a test advertised as fixed-speed, loss of speed is not part of the intended
input condition: steering response should be measured only while measured
vehicle speed remains near the target. Speed loss under an uncorrected or
loose-gated turn is still valuable, but it belongs in a separate coupled
longitudinal/lateral response dataset. Do not mix those observations into a
fixed-speed surface as though they were measurements at the target speed.

The full-surface 3 m/s capture
`live_runs/openplane_rootless_30_full_surface_20260927_01/run/run_0.db3` used
the previous permissive gates (median error 0.35 m/s, p95 error 0.65 m/s).
All 84 steering probes passed those gates, but when re-scored at 0.12/0.20 m/s
only 29/84 passed. The median across probes of their per-phase median and p95
absolute speed errors was 0.121 and 0.219 m/s, respectively. Therefore the
previous `valid` label did not establish constant-speed data.

Two real-simulator controller checks were made using the same 3 m/s, 84-probe
schedule and 0.12/0.20 m/s gates:

| Controller | Result |
| --- | --- |
| Proportional gain 0.50 | Aborted after 23 probes on an odometry timeout; 0/23 passed. Zero throttle occurred in 22/23 probes (7% of command samples), consistent with engaging the simulator's active-braking branch. |
| P=0.10, I=0.08 | Completed all 84 probes without collision. In a same-seed, same-order comparison with P=0.10/I=0, 28/84 versus 29/84 probes passed both gates. |

The paired comparison grouped the three repetitions by each of 28 signed
steering conditions and bootstrapped those conditions. I=0.08 reduced the
median per-phase p95 error by 0.0179 m/s (95% interval -0.0242 to -0.0141),
but increased median per-phase median error by 0.0118 m/s (95% interval
0.0058 to 0.0132). That trade-off did not improve the joint acceptance rate;
the PI option remains development-only and is not the default controller.
Increasing P gain to 0.50 is rejected. No speed-loop setting from these runs
is promoted to the vehicle controller or MPC.

The paired PI run retained 39.96 Hz odometry/steering/encoder/IMU/packet
streams (receipt-time p95 gaps 26.5–27.0 ms; maximum 49.7–50.7 ms), zero
collisions, and zero bridge timing faults. The simulator stayed running with
no OOM/restart. The separate P=0.50 abort was an odometry silence timeout, not
a simulator process crash; a subsequent read-only bridge probe recovered
approximately 39.9–40.0 Hz. This does not prove the transient transport stall
cause.

The speed controller's P/I gains and phase gates are now recorded with each
run, and the default fixed-speed acceptance gates are 0.12 m/s median and
0.20 m/s p95. Future fixed-speed captures that miss either gate must be
marked invalid for fixed-speed fitting; keep their raw bags for the separate
natural speed-droop analysis. Older bags must be rescored from their logged
per-phase error metrics rather than trusting their more permissive `valid`
flags. The controller change only affects this development experiment harness;
it does not alter simulator physics or the competition/runtime stack.

### Cross-speed hybrid yaw response: held-out speed validation (2026-09-27)

The measured steady response was refit in the lateral-acceleration coordinate
`a_y = |u r|`: speed-scaled steering demand `q = |u tan(delta)|` below 0.25 rad,
steering angle above 0.275 rad, and a smooth blend between those regions. A
separate low-/high-angle yaw-response time constant was fit from repetitions
1--2 at each training speed. No measured speed or steering from a holdout was
used to fit its surface or response lag.

Using source speeds 3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 6.5, and 8.25 m/s, the
first check held out two independent complete 5.0 m/s captures. With the
production fixed 15 ms time constant, this map failed the strict 0.05 rad/s
yaw-error gate at 25--125 ms (0.095/0.081 rad/s), despite reducing error by
94--97% against the configured model. Fitting the response lag only on the
training repetitions passed all five horizons:

| Horizon | Candidate yaw RMSE | Configured yaw RMSE | Candidate planar-position RMSE | Configured position RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 25 ms | 0.0185 rad/s | 1.5263 rad/s | 0.0000 m | 0.0012 m |
| 125 ms | 0.0342 rad/s | 1.8854 rad/s | 0.0008 m | 0.0548 m |
| 250 ms | 0.0409 rad/s | 1.8703 rad/s | 0.0037 m | 0.2476 m |
| 500 ms | 0.0433 rad/s | 1.8672 rad/s | 0.0184 m | 0.9835 m |
| 750 ms | 0.0457 rad/s | 1.8670 rad/s | 0.0441 m | 1.9959 m |

The 5.0 m/s holdout matched 53/58 resolved local yaw-Jacobian signs (91.4%),
above the existing 90% screen. A second whole-speed holdout at 4.5 m/s passed
with 0.0163, 0.0237, 0.0223, 0.0178, and 0.0344 rad/s at the same horizons,
and 52/56 (92.9%) resolved Jacobian signs. Repetition 3 at 6.5 m/s, including
the independent confirm capture, passed 28/28 resolved Jacobian signs with
0.0012--0.0028 rad/s yaw RMSE. These are conditional yaw/position rollouts:
measured longitudinal speed and steering are replayed as inputs, and lateral
velocity is held. They do **not** validate a closed-loop MPC or a full
longitudinal `u,v,r` plant.

A deliberate leave-6.5-m/s-out test trained on neighboring speeds and held out
6.5 m/s. Output errors were still small (0.0042--0.0129 rad/s), but only
24/28 (85.7%) resolved local Jacobian signs matched. The missing feature is
not generic noise: the 6.5 m/s data contain a narrow steering-response trough
that cannot be safely smoothed across speed. Include a measured 6.5 m/s source
surface when operating there; do not interpolate through it from adjacent
speeds. Using the fixed-angle low-steer coordinate instead of speed-scaled
demand was worse on the 5.0 m/s holdout (0.186--0.229 rad/s; 48/58 Jacobians),
so retain the validated demand coordinate.

The fitting code is
[`evaluate_open_plane_fullspeed_hybrid_rollout.py`](../../tools/evaluate_open_plane_fullspeed_hybrid_rollout.py).
An MPC implementation of this surface passed its finite-difference check
(maximum normalized Jacobian discrepancy `0.000294` across 66 speed/steering
conditions), but that only validated the code and derivatives. The following
same-track live transfer test rejected the model, so the implementation was
removed rather than left as an accidental development option.

### Practice-track transfer check (2026-09-27)

The unchanged practice yaw law was evaluated on the successful 12-lap
practice bag `live_runs/practice_current_baseline_20260926_codex1/run/run_0.db3`.
Using measured speed and steering as conditional inputs, its one-step yaw
prediction RMSE was `0.01817 rad/s`, and its five-step conditional rollout
RMSE was `0.02118 rad/s`. The open-plane surface did not transfer: its
training-fitted response lag gave `0.0535`, `0.1666`, `0.2689`, `0.4038`, and
`0.4242 rad/s` at 25, 125, 250, 500, and 750 ms; fixing the lag at the
production `15 ms` still gave `0.1200`, `0.1467`, `0.1462`, `0.1350`, and
`0.1096 rad/s`. Thus neither retuning the lag alone nor the steady-state
surface alone explains the track response.

The live A/B used the same practice map and raceline. The baseline completed
1 warmup + 10 racing + 1 extra lap with zero collisions; scored mean was
`6.0695 s`. The surface candidate completed no lap and had zero collisions,
but accumulated 6,617 rejected nonlinear rollouts, fell into hard fallback,
and stopped making progress. Its divergence started on the track well before
any collision. AMCL remained consistent with odometry, and LiDAR stayed at
`39.967 Hz` throughout the candidate recording. The evidence points to a
non-transferring model, not localization or transport loss.

The surface also learned a `110 ms` low-angle response constant at 3.5 m/s,
versus the practice configuration's independently validated `15 ms`. This
is one concrete mismatch, but it is not the whole cause: the surface still
failed the practice-bag replay with the lag fixed at 15 ms. Keep the
open-plane surface as an offline result only; it must not be used by the
practice or competition MPC unless a new model passes held-out practice-bag
prediction and a live same-track screen.

### Fixed-speed experiment controller: proportional-loop oscillation — 2026-09-27

The 3 m/s `isolated_transition_full_surface` schedule was repeated with the
same seed (`20260929`) and pinned Explore player while changing only the
speed-loop proportional gain. All captures used the same 0.12 m/s per-phase
median and 0.20 m/s p95 speed-error gates; older captures were rescored from
their per-phase errors rather than trusting older permissive `valid` flags.

| Speed-loop P / I | Strictly gated probes | +0.08 rad speed ripple (p-p) | Peak frequency | Collisions / timing faults |
| --- | ---: | ---: | ---: | ---: |
| 0.10 / 0.00 | 29/84 | 0.5565 m/s | 4.76 Hz | 0 / 0 |
| 0.10 / 0.08 | 28/84 | 0.5790 m/s | 4.75 Hz | 0 / 0 |
| 0.02 / 0.00 | 84/84 | 0.0179 m/s | 4.62 Hz | 0 / 0 |

The P=0.10 cases also had about 0.19 m/s standard deviation in the 0.08 rad
speed trace; P=0.02 reduced it to 0.0054 m/s. The oscillation was present with
and without integral action, so the proportional gain—not the I term—was the
cause supported by this A/B. With P=0.02, all six repetitions at every tested
steering angle passed. The difficult 0.25 and 0.275 rad conditions stayed
near 2.90 and 2.92 m/s respectively (median errors 0.100 and 0.083 m/s;
median per-phase p95 errors 0.101 and 0.131 m/s). This small feedback term
therefore damps the loop while recovering the localized speed droop that a
fixed 0.120 throttle leaves in that band.

The P=0.02 run completed all 86 schedule phases with exit status 0. Odom,
steering, rear encoders, IMU, and packet-timing streams were 39.963 Hz, with
26.4--26.7 ms p95 and 48.1--48.9 ms maximum receipt gaps; the actuator command
stream was 39.569 Hz. There were zero collisions and zero bridge timing
faults. Bag:
`live_runs/openplane_speedhold_p002_20260927_01/run/run_0.db3`.

### Cross-speed confirmation and selected harness gain — 2026-09-27

The P=0.02 result did not generalize across the measured speed range: at 4 m/s
its full-surface run failed only at ±0.20 rad in all three repetitions. Those
six phases had median absolute speed errors of 0.140–0.141 m/s and p95 errors
of about 0.142 m/s, just outside the unchanged 0.12/0.20 gates. Rear encoder
slip proxies reached about 0.13 on one driven wheel in those blocks, while the
other stayed near zero. This is repeatable speed loss under combined cornering
and drive load, not the 4.7 Hz speed-loop oscillation observed at P=0.10.

The gain was increased once, to P=0.04/I=0, and tested with the same randomized
full-surface schedule, seed, and gates:

| Target speed | Valid probes | ±0.20 rad median / p95 absolute speed error | +0.08 rad speed ripple (p-p) | Bag |
| ---: | ---: | ---: | ---: | --- |
| 3.0 m/s | 84/84 | within gates | 0.0158 m/s | `live_runs/openplane_speedhold_p004_3mps_20260927_01/run/run_0.db3` |
| 4.0 m/s | 84/84 | 0.107 / 0.117 m/s | 0.0143 m/s | `live_runs/openplane_speedhold_p004_4mps_20260927_01/run/run_0.db3` |
| 4.5 m/s | 84/84 | 0.053 / 0.053 m/s | 0.0016 m/s | `live_runs/openplane_speedhold_p004_45mps_20260927_01/run/run_0.db3` |

The 4 m/s ±0.20-rad result also passed a separate, narrower 18/18 bridge
schedule at P=0.04. The low-steer ripple stayed roughly 0.014–0.016 m/s at
3–4 m/s and 0.002 m/s at 4.5 m/s, compared with 0.56–0.58 m/s for P=0.10.
The small residual near 4–5 Hz is not a material speed excursion at P=0.04.
The three full-surface runs completed all 86 scheduled phases, with 39.96–
39.97 Hz odometry, steering, encoder, IMU and packet-timing streams, zero
collisions and zero bridge timing faults. The six-probe P=0.02/4 m/s failure
bag is retained at
`live_runs/openplane_speedhold_p002_4mps_20260927_01/run/run_0.db3`.

The development excitation default is now P=0.04/I=0.00. This setting is
validated only at the discrete 3.0, 4.0, and 4.5 m/s targets in the Explore
plane experiment; do not extrapolate it beyond that range without another
gated run. It is only the development measurement controller—it does not
change MPC, odometry/localization, the competition container, or simulator
physics. A probe that misses either speed gate remains invalid for fixed-speed
model fitting, while its raw bag can still be used for coupled speed/steering
analysis.

### Full-input nonlinear body model and encoder-state rejection — 2026-09-27

Three clean full-input Explore captures now cover randomized combinations of
all steering commands from zero through exact +/-0.5236 rad and throttle
feedback from zero to 0.50. The first capture has 504 valid phases (steering
through +/-0.50 rad); the second training capture and independent whole-run
holdout each have 552/552 valid phases. Speed reaches 8.15--8.16 m/s. Odom,
steering, throttle, both encoders, IMU, and bridge packet timing stayed at
39.974--39.977 Hz, with p95 gaps about 25.5--25.7 ms; collisions and bridge
timing faults remained zero. Bags are under 180 MB each:

* `live_runs/openplane_full_input_excitation_20260927_train1/run/run_0.db3`
* `live_runs/openplane_full_input_excitation_20260927_train2/run/run_0.db3`
* `live_runs/openplane_full_input_excitation_20260927_holdout_cache/run/run_0.db3`

The whole-run fit uses a cubic tensor-product B-spline over rear-axle body
state, measured steering/throttle, and frame-corrected front/rear slip-angle
proxies. This is explicitly nonlinear in speed, steering, sideslip, and their
selected interactions; the affine model is only a reference baseline. Fitting
on the first two captures with ridge 0.1 and evaluating on the untouched third
gave these one-step state RMSEs:

| State | Affine baseline | Nonlinear body surface |
| --- | ---: | ---: |
| longitudinal speed | 0.0800 m/s | 0.0409 m/s |
| rear-axle lateral speed | 0.0679 m/s | 0.0283 m/s |
| yaw rate | 0.1410 rad/s | 0.1184 rad/s |

The high-steering 4--6 m/s subset (2,672 transitions, |steer|=0.40--0.524 rad)
improved from 0.0854/0.0849/0.0895 to 0.0241/0.0340/0.0837 for
longitudinal speed/lateral speed/yaw rate. At 6--8 m/s and the same steering
band (3,553 transitions), it improved from 0.0641/0.0108/0.1070 to
0.0437/0.0080/0.1052. Thus the state surface generalizes through full lock and
high speed for body velocity, but it has **not** solved high-speed yaw
prediction. Its conditional 750 ms yaw-rate RMSE is still 0.786 rad/s versus
2.096 for affine. These rollouts replay measured actuator feedback, so they
are not yet a standalone vehicle simulation.

The holdout independently shows a strong nonlinear steering-speed response.
For example, median yaw gain falls from 2.948 to 0.456 1/m when steering moves
from 0.10--0.20 to 0.50--0.524 rad at 4--6 m/s, and from 1.418 to 0.243 1/m at
6--8 m/s. Lateral acceleration remains roughly 6--8 m/s^2 across much of those
steering bands. These are empirical response associations, not a fitted
per-wheel force law; throttle and past states are not independently matched
in these broad slices.

A first attempt to add both 100 ms encoder-derived rear-wheel surface speeds
and their tensor interactions was rejected. Despite 99.7% marginal feature
range support, at ridge 0.1 its held-out one-step errors ballooned to
0.249 m/s longitudinal, 5.892 m/s lateral, and 17.161 rad/s yaw. This exposes
an evaluator limitation: checking each feature's range does not establish
support for a particular joint speed/throttle/slip combination, and a large
unconstrained tensor surface can extrapolate badly between observed joint
states. Do not feed this candidate into MPC or call it an identified tire
model.

The encoder stream itself is not mistimed in these captures: header and bag
receipt intervals agree at 40 Hz, with stable sub-millisecond stamp offset.
Yet in straight 4--6 m/s samples at the same 0.20 throttle command and
feedback, training runs have median rear-wheel surface speed about 5.07 m/s
at body speed 4.92 m/s, while the holdout is about 5.77 m/s. A fourth,
independent partial run repeats the higher branch at 5.82 m/s. The difference
persists beyond 0.7 s in the 1.3--2.5 s phases; during the matched slice,
median body acceleration remains near zero in all runs. The velocity slip
proxies correspond to about 3% and 17% longitudinal slip ratio, respectively.
The latter is near the guide's published longitudinal-force extremum at 15%
slip, making the post-peak tire response a testable hypothesis, not a measured
force explanation. The two branches are not explained by the immediately
preceding throttle/steering command alone; longer history or another hidden
vehicle state remains unidentified.

#### High-steer 3-D departure and correction to the planar audit

The fourth randomized capture used seed `20260933` and stopped after 232
completed phases when the emergency speed guard reached 9.405 m/s. The closed
55 MB bag has zero collisions and zero timing faults; core streams were
39.977 Hz. It remains partial and is not a complete-run holdout, but its 232
completed phases are usable as independent validation. The body-only spline
trained on runs 1--2 retained lower one-step RMSE than affine on this subset
(0.0479/0.0422/0.1322 versus 0.0843/0.0706/0.1494 for longitudinal speed,
rear lateral speed, and yaw rate). At 750 ms its conditional rollout RMSE was
0.411/0.812/0.719 versus 2.066/1.182/2.006. Measured actuator feedback is
replayed, and the run ends before all schedule conditions are covered.

The abort exposed an unmodelled 3-D boundary. The tilt angle computed from the
full odometry quaternion stayed below 6.2 degrees in each completed capture.
In the partial run, phase 231 (`steer=+0.20 rad`, `throttle=0.50`) followed a
high-speed steering reversal. Tilt was 3.8 degrees at phase start; about 1.4 s
later, at 6.37 m/s, it reached 8.5 degrees, then rose through 14.2, 51.8, 88.1,
and 122.5 degrees over the next 0.3 s. The vehicle was already at about 137
degrees tilt when phase 232 (`steer=-0.50 rad`, `throttle=0.28`) began. It
continued through a large vertical excursion and the speed guard ended the
experiment after the rollover had begun. Therefore phase 232 is not the
initiating cause; the event begins in phase 231's steering/throttle/history
combination. Collision telemetry remained zero.

An initial *planar* comparison misleadingly suggested a 5.76 m/s
pose--twist disagreement and a roughly -41 m/s^2 lateral acceleration. That
calculation omitted roll/pitch and the full 3-D COM lever arm, so it is
invalid for this event. Recomputing the twist-to-rear-axle shift with the full
quaternion and 3-D angular velocity reduced the pose-derived velocity
residual to 0.097 m/s in the event (0.104 m/s over the partial capture). This
supports a real out-of-plane vehicle response, not a bridge timing or odometry
conversion fault. Do not train this state as ordinary planar cornering.

The simulator has sprung/unsprung suspension and slip-dependent nonlinear
tire forces, as described in the [official vehicle dynamics guide,
§1.3.2](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics).
The observed transition therefore calls for separate models: an empirically
validated planar nonlinear body model inside its measured domain, plus a
roll/sideslip stability boundary (and, if needed, a 3-D roll-state model).
Do not hide the boundary by flattening the tire curve or treating the event as
ordinary yaw saturation. No simulator physics, competition runtime, odometry,
or MPC configuration changed.

### Command-driven rollout and support audit — 2026-09-27

The full-input spline was trained on the two clean training bags above and
scored against the untouched 552-phase `holdout_cache` run. To remove future
actuator feedback from recursive predictions, each forecast uses the measured
actuator state only at its initial instant, then replays the recorded command
history with a 50 ms command-to-feedback alignment. This delay is an empirical
bridge-timeline hypothesis, not a physical steering-servo model.

The first support audit reported only 474/552 supported 750 ms rollouts. A
feature-level check found that 69 of 78 apparent exits were caused by a
Float32 boundary mismatch: recorded full-lock feedback is
`±0.52359998226165771` rad, while double-precision command conversion produced
`±0.52360000000000000` rad. The scorer now quantizes the converted steering
command to Float32, matching the ROS wire value. This removes only the
`1.8e-8` rad numerical overrun; it does not clip real vehicle states or alter
the fitted dynamics.

After that correction, recursive command-driven results are:

| Horizon | Supported rollouts | `u` RMSE | `v_rear` RMSE | yaw-rate RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 25 ms | 552/552 | 0.013 m/s | 0.011 m/s | 0.048 rad/s |
| 125 ms | 552/554 | 0.213 m/s | 0.138 m/s | 0.770 rad/s |
| 250 ms | 549/553 | 0.375 m/s | 0.268 m/s | 0.892 rad/s |
| 500 ms | 546/553 | 0.453 m/s | 0.531 m/s | 0.827 rad/s |
| 750 ms | 543/552 | 0.654 m/s | 0.906 m/s | 0.800 rad/s |

The nine 750 ms exits are genuine predicted-state extrapolations: three
first cross the training `u` bound and six cross the yaw-rate bound. Their
median predicted state at exit is `u=-1.214 m/s`, `|v|=3.399 m/s`, and
`|r|=7.124 rad/s`. They arise from diverse steering/throttle phases, not one
repeatable command cell, so these are not yet evidence of one missing
steady-state tire-curve knot. They are excluded from the RMSE denominator and
remain an explicit failure mode.

To test whether these exits were caused by the 25 ms explicit-Euler step, the
same fitted spline, holdout, command history, and initial states were replayed
with five 5 ms internal Euler substeps per 40 Hz sample. At 750 ms this gave
543/552 supported rollouts and RMSE `0.671/0.933/0.793` for `u/v/r`, versus
`0.654/0.906/0.800` with one step. The same nine phases left support. Smaller
integration steps therefore do **not** explain or cure the divergence; the
remaining issue is model state/dynamics or its learned response surface, not a
simple time-step instability.

The command-driven model remains a nonlinear candidate, not a standalone
simulator: even inside its support it has roughly `0.8 rad/s` 750 ms yaw error,
and it cannot predict through the nine spin-like states. The affine fit stays
only as a diagnostic baseline. No parameters were changed in simulator
physics, odometry, localization, MPC, or the competition runtime. Reproduce
the command-history holdout with `tools/evaluate_open_plane_body_dynamics.py`
using `--command-alignment-delay-ms 50 --integration-substeps 1
--integration-substeps 5` for the numerical-integration comparison.

### Whole-run state/history sufficiency diagnostic — 2026-09-27

Added `tools/analyze_vehicle_state_sufficiency.py` as a diagnostic over
closed bags. It forms central, approximately 50 ms `[u_dot, v_dot, r_dot]`
targets and compares six causal feature sets with a robustly scaled,
32-neighbor conditional predictor. It also estimates joint support using the
p95 nearest-neighbor distance measured across the two training runs. This is a
nonlinear matching/ablation diagnostic, **not** a differentiable plant, MPC
model, or physical tire-force identification. History features use only
samples at or before the prediction time; the capture reader now retains
500 ms of pre-phase context without adding those pre-phase samples to fitting.
Crucially, M0–M5 use the bridge's simulator-truth current `u,v,r` as state
features and the same truth stream for future targets. This is an
**oracle-conditioned** state-sufficiency probe, not a legal-input training
result.

Training was fixed to the complete `train1` and `train2` bags above. Whole-run
holdouts were the independent 3, 4, and 5 m/s response captures plus a clean
12-lap practice bag. The latter has 12 completed laps, no collision increase,
no bridge timing-fault event, 39.95 Hz core streams, 26.1 ms p95 receipt gaps,
and 43.9 ms maximum core-stream gap. The practice baseline bag from the next
day was **rejected** because its timing-fault detail records a simulator socket
disconnect, despite reaching 12 laps. The loader permits an unphased practice
bag only when its lap counter reaches at least 12 and all collision, timing,
stream-rate, and gap gates pass.

The reported metric is next-sample state-increment RMSE in `u`, `v`, and yaw
rate, on identical M5-complete rows for every level. Values below are in
`m/s`, `m/s`, and `rad/s`; they are conditional one-step errors, not accumulated
trajectory errors:

| Whole-run holdout | M0 `(u,δ)` | M1 `+(v,r)` | M2 `+throttle` | M3 `+rear slip` | M4 `+rates/commands` | M5 `+history` |
| --- | --- | --- | --- | --- | --- | --- |
| 3 m/s open plane | `.1281/.0655/.1565` | `.0836/.0037/.1128` | `.0716/.0037/.1143` | `.0802/.0038/.1191` | `.0810/.0037/.1034` | `.0819/.0036/.1045` |
| 4 m/s open plane | `.0986/.0909/.1024` | `.0299/.0017/.0595` | `.0176/.0017/.0630` | `.0166/.0017/.0644` | `.0192/.0015/.0380` | `.0190/.0013/.0423` |
| 5 m/s open plane | `.0674/.0871/.1922` | `.0202/.0020/.0547` | `.0104/.0020/.0555` | `.0108/.0020/.0553` | `.0125/.0015/.0345` | `.0140/.0016/.0384` |
| 12-lap practice | `.0867/.0178/.1183` | `.0986/.0028/.0673` | `.0579/.0021/.0683` | `.0611/.0018/.0664` | `.0578/.0018/.0653` | `.0583/.0017/.0633` |

The evidence is mixed but useful:

- Adding `v` and `r` (M1) sharply reduces lateral/yaw one-step error on the
  independent open-plane holds. M1 alone transfers unevenly to practice:
  practice longitudinal error rises versus M0, while yaw error falls.
- Adding throttle feedback (M2) consistently helps longitudinal prediction on
  all three isolated-speed holds and materially improves it on practice.
- Rear-wheel slip-velocity proxies (M3) do **not** consistently improve any
  output. They are not measured tire slip and this test does not justify adding
  encoder-derived slip to MPC.
- Actuator rates/commands and causal history (M4/M5) reduce yaw error at 4 and
  5 m/s and slightly on practice, but regress yaw error at 3 m/s. This is a
  regime-dependent hint of missing actuator/history state—not evidence for one
  universal latent state.
- The practice holdout never reaches `|steering|=0.30 rad`; its maximum is
  `0.2833 rad`. It therefore provides **no high-angle track validation**.
  Open-plane high-angle data remain necessary for that regime.
- M0 joint support on practice is 76.3%; M1–M3 report 100% under the
  cross-run-calibrated distance threshold, with M4/M5 at 95.4%/95.9%. These
  percentages are feature-space diagnostics, not calibrated prediction
  confidence; increasing dimension changes the distance scale.

These bags expose `/autodrive/roboracer_1/odom`, which
`tools/analyze_practice_state_accuracy.py` explicitly documents as development
ground truth. The analysis uses it for both current body-state features and
future transition targets; it does not read `/ips` or transform topics. It
does not use the team-produced legal `/odom` estimate, and therefore does not
establish that the state variables in M0–M5 are available with equal accuracy
to MPC. Do not interpret this test as a legal-input or production-transfer
pass.

This table is an oracle-state diagnostic only. A later attempt to substitute
observer states initially left M0–M2 connected to truth; those intermediate
scores were discarded. The corrected legal-state rerun is described below.
Even that KNN report's “next interval increment” error is only derivative error
times the interval; it does not include the observer's current state error and
must not be read as an absolute state-prediction score.

### Observer-state failure under wheelspin — 2026-09-28

The offline replay now joins each sample to the exact `sensor_odometry_node`
replay by original source stamp. The replay uses only the two encoder streams
and IMU, disables TF publication, is network-isolated, and verifies the image's
observer source/config hashes against this worktree. It covers 100% of the
selected samples. On the practice bag, the normal replay also matches the
recorded team `/odom` twist exactly at every source stamp. No simulator
physics, MPC, or runtime odometry configuration was changed for this study.

The clean 12-lap practice run remains well behaved: replayed forward-speed
RMSE against truth is `0.139 m/s`, lateral-speed RMSE is `0.009 m/s`, and yaw
rate is exact. The same observer fails badly on the deliberately broad
full-steering/full-throttle open-plane excitation:

| Whole-run data | Observer `u` median | Truth `u` median | `u` bias | `u` RMSE | `v` RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Excitation train 1 | 11.36 m/s | 3.55 m/s | +12.35 m/s | 21.49 m/s | 0.95 m/s |
| Excitation train 2 | 11.41 m/s | 3.67 m/s | +13.41 m/s | 20.58 m/s | 0.92 m/s |
| Independent excitation holdout | 9.94 m/s | 3.60 m/s | +8.56 m/s | 13.12 m/s | 0.88 m/s |

Yaw-rate RMSE remains exactly zero because the observer publishes the aligned
gyro rate. The forward-state error is therefore not a source-time join or yaw
frame error. It is specific to the aggressive open-plane envelope and is not
evidence that the 12-lap practice run has the same speed failure.

The encoder data identify a repeatable nonlinear wheel-spin regime. Encoder
surface speed (using the configured `0.059 m` radius) was compared with the
ground-truth rear-wheel longitudinal contact speed averaged over the matching
100–120 ms window. For `|steering| >= 0.30 rad`, throttle `>= 0.30`, and truth
speed below `6 m/s`, the excess wheel speed was:

| Whole run | Samples | Median excess | p90 excess | Fraction above 3 m/s |
| --- | ---: | ---: | ---: | ---: |
| Train 1 | 2,364 | 8.05 m/s | 10.60 m/s | 99.2% |
| Train 2 | 3,280 | 7.75 m/s | 10.63 m/s | 98.3% |
| Holdout | 3,541 | 7.86 m/s | 10.94 m/s | 97.5% |

At throttle `<= 0.15` and steering `0.45–0.54 rad`, median excess ranges
from `-0.06` to `+0.16 m/s` across these runs. At throttle `0.30–0.45`, even the
`|steering| < 0.15 rad` cohort has median excess `6.12–6.72 m/s`. Thus a
single wheel-radius/gain correction cannot explain both regimes: the measured
encoder-to-body relationship changes sharply with throttle and persists on an
independent run. This is wheelspin/surface-slip evidence, not a force or normal-
load measurement.

One synchronized holdout sequence makes the state failure concrete: truth
longitudinal body velocity crosses from about `+0.10` to `-1.26 m/s` while the
rear encoder surface speed remains near `11 m/s`. The observer publishes
positive speed and then, after rejecting a burst, continues increasing it at
the configured positive-acceleration limit (`4.4 m/s²`) while wheel updates
are unavailable. The current observer takes absolute wheel speed, constrains
the turn lateral-velocity model to `0.35 m/s`, and has no learned wheelspin
measurement model. It therefore cannot represent the reverse body-longitudinal
component of a spin, and its positive-only dropout propagation can preserve a
large stale estimate. This is the empirically supported failure mechanism;
changing a constant wheel radius would hide the low-throttle agreement and
still fail the high-throttle holdout.

The corrected KNN ablation remains useful only for derivative sufficiency.
Adding `u,v,r` and throttle reduces conditional transition error, but the
increment metric omits the large initial observer-state error above. A
differentiable spline trained on legal observer state and truth labels likewise
has `13.1 m/s` one-step absolute forward-state RMSE on the independent
open-plane holdout; the nonlinear fit's `0.008 m/s` one-step improvement over
the affine forward-speed baseline is negligible, while its recursive forward
error is worse. It is rejected for MPC promotion.

An offline A/B of the existing `integrate_lateral_acceleration_in_turn=true`
parameter shows why the old switch was disabled. On the full-input holdout it
reduces `u` RMSE from `13.12` to `5.61 m/s`, but increases `v` RMSE from `0.88`
to `6.71 m/s`; on practice, it worsens `(u,v)` RMSE from `(0.139, 0.009)` to
`(0.161, 0.899) m/s`. This rejects global promotion. It suggests testing a
separated, bounded longitudinal turn update while retaining the validated
lateral estimate, but that needs its own whole-run replay and closed-loop
confirmation before any runtime edit.

The next identification target is therefore not another global steering gain
or MPC weight sweep. Fit the wheel-spin measurement surface over throttle,
wheel rate, steering, yaw rate, and causal history using train 1/2; require
improvement on the held-out open-plane run and no practice regression. Keep the
per-wheel force interpretation limited: the captures do not contain suspension
travel, normal load, or tire-force telemetry. The current code changes are
offline analysis/replay tools only.

### Causal observer screens and corrected plant-transfer evaluation — 2026-09-28

The exact practice replay was regenerated from
`practice_current_baseline_20260926_codex1` because the earlier `/tmp` practice
sidecar belonged to a different source run. The replay used the image-built
production observer, only the two encoder topics and IMU, isolated ROS domain,
and `publish_tf=false`; its 2,990 source-stamped twists match the recorded
`/odom` exactly. The source bag has 12 laps, zero collisions, one timing-fault
flag, and only `0.282 rad` maximum steering. That single flag is retained as a
quality caveat rather than silently calling this run fault-free.

Two low-complexity speed-estimator ideas were screened offline and rejected:

* A KNN truth-state correction gated by causal rear encoder slip-velocity
  proxy and throttle was trained on open-plane runs 1–2. Across the independent
  full-input holdout, a `0.5 m/s` slip threshold activated on 21.4% of samples
  and changed whole-set `u` RMSE only `13.12 -> 12.95 m/s`; more conservative
  gates changed it still less. Leave-one-training-run-out gating did not
  materially improve the approximately `21 m/s` observer error. It is not a
  robust, practice-safe correction law.
* Integrating the signed longitudinal IMU acceleration from a stationary
  startup estimate, with the existing COM-reference and braking calibration,
  diverged over the long open-plane runs: `u` RMSE was `494`, `504`, and
  `616 m/s` for train 1, train 2, and the independent holdout. Mean recorded
  `ax` was `+0.75`, `+0.71`, and `+0.68 m/s²`, despite near-zero startup bias.
  Practice IMU-only RMSE was `3.84 m/s` versus `0.12 m/s` for the replayed
  observer. A fixed wheel/IMU innovation blend did not repair the open-plane
  drift. Do not replace the wheel observer with raw acceleration integration;
  the acceleration signal requires a validated operating-regime measurement
  model and periodic independent velocity anchoring.

The independent offline-plant evaluation used full-input runs 1–2 for fitting
and the complete `holdout_cache` run for testing. Ground-truth body state was
used as the training state coordinate and once at the start of each held-out
phase rollout; all subsequent rollout states were model-predicted. Actuator
feedback was replayed, so this is a conditional body plant—not yet a complete
simulator or closed-loop MPC test. The nonlinear spline's held-out one-step
RMSE was `0.0404/0.0375/0.1392` for `u/v/r`, compared with affine
`0.0798/0.0679/0.1408`. For 50 ms command-aligned free rollouts, nonlinear
750 ms `u/v/r` RMSE was `0.660/0.984/0.849` (543/552 supported), versus
`1.987/1.119/1.942` affine. Five integration substeps did not materially
improve it. Thus nonlinear structure helps, but yaw prediction is still far
from sufficient at high demand.

Before scoring that model on a complete unphased practice bag, the evaluator
was corrected. Previously it found one horizon target relative to the whole
bag's first timestamp, so later continuous segments were omitted. It now
evaluates a 0.75 s recursive window every 0.5 s in every unphased continuous
segment; phase-labeled excitation still uses the phase start. Applying this
to the 12-lap practice bag yielded 144 rollout starts across five segments.
The open-plane-trained nonlinear model had `100%` marginal feature-range
support but 750 ms `u/v/r` RMSE `0.540/1.196/0.894`; at 500 ms they were
`0.368/0.541/0.677`. This contradicts promotion: marginal ranges do not prove
joint support or domain transfer. The practice test does not cover the
high-steering region, and one timing fault remains in its source bag.

Next model work should train on a balanced mixture of complete practice and
open-plane runs, hold out an untouched practice run and an untouched
full-input open-plane run, and report joint-support plus local steering
sensitivity—not just marginal ranges and state RMSE. Keep wheel/contact labels
from the public-source player excluded: its high-angle public response failed
the pinned-player equivalence gate. No production odometry/MPC parameter,
simulator physics, or controller topic policy changed in these screens.

### Actuator surrogate and free-running body-model check — 2026-09-28

`tools/evaluate_open_plane_actuator_dynamics.py` compares zero-order, delayed,
first-order, rate-limited, and combined command-to-feedback models. Fitting
uses the two complete full-input training runs with equal run weight. The
selected steering hypothesis is a 25 ms command delay, 25 ms first-order lag,
and the documented 3.2 rad/s rate limit (training one-step RMSE `0.0343 rad`);
the selected throttle hypothesis is a 25 ms delay and 25 ms first-order lag
(training RMSE `0.0474` normalized command). These compact equations predict
steering and throttle feedback only; they do not predict wheel speed, tire
slip, suspension, or contact forces.

The actuator candidate was passed into the body evaluator on the untouched
track holdout. Each rollout starts from measured body and actuator state,
then predicts both actuator channels from recorded command history and the
body state from the nonlinear-u/r plus affine-v model. Comparing identical
matched rollout starts against the conditional version that replays measured
future actuator feedback:

| Horizon | Body `u` RMSE, measured → predicted actuator | Body `v` RMSE | Yaw-rate RMSE |
| ---: | ---: | ---: | ---: |
| 125 ms | 0.1059 → 0.1349 m/s | 0.0121 → 0.0122 m/s | 0.2154 → 0.2421 rad/s |
| 250 ms | 0.1535 → 0.2364 m/s | 0.0183 → 0.0180 m/s | 0.3476 → 0.5890 rad/s |
| 500 ms | 0.2196 → 0.2769 m/s | 0.0374 → 0.0381 m/s | 0.4670 → 0.5139 rad/s |
| 750 ms | 0.2315 → 0.3157 m/s | 0.0564 → 0.0580 m/s | 0.4423 → 0.5960 rad/s |

Whole-lap paired bootstrap at 750 ms confirms the degradation is not driven
just by pooling many adjacent samples: predicted-actuator minus measured-
actuator error was `+0.0834 m/s` in forward speed (95% interval
`[+0.0621,+0.1065]`, 0/16 complete-lap clusters favored prediction),
`+0.0017 m/s` in lateral speed (`[+0.0009,+0.0027]`, 3/16 favored), and
`+0.1522 rad/s` in yaw (`[+0.0925,+0.2181]`, 2/16 favored). The fit's
training/one-step plausibility therefore does not establish sufficiently
accurate actuator state over a recursive horizon. It is **not accepted** as
an offline plant or MPC input.

For context, the measured-feedback hybrid body model's 750 ms error on this
track run is `0.232/0.056/0.438` for `u/v/r`, while the command-predicted
version is `0.316/0.058/0.596`; both are materially better than the affine
reference, but neither resolves actuator-free yaw rollout. On the independent
open-plane transfer, the measured-feedback hybrid is `0.724/0.687/0.932` and
the predicted-actuator hybrid is `0.801/0.647/0.832`. These cross-domain
results reject a single universal fit and motivate a focused audit of
steering-response residuals around reversals, high steering rates, and
high-steer/yaw regimes. Any candidate selection must use training runs only;
retain whole track runs and the independent open-plane run for final tests.

This actuator/body experiment used ground-truth current state for initial
conditions and labels, so it is an offline diagnostic—not a compliant runtime
controller, odometry estimator, or end-to-end sim. It does not resolve
per-wheel force attribution. No runtime or simulator behavior changed.

### Throttle-versus-steering causal attribution — 2026-09-28

The body scorer now allows a counterfactual one-channel actuator rollout:
predict steering while replaying measured throttle, or predict throttle while
replaying measured steering. This separates channel contributions without
changing the fitted body model or selecting a parameter from a different
rollout setup. On the same track holdout and 16 complete-lap clusters,
steering-only prediction is neutral at 750 ms (`u 0.2321 -> 0.2322`,
`v 0.0558 -> 0.0553`, `r 0.4378 -> 0.4391`); the yaw interval spans zero.
Throttle-only prediction reproduces essentially all the combined-model
regression: `u 0.2315 -> 0.3158 m/s`, `v 0.0564 -> 0.0584 m/s`, and yaw
`0.4423 -> 0.5917 rad/s`. The paired complete-lap increases are `+0.0834
m/s` for forward-speed RMSE (`95% CI [+0.0619,+0.1065]`) and `+0.1476 rad/s`
for yaw (`[+0.0881,+0.2135]`). The measured-feedback conditional baseline is
still better; this actuator candidate is not accepted.

The deterioration clusters with throttle cuts and high speed rather than
large steering. On 367 matched 750 ms windows:

| Subset | Count | Forward RMSE, measured → predicted throttle | Yaw RMSE |
| --- | ---: | ---: | ---: |
| Any command reaches zero | 206 | 0.233 → 0.358 m/s | 0.481 → 0.712 rad/s |
| Command drops by ≥0.10 | 103 | 0.258 → 0.422 m/s | 0.469 → 0.881 rad/s |
| Initial speed ≥6 m/s | 95 | 0.199 → 0.311 m/s | 0.246 → 0.786 rad/s |
| Peak steering <0.15 rad | 150 | 0.217 → 0.330 m/s | 0.186 → 0.625 rad/s |
| Peak steering ≥0.30 rad | 125 | 0.266 → 0.328 m/s | 0.665 → 0.676 rad/s |

The command-to-feedback model's per-step throttle error is `0.0223` normalized
over all rollout intervals and `0.0387` in command-cut windows. This points to
missing throttle/braking history or to excessive throttle sensitivity in the
body transition surface. It does **not** tell which is responsible: the model
is conditioned on ground-truth start states, and this is still a conditional
body prediction, not a complete simulator. A simple command lag alone has not
explained the failure.

The actuator fit was repeated with an additional complete track run included
in the equal-run training set. Its selected throttle model remained a 25 ms
delay plus 25 ms first-order lag. Coupling this model to the body predictor
produced essentially unchanged held-out track errors. On the independent
open-plane holdout, 750 ms `u` RMSE changed `0.724 -> 0.777 m/s`, yaw
`0.932 -> 0.940 rad/s`, and `v` `0.687 -> 0.686 m/s`. The added track data
therefore did not fix the free-running mismatch. Because the practice holdout
has already been inspected, do not use it to tune a new lag constant and then
call the same run independent validation. The next model should represent
throttle cuts/braking history or validate the body model's throttle-to-yaw
response using training-run splits, followed by a fresh complete-run test.

### Raw receipt-time correction and actuator/body re-evaluation — 2026-09-28

The preceding actuator results used command values rejoined to odometry-grid
samples. Steering/throttle command topics are Float32 without source headers;
the rejoin could lose their independent receipt-time alignment. The actuator
fit is therefore rerun from each command and feedback topic's raw bag receipt
timestamps with strictly causal zero-order-hold lookup. This supersedes the
preceding selected 3.2 rad/s steering-rate cap and 25/25 ms throttle-lag
interpretation as the preferred timing model.

Using two complete randomized open-plane runs plus the `amcl_startup_lock010b`
practice run for equal-run training, the one-step raw-clock ranking selected:

| Channel | Selected receipt-time surrogate | Training RMSE |
| --- | --- | ---: |
| Steering | 25 ms grid delay, 50 ms first-order lag, no rate cap | 0.023801 rad |
| Throttle | 25 ms grid delay, zero-order response, no rate cap | 0.014527 normalized |

The grid delay is not an exact physical delay estimate. Commands have no
source timestamps; their receipt streams run near 39.5 Hz and include one
106 ms maximum gap. On the open-plane phase-edge holdout, large command spans
also defeat a single smooth response: steering first-order RMSE is `0.1943
rad` on large changes versus `0.0401 rad` on small changes; throttle zero-order
RMSE is `0.0384` on large changes versus `0.0019` on small changes. This
supports distinct transition regimes, not a universal rate/lag equation.

The raw-clock actuator candidate was then coupled to the history-conditioned
body spline. On the existing 16-lap practice holdout, 750 ms hybrid
`u/v/r` RMSE changed from `0.2103/0.0502/0.3398` with measured future
actuators to `0.2049/0.0476/0.3337` with predicted actuator state. The paired
yaw change `-0.0066 rad/s` has interval `[-0.0150,+0.0001]` across complete
lap clusters and does not establish an improvement. On the whole-run
open-plane transfer, the hybrid's 750 ms error changed from
`0.8126/0.7871/0.9800` to `0.8099/0.7899/0.9490`; only 482/551 candidate
rollouts remained supported. This corrects the earlier impression that
actuator prediction alone caused a large universal degradation, but it does
not validate a free-running offline simulator: yaw error remains high and
support is incomplete. The same bags have already informed model choices, so
they are not a fresh final test.

The body evaluator now rejects windows lacking fresh causal command history
instead of throwing or using a future sample. A new randomized full-input
capture on the pinned Explore player completed 552/552 phases with zero
collisions, timing faults, invalid phases, or abort. A `111.8 ms` core-stream
gap occurred before the first phase, during unassigned bridge/player warm-up;
all active phase windows passed: core streams averaged `39.966 Hz` with
`25.76 ms` p95 and `49.21 ms` maximum gap, while command streams averaged
`40.000 Hz` with `27.37 ms` maximum gap. The validator checks measured phase
intervals used for model fitting and continues to report whole-bag timing, so
the startup gap remains visible and active-data limits are unchanged.
Bag: `live_runs/openplane_full_input_validation_20260928/run/run_0.db3`.
The frozen candidate's holdout score is complete. The truth-state nonlinear
spline cuts 750 ms recursive `u/v/r` RMSE from `1.819/1.172/1.837` to
`0.570/0.971/0.847`, but only 508/552 starts are supported. The hybrid
nonlinear-`u/r` plus affine-`v` model scores `0.933/0.890/0.974` on 487/552
starts. Predicting actuators from commands barely changes the hybrid at
750 ms (`0.934/0.891/0.975` to `0.937/0.890/0.995` on matched starts). Full
nonlinear one-step yaw error remains worse than affine (`0.158` vs
`0.143 rad/s`); the recursive gain does not make this a validated free-running
plant. No competition runtime, simulator physics, or vehicle behavior changed.

### Causal-history sufficiency and practice-domain transfer — 2026-09-28

The new independent open-plane holdout was also evaluated with the exact
source/config sensor-odometry node replayed from encoder and IMU streams.
Training stayed fixed to the two earlier whole runs. The capture completed
552/552 valid phases with no collision or timing-fault increase. A separate
oracle-state run used bridge body state only for offline diagnosis; no
controller received it.

| State features | Next-interval `u/v/r` RMSE | Cross-run support |
| --- | --- | ---: |
| Oracle M5 (body state + causal actuator/wheel history) | `0.0169 / 0.0171 / 0.0406` | 98.1% |
| Replayed sensor-odometry M5 | `0.0401 / 0.0415 / 0.0878` | 98.1% |

Values are conditional one-step increments in `m/s`, `m/s`, and `rad/s`, not
absolute speed error or recursive rollout. Replayed observer state makes the
increment errors about 2.4x larger than oracle state. Causal history is still
informative: replay-state M0 `u/v/r` error is `0.1273/0.0915/0.1572`; M5 is
`0.0401/0.0415/0.0878`.

A 32-neighbor supervised correction maps causal sensor/actuator features to
current bridge body state. Trained on open-plane runs 1–2, it reduces the new
open-plane holdout's replay-observer `u/v/r` RMSE from `19.70/0.93/0.00` to
`0.92/0.26/0.11`; forward-speed absolute-error p95 remains `1.94 m/s`. This
within-open-plane result does not transfer to the independent
`amcl_startup_lock010b_20260922` practice capture: exact-source replay error
against bridge truth is `0.126/0.011/0.000`, while open-plane M5 correction
gives `1.539/0.013/0.271`, despite 98.4% nominal feature support. The replayed
observer also differs from that older bag's recorded team `/odom` (`u` RMSE
`0.082 m/s`, max `1.022 m/s`; p95 aligned-pose residual `0.785 m`), so this is
a counterfactual replay with the current binary, not validation of the
historical controller state. Reject the KNN correction: feature-space support
did not catch the domain-transfer failure.

Current evidence supports causal state/history as useful for an empirical
nonlinear transition model, but not a universal body model. It also confirms
the open-plane wheel-spin observer failure and shows that a generic learned
speed correction can break a low-slip practice regime. Continue with an
explicit causal wheel-spin/traction state fitted from encoder, throttle,
steering, and IMU history. Select model structure on training-run splits, then
require a fresh full-input holdout and practice transfer. Do not promote any
KNN correction or MPC change. The bags do not expose tire forces, suspension
travel, or per-wheel normal load; claims about those quantities still require
pinned-player-equivalent instrumentation.
