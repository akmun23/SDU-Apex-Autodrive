# Engineering state: open-plane dynamics identification

Updated: 2026-09-27

## 2026-09-26 frame audit — applied to development analyses

Raw bridge `/odom` pose is at the rear axle, but its copied linear twist is at
the simulator COM. Pose differentiation confirms the COM reference 4.38x
better than the rear-axle hypothesis. Development slip/force analyses now
shift raw `v_y` by `-yaw_rate * 0.15532 m` before wheel kinematics and use the
bridge-consistent Ackermann side ordering. This corrects the analysis only;
the competition observer does not consume raw `/odom`, and no runtime model or
topic policy changed.

The corrected 4 m/s data show front `|Sy|` crossing from about 0.027 to 0.113
at the 0.21 rad high-/low-yaw branches, consistent with—but not proof of—the
published tire curve's declining region. A prior-state slip yaw model fails
the predeclared 5-horizon rollout gate. The guide-informed four-wheel force
model fits held-out aggregate lateral force (1.181 N at 4 m/s transition) but
cannot produce a physically usable, speed-consistent yaw/lateral rollout.
Reject it for MPC promotion. The detailed results and limitations are in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).

## Fixed-speed experiment loop — P=0.04 validated at 3–4.5 m/s — 2026-09-27

Matched Explore runs confirmed that P=0.10 causes a 4.6–4.8 Hz speed limit
cycle: only 29/84 and 28/84 probes passed the strict gates with P=0.10/I=0
and P=0.10/I=0.08 respectively, and the low-steer ripple was 0.56–0.58 m/s
peak-to-peak. P=0.02 passed 84/84 at 3 m/s but failed six probes at 4 m/s,
all three repetitions of ±0.20 rad (median error 0.140–0.141 m/s). P=0.04/I=0
then passed full 84-probe schedules at 3.0, 4.0, and 4.5 m/s; its low-steer
ripple was 0.016, 0.014, and 0.002 m/s respectively. All runs retained
39.96–39.97 Hz public streams, zero collisions, and zero bridge timing faults.
The development excitation default is now P=0.04/I=0.00. This validates the
measurement loop only through 4.5 m/s; it changes no vehicle/competition
controller or simulator physics. Detailed per-angle evidence and bag paths
are in `OPEN_PLANE_PER_WHEEL_DYNAMICS.md`.

Frame-corrected odometry replay with oracle speed/yaw gives 0.0061 m/s held-out
rear-axle lateral-velocity RMSE for the current kinematic law. IMU integration
is worse at 0.0680 m/s RMSE. Acceleration reference fitting independently
supports the existing 0.15532 m COM correction: the bag's IMU acceleration is
COM-referenced despite its x=0.08 m TF placement. No runtime odometry setting
was changed.

## Current work package — IMPLEMENTED

The explore-simulator capture path has `isolated_boundary`,
`isolated_force_3mps`, `isolated_force_4mps`, and `isolated_force_5mps`
profiles. They randomize signed steering probes at 0.30,
0.42, 0.46, and 0.50 rad, with three repetitions per condition. Before each
probe, the runner restores a straight, matched speed/lateral-velocity/yaw-rate
state using rolling 0.5 s medians; it aborts if that state cannot be
recovered. The 3.0 and 5.0 m/s profiles test how the high-angle response
changes with speed. These are development-only direct-actuator
experiments and do not change simulator physics or enter the competition
runtime.

Run the official explore image in batchmode and then start the runner in a
second terminal:

```bash
SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode ./tools/start_simulator.sh
SDU_APEX_EXPERIMENT_PROFILE=isolated_boundary \
SDU_APEX_EXPERIMENT_SEED=20260928 \
SDU_APEX_EXPERIMENT_TIMEOUT_S=180 \
./tools/run_open_plane_experiment.sh
```

The runner records only the small dynamics topic set. The pinned simulator
image is `sha256:1a26bcc8b91fe845f2caafb95533bf3606092502430efca493709e167a5adb40`;
the API/ROS image is `sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7`.

## Accepted capture

Run: `openplane_isolated_boundary_window_20260926_172626`

Bag: `live_runs/openplane_isolated_boundary_window_20260926_172626/run/run_0.db3`

Analyzer: `tools/analyze_open_plane_dynamics.py`

Acceptance: 24/24 probes valid; 24/24 starting states matched; all recorded
streams 39.96 Hz with p95 gaps below 27 ms; zero collisions and zero bridge
timing faults. The bag is about 13 MB.

Measured yaw gain `r/(vx*tan(delta))` at 2.2 m/s:

| Steering magnitude | Result |
| --- | --- |
| 0.30 rad | 2.949–2.950 1/m across signs; very low repeat spread |
| 0.42–0.46 rad | Two stable response branches, roughly 1.5–1.7 or 2.9–2.95 1/m; the early/late half comparison rules out a short transient |
| 0.50 rad | 1.38–1.59 1/m across signs; repeatable authority loss |

This shows the current steering-angle-only yaw taper is not a sufficient
identified model through the 0.42–0.46 rad transition. Do not change MPC or
raceline parameters from this capture alone. With the corrected COM-to-rear-
axle velocity shift, front lateral-slip proxies cross 0.10 on the low-yaw
branch, while rear lateral-slip proxies stay about 0.0014–0.0040. Rear
longitudinal slip is a separate signal. These are kinematic proxies, not
per-tire force measurements. Guide equations and constants:
[AutoDRIVE Technical Guide, §§1.3.2–1.3.4](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics).

## Diagnostic aborted captures

Runs `openplane_isolated_boundary_20260926_171623` and
`openplane_isolated_boundary_smooth_20260926_172247` aborted before the first
probe, with no collision. The initial pointwise speed gate rejected brief
40 Hz speed excursions although straight-motion and steering states were
otherwise settled. Its hard zero-throttle branch also caused avoidable
braking near the setpoint; removing that branch alone was insufficient. The
accepted capture uses continuous speed-hold throttle and rolling-window state
qualification; both changes are confined to the development excitation
process.

## Follow-up model investigation — IMPLEMENTED (offline decision)

Run `openplane_isolated_force_3mps_holdout_20260926` completed all 24/24
matched probes at 39.972 Hz with zero collisions. An offline constant lateral
acceleration envelope fitted on repetitions 1–2 reduced held-out yaw-rate
RMSE from 0.9881 to 0.3413 rad/s (65.5%); per-speed reductions were 38.6% at
2.2 m/s and 96.2% at 3.0 m/s. The fitted limit was 4.460 m/s^2 (0.455 g).
However, the hard cap had zero local steering sensitivity at 15/16 holdout
points, so it is not approved for MPC use despite passing the output-error
screen.

The guide-informed static-load, lateral-only per-wheel model was rejected:
it required a 4.271x force scale, had 15.342 N held-out net lateral-force
RMSE and 233.595 rad/s^2 held-out yaw-acceleration RMSE, and diverged in
recursive rollout. This only rejects the assumptions/implementation; missing
dynamic normal loads, AWD combined slip and per-wheel force telemetry prevent
calling the guide's model wrong.

Condition-matched tests found a useful low/mid-speed nonlinear state feature:
mean absolute rear longitudinal slip improves held-out yaw-gain prediction by
74.2% at 2.2 m/s and 61.7% at 3.0 m/s, with a positive fitted coefficient
at both speeds. The feature uses the two rear encoder sensors, which the
competition guide lists as permitted input topics. It predicts the otherwise
unexplained repeat-to-repeat high-angle branch after controlling for speed,
steering magnitude, and turn direction. This supports an empirical local
model `K_yaw = K0(speed, steer) + beta(speed)*(mean|Sx_rear|-reference)`.
It remains an association and a steady-block model; future-horizon slip
dynamics are not yet established. At 5.0 m/s the same feature failed its
predeclared screen (1.1% error reduction, coefficient sign reversed), so it
must not be treated as a global correction. The reusable analysis is
`tools/evaluate_open_plane_rear_slip_effect.py`.

Run `openplane_isolated_force_5mps_20260926` completed 24/24 matched probes
at 39.977 Hz with zero collisions and zero timing faults. Its high-speed yaw
gain is nearly deterministic by steering (0.679, 0.509, 0.471, 0.448 1/m at
0.30, 0.42, 0.46, 0.50 rad). A shape-preserving cubic yaw-gain map passed
leave-one-repetition-out intermediate-angle tests: RMSE 0.00269 1/m, max
error 0.00302 1/m, yaw-rate RMSE 0.00633 rad/s, and positive yaw-rate
sensitivity 0.765–1.024 1/s. This is a validated steady-state steering
surface at 5.0 m/s, not yet speed interpolation or transient rollout. The
evaluator is `tools/evaluate_open_plane_yaw_spline.py`.

## Speed-dependent high-angle model — IMPLEMENTED (offline identification)

The new matched 4.0 m/s capture
`live_runs/openplane_isolated_force_4mps_20260926/run/run_0.db3` completed
24/24 probes with 24/24 matched starts, zero collisions, no bridge timing
faults, and 39.976 Hz source streams. The three-repetition response is highly
repeatable: yaw gain is 0.967, 0.720, 0.659, and 0.618 1/m at steering
magnitudes 0.30, 0.42, 0.46, and 0.50 rad. The corresponding lateral
acceleration rises smoothly from 0.483 g to 0.545 g. Rear-slip correction
improves held-out yaw-gain RMSE by only 5.8% (0.0008 to 0.0007 1/m), below
the 10% gate; do not add rear |Sx| as a 4 m/s correction.

A compact high-speed response surface now has an independent real-simulator
holdout. At each measured speed, fit `K_yaw` versus steering with a
shape-preserving cubic; interpolate signed lateral acceleration between speed
knots, then recover yaw rate:

```text
a_y(v, delta) = speed_interpolate[v_i^2 * tan(delta) * K_yaw(v_i, delta)]
r_ss = a_y / v
K_yaw(v, delta) = a_y / [v^2 * tan(delta)]
```

Using only the existing 3 and 5 m/s bags, the entire 4 m/s bag was held out;
0.42 and 0.46 rad were also withheld from the steering curves. Across 12
leave-one-repetition-out predictions, the lateral-acceleration surface gave
0.00779 1/m yaw-gain RMSE and 0.01431 rad/s yaw-rate RMSE, versus 1.37372
1/m for the configured steering gain. All tested local steering sensitivities
were finite and positive (0.404–0.803 s^-1). Direct 4 m/s PCHIP validation
independently gave 0.00376 1/m gain RMSE, 0.00708 rad/s yaw-rate RMSE, and
0.608–0.767 s^-1 positive sensitivity. This is accepted as a steady-state
response model for the tested 3–5 m/s high-steering band only; it has not yet
been inserted into the MPC or validated in a practice-lap rollout.

An additional unseen-speed check used only the complete 4.5 m/s group from
the later-aborted high-speed capture, with 3 and 5 m/s bags as training data.
At held-out 0.35/0.42 rad angles (both signs, three repetitions), the
lateral-acceleration blend achieved 0.00889 1/m yaw-gain RMSE and
0.01507 rad/s yaw-rate RMSE, reducing configured-law RMSE by 99.6%; repeated
condition SD was at most 0.00055 1/m. This supports steady-state interpolation
at a second unseen speed. It does not model the separately observed negative
yaw-response slope from 0.15 to 0.20 rad, validate transients or per-wheel
forces, or justify MPC integration by itself. The later 6.5 m/s recovery
aborted, so no complete high-speed sweep is claimed.

Do not extend that interpolation down to 2.2 m/s. The 2.2→5 m/s linear yaw-
gain blend missed the held-out 3 m/s capture by 0.625 1/m. Blending lateral
acceleration reduced error to 0.189 1/m but produced negative steering
sensitivity, so that broad interpolation is rejected. At 3 m/s the direct
steering spline predicts held-out outputs at 0.0487 1/m RMSE, but its
sensitivity changes sign at some high-angle points; the rear-slip state
feature remains useful there. At 2.2 m/s the direct steering spline is
inaccurate (0.496 1/m), consistent with the separate rear-slip-dependent
response branch.

One 5 m/s throttle-contrast capture,
`live_runs/openplane_combined_slip_5mps_20260926/run/run_0.db3`, showed why
longitudinal slip cannot be interpreted without conditioning on speed. The
0.08 throttle phases decelerated from 5.0 to about 1.93 m/s, while 0.38 held
near 5.3 m/s; rear |Sx| separated strongly but speed and slip were confounded.
Adding rear |Sx| improved held-out yaw-gain RMSE only 4.1%, with the expected
effect direction in 2/4 conditions. The high-speed rear-slip correction is
rejected. The non-identifying fixed-throttle profile was removed from
the excitation scheduler; preserve its bag and evaluator as negative evidence.

The reusable cross-speed analysis is
`tools/evaluate_open_plane_speed_steering_surface.py`. Do not extrapolate
below 3 or above 5 m/s, assume transient response is captured by this
steady-state map, or promote it to the competition MPC until horizon
prediction and a gated practice run pass. The next distinct work package is
integration of this bounded response surface and explicit validation of the
unmodelled low-speed slip branch.

The guide-informed lateral-only force model fit net lateral force well at
5.0 m/s (1.724 N holdout RMSE) but inferred non-positive yaw inertia under
both Ackermann label assignments. Its current settled-only fit has inadequate
yaw-acceleration excitation; do not interpret that rejection as disproving
the guide curve.

See [`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md)
for the full equations, data, limitations and promotion gate. No production
MPC or physics change was made. The accepted empirical model is bounded to
steady high-angle behavior from 3.0 to 5.0 m/s; below 3.0 m/s the rear-slip
branch remains separate and above 5.0 m/s no interpolation has been validated.
The next work package is integrating this bounded response surface into the
MPC stage model, validating horizon predictions and optimizer sensitivities,
and gathering only the missing low-speed-slip and >5 m/s evidence. Do not
repeat the completed steady-steering sweep. Keep the simulator image and
physics unchanged.

## Latest transient-response package — 2026-09-26

The new `transient_4mps` open-plane profile completed in the pinned explore
simulator's Xvfb batchmode. The fresh bag is
`live_runs/openplane_transient_4mps_holdout_20260926/run/run_0.db3` (about
14 MB): 56/56 step phases valid, 8/8 matched sequence starts, all streams
39.977 Hz, zero collisions and zero timing faults. The simulator container
was stopped after the capture; no Docker containers remain running.

Frame-corrected front lateral-slip proxies rise from 0.2125 at 0.30 rad to
0.4345 at 0.50 rad, while rear lateral-slip proxies stay around 0.0027–0.0031.
Over the same change,
`K_yaw` falls from 0.968 to 0.618 1/m (36.1%) but sign-corrected yaw rate
increases from 1.190 to 1.344 rad/s (12.9%); local finite-difference yaw
sensitivity remains positive. Paired up/down differences are <=0.004 1/m,
below the predeclared 0.05 1/m history-effect gate, including the untouched
fourth repetition. The supported explanation is a repeatable nonlinear
steering-to-yaw surface consistent with front-tire authority loss, not a
steady hysteresis state. Slip remains a proxy, not a force measurement.

An offline model trained on repetitions 1–3 and scored once on repetition 4
fit the measured high-angle yaw knots plus a first-order yaw state (effective
tau=0.010 s, no added sample delay). Its conditional open-loop yaw RMSE was
0.00089, 0.03597, 0.00898, 0.00710 and 0.01280 rad/s at 25, 125, 250, 500
and 750 ms; the configured yaw law gave 1.43373, 2.25123, 2.31916, 2.31652
and 2.31467 rad/s on the same samples. This passes the predeclared yaw gate
for this narrow 4 m/s test. The analogous fitted `v_y` response passes 125–750
ms but fails at 25 ms (0.00030 vs 0.00009 m/s), so it is rejected for MPC
promotion. No runtime/controller parameters changed. Full per-wheel tire
force identification remains blocked by absent load, force and front-wheel
speed observations.

The durable capture, held-out response and current limitations are recorded
in [`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).
Continue by validating speed interpolation and optimizer Jacobians with this
yaw model; do not carry the rejected `v_y` model into MPC or infer tire forces
from kinematic slip alone.

## Rear-encoder slip versus the 6.5 m/s yaw trough — REJECTED — 2026-09-27

The existing rear-slip feature was tested on the original fine 6.5 m/s
transition capture and its independent confirmatory run. In each bag,
repetitions 1–2 fit a condition-centered rear-slip coefficient and repetition
3 was held out (32 training and 16 held-out steering blocks per bag). The
condition-only yaw-gain RMSE was 0.0136 1/m. Adding mean absolute rear
longitudinal slip reduced it by only 0.3% in the original bag and 1.0% in the
independent run, below the predeclared 10% criterion. The fitted coefficients
also varied from +5.394 to +2.061 yaw-gain units per slip ratio.

This is a negative result for the *mean rear-slip correction* at this
operating point, even when body speed is taken from the offline simulator
odometry. Rear encoder-derived angular rates remain available legal sensor
data, but they do not explain the narrow 0.145–0.180 rad yaw-rate minimum in
these captures. Do not add this feature to the MPC/offline transition model
for that regime. It does not rule out front-wheel slip, load transfer,
combined slip, or steering/tire transient state. Reproduce with:

```bash
source /opt/ros/jazzy/setup.bash
python3 tools/evaluate_open_plane_rear_slip_effect.py \
  live_runs/openplane_transition_65mps_20260927_01/run/run_0.db3 \
  live_runs/openplane_transition_65mps_confirm_20260927_01/run/run_0.db3
```

Work-package status: **REJECTED** for rear-slip augmentation of this yaw
surface; continue toward synchronized per-wheel measurements or a complete
legal-input `u,v,r` transition validated on whole held-out sequences.

## Simulator availability — BLOCKED on host container metadata repair — 2026-09-27

The host Docker daemon is in a restart loop, so no fresh real-simulator run
could be started in this continuation. Read-only service logs show containerd
panicking in bbolt metadata page traversal (`freepages` key ordering), while
dockerd separately panics in the libnetwork bbolt store (`invalid page type`).
The Docker API and Docker Desktop socket are unavailable. This is a host
runtime metadata failure, not a ROS/controller or vehicle-model failure.
Do not delete Docker/containerd snapshots or database files to work around it:
repair requires a privileged, backed-up recovery of the affected runtime
metadata before launching the pinned Explore image. Offline evaluation of the
saved bags can continue; new instrumented or pinned-player evidence must wait
for the daemon to be restored.

Live recheck at 2026-09-27 09:05 +02:00 confirms the same root cause: systemd
shows Docker `activating` with 27 restarts and containerd `activating` with 332
restarts. Containerd panics at bbolt leaf page 40 because the `containers`
key sorts before its `snapshots` ancestor; dockerd independently panics in its
libnetwork bbolt `List` with `invalid page type: 7:10`. `docker info` returns
client details but times out before server details. The rootful socket exists,
but `sudo -n` cannot authenticate. The Docker Desktop context points to a
missing socket; Podman and nerdctl are not installed, and no pre-existing
rootless Docker/containerd store is present. Root filesystem capacity is 249
GB free (72% used), with 7% inode use, so current failure is not disk or inode
exhaustion. No Docker/containerd metadata or snapshots were modified. A fresh
rootless daemon would require a separate image store and likely duplicate the
simulator image, so it is not an acceptable silent workaround under the
workspace's storage constraints. Required external action: an administrator
must stop the restart loop, preserve cold backups of the exact affected
metadata stores, and restore/recover those stores before the pinned Explore
image is attempted again. Do not prune or reinitialize either store before
those backups exist.

## Latest package — full-angle 4 m/s yaw response — IMPLEMENTED

Run `openplane_fullsteer_4mps_20260926` completed in the pinned explore
simulator through the established batchmode/Xvfb launcher. Bag:
`live_runs/openplane_fullsteer_4mps_20260926/run/run_0.db3`. It passed 114/114
valid steering steps, 6/6 matched starts, 39.972 Hz on all recorded dynamics
streams, zero collisions and zero bridge timing faults. The simulator was
stopped after the run.

The previously unmeasured 0.20–0.25 rad region contains a sharp,
repeatable steering-to-yaw collapse: mean yaw falls about 49.5% while front
lateral-slip proxy rises from 0.070 to 0.115. The 4 m/s effective steering
map plus first-order yaw state passed repetition-3 holdout (0.03852 rad/s
steady yaw RMSE; correct sensitivity sign throughout; 88–98% yaw-horizon error
improvement versus the configured law). This establishes a useful bounded
nonlinear response model, not a per-wheel force model or closed-loop result.

The guide's public data schema exposes no per-wheel contact forces, dynamic
loads, suspension displacement, or front wheel speeds. Keep those latent; use
the validated empirical response and measured kinematic slip proxies without
claiming force identification. Next: refine the 0.18–0.28 rad transition at
3/4/5 m/s, then test the speed-conditioned map and its Jacobian in the MPC
stage model. Details and exact evaluator results are in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).

## Four-second branch dwell — 2026-09-26 — REJECTED for MPC promotion

Run `openplane_transition_dwell_20260926` used the pinned explore simulator
through the established Xvfb batchmode launcher. It held fixed throttle
command 0.164 and steering levels 0.20–0.23 rad for four seconds, in both
turn directions and three repetitions. Repetitions 1–2 were the development
set; repetition 3 was held out.

The capture passed: 42/42 scored phases and 6/6 matched starts; odom,
steering, both rear encoders, IMU, and packet-timing streams ran at 39.971 Hz;
there were zero collisions, timing faults, or quality failures. The command
stream ran at 39.76 Hz. Bag:
`live_runs/openplane_transition_dwell_20260926/run/run_0.db3`.

At 0.21 rad, the up sequence retained high yaw gain (2.8646 1/m) while the
down sequence retained low yaw gain (1.4279 1/m); the same difference
reproduced in held-out repetition 3. Within each branch the first 0.5–1.2 s
and final 0.7 s gains were effectively unchanged. At 0.22 rad, the up
sequence transitioned from 2.5429 to 1.2986 1/m during the four-second hold;
the down sequence was already on the low branch. This is strong evidence of
a long-lived history-conditioned state and a sharp transition in this range,
but four seconds does not establish distinct steady-state equilibria or prove
which physical state carries the memory.

The fixed throttle command removes changing speed-controller output as the
explanation, but does not match actual speed: the 0.21-rad branches averaged
3.863 and 4.007 m/s. Rear longitudinal-slip proxies also differed (0.0738
versus 0.0338), as did front lateral-slip proxies (0.0208 versus 0.1161).
These are useful correlates, not isolated causes. The current rear-slip
blockwise evaluator expects isolated probes and cannot score this sequenced
profile; no causal slip-conditioned predictor is accepted from this run.

The four-wheel guide-curve plus effective linear yaw-drag fit again inferred
an impossible inertia (unconstrained -0.15575 kg m^2; bounded solution stuck
at the 0.005 lower bound). Held-out yaw-acceleration error rose from 0.402
rad/s^2 for the zero-acceleration baseline to 29.788 rad/s^2. Reject this
force/moment model; do not tune its coefficients or put it in MPC.

The current competition stage model also has a directly measured mismatch:
its steering-gain taper starts only at 0.41 rad (2.95 1/m below that), while
the 4 m/s holdout measured 1.152 1/m at 0.25 rad and 0.968 1/m at 0.30 rad.
At 0.50 rad the configured taper gives 1.17 1/m versus 0.618 measured. Thus
the model overpredicts steady yaw authority by 2.6x at 0.25 rad, 3.0x at
0.30 rad, and 1.9x at 0.50 rad. At 0.21 rad it also predicts one response
where the holdout reproduces two history-conditioned branches (2.86 and
1.43 1/m). This is a confirmed model mismatch and a plausible source of bad
high-angle rollouts; existing data do not yet prove it is the cause of a
specific solver rejection or fallback.

No production physics, MPC, localization, odometry, or topic policy changed.
This work package is **REJECTED for model promotion**: the capture confirms
the branch but the fitted force model fails and the available telemetry does
not causally identify the hidden state. Do not repeat the same dwell sweep.

## Fallback package — rear-slip causal test if internal telemetry is unavailable

Use a controlled throttle/steering experiment that matches actual speed and
steering while creating independent variation in rear encoder-derived
longitudinal slip. Keep turn signs and sequence order balanced; use complete
repetitions as holdouts. Compare only the predeclared causal candidates:
steering-only response; response conditioned on rear longitudinal slip; and
a compact first-order slip/yaw state. Ground-truth pose and velocity are
offline targets, not inputs to a competition predictor. If this excitation
cannot separate speed, drive slip, and lateral tire saturation, record the
model as non-identifiable from public telemetry rather than fitting more
parameters.

Predeclared acceptance: every scheduled probe is valid; all six measured
streams are >=38 Hz with p95 gap <=35 ms and maximum <=60 ms; zero collision,
bridge timing fault, or abort. Repetition 3 is held out. The capture is
identifying only if it yields at least two repetitions of paired samples at
the same measured steering within 0.005 rad and speed within 0.05 m/s, with
rear `|Sx|` separated by >=0.02. Otherwise reject the capture as
non-identifying and do not fit. A candidate model must lower held-out yaw-rate
RMSE by >=20% at three or more horizons in 125–750 ms, regress no horizon by
>10% versus the deployed model, avoid >10% first-step regression at 25 ms,
and predict the observed finite-difference steering-sensitivity sign wherever
the measured magnitude is >=0.05 s^-1.

This public-signal test is no longer the next action. Use it only if a
source-built telemetry player fails the behavior-equivalence gate below; do
not rerun the prior sweeps or simple early-feature fit. Promotion would still
require a competition-input-only rollout to improve held-out yaw prediction,
preserve the measured sensitivity sign, and avoid first-step/long-horizon
regression before any MPC or real-lap change.

## Latest identification package — 2026-09-27 — REJECTED for promotion

The saved 4 m/s transition bag was re-evaluated with a leave-one-repetition-out
body-state model. A smooth steering RBF was interacted with predicted lateral
velocity and yaw rate; those states were propagated recursively. The model
improved 125/250 ms yaw RMSE by 25.5/45.3%, but regressed 82.1% at 25 ms,
172% at 500 ms, and diverged before 750 ms. Lateral-velocity error also
regressed at 25 ms and after 250 ms. It fails the predeclared coupled-state
gate and must not enter the MPC. Future ground-truth speed and measured
steering were still conditioned inputs, so this was not a full offline-sim
validation. Evaluator: [`evaluate_open_plane_coupled_lateral_model.py`](../../tools/evaluate_open_plane_coupled_lateral_model.py).

A separate held-out feature check on the transition sequence found that early
legal IMU, steering/throttle feedback and rear encoder rates did not improve
the steering/direction baseline: pooled yaw-gain RMSE changed from 0.0117 to
0.0119 1/m; only one of three held-out repetitions improved. This does not
prove that the branch is unobservable, but does reject that simple linear
history correction. The run used saved data; no simulator or runtime stack was
changed. This was an ad-hoc diagnostic, not a standalone reusable evaluator;
the bag, feature windows, split and scores are documented in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).

The same 4 s dwell bag confirms that each rear `JointState` has no velocity or
effort values. Both angle streams ran at 39.971 Hz. Differencing encoder angle
over 100 ms gives median rear rates of 69.95/69.93 rad/s in the moving capture;
25 ms differencing is noisier. These are measured rear-wheel rates, not body
speed, and the full-run slip quantiles include transitions. There is no front
wheel-rate signal in the public ROS stream. See
[`UNITY_SIMULATOR_ASSET_AUDIT.md`](UNITY_SIMULATOR_ASSET_AUDIT.md) for the
serialized physics values and sensor/hidden-state inventory.

## Source observability package — 2026-09-27 — IMPLEMENTED (audit only)

The public AutoDRIVE simulator source branch is usable as a telemetry
instrumentation candidate: it uses the same Unity editor version as the
pinned player, and its RoboRacer/F1TENTH prefab's wheel, suspension, friction,
steering and drive settings match the loaded player. The pinned binary and
public source do not name the same active scene (`Copy` versus `Explore`), so
whole-player equivalence is not assumed. The binary's active floor and
inactive tracks are consistent with the open-plane setup. The detailed
comparison is in [`UNITY_SIMULATOR_ASSET_AUDIT.md`](UNITY_SIMULATOR_ASSET_AUDIT.md).

The source's `WheelEncoder.RPM` reads native `WheelCollider.rpm`, but the
bridge exports only quantized rear encoder angles/ticks. A development-only
instrumented build can read all four wheel RPMs, Ackermann angles, contact
slips/geometry/force magnitude and applied motor/brake torque without writing
to physics state. Unity's public API does not expose separate per-wheel
longitudinal and lateral force components. The runtime's actual wheel-local
slips must be computed from COM velocity plus yaw-rate-at-wheel and each
wheel's Ackermann angle; the guide's equations alone are not a complete fitted
plant. Ground truth constrains aggregate forces/moments but cannot uniquely
recover four tires' forces.

This audit package is **IMPLEMENTED**; no simulator, production physics, MPC,
localization, odometry or topic policy changed. The instrumented-player run is
not yet accepted. Before using any new telemetry as labels, replay the same
open-plane command trace in the pinned and source-built players. Accept only
if their public motion remains within the existing same-profile repeat
envelope, stream rates stay at the established 40 Hz quality, and there are
zero collisions/timing faults. Otherwise reject that build's internal data and
request/use the exact project for the pinned scene. Only after equivalence
passes should a new high-angle identification capture be designed; do not
continue blind tire-curve fits or add more arbitrary MPC state terms.

## High-rate per-wheel observability run — 2026-09-27 — capture implemented; model promotion rejected

The development-only recorder in `tools/unity/OpenPlaneWheelDynamicsCapture.cs`
was built into a temporary source player and run with the established explore
container in Xvfb batchmode. Run and data are preserved at
`live_runs/openplane_source_player_equivalence_retry2_20260927/`. Public
acceptance passed: all 24 steering probes valid with matched starts, 39.98 Hz
telemetry, p95 gaps below 27 ms, zero collisions, zero in-run timing faults.
The 1 ms capture found a front-slip/yaw-authority association: between 0.30 and
0.50 rad, yaw gain fell about 57% as front `|Sy|` rose about 31x; the rear
lateral-slip proxy stayed near zero. A 0.42 rad split branch was also visible.
This gives a concrete candidate mechanism that pose-only fits could not see.

The temporary source player's high-angle public response is below the
pinned-player repeats at 0.42–0.50 rad, so its internal wheel labels are
**not valid training data for the pinned model**. Unity `WheelHit` exposes a
contact-force magnitude, not per-wheel tangential force components; the current
capture cannot directly identify `Fx(Sx)`/`Fy(Sy)`. No MPC, localization,
odometry, runtime topic policy, or simulator physics was changed. Exact source
for the pinned active scene/build is now the external dependency for a valid
per-wheel force fit. The pinned image uses IL2CPP: pinned `GameAssembly.so` is
99,868,608 bytes with 18,458,992-byte global metadata; the public-source build
produced a 569,883,160-byte library with 13,887,812-byte metadata and different
assembly manifests. A library-only swap is unsafe and was not attempted.
Until exact-build source is available, preserve the accepted bounded
body-response surface and do not repeat the same angle sweep or tune MPC
weights against source-player data.

## Rear-wheel odometry screen — 2026-09-27 — REJECTED as a standalone correction

Existing matched-start high-steering bags at 2.2/3/4/5 m/s were analyzed
offline. Rear encoder rates from 100 ms angle slopes, converted with the
documented 0.059 m radius, miss GT wheel-local longitudinal speed by p50/p95
0.187/0.312, 0.150/0.186, 0.130/0.138 and 0.123/0.137 m/s, respectively.
The YAML speed-scale table changes these only marginally and slightly worsens
the 4 m/s median. Sparse straight windows cannot identify a wheel-radius
correction; the left/right differential also fails held-out prediction of
common-mode slip. This was an offline sensor-model screen, not full C++
observer replay. No odometry setting changed. Full method and results:
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).

Work-package status: **REJECTED** for direct rear-wheel-speed substitution or
an extra left/right correction. Continue with a causal observer/history
candidate only if it can be trained and evaluated on complete held-out runs
using permitted inputs, without GT at inference; otherwise retain current
observer. This speed residual test does not identify the tire-force
nonlinearity responsible for the high-angle yaw branches.

## High-speed steering transition — 2026-09-27

The 4.5 m/s portion of `openplane_isolated_highspeed_surface_20260927`
completed 60/60 matched probes across three repetitions. From 0.15 to 0.20
rad, yaw gain fell 2.925→1.199 1/m and net lateral acceleration fell
0.890→0.496 g; reconstructed front lateral-slip proxy rose 0.0135→0.126.
This closely tracks the documented tire peak-to-asymptote force ratio and
strongly implicates front-tire saturation, while remaining an inference from
chassis motion. At 6.5 m/s, the front proxy was already beyond 0.1 and lateral
acceleration stayed near 0.55 g as yaw gain fell; speed and throttle demand
therefore change the response regime. The 4.5 m/s train/holdout curve predicts
16 withheld angle/sign points at 0.0238 1/m RMSE, but the full run aborted
before 7.5 m/s and the observed steering Jacobian is negative across the
transition. No model or MPC weights changed.

The run ended after releasing +0.35 rad at 6.5 m/s: yaw reversed and reached
about 3.2 rad/s while rear encoder rate rose from 114 to 215 rad/s; the
speed-hold loop then saturated throttle and the safety cutoff fired at
9.465 m/s. Collision count and bridge timing faults remained zero; sensor
streams stayed at 39.977 Hz. Do not repeat the same speed-hold straight reset
after a high-angle transient. Treat the cross-speed capture as rejected for
promotion; retain its complete 4.5 m/s subset as exploratory evidence.
Detailed analysis is in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).

## Exact-speed transition model — 2026-09-27 — REJECTED for MPC promotion

To separate cross-speed error from steering-curve error without another
simulator run, repetitions 1–2 of the complete 4.5 m/s group trained a
steering PCHIP with 0.21/0.22/0.23 rad removed; repetition 3 at those angles
and both signs was held out. Yaw-gain RMSE was 0.01312 1/m, maximum error
0.03004 1/m, and 99.3% below the configured-law RMSE. However, the analytic
local steering derivative had the wrong sign near 0.215 rad for both signs
and near 0.24 rad for one sign. One held-out +0.23 rad phase had p95 speed
variation of 0.299 m/s, so state/speed confounding remains. This validates a
close steady-output fit, not an optimizer-safe Jacobian. Do not integrate it
or retune MPC weights.

Separate cross-speed interpolation through the 0.15–0.25 rad band scored
0.65502 1/m RMSE against a 0.10 gate and had multiple wrong sensitivity
signs. A front-slip-only response curve scored 0.13663 1/m and also failed
the derivative gate. These are not failures of the published equations: the
public bag contains neither true individual-wheel slip/load nor signed
per-wheel force, and GT chassis motion constrains only aggregate force and
moment. A physically meaningful four-wheel fit requires equivalent-build,
read-only wheel/load/contact instrumentation; absent that, the next valid
target is a bounded legal-input body-response model with held-out Jacobian
and trajectory validation. The exact test and acceptance rationale are in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).

## Per-wheel capture integrity — BLOCKED pending Unity integration — 2026-09-27

The 60k-row source-player wheel CSV from
`openplane_source_player_equivalence_retry2_20260927` overwrote 50,716
samples and has no shared-clock timestamps. Its steering sequence does not
match the attached ROS bag after the initial +0.50 rad command, so the files
are not a defensible same-run pair. Keep the CSV as unpaired source-only
telemetry; do not draw cross-sensor or phase-fit conclusions from it. The
recorder source now has a 180k bounded buffer, per-sample UTC-mapped
timestamps, and the missing inertia-tensor frame rotation. Paired analysis
also requires body-speed and actuator-command agreement. The Unity project is
unavailable here, so those edits remain uncompiled and unvalidated until a
new instrumented open-plane run. The offline evaluator and bag reader are
implemented; syntax, bag decoding and stale-capture rejection pass. Work
package status: **BLOCKED** on the exact Explore simulator source project. No
vehicle or competition-runtime code changed.

## Cross-speed steering and excitation integrity — 2026-09-27

The first `openplane_grid_combo_20260927_01` capture supplied the initial
2.5/4.5/6.5/7.5 m/s surface, with only one observation per sign/angle. At
7.5 m/s, yaw gain was already
0.647 1/m at 0.15 rad and the reconstructed front `|Sy|` proxy is 0.123,
past the guide's asymptote landmark; by 0.42 rad gain is 0.261 1/m and the
proxy is 0.420. This supports a speed-dependent front saturation transition,
not a statistically validated per-wheel tire curve. The next high-speed
capture must resolve below 0.15 rad and hold out complete repeats.

The grid also found that an overspeed guard changed fixed-throttle commands to
zero inside a phase. Since idle torque is active braking, its coast/brake
segments are rejected for model fitting. The development excitation now
applies that guard only to speed-regulated phases, logs each requested fixed
input in phase metadata, and the bag analyzer gates recorded fixed-command
fidelity. The 9 m/s emergency cutoff remains unchanged. This is a development
measurement-integrity fix only; no vehicle physics or competition stack was
modified. See the detailed gate and 4.5/6.5/7.5 m/s measurements in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).

After removing the confounded combined-throttle tail, the corrected steering
grid completed 48/48 phases in the pinned Explore simulator's established
Xvfb batchmode. All 40 steering probes passed; odom/steering/encoders/IMU and
packet timing were 39.981–39.982 Hz, p95 gaps 25.56–25.68 ms, maximum gaps
47.91–47.99 ms, with zero collisions/faults. Across three captures the
4.5/6.5/7.5 m/s steering responses reproduce within 0.0012 1/m at shared
0.15/0.25/0.42-rad angles. At 7.5 m/s, yaw gain is 0.647 at 0.15 rad and
0.261 at 0.42 rad;
IMU lateral acceleration independently corroborates the body response.

A 4.5/7.5-to-6.5 m/s lateral-acceleration interpolation achieved 0.0845 1/m
held-out yaw-gain RMSE (96.5% below the configured model), but only 6/8 local
steering-derivative signs matched; actual held-out slopes were positive
0.224–0.738 1/s while the candidate had a negative region. **REJECTED** for
MPC promotion. The configured speed-independent gain 2.95 1/m is plainly
wrong for this operating band, but this fit is not optimizer-safe. The grid
also couples target speed with the speed-hold throttle (roughly 0.18 at
4.5 m/s, 0.31 at 7.5 m/s), so it does not identify a pure speed-only tire
law. Next: targeted low-angle (<0.15 rad) high-speed capture and short,
speed-matched throttle perturbations; include actual throttle and rear
encoder slip in offline candidates, with whole-run and Jacobian holdouts.
Exact per-wheel force identification remains blocked on synchronized native
wheel/load instrumentation from a behavior-equivalent pinned build.

## Low-angle high-speed response — 2026-09-27

The targeted `isolated_highspeed_crossfactor` profile replaced the broad
steering-only grid with 0.05–0.25 rad points and small fixed-pedal offsets.
The initial ±0.025 throttle contrast was stopped at 35/184 phases: it moved
speed by over 1.2 m/s and was non-identifying. The narrowed ±0.003 run
completed its full 6.5 m/s subgroup (90/90 probes) and only one 7.5 m/s
repetition before the 9.38 m/s emergency cutoff during zero-steer recovery.
Neither capture had collisions or bridge timing faults; the second's public
streams remained 39.978 Hz. Do not repeat its high-speed return-to-zero
recovery sequence.

The 6.5 m/s speed-hold surface shows yaw gain dropping from 2.959 at 0.05 rad
to 2.374 at 0.10, 1.343 at 0.125, 0.844 at 0.15, then 0.630 at 0.20. This
locates the onset far below the current model's 0.41 rad taper. A PCHIP trained
on repetitions 1–2 predicts repetition 3 at 0.0213 1/m yaw-gain RMSE (versus
1.8080 for production), but its local derivative signs match only 14/16
intervals. It predicts positive slope from 0.15–0.175 rad while both held-out
turn directions measure a small negative slope. **REJECTED for MPC promotion**;
the output fit alone is not enough.

The 6.5 m/s ±0.003 pedal contrast met the speed-gap limit in 18/18 pairs and
its throttle feature improved holdout RMSE by 14.1%; a model using measured
speed alone scored nearly identically (0.0488 versus 0.0483 1/m). No
independent throttle coefficient is accepted because speed and throttle are
not separated. The 7.5 m/s response is single-repetition descriptive data
only. Detailed results are in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md) and
[`evaluate_open_plane_highspeed_crossfactor.py`](../../tools/evaluate_open_plane_highspeed_crossfactor.py).

The principal missing MPC physics is now explicit: `vehicle_model.c` uses a
first-order yaw-rate map and holds predicted lateral velocity constant; it
does not represent the measured high-speed steering trough or its changing
Jacobian. This motivated the fine 6.5 m/s capture summarized below. Do not
tune weights to compensate for the missing state transition; promote no
candidate until its one-step, recursive rollout, and local-Jacobian holdouts
pass.

That fine 6.5 m/s capture is now complete at
`live_runs/openplane_transition_65mps_20260927_01/run/run_0.db3`: all 48 probes
valid, 39.977 Hz streams, no collisions or timing faults. A repetition-3
holdout exposed a narrow yaw-rate minimum around 0.160 rad (about 0.819 rad/s)
between 0.145 and 0.180 rad. Gain-PCHIP had excellent pointwise fit
(0.0001 1/m RMSE) but only 12/14 local derivative signs matched, so it remains
rejected. Direct interpolation of yaw-rate magnitude scored 0.00016 rad/s and
14/14 signs on that same holdout, but was formulated after inspecting the
holdout. It was frozen and then independently tested on a new randomized
run: 0.00022 rad/s RMSE across 48 probes and 42/42 resolvable Jacobian signs,
with zero collisions/faults and 39.978 Hz streams. **Accepted only as a
steady-state yaw-response component**, not an MPC-ready model. Next fit and
validate the full `u,v,r` transition, including steering/throttle feedback
and rear wheel rates, on untouched sequence holdouts and 25–750 ms rollouts.
Details and exact commands are in `OPEN_PLANE_PER_WHEEL_DYNAMICS.md`.

## Explore runtime recovery and fresh matched-start data — 2026-09-27

The system Docker/containerd stores are unusable: containerd reports a bbolt
snapshotter free-page/ancestor mismatch and dockerd reports an invalid
libnetwork bbolt page. I left those stores untouched and installed the
standard user-level rootless Docker service, selecting its `rootless` context.
The repository's `tools/docker_env.sh` automatically selects its socket. The
pinned Explore and API images now run from the separate
`~/.local/share/docker` store; the user service is enabled and active.

Two launches of the official Explore image through
`tools/start_simulator.sh` in the established Xvfb `-batchmode` path initialized
Vulkan on the GTX 1080 and PhysX without a segmentation fault. A matched-start
`isolated_force_3mps` capture completed all 24/24 scored probes: every scored
phase passed, source streams were 39.975 Hz (p95 gaps 25.59–25.73 ms, max
48.9–49.0 ms), collisions and timing faults were zero, and the run was not
aborted. The bag is
`live_runs/openplane_rootless_isolated3mps_20260927_01/run/run_0.db3`.

On this capture, leave-one-repetition-out interpolation at 0.42/0.46 rad gave
0.02549 1/m yaw-gain RMSE versus 1.06342 1/m for the current fixed steering
taper; measured steering sensitivities were positive (minimum 0.0941 s^-1).
At 3 m/s, repeated mean gain falls from 1.567 1/m at 0.30 rad to 1.090 at
0.42, 0.996 at 0.46, and 0.949 at 0.50. The front lateral-slip proxy rises
from about 0.15 to 0.37 while the rear proxy remains near 0.002. This supports
an empirical nonlinear front-authority loss; the proxy is not a direct tire
force measurement and does not imply a sinusoidal law. This is within-run
steady-state cross-validation only, not a recursive or whole-run MPC holdout.
Do not promote it to production MPC on this evidence alone.

The `high_angle_boundary` smoke captures are runtime evidence only: some phase
starts retained up to 3.2 rad/s yaw from the preceding probe despite passing
speed/steering checks, so their gains are confounded and must not train a
steady map. The bridge now handles SIGINT/SIGTERM through its normal cleanup;
the repeated real run emitted no false fatal publisher-context error. Both
bags had zero recorded timing faults. No simulator physics, odometry, AMCL,
EKF, or MPC behavior parameters changed during this recovery.

An independent-run interpolation check trained only on the prior 3 m/s
capture's ±0.30/±0.50 rad points, then predicted all 12 new-run ±0.42/±0.46
rad blocks. Interpolating yaw gain gave 0.07881 1/m / 0.10827 rad/s RMSE
versus 1.06342 1/m / 1.41134 rad/s for the production taper, but predicted a
negative local yaw-rate derivative on both turn signs while the held-out
finite differences were positive. Directly interpolating curvature
`q=|r|/u` gave 0.03195 rad/s held-out yaw-rate RMSE and positive derivatives
(+0.223/+0.176 s^-1), matching both measured signs (+0.490/+0.126 s^-1).
This is the strongest candidate so far for the measured 3 m/s high-steering
response, but the evidence is limited to one speed and two withheld
intermediate angles. It does not establish transient `u,v,r` prediction or
justify MPC integration. Fit the optimizer's output and derivative together:
a gain curve can fit the values while yielding a wrong Jacobian.

## Fresh 3–5 m/s response and cross-speed holdout — 2026-09-27

The pinned Explore image completed new `isolated_force_4mps` and
`isolated_force_5mps` captures in the same Xvfb `-batchmode` launch. Both
completed 26/26 phases and 24/24 matched probes, with no collisions, timing
faults, or quality failures. Public odometry averaged 39.974 and 39.969 Hz.
The first 5 m/s run had a single 100.14 ms source-response gap; it did not
recur after restarting the same simulator and randomizing the order. The
repeat completed with a 39.968 Hz rate and 48.45 ms maximum gap. The 4 m/s capture is
`live_runs/openplane_rootless_isolated4mps_20260927_01/run/run_0.db3`; the 5
m/s repeat used for strict model validation is
`live_runs/openplane_rootless_isolated5mps_repeat_20260927_02/run/run_0.db3`.

Median yaw gain `K=r/(u*tan(delta))` over both turn directions changes
substantially with speed:

| Speed | 0.30 rad | 0.42 rad | 0.46 rad | 0.50 rad |
| ---: | ---: | ---: | ---: | ---: |
| 3 m/s | 1.584 | 1.096 | 1.007 | 0.924 |
| 4 m/s | 0.967 | 0.721 | 0.659 | 0.618 |
| 5 m/s | 0.679 | 0.509 | 0.471 | 0.448 |

These results do not support one steering-only gain or a sinusoid as a
mechanistic explanation. The observed response is better represented as a
speed- and steering-conditioned lateral acceleration `a_y(u,delta)`, then
`r_ss=a_y/u`. A model trained on independent 3 and 5 m/s captures, with the
fresh complete 4 m/s capture held out, blended lateral acceleration across
speed and used PCHIP in steering. On 12 withheld ±0.42/±0.46 blocks it scored
0.01584 1/m yaw-gain RMSE and 0.02872 rad/s yaw-rate RMSE, versus 1.37362 1/m
for the current configured gain. Its local `d(r_ss)/d(delta)` signs were
positive (0.512–1.258 1/s). Directly blending yaw gain scored 0.07827 1/m and
0.14828 rad/s. This supports the lateral-acceleration response coordinate,
but remains an aggregate steady-state model, not identified per-wheel tire
physics. The earlier 0.00927/0.01749 result used a different 5 m/s source;
the repeat above is the stricter, fully cadence-qualified evidence.

A recursive rollout trained from the fresh 3 m/s capture and the clean 5 m/s
repeat, then tested all 24 phases of the fresh 4 m/s run, reduced yaw-rate
RMSE by 92.6%, 97.8%, 99.2%, and 98.9% at 125, 250, 500, and 750 ms. At 25 ms
it regressed from `5.6e-5` to
`7.0e-5` rad/s (absolute difference `1.4e-5` rad/s); the fixed acceptance gate
does not permit that relative regression for the fitted 5 ms time constant.
Because tau is unresolved below 40 Hz, the same unchanged test was also run
with the existing 15 ms production time constant held fixed (not fit to the
holdout). That yaw-only candidate matched baseline RMSE at 25 ms and improved
by 90.2%, 97.8%, 99.2%, and 98.9% at 125–750 ms; it **passes the unchanged
recursive yaw gate**. This qualifies the empirical yaw surface only within
3–5 m/s and 0.30–0.50 rad. It does not validate longitudinal-speed or
lateral-velocity dynamics, low steering, or speeds outside that band, so it
is not yet integrated into production MPC. Lateral-velocity fitters still
regress at 25 ms; the production held-`v` model remains. No MPC, odom,
localization, or simulator behavior parameters changed. Reproduction details are in
[`OPEN_PLANE_PER_WHEEL_DYNAMICS.md`](OPEN_PLANE_PER_WHEEL_DYNAMICS.md).
