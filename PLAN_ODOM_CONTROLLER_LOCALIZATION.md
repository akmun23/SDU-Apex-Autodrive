# SDU Apex AutoDRIVE — Odom, Controller, and Localization Re-plan

**Status:** discussion draft; no implementation work is authorized by this file
**Date:** 2026-09-03
**Repository:** `/home/akselmo/Documents/GitHub/SDU-Apex-Autodrive`

This is the working plan for the next phase. It is deliberately separate from
`IMPLEMENTATION_STATUS.md` and from the downloaded handoff audit. The handoff
is technical reference material; this file records the user’s goals, the
evidence currently available, and decisions that still need to be agreed.

## 1. User requirements and hard constraints

These are the requirements to preserve while we discuss the implementation:

1. Build an accurate vehicle-motion model from IMU and encoder inputs,
   compared offline against simulator ground truth.
2. Include slip when the measurements demonstrate that it is necessary. Any
   runtime speed used for slip must be estimated from legal runtime sensors,
   never taken from ground truth.
3. Create two separate physical-unit controller interfaces:
   - Pure Pursuit publishes a speed target on `/cmd/speed`.
   - MPC publishes an acceleration target on `/cmd/acceleration`.
4. The speed test must accelerate quickly toward a target and then hold a
   constant or near-constant throttle that maintains that target. A changing
   throttle signal is not evidence of a successful steady-speed test.
5. Every open-world grid point must reset the car before the next point. A
   reset is permitted only after braking has continued until IMU, encoders,
   odometry, and diagnostic ground truth agree that motion has stopped.
6. A frozen encoder while the car is still moving is slip data, not a reason
   to reset or discard the sample.
7. Ground truth may be logged and used for offline fitting and diagnostics in
   these tests. It must not enter runtime odometry, EKF, AMCL, or control.
8. AMCL is not tested on the open map. Track localization must use the same
   map and raceline as the track simulator.
9. Native source rates and timestamps must be measured from messages rather
   than assumed from recorder rate or simulator frame rate.
10. Do not repeatedly run uncontrolled tests. Each planned run must have a
    defined purpose, finite matrix, reset/stop safety, output location, and
    analysis report.

## 2. Confirmed evidence before the re-plan

### 2.1 Current speed-test result

The most recent corrected open-world speed test was not valid controller
evidence and must remain rejected:

- The zero-speed phase and reset completed correctly.
- The first nonzero target was 4.0 m/s.
- Ground-truth speed in the tail was approximately 6.30 m/s.
- Local `/odom` reported approximately 3.30 m/s.
- Relative target-speed error was approximately 57.5% median.
- Throttle standard deviation was approximately 0.236 on a 0–1 scale, so
  there was no constant-throttle plateau.
- The run never reached the later 6–20 m/s targets.
- Native sensor cadence was approximately 11.5 Hz; speed commands were
  approximately 50 Hz.

This demonstrates a failure mode: the speed loop trusted an odometry estimate
that was below ground truth during slip, so it continued changing throttle
while the car was already overspeeding. It does not prove that the final
speed-controller architecture is impossible, but it does prove that the test
and odom/controller dependency must be separated more carefully.

The rejected raw run and its derived metrics are retained under:

```text
sdu_apex_autodrive/artifacts/calibration/archive/
rejected-speed-controller-20260903/
```

### 2.2 Current open-world odometry evidence

The latest post-fit open-world identification remains the best current odom
evidence, but it is not yet a final acceptance claim:

- At cumulative distances of at least 100 m, position error was approximately
  1.41% median and 2.52% p95.
- In the 15–20 m/s band, odom speed relative error was approximately 0.87%
  median and 1.06% p95.
- In the 20–23 m/s band, it was approximately 0.77% median and 0.84% p95.
- The identified slip data contains a wide traction distribution, so a single
  hard-coded traction correction would be unsafe without an independent run.
- A moving/frozen-encoder braking fit exists, but its runtime use still needs
  independent validation.

The important conclusion is that the current estimator is promising at speed
in selected regimes, while low-speed/transient/slip behavior and integrated
position acceptance remain open.

### 2.3 Current architecture boundary

The intended runtime boundary is:

```text
official encoders + IMU -> sensor odom -> local prediction/EKF state
track map + LiDAR       -> AMCL -> independent map-to-odom correction
Pure Pursuit            -> /cmd/speed       -> actuator abstraction -> throttle
MPC                     -> /cmd/acceleration -> actuator abstraction -> throttle
```

Simulator odometry, IPS, collision, reset, and other truth/debug topics are
diagnostic-only. AMCL must not be used to improve open-world IMU/encoder
identification.

The BachelorProject is a reference for the AMCL/EKF correction responsibility
split only. It is not the template for simulator test setup.

## 3. Proposed work sequence

No later stage should be called accepted until the preceding stage has a
report with raw-data provenance, native rates, metrics, plots or tables as
appropriate, and a clear pass/fail decision.

### Stage 0 — Freeze definitions and data provenance

Before another live run:

- define the exact legal runtime inputs for odom, EKF, AMCL, speed control,
  acceleration control, Pure Pursuit, and MPC;
- define which ground-truth topics are diagnostic-only;
- inventory the active raw and derived CSVs and mark rejected runs clearly;
- record the simulator image/scene, map choice, raceline choice, bridge
  revision, code revision, parameter files, and launch command;
- decide whether each run is a training run, a validation run, or a track
  acceptance run;
- preserve source timestamps and record callback/recorder timestamps
  separately.

**Deliverable:** one run manifest and one metric-definition document. No data
from intentional reset transitions may be included in motion-error metrics.

### Stage 1 — One controlled open-world identification suite

This is the main IMU/encoder/vehicle-model data run. It is one finite suite,
but it contains explicit blocks and resets at every independent operating
point. The recorder must log all blocks in one schema rather than creating
unexplained CSV variants.

#### Required blocks

1. **Stationary bias and noise:** zero throttle, stationary IMU, both encoder
   angles, odom, and ground truth.
2. **Throttle response from rest:** several throttle levels, including small
   throttle near breakaway and high throttle, with acceleration and coast
   tails.
3. **Speed-conditioned throttle response:** reach a measured speed band using
   a documented open-loop/pre-fit command, then probe throttle around that
   speed. The speed band is a label from ground truth for fitting only.
4. **Steady-speed plateaus:** for each target, select a candidate equilibrium
   throttle, accelerate quickly, then hold that throttle for a fixed dwell.
   Log whether the plateau actually remains stable; do not silently replace a
   failed plateau with feedback oscillation.
5. **Acceleration/deceleration:** measure positive acceleration by speed band
   and passive zero-throttle deceleration. Negative commanded acceleration is
   a separate experiment until a verified brake channel exists.
6. **Encoder/slip cases:** retain samples where encoder speed, IMU-integrated
   speed, and ground truth disagree, including the moving/frozen-encoder
   braking condition.
7. **Straight-line pose propagation:** use long enough isolated runs to
   measure integrated distance and yaw drift without crossing the open-world
   boundary.
8. **Optional steering/slip probes:** only after the longitudinal portion is
   safe and the steering sign, units, and source cadence are verified.

Each block must specify its throttle/target, minimum dwell, stop gate, reset
gate, and expected maximum travel before the run starts. The runner must stop
with an explicit failure if a target cannot settle; it must not continue into
the next grid point using invalid data.

#### Data to record

At every recorder row, plus source-event timestamps and event counters:

- raw throttle command and bridge throttle feedback;
- `/cmd/speed` and `/cmd/acceleration` when active;
- left/right encoder position, source timestamp, derived angular speed, and
  sample interval;
- IMU linear acceleration, angular velocity, orientation, source timestamp,
  and sample interval;
- team `/odom`, odom diagnostics, covariance, source timestamp, and interval;
- diagnostic ground-truth odometry, IPS, collision count, and reset epoch;
- calculated wheel speed, estimated body speed, estimated acceleration, and
  slip features;
- phase, target, throttle plateau statistics, stop-gate state, reset-gate
  state, and safety/boundary state.

The report must show native rate distributions, not only one average. The
recorder’s nominal 50 Hz rate is not a sensor-rate claim.

### Stage 2 — Fit the legal IMU + encoder odometry model

The offline fit should compare runtime-observable candidate estimates against
timestamped ground truth, then promote only the model that also passes an
independent run.

#### Candidate model structure

Use a piecewise/regime estimator rather than forcing one global polynomial:

```text
state: position, yaw, body speed, IMU acceleration bias, observer confidence
inputs: IMU acceleration/angular rate, left encoder, right encoder
optional derived feature: estimated body speed for slip regime selection
measurements: wheel-speed observation and IMU propagation
```

The estimator should:

- keep the physical wheel radius fixed at 0.0590 m;
- estimate and track IMU acceleration bias;
- use encoder angle differences with source timestamps and reset-safe
  re-baselining;
- adapt wheel measurement confidence using speed, acceleration, IMU/encoder
  disagreement, yaw rate, and frozen-encoder state;
- continue IMU propagation while an encoder freezes;
- model remaining slip distance during braking without moving the pose
  backward or resetting the odom origin;
- publish covariance that reflects the current observation quality;
- never consume ground truth, throttle, IPS, collision, AMCL, or map data.

#### Required fit outputs

- speed/acceleration residuals by speed, throttle, acceleration, and slip
  regime;
- wheel-speed-to-body-speed map and uncertainty;
- slip classifier/confidence calibration;
- frozen-encoder braking prior and uncertainty;
- position, yaw, and continuity residuals per reset epoch;
- a replayable parameter file with provenance and source-data hashes.

### Stage 3 — Independent odom verification

Run one withheld open-world validation suite with different ordering and
different throttle/target combinations from the fitting run. It must include
at least one moving/frozen-encoder event and one long straight segment.

Report errors in both absolute and relative forms:

- speed: absolute m/s plus percentage of actual/target speed; below a declared
  low-speed denominator threshold, report absolute error only;
- position: absolute metres plus error divided by cumulative ground-truth
  distance since the latest reset;
- yaw: degrees plus degrees per metre where meaningful;
- slip: wheel/body residuals by regime, not one overall average;
- continuity: maximum position/yaw/speed jump at encoder resets and dropouts;
- stop: time and distance after brake command until every stop condition agrees.

The acceptance thresholds are intentionally not fixed in this draft. They
must be chosen after inspecting the error distributions and must distinguish
transient, steady, braking, low-speed, and high-speed regimes.

### Stage 4 — Speed controller

Only after Stage 3 is accepted:

- Pure Pursuit publishes only a physical speed target on `/cmd/speed`;
- one actuator abstraction converts speed target to throttle;
- feed-forward comes from the identified equilibrium map;
- feedback is bounded and used for trim/recovery, not as a permanently
  oscillating replacement for the feed-forward plateau;
- speed-loop slew limits are independent from acceleration-loop limits;
- the controller uses only legal runtime state, primarily accepted `/odom`.

#### Speed-controller test definition

For each target:

1. reset and confirm spawn;
2. accelerate quickly using the controller;
3. enter a hold window only after the target is reached;
4. hold the throttle at the candidate equilibrium, allowing only a declared
   small trim band;
5. measure rise time, overshoot, steady-state speed error, throttle variance,
   acceleration, slip, and recovery from a small disturbance;
6. brake/coast until the full stop gate agrees;
7. only then reset for the next target.

The test must report target speed, actual ground-truth speed, odom speed,
throttle mean/std/range, hold duration, and the percentage errors. A target
with no stable plateau is a failed point, not an averaged success.

### Stage 5 — Acceleration controller

After the speed abstraction is stable:

- MPC publishes only physical acceleration on `/cmd/acceleration`;
- the actuator layer realizes positive acceleration using a speed-dependent
  inverse model and measured feedback;
- passive zero-throttle braking is modelled separately from controllable
  negative acceleration;
- achievable acceleration limits are identified by speed regime and enforced
  by MPC;
- acceleration rise time, bias, p95 error, saturation, and speed dependence
  are measured independently from the speed-controller test.

### Stage 6 — AMCL/EKF on the matching track scene

Do not start this stage on the open map. Use the competition track simulator
with the exact paired assets:

```text
map:     f1tenth_planning/maps/autodrive_compete_2026.yaml
raceline:f1tenth_planning/trajectories/icra_2025_raceline.csv
```

Before running, verify the loaded map and raceline identity, dimensions,
resolution, frame, bounds, and free-space compatibility.

The localization responsibility must follow the BachelorProject principle
without copying its simulator test setup:

- sensor odom is the continuous local IMU/encoder prediction;
- AMCL independently estimates map-frame pose and map-to-odom correction;
- AMCL must not be fed directly into the sensor-odom estimator;
- the EKF must not double-count the same AMCL correction as both a pose input
  and a separate transform;
- delayed/out-of-order corrections must be measured explicitly.

Required track tests:

1. correct-start AMCL convergence;
2. wrong-start AMCL convergence;
3. steady-state AMCL error and covariance calibration;
4. EKF/map pose error and correction latency;
5. temporary LiDAR degradation or localization recovery;
6. Pure Pursuit using the same map/raceline and accepted speed controller;
7. five clean laps with cross-track error, speed error, odom/AMCL/EKF error,
   collisions, resets, and native topic rates.

Ground truth is still diagnostics-only for scoring these experiments.

### Stage 7 — MPC and production cleanup

Only after odom, localization, and both controller abstractions have evidence:

- identify MPC `dt` from the actual odom event cadence or use an explicit
  validated predictor;
- identify longitudinal time constant and feasible acceleration envelope;
- identify steering sign, delay, rate, and effective response;
- identify or justify lateral velocity `v_y` and sideslip treatment;
- add numerical Jacobian, replay, cadence, and actuator-feasibility tests;
- tune MPC weights only after model parameters are identified;
- keep exactly one production controller and one actuator interface active;
- remove diagnostic reset/collision behavior from production paths;
- retain diagnostics in separate development launches.

## 4. Proposed acceptance report format

Every stage should end with one short report containing:

| Field | Required content |
|---|---|
| Run identity | date, revision, simulator scene, config, map/raceline if applicable |
| Purpose | fit, validation, controller, or track acceptance |
| Data | raw CSV path, row count, unique source-event count |
| Rates | native timestamp distributions for every relevant topic |
| Conditions | target, throttle, speed, steering, reset and stop rules |
| Errors | absolute, relative, p50, p95, max, and regime split |
| Safety | collisions, boundary guard, resets, stop confirmation |
| Decision | accepted, rejected, or needs another specifically defined run |
| Next action | one concrete follow-up, not an open-ended rerun |

## 5. Decisions to spar about before implementation

1. **Primary odom acceptance:** what speed percentage and per-distance
   position percentage are good enough to move to track work?
2. **Low-speed reporting:** should the relative-speed denominator be target
   speed, actual speed, or be omitted below a fixed threshold such as 1 m/s?
3. **Speed hold definition:** should “constant throttle” permit a narrow trim
   band, and if so what standard deviation/range is acceptable?
4. **Model complexity:** should the runtime model use a small piecewise table
   plus regime logic, or a compact observer with more online computation?
5. **Training/validation split:** should the next open-world suite be used
   only for fitting, followed by a separate validation run, or should the
   existing identification data be treated as the fit and the next run be
   validation only?
6. **Acceleration limits:** is passive braking sufficient for the first MPC
   track test, or must a controllable negative-acceleration channel be
   implemented first?
7. **Track gate:** should Pure Pursuit be tested immediately after odom and
   speed-control acceptance, with MPC deferred until the same localization
   metrics are proven?

## 6. Current recommendation

The safest next implementation target is not another full speed-controller
grid. It is:

1. agree the metrics and acceptance thresholds;
2. define the single open-world identification suite and its reset/hold
   semantics;
3. fit and independently validate the slip-aware IMU/encoder odom model;
4. rerun the speed controller with a measurable equilibrium-throttle hold;
5. only then move to the matching-map AMCL/EKF and raceline track tests.

Until those decisions are agreed, this file is a discussion record only.
