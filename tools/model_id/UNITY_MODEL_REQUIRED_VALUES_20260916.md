# Unity model-identification value contract

This is the value contract for reproducing the virtual F1TENTH car. It is
intentionally based on the Unity implementation, not on a real-vehicle tire
model. It does not change Unity, the simulator physics, or production MPC.

## Values already available from the Unity source

From `Assets/Prefabs/F1TENTH/F1TENTH.prefab` and
`Assets/Scripts/VehicleController.cs`:

| Group | Required values | Current source value |
|---|---|---|
| Body | mass | `3.47 kg` |
| Body | serialized COM | `(-0.00008, 0.06434, -0.00468) m` |
| Body | Rigidbody drag / angular drag | `0.273 / 0.1` |
| Wheel geometry | collider radius | `0.059 m` |
| Wheel geometry | collider mass | `0.109 kg` |
| Wheel geometry | WheelCollider center / local positions | center `(0,0.025,0)`; FR `(0.118,0.06,0.17)`, FL `(-0.118,0.06,0.17)`, RL `(-0.118,0.05,-0.16)`, RR `(0.118,0.05,-0.16)` m |
| Wheel geometry | front WheelCollider local rotation | FR `+5 deg`, FL `-5 deg` about local z; rear wheels `0 deg` |
| Suspension | spring / damper / target / travel | `500 / 100 / 0.5 / 0.05` |
| Suspension | wheel damping / force application point | `0.25 / 0` |
| Sideways law | extremum slip/value, asymptote slip/value, stiffness | `0.01 / 1.0 / 0.1 / 0.5 / 1.0` |
| Forward law | extremum slip/value, asymptote slip/value, stiffness | `0.15 / 0.9 / 0.25 / 0.58 / 0.8` |
| Steering | VehicleController wheelbase / track | `0.324 / 0.236 m` |
| Steering | steering limit / rate | `30 deg / 183.346 deg/s` (`0.523599 rad / 3.2 rad/s`) |
| Active competition controller | drive / brake / steer / mode / motor torque | `CAWD / CAWB / FrontWheelSteer / autonomous / 428 Nm` |
| Braking/input limits | throttle limit | `1.0` |
| Input semantics | applied steering sign | `AppliedSteering = -SteeringAngle` |

The active `VehicleController.Steer()` actuator is now represented as an
explicit offline contract named `unity_vehicle_controller`. It executes the
serialized `SteeringRate=3.2 rad/s` update at the captured
`fixedDeltaTime=0.001 s`, in the source's sign-inverted internal angle, and
retains the source comparison/clamp branch. It is not interchangeable with a
generic symmetric `rate_limited` update or a fitted first-order lag. The
transition state is the post-controller `AppliedSteering` value; `V1 Steering`
is the pre-update getter and remains diagnostic only.

The measured Unity forward-slip coordinate is
`(wheel_surface_speed - ground_speed) /
max(abs(wheel_surface_speed), abs(ground_speed), 0.5 m/s)`, clipped to
`[-1, 1]`. Here `wheel_surface_speed = RPM * 2*pi/60 * 0.059 m`. This is the
simulator's normalized coordinate, not a real-vehicle slip-ratio definition.

The exact diagnostic-dump replay profile uses `AppliedSteering` as its input.
Unity internally writes `SteeringAngle = -AppliedSteering` in its x-right
frame, while the replay model maps that frame to x-forward/y-left; the two
sign changes cancel, so the effective model Ackermann angle is positive for a
positive `AppliedSteering`. This conversion is explicit in the dump loader and
must not be confused with the raw Unity `WheelCollider.steerAngle` field.

The controller’s `WheelRadius=0.0325 m` is not the WheelCollider radius and
is not used by the CAWD torque branch. It must not replace the `0.059 m`
collider radius in the wheel model.

The prefab class default contains `MotorTorque=85.6 Nm`, but the competition
scene override is `428 Nm`; the runtime snapshot confirms the latter. The
active CAWD branch therefore applies `428 * throttle / 4 Nm` to each wheel,
while the zero-throttle CAWB branch applies `428 Nm` brake torque to each
wheel. The scene override, not the class default, is the required competition
value.

Unity project-level values also required for exact replay are fixed timestep
`0.001 s`, maximum allowed timestep `0.1 s`, gravity `(0,-9.81,0)`, solver
iterations `6`, solver velocity iterations `1`, contact offset `0.01 m`,
maximum depenetration velocity `10 m/s`, and maximum angular speed `7 rad/s`.

## Minimum value set for the offline vehicle model

The model must consume these values, grouped by authority:

| Authority | Required values | Status |
|---|---|---|
| Unity source/scene | mass, COM, rigid-body drag, angular drag, gravity, solver/timestep settings, collider geometry, wheel radius/mass, suspension, both serialized friction curves, Ackermann wheelbase/track, steering limit/rate, CAWD/CAWB/FrontWheelSteer branch, motor/brake torque, throttle limit, and steering sign | exact and already captured |
| Unity runtime state | body-y inertia each fixed step, wheel sprung mass each fixed step, wheel pose/compression, WheelHit force/contact geometry, forward/sideways slip, RPM, motor/brake torque, actual wheel steer angles, grounded flags, body pose and body-frame velocities, applied command/step sequence | required trace inputs; captured by the diagnostic build |
| Causal solver identification | wheel rotational state response, dynamic suspension/load response, and the mapping from each wheel's local slip/contact state to solver wrench | not a source constant; requires blind excitation/holdout data |
| MPC interface | state order, legal command inputs, 40 Hz command period, 30 commands, `0.75 s` horizon, and the `16 m/s` operating ceiling | fixed project contract |

The first two rows are inputs to an exact Unity replay model. The third row
is the only part that may be identified from data. It must not be represented
by a generic real-car cornering stiffness, friction coefficient, or tire peak
unless the corresponding mechanism is explicitly present in the Unity source.

## Compound collision values are part of the model boundary

The F1TENTH Rigidbody is not composed only of four WheelColliders. The
prefab has fourteen active child colliders on layer `0`, all non-trigger and
without an explicit physics material: two `Vehicle` BoxColliders, eight named
body/vehicle MeshColliders (`Chassis-1-solid1`, `Rear Shock Tower-1-solid1`,
`Front Shock Tower-1-solid1`, `Front Crash Member-1-solid1`, `Battery-1-solid1`,
`Platform Deck-1-solid1`, and the two Hokuyo meshes), and four active wheel
child MeshColliders named `Collider`. The active body MeshColliders are convex
with cooking options `30`; four additional imported wheel MeshColliders are
present but disabled. The two box values are:

| Collider | Size (m) | Center (m) |
|---|---|---|
| `Vehicle` | `(0.084305726, 0.0567199, 0.07943246)` | `(-0.034917377, 0.12554008, -0.15774795)` |
| `Vehicle` | `(0.08055779, 0.04038982, 0.1006026)` | `(-0.03494218, 0.11738583, -0.05888412)` |

In the open identification scene, the `Floor` is also an active layer-0,
non-trigger BoxCollider with no explicit physics material, size `(1,2,1)`,
transform position `(0,-0.01,0)`, and scale `(1000,0.01,1000)`; its top is at
approximately `y=0`. These colliders must be captured as an operating-envelope
condition, not replaced by a guessed tire friction coefficient.

The collision audit of the unchanged open scene recorded `10,918`
`Chassis-1-solid1`--`Floor` contact callbacks during the 70 s combined-slip
run, in the intervals `5.270--10.238 s` and `46.085--52.033 s`. It also
recorded `279,652` wheel-child `Collider`--`Floor` callbacks. However, every
chassis callback had positive separation (`0.0183--0.0200 m`) and exactly zero
`ContactPoint.impulse` and `Collision.impulse`; these are speculative/contact
callbacks, not evidence of a body force. The collision geometry remains an
exact replay input, but those rows must not be treated as a missing force
channel or used to justify changing a tire parameter. A no-chassis-contact
split is diagnostic only because it changes the operating-regime selection.

## Values that must be captured at runtime from the exact competition scene

These values are calculated by Unity or change during the run, so they cannot
be safely copied from the prefab or inferred from a generic car model:

1. The initial/static Rigidbody inertia tensor, inertia-tensor rotation, and
   the physical yaw inertia about the Unity body-y axis (`I_y`, mapped to
   model `I_z`).
2. Every fixed step’s runtime Rigidbody inertia tensor, inertia-tensor
   rotation, and body-y yaw inertia. The post-settling value is not guaranteed
   to equal the initial/static value.
3. Each wheel’s runtime `sprungMass` after the scene has settled, plus the
   per-fixed-step value if it changes during suspension/contact transitions.
4. Every fixed step’s wheel suspension pose from `WheelCollider.GetWorldPose`.
   This is needed to calculate compression and rebound using the configured
   spring/damper law.
5. Every fixed step’s `WheelHit.force`, contact point, contact normal,
   `forwardDir`, and `sidewaysDir`.
6. Every fixed step’s `forwardSlip` and `sidewaysSlip` for all four wheels.
7. Every fixed step’s wheel `rpm`, `motorTorque`, and `brakeTorque` for all
   four wheels.
8. Every fixed step’s actual `steerAngle` for all four wheels.
9. The body pose, world and body-frame velocity, world and body-frame angular
   velocity, and their timestamps.
10. The controller’s applied throttle, applied steering, applied steering in
    radians, command sequence, and physics-step sequence.
11. Grounded state for each wheel, including transitions to/from ungrounded.
12. The wheel-solver force response itself, or an equivalent causal
    reconstruction from body acceleration and wheel rotational balance. Unity
    2022.3 exposes `Rigidbody.GetAccumulatedForce/GetAccumulatedTorque`, but
    those calls do not include the internal WheelCollider solver in this
    vehicle; the exact audit returned zero for every component on all 70,028
    fixed steps. They are therefore an API-boundary check, not a required
    force measurement.

The post-settling runtime snapshot was added as a verification step. It now
also records the active `brakeType`, `ThrottleLimit`, and `DrivingMode`, so
the command-to-actuator branch is not inferred from the prefab default. For the
F1TENTH competition prefab it matches the serialized WheelCollider suspension
values (`500 N/m`, `100 N s/m`, `0.05 m`, force point `0`); the repository's
optional `Suspension.cs` is not attached to this prefab and must not be
mistaken for an active competition parameter.

The diagnostic source now also exports the dynamic `GetWorldPose` wheel pose
and rotation, per-fixed-step Rigidbody inertia, body-y yaw inertia, and
per-wheel sprung mass. This is a read-only observation and does not alter
physics. Old 110-column traces remain valid historical traces; new traces have
the added runtime body fields, sprung-mass field, and seven pose columns per
wheel. The exact competition-scene capture completed
on 2026-09-16. The initial/static snapshot measured
`yawInertiaBodyFrame = 0.0961907506 kg m^2`, agreeing with the manifest value
`0.0961908 kg m^2`; the old `0.0276976 kg m^2` value was the wrong
principal-axis projection and is rejected. After WheelCollider settling, the
open-scene runtime snapshots measured approximately `0.09564--0.09569 kg m^2`,
so the initial value must not be treated as a fixed runtime constant. The
required trace value is the per-fixed-step runtime body-y inertia. The same
settled snapshots measured runtime sprung masses approximately
`[0.817195, 0.816019, 0.918993, 0.917794] kg`, or static loads
`[8.01668, 8.00515, 9.01531, 9.00356] N`.
The combined exact-scene capture contains 12,000 rows at 1 kHz with no timing
gaps; the stationary capture is the settled-pose reference for deflection
analysis.

The accumulated-force boundary audit is stored in
`model_fits_v2/unity_exact_open_combined_slip_accumulated_force_v4_20260916/`.
It adds six read-only fields to the trace. All are zero because the active
`VehicleController` assigns WheelCollider inputs and the internal WheelCollider
solver applies the contact forces during simulation; no explicit
`Rigidbody.AddForce` channel exists in that controller. The active source audit
also found no attached optional `Suspension.cs`, anti-roll-bar, or down-force
component in the identification vehicle. The horizontal force model must
therefore identify the WheelCollider solver response from the causal state
already available in the trace; it must not treat the zero accumulator as a
missing physical force or multiply an assumed real-tire coefficient into the
model.

## Values that must not be silently fitted as substitutes

- No cornering stiffness, friction coefficient, or tire peak should be added
  to compensate for a mismatch when the Unity sideways curve is the active
  law.
- `WheelHit.force` is an exposed contact-force magnitude, not a public per-wheel
  decomposition of the internal forward/sideways solver force. The exact
  curve-shaped regressors remain diagnostic until the causal solver response
  is validated on blind holdouts. A fitted front/rear gain is not allowed to
  stand in for an unidentified load, wheel-state, or solver mechanism.
- No dynamic load-transfer coefficient should be promoted from the combined
  trace without a causal suspension-state model and blind holdouts. The exact
  scene has confirmed the static sprung masses, but dynamic `WheelHit.force`
  still varies with acceleration, steering, contact geometry, and wheel state.
- `I_z=0.0961907506 kg m^2` is the verified initial/static exact-scene value,
  not a guaranteed fixed runtime value. The runtime trace must supply the
  per-step body-y inertia. The old `0.0276976 kg m^2` value is retained only
  as rejected diagnostic provenance.

## Required excitation data

Use the exact competition scene and record at the full internal fixed-step
rate, while retaining the 40 Hz applied-command sequence:

- stationary suspension settling;
- straight throttle steps and ramps up to the project ceiling of `16 m/s`;
- zero-throttle/braking events, because zero throttle applies the configured
  all-wheel brake;
- left and right steering steps at raceline-relevant speeds, approximately
  `2, 4, 6, 8, 10, 12, 14, 16 m/s`;
- combined throttle-plus-steering runs with physically reachable raceline
  steering, not arbitrary high-speed/large-steering stress cases;
- repeated track/raceline runs after the open-scene mechanisms are identified.

The first identification target is the causal Unity suspension/load, runtime
inertia, and wheel rotational state. Only after those values are confirmed on blind holdouts
should the direct model be considered for MPC integration.

The first causal axle-force inversion is implemented in
`tools/model_id/identify_unity_axle_force_response.py`. It uses the recorded
wheel RPM/torque state to remove longitudinal force, then solves the remaining
body lateral-force/yaw equations for front and rear axle force. On two
independent captures of the same unchanged Unity schedule, the recovered
force-to-`WheelHit.force * UnityCurve(sidewaysSlip)` gains were `0.6489 / 0.6869`
(front/rear) with chronological holdout RMSE `0.179 / 0.195 N`, differing by
less than `1e-4` between repeats. These are stable Unity-solver observability
results, not tire coefficients; they must still be validated on a fresh
raceline-relevant schedule before being used by any recursive predictor.

That fresh schedule is stored in
`model_fits_v2/unity_exact_open_raceline_relevant_holdout_v1_20260916/`.
It reached `15.36 m/s` and kept applied steering within `+-3.6 deg`; the
selected turning rows cover `5.94--15.31 m/s` after lowering the offline
minimum-steering screen to `0.005 rad`, including high-speed, small-steering
raceline conditions. Its chronological holdout returned front/rear proxy
gains `0.7932/0.6781`, with force RMSE `0.234/0.183 N`.
Because these values differ from the previous schedule's `0.6489/0.6869`, a
single front/rear gain is rejected as a hidden parameter. The missing split
must be identified against wheel state, suspension/load state, steering
transition state, and speed before recursive model fitting.

## Wheel-state result requiring further validation

The bounded powered-drive repeat capture is stored in
`sdu_apex_autodrive/artifacts/model_id_work/model_fits_v2/unity_exact_open_powered_drive_repeat_20260916/`.
Its explicit torque-balance audit identifies a speed-dependent effective
wheel-state response: `I_eff` is approximately `0.000366 kg m^2` below
`6 m/s`, `0.000441 kg m^2` from `6--12 m/s`, and `0.000436 kg m^2` from
`12--16 m/s`. The damping term stays near `0.250 N m s`. The per-regime
holdout balance error is `0.11--0.14 N m`.

These are not copied source values and are not a universal physical wheel
inertia. They are explicitly identified effective coefficients of the
unchanged Unity WheelCollider response. The speed dependence is evidence that
one constant inertia would hide a simulator mechanism. The zero-throttle
`428 Nm` CAWB brake event is a separate hybrid lock transition and must not be
fit as a smooth wheel-inertia response. No coefficient is promoted to the
runtime plant until a separate raceline-relevant holdout and recursive parity
test accept it.

## Current mechanism-identification status

The compact cross-schedule screen is implemented in
`tools/model_id/screen_unity_axle_force_cross_schedule.py`. It validates
complete experiments against one another using only recorded Unity values;
it does not alter the simulator or create runtime parameters. The tested
proxy, speed, slip, serialized-curve-demand, and load-power bases do not
generalize between the previous combined-slip and raceline-relevant
experiments. Therefore no front/rear gain, load exponent, combined-slip
coefficient, or tire peak is a required model value yet.

The latest bounded speed-aware diagnostic run completed with normal graphics
in batchmode, captured `90,039` fixed steps over `90.038 s`, had no timing
gaps, reached `15.22 m/s`, and did not exceed the `16 m/s` ceiling. Its
lower-speed steering plateaus reached normalized sideways slip `0.998`; the
result is retained as a rejected excitation because it entered a spinning/
transient regime and did not identify a schedule-independent force mapping.
The next required dynamic value is consequently not another fitted gain: it
is a causal, observable WheelCollider solver/load state that predicts both
small-steering raceline data and a non-spinning lower-speed holdout.

## How the captured values are used

The captured Unity values are divided into three different uses:

1. **Direct plant inputs.** Mass, COM geometry, yaw inertia, rigid-body drag,
   wheel locations, collider radius, Ackermann geometry, steering rate, drive
   branch, brake branch, and the two serialized friction curves are consumed
   directly by the offline replay model. These values are not optimizer
   parameters.
2. **Causal-state identification.** Per-wheel RPM, torque, brake torque,
   measured slip, actual steering angle, `GetWorldPose`, sprung mass, inertia,
   `WheelHit.force`, contact geometry, grounded state, and the applied command
   sequence are used to identify four wheel rotational states and the
   suspension/contact state. A gain is not accepted when it changes between
   schedules without an identified state explaining the change.
3. **Odometry calibration reference.** Simulator pose, velocity, and yaw rate
   are used only offline to score estimates made from encoder, IMU, and
   steering inputs. They must never be passed into `/odom`, EKF, AMCL, or the
   production controller at runtime.

The exact-geometry mechanical CG load-transfer screen is stored in
`model_fits_v2/direct_unity_wheel_collider_benchmark_075_mechanical_cg_transfer_20260916.json`.
It uses only Unity mass, COM height, geometric contact wheelbase, and track
width. On the same blind direct benchmark it changed 0.75 s CORE p95 from
`0.043080` to `0.042580 m` cross-track, `0.052810` to `0.051882 rad` heading,
and `0.109821` to `0.108606 rad/s` yaw rate. This is a small, non-diagnostic
improvement, not evidence that the eight-state model has captured Unity's
suspension. It therefore remains an offline screen and the canonical direct
model continues to use the measured static sprung masses until a causal
suspension state passes a fresh blind holdout.

The currently missing model values are therefore state-transition values, not
static constants: per-wheel rotational response during raceline-relevant
combined drive/turning, suspension compression/deflection dynamics and
ground-contact transitions, and the WheelCollider solver's causal mapping
from each wheel's local slip/contact state to force and yaw moment. These are
the next identification targets. The target for odometry is separate: use
the exact `0.059 m` collider radius and measured steering geometry as physical
anchors, then retune only the encoder/IMU observer from legal sensor streams
against simulator truth in offline holdouts.

The v2 compact row exports now retain the runtime-state fields needed for that
identification: per-wheel angular speed and derivative, actual wheel steering
angle, sprung mass, and `GetWorldPose` vertical position. The cross-schedule
screen tested these fields as an explicit state basis. It did not generalize:
the front test error became very large in the old-high-steering-to-raceline
direction, while the reverse direction and rear axle also failed to beat a
stable universal basis. This is evidence that the raw fields need a causal
transition model and proper deflection/reference-state construction; they must
not be collapsed into a new gain.

The v4 compact exports additionally retain the reconstructed target body
force/moment and the wheel-torque longitudinal force/moment in separate
columns. They are stored in
`model_fits_v2/unity_exact_open_axle_force_response_cross_schedule_old_rows_v4_20260916.csv`
and
`model_fits_v2/unity_exact_open_axle_force_response_cross_schedule_raceline_rows_v4_20260916.csv`;
both are below the 100 MB repository limit. These channels make it possible
to test longitudinal contact response directly against the Unity curve and
torque balance, instead of using position drift to tune an opaque parameter.

### 2026-09-16 explicit four-wheel plant screen

The captured values are now consumed by a separate offline plant in
`tools/model_id/simulator_native_model.py` and scored by
`tools/model_id/benchmark_unity_native_wheel_model.py`. Its state is the
body pose/twist plus four wheel angular rates. It uses the dumped body-Y
inertia, per-wheel sprung masses, exact wheel radii, serialized forward and
sideways curves, Ackermann geometry, rigid-body drag, and the active
VehicleController CAWD/CAWB torque branch. The effective wheel rotational
inertia regimes from the exact trace remain explicitly labelled identified
solver-response values; they are not source `WheelCollider.mass` inertias.

The first causal replay found a brake-lock integration defect in the new
diagnostic model. That was fixed with a static-brake/no-sign-flip transition,
not by changing Unity or a tire value. On the combined straight/turn screen,
0.75 s position p95 changed from `1.433 m` to `0.644 m`; the remaining error
is still too large for promotion. The one-step and short-horizon behavior is
substantially better, but the long-horizon longitudinal and turning residuals
show that the Unity WheelCollider's internal contact/torque transition is
still not fully identified. The benchmark artifacts are
`model_fits_v2/unity_native_wheel_model_benchmark_initial_20260916.json` and
`model_fits_v2/unity_native_wheel_model_benchmark_brake_lock_20260916.json`.

This is the first point at which the captured values form an executable,
auditable model rather than only regression screens. It remains offline-only;
no production MPC, `/odom`, EKF, AMCL, simulator physics, or simulator
timing was changed. The next model-identification work is to split the
remaining longitudinal WheelCollider contact response from the hybrid
brake-lock transition and then identify causal suspension/load-transfer state
on a clean, non-spinning raceline holdout. Odometry tuning can use the same
runs only as offline truth-scored sensor replay and must remain a separate
acceptance step.

### 2026-09-16 native suspension travel audit

The first body-pose-only suspension screen was not a valid raceline model:
fixed Euler-angle extraction mixed accumulated yaw into pitch/roll. The audit
was corrected to use the actual `WheelCollider` contact geometry, equivalent
to the source `Transform.InverseTransformPoint(WheelHit.point)` calculation
with each wheel's local position, local rotation, radius, and suspension
distance. Rows with non-flat contact, braking, or non-upright body pose are
excluded from identification.

The corrected law is explicit and source-derived:

```
Fz_i = max(0, sprungMass_i * g
             - spring_i * suspensionDistance_i * travel_i
             - damper_i * suspensionDistance_i * travelRate_i)
```

It is implemented as `unity_wheel_travel_loads()` in
`tools/model_id/simulator_native_model.py`. `targetPosition` remains recorded
as a Unity configuration value; it is not treated as a free force gain. The
offline suspension plant uses the same law through its heave/pitch/roll
deviation state.

On the exact raceline-relevant holdout, the serialized law has correlation
`0.9987--0.9989`; chronological holdout load MAE is `0.082--0.090 N` and p95
is `0.117--0.156 N` across the four wheels. The chronological open-drive
holdout is noisier (`0.372--0.397 N` MAE, `0.476--0.513 N` p95), but the
identified normalized travel coefficients remain close to the source values:
approximately `-25 N/travel` and `-5 N s/travel`. These results identify the
native local travel/load mechanism; they do not introduce an arbitrary fitted
parameter.

The complete audits are
`model_fits_v2/unity_exact_open_suspension_pose_load_audit_20260916.json` and
`model_fits_v2/unity_exact_raceline_suspension_pose_load_audit_20260916.json`.
They are diagnostic only. The remaining model task is to initialize and
propagate the hidden heave/pitch/roll state causally from legal runtime
signals, then re-score the full 30-step/40 Hz prediction horizon. No value
has been migrated into MPC, `/odom`, EKF, or AMCL.

### 2026-09-16 reset-safe competition audit

The exact competition excitation traces preserve monotonically increasing
timestamps across experiment resets. A reset therefore cannot be detected by
an ordinary timing-gap test: it appears as a large root-position jump to the
stationary start pose. The suspension analyzer now segments such boundaries
and excludes a documented 10-row neighbourhood from derivative-based load
identification, while retaining all valid motion between resets.

The corrected competition excitation audit detected 10 boundaries and the
longer competition trace detected 76. After segmentation, the serialized
travel/load law reached `0.9886--0.9953` correlation and chronological holdout
MAE `0.048--0.062 N` on the 10-segment excitation; on the longer trace it
reached `0.9989--0.9991` correlation and holdout MAE `0.020--0.023 N`. The
previous impossible derivative spikes were reset artifacts, not evidence for
a new suspension or tire parameter. The corrected reports are
`model_fits_v2/unity_exact_competition_combined_excitation_suspension_pose_load_audit_20260916.json`
and
`model_fits_v2/unity_exact_competition_combined_full_suspension_pose_load_audit_20260916.json`.

### 2026-09-16 force-response inversion and data-to-model gate

The same reset-safe segmentation is now applied to
`tools/model_id/identify_unity_axle_force_response.py`. Body acceleration,
angular acceleration, wheel angular acceleration, and wheel-pose derivatives
are computed independently within each continuous trace segment. The open
combined-slip and raceline holdouts contained no detected reset boundaries;
their remaining force residuals are therefore not caused by the reset
artifact found in the competition traces.

The direct serialized sideways-curve proxy remains schedule-dependent. On
the open excitation holdout, the apparent front/rear proxy gains are
`0.678/0.692` with force MAE `0.356/0.125 N`. On the raceline-relevant
holdout they are `0.792/0.681` with MAE `0.132/0.137 N`. The compact
cross-schedule screen found no clean universal gain, speed/slip correction,
load-power law, or uninitialized wheel runtime-state regression. The latter
is numerically unstable across schedules and is rejected as a hidden
absorber, not promoted as a model term.

This is actionable model evidence: serialized wheel geometry, curve shape,
sprung-mass/suspension law, body mass, drag, and dumped inertia remain direct
source inputs, while the unresolved front/rear discrepancy must be split
through a causal WheelCollider contact/force transition or a better
source-derived slip/force reconstruction. The outputs are
`model_fits_v2/unity_exact_open_axle_force_response_cross_schedule_old_v5_20260916.json`,
`model_fits_v2/unity_exact_open_axle_force_response_cross_schedule_raceline_v5_20260916.json`,
and
`model_fits_v2/unity_exact_axle_force_cross_schedule_screen_v4_20260916.json`.
The selected rows remain below the 100 MB limit; both tables are stored as
pandas-compatible gzip CSVs: `...old_rows_v5_20260916.csv.gz` and
`...raceline_rows_v5_20260916.csv.gz`.

These results can now be used to build and score the next causal model and to
replay-test odometry, but they do not justify changing `/odom`, EKF, AMCL, or
production MPC yet. The promotion gate remains a blind 30-step/40 Hz causal
replay using only legal runtime state, followed by fresh live sensor-only
odometry validation.
