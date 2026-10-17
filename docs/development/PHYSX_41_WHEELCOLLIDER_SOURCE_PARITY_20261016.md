# PhysX 4.1 / Unity WheelCollider source-parity investigation — 2026-10-16

## Result in brief

There is enough source and matching local Unity-player evidence to reproduce
the configured WheelCollider vehicle path substantially more directly than a
data-fitted model. The source-grounded optimizer plant removes the wheel-spin,
tire-force, and suspension-load fit artifacts. A separate research-only
one-tick composition now joins the identified PhysX 4.1 / Unity equations for
four wheels on caller-supplied static hit planes.

This is not yet a complete Unity/PhysX simulator, nor a production odometry or
MPC implementation. A research harness now runs public PhysX 4.1.2 recursively
from one initial state and WheelCollider commands, without feeding it future
body or wheel measurements. The differentiable optimizer has also been
changed to use a source-derived composition of PhysX's wheel-speed integration
and finite brake torque. The OCP still does not run the full native contact
solver in its constraints, and its wheel reaction torque is averaged over
spatial intervals. Query selection, dynamic hit actors, sticky-tire and
suspension-limit constraints, active course contacts, and Unity's patched
native binary remain parity gaps. The long recursive result is strong
evidence for the source plant over its tested captures, not universal
accuracy or sensor-only odometry validation.

## Source and binary evidence

The workspace contains the Unity 2022.3.52f1 project and a pinned public
PhysX 4.1 source checkout at commit
`a2c0428acab643e60618c681b501e86f7fd558cc`. The deployed `UnityPlayer.so` has
a matching `UnityPlayer_s.debug` BuildID/debuglink. Local symbols and targeted
disassembly identify these native wrapper functions:

| Unity native symbol | Address |
|---|---:|
| `WheelCollider::ComputeWheelCentreOfMassOffset` | `0x2342480` |
| `WheelCollider::UpdateVehicleSpringCMOffsetsAndMasses` | `0x23420f0` |
| `WheelCollider::AddWheelToExistingVehicle` | `0x2342870` |
| `WheelCollider::AddWheelToRecreatedVehicle` | `0x2342ba0` |
| `VehicleComputeTireForce` | `0x2343130` |
| `VehiclesManager::VehicleConfig::UpdateVehicles(float)` | `0x2346430` |
| `ConfigureVehicleSubsteps` | `0x23446b0` |

The binary evidence shows the wrapper uses `PxVehicleNoDrive`, configures
`setSubStepCount(5.0, 5, 1)`, installs a custom tire-force callback, sets the
minimum longitudinal-slip denominator to 4.0, and in the manager update calls
`PxVehicleSuspensionRaycasts` followed by `PxVehicleUpdates`. The five-vs-one
substep setting is confirmed; Unity's overall fixed-step scheduling and every
integration/contact boundary have not been proven bit-for-bit from this call
trace.

The same `VehiclesManager::VehicleConfig::UpdateVehicles` disassembly shows it
allocates one tire type and one surface type and calls friction-pair `setup`
with one of each. PhysX 4.1 initializes every pair to `1.0f`; for this one-pair
table the tire-force callback receives a surface friction multiplier of 1.0.
This removes the unrecorded surface-multiplier uncertainty for the active
player path.

The source tree and player symbols answer much of the WheelCollider question,
but they do not expose Unity's complete patched native engine build metadata.
The public PhysX source provides the algorithms; the matching local binary
provides wrapper behavior and values. Differences in Unity patches, build
options, floating-point/compiler behavior, scene queries, and call phase must
still be checked before claiming binary-equivalent output.

## Configured vehicle values

These are read from the local project/prefab/scenes, not fit from captures:

| Component | Configured value |
|---|---|
| Unity project | 2022.3.52f1 |
| Fixed physics timestep | 0.001 s |
| Rigidbody mass / drag / angular drag | 3.47 kg / 0.273 / 0.1 |
| Runtime COM assigned in `Start` | (-0.00008, 0.06434, -0.00468) m |
| WheelCollider radius / mass | 0.059 m / 0.109 kg |
| Wheel damping | 0.25 |
| Suspension | 500 N/m spring, 100 N·s/m damper, 0.5 target, 0.05 m travel |
| Forward friction curve | extremum (0.15, 0.9), asymptote (0.25, 0.58), stiffness 0.8 |
| Sideways friction curve | extremum (0.01, 1.0), asymptote (0.1, 0.5), stiffness 1.0 |
| Vehicle geometry | wheelbase 0.324 m, track 0.236 m |
| Active controller | CAWD, CAWB, front steer; torque scalar 428; 30° limit; 183.346°/s steering rate |
| Physics settings | gravity -9.81 m/s²; solver 6 position / 1 velocity iteration; max angular speed 7 rad/s |

The four wheels share the serialized radius, suspension, and friction curves.
The active sim-racing scenes override `MotorTorque` from 85.6 to 428 N·m and
serialize the forward curve to stiffness/extremum values 0.8/0.9. The prefab's
`TireFriction` component is not attached in these scenes.

The WheelCollider/encoder radius is 0.059 m. A distinct controller field,
`WheelRadius = 0.0325`, is used only by its SkidSteer torque conversion; the
active CAWD path does not use it as a hidden radius gain. CAWD divides positive
drive torque across four wheels (107 N·m each at full command); at zero
throttle, CAWB assigns the configured 428 N·m brake torque to each wheel. The
controller's [-1,1] clamp permits negative autonomous throttle, which enters
the negative-motor-torque path for CAWD/CAWB; it is not the special IAWB/IRWB
brake branch.

## WheelCollider geometry, suspension, and load

The native wrapper's wheel-COM offset is the transformed collider center minus
the Rigidbody COM, with a target-position shift along the suspension direction:

`wheelCenterOffset += suspensionDirection * ((1 - targetPosition) * suspensionDistance)`.

It also configures suspension/force application points. The wrapper recomputes
four-wheel sprung masses with `PxVehicleComputeSprungMasses`, using the body
COM, wheel offsets, gravity direction, and Rigidbody mass. Reproducing that
four-wheel solve from scene geometry gives (FL, FR, RL, RR):
`(0.8171947001, 0.8160186332, 0.9189924292, 0.9177942374) kg`.

PhysX suspension rays use the chassis suspension-down direction, not each
wheel's camber axis. For a ray hit, the source computes jounce from rest travel
minus hit distance, and jounce speed from the change in jounce over the
vehicle substep. In the flat-ground, contact-present case, the source spring
and damper magnitude is based on `max(0, sprungMass * alpha + spring * jounce)`
plus `damper * jounceSpeed`, where `alpha=max(0, gravity·suspensionDirection)`.
PhysX then projects the suspension/gravity resultant onto the hit normal when
suspension-force projection is enabled. Ray hit selection, contact loss,
expansion-velocity limits, arbitrary terrain normals, and the exact native
substep jounce derivative remain outside the current flat-plane optimizer
translation.

The tire callback does not consume raw suspension force directly. PhysX
normalizes suspension force by gravity times `(wheelMass + sprungMass)` and
filters the normalized load. The default `PxVehicleTireLoadFilterData` points
are `(0, 0.2308, 3, 3)`; the Unity wrapper copies custom filter data on vehicle
recreation if present, otherwise the PhysX default applies. The local callback
uses the default because no custom scene value was found.

## Slip, tire force, and wheel rotation

For the non-tank `PxVehicleNoDrive` branch with nonzero motor torque or brake,
PhysX computes longitudinal slip as

`(radius * wheelOmega - wheelLongSpeed) / (max(abs(wheelLongSpeed), abs(radius * wheelOmega)) + 0.1 * lengthScale)`

and lateral slip as

`atan(wheelLateralSpeed / (abs(wheelLongSpeed) + 1 * lengthScale))`.

The configured 4.0 m/s minimum denominator is used in the *no applied
drive/brake torque* branch: its denominator is
`max(4.0 m/s, abs(wheelLongSpeed), abs(radius * wheelOmega))`. It is not used
in the active-torque branch above. PhysX sets the active-torque predicate from
nonzero applied drive/brake torque, so negative motor torque in the active
CAWD/CAWB branch also selects the active-torque slip formula.

The local Unity tire callback evaluates the configured two-piece cubic forward
and sideways friction curves, applies the combined-slip ellipse scaling, and
returns the source signs and force relationships (`Fx = filteredLoad *
longResponse * scale`, `Fy = -filteredLoad * latResponse * scale`). The wheel
reaction torque follows `-Fx * radius`. Local source inspection and binary
disassembly agree on the piecewise curve callback path.

The wrapper sets wheel moment of inertia to `0.5 * wheelMass * radius^2` and
wheel damping to 0.25. The public `PxVehicleNoDrive` update integrates wheel
speed with an implicit damping Euler step and locks a braked wheel to zero when
the step crosses through zero. The optimizer now uses the closed-form
composition of this 1 ms update for held torque, applies 428 N m brake torque
per wheel at exact-zero throttle, and models zero-crossing lock instead of
forcing wheel speed to zero at every braking node. Across each spatial
interval, tire reaction torque is trapezoidally averaged and a fractional-step
extension handles variable segment duration. Thus the damping/brake
integration is source-derived, but changing tire torque within an interval
remains an OCP discretization approximation.

## Code implementation and checks

### Rigidbody unconstrained-velocity source step

The Unity project fixes the physics timestep at 0.001 s, gravity at
`(0, -9.81, 0) m/s²`, Rigidbody linear/angular damping at 0.273/0.1 s⁻¹,
and the default maximum angular speed at 7 rad/s. PhysX 4.1 initializes a
dynamic body's maximum linear-velocity-squared to `1e32`; there is no
vehicle-specific serialized linear-speed cap in the active prefab. The
source operation in `DyBodyCoreIntegrator.h::bodyCoreComputeUnconstrainedVelocity`
is explicitly:

1. Add `gravity * dt * accelScale` to linear velocity unless gravity is
   disabled.
2. Multiply linear and angular velocity by `max(1 - damping * dt, 0)`.
3. Clamp each vector by its configured squared magnitude limit.

That float32 primitive is ported in
`tools/vehicle_dynamics_learning/physx_rigidbody_source.py`, with Unity's
configured 7 rad/s angular cap and PhysX's dynamic-body linear threshold as
defaults. It is intentionally a standalone component: it does not apply tire,
suspension, or contact impulses and does not integrate the body pose. The
solver can change velocities after this stage, so the 7 rad/s cap must not be
treated as a final-state yaw-rate bound or directly imposed on the optimizer
as though it ran after contact solving.

The same module now ports `PxVehicleUpdate.cpp::integrateBody`, the chassis
integrator inside each NoDrive substep: apply summed wheel force/torque to
linear/angular velocity, advance the temporary vehicle transform with updated
velocity, then update its quaternion using the vehicle SDK's first-order
normalized quaternion step. It separately ports the ordinary scene solver's
`Dy::integrateCore` **free-body/no-contact pose branch**, which uses the
post-solver velocity and a closed-form quaternion increment. These are
distinct operations; they must not be collapsed into one integrator.

In the upstream PhysX velocity-change update mode, Unity's manager performs
the suspension scene queries once, then NoDrive repeats suspension-plane
intersection/jounce evaluation, tire evaluation, wheel integration, force
accumulation, and `integrateBody` over its configured vehicle substeps. It
reuses the selected hit plane across those substeps; it does **not** issue a
fresh scene raycast at each 0.2 ms internal step. Afterward it writes the
internally updated chassis velocities to the actor. The scene solver
subsequently prepares unconstrained velocity, solves contacts, and performs
final pose integration.
The internal vehicle transform is used for subsequent substep contact geometry;
it is not a replacement for the scene solver's final pose. The active Unity
binary calls `PxInitVehicleSDK`, whose disassembly calls PhysX's
`setVehicleDefaults` routine; that routine writes `gApplyForces = false`. The
player contains no `PxVehicleSetUpdateMode` symbol/call site, so this path uses
the SDK's default velocity-change mode. The manager passes a null
concurrent-update buffer to `PxVehicleUpdates`, so the SDK applies the
post-update writes immediately. The new ports cover the integration equations,
not ray queries, contact impulses, or the full composed rollout.

The cached ray-plane calculation and suspension-force stage are ported in
`tools/vehicle_dynamics_learning/physx_vehicle_source.py`. For each internal
substep it uses the current temporary chassis pose and configured COM-relative
wheel-center offset to rebuild the suspension ray start, applies the source's
start-inside-plane rejection, intersects the ray with the cached plane,
computes `jounce = 2*radius + maxCompression - rayParameter`, and obtains
jounce speed from the change over the *internal* substep dt (zero on the first
uninitialized jounce). The upstream intersection code does not separately
reject a plane solely because its normal points with the ray; the port now
preserves that behavior. A parallel ray is returned as no intersection instead
of reproducing the source's undefined divide-by-zero. The spring/gravity
expression and default projection onto the hit normal are translated from
`processSuspTireWheels`. This replaces neither scene-query hit selection nor
dynamic-hit actor response; those remain separate stages.

The same module now contains float32 translations of the source contact-frame
construction (`computeTireDirs`), rigid wheel-bottom point velocity,
both NoDrive slip-denominator branches, the normalized tire-load filter, and
the Unity custom two-piece friction callback with combined-slip scaling and
wheel reaction torque. The contact frame follows PhysX's `(right × normal)`
then `(normal × longitudinal)` construction before wheel steer rotation. These
are source equations, not a model fitted to capture data.

`tools/vehicle_dynamics_learning/physx_no_drive_plant.py` composes these
operations into one 1 ms tick: select the 5/1 vehicle substep count from the
tick-start forward speed; reuse each wheel's supplied hit plane; calculate
jounce/load, contact-frame slip and Unity tire forces; integrate each wheel
and the temporary chassis through every vehicle substep; then run the separate
ordinary no-contact gravity/damping and pose step. It preserves the source
frame split: PhysX takes the COM translation but resets the temporary vehicle
chassis rotation to actor rotation, while the outer mass-frame pose uses the
full COM-local pose. Wheel-ray center and force-application offsets are
derived from the serialized F1TENTH wheel transforms, WheelCollider center,
target position, travel, and radius by
`unity_wheel_collider_offsets_com_body`.

The composition accepts `cached_hit_plane_world_xyzd` as an input; it does
not yet perform Unity/PhysX scene queries, choose among candidate hits, apply
include/exclude-layer filters, or reproduce query ordering. Dynamic-hit force
exchange, low-speed sticky-tire constraints, and ordinary solver contacts are
not part of this Python composition. The active vehicle registers no anti-roll
bars, and its `PxVehicleWheelsSimData` expansion-limit flag remains at the
PhysX default of zero (evidence and confidence are documented below), so those
two omitted branches are inactive for the inspected configuration. The
no-contact branch remains valid only when no scene contacts or locks are
active. At this stage the component suite had not yet compared the composition
against Unity captures; that check is recorded in the continuation below.

Existing r07/r11/r12g body-angular-velocity captures have maximum 3-D angular
speed norms of 3.483/2.055/2.410 rad/s respectively, below the 7 rad/s
pre-solver cap. This establishes that the cap was inactive in those captures;
it does not show that the cap is irrelevant for more aggressive candidate
motions. Fourteen focused tests cover gravity/damping order, gravity disable,
velocity clamps, both quaternion integrators, rotated inertia, damping floor,
and invalid inputs.

The cached-plane suspension port covers contact-plane projection during motion,
start-inside handling, first-substep jounce rate, spring/damper ordering, and
projected/unprojected force branches. The source contact/tire tests also cover
flat and tilted planes, point velocity, active/coast slip, friction-curve
landmarks, combined slip, and load filtering. Across the focused
source-mechanics suite (steering, wheel update, rigidbody integrators,
suspension/raycast, tire-force checks, composed no-drive step, and wheel-mesh
contact geometry), **101 tests pass**.

### Unity fixed-step ordering and moving wheel-mesh collision shapes

The matching Unity source and the native player resolve a previously missing
coupling that the static-plane Python plant does not include. In
`VehicleController.FixedUpdate`, the active order is:

1. Read controls, update Ackermann `WheelCollider.steerAngle`, and assign the
   current per-wheel motor/brake torques.
2. Call `UpdateWheelPoses` for all four wheels. Each call invokes
   `WheelCollider.GetWorldPose`, then writes the returned world position and
   rotation onto the wheel-transform GameObject.
3. Capture development telemetry; Unity then performs the physics step.

The exact-player `WheelCollider::GetWorldPose` wrapper (debug-symbol address
`0x2344440`) calls `PhysicsManager::AutoSyncTransforms`, then `GetLocalPose`,
then composes that pose with the wheel transform's world transform. The local
pose uses the current wheel query's suspension jounce and steering angle plus
the dynamic wheel rotation angle. `GetGroundHit` reads the per-wheel
`PxVehicleWheelQueryResult`; the 1 kHz development capture therefore records
the previous physics result at the next `FixedUpdate`, not a separate
post-solver tire-contact impulse measurement.

Those wheel transforms are not only render state in this prefab. The active
rigidbody has four active convex wheel-mesh colliders, and their transforms are
the four `Front/Rear * WheelTransform` references written by
`UpdateWheelPoses`. Consequently the ordinary rigid-body scene solver sees
moving wheel-mesh shapes in addition to the four WheelCollider ray/suspension
and tire-force paths. `WheelCollider.m_ProvidesContacts=0` only describes the
WheelCollider component; it does not disable these separate MeshColliders.
The source-hull clearance analysis shows the wheel meshes approach/cross the
open-plane surface, so their contact manifolds can contribute impulses and
affect chassis velocity/yaw. This is a concrete missing term in the previous
static-plane/no-contact composition, not a tire-curve fitting issue.

The model must therefore reproduce the actual order: update steer/torque;
derive wheel visual/collision poses from the *previous* wheel-query state;
sync transforms; perform one suspension-raycast batch; execute the cached-hit
vehicle substeps; then execute the ordinary scene contact solver with the
current synchronized compound shapes. A model that computes tire forces but
omits the last contact stage cannot be called 1:1.

The binary wrapper has now also been decompiled for
`GetGroundHit`, `GetLocalPose`, and `GetWorldPose`. `PxVehicleWheelsSimData`
allocation in the pinned upstream PhysX 4.1 source initializes flags to zero;
the Unity setup path configures no anti-roll bars and does not enable the
suspension-expansion-velocity flag. This rules those features out for the
active vehicle path unless a later runtime call changes them; the wrapper
search found no such call in the active scene/build path. The native raycast
prefilter rejects inactive/trigger colliders and delegates collision-ignore
and layer-override checks to Unity's PhysicsManager. Full native query hit
ordering and all Unity-patched solver details remain to be reproduced.

The active body and wheel convex MeshColliders serialize
`m_CookingOptions: 30`. The native contact harness reconstructs their source
convex-hull vertices and recooks them with upstream PhysX defaults plus vertex
welding; it does not yet prove that Unity's exact convex-cooking flags and
version-specific cooking path are identical. Unity documents these options as
affecting cooked query/contact structures; `UseFastMidphase` is primarily
relevant to triangle-mesh midphase, while these attached vehicle shapes are
convex. This is a bounded shape-cooking parity gap, not evidence that the
open-plane contact geometry is materially wrong. The audit JSON retains each
collider's serialized options and source hull for later comparison.

The full-step validation source capture
`live_runs/unity_mechanics_dynamic_validation_20261013_r12/unity_source_capture/`
contains 137,349 samples on a configured 1 ms grid, including body truth,
wheel RPM/steering/torque, WheelHit values, and WheelCollider world poses. It
is being used as a no-fit held-out mechanics check. These records make it
possible to isolate the single-tick wheel/force/contact discrepancy without
starting another Unity run; they do not alone reveal PhysX's internal contact
manifold or impulses.

Public PhysX 4.1.2 native libraries have also been compiled locally against
GCC 13 for a direct SDK execution harness. The only build adjustment is
relaxing upstream warning-as-error treatment for modern-compiler diagnostics;
the PhysX source/physics code is unchanged. A direct native test still needs
to include Unity's moving compound shapes and the verified update ordering
before its output can be used as an optimizer plant.

### Active controller steering and fixed-step boundary

The active scene's controller is `VehicleController` on `F1TENTH.prefab`,
with the scene override selecting autonomous mode and `MotorTorque=428`. Its
per-callback order is:

1. `VehicleController.FixedUpdate()` polls input, then calls `Steer()`.
2. `Steer()` updates internal center angle from latched normalized
   `AutonomousSteering`, then writes the two front Ackermann angles; the rear
   angles are zero for the active front-steer mode.
3. The same callback calls `Drive()` to set per-wheel motor/brake torques,
   updates wheel visual poses, and then the development capture records
   command, wheel angles, and torques.
4. Unity performs physics calculations immediately after `FixedUpdate`. The
   player-wrapper trace shows suspension raycasts followed by
   `PxVehicleUpdates`.

Unity's 2022.3 manual documents physics immediately after `FixedUpdate`, but
does not guarantee ordering among separate scripts' same event unless an
execution order is explicitly configured. So the order above is established
inside `VehicleController.FixedUpdate`; sensor-script phase is not inferred
from it. See the [Unity 2022.3 execution-order manual](https://docs.unity3d.com/2022.3/Documentation/Manual/ExecutionOrder.html).

Autonomous steering is not the symmetric physical-angle rate limiter currently
used by the IPOPT transcription. In Unity float arithmetic, for normalized
command `c`, previous internal angle `a` in degrees, fixed step `dt`, rate `R`,
and limit `L`, the source performs:

```text
if c > a/L:
    a = a - R*dt
    if a <= -c*L: a = -c*L
else:
    a = a + R*dt
    if a >= -c*L: a = -c*L
lastSteeringAngle = a
```

The sign comparison makes the response asymmetric; some changes snap to the
target. A float32 source port is in
`tools/vehicle_dynamics_learning/unity_vehicle_controller_source.py`, and
`unity_vehicle_mechanics.step_steering()` now delegates to it. The no-fit
capture replay predicts Ackermann wheel angles using the previous captured
steering state and current row's command, at the recorded 1 ms grid:

| Existing capture | Rows | Valid 1 ms steps | Wheel-angle RMSE | p99 absolute | Max absolute |
|---|---:|---:|---:|---:|---:|
| source parity r07 | 289,450 | 289,449 | 6.20e-7° | 1.95e-6° | 1.15e-5° |
| exact replay r11 | 297,415 | 297,413 | 4.27e-7° | 1.95e-6° | 5.74e-6° |
| dynamic r12g | 300,000 | 299,999 | 3.50e-7° | 1.91e-6° | 1.18e-5° |

There is one incomplete row in r11; it is excluded from consecutive-step
checks. These results verify the C# actuator recurrence and capture row phase
to float precision for the tested captures. They do **not** establish that an
optimizer-generated steering path is reachable: the OCP still optimizes
physical `steering_rad` and symmetric bounded `qdelta`, rather than normalized
command passed through this exact recurrence at 1 kHz. That is now a concrete
optimizer model mismatch to remove before treating its steering prediction as
Unity-equivalent.

The PhysX 4.1 `integrateNoDriveWheelSpeeds` float32 equation and Unity's
observed 5/1 substep selection are now separately ported in
`tools/vehicle_dynamics_learning/physx_no_drive_source.py`. The replay tool
`validate_physx_no_drive_wheel_step.py` uses recorded current-step wheel slip
and suspension force to evaluate the source tire callback, then applies the
source wheel integrator to predict the next captured wheel RPM. No coefficients
are fitted. On all three captures, the **at-or-above-5 m/s / one-substep** band
has low error:

| Capture | Wheel transitions | Wheel angular-speed RMSE | p95 absolute | Maximum absolute |
|---|---:|---:|---:|---:|
| source parity r07 | 660,100 | 0.00999 rad/s | 0.01237 rad/s | 0.8911 rad/s |
| exact replay r11 | 660,412 | 0.01011 rad/s | 0.01209 rad/s | 0.8992 rad/s |
| dynamic r12g | 789,024 | 0.00680 rad/s | 0.00747 rad/s | 0.7934 rad/s |

The below-5 m/s band (five substeps) reports RMSE 0.0100–0.0212 rad/s, but
that replay holds captured slip and load constant across those internal
substeps; it is diagnostic rather than a complete re-simulation. The
one-substep validation also uses captured slip/load, so it verifies the
source tire-force-to-wheel-speed step when those current-step quantities are
known; it is **not** a free rollout and cannot establish open-loop wheel-speed
prediction accuracy.
Largest residuals cluster in isolated transients (up to 0.79–0.90 rad/s),
while the p95 stays below 0.013 rad/s. They must be classified before claiming
full native parity.

`Socket` assigns received throttle and steering directly to the autonomous
properties (clamped to [-1, 1]); those fields remain latched until changed.
At exactly zero throttle, `Drive()` assigns 428 N m brake torque to every
wheel. Any nonzero throttle clears the brake and applies one quarter of
`428 * throttle` as motor torque to each wheel. Zero is therefore a discrete
brake-mode switch, not a smooth zero-torque point.

`f1tenth_planning/autodrive_mintime/unity_wheel_model.py` now derives its wheel
inertia, sprung masses, suspension setup, tire curves, slip, and tire-load
filter from the local Unity configuration and PhysX source behavior. The model
builder no longer reads the three fitted wheel-spin, tire-force, and
suspension-load artifacts. The suspension translation is currently limited to
the flat-plane/contact-present form described above. This is an optimizer
plant change only: it has not been integrated into runtime odometry or MPC.

Focused tests cover the source wheel inertia and sprung masses, absence of fit
inputs, curve interpolation/signs/combined slip, applied-torque slip, body-down
suspension ray, zero-jounce static support, load filtering, and inverse slip
calculation. At the time of this report they pass: **15 tests passed** in
`tools/racing/test_unity_wheel_model_native_tire_force.py`.

### Existing-capture check (not a fit)

The new flat-plane suspension translation was compared with `WheelHit.force`
from three already-existing 1 kHz Unity captures, with no parameter search or
coefficient adjustment. Samples were restricted to grounded wheel hits,
reconstructed suspension speed magnitude below 0.05 m/s, and suspension-down
alignment cosine above 0.9. Suspension travel and its derivative came from
body pose at the recorded 1 ms interval; this is not the native 0.2 ms
substep's exact jounce derivative below 5 m/s.

On the two parity captures (`r07` and exact replay `r11`), per-wheel RMSE was
0.5305–0.5424 N and mean bias +0.3472–+0.3587 N. On the dynamic capture
(`r12g`), rear-wheel RMSE was 0.5170–0.5221 N and front-wheel RMSE
0.7371–0.7531 N. Large isolated maxima (up to about 104 N in the dynamic
capture) remain; this low-jounce selection does not explain those transients.
The comparison supports the static/slow-jounce mapping, but not exact dynamic
load parity. Timing, telemetry phase, internal substeps, wheel-hit state, and
projection conditions are candidate explanations still to discriminate; none
was tuned away.

### No-fit 1 ms replay and newly identified scene-contact path

The research-only composition was replayed on 4,991 evenly spaced transitions
from the existing 289,450-row r07 open-plane capture. Each transition starts
from that row's observed rigid-body and wheel states, uses the current row's
steering/torque, and predicts one 1 ms step. It does not recurse from an
earlier prediction. Static ground is supplied as the known y=0 plane; the next
capture row is scoring truth only. WheelHit slip and force, future state, and
fitted coefficients are not predictor inputs. The output is saved at
`live_runs/unity_exact_binary_audit_20261012/physx_no_drive_source_one_step_r07_20261016.json`.

| Tick-start forward-speed band | Body linear-velocity RMSE | Body angular-velocity RMSE | Maximum angular error | Position RMSE | Yaw-angle RMSE |
|---|---:|---:|---:|---:|---:|
| < 5 m/s (2,145 samples) | 0.0000556 m/s | 0.000446 rad/s | 0.00805 rad/s | 0.000000694 m | 0.000000288 rad |
| >= 5 m/s (2,846 samples) | 0.000178 m/s | 0.001681 rad/s | 0.06659 rad/s | 0.00000191 m | 0.00000143 rad |

The small aggregate one-step errors are encouraging for the source equations,
but the 0.0666 rad/s maximum is a material isolated miss and these numbers do
not measure a recursive rollout, production odometry, or full Unity parity.
Wheel-speed errors are also not uniformly negligible: the sampled low-speed
window around 3.6–3.7 m/s reaches 0.200 rad/s on one wheel, while the high-speed
window around the angular outlier has wheel errors below 0.021 rad/s. These
residuals remain to be explained, not fitted away.

The outlier aligns with a previously unmodeled active collision shape in the
Unity prefab. Each wheel visual has an enabled, non-trigger, convex
`MeshCollider` using Unity built-in Cylinder mesh fileID 10206, at scale
`(0.11, 0.022, 0.11)` and rotated 90 degrees around local Z. This gives a
nominal 0.055 m radius and 0.022 m half-height, separate from the 0.059 m
WheelCollider raycast tire. The child shapes are attached to the same dynamic
vehicle rigidbody. `VehicleController.FixedUpdate()` updates the wheel visual
transforms from `WheelCollider.GetWorldPose()` before Unity's physics step;
`Physics.autoSyncTransforms` is enabled. Thus the ordinary scene solver can
apply additional contact impulses and friction through these compound wheel
shapes in addition to the WheelCollider force path. The current Python plant
does not solve these shape contacts.

The installed Unity 2022.3.52f1 editor contains the exact built-in `Cylinder`
resource referenced by fileID 10206. It was extracted from
`Editor/Data/Resources/unity default resources`: 88 source vertices and 240
indices (80 triangles), forming a 20-sided cylinder mesh. The source vertices,
editor version, and resource SHA-256 are retained in
`live_runs/unity_exact_binary_audit_20261012/unity_builtin_cylinder_mesh_2022.3.52f1.json`.
`tools/vehicle_dynamics_learning/analyze_unity_wheel_mesh_clearance.py` now
computes support of those mesh vertices after the serialized child scale and
pose transforms; this equals the convex hull's support in each direction.
Across all 289,450 r07 rows, every wheel shape is within the project's default
combined contact-offset band (10 mm per collider, 20 mm per pair) of the open
plane. The extracted source-mesh hull has negative clearance on the
front-right in 0.1406% of frames (minimum -0.1271 mm); the other wheels remain
geometrically separated from the plane in this capture. This is source-mesh
geometry, not proof of the post-cooking contact shape or a solver impulse.
The per-wheel audit is saved at
`live_runs/unity_exact_binary_audit_20261012/physx_wheel_mesh_contact_clearance_r07_20261016.json`.

At the known one-step angular-residual row 181366 (10.447 m/s), the extracted
source-mesh clearances are FL +1.102 mm, FR -0.053 mm, RL +1.542 mm, and
RR +1.575 mm; they are included in the JSON inspection section. This supports
the ordinary rigid-body contact path as a plausible contributor: the shapes
are within the configured contact-offset band, and front-right source-mesh
vertices slightly cross the plane. It is not proof that a solver impulse
occurred at that exact tick: the capture records
WheelCollider `WheelHit` data but not rigidbody `OnCollision` contact
manifolds or impulses. The support calculation deliberately does not fabricate
a contact force or claim a causal correction. Unity documents
contact generation when collider separation is below the sum of the two
contact offsets ([`Physics.defaultContactOffset`](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Physics-defaultContactOffset.html)); the serialized 10 mm setting is therefore relevant even when the exact signed clearance is positive.

### Exact player compound-shape and global-solver audit — 2026-10-16

The audit now reads the compiled player's `level0`, `globalgamemanagers`, and
shared mesh assets rather than relying only on the public prefab. Build
settings and the r07 capture both identify Unity `2022.3.52f1`; the checked-out
upstream source reports PhysX `4.1.2.29882248`. Unity announced the 4.1.2
update in its 2022.1 line, but that does not prove the shipped engine has no
private patches relative to public source.

The compiled scene has 14 active, non-trigger colliders parented to the car's
3.47 kg Rigidbody: 12 convex MeshColliders (four 20-sided Cylinder wheel
shapes and eight component meshes) plus two BoxColliders. Four legacy CAD
wheel MeshColliders are disabled and excluded. The active floor is a
BoxCollider whose top is exactly world `y=0`; car and floor are on layer 0,
and layer 0 collides with itself. Details and source-hull vertices are in
[`vehicle_compound_shapes_clearance_r07_20261016.json`](../../live_runs/unity_exact_binary_audit_20261012/vehicle_compound_shapes_clearance_r07_20261016.json),
reproducible with
[`audit_unity_player_compound_shapes.py`](../../tools/vehicle_dynamics_learning/audit_unity_player_compound_shapes.py).

Exact serialized global settings are: 1 ms fixed step, 100 ms maximum allowed
step, gravity `(0,-9.81,0)`, automatic simulation and transform sync, 10 mm
default contact offset, 6 position / 1 velocity solver iterations, 7 rad/s
maximum angular speed, 10 m/s maximum depenetration velocity, solver type 0
(Projected Gauss-Seidel), and friction type 0 (Patch). Neither car shapes nor
floor has an explicit PhysicMaterial, and PhysicsManager has no explicit
default material. The 2022.3 Physics manual identifies PGS as the default
solver and explains pair contact generation from the sum of contact offsets
([Physics Manager](https://docs.unity3d.com/2022.3/Documentation/Manual/class-PhysicsManager.html)).

Across all 289,450 r07 rows, every wheel-mesh convex hull is within the 20 mm
pair-offset band. Only the front-right source hull crosses the plane (0.1406%
of frames; minimum -0.1271 mm). No fixed component hull or box penetrates the
plane. The chassis mesh comes closest at 17.684 mm and is inside the 20 mm
band for 23.20% of rows. Its support vertex is moving toward the plane on
17.33% of rows, generally at very low normal speed. This is a potential
contact constraint, not measured contact force.

At the known one-step yaw outlier (row 181366, 181.366 s), chassis clearance
is 18.249 mm and its support-normal velocity is -0.00126 m/s; the front-right
wheel source-hull clearance is -0.053 mm. The audit cannot establish that
Unity generated a cooked-hull manifold or applied an impulse. Serialized
`Mesh` assets contain source vertices but no baked convex-cooking data, so the
recovered geometry is not claimed to be bit-identical to the runtime-cooked
shape.

## What can and cannot be reverse-engineered

### Active F1TENTH configuration, not generic project scripts

The instantiated `Assets/Prefabs/F1TENTH/F1TENTH.prefab` selects
`VehicleController` with `driveType=CAWD`, `brakeType=CAWB`, and front-wheel
Ackermann steering. Its active root Rigidbody is 3.47 kg, gravity-enabled,
drag 0.273, angular drag 0.1, continuous-dynamic collision mode, with runtime
COM set by `VehicleController.Start()` to `(-0.00008, 0.06434, -0.00468) m`.
The prefab serializes the WheelCollider radius as 0.059 m, while the
controller's separate `WheelRadius` field is 0.0325 m; with this CAWD
configuration that field is only used by the skid-steer branch, which is not
selected. The wheel visual MeshCollider has nominal 0.055 m radius. These are
three distinct values and must not be substituted for each other.

The checked-in prefab default is `MotorTorque=85.6 N m`, but the active
sim-racing scene and exact-player configuration override it to 428 N m. Thus
the active CAWD path is `DriveTorque = throttle * 428 N m`, split to 107 N m
per wheel; nonzero torque clears brake torque. At exactly zero torque, CAWB
sets 428 N m brake torque on every wheel. Negative throttle enters the CAWD
negative-motor-torque path, not the IAWB/IRWB brake branches. The active r07
capture confirms the 428 N m scene configuration and per-wheel friction curves.
Front steering uses a 183.346 deg/s rate limit and a 30 deg normalized limit,
followed by the source Ackermann formula using 324 mm wheelbase and 236 mm
track width. Each WheelCollider is configured with 0.05 m travel, 0.5 target
position, 500 N/m spring, 100 N s/m damper, 0.109 kg wheel mass, and 0.25 wheel
damping; the effective per-wheel forward/lateral curves are in the capture
metadata and the active Unity prefab/scene.

The generic `Suspension`, `TireFriction`, and `AntiRollBar` scripts are present
in the Unity project but are **not attached** to this F1TENTH prefab. Their
per-tick spring rewriting, material/map stiffness scaling, and anti-roll
forces are therefore not active in this vehicle. The attached `IMU` has
`GravityCompensation=1`; it only computes sensor output and does not apply a
force to the Rigidbody. Its code sets the gravity term to zero in that mode,
so reported finite-difference acceleration retains gravity instead of
subtracting it. The prefab's own WheelCollider fields and
`VehicleController` are the active suspension/friction/actuation
configuration to reproduce.

The project PhysicsManager uses gravity -9.81 m/s², 1 ms fixed timestep,
automatic simulation and transform synchronization, 6 position / 1 velocity
solver iterations, 10 mm default contact offset, Patch friction, and no
explicit PhysicMaterial on the wheel child shapes. Unity's default PhysicMaterial
uses 0.6 static/dynamic friction, zero bounciness, and Average combine for
ordinary collider contacts. This is separate from WheelCollider tire friction:
the active VehiclesManager configures a single surface/tire pair, whose PhysX
default is 1.0. Rigidbody compound colliders remain part of the ordinary
contact path; WheelCollider raycast tire forces are not the only possible
vehicle forces.

The WheelCollider vehicle path can be reconstructed much more faithfully than
the former fitted model: wrapper setup and values are visible in the matching
player binary; public PhysX 4.1 provides vehicle update, suspension, tire, and
NoDrive algorithms; the Unity project supplies scene/prefab/controller values.
That is a viable basis for a source-directed standalone implementation.

It is not yet an exact replica of the complete Unity simulation step. A full
copy also needs the exact Unity-patched PhysX build/revision, query filtering
and hit ordering, rigid-body force application timing, chassis integration,
and the scene's ground/wall contact behavior and solver configuration. Unity
uses a general rigid-body/contact solver in addition to the WheelCollider's
raycast suspension and tire model; reproducing only the wheel callback does not
reproduce chassis-vs-ground/wall contacts. The local Unity project does not
contain that entire native patched engine implementation. The public upstream
PhysX solver can be built and investigated, while local binary disassembly can
constrain Unity's wrapper behavior, but matching them is still an engineering
and parity-validation task—not something established by the current equations.

## 2026-10-16 continuation: callback parity, capture phase, and native scene replay

The planned compound-contact replay now exists in
`tools/vehicle_dynamics_learning/physx_native_vehicle_step.cpp`. It attaches
the 14 recovered chassis/wheel compound shapes, uses the 1 ms scene step and
6/1 solver iterations, runs the public PhysX vehicle update before the scene
step, and filters wheel rays to the open-plane ground. This is a native-source
experiment; the convex meshes are recooked from source vertices, not Unity's
runtime cooked blobs.

Re-reading `VehicleComputeTireForce` exposed a source-parity mismatch. The
decompiled callback normalizes combined longitudinal/lateral slip using the
raw spline responses, then applies each curve's stiffness to the final force.
The Python and native ports had applied stiffness before that normalization.
The active forward stiffness is 0.8, so this affected the current vehicle.
Both ports now apply stiffness after combined-slip scaling, and the Python
plant passes the surface-pair friction factor explicitly. Focused regression
tests pass (33 across the WheelCollider source equations and NoDrive plant).

On the same 30 consecutive 1 ms samples at capture rows 185370–185399, the
corrected callback reduced next-sample wheel-speed absolute-error RMSE from
0.00461 to 0.00149 rad/s (pooled across four wheels); pooled p95 fell from
0.00766 to 0.000286 rad/s. Angular-velocity RMSE excluding the one
frame-suspect sample fell from 0.00117 to 0.000806 rad/s. Including that one
sample leaves a 0.2267 rad/s maximum and 0.0414 rad/s RMSE; it is not counted
as a model gain. On this short window, predicted position RMSE is 0.000016 m
and maximum attitude error is 0.000752 rad. These are one-window results,
not full-run or recursive-rollout validation; the angular target still comes
from the legacy capture frame described below.

A second 30 ms segment was run through the same native vehicle + compound
contact harness, at r12g rows 237639–237668. It is a high-speed/high-steering
case (mean body speed 7.55 m/s; maximum front-wheel steering magnitude
0.631 rad) and has four active scene contact pairs per step. Relative to the
following captured sample, pooled wheel-speed RMSE is 0.00382 rad/s (p95
0.000258, maximum 0.03074); body linear-velocity vector RMSE is
0.0000075 m/s; body angular-velocity vector RMSE is 0.00153 rad/s (maximum
0.00837); and position RMSE is 0.0000020 m (maximum 0.0000066 m). These are
truth-initialized one-step comparisons with the captured body and wheel state
reset at every millisecond—not a recursive rollout. They also depend on
reconstructing legacy local telemetry in the Rigidbody frame, so the small
numbers are conditional and are not a clean independent measurement of
Unity-vs-PhysX pose/twist closure.

The earlier isolated mismatch at row 185380 is not explained by the compound
contact addition: the corrected native replay still differs from the
legacy-capture angular target by 0.22667 rad/s, while the ordinary scene solve
changes angular velocity by only 0.0001266 rad/s. It reports five contact
pairs but zero axis solver constraints. Because the old capture omitted the
Transform pose and world twist needed to resolve its local-frame ambiguity,
this row cannot distinguish a telemetry-frame reconstruction error from an
unmodeled native update. It remains open and is not treated as evidence that
the PhysX contact solver itself is wrong.

The second window's outputs were unchanged to reported float precision between
the pre-correction and corrected native callback binaries. Thus the stiffness
ordering correction is material in the first tested window but has no
measurable effect in this high-steering segment; it must not be presented as a
universal accuracy gain.

The same player disassembly establishes update ordering: the vehicle manager
performs suspension raycasts and `PxVehicleUpdates` before `PhysicsScene::Simulate`;
it first synchronizes transforms. `VehicleController.FixedUpdate` sets
steering/torque and captures telemetry before that native physics step. The
capture therefore has a pre-step command/state phase, but legacy vectors are
not all in the same frame: pose comes from `Rigidbody.position/rotation`, while
velocity and several wheel/hit local vectors were converted using `Transform`.
The prefab enables Rigidbody interpolation. Thus the old 1 ms kinematic
closure report is a conditional diagnostic—not a clean measure of PhysX
pose/twist error.

All seven >0.1 m pose jumps in the retained r12g ring buffer match seven
`/autodrive/reset_command` rising edges in the ROS bag. A single clock offset
of 0.60933 s aligns them with residuals from -8.8 to +11.3 ms. They are reset
teleports, not vehicle trajectories; the two earlier reset edges precede the
retained window. The current capture's `capture_unix_time_ns` is constant
across rows: its source used 10 ns per .NET tick instead of 100, and the
realtime subtraction/addition cancels the per-row change. The development-only
capture source now fixes that conversion, labels schema v2, logs world and
Rigidbody-local twist plus Transform pose/local twist, and stores wheel-pose/
WheelHit vectors in the Rigidbody pose frame. No physics behavior is changed.
No Unity Editor executable was found in this workspace, so this new capture
schema is statically reviewed but not runtime-compiled or tested.

## Recursive plant check and optimizer connection (2026-10-16 continuation)

The native harness now has a `recursive_command_only` mode. It seeds body pose,
body velocity, angular velocity, four wheel speeds/angles, and suspension
jounce once. For each subsequent 1 ms tick it receives only the captured
per-wheel steering, motor torque, and brake torque at the WheelCollider
boundary; wheel visual collider poses recurse from its own jounce and wheel
rotation. It does not read future body pose, body twist, wheel RPM, slip, or
contact telemetry. The evaluator reads capture truth only after rollout.

On an uninterrupted 38.0 s dynamic capture (mean speed 7.15 m/s, max physical
front steering 0.631 rad, 223.3 m travel, throttle 0–0.34), the public PhysX
4.1.2 plant achieved:

| Metric | Recursive error |
| --- | ---: |
| Planar position RMSE / p95 / max | 0.0272 / 0.0528 / 0.0536 m |
| Yaw RMSE / absolute max | 0.000292 / 0.000452 rad |
| Speed-magnitude RMSE / max | 4.90e-6 / 1.85e-4 m/s |
| Wheel angular-speed RMSE / p95 | 2.40e-4 / 5.06e-4 rad/s |

At 5 s, planar position error was 1.13 mm; at 10 s, 3.80 mm; at 20 s,
18.2 mm; at 30 s, 36.8 mm. This is unusually close source-parity evidence
for a long recursive interval. Position drift tracks the small accumulated
yaw error. It is not yet a deployment odometry result: the initial body state
is simulator truth, per-wheel WheelCollider torques/steering are provided as
inputs, the capture's legacy twist lacks explicit frame metadata, and the
native public PhysX build recooks reconstructed hulls rather than loading
Unity's exact runtime cooked convex data. Input and output JSON were gzip
compressed after scoring (24 MB to 0.7 MB, 188 MB to 28 MB); the CSV capture
and metrics remain intact.

This evidence led to one actual differentiable-optimizer revision. In
`unity_wheel_model.py` the wheel update is now the closed-form composition of
PhysX 4.1 `integrateNoDriveWheelSpeeds` implicit-Euler steps for held torque,
including direction-dependent finite brake torque and zero-crossing lock.
The OCP applies source CAWD quarter motor torque at nonzero throttle and
source CAWB 428 N m per-wheel braking at exact zero; a brake/drive mode edge is
split into two half-intervals instead of setting wheel RPM instantly to zero.
Tire reaction torque is trapezoidally averaged across each spatial interval.
The fractional-step duration extension is smooth for IPOPT; it matches the
source recurrence exactly for integer 1 ms counts under constant torque, but
the spatial interval's held/averaged tire torque is still an approximation.

Verification so far: 61 focused source/optimizer tests pass. The OCP's
`--diagnose-initial` check reports a maximum normalized defect of 0.794 on the
old captured seed (including 0.396 on one wheel update), so the seed is not
feasible under the revised dynamics. The first bounded IPOPT run (120
feasibility iterations, trust region 25) reduced the maximum defect to 0.0530
but did not produce a feasible incumbent; its largest remaining violations
were lateral force balance (0.0530) and the captured-motion trust region
(0.0491). A second targeted run with a 600-iteration budget and a larger trust
region is being used to distinguish iteration/seed restrictions from a real
inconsistency in the current OCP equations. Do not call the optimizer revision
validated until it returns a feasible incumbent and its candidate is checked
by the native recursive plant. No production odometry source or runtime
physics was changed in this continuation, and no Unity simulator was launched.

## Remaining work, in order

1. Complete the current bounded OCP feasibility/time run. If it fails, inspect
   the first dynamic defects and adjust only the source-derived integration
   or initialization, then re-run the same bounded case.
2. For a feasible candidate, evaluate its one-lap command profile with the
   recursive native source plant from the candidate's initial state. Score
   path, speed, yaw, wheel speed, and wall clearance; only then consider it as
   an optimizer improvement.
3. Build an odometry experiment that starts from sensor-estimated state and
   uses only the deployable encoder, IMU, steering, and throttle feedback
   channels. Replay independent existing whole-run bags through the exact
   production C++ observer and compare pose/speed/yaw with ground truth.
   The current long native rollout cannot substitute for this because it was
   initialized with ground truth and consumes per-wheel torque outputs.
4. Obtain schema-v2 Unity telemetry when the matching Unity Editor is
   available, then finish query/hit selection, surface/contact, and cooked
   convex parity. Keep the proven reference trajectory and runtime physics
   unchanged until an improved optimizer candidate passes the established
   simulator validation.

## 2026-10-16 continuation: odometry reference frame and 40 Hz IMU check

### Corrected twist comparison

The first exact-production replay CSV compared the observer's rear-axle
`body_v_mps` directly with the Unity bridge's raw `odom.twist.linear.y`. That
was a frame mismatch: the bridge twist is at the Rigidbody COM, while the
production observer's `base_link` velocity is at the rear axle. The scorer now
performs the Unity-geometry conversion

`v_rear = v_com - yaw_rate * 0.15532 m`

before comparing lateral velocity. On the existing 75.73 s r02 bag (3,029
aligned samples), this changes the current-source replay's lateral-velocity
RMSE from the invalid direct-frame value `0.24524 m/s` to `0.00827 m/s` (p95
`0.01661 m/s`, max `0.02930 m/s`). The recorded production output scores
`0.00946 m/s` after the same conversion. The replay is not a promotion test:
the bag's recorded configuration hash differs from today's configuration.
Planar position remains a separate comparison: current-source replay RMSE is
`0.20758 m`, versus `0.79399 m` for the historical recorded output, so the
observer replay still needs a configuration-matched run before attributing
that difference to code changes.

This corrects earlier statements in this investigation that treated raw
simulator twist `linear.y` as rear-axle velocity. The development truth topic
is still only an offline label and is not consumed by production.

### Does 40 Hz sampling alone explain the lateral-acceleration failure?

`tools/vehicle_dynamics_learning/analyze_unity_imu_lateral_sampling.py` now
reconstructs the documented Unity `IMU.cs` finite-difference acceleration
from the moving r12g 1 kHz capture, applies the COM-to-rear-axle transform, and
tests all 25 phases of a nominal 25 ms grid. A prior invocation accidentally
used the stationary r12 capture; its output was removed and its apparent
sampling conclusion is withdrawn.

The correct r12g capture contains 300,000 samples over 300 s, with forward
speed from approximately 0 to `11.20 m/s`, front-wheel steering to
`0.631 rad`, and yaw rate to `1.91 rad/s`. Seven pose discontinuities split
it into eight continuous segments. Results use truth lateral velocity only
for each diagnostic window's initial state and scoring; it is never fed back
after initialization. They are not production-observer results, and this
capture is not treated as a sealed holdout.

| Rear lateral-velocity predictor | Median phase RMSE | Worst phase RMSE | Notes |
|---|---:|---:|---|
| Instantaneous IMU sample integrated at 25 ms | `0.110 m/s` | `1.334 m/s` | Left/right/trapezoid variants are nearly identical; worst phase is a specific dynamic segment |
| Average of the 1 kHz IMU acceleration over the completed 25 ms interval, endpoint `u/r` | `0.0289 m/s` | `0.133 m/s` | Requires a source-side interval aggregate not present in the existing 40 Hz packet |
| 1 kHz equation-integral oracle | `0.00822 m/s` | `0.0227 m/s` | Diagnostic lower bound; it uses the hidden within-interval `r*u` sequence |
| Current configured kinematic lateral base, using captured truth `u/r` | `0.00914 m/s` | `0.0122 m/s` | Sensor-only algebraic comparator, but not an end-to-end observer test |

Across all 300,000 samples, that kinematic base has `0.00890 m/s` RMSE,
`0.01960 m/s` p95, and `0.0661 m/s` maximum absolute error. Joint speed/physical
steering counts and errors are retained in
`live_runs/unity_mechanics_dynamic_validation_20261013_r12g/analysis/imu_lateral_sampling_20261016.json`.
This test supports keeping the production kinematic lateral path rather than
enabling raw 40 Hz acceleration integration. The interval-average result is a
plausible future experiment, not an implemented change: it would require
source-side 1 kHz aggregation and a 40 Hz observer update designed for an
interval-average signal. No IMU topic or Unity physics behavior was changed.

### Full 3D body-frame term check

The Unity-to-ROS frame mapping gives roll rate `p = -omega_unity.z` and
vertical body speed `w = velocity_unity.y`. Rigid-body kinematics therefore
add `p*w` to the lateral COM acceleration equation; converting that to the
rear-axle reference also retains the `-yaw_acceleration * COM_to_rear` term.
The analyzer evaluates this identity on both moving 1 kHz captures. It
materially reduces the long-run 1 kHz integration discrepancy, but does **not**
justify replacing the production lateral estimate:

| Capture / method | Planar 1 kHz integration RMSE | Full-3D 1 kHz RMSE | Current kinematic base RMSE |
|---|---:|---:|---:|
| Practice whole run, 74.85 s | `0.12491 m/s` | `0.03458 m/s` | `0.01031 m/s` |
| Dynamic capture, all continuous segments | `0.01176 m/s` | `0.00770 m/s` | `0.00890 m/s` |

On practice, median 25 ms phase RMSE for point-sampled acceleration is
`0.2698 m/s` planar and `0.3909 m/s` with `p*w`; completed-interval IMU
averaging is `0.08310` and `0.11694 m/s`, respectively. On the dynamic
capture, full-3D point sampling improves median phase RMSE from `0.1102` to
`0.08427 m/s`, while interval averaging is effectively unchanged (`0.02887`
versus `0.02892 m/s`). The best source-equation integration still uses
unavailable within-interval 1 kHz terms. Thus `p*w` explains part of the
high-rate integration discrepancy, particularly in practice, but 40 Hz
acceleration integration remains inferior to the existing kinematic estimate
on the practice run. This is evidence for the equation, not a production
observer change. Detailed phase/regime results are retained in the two JSON
reports under the corresponding capture `analysis` directories.

The existing r02 bag's IMU, truth, and encoder header stamps are identical
within each source packet. Receive-order deltas are mostly sub-millisecond,
but receive gaps range from about `3.1 ms` to `47.7 ms`; this is delivery
cadence jitter/bursting, not evidence of a fixed two-packet source delay. A
cross-correlation shift is not an authorized correction. The exact raw timing
summary is kept at
`live_runs/odom_source_revalidation_20261012_r02/analysis/production_odom_exact_replay_20261016/sensor_phase_audit.json`.

### Optimizer substep correction and current solve

PhysX calls `computeNumberOfSubsteps` once at the start of each outer vehicle
tick, selecting five `0.2 ms` substeps below `5 m/s` and one `1 ms` substep at
or above the threshold based on the chassis's longitudinal velocity at that
tick start. The spatial OCP had used the collocation-interval midpoint speed;
it now selects from the interval-start speed, the nearest state represented
by that transcription. A spatial interval that crosses `5 m/s` still cannot
reproduce per-millisecond branch changes exactly.

Focused checks now pass: `131 passed` across the PhysX source, vehicle plant,
optimizer, controller-command exporter, and production-odometry shim suites
(`0.56 s`; `git diff --check` clean). The 120-iteration r05 OCP
attempt did not produce a feasible candidate; its maximum final constraint
violation was `0.6677`, dominated by lateral force balance. A 600-iteration
attempt with the corrected start-speed branch ended at
`Maximum_Iterations_Exceeded` with no feasible incumbent. Maximum constraint
violation fell from `0.7941` at the initial guess to `0.1393`; the largest
remaining residuals are lateral force balance at nodes 140 and 139 (`0.1393`
and `0.0616`). A diagnostic repetition with stage reporting confirms that
only the `captured-motion feasibility` stage ran; minimum-time optimization
was never entered. Its 600-iteration stage ended at primal infeasibility
`0.1235`, with lateral force balance still the largest named residual. The
capture-derived periodic seed itself closes poorly: its
start/end differences include `-0.200 m` lateral offset, `+0.511 m/s` forward
speed, and `-0.440 rad/s` yaw rate. This solve does not demonstrate a valid
optimizer result. The failure-report writer now preserves per-stage IPOPT
status and residuals for subsequent diagnosis. The targeted 1,800-iteration
continuation did not improve that basin: final maximum residual was `0.1408`
and primal infeasibility `0.1249`, still dominated by lateral force balance at
nodes 140/139. This rules out a simple 600-iteration budget shortage for the
raw captured seed; time minimization was again never entered. No raceline from
either raw-seed attempt is valid until it has a feasible OCP incumbent and is
replayed through the native recursive source plant.

### Legacy candidate slip-coordinate handoff

The prior `r21_scaled_force_balance` state CSV contains no wheel angular-speed
columns; its `slipx_*` fields must be inverted using the convention declared in
its own optimizer report. That report says the legacy guide coordinate was
`Sx=(R*omega-vx)/max(vx,0.5)`. The current warm-start loader had instead used
the PhysX applied-torque expression. A high guide-slip sample then inverted
through `(1-Sx)` to a spurious negative wheel speed, producing an initial
`forward_wheel_rotation` violation of `1172.79 rad/s`. The loader now selects
the inverse from explicit model metadata and no longer forces every zero-
throttle node to zero wheel speed; a physically locked legacy node reconstructs
zero from its recorded slip. Six focused inverse checks pass.

Re-running the r21 seed diagnostic with the correct inverse removes that
spurious negative-speed defect. The remaining maximum initial constraint
residual is `3.7925`, at longitudinal force balance; residuals also remain in
the new finite source wheel update (about `1.77` at two nodes) and lateral
force balance (about `2.26`). Thus the previous r21 path is useful as an
initializer but is not compatible/feasible under the source-grounded plant as
saved. A slower prior r26 candidate gives a lower initial maximum residual
(`2.0415`, longitudinal force balance) under the same metadata-aware wheel
reconstruction, and it is already a periodic free-gate path. The first
source-model feasibility solve from r26 reduced maximum residual to `0.06367`
after 2,000 iterations, with the largest remaining defects at lateral force
balance nodes 76 (`0.06367 m/s^2`), 248 (`0.05576 m/s^2`), and 124
(`0.05116 m/s^2`). It still had no feasible incumbent, and no minimum-time
stage ran. The lateral residual is physically small (about `0.22 N` at the
3.47 kg chassis), but it remains above the strict optimizer acceptance limit;
the iterate is not promoted.

The first checkpoint attempt exposed a handoff bug: although IPOPT finished at
the same `0.06367` maximum residual, `solve_bottom_up` discarded every
unsuccessful iterate above the normal `0.005` warm-start threshold before the
diagnostic writer could see it. The return gate now separately retains a
finite failed iterate only when its maximum residual is at most `0.1`; the
caller marks and serializes it as `diagnostic_iterate_only`, while retaining
the stricter `0.005` threshold for ordinary warm starts. The first reload
diagnostic found that wheel and per-wheel load seeds were being reset/rebuilt;
the initializer now preserves those solver variables when resuming a
model-consistent state. Rechecking r14 reproduces its original maximum
residual exactly (`0.06367476046`) with the same worst constraints. A
2,000-iteration continuation (r17) reduced the maximum defect to
`0.0044271 m/s^2` (about `0.0154 N` at the chassis mass) and reached a
`5.21241 s` predicted time, but IPOPT ended with
`Maximum_Iterations_Exceeded`; it has no feasible incumbent and is retained
only as a near-feasible warm start under the `0.005` restart threshold. The
dominant remaining residual is lateral force balance at node 247. A further
bounded continuation from r17 (r18) is running to seek a converged feasible
seed before any minimum-time candidate is considered. This state can only
seed a bounded solver continuation, never be emitted as a raceline. Neither
the old r21 nor r26 fitted-model trajectory is a source-model candidate.

### Recursive practice-lap replay from the 40 Hz command boundary

The r119 practice capture contains one completed `5.75 s` lap and 5,750
contiguous 1 ms plant samples. A stricter replay was added: sample the
captured `AutonomousSteering` and `AutonomousThrottle` only at each 25-tick
boundary, then reconstruct the 1 kHz wheel inputs with the public Unity
`VehicleController.Steer()`/Ackermann/CAWD/CAWB source equations. The 230 held
command intervals reproduced the captured wheel outputs to a maximum error of
`4.54e-6 deg` steering, `3.06e-6 N m` drive torque, and exactly zero brake
torque error. This establishes that the command expansion matches the
captured 40 Hz controller boundary for this run.

The recursive public PhysX 4.1.2 native plant then used one captured initial
body/wheel state plus only those 40 Hz-held commands for the whole lap. It did
not read future pose, velocity, wheel RPM, contact, or per-wheel actuation
from the capture. Against the sealed capture labels it achieved:

| Whole-lap metric | Result |
|---|---:|
| Duration / mean speed | `5.750 s` / `5.027 m/s` |
| Planar position RMSE / p95 / max | `0.00126 / 0.00230 / 0.00232 m` |
| Yaw RMSE / absolute max | `0.000151 / 0.000220 rad` |
| Speed-magnitude RMSE / absolute max | `0.0000169 / 0.0000779 m/s` |
| Wheel angular-speed RMSE | `0.000333 rad/s` |
| 5 s planar position error | `0.00229 m` |

This is strong evidence that the source-derived plant can serve as a useful
offline simulator for the measured practice-lap regime, including its actual
40 Hz controller output and internal 1 kHz steering ramp. It is not yet a
sensor-only odometry result: initialization uses simulator truth (including
wheel state/pose), and the native public PhysX build recooks source geometry
rather than sharing Unity's private patched binaries. It also does not prove
that a newly optimized path will attain the same accuracy; that requires
feeding its planned commands into this plant and checking the new path. The
captured r119 lap itself reports zero collisions.

An independent second closed-course window was then extracted from the older
r07 practice-scene capture. Its 5.95 s segment travels 28.05 m, returns within
2.4 cm of its starting point, and ends within 0.0067 rad of its starting yaw.
Its mean speed is 4.719 m/s, speed range 2.966–7.777 m/s, maximum physical
steering 18.48 degrees, and throttle range 0–0.327. The 238 held 40 Hz command
intervals reproduce measured per-wheel outputs within `2.99e-6 deg` steering,
`2.40e-6 N m` drive torque, and zero brake-torque error. From one captured
initial body/wheel state and only those commands, the recursive plant obtains:

| Whole-lap metric | r07 result |
|---|---:|
| Planar position RMSE / p95 / max | `0.000383 / 0.000740 / 0.000745 m` |
| Yaw RMSE / absolute max | `0.0000427 / 0.0000782 rad` |
| Speed-magnitude RMSE / absolute max | `0.00000916 / 0.0000355 m/s` |
| Wheel angular-speed RMSE | `0.000275 rad/s` |
| 5 s planar position error | `0.000651 m` |

Together, r07 and r119 are two separate collision-free Unity capture windows
with closed lap geometry. This materially improves confidence in the
source-derived plant over these measured, moderate-steering practice regimes.
It still does not establish accuracy at the high-steering edge, on a newly
optimized trajectory, or from a sensor-derived initial state.

An additional r117 13.175 s recursive segment is retained only as a long
motion diagnostic, not counted as a lap: its endpoints are 33.27 m apart and
its heading changes by 0.916 rad, so its capture-side `raceline_lap_completed`
flag does not establish geometric lap closure. It reached 4.06 mm maximum
position error, but is not used to claim another practice-lap validation.

No Unity simulator was launched in this continuation, and no runtime physics
or production odometry algorithm was changed. The replay evaluator was fixed
to compare equivalent velocity reference points; the existing C++ observer
remains unchanged pending sensor-only validation. The source plant is now
validated recursively over two separate closed practice-capture laps from
40 Hz command boundaries, and is the intended high-fidelity dynamics
validator for subsequent raceline-optimizer candidates. The remaining Odom
gap is initial-state and measurement correction using only permitted
encoders, IMU, and actuator feedback; simulator truth remains scoring-only.

## Primary references

- [Unity 2022.3 Wheel colliders](https://docs.unity3d.com/2022.3/Documentation/Manual/wheel-colliders.html)
- [Unity 2022.3 primitive objects](https://docs.unity3d.com/2022.3/Documentation/Manual/PrimitiveObjects.html) (built-in primitive geometry; exact installed-editor vertices are retained in the audit JSON)
- [Unity 2022.3 `Physics.defaultContactOffset`](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/Physics-defaultContactOffset.html)
- [Unity 2022.3 Physic Material](https://docs.unity3d.com/2022.3/Documentation/Manual/class-PhysicMaterial.html)
- [Unity 2022.3 Physics Manager](https://docs.unity3d.com/2022.3/Documentation/Manual/class-PhysicsManager.html)
- [Unity 2022.3 `MeshColliderCookingOptions`](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/MeshColliderCookingOptions.html)
- [Unity 2022.3 `MeshColliderCookingOptions`](https://docs.unity3d.com/2022.3/Documentation/ScriptReference/MeshColliderCookingOptions.html)
    - [Unity C# WheelCollider bindings](https://github.com/Unity-Technologies/UnityCsReference/blob/master/Modules/Vehicles/Vehicles.bindings.cs)
- [PhysX 4.1 `PxVehicleUpdate.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleUpdate.cpp)
- [PhysX 4.1 `PxVehicleNoDrive.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleNoDrive.cpp)
- [PhysX 4.1 `DyBodyCoreIntegrator.h`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/lowleveldynamics/src/DyBodyCoreIntegrator.h)
- [PhysX 4.1 `DyDynamics.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/lowleveldynamics/src/DyDynamics.cpp)
- [PhysX 4.1 Vehicles SDK guide](https://nvidiagameworks.github.io/PhysX/4.1/documentation/physxguide/Manual/Vehicles.html)
