# PhysX 4.1 / Unity WheelCollider source-parity investigation — 2026-10-16

## Result in brief

There is enough source and matching local Unity-player evidence to reproduce
the configured WheelCollider vehicle path substantially more directly than a
data-fitted model. The source-grounded optimizer plant now removes the wheel-
spin, tire-force, and suspension-load fit artifacts and implements the
identified PhysX 4.1 / Unity equations described below.

This is not yet a complete Unity/PhysX simulator, nor a production odometry or
MPC implementation. In particular, the model does not reproduce Unity's whole
1 kHz rigid-body/contact solve, dynamic raycast scene, or the exact internal
vehicle substep sequence through its full state rollout. The optimizer still
uses a continuous wheel-state transition between its knots rather than
integrating `PxVehicleNoDrive` at every native substep. No claim of exact
long-horizon prediction or production accuracy follows from source parity of
these components alone.

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
the step crosses through zero. The current optimizer wheel state has not yet
been changed to execute that exact discrete equation at every 1 ms / vehicle
substep; it still approximates wheel-state transition continuously between
optimizer knots. This is a known source-parity gap, not a fitted parameter.

## Code implementation and checks

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
prediction accuracy. The capture schema does not include the native drivable
surface friction-pair multiplier; the current check assumes multiplier 1.
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

## What can and cannot be reverse-engineered

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

## Next source-directed step

First classify the isolated wheel-step residuals and recover the surface
friction-pair multiplier from the active Unity player wrapper/scene; neither
may be tuned. Next, make a source-driven 1 kHz replay recompute slip and tire
force from its own predicted body/wheel state at every PhysX substep, rather
than using future logged slip/load. Then port the wheel state into the
optimizer's discrete transition and represent normalized steering commands
through the verified C# recurrence. Compare wheel/body motion on held-out
captures without fitting. Remaining major work is the full 3-D rigidbody,
raycast/contact selection and Unity scene collision solve. These are still
research-model tasks; no production odometry/MPC change is claimed. No
simulator was launched for this report; no raceline was accepted or changed.

## Primary references

- [Unity 2022.3 Wheel colliders](https://docs.unity3d.com/2022.3/Documentation/Manual/wheel-colliders.html)
- [Unity C# WheelCollider bindings](https://github.com/Unity-Technologies/UnityCsReference/blob/master/Modules/Vehicles/Vehicles.bindings.cs)
- [PhysX 4.1 `PxVehicleUpdate.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleUpdate.cpp)
- [PhysX 4.1 `PxVehicleNoDrive.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleNoDrive.cpp)
- [PhysX 4.1 Vehicles SDK guide](https://nvidiagameworks.github.io/PhysX/4.1/documentation/physxguide/Manual/Vehicles.html)
