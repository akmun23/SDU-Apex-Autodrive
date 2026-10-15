# Unity WheelCollider / PhysX source audit — 2026-10-12

## Purpose

This audit identifies which source files can support a mechanics-based 1 kHz
offline plant and 25 ms odometry observer, and which parts of the simulator
remain unavailable. It is not a claim that public PhysX code is the exact
native solver in the competition player.

## Files present in the workspace

The downloaded public AutoDRIVE Unity project is
`external/autodrive-unity-source`, commit
`75b7df5216524a3c6b3150d698139190c6c44f8f` (`AutoDRIVE-Simulator`). The files
most directly relevant to vehicle motion are:

- `Assets/Scripts/VehicleController.cs`: common-actuator torque distribution,
  throttle/brake behavior, steering-rate limit and Ackermann wheel angles.
- `Assets/Scripts/WheelEncoder.cs`: samples `WheelCollider.rpm`, accumulates
  revolutions at `Time.deltaTime`, quantizes ticks, and derives cumulative
  angle. The socket exports angle/ticks, not independent wheel-speed truth.
- `Assets/Scripts/IMU.cs`: rotates rigid-body velocity differences into the
  body frame and publishes the selected linear/angular quantities.
- `Assets/Scripts/Socket.cs`: serializes steering/throttle feedback, encoder
  angles, and IMU values into telemetry.
- `Assets/Scripts/OpenPlaneWheelDynamicsCapture.cs`: development-only,
  read-only 1 kHz telemetry ring buffer for body motion and all four
  WheelColliders; enabled only by `SDU_APEX_DYNAMICS_DIR`.
- `Assets/Prefabs/F1TENTH/F1TENTH.prefab`: the public Explore vehicle
  configuration and component connections.
- `ProjectSettings/TimeManager.asset`: `Fixed Timestep: 0.001` s.

The public prefab defaults to `driveType: 5` (`CAWD`, common-actuator AWD),
front-wheel Ackermann steering, `MotorTorque: 85.6`, `DrivingMode: 0`,
`SteeringRate: 183.346` deg/s, `Wheelbase: 0.324` m, `TrackWidth: 0.236` m,
controller `WheelRadius: 0.0325` m, and `WheelCollider.radius: 0.059` m. The
active `RoboRacer - Sim Racing - Explore` scene overrides `MotorTorque` to
428 and `DrivingMode` to autonomous (1), matching the exact pinned player's
serialized controller. Thus the prefab default is not the effective Explore
runtime value. The scene also overrides one wheel's longitudinal friction
stiffness/extremum value to the same values already present in the prefab.
The two radii have different roles: the controller value is only used by its
differential-drive path; the normal CAWD `Drive()` path distributes
`DriveTorque` directly. The colliders use 0.059 m for contact and encoder
surface-speed calculations. The public Explore scene is still not the pinned
player's exact track scene, so motion equivalence must be checked separately.

The source computes feedback in the same `FixedUpdate` that updates steering
and drive. It assigns `currentS/currentT` before calling `Steer()/Drive()`, so
the exported feedback can reflect the preceding physics tick (1 ms). Steering
feedback is in physical radians in this source; throttle feedback is
normalized. This is consistent with the workspace bag analysis, where
`/steering` is already radians and command steering is normalized and scaled.

The prefab wiring resolves the encoder ambiguity: the left and right
`WheelEncoder` components each observe one rear collider (rear-left and
rear-right respectively), not an axle average. Both serialize `PPR: 16` and
`GearRatio: 120`, i.e. 1,920 encoder ticks per wheel revolution. `FixedUpdate`
integrates `WheelCollider.rpm` at the 1 ms step, truncates cumulative ticks to
an integer, and publishes cumulative angle. Thus the two ROS encoder streams
are quantized rear-wheel angular-position observations; the front-wheel RPMs
remain latent in the public telemetry.

The configured actuation path is also sharply nonlinear at zero throttle.
`CAWD` distributes requested drive torque equally over four colliders. In the
active Explore scene, full input requests 428/4 = 107 N·m per wheel. At exactly
zero drive torque, the `CAWB` branch instead applies 428 N·m brake torque to
all four wheels.
For nonzero throttle, the drive path clears brake torque and applies the
commanded motor torque; this branch does not convert negative throttle to a
separate brake command for `CAWB`. The configured steering rate is
183.346 degrees/s with a 30-degree limit. This is consistent with the pinned
controller, but does not prove whole-player physics equivalence.

## Public Unity code boundary

Unity's `UnityCsReference/Modules/Vehicles/Vehicles.bindings.cs` declares
`WheelCollider`, `WheelFrictionCurve`, `WheelHit`, and their native bindings.
It documents slip-based tire friction, wheel RPM, motor/brake torque, and
suspension properties, but methods are `extern` and the class points to the
native header `Modules/Vehicles/WheelCollider.h`. The native implementation is
not present in this reference repository. The 2022.3 WheelCollider manual
describes the wheel as a suspension-direction raycast and documents separate
forward and sideways friction curves.

The AutoDRIVE 2026 guide is more specific about its documented model: separate
longitudinal/lateral tire-slip response, per-wheel loads and suspension, and
two-piece cubic friction splines with the published extremum/asymptote points.
That guide is the appropriate public specification for a Unity-style
surrogate, but it does not expose all native solver details or the precise
cubic spline tangents.

## PhysX source files found online

NVIDIA publishes PhysX 4.1 vehicle source. The most useful files are:

- `physx/source/physxvehicle/src/PxVehicleUpdate.cpp`: wheel slip,
  suspension/tire update ordering, default combined-slip tire-force shader,
  and force/moment application.
- `physx/source/physxvehicle/src/PxVehicleSuspWheelTire4.cpp`: wheel,
  suspension, tire and sprung-load state setup.
- `physx/source/physxvehicle/src/PxVehicleTireFriction.cpp`: tire/surface
  friction-pair mapping.
- `physx/source/physxvehicle/src/PxVehicleWheels.cpp`: wheel simulation and
  dynamic data.
- `physx/source/physxvehicle/src/PxVehicleComponents.cpp` and
  `physx/include/vehicle/PxVehicleComponents.h`: defaults and definitions for
  tire stiffness, tire-load dependence, and the friction-vs-slip graph.
- `physx/documentation/PhysXGuide/_sources/Manual/Vehicles.txt`: vehicle SDK
  setup and model documentation.

The most relevant source routines in `PxVehicleUpdate.cpp` are
`computeTireSlips` and `PxVehicleComputeTireForceDefault`. They show that the
PhysX 4.1 SDK reference model regularizes low-speed slip, changes the
longitudinal denominator depending on whether torque is applied, computes
load-dependent lateral stiffness, and couples longitudinal/lateral forces
through a combined-slip saturation law. `PxVehicleComponents.h` gives default
stiffnesses and a three-point piecewise-linear friction graph.

These files are valuable mechanics references, **not a drop-in exact model of
Unity `WheelCollider`**. The documented Unity/AutoDRIVE curves are separate
forward/sideways two-piece cubic curves, while the PhysX SDK default shader
uses a different force construction. Unity's public API does not reveal how
its native wrapper maps `WheelFrictionCurve` into its internal solver. The
exact Unity native `WheelCollider` implementation is not available in the
downloaded project or public C# reference source; access to Unity's engine
source would be needed to inspect that implementation directly.

## Consequence for the estimator and offline plant

The source-supported path is to keep rigid-body motion explicit, use the
actual measured actuator feedback as the held input, propagate wheel/body
states with 1 ms internal steps, and treat cumulative rear encoder angle as
an integrated wheel-rotation observation at each 25 ms packet boundary. Tire
forces should use the AutoDRIVE guide's separate slip curves unless the
captured pinned-player data demonstrates a better mapping. IMU and encoder
measurements then correct the observer at packet boundaries. Fit only
unreported parameters (for example spline tangents, effective inertia, and
load response) on training captures; score recursive prediction on whole
unseen runs and practice. Ground truth remains offline-only.

The PhysX 4.1 equations can inform hypotheses about low-speed regularization,
combined-slip saturation, load response, and numerical substeps. They must be
treated as competing physics hypotheses and accepted only if the pinned
simulator's held-out data supports them. No simulator physics, reference
trajectory, or competition topic policy was changed in this audit.

## Existing development telemetry hook

`Assets/Scripts/OpenPlaneWheelDynamicsCapture.cs` is already called from the
vehicle's `FixedUpdate` in development builds. When `SDU_APEX_DYNAMICS_DIR`
is set, it buffers 300,000 1 kHz samples (300 seconds) and writes a CSV on
application exit. It records all four wheel RPMs, actual steer/motor/brake
torque, `WheelHit` slips and normal contact load, body pose/velocity/angular
velocity, wheel contact directions, and vehicle/wheel/suspension/friction
metadata. The capture also serializes `WheelCollider.mass`, relevant to
wheel-spin response and documented separately from sprung mass in Unity's
[2022.3 manual](https://docs.unity3d.com/2022.3/Documentation/Manual/class-WheelCollider.html).
The hook is read-only.

The source project compiled successfully as a Linux Development player. Its
startup faults were scene-level rather than physics: the GUI connection button
is under inactive `Menu`, and the serialized front-camera reference is a
stripped component that makes `FrameGrabber` throw during connection. The
development bootstrap runs from the active UI root and invokes the existing
GUI toggle after callbacks register. The kinematics-only scene omits camera
image arrays; vehicle physics and motion/IMU/encoder/actuator/LiDAR references
are unchanged. The bootstrap is currently gated by
`SDU_APEX_DYNAMICS_DIR`, so replay runs enable the read-only capture as well.

The 260-second source capture `r07` completed with 289,450 1 kHz samples,
0 overwritten rows, all four wheels grounded for 99.99965% of samples, zero
collisions and no bridge timing faults. Its public telemetry ran at 39.95–40.02
Hz. It covers 0–11.1 m/s, steering through the allowed range, actual wheel RPM,
WheelHit slip and contact load. The closed capture is
`live_runs/unity_exact_binary_audit_20261012/dynamics_source_parity_20261012_r07/`;
its ROS bag is `live_runs/unity_vehicle_model_parity_source_20261012_r07/run/`.

### What 1 kHz WheelHit data establishes

Using body velocity, angular velocity, the measured contact point and Unity's
reported tire basis directions, the rear-wheel sideways-slip observation is
reproduced by

`sidewaysSlip ≈ v_contact_side / (abs(v_contact_forward) + 1 m/s)`.

On the r07 capture, RMSE is about `5.5e-4` across samples above 1 m/s; within
6–12 m/s it is `3.2e-5` to `7.0e-5`. This is a strong empirical reconstruction
of the native `WheelHit.sidewaysSlip` signal, not a tire-force validation and
not yet an independent-player validation. The guide's idealized
`v_y / abs(v_x)` relation is close but not identical to the native signal.

For longitudinal WheelHit slip, the analogous regularized candidate
`(R * wheelOmega - v_contact_forward) / (abs(v_contact_forward) + 1 m/s)` has
RMSE `0.0048–0.0049` in the 9–12 m/s bands, but about `0.030` at 1–3 m/s.
The low-speed residual is still unresolved; possible causes include the native
slip definition, wheel-point/timing alignment, or low-speed regularization. It
is not being hidden with an arbitrary correction. The official guide's
unregularized slip equations and the measured native WheelHit values are
separate evidence and must not be conflated.

The `r02` pinned-player vs. `r07` source-player comparison is not an acceptance
result: the closed-loop runs used the same planned sequence but differed in
actual sent commands (packet-sequence RMSE `0.034` steering-normalized,
`0.0084` throttle-normalized). Their accumulated pose diverged substantially.
The `r11` source capture is now closed and scored, but its public command
replay did not match the `r02` request boundary. It is a distinct held-out
Unity trajectory, not exact command-parity evidence. It contains 297,415 rows;
the 300,000-row ring buffer overwrote 70,665 earlier rows. It is retained for
cross-run transfer checks, not labeled as a pinned-player replicate. The
earlier `r08–r10` attempts did not send commands and remain excluded.

### Direct state equations and what the captures prove

The source gives a useful decomposition rather than one fitted vehicle model:

1. **Actuation and steering.** `VehicleController` maps normalized throttle to
   four wheel torques and turns exact zero throttle into four-wheel braking.
   The Explore scene override is `MotorTorque=428 N m`; `CAWD` therefore
   applies `107 * throttle` N m to each wheel, and zero applies `428 N m` brake
   per wheel. This is confirmed in `r07/r11` actuator columns. Steering is
   rate-limited at `183.346 deg/s` with a 30-degree center limit and then
   mapped through Ackermann geometry. The radians-per-second constant in the
   mechanics helper is the degree value converted to radians.
2. **Contact-point kinematics.** For wheel `i`,
   `v_i = v_CG + omega x (p_i - p_CG)`. Projecting this velocity and wheel
   angular speed into each measured tire frame gives the documented
   longitudinal/lateral slip candidate. The guide's zero-speed singularity
   remains real; do not silently extend those equations through rest.
3. **Suspension/load.** The active prefab has `spring=500 N/m`,
   `damper=100 N s/m`, `targetPosition=0.5`, and `suspensionDistance=0.05 m`.
   The separate `Suspension.cs` and `AntiRollBar.cs` scripts are not attached
   to this vehicle prefab. Fitting each wheel's captured contact load from
   wheel-pose height and its 11 ms derivative gives `k=491.5–504.9 N/m` and
   `c=99.35–100.16 N s/m` on `r07`. Frozen on `r11`, the serialized 500/100
   coefficients with one learned per-wheel target-height calibration yield
   `0.394–0.402 N` load RMSE per wheel on `r11`; refitting `k,c` changes this
   only slightly (`0.394–0.396 N`). More importantly, wheel travel solved from
   body pose, the serialized wheel transforms, tire radius, and the known flat
   plane reconstructs captured wheel-center height within `0.17–0.19 mm` RMS.
   The geometry-only load prediction is `0.52–0.55 N` RMSE on `r11`; learning
   each wheel's target-height correction on `r07` reduces it to
   `0.375–0.403 N` on `r11`. Those corrections are only `0.59–0.69 mm` from the
   prefab's target center. `wheel_suspension_travel_from_body_pose`,
   `wheel_suspension_compression_m`, and `wheel_suspension_load_n` now encode
   this geometry and spring/damper law as tested reusable mechanics helpers.
   This validates a useful load-state component on a flat plane, not yet a
   runtime observer or recursive full-vehicle rollout; the road plane and
   body pose must be estimated, and uneven/banked terrain remains untested.
4. **Tire force.** The guide's two independent two-piece curves are useful
   structure, but their native spline tangents and exact WheelCollider mapping
   are not public. Fitting a shared scale and zero-slip tangent to r07 force
   balance reduces the held-out net-force error to about `1.76 N` vector RMS
   on r11. However, the same model leaves `0.256 N m` yaw-torque RMSE. A
   3-axis force-and-moment joint fit reduces held-out yaw angular-acceleration
   RMSE from `2.42` to `1.86 rad/s^2`, but worsens held-out translational force
   vector RMSE from `1.76` to `2.34 N`; other moment axes are not uniformly
   improved. Keep this physically structured fit as a diagnostic candidate;
   the rollout tradeoff identifies a mismatch to explain, not grounds to
   discard the more source-faithful force/moment equations. It is not yet a
   production tire model.
5. **Rigid body and pose.** Body translation follows `m(a-g)=sum(F)` after
   drag; body rotation follows
   `I*alpha + omega x (I*omega)=sum(r x F)` with the runtime-captured rotated
   inertia tensor and angular drag. The gyroscopic/off-diagonal yaw correction
   is only about `0.02 N m RMS`, much smaller than the remaining `0.24–0.29 N m`
   yaw torque residual. The dominant remaining issue is therefore not simply
   a missing scalar-vs-3D yaw equation; per-wheel force distribution, contact
   dynamics, or timing is still wrong.

The 1 kHz command-only planar replay is still a failure: it reaches roughly
`0.24–0.28 rad/s` yaw-rate error by 100 ms, `0.76–0.87 rad/s` by 250 ms, and
about `15.5–16.6 m` position RMSE over 5 s. Therefore no complete offline
plant or odometry improvement is claimed. The new reusable
`wheel_suspension_load_n` helper implements the validated spring/damper law,
but it is a mechanics primitive; it has not yet been composed into a validated
full 3D command-only observer or promoted into production odometry/MPC.

The next model step is to place this geometry-derived suspension state inside
a full 3D wheel/body teacher and test recursive position, speed, and yaw on
whole held-out runs. Tire force must fit translation and moment together
without trading away translational accuracy. IMU/encoder measurements may
correct the observer only at their actual packet boundaries; simulator truth
and WheelHit labels stay training/scoring-only. No simulation physics or
reference trajectory was changed.

## Follow-up: force and frame residual diagnosis — 2026-10-13

The source does provide a mechanics decomposition; the model does not need to
learn every state from scratch. The directly calculable parts are

\[
v_i = v_{CG} + \omega \times (p_i-p_{CG}),\qquad
S_{x,i}=\frac{R_i\omega_i-v_{i,long}}
 {\max(|R_i\omega_i|,|v_{i,long}|)+\epsilon},
\]

plus the measured wheel frame, source steering geometry, suspension ray/plane
intersection, spring/damper load, wheel torque balance, and Newton–Euler rigid
body equations. The project source fixes command-to-torque and steering
actuation; the serialized prefab fixes geometry and friction-curve landmarks.
The *native* WheelCollider still owns the exact contact-frame construction,
effective wheel inertia, friction response details, and internal update phase.
Those are the things that need identification; this is not evidence that
modelling is impossible.

The latest frozen training/held-out report is
`live_runs/unity_exact_binary_audit_20261012/wheel_spin_state_inputs_v10_20261013.json`.
All fitted quantities below use r07 training only; r11 is held out. r11 is an
independent source-player trajectory, but not an exact replay of r07's
commands.

| Current-state test | r11 result | Decision |
| --- | ---: | --- |
| Reconstructed-input wheel spin, 1 ms step | RMSE `0.05340 rad/s`, p95 `0.09163`; 1,189,653 wheel transitions | Useful wheel-subsystem evidence only; large rare zero-RPM release misses remain, and body truth is supplied every step. |
| Frozen tire response + reconstructed slip/load, body force | `3.122 N` vector RMSE | Baseline; not a full rollout. |
| Shared lateral response fit to r07 lateral body-force component | Force improves to `1.892 N`; yaw moment `0.326→0.238 N·m`; pitch-axis angular-acceleration RMSE worsens `10.77→14.02 rad/s²` | Reject as a complete model: it trades away a rotational state. |
| Separate front/rear lateral curves fit to all r07 force/moment axes | Force worsens to `5.495 N`; moment and angular-acceleration axes improve | Reject: no joint force/moment improvement. |
| Per-wheel lateral curves | Same held-out force regression (`5.501 N`) despite lower moment residual | Reject; added degrees of freedom do not solve force balance. |
| Friction-circle coupling candidate | Fit drives `mu` to its upper bound; force worsens to `5.507 N` | Reject; no evidence this simple combined-slip law explains the residual. |
| Constant body-frame contact offsets fit on r07 WheelHit points | Contact-point RMS falls from `8.65–8.76 mm` to `4.92–5.42 mm`; force worsens `3.122→4.774 N`, yaw moment `0.326→0.731 N·m` | Reject as a physics correction. A closer point coordinate alone is not a better movement model. |
| Serialized rest-base force point + r07 joint lateral-curve refit | On r11 force-vector RMSE `3.122→1.993 N`; torque axes `[0.054,0.326,0.276]→[0.044,0.277,0.124] N·m`; angular-acceleration axes `[0.579,3.597,10.772]→[0.476,3.524,5.801] rad/s²` | Positive one-step whole-run transfer: all force and angular-acceleration axes improve. This is the first joint force/moment candidate that does not trade one state for another. |

The lever-arm result required a unit correction. The capture's calibrated
target travel is already in metres; an initial probe multiplied it by
`suspensionDistance` a second time and produced an invalid 2.4 cm point error.
Those numbers were discarded. The corrected source reconstruction uses the
serialized target *fraction* `t` and suspension distance `d`:

\[
p_{app}=p_{mount}+R_wc+R_w(0,-1,0)\big((1-t)d+R_{tire}\big)
 +(0,\texttt{forceAppPointDistance},0),
\]

where the final offset follows the rigid-body local up axis. This follows the
Unity manual's convention that target position 1 is fully extended, 0 is fully
compressed, and the force application distance is measured from the resting
wheel base ([Unity Wheel Collider manual,
2021.2](https://docs.unity.cn/2021.2/Documentation/Manual/class-WheelCollider.html)).
The corrected point is within roughly 7–8 mm of the measured WheelHit point in
the dominant axes. Using that serialized point *without refitting* leaves
translation unchanged, improves roll/pitch torque residual, but worsens yaw
torque (`0.326→0.373 N·m`) and yaw angular-acceleration RMSE
(`3.60→5.35 rad/s²`). That alone is not a win.

The important result is the **joint** step: freeze this source-derived point,
freeze the longitudinal tire response, and fit only the two shared lateral
curve parameters against all three r07 force and all three moment residuals.
The r07-only result is lateral scale `1.13191` and zero-slip tangent `76.0294`
(previous shared fit at the WheelHit lever: `1.15436`, `128.8003`). Frozen on
r11, the force-axis errors all fall (`[2.729,1.277,0.818]→[1.392,1.182,0.799] N`),
and all three angular-acceleration errors also fall. Repeating the scoring with
101 ms rather than 31 ms differentiation preserves the held-out gain: force
vector RMSE `3.183→2.084 N`; torque axes
`[0.058,0.349,0.275]→[0.049,0.300,0.121] N·m`; angular-acceleration axes
`[0.618,3.876,10.750]→[0.528,3.781,5.762] rad/s²`. This is a genuine
training-to-held-out improvement in current-state acceleration prediction,
not yet a recursive plant result.

The state-only lateral-slip residual is mainly tire-frame reconstruction, not
the contact-point location. On r11, using reconstructed point *and* frame gives
side-slip RMSE about `0.00153`; substituting the measured wheel frame while
keeping the reconstructed point gives `0.00086`, while measured point and
measured frame gives `0.00032`. A ±1 ms steering-index sweep has minimum average
frame/slip error at the next recorded steering sample on both captures
(`0.00250→0.00206 rad` on r07, `0.00230→0.00207 rad` on r11), but p95 barely
moves and rare tail errors remain. Since that uses a future sample, it is only
a diagnostic for telemetry/physics phase, not a causal observer correction.

### What this means for a practical observer / teacher

Separate model state into:

1. **Known inputs and actuator state:** the actual command, the source's
   nonzero-throttle torque law and exact-zero brake branch, and the
   rate-limited Ackermann steering state.
2. **Measured/estimated dynamic state:** body pose, body-frame `u/v`, 3-axis
   angular rate, four wheel angular speeds, and suspension travel/rate. A
   25 ms observer can update this from permitted IMU/encoder/actuator
   measurements; simulator truth is for offline targets only.
3. **Direct mechanics:** contact-point velocity, wheel surface speed, slip
   candidates, spring/damper load, rigid-body force/moment integration, and
   pose integration. These equations can be computed from state and geometry.
4. **Empirical native residual:** the remaining wheel friction/application
   response and the exact phase of Unity's WheelCollider outputs. Fit that
   component against r07 labels and hold it fixed on whole-run r11; do not
   absorb it into unrelated odometry gains.

This is the first model candidate that improves all six held-out force and
angular-acceleration components in the one-step Newton–Euler balance. It is
now the candidate to carry into the next test: a 1 kHz, command-only recursive
3-D body/wheel rollout, initialized once and scored against r11 position,
speed, and yaw at increasing horizons, without refreshing state from future
truth. That will tell us whether this local mechanics gain survives recursion.
Until that passes, it is not a deployable observer and it does not establish
accurate long-horizon state prediction. No production odometry/MPC, simulator
physics, or reference path was changed in this follow-up.

## Primary references

- [Unity 2022.3 WheelCollider manual](https://docs.unity3d.com/2022.3/Documentation/Manual/class-WheelCollider.html)
- [Unity C# reference bindings for Vehicles](https://github.com/Unity-Technologies/UnityCsReference/blob/2022.3/Modules/Vehicles/Vehicles.bindings.cs)
- [NVIDIA PhysX 4.1 `PxVehicleUpdate.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleUpdate.cpp)
- [NVIDIA PhysX 4.1 `PxVehicleSuspWheelTire4.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleSuspWheelTire4.cpp)
- [NVIDIA PhysX 4.1 `PxVehicleTireFriction.cpp`](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleTireFriction.cpp)
- [NVIDIA PhysX 4.1 vehicle guide source](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/documentation/PhysXGuide/_sources/Manual/Vehicles.txt)
- [AutoDRIVE RoboRacer technical guide, vehicle dynamics](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics)

## Follow-up: recursive rollout and brake constraint — 2026-10-13

The previously recorded command-only planar replay was re-run with a targeted
comparison of its wheel-brake integration. This follows directly from source:
`VehicleController.Drive()` sets all four `WheelCollider.brakeTorque` values to
`MotorTorque=428 N·m` when `DriveTorque==0` (the active prefab is CAWB), and the
r11 capture confirms all four wheel RPM values are exactly zero in 99.99% of
zero-throttle samples. The old replay integrated this brake as continuous
Coulomb torque over a 1 ms RK2 step, which can numerically reverse a stopped
wheel. The evaluator now includes a unilateral angular-impulse mode that
cannot reverse wheel rotation under braking. This is an offline model change;
no production odometry, Unity source physics, or controller was changed.

The complete 5 s rolling-window comparison is saved at
`live_runs/unity_exact_binary_audit_20261012/source_mechanics_command_rollout_v12_20261013.json`.
It uses 52 r07 training windows and 39 untouched r11 windows per candidate,
initialized once per window and then driven only by the recorded command trace.
The correction removes the old wheel-speed explosion (about `389 rad/s` wheel
RMSE at 0.75–1 s) and holds wheel error around `0.7–1.0 rad/s` at 0.75–2 s.
This confirms the brake integration defect, but it does **not** validate the
vehicle plant: r11 yaw-rate RMSE remains `0.278 rad/s` by 100 ms and
`0.87 rad/s` by 250 ms; lateral-speed RMSE is `0.112 m/s` and `1.15 m/s` at
those horizons. At 5 s, position radial RMSE remains `16.57 m` and yaw-rate
RMSE `6.67 rad/s`.

The r07-only lateral curve refit that improved current-state force balance was
also inserted as a recursive ablation, with every other parameter frozen. It
does not materially improve r11 recursive motion: at 100 ms yaw-rate RMSE is
`0.279 rad/s` versus `0.278` for the frozen force fit, and at 250 ms it is
`0.870` versus `0.873`; lateral-velocity and 5 s position results are slightly
worse. Therefore the one-step force-balance gain is not promoted as a
long-horizon movement-model improvement.

The first divergence is now localized more tightly: steering feedback is
predicted within about `0.001 rad` on most held-out windows, but yaw error is
already `0.075 rad/s` by 25 ms, while wheel-speed error remains about
`0.05–0.08 rad/s`. By 100 ms the main errors are yaw and lateral body speed,
not actuator steering or wheel-speed blow-up. The exact source Ackermann map
reconstructs the two measured front steering angles with `0.20°` RMS on r07
(the r11 capture has some non-finite rows); a ±1-sample command-phase check
changes steering error only marginally. These checks do not justify an
arbitrary steering delay/gain.

**Current conclusion:** the brake equation is now physically constrained in
the research replay, but the full model still fails long-horizon prediction.
The next required model component is the lateral tire/body response and its
roll/pitch/suspension coupling. The corrected source force-application-point
and lateral-curve candidate is still only a one-step force-balance result; it
must be incorporated into a 3-D recursive model with attitude and suspension
states and then scored on whole held-out runs. No simulator run or production
stack change has been made during this follow-up.

## Follow-up: explicit 3-D command-only plant — 2026-10-13

The next step was implemented in
`tools/vehicle_dynamics_learning/replay_unity_source_3d_plant.py`. Unlike the
previous planar rollout, the state now carries world position and quaternion,
3-axis body velocity and angular velocity, four wheel angular speeds, and
center steering. Each 1 ms step computes wheel contact-point velocity, the
capture-calibrated longitudinal slip and lateral slip angle, ray/plane
suspension travel and spring/damper load, fitted tire-curve forces, full
Newton–Euler moment balance with the captured rotated inertia tensor, and the
source steering/torque/brake laws. Rigidbody transform-origin position is
advanced separately from COM velocity using
`p_origin_dot = R * (v_COM - omega_body x COM_offset)`. The zero-throttle
brake uses a unilateral stop. Only the initial state and the current recorded
steering/throttle command enter a rollout; later truth is used for scoring.

All fitted values are frozen from r07: force-curve coefficients, the
longitudinal slip regularizer/radius correction, suspension target-travel
calibration, effective wheel inertia, and the source-force-point lateral
curve fit. The independent r11 capture was not used to fit them. A 5 s
command-only evaluation uses 52 non-overlapping r07 windows and 39 r11 windows;
it does not refresh state within a window. Full per-window errors are retained
in
`live_runs/unity_exact_binary_audit_20261012/source_mechanics_3d_plant_5s_diagnostics_20261013.json`.

On independent r11, the 3-D model's 1 s horizontal position RMSE is `0.170 m`
(p95 `0.282 m`), heading RMSE `0.068 rad`, yaw-rate RMSE `0.088 rad/s`, and
body-velocity component RMSEs are `[0.012, 0.002, 0.026] m/s`. At 5 s, position
RMSE is `1.874 m` (p95 `3.432 m`), heading RMSE `0.239 rad`, yaw-rate RMSE
`0.085 rad/s`, body-velocity component RMSEs `[0.012, 0.003, 0.018] m/s`, and
per-wheel angular-speed RMSEs `[0.082, 0.082, 0.066, 0.060] rad/s`. The 5 s
heading bias is about `-0.052 rad`. This is a large improvement over the
previous planar command replay at 1 s (position `0.578 m`, yaw-rate
`1.831 rad/s`), but it is **not** a negligible-error full-lap simulator.
The comparison changes multiple mechanics together, so it establishes a
promising combined candidate, not the contribution of each individual term.

The held-out windows localize the remaining errors. Some 7–10 m/s, low-steer
windows accumulate a roughly `0.07 rad/s` yaw-rate offset for several seconds;
one 5 m/s window beginning near `0.30 rad` steering develops a sharp yaw-rate
error around `0.5 s` before the final rate re-converges. Speed and wheel errors
remain small in these cases while heading/position do not. This points to
remaining yaw/tire transient response, not a missing wheel-speed equation. It
does **not** yet identify whether the missing mechanism is tire-force memory,
combined slip, or another WheelCollider force detail.

A single shared friction-ellipse combined-slip hypothesis was also tested
using r07-fitted parameters and r11 force/moment labels. Fitting its one cap
factor chose `1.449`, i.e. it is mostly inactive; held-out force-vector RMSE
changed only `1.762 N` to `1.695 N`, and yaw-moment RMSE `0.2562 Nm` to
`0.2555 Nm`. A hard cap of `1.0` worsened both. This does not explain the
remaining yaw/position error and is not included in the 3-D plant. NVIDIA's
[PhysX 4.1 vehicle source](https://github.com/NVIDIAGameWorks/PhysX/blob/4.1/physx/source/physxvehicle/src/PxVehicleUpdate.cpp)
is only a competing reference model; it is not proof of Unity's native
WheelCollider implementation.

**Current state:** a source-backed 3-D free-running plant is implemented and
has substantially better held-out motion prediction, but 5 s pose drift is
still too large for a trusted offline raceline simulator. No production odom,
MPC, Unity physics, or track reference was changed. The next focused step is to
measure the yaw-rate residual by speed/steering and transition history on r07,
fit a causal yaw-force residual using r07 only, and replay the same windows on
r11. Promote it into the research plant only if both one-step and recursive
held-out yaw, position, and other-state errors improve; otherwise preserve the
current 3-D model as the comparator and diagnose the transient more narrowly.

## State decomposition and command-phase follow-up — 2026-10-13

The Unity source does specify most of the state-transition structure. The
research plant carries

`x = (world position, body quaternion, COM body velocity, body angular
velocity, four wheel angular speeds, center steering)`.

At each 1 ms step, the known source and rigid-body equations provide:

- actuator steering saturation/rate limit, Ackermann front-wheel angles,
  common motor torque, and the exact-zero-throttle brake branch;
- wheel contact-point velocity, projected into each tire frame;
- longitudinal/lateral slip estimates from contact velocity and wheel surface
  speed, with the documented low-speed regularization kept explicit;
- suspension ray travel and spring/damper normal load from body pose, wheel
  geometry, and its velocity;
- `m(a-g)=sum(F)` and
  `I alpha + omega x (I omega)=sum(r x F)`, followed by quaternion, COM and
  transform-origin pose integration.

The source does **not** disclose the native WheelCollider tire/contact-force
solver or every hidden solver state. Those are the fitted residual terms; the
model is not learning pose kinematics, torque assignment, or Newton–Euler
integration from scratch. A 1 kHz integrator matches the simulator's fixed
step, but creates no extra sensor information.

The summary-only command-profile experiment used 10,409 r07 25 ms transitions
(including all 1,918 intervals previously excluded for within-interval input
changes) and 39 disjoint 5 s windows on held-out r11. Relative to the
source-mechanics plant without a learned yaw residual, the DAgger residual
candidate changed held-out 5 s horizontal position RMSE `1.874 -> 0.810 m`,
heading RMSE `0.239 -> 0.055 rad`, and yaw-rate RMSE `0.085 -> 0.015 rad/s`;
position improved in 33/39 windows. The 5 s position p95 remains `1.371 m`,
so this is a useful research improvement, not an accurate full-lap plant or a
production odometry change. All four wheel-speed RMSEs remain around
`0.063–0.086 rad/s` (about `0.004–0.005 m/s` at the wheel surface).

A paired 25 ms transition explains why steering/throttle history matters. At
one r07/r11 pair, start body/wheel states differed by only about `0.005 m/s`,
`0.001 rad/s`, and `0.003 rad/s` per wheel. The same steering unwind and
throttle rise occurred 2 ms later in r11. r07's rear wheel rose from about
`1,339` to `1,757 rpm`; r11 stayed near `1,335 rpm`. Both remained grounded
and received the corresponding source motor torque. The base 3-D plant
predicted the endpoint wheel speeds within `0.3 rad/s`, but missed yaw-rate
increments in both runs (`+0.047` and `+0.056 rad/s` endpoint error). This
points to a high-sensitivity, millisecond-phase tire/yaw response—not a need
to fit wheel radius or a gross forward-speed state error.

The new experiment passed the **entire 25-sample-per-command profile** to the
residual model, instead of compressing a 25 ms interval to
mean/range/total-variation. On the same r11 windows, full-profile DAgger gives
5 s position RMSE `0.794 m` versus `0.810 m` for the summary-profile model,
but position p95 is slightly worse (`1.392` versus `1.371 m`); heading RMSE is
`0.050` versus `0.055 rad`, and yaw-rate RMSE is essentially unchanged
(`0.0150` versus `0.0151 rad/s`). Per-window position error improves in only
17/39 cases relative to the summary model, and mean absolute position error is
effectively identical (`0.5152` versus `0.5149 m`). The extra 50 command
features therefore do not provide a durable transfer gain; retain the simpler
summary-profile candidate as the comparator. The command timing explains the
wheel-spin difference and is already represented by the 1 ms plant input, but
does not resolve the yaw-force residual. The r11 capture remains
evaluation-only. No Unity physics, runtime odometry/MPC, or reference
trajectory has been changed.

## Runtime observability and 3-D term check — 2026-10-13

The equations are available, but not every state needed to evaluate them is a
competition-side measurement. The distinction is important: using a physical
equation does not make its hidden inputs observable.

| Quantity | Source equation / evidence | Available to production odometry? |
| --- | --- | --- |
| Steering state and front wheel angles | Rate-limited center angle, then Ackermann geometry in `VehicleController.Steer()` | Command and feedback are available; actual wheel angles are not separately published. Reconstruct from feedback and source parameters. |
| Motor and brake torque | Exact CAWD/CAWB branch in `VehicleController.Drive()` | Reconstruct from the actual throttle feedback and source configuration; zero throttle is the separate brake-torque branch. |
| Body angular rate and attitude | Unity IMU source transforms Rigidbody angular rate and orientation | ROS IMU contains these, but `sensor_odometry_node` currently consumes only yaw rate and yaw. |
| Body acceleration | Source differentiates world Rigidbody velocity and rotates it into body coordinates | ROS IMU contains acceleration; current odometry consumes only x/y. The source prefab sets `GravityCompensation=1`, which sets the subtraction vector to zero in `IMU.Start()`: despite the flag name, this is uncorrected world-velocity derivative, not gravity-removed specific force. A 3-D model must preserve that convention and must not subtract gravity twice. |
| Rear wheel angular speed | Encoder angle is accumulated from `WheelCollider.rpm` then quantized to cumulative ticks | Only rear left/right angles are published. Differentiate with source-time-aware filtering; wheel speed is not body speed under slip. Front wheel speeds remain latent. |
| Contact speed and slip | `v_contact = v_CG + omega x r`; project into the tire frame and compare to `R*omega` | Requires estimated body velocity, 3-D attitude/rate, wheel orientation and contact geometry. Native `WheelHit` slip is development-only scoring data, not a runtime input. |
| Per-wheel normal load and force | Suspension spring/damper and tire slip curves | Load can be reconstructed on known terrain from suspension geometry; per-wheel force and combined-slip response are hidden behind native WheelCollider. |
| Rigid-body state propagation | Newton–Euler force/moment equations and quaternion kinematics | Direct once initial state and all tire forces are known; currently limited by hidden tire/contact forces, not by missing pose equations. |

The development bridge records simulator position and linear velocity inside
`/autodrive/roboracer_1/bridge_packet_timing` for offline diagnosis. The
production odometry node does not subscribe to that topic; its inputs remain
the two encoder topics and `/imu`. Those simulator values must remain
training/scoring labels only.

The planar observer's body-frame equations

`u_dot = ax + r*v`, `v_dot = ay - r*u`

are the planar reduction of `v_dot_body = a_body - omega_body x v_body`.
The full 3-D reduction additionally contains `-q*w` in `u_dot`, `+p*w` in
`v_dot`, and the corresponding vertical equation. We measured these terms
from the 1 kHz source captures rather than assuming roll/pitch mattered:

| Capture | Full-attitude vs yaw-only horizontal velocity | Omitted 3-D cross-term RMS (u, v) | Omitted-term absolute p95 (u, v) |
| --- | ---: | ---: | ---: |
| r07, 289,450 samples | 0.01054 m/s RMSE; 0.01969 m/s p95 | 0.01092, 0.01323 m/s² | 0.02113, 0.01942 m/s² |
| r11, 297,415 samples | 0.01013 m/s RMSE; 0.01958 m/s p95 | 0.01033, 0.01281 m/s² | 0.02113, 0.01716 m/s² |

This confirms a small, repeatable 3-D correction, but it is far too small by
itself to explain the existing decimetre-to-metre recursive position error.
Keep full attitude/rates in the research teacher for correctness; prioritize
the unresolved tire-force/yaw transient over adding a roll-specific empirical
term. No production odometry behavior changed from this check.

## Tire-force slip versus WheelHit telemetry — 2026-10-13

An equation audit found a distinction that must remain explicit in every
model. Section 1.3.2 of the [AutoDRIVE vehicle guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics)
defines the inputs to its tire-force curves as

`Sx = (R*omega - vx) / vx`,  `Sy = vy / abs(vx)`.

The 1 kHz Unity capture also exposes `WheelHit.forwardSlip` and
`WheelHit.sidewaysSlip`. Reconstructing the latter from contact velocity with
`atan2(v_side, abs(v_forward)+1 m/s)` matches the captured telemetry very
closely (r07 RMSE about `5.5e-4`; independent r11 about `3.2e-4`). That is a
reconstruction of a Unity diagnostic output; it does **not** establish that
these regularized coordinates are the inputs to the guide's tire-force
curves. The earlier greybox revision substituted those telemetry fits into
the force model. That interpretation was incorrect.

The greybox now computes the documented force coordinates, using a
`0.5 m/s` denominator floor only to regularize the guide's zero-speed
singularity. For forward motion at `vx >= 0.5 m/s`, this is exactly the
documented equation. The contact-slip analyzer now labels its WheelHit fits
as telemetry diagnostics and explicitly warns against reusing them as tire
force inputs.

This distinction was checked by recursive replay, not just formula matching.
All candidates below use the same sealed R03/R04 practice holdout (12
lap-count sequences per run); no validation or practice frames were used for
fitting. The baseline is the prior fitted-actuator greybox. Errors are macro
averages over the 12 sequences in each independent run.

| Candidate | R03 position radial RMSE / speed RMSE | R04 position radial RMSE / speed RMSE | R03/R04 250 ms yaw-rate abs error |
| --- | ---: | ---: | ---: |
| Baseline equations/checkpoint | `6.843 m / 3.697 m/s` | `6.708 m / 3.595 m/s` | `0.0948 / 0.0879 rad/s` |
| WheelHit-calibrated slips put into force law; same checkpoint | `7.362 m / 3.991 m/s` | `7.515 m / 4.017 m/s` | `0.0794 / 0.1336 rad/s` |
| WheelHit-slip refit on clean train/validation split | `7.670 m / 4.069 m/s` | `7.621 m / 4.062 m/s` | `0.0863 / 0.0684 rad/s` |
| Guide force slips, refit on same clean split | `6.718 m / 3.556 m/s` | `6.801 m / 3.584 m/s` | `0.0456 / 0.0460 rad/s` |

The guide-slip refit halves the 250 ms yaw-rate error, but position and speed
are not jointly improved, and whole-sequence heading RMSE worsens to `1.292`
and `1.263 rad` (from `1.082` and `1.159`). This does **not** falsify or
disqualify the documented slip equations: they remain the source-faithful
equation candidate, while the worse recursive trajectory is evidence that
other mechanics or assumptions in the composed plant are still mismatched.
The WheelHit telemetry reconstruction answers a different question (matching
Unity's reported diagnostic slip) and must not be substituted for tire-force
slip merely because it has lower telemetry RMSE. Retain both variants for
causal diagnosis; neither is yet a fully validated offline plant. Reports are retained at
`live_runs/derived_dynamics_learning_20260928/command_only_750ms_rollout_20261013/`.

## Source-actuator wheel-spin screening — 2026-10-13

The pinned Explore scene configures CAWD/CAWB: `VehicleController.Drive()`
applies `428 * throttle` total motor torque, divided equally across the four
wheels for nonzero throttle; exactly zero throttle takes a separate `428 N m`
brake branch. The 1 kHz capture records each WheelCollider torque directly;
reconstructing that source mapping from throttle agrees to about `7e-7 N m`
RMSE. The guide's solid-cylinder wheel inertia from `m=0.109 kg`, `R=0.059 m`
is `0.0001897 kg m^2`; a separate r07/r11 wheel-spin identification estimates
an effective WheelCollider inertia near `0.0004366 kg m^2`.

The default four-wheel greybox had instead learned a free `motor_torque_scale`
near `0.48 N m` total. This is an effective fit parameter, not the physical
Unity torque, and should never be presented as such. Added an explicit
source-actuator training mode with 25 internal substeps per 25 ms transition
(1 kHz), exact 428/4 CAWD torque, and a unilateral zero-throttle stop. Its
guide-slip screening fit (400 updates, 0.25 s training horizon) improved its
held-out open-plane normalized score to `0.399`, but failed the independent
practice transfer decisively: per-lap position radial RMSE was `10.58 m` and
`10.23 m`, whole-sequence speed RMSE `1.48` and `1.58 m/s`, and rear-wheel
surface-speed RMSE about `453 m/s`. It is not promoted. This exposes the
remaining mismatch between the guide tire-force surrogate and the Unity
WheelCollider response under source-level motor torque; merely inserting
the correct actuator torque into that simple force model is insufficient.

The source-actuator mode and tests are experimental and offline-only. There
was no simulator run, production odometry/MPC change, or reference raceline
change. The exact CAWD torque and zero-throttle brake mapping are source facts
and must remain exact in the model; the failed wheel-speed rollout is evidence
of a missing or incorrect wheel/contact response, not a reason to substitute
an effective torque that contradicts the controller code. In particular, the
`453 m/s` rear-wheel surface-speed failure points to an unresolved wheel
rotational/contact equation or integration issue that must be isolated before
using this candidate. The separate 1 ms 3-D model with fitted WheelCollider
force response remains a comparator (r11 5 s position RMSE `0.794 m`), not a
trusted offline simulator and not yet verified on practice.

## Source-fidelity decision — 2026-10-13

Model selection is now explicitly two-axis: (1) whether an implementation
matches the Unity/AutoDRIVE equations and serialized configuration, and
(2) whether its recursive outputs match held-out Unity motion. A poor score on
(2) does not erase evidence of correct source mechanics under (1); it
requires locating the remaining model boundary, hidden native behavior,
timing, state, or implementation error. Keep equation-faithful candidates
available and report their output errors separately. Do not silently replace a
source equation with an empirically convenient law just because the latter
scores better on one metric.

The public Unity C# project is authoritative for `VehicleController`, sensor
wrappers, scene wiring, and serialized project settings. It is not the complete
Unity engine source: `WheelCollider`'s force solver is native and unavailable
in the project. Therefore the most literal offline reference for that part is
the matching Unity runtime itself; any standalone WheelCollider reimplementation
must state its engine assumptions and be checked against the pinned player.
Reinforcement learning and learned residuals are not the default route when a
relevant source equation exists. The first source-player/pinned-player replay
was not a valid physics comparison because the ROS-side replay emitted queued
command callbacks in bursts; the direct bridge-request replay correction and
its next real-simulator validation are documented below.

## Source-player / pinned-player command-parity attempt — 2026-10-13

The first 10,406-sample replay (`r02`) completed with zero collisions, zero
bridge timing faults, and a 40 Hz packet stream (median interval `24.962 ms`,
95th percentile `25.838 ms`). It is **not** valid evidence about physics
equivalence: the ROS replay node reported up to `130.283 ms` scheduling
lateness, and packet-sequence-aligned commands differed (`0.04157` normalized
steering RMSE; `0.02050` throttle RMSE). The apparent `8.92 m` relative-position
RMSE therefore cannot be assigned to the Unity model or physics.

The failure mechanism is the harness boundary: the bridge keeps only the
latest ROS command, while a delayed subscriber can drain several queued packet
callbacks and publish a burst between 40 Hz requests. Some intermediate
commands are then never sent. `build_unity_packet_command_profile.py` now
extracts the commands the source run actually placed in each request. In
profile mode, `bridge_40hz.py` selects one sample synchronously at its existing
request-send boundary, rather than sending through a separately scheduled ROS
publisher. The mode is opt-in through
`SDU_APEX_PACKET_COMMAND_PROFILE`; it adds no topic subscription and is absent
from normal launches. The parity runner uses it for the next pinned Explore
replay and requires exact packet-sequence command agreement before interpreting
motion differences.

No learned model, odometry, MPC, raceline, or simulator physics was changed by
this diagnosis. The `r02` bag remains a small harness-diagnostic capture and
must not be used to fit or score vehicle dynamics.

## Direct source-Unity offline replay — 2026-10-13

The next step was to run the source vehicle itself, not a greybox or learned
replacement. `OfflineCommandProfileRunner.cs` is an opt-in development runner:
it disables only Socket/Socket.IO, sets the actual `VehicleController` to
autonomous mode, applies each command at the Unity fixed-step boundary, and
holds it for 25 physics steps. It requires the project's serialized
`Time.fixedDeltaTime=0.001 s`; it does not edit any physics parameter. The
official Explore Docker image was launched through `tools/start_simulator.sh`
in batch mode under Xvfb, with the source-built Linux player mounted read-only.
No GUI or `-no-graphics` mode was used.

Run evidence is in
[`unity_source_exact_offline_plant_20261013_r01`](../../live_runs/unity_source_exact_offline_plant_20261013_r01/):

- Unity build succeeded with the repository's Unity `2022.3.52f1` editor.
- `offline_result.json`: `complete`, 10,487/10,487 commands, 262,175/262,175
  fixed steps, 1 ms timestep, zero overwritten samples.
- `wheel_dynamics_...csv`: 262,175 1 kHz samples, about 350 MB, with actual
  rigid-body state plus all four WheelCollider RPM, contact, slip, force,
  torque, steering, and pose outputs. This is preserved workspace data, not a
  temporary file.
- The command comparator verifies the packet command sequence against the
  Unity-applied command on every 1 ms step: steering and throttle max error are
  both exactly `0.0`.
- State comparison uses the `GPS.cs` rear-axle point, not the rigid-body origin.
  The latter is about 0.16 m forward and 0.05 m below the GPS child; this is
  explained by the serialized `Rear Axle Center` child offset `(0, 0.05,
  -0.16) m`, not a fitted correction. The reconstructed initial pose agrees
  with the reference to about 1 mm.

Against source-built Explore capture r07, at the nominal same-command packet
phase, the replay gives body-velocity vector RMSE `0.00128 m/s` (p95
`0.00194`), yaw RMSE `0.00748 rad` (p95 `0.0182`, max `0.0250`), and yaw-rate
RMSE `0.00468 rad/s`. Relative position RMSE is `0.443 m`, p95 `1.087 m`, and
endpoint error `0.259 m` over 262.2 s. Position residual is small early
(`0.030 m` RMSE over 0–30 s) but grows during the later fast segment: `1.047 m`
over 210–240 s and `0.612 m` over 240–262 s. These are not lap scores, and the
long-run position drift is not negligible; it remains a validation gap rather
than being hidden by the low one-step velocity/yaw errors. The report includes
a ±25 ms capture-phase sweep only as a timing diagnostic; it does not select a
per-state best offset as a model correction.

This is the first working offline plant that actually executes the Unity
`VehicleController`, scene configuration, and native Unity WheelCollider solver
at 1 kHz. It establishes that a learned plant is not required to reproduce the
simulator's mechanics. The remaining discrepancy is narrowed to cross-run /
packet-phase sensitivity and long-horizon state divergence; it is not evidence
that source equations should be replaced with RL. The source player is usable
as an exact-code offline teacher and command-profile evaluator, but this 262 s
replay does not yet establish negligible error on a complete practice lap or
justify promoting a production odom/MPC change. Next, use this same runtime on
recorded practice commands and account explicitly for actuator/packet
observation phase before making lap-ranking claims.

## Practice race-scene confirmation — 2026-10-13

That follow-up is complete for the current 12-lap practice command stream.
Build/replay r07 used the source race scene, kept its vehicle and native
physics parameters intact, and added the exact active practice course
MeshCollider extracted from the matching official simulator image. The OBJ
was imported as one mesh after removing only its three non-geometric `g`
group markers (vertex/face content unchanged); its SHA-256 is
`3c3d2a7c1e0e4d9b72900fd30c06e9cb151ecfb955c3edbff270f42cd60e2405`.
The source scene spawn and practice-capture initial pose agree within the
scene/capture frame transform, so this is a meaningful same-course replay.

Result: Unity 2022.3.52f1 completed all `2,994/2,994` captured command pairs
and `74,850/74,850` fixed steps at 1 ms, with exact steering/throttle command
match and zero course collisions. There was one missing 25 ms state sample in
the reference capture, explicitly handled by the comparator. The diagnostic
capture phase is -26 fixed steps and was selected by an independent timing
sweep, not by fitting state dynamics.

Timing-aware parity metrics are: body-velocity vector RMSE `0.002354 m/s`
(p95 `0.005591`); yaw-rate RMSE `0.0007583 rad/s`; body-angular-velocity
vector RMSE `0.003220 rad/s`; yaw-orientation RMSE `0.02109 rad`; and relative
position vector RMSE `0.09386 m` (p95 `0.18581 m`, max `0.23835 m`). Endpoint
relative-position error is `0.10486 m`. Per-lap local path-increment RMSE is
`0.00396–0.02206 m`, while lap-end increment error is `0.00458–0.03257 m`.
Thus the same-source plant reproduces instantaneous movement closely, but
small per-lap differences still accumulate. This is not evidence of
perfect/full-lap odometry and no production odom, MPC, or raceline change is
claimed.

The result directly supports a non-learning offline plant: when exact Unity
code and the matching engine build are available, execute that source and
native WheelCollider implementation rather than train a replacement for
simulator-defined behavior. The Unity project does not contain the native
WheelCollider solver source, so a stand-alone port cannot be called
apples-to-apples without separately reproducing and validating that engine
implementation. The preserved source player, 1 kHz capture, comparator output,
and build log are in
[`unity_source_practice_scene_command_replay_20261013_r07`](../../live_runs/unity_source_practice_scene_command_replay_20261013_r07/).

## Official practice settings and command-phase trial — 2026-10-13

The official practice image's `Data/globalgamemanagers` was extracted
read-only; the preserved file is
`live_runs/unity_source_practice_scene_command_replay_20261013_r08/official_assets/globalgamemanagers`
(SHA-256 `e2342faa929031f5e2864ffff207b5cccc89b1fe95d0dbade5bd69ae939a1a31`).
Its BuildSettings select the same race-scene asset path as the source project,
`Assets/Scenes/RoboRacer - Sim Racing.unity`. The serialized global settings
match source ProjectSettings: 1 ms fixed step, 100 ms maximum step, scale 1,
gravity `(0,-9.81,0)`, 0.01 m contact offset, bounce threshold 2, maximum
depenetration speed 10, sleep threshold 0.005, solver iterations 6/1,
automatic transform sync, and max angular speed 7 rad/s. The active official
practice scene resolves its controller component to `Assembly-CSharp.VehicleController`;
serialized rigid-body and four WheelCollider parameters match the source
runtime capture. This rules out a mismatch in these compared global and
vehicle settings as the explanation for the observed replay residual, though
it does not independently prove identical C# method bodies in the public
source and official binary.

A targeted 1 ms command-phase variant (`r08`) was built from the same source
scene, Unity engine version, and track, then replayed through the established
practice batch-mode launcher. It completed 2,994/2,994 commands and 74,851
fixed steps with exact command values and zero course collisions. Compared at
the same diagnostic capture phase as r07, it worsened position RMSE from
`0.09386` to `0.09561 m`, yaw RMSE from `0.02109` to `0.02201 rad`, body-velocity RMSE from
`0.002354` to `0.004254 m/s`, and yaw-rate RMSE from `0.000758` to
`0.001193 rad/s`. Therefore the one-step shift is rejected; r07's zero-phase
profile remains the best measured source replay. The phase control is an
offline diagnostic only and changes no physics/runtime settings.

Source inspection confirms the likely chronology boundary to examine next:
`Socket.OnBridge` assigns the throttle/steering setpoints; telemetry is queued
separately; the reported `CurrentThrottle` value is refreshed inside
`VehicleController.FixedUpdate`. The current replay injects setpoints at a
deterministic fixed-step phase rather than reproducing the Socket callback and
telemetry queue. This is evidence for testing exact event ordering, not proof
that it explains all accumulated pose error. No ML/RL dynamics substitute,
production odom/MPC update, or vehicle physics change was promoted. The r08
capture, result, profile, official settings asset, and build log are preserved
in [`unity_source_practice_scene_command_replay_20261013_r08`](../../live_runs/unity_source_practice_scene_command_replay_20261013_r08/).

## Independent exact replay — 2026-10-13

R09 repeats r07's zero-phase source build and full 2,994-command practice
profile from a separate process. It again completed `74,850/74,850` 1 ms
steps, with zero collisions. Numeric capture comparison is exactly equal
after CSV parsing across all 74,850 rows: simulation time, rigid-body position
and quaternion, body linear/angular velocity, throttle/steering commands, and
all recorded wheel values (RPM, steering angle, torque, grounded flag, slip,
and contact force). Every reported maximum absolute difference is exactly
zero.

Therefore the source-built Unity/engine plant itself is repeatable for this
initial scene and command sequence, and is a valid deterministic offline
evaluator without learned dynamics. The remaining `9.39 cm` global position
RMSE / `0.0211 rad` yaw RMSE against the practice capture is specifically a
cross-run parity gap—not nondeterminism between offline replays. It remains
unresolved; potential initial-state, command/event timing, and public-source
versus official-binary differences must be separated before claiming the
source code fully matches the official executable. No production odom/MPC or
physics parameter was modified. R09's preserved capture is in
[`unity_source_practice_scene_command_replay_20261013_r09_repeat_zero_phase`](../../live_runs/unity_source_practice_scene_command_replay_20261013_r09_repeat_zero_phase/).
