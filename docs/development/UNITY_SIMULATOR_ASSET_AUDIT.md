# Simulator physics assets and wheel-state observability

Updated: 2026-09-27

This records values read from the pinned explore simulator's serialized Unity
scene/vehicle components and measured from its ROS stream. Serialized settings
are not all runtime states: WheelCollider contact forces, dynamic normal
loads, suspension motion and the Rigidbody's implicit inertia tensor evolve
inside Unity and are not exported by the stock bridge.

## Confirmed vehicle settings

| Quantity | Serialized/guide value | Evidence and limitation |
| --- | ---: | --- |
| Rigidbody sprung mass | 3.470 kg | Active vehicle Rigidbody; guide agrees. |
| WheelCollider mass | 0.109 kg each | Four wheels; totals 0.436 kg unsprung mass. |
| WheelCollider radius | 0.0590 m | Same on all four active colliders and used by the rear encoders. |
| Controller `WheelRadius` field | 0.0325 m | Exact pinned-player field. The active `CAWD` branch does not use it for drive-torque distribution; it is used only by the alternate differential-drive branch. It is neither the collider radius nor an odometry gain. |
| Wheelbase / track | 0.324 / 0.236 m | Controller/guide values. Active collider centers span 0.330 m longitudinally and 0.236 m laterally; the 6 mm longitudinal difference remains unexplained. |
| COM from rear-axle ground frame | forward=0.15532 m, left≈0 m, up=0.01434 m | Exact controller field in Unity local axes is `(-0.00008, 0.06434, -0.00468)` m. Transforming Unity `(x lateral, y up, z forward)` and subtracting the rear-axle-ground origin (`y≈0.05`, `z=−0.16`) gives the guide-frame value. |
| Suspension | 500 N/m, 100 Ns/m, 0.050 m travel | Active WheelCollider spring, damper and suspension travel values. |
| Tire curve landmarks | Longitudinal: (0.15, 0.72), (0.25, 0.464); lateral: (0.01, 1.0), (0.10, 0.50) | Active WheelCollider friction curves after stiffness scaling; curve landmarks do not expose combined-slip coupling or dynamic normal loads. |
| Steering | ±0.5236 rad, 3.2 rad/s | Controller/guide limit and rate. |
| Drive | AWD | Guide and active vehicle controller configuration. |
| Rigidbody yaw inertia | Not serialized as a scalar | `m_ImplicitTensor=true`; Unity computes the tensor from attached colliders. Exact runtime tensor is not in the ROS packet. |

The AutoDRIVE guide documents the tire-slip/friction-curve and suspension
structure, but its ROS topic table exports only rear encoders, IMU, steering,
throttle and other listed sensors. It does not export individual contact
forces, suspension states or front-wheel rpm. Ground-truth IPS/raw simulator
odometry are restricted only during autonomous runtime and remain available
as offline development labels. See the [vehicle-dynamics section](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#132-vehicle-dynamics)
and [data-stream table](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#23-data-streams).

## What the encoders actually measure

The raw packet contains left/right rear encoder angles and tick counts. There
is no independent wheel-speed field. The ROS `JointState` publisher fills
position only; velocity and effort arrays are empty. Empirical calibration
matches the configured 16 PPR × 120 ratio, or 1,920 ticks/revolution
(305.5775 ticks/rad). Therefore rear angular velocity is estimated by
differentiating cumulative encoder angle:

```text
omega_rear ~= (theta(t) - theta(t - T)) / T
```

On `openplane_transition_dwell_20260926`, each encoder ran at 39.971 Hz over
7,989 samples. A 100 ms slope gave median rates of 69.95 rad/s (left) and
69.93 rad/s (right), equivalent to about 4.13 m/s at 0.059 m radius. A
25 ms slope gave similar central rate but noisier slip estimates; 100–200 ms
windows were more stable but trade response lag for reduced quantization/noise.
The all-moving-sample absolute-slip p95 was about 0.15 and includes steering
and speed transitions, so it is not a steady tire parameter. It only
characterizes the derivative tradeoff in this capture.

These are actual rear wheel rates and are competition-permitted encoder
measurements. They are not vehicle speed: rear longitudinal slip must be
estimated against body speed from the allowed-input observer. No front
encoder is published, so front circumferential speed and front longitudinal
slip cannot be directly measured from the current public stream.

## Hidden states and next measurement

The current evidence distinguishes three categories:

1. **Known scene parameters:** mass, collider radius, suspension spring and
   damper, curve landmarks, steering geometry and COM offset above.
2. **Recoverable rear sensor state:** rear wheel angular rate from encoder
   angle history, subject to the 40 Hz window/noise tradeoff.
3. **Unexported simulator state:** front wheel rpm; per-wheel forward/sideways
   contact slip and force; normal loads; suspension compression/velocity;
   wheel torques; and the Rigidbody's computed inertia tensor.

Ground-truth body pose/velocity alone constrains aggregate planar forces and
moments, not a unique allocation to four tires. Fitting separate per-wheel
curves from those labels without wheel loads/slips/contact forces is therefore
underdetermined. If the development simulator can be instrumented without
changing its physics, record the Unity WheelCollider rpm, forwardSlip,
sidewaysSlip and ground-hit force, suspension state, motor torque and
Rigidbody inertia tensor as **offline labels only**. The deployed estimator
and MPC must continue to consume only permitted sensors and team-derived
state. If no source/build pair can be shown equivalent to the pinned player,
keep the front-wheel quantities latent and target a statistically validated
body-motion predictor instead of claiming they were recovered.

## Source-level observability audit — 2026-09-27, amended 2026-10-11

The public AutoDRIVE Unity source branch (`AutoDRIVE-Simulator`, commit
`75b7df5216524a3c6b3150d698139190c6c44f8f`, dated 2026-09-21) identifies Unity
`2022.3.52f1`, the same editor version embedded in the pinned player. The
public F1TENTH prefab serializes four 0.059 m, 0.109 kg colliders; 500 N/m
springs; 100 Ns/m dampers; 0.050 m travel; the friction-curve values above;
324/236 mm wheelbase/track; 85.6 motor torque; a 30 degree steering limit;
and 183.346 degree/s steering rate. These prefab values are source-project
evidence. The matching Unity editor version does not establish that this
commit or scene is the source used to build the pinned player.

Upstream refs were checked on 2026-10-11. The repository's top-level
`AutoDRIVE` branch contains project documentation/images but no Unity project;
the available simulator assets remain on `AutoDRIVE-Simulator`. No published
tree inspected here contains the pinned binary's `RoboRacer - Sim Racing -
Copy.unity`, so exact source/build identity remains unverified.

The 0.0325 m controller `WheelRadius` is used in the source project's
`ExtendedDifferentialDrive` torque conversion (`motorTorque = (...) /
(2*WheelRadius)`). Exact pinned-player disassembly confirms the active
`driveType=CAWD` path instead divides total drive torque among four wheels;
the 0.0325 m field is not used by that path. Separately, the 0.059 m
`WheelCollider.radius` is the geometric radius used by `WheelEncoder` when
integrating RPM into ticks/angle and computing its local speed properties.
Neither radius is an odometry gain; the encoder publisher sends cumulative
ticks and angle.

This strongly matches the car definition, but does not prove identical
whole-player provenance: the pinned binary's BuildSettings names
`RoboRacer - Sim Racing - Copy.unity`; the public source branch names the open
scene `RoboRacer - Sim Racing - Explore.unity`. In the pinned binary, the
flat `Floor` is active and all serialized racetrack objects are inactive.
Treat the public source as an instrumentation candidate and require an
identical-command A/B motion comparison before using its labels. Do not
silently substitute a rebuilt player for the pinned simulator.

Correction — the enum in `Assets/Scripts/VehicleController.cs` is ordered
`IRWD, IFWD, IAWD, CRWD, CFWD, CAWD, SkidSteer`; therefore the prefab's
`driveType: 5` is CAWD, not SkidSteer (`6`), and `steerType: 0` is
front-wheel steering. In this source branch, `FixedUpdate` assigns
`CurrentSteeringAngle` from `-SteeringAngle * PI/180` and normalized
`CurrentThrottle` from the previous `DriveTorque`, immediately before calling
`Steer()` and `Drive()` for the next physics tick. Thus the steering feedback
is physical radians, while throttle feedback is normalized and both reflect
the preceding 1 ms controller state. This is consistent with the observed
packet units and one-physics-tick actuator timing. It does not prove that the
pinned player uses this exact public prefab/controller build; preserve the
source/build-equivalence caveat, but do not use the former alleged
SkidSteer-vs-radian mismatch as a model explanation.

The exact-player `level0` extraction contains `Fixed Timestep: 0.001` s and
`Maximum Allowed Timestep: 0.1` s. This establishes the serialized physics
setting in the pinned build; a runtime script changing it after load has not
been found. The 40 Hz ROS output is a separate sensor/bridge rate, not the
physics integration rate.

The source confirms an actionable telemetry gap. `WheelEncoder.RPM` already
reads `WheelCollider.rpm`, but the bridge serializes only cumulative encoder
ticks and angles. A development-only source build can sample all four wheels'
RPM, Ackermann steer angle, `GetGroundHit` status, forward and sideways slip,
contact-force magnitude, contact point/normal and contact-frame directions,
wheel pose, and motor/brake torque. The repository-side candidate recorder is
[`OpenPlaneWheelDynamicsCapture.cs`](../../tools/unity/OpenPlaneWheelDynamicsCapture.cs).
It also writes configured spring/damper, suspension travel, force-application
point and wheel damping. `GetWorldPose` records the ground-adjusted wheel
pose; this is a suspension/steering kinematic observation, not a direct
measurement of suspension force or independently exposed compression. These
are read-only observations; the instrumentation must not write to a wheel,
body, or actuator. Its 300,000-sample buffer is allocated only when the
development capture environment variable is set, so normal/non-capture runs
do not allocate the recorder buffer. Unity's public `WheelHit` API exposes
slips, contact geometry and one force magnitude, not independent per-wheel
longitudinal/lateral tire-force components. Those components therefore remain
aggregate-force inferences unless deeper simulator/PhysX instrumentation is
obtained. This recorder has not yet been integrated into or compiled with the
simulator source project, and its measurements cannot be treated as labels for
the pinned player until same-command A/B motion and 40 Hz stream equivalence
are demonstrated. See the [AutoDRIVE source branch](https://github.com/Tinker-Twins/AutoDRIVE/tree/AutoDRIVE-Simulator),
[Unity `WheelHit`](https://docs.unity3d.com/2022.3/ScriptReference/WheelHit.html),
and [Unity `WheelCollider.rpm`](https://docs.unity3d.com/2022.3/ScriptReference/WheelCollider-rpm.html).

The public project contains a `Suspension` script that can recompute spring,
damper and force-application settings, but that component is not referenced by
the public Explore scene or its RoboRacer vehicle prefab. It is not evidence
that this car's 500/100 settings are dynamically overwritten. Confirm component
attachments against the exact built scene if that scene is rebuilt.

The guide equations specify a model structure, not a complete identified
plant. The actual per-wheel slip must be computed in each steered wheel frame.
For wheel position `(x_i,y_i)` relative to the COM and yaw rate `r`, first use
`v_xi = u - r*y_i`, `v_yi = v + r*x_i`, then rotate by that wheel's Ackermann
angle `delta_i`; only those local velocities belong in that wheel's slip
equations. The configured WheelFrictionCurve landmarks and stiffness are
known and define the documented two-piece spline; they are not the missing
parameters. What is missing from ordinary ROS bags is each wheel's
time-varying local slip, contact/load state and resulting tire-force vector.
`WheelHit` exposes only a contact-force magnitude, not the separate
longitudinal/lateral force components. Ground-truth body motion therefore gives
aggregate force/moment constraints, not a unique decomposition into four
tires; fitting another cubic curve to body pose alone cannot recover that
decomposition. Unity's native WheelCollider/PhysX integration still has to be
represented accurately in the MPC predictor.

Instrumentation acceptance must be set before running it: the source-built
development player must produce the same public motion as the pinned image
under the same open-plane command trace, within the existing repeated-run
envelope, while preserving the current 40 Hz public stream quality. If it
doesn't, reject its internal measurements as labels for the pinned simulator.
Keep this telemetry strictly in the development/offline pipeline; it must not
become a competition-controller input.

The existing source-player runtime dump recorded the `inertiaTensor` principal
moments but omitted `inertiaTensorRotation`. Those values alone do not identify
body-axis yaw inertia: rotate the principal diagonal tensor into the
Rigidbody frame using Unity's principal-frame quaternion. The updated
development recorder writes both. Its 300,000-row ring buffer is bounded for
the 260 s continuous race-domain schedule; output is expected to be about
200 MB based on the previous 60,000-row capture. Per-sample Unix-UTC-mapped
timestamps and steering/body-speed alignment checks are required before
joining wheel data to a ROS bag. This source update remains unverified in
Unity until the simulator source project is available.

## Exact pinned-player configuration extraction — 2026-10-12

This section supersedes the earlier source-only caveats for static vehicle
configuration. It was decoded from the exact competition image
`autodriveecosystem/autodrive_roboracer_sim:2026-icra-compete`, digest
`sha256:0511fb6d6db7f31ec0d20787c0498eb05f3bd45ae6b9683165e8680b61e9b159`.
The retained hashes and serialized `VehicleController` record are in
[`exact_vehicle_controller_config.json`](../../live_runs/unity_exact_binary_audit_20261012/exact_vehicle_controller_config.json);
the repeatable decoder is
[`extract_exact_vehicle_config.py`](../../live_runs/unity_exact_binary_audit_20261012/tools/extract_exact_vehicle_config.py).

| Exact pinned setting | Value | Model consequence |
| --- | ---: | --- |
| Controller modes | CAWD, CAWB, front Ackermann; autonomous | Determines active steering, drive and brake branches. |
| Motor torque field / throttle limit | 428 N·m / 1 | For nonzero throttle, the active CAWD branch distributes total commanded torque equally: 107 N·m per wheel at full command. This is a wheel torque request, not tire force. |
| Zero-throttle brake | 428 N·m per wheel | The exact zero-input branch applies common all-wheel braking; zero throttle is not free coasting. |
| Negative throttle | Negative motor torque in CAWD | It does not enter either independent-brake branch for this brake type. |
| Steering | 30° limit; 183.345993°/s rate | Steering is rate limited. Actual steering feedback is the causal model input when available. |
| Ackermann geometry | 324 mm wheelbase; 236 mm track | Used to calculate individual front-wheel steer angles. |
| Collider centers | Front z=+0.17 m, rear z=−0.16 m; x=±0.118 m (Unity local) | 330 mm center spacing and 236 mm track for force moment arms; center spacing differs by 6 mm from the Ackermann wheelbase. |
| Controller radius | 32.5 mm | Not used in the active CAWD torque split; do not use as a wheel-speed scale. |
| WheelCollider radius/mass | 59 mm / 0.109 kg each | Geometry and serialized wheel mass; effective rotational inertia is not separately serialized. |
| Rigidbody mass/drag/angular drag | 3.47 kg / 0.273 / 0.1 | Sprung rigid-body contribution; four wheel masses add 0.436 kg (guide total mass 3.906 kg). |
| Suspension | 500 N/m, 100 Ns/m; travel 50 mm; target 0.5 | Settings known; per-wheel dynamic compression/load is not in compliant ROS data. |
| Longitudinal friction | slips 0.15/0.25; values 0.9/0.58; stiffness 0.8 | Exact curve landmarks. Do not interpret these directly as constant-μ force limits without identifying Unity's load/force scaling. |
| Lateral friction | slips 0.01/0.10; values 1.0/0.50; stiffness 1.0 | Separate exact curve landmarks. |
| Fixed physics step | 1 ms; max allowed step 100 ms | Serialized pinned-build setting, distinct from 25 ms sensor/bridge samples. |

The exact raw COM is Unity-local `(-0.00008, 0.06434, -0.00468)` m.
Transforming to the guide's rear-axle frame and ground-height origin gives
forward 0.15532 m, lateral approximately zero, and 0.01434 m above the ground
reference. Using Unity-local axes as vehicle-frame axes without this transform
would put the COM in the wrong place.

The public prefab defaults differ from the pinned player (`MotorTorque=85.6`,
`DrivingMode=0`), but the active `RoboRacer - Sim Racing - Explore` scene
overrides those fields to `428` and autonomous mode. These scene overrides
match the exact pinned controller values. Therefore source-built runs of that
Explore scene are not invalidated by the prefab defaults; prior runs still
need their build scene and player provenance checked before their labels are
accepted. The Explore scene also overrides the same forward-friction values
on one wheel that are already serialized in the prefab. Targeted native
disassembly and IL2CPP metadata confirmed the active controller branches;
full Ghidra auto-analysis was stopped after sustained disassembly failures
and no useful pseudocode output. The native WheelCollider force solver is
still not recovered, so the documented physics must be identified against
instrumented captures rather than claimed as an exact PhysX reimplementation.

## Slip-sign and low-speed convention to resolve

The guide's longitudinal definition is `Sx=(R*omega-vx)/vx`; with a
conventional positive wheel-spin sign it is positive when the driven wheel
surface outruns the road. Unity's API documentation says
[`WheelHit.forwardSlip`](https://docs.unity3d.com/2022.3/ScriptReference/WheelHit-forwardSlip.html)
is negative under acceleration and positive under braking. This may be only a
wheel-axis/sign-convention difference, but the two values must not be equated
or sign-flipped by assumption. The first instrumented check should compare
raw RPM and `WheelHit.forwardSlip` during straight positive throttle, coast,
and braking, then repeat under reverse. The guide's normalized slip ratios
also divide by longitudinal speed, so a separate, measured low-speed
representation is required around zero rather than direct division.
