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
| Controller `WheelRadius` field | 0.0325 m | A separate VehicleController field, not the collider radius. Its code usage has not been resolved; do not use it as an odometry gain. |
| Wheelbase / track | 0.324 / 0.236 m | Controller/guide values. Active collider centers span 0.330 m longitudinally and 0.236 m laterally; the 6 mm longitudinal difference remains unexplained. |
| COM from rear axle | x=0.15532 m, z=0.01434 m | Reconstructed from the active scene component transforms; agrees with the guide. |
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

## Source-level observability audit — 2026-09-27

The public AutoDRIVE Unity source branch identifies Unity `2022.3.52f1`, the
same editor version embedded in the pinned player. Its RoboRacer/F1TENTH
vehicle prefab matches the loaded player's wheel configuration: four 0.059 m,
0.109 kg colliders; 500 N/m springs; 100 Ns/m dampers; 0.050 m travel; and the
same longitudinal/lateral friction-curve landmarks and stiffnesses. Its
vehicle-controller fields also match the known 324/236 mm wheelbase/track,
85.6 motor-torque setting, 30 degree steering limit, and 183.346 degree/s
steering rate. The controller's separate `WheelRadius=0.0325` remains distinct
from the physical 0.059 m collider radius.

This strongly matches the car definition, but does not prove identical
whole-player provenance: the pinned binary's BuildSettings names
`RoboRacer - Sim Racing - Copy.unity`; the public source branch names the open
scene `RoboRacer - Sim Racing - Explore.unity`. In the pinned binary, the
flat `Floor` is active and all serialized racetrack objects are inactive.
Treat the public source as an instrumentation candidate and require an
identical-command A/B motion comparison before using its labels. Do not
silently substitute a rebuilt player for the pinned simulator.

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
body, or actuator. Its 180,000-sample buffer is allocated only when the
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
development recorder writes both. Its 180,000-row ring buffer is bounded for
the 153 s maximum excitation schedule; estimated CSV output is under 150 MiB
per run based on the previous 60,000-row capture. Per-sample Unix-UTC-mapped
timestamps and steering/body-speed alignment checks are required before
joining wheel data to a ROS bag. This source update remains unverified in
Unity until the simulator source project is available.
