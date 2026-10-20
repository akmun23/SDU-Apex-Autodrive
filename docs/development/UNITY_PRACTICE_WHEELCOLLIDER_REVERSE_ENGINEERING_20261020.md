# Practice Unity/PhysX WheelCollider reverse-engineering

Updated: 2026-10-20

## Scope and binary identity

This audit targets the practice simulator's exact Unity player, not an Explore
vehicle or a generic PhysX build:

- `UnityPlayer.so` Build ID: `79f401cda3570218568e40b5331a0e68aa86f11c`
- `UnityPlayer.so` SHA-256: `9522ea13f26f71ddb439996643b382c6466c5cfafa8f0deb58517b8b82fe7b4f`
- Matching debug ELF: `external/unity-practice-image-analysis/UnityPlayer_s.debug`
- Debug ELF SHA-256: `ba315e1c4c01498960045e95770a1e45f2515e612d93653e635ac1dc2f83af31`
- Unity version reported by the player: 2022.3.52f1

The debug ELF has the same Build ID as the shipped player. Native symbols and
selected functions were inspected with `nm`, `objdump`, and targeted Ghidra
decompilation. The reproducible target list is
[`ExportWheelCollider.java`](../../tools/unity_reverse_engineering/ExportWheelCollider.java).
It imports the binary without whole-program analysis and exports only named
WheelCollider, manager, and PhysX update functions. The raw disassembly and
decompiler export are local analysis artifacts under
`~/.cache/sdu-apex-ghidra-12.1.4/`; they are not runtime dependencies.
The Unity wrapper is native and is not shipped as C++ source, but it is not an
opaque black box here: the exact matching debug ELF permits targeted
disassembly/decompilation. The underlying PhysX 4.1 source is also present in
the repository. This is enough to reconstruct deterministic vehicle setup and
update logic; it does not make every runtime scene contact or callback timing
observable from an ordinary bag.

## Reconstructed execution path

The practice player configures a native `PxVehicleNoDrive`. The relevant call
path is:

1. `VehiclesManager::VehicleConfig::UpdateVehicles` runs the batched wheel
   suspension raycasts with `VehicleWheelRaycastPreFilter`.
2. It creates the surface-to-tire friction pairs and calls
   `PxVehicleUpdates` for the vehicle.
3. `PxVehicleUpdate::updateNoDrive` calculates tire/suspension forces in
   configured internal substeps, updates wheel speeds, and returns a final
   chassis velocity to Unity's ordinary scene simulation.
4. The Unity/PhysX scene step applies its body integration and collision
   constraints after the vehicle update. The private scene contact solve is
   not replaced by the source plant.

Relevant exact-build symbols include:

| Function | UnityPlayer link address | What it establishes |
| --- | ---: | --- |
| `WheelCollider::AddWheelToExistingVehicle` | `0x15134b0` | Native wheel/suspension setup and the `setSubStepCount(128,5,1)` call |
| `VehicleComputeTireForce` | `0x1513d40` | The simulator's custom per-wheel tire callback |
| `WheelCollider::ComputeWheelCentreOfMassOffset` | `0x15130c0` | Wheel centre and tire/suspension force application offsets |
| `WheelCollider::SetSuspensionExpansionLimited` | `0x1513f70` | Maps the property to the native suspension flag |
| `WheelCollider::GetRpm` | `0x15151c0` | Reads PhysX wheel angular speed and converts it to RPM |
| `VehiclesManager::VehicleConfig::UpdateVehicles` | `0x1516e10` | Raycast batch and `PxVehicleUpdates` orchestration |
| `physx::processSuspTireWheels` | `0x1b28fd0` | Contact, suspension, slip, sticky-tire, and tire callback preparation |
| `physx::PxVehicleUpdate::updateNoDrive` | `0x1b32e20` | `PxVehicleNoDrive` substep/update order |

The exact native debug ELF corrects two earlier uncertainties:

- **Wheel substeps:** `AddWheelToExistingVehicle` passes `128.0f, 5, 1` to
  `setSubStepCount`. Thus the race-speed branch is five internal vehicle steps
  per Unity fixed step; the one-step branch begins at `|forward speed| >= 128`
  in the player's units. The old 5 m/s model threshold was wrong and is
  removed from production code. The separate public-PhysX 5 m/s value is only
  the threshold for low-speed *wheel pose-angle blending*.
- **Wheel encoders:** `WheelCollider::GetRpm` calls
  `PxVehicleWheelsDynData::getWheelRotationSpeed`. It reports native wheel
  angular speed converted to RPM; it does not read throttle and does not
  derive RPM from the accumulated visual wheel angle.

The ray-query geometry and ordering are source-confirmed, not guessed:

- `PxVehicleSuspensionRaycasts` forms the chassis query pose as
  `actor.globalPose * actor.CMassLocalPose`, after replacing the local COM
  rotation with identity. For each enabled wheel it rotates the native
  suspension-travel direction, transforms the wheel-center offset, and starts
  the ray at the wheel center shifted toward maximum compression by
  `radius + maxCompression`.
- Query length is `3*radius + maxCompression + maxDroop`; each ray requests
  position, normal, and UV, and the batch is executed before vehicle update.
  With the active `0.059 m` radius and `0.025 m` compression/droop this is
  `0.227 m` per wheel.
  The query includes static and dynamic actors and invokes Unity's
  `VehicleWheelRaycastPreFilter`, which applies the candidate collider and
  PhysicsManager ignore/layer checks.
- Tire/suspension processing rejects a missing/zero-distance hit, a ray whose
  start is already behind the reported hit plane, a hit with an invalid
  suspension-direction normal, and a plane intersection beyond max droop.
  For ray hits, the native `intersectRayPlane` computes `T=-(n·v+d)/(n·w)`,
  `jounce=(2*radius+maxCompression)-T`, and the wheel-bottom point from that
  same ray. The shared plant implements this intersection for one known plane.

So the remaining contact-query mismatch is more specific than “PhysX contact
solver”: production does not yet obtain per-wheel hit plane/normal/material
from the practice scene's collision mesh and reproduce the exact query filter
result. Once a hit plane is known, the ray-to-plane geometry itself is already
implemented. This distinction matters: changing suspension/tire constants
cannot correct a wrong mesh hit or surface normal.

## Native equations and settings now represented

### Wheel and command setup

`AddWheelToExistingVehicle` enables each wheel, installs the `128/5/1`
substep schedule, copies the serialized WheelCollider radius/mass/damping,
sets wheel moment of inertia to `0.5 * mass * radius^2`, installs a 0.1 m
width and native wheel data, sets suspension direction `(0,-1,0)`, computes
the COM-relative wheel and force application offsets, and points the tire
shader data at the WheelCollider's two friction curves. The active prefab
contains `radius=0.059 m`, `mass=0.109 kg`, `dampingRate=0.25`,
`spring=500 N/m`, `damper=100 N s/m`, and `suspensionDistance=0.05 m`.

The ordinary controller's CAWD command path is in
[`VehicleController.cs`](../../external/autodrive-unity-source/Assets/Scripts/VehicleController.cs):
nonzero positive throttle gives each wheel one quarter of motor torque;
exactly zero gives no drive torque and full configured brake torque to each
wheel. The WheelCollider setters forward those values to
`PxVehicleNoDrive::setDriveTorque` and `setBrakeTorque` without a second
conversion.

### Tire callback

The native callback copies and evaluates the serialized forward and sideways
`WheelFrictionCurve` data. For each curve it uses the signed slip and the
absolute-slip magnitude; the rising segment is
`extremumValue * (t + t^2 - t^3)`, the second segment is a smooth cubic
transition `extremumValue + (asymptoteValue - extremumValue) * (3t^2 - 2t^3)`,
and the response is constant at the asymptote beyond its slip coordinate.

The callback's combined-slip arithmetic is unusual but directly visible in
the practice-player machine code. It computes

```text
L = raw_forward / forward_extremum_value
S = raw_side / sideways_extremum_value
scale = max(1, sqrt(L*L + S*S))
Fx = filtered_tire_load * raw_forward * forward_stiffness / scale
Fy = -filtered_tire_load * raw_side * sideways_stiffness / scale
wheel_torque = -wheel_radius * Fx
aligning_moment = 0
```

This is the native callback's literal arithmetic, not a conventional friction
ellipse. Tire load scales both output forces, but does not enter either
normalized demand in the combined-slip denominator. Exact-build disassembly
shows `response_y / sideways_extremum_value` in the utilization term, while
the force output is separately `tireLoad * response_y * sideStiffness`.

**Correction, 2026-10-20:** the previous transcription incorrectly multiplied
the normalized lateral demand by filtered tire load. That extra factor was
not in the Unity callback and could suppress lateral force by a large factor
under load. The production plant and its direct callback-equation test have
now been corrected to the disassembled expression. The earlier replay scores
in this report predate this correction and must not be treated as scores of
the corrected source; see the runtime progress log for the fresh replay.

### Suspension flag default

The public PhysX `PxVehicleWheelsSimData` constructor initializes its flags to
zero. The exact `AddWheelToExistingVehicle` path does not set them. The
practice prefab and repository scripts do not call
`SetSuspensionExpansionLimited(true)`. The native setter passes its Boolean
directly as the full flags value, so the expansion-velocity limiter is off
for this configured vehicle; default suspension-force projection remains on.
The source plant documents this at its suspension-force implementation and
does not add the optional limiter.

### Wheel pose angle

Public `integrateNoDriveWheelRotationAngles` uses actual wheel angular speed
when drive or brake torque is active, and otherwise blends the rolling speed
with wheel angular speed below 5 m/s. It wraps each accumulated wheel pose
angle once at `+/-10*pi`. The shared plant now implements those exact
conditions and wrap. This is a native wheel-pose-state correction only:
`WheelCollider.GetRpm` reads the unblended wheel angular speed.

### Rigidbody inertia and wheel sprung-mass construction

The matching ELF also exposes the Unity wrapper and PhysX mass-property path;
this is no longer an unexplained inertia constant:

1. `Rigidbody::SetCenterOfMass` and
   `Rigidbody::SetAutomaticInertiaTensor` both reach
   `Rigidbody::UpdateMassDistribution`.
2. `UpdateMassDistribution` enumerates shapes attached to the actor, excludes
   trigger shapes, and calls
   `PxRigidBodyExt::setMassAndUpdateInertia` with the Rigidbody mass and either
   an explicit COM or a null COM pointer according to the Unity setting.
3. The exact PhysX 4.1 implementation is present in
   `external/physx-4.1-source/physx/source/physxextensions/src/ExtRigidBodyExt.cpp`.
   It includes only simulation shapes by default; gets each supported shape's
   mass properties (convex meshes use cooked-mesh `getMassInformation`);
   transforms and aggregates them in actor space; scales total density to the
   requested 3.47 kg; translates inertia to the specified COM or computes and
   recenters at the automatic COM; diagonalizes the tensor; then writes mass,
   principal inertia and COM pose to the native actor. Thus the vehicle's
   convex-collider geometry/cooking and local poses determine the tensor.
4. The active serialized Rigidbody has mass `3.47 kg`, drag `0.273`, angular
   drag `0.1`, `m_ImplicitTensor=1`, and `m_ImplicitCom=1`. The vehicle's
   `VehicleController.Start()` also assigns COM `(-0.00008, 0.06434,
   -0.00468)` in Unity coordinates. The plant's principal moments and
   principal-axis rotation are the values reported by the practice player in
   its capture; they are fixed-vehicle runtime properties in the plant, not
   parameters fitted to yaw error. The code's separate vehicle-substep and
   scene-constraint inverse-inertia representations correspond to the two
   integrations and must not be collapsed into one tensor.

The wheel-sprung-mass path is independently reconstructed from
`WheelCollider::UpdateVehicleSpringCMOffsetsAndMasses` and
`PxVehicleComputeSprungMasses`:

- For four enabled wheels, Unity computes each wheel-center offset from the
  chassis COM and calls the public PhysX constrained minimum-variance mass
  allocation with total mass `3.47 kg`, COM zero in the wheel-offset frame,
  and gravity axis `1` (Unity up).
- Using the exact scene wheel-center offsets, an independent solve of those
  same three constraints (sum of masses, longitudinal COM, lateral COM)
  gives `[0.817194701, 0.816018680, 0.918992382, 0.917794237] kg`. This
  matches the four production `kSprungMassKg` constants to float-rounding
  precision. The current production constants therefore have a direct,
  reproducible PhysX derivation; replacing them with equal quarter-masses
  would be incorrect.
- The Unity wrapper also has an equal-share branch if the computed result is
  invalid/negative. That branch is not taken for the active practice geometry.

The exporter now includes `Rigidbody::UpdateMassDistribution`, the Unity
center-of-mass and automatic-tensor setters, PhysX mass-property functions,
the wheel sprung-mass updater/resetters, sprung-mass helper, and the vehicle
raycast prefilter. This keeps the next assembly pass targeted and repeatable;
it does not invoke or link Unity/PhysX at runtime.

## Code changes in this working tree

- Corrected the production wheel-substep schedule to five steps below 128 m/s
  and one at/above it; added exact boundary tests.
- Removed filtered tire load from normalized lateral utilization. The Unity
  callback normalizes the raw friction-curve responses; filtered tire load
  scales the resulting force, but is not part of the normalized lateral term.
- Reconstructed wheel pose angle integration, including the source's low-speed
  free-roll blend and `+/-10*pi` wrap.
- Added focused Ghidra targets for wheel setup, sprung-mass geometry,
  suspension-limit flag, command setters, RPM getter, Rigidbody mass
  distribution, PhysX mass-property calculation, sprung-mass calculation, and
  vehicle raycast filtering.
- The same `unity_source_plant.cpp` is compiled into both MPC and production
  localization; these are not offline-only equations.

## What this does not establish

This work is a source-grounded implementation of the identified vehicle
equations, not a claim that the plant is a bit-exact replacement for the
practice simulator. The Unity player performs actual scene raycasts and then
uses PhysX's scene solver. The current reduced plant's contact query is an
analytic ground-plane intersection and it does not reproduce arbitrary
track-mesh hit selection, query filtering, dynamic hit actors, wall/obstacle
contacts, or the complete PhysX scene constraint solver. Those are explicit
remaining mismatches, not reasons to tune tire constants.

The exact player binary plus public PhysX source make further reverse
engineering practical, but any additional native-to-production port still
needs held-out practice replay metrics. Matching formulas alone does not prove
prediction accuracy. No practice simulation was launched for this audit, and
no new live accuracy claim is made here.

## Verification state

- Targeted Ghidra export completed against the matching debug ELF. The callback
  equation was independently checked in the Ghidra decompilation and x86-64
  register/dataflow. This caught and corrected the prior transcription error
  in `S`: tire load was incorrectly included in the normalized lateral term.
- The corrected equation is in the production-shared
  `f1tenth_mpc/src/unity_source_plant.cpp`; the focused callback equation test
  now asserts the native expression. `test_unity_source_plant` passes. The
  `mpc_core`, `mpc_controller_node`, and production `sensor_odometry_node`
  targets all compile with the correction.
- Replayed the two independent Oct 19 practice captures through the rebuilt
  source-plant evaluator using the recorded initial plant state and 25 ms
  command sequence. These are open-loop full-state rollouts; future state
  measurements are not fed into the plant. At 750 ms, `run/` has
  `u RMSE=0.00521 m/s`, `v=0.00038 m/s`, `yaw-rate=0.00300 rad/s`,
  `heading=0.00135 rad`, and `XY endpoint RMSE=0.00570 m`; `run_1/` has
  `u=0.00678 m/s`, `v=0.00069 m/s`, `yaw-rate=0.00558 rad/s`,
  `heading=0.00188 rad`, and `XY endpoint RMSE=0.00787 m`. Compared with the
  pre-correction `~1.381 m` XY RMSE on both bags, this is a repeatable,
  approximately 99.5% reduction. Both bags are below the requested 5 cm RMSE.
- Tail accounting is retained rather than hidden: `run/` has no endpoint or
  intermediate path sample above 5 cm. `run_1/` has one endpoint at `50.31 mm`
  and two intermediate path samples above 5 cm (maximum `53.7 mm`). The traced
  worst samples have plant-vs-GT-twist increment errors under 0.4 mm while the
  recorded GT pose-vs-twist increment differs by 2.9–3.8 cm in a single 25 ms
  interval. This is a real limitation of pose-based tail scoring in those
  samples, not evidence to add a force gain. Keep the strict max-tail result
  visible if the acceptance criterion is changed from RMSE to maximum error.
- No simulator was launched: this is a two-capture offline held-out replay, not
  a new online accuracy claim. The reduced plant still uses a known-plane
  analytic contact query and does not reproduce arbitrary mesh-hit selection,
  dynamic contacts, or the full scene constraint solver. Do not infer that
  the model is now a bit-exact Unity replacement or that every untested regime
  is solved. The next useful validation is a current-code practice run; if it
  disagrees, inspect the runtime stream and the exact scene contact query
  before altering this now source-correct callback equation.
