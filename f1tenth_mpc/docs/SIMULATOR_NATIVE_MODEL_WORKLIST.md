# Simulator-native prediction worklist

## Non-negotiable model boundary

The controlled object is the current AutoDRIVE Unity vehicle.  It is not a
physical F1TENTH car and its predictor must not acquire parameters merely
because they resemble conventional vehicle quantities.  The only authorities
for the nominal model are:

1. the current Unity `VehicleController.cs` command semantics;
2. the current scene/prefab serialized values and WheelCollider setup; and
3. blind, recursively-scored simulator responses.

Simulator truth, contact state, wheel state, collision state, and lap state
may be used offline to explain residuals.  They are never runtime inputs to
odometry, AMCL, Pure Pursuit, or MPC.  Unity physics, scenes, timing and
vehicle behaviour are read-only for this work.

The online acceptance target is a 30-stage, 40 Hz (0.75 s) prediction from
only causal rear encoder, IMU, LiDAR-localized pose, and command-history data.
The project command ceiling is 16 m/s.

## Reviewed suggestions

| Work item | Status | Decision / evidence required |
| --- | --- | --- |
| Score every candidate recursively at 0.10, 0.25, 0.50 and 0.75 s from legal runtime inputs | **Implemented** | `score_legal_n30.py` rejects non-legal origins, non-40 Hz cadence, duplicate origins, and any horizon outside N4/N10/N20/N30. Only legal-state N30 results can qualify a controller model. |
| Keep simulator truth offline-only | **Implemented** | The controller launch path and model scorer keep truth out of runtime consumers. |
| Separate state estimation from prediction | **Planned** | One causal state handoff will serve MPC; do not make `/odom`, AMCL and MPC independently invent incompatible hidden states. |
| Exact source steering target, limit and rate update | **Implemented in the MPC command path** | The fitted steering pole is removed. MPC limits target changes to the source's +/-0.5235988 rad and 3.2 rad/s physical update envelope. |
| Preserve left and right rear encoders separately | **Recorded; not yet used by the online predictor** | Use each encoder and gyro/geometry to construct causal rear-contact features. Do not average them before the ablation. |
| Add an IMU-derived longitudinal-motion state | **Planned ablation** | Retain only if blind N30 longitudinal error improves across held-out track data. |
| Add a causal drivetrain-memory state | **Planned ablation** | Test after the longitudinal-motion state; it must improve multi-step results, not one-step fit alone. |
| Add lateral/yaw residual states | **Planned ablation** | Start with zero/one state at a time. Retain a state only with a reproducible held-out N30 benefit. |
| Use source-shaped WheelCollider force curves | **Parity experiment required** | Current source curves are an anchor, but the native WheelCollider/contact solver and unobserved loads must be checked against diagnostics before a runtime surrogate is promoted. No Pacejka replacement is an authority. |
| Use high-rate wheel/contact diagnostics as an offline teacher | **Implemented policy; analysis remains** | Use them to find a causal observable proxy, never to add hidden runtime state. |
| Correct IMU lever-arm geometry in the estimator measurement model | **Needs source-to-runtime verification** | Add only after verifying the current Unity sensor transform and the estimator reference point. |
| Treat LiDAR localization as the global-pose anchor while odometry supplies local motion | **Existing architecture; live validation pending** | Improve the current AMCL/EKF contract only with fresh sensor-legal validation. |
| Small learned/local residual after a source-shaped baseline | **Deferred** | Consider only after the deterministic observable model passes ablation tests; validate recursively and bound it. |
| SynPF localization | **Rejected by project direction** | Do not investigate or implement. |
| Cartographer localization | **Rejected by project direction** | Do not investigate or implement. |
| Full four-wheel/suspension/RPM reconstruction as online MPC state | **Rejected for runtime** | Useful only to diagnose residual mechanisms offline; front-wheel and suspension states are not causal runtime measurements. |

## Cleanup acceptance checklist

- [x] Remove the active Pacejka/cornering-stiffness/friction-limit bicycle
      contract and direct Unity-acceleration actuator input from `f1tenth_mpc`.
- [x] Remove the separate offline `vehicle_plant` from the MPC build and tests.
- [x] Replace artefact-pinned manifests with one current-source simulator
      contract, independent of prior repository revisions and old reports.
- [x] Retain one MPC code path: BachelorProject solver plus the strictly
      necessary ROS adapter and a deliberately inhibited source-command
      baseline.
- [x] Keep the existing launchable Pure Pursuit controller and add a launchable
      MPC controller whose command authority is inhibited pending validation.
- [x] Name MPC's second decision variable `target_speed_rate` and always send
      zero through Ackermann's direct acceleration field; the established
      actuator interface remains the sole speed-to-throttle owner.
- [x] Remove obsolete generated fitting artefacts and old high-speed campaigns.
      Fresh 16 m/s-or-lower source-valid captures are required for every future
      ablation; historical reports are not input evidence.

## Promotion rule

An MPC model is not promoted because it resembles a car, has a low one-step
fit, or uses unavailable Unity state.  Promotion requires: source-contract
review, clean 40 Hz causal input timing, blind legal-state recursive scores at
the four horizons, controller watchdog/failsafe tests, and a collision-free
batchmode lap using the same legal runtime inputs.
