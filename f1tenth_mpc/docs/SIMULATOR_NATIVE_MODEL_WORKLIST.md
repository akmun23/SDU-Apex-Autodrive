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

The online model remains a 30-stage, 40 Hz (0.75 s) predictor using only
causal rear encoder, IMU, LiDAR-localized pose, and command-history data. For
development acceptance, recursively scored N5 (0.125 s) is the primary
control-relevant metric because the controller replans at 40 Hz; N4/N10/N20
and N30 remain required diagnostics, and N30 is still the full-horizon check.
Report longitudinal and lateral errors separately in the truth-heading frame;
retain Euclidean position error only as a secondary summary. The project
command ceiling is 16 m/s.

## Test-to-model utilization rule

No run is started without recording (1) the specific model, odometry,
localization, timing, or controller question; (2) the legal runtime inputs the
result is meant to improve; (3) the primary score and a held-out rejection /
promotion rule; and (4) the exact implementation or next-test decision that
each possible result will trigger. After a run, its data must be scored and
either used to make a bounded code/model change or to reject that change with
evidence. Do not repeat a collection when the expected result cannot change
the implementation decision. Reuse accepted traces first; collect more only
when a named excitation, coverage gap, or independent holdout is missing.

For recursive prediction replay, legal initial states and planned control
sequences are allowed inputs. Future IMU, encoder, pose, simulator truth, or
other future measurements are not model inputs; truth is only an offline
target. Every candidate score must report signed/absolute longitudinal and
lateral position error plus yaw error at N5, with N4/N10/N20/N30 as required
diagnostics. A fit that only improves one-step or training error is rejected.

## Source-time acceptance policy — 2026-09-17

40 Hz is the Bridge request target, not a timestamp to assign to every
response. Unity reports the `Time.fixedTimeAsDouble` and physics step at which
each telemetry state is assembled. The bridge preserves that source-time
interval in ROS headers; it does not replace it with 25 ms or infer it from
host arrival spacing. It accepts increasing source intervals from 1 ms (one
Unity fixed step) up to the observer's 250 ms integration bound, provided
physics-step and telemetry sequences also advance. Odometry uses the measured
interval and marks intervals above 35 ms non-nominal; it never stretches them
to 25 ms. Host arrival spacing is independent and may bunch; its current
bounded transport/response window is 150 ms, with seven outstanding request
slots. Thus a contiguous 77 ms host gap is accepted when its source metadata
is valid, while a source gap above 250 ms or a host gap/response above 150 ms
still fails closed. The 40 Hz figure is the request target, not a guarantee for
every sample: report source-time and arrival-time distributions separately.
Earlier 15–35 ms bridge rejection notes below are historical and superseded.
The MPC node has a separate 75 ms source-header-age stop gate. The latest
batchmode attempt delivered regular 40 Hz messages but their headers appeared
about 78–80 ms old to the node; reconcile that clock/transport offset before a
live MPC acceptance run rather than weakening the gate without evidence.

## Reviewed suggestions

| Work item | Status | Decision / evidence required |
| --- | --- | --- |
| Score each candidate recursively from legal runtime inputs | **Implemented; priority updated 2026-09-17** | `score_legal_n30.py` now requires N4/N5/N10/N20/N30, reports truth-heading longitudinal/lateral errors, and identifies N5 (0.125 s) as primary. N30 remains a required 0.75 s diagnostic; no candidate is promoted by a single horizon. |
| Keep simulator truth offline-only | **Implemented** | The controller launch path and model scorer keep truth out of runtime consumers. |
| Keep simulator collision telemetry out of racing/model-ID control | **Implemented; live checked 2026-09-17** | Racing and model-ID launch paths disable both collision reset and terminal stop. The live actuator had both parameters false and `/collision_count` had zero subscribers. Mapping retains its explicit diagnostic abort. |
| Separate state estimation from prediction | **Planned** | One causal state handoff will serve MPC; do not make `/odom`, AMCL and MPC independently invent incompatible hidden states. |
| Exact source steering target, limit and rate update | **Implemented in the MPC command path** | The fitted steering pole is removed. MPC limits target changes to the source's +/-0.5235988 rad and 3.2 rad/s physical update envelope. |
| Preserve left and right rear encoders separately | **Recorded; offline side ablation now available** | `tools/model_id/analyze_encoder_sides.py` compares left-only, right-only and paired-mean speed with the deployed radius/window/speed calibration. It reports per-side freshness and brake/turn strata. No runtime split is promoted until a matched holdout shows repeatable benefit without turn-direction bias. |
| Stratify observer error by powered drive versus full braking | **Offline observer replay complete; no brake calibration promoted** | The 11,921-packet track trace has 1,157 moving zero-throttle/full-brake samples (160 straight, 997 turning). Direct COM body-forward speed is a cleaner speed reference than finite-differencing the GPS point: full-brake p95 error is 0.158/0.130 m/s (straight/turn) against COM, versus 0.242/0.218 m/s against pose differencing. A chronological 60/40 replay split found no transferable improvement: the best training candidate (`decel_ax_scale=0.95`, `offset=-0.15`) lowered training p95 from 0.154 to 0.123 m/s but worsened held-out p95 from 0.116 to 0.155 m/s. The independent short trace moved slightly in the candidate's favor, but only 68 brake samples were available and MAE changed by just 0.001 m/s. Keep the deployed 1.005/0.020 coefficients; obtain an isolated speed-stratified brake holdout before adding a brake mode. |
| Gate odometry drive/brake response with causal controller history | **Planned; candidate data support a focused test** | The sensor observer currently consumes encoders and IMU only. Any mode input must come from legal controller/actuator command history, be aligned to the observed one-packet lag, and be scored on held-out live data; never use simulator state or bridge truth at runtime. |
| Fit source-command longitudinal speed response for MPC | **Fitted; recursively validated offline on a held-out PP trace 2026-09-18; live MPC remains unverified** | The MPC carries its actuator target as a state and predicts body-speed response from legal target-speed history. A capped batchmode attempt was stopped at zero command because valid 40 Hz source messages failed the 75 ms age gate; keep command authority disabled until the timestamp handoff is reconciled and a closed-loop run passes. |
| Add an IMU-derived longitudinal-motion state | **Planned ablation** | Retain only if blind N5 longitudinal error improves across held-out track data without an unacceptable N30 regression. |
| Add a causal drivetrain-memory state | **Planned ablation** | Test after the longitudinal-motion state; it must improve multi-step results, not one-step fit alone. |
| Add lateral/yaw residual states | **Planned ablation** | Start with zero/one state at a time. Retain only with reproducible held-out N5 benefit and no unacceptable N30 regression. |
| Use source-shaped WheelCollider force curves | **Parity experiment required** | Current source curves are an anchor, but the native WheelCollider/contact solver and unobserved loads must be checked against diagnostics before a runtime surrogate is promoted. No Pacejka replacement is an authority. |
| Use high-rate wheel/contact diagnostics as an offline teacher | **Implemented policy; analysis remains** | Use them to find a causal observable proxy, never to add hidden runtime state. |
| Correct IMU lever-arm geometry in the estimator measurement model | **Implemented and source/runtime verified 2026-09-17** | Keep the 0.08 m IMU TF mount separate from the 0.15532 m Rigidbody-COM-to-rear-axle acceleration reference. Candidate runtime parameters were checked live; turn lateral velocity is translated to the rear-axle pose point. |
| Treat LiDAR localization as the global-pose anchor while odometry supplies local motion | **Existing architecture; 2026-09-17 test failed precision target** | The sensor-only partial run accepted local scan corrections, but `/current_map_pose` position p95 was 1.221 m and track-frame lateral error p95 was 0.537 m. Keep the architecture; do not call its current tuning validated. |
| Lower the local AMCL cluster-weight threshold from 0.90 to 0.70 | **Tested candidate; not promoted** | The fresh PP capture retained 2,500 particles; the winning-cluster weight was high (median 0.973, p95 0.998), yet `/current_map_pose` position error p95 was 0.107 m. This does not identify cluster weight as the remaining limiting error; require a matched threshold ablation before further tuning. |
| Split localization trajectory residuals into longitudinal and lateral components | **Implemented in offline reports 2026-09-17** | `/odom`, `/ekf_odom`, AMCL and current map pose now report signed and absolute truth-heading components per sample and by motion regime. Euclidean distance remains secondary; EKF covariance changes cannot correct its mean trajectory while its mean follows `/odom`. |
| Model the simulator's zero-throttle braking branch | **Mode semantics recorded; force mapping not yet resolved** | Zero throttle is treated as full brake (428 N per the user-provided simulator value), never as coast-down. The checked prefab/controller serializes 85.6 N·m `MotorTorque` per wheel and exposes no literal 428 N field, so reconcile how that effective force is produced before using it as a predictor coefficient. Do not insert 428 N as a constant deceleration; the IMU/encoder replay found no held-out gain from retuning the current deceleration affine term. |
| Use AMCL to correct along-track odometry drift | **Existing correction measured; gain change not promoted** | The long trace has raceline along-track p95 0.0862 m and cross-track p95 0.0390 m. An immediate one-scan doubled-along-gain counterfactual improves along p95 only slightly (0.0864 to 0.0827 m); a separate short trace moves in the wrong direction. Require a matched live A/B before changing the deployed gain (0.25); do not tune EKF covariance as a substitute for mean-pose correction. |
| Small learned/local residual after a source-shaped baseline | **Deferred** | Consider only after the deterministic observable model passes ablation tests; validate recursively and bound it. |
| SynPF localization | **Rejected by project direction** | Do not investigate or implement. |
| Cartographer localization | **Rejected by project direction** | Do not investigate or implement. |
| Full four-wheel/suspension/RPM reconstruction as online MPC state | **Rejected for runtime** | Useful only to diagnose residual mechanisms offline; front-wheel and suspension states are not causal runtime measurements. |

## Earlier live experiment sequence (superseded by the MPC rewrite)

The straight open-scene braking sweep at 12, 14, and 15.3 m/s is complete; do
not repeat those points without a new coefficient decision to test. The AMCL
fast along-track gain A/B is also complete (2026-09-18); gain `1.0` was not
promoted because cross-track localization and actual vehicle cross-track p95
regressed. Do not repeat that A/B unchanged.

The planned recursive speed-response score on the two accepted captures is
complete; see **MPC speed-response replay and focus transition — 2026-09-18**
below. Do not recollect those traces or repeat the completed AMCL gain A/B.
This experiment order was current before the full MPC rewrite and is now
superseded by the ordered phases in the 2026-09-18 handoff. The active next
gate is a non-commanding MPC shadow run alongside Pure Pursuit, in batchmode;
do not enable MPC authority or change localization/model parameters for that
run. Keep simulator truth offline-only and do not edit Unity physics, scene
geometry, sensor behavior, or timing.

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

## Historical Riccati and actuator-path recode — 2026-09-18

This section records the implementation snapshot from that date. The active
controller is now the nine-state RTI implementation described above; the
superseded 10-state controller and its compatibility-only files have been
removed from the repository.

The production core uses a fixed 30-stage/25 ms horizon. The augmented state
order is `[e_y, e_psi, u, v, yaw_rate, target_speed, commanded_steering,
previous_steering_rate, previous_target_speed_rate]`.
The two optimizer inputs are steering rate and target-speed slew.

The Riccati backward/forward pass now evaluates full matrices for every active
dimension, without the former hard-coded dense-eight/two-tail row assumption.
With affine dynamics `x+ = A x + B u + d`, diagonal state/control quadratics
`Q/R`, linear terms `q/r`, and cross-cost `N`, each backward step uses
`M=B'P`, `S=R+B'PB`, `G=B'PA+N'`, `K=-S^-1G`, and the affine gradient shift
`p_bar=p+Pd`. The value updates are `P=Q+A'PA+G'K` and
`p=q+A'p_bar+G'k`; the forward rollout applies every row and column of `A`,
`B`, and `d`. The 2x2 control Hessian must be positive definite; singular or
indefinite cases now fail explicitly instead of silently dropping
off-diagonal terms.

In that superseded ten-state snapshot, all augmented states had explicit
nonzero default costs at stage and terminal nodes. The active controller is
the nine-state implementation described above; its weights are declared as
ROS parameters in `config/mpc_autodrive.yaml` and still need driving
validation.

Validation in the Humble workspace container:

- `riccati_solver_test` compares a dense 10-state, 2-input affine/cross-cost
  problem against an independently condensed QP solved by pivoted Gaussian
  elimination. Optimized controls and propagated states agree within
  `2e-4`.
- The same test exercises state-box and control-box ADMM projections for the
  10-state problem and reaches the configured residual tolerance.
- `vehicle_model_test` covers the authoritative seven-state source-command
  stage, including its clipping, braking, and constant-turn branches.
- All three package tests pass. A 500-sample N30 core benchmark (no ROS/DDS)
  measured p50/p95/p99/max of `0.110/0.155/0.163/0.165 ms`; none of the solves
  hit the iteration limit (maximum 8 of 50 iterations).
- The ROS adapter emits steering angle plus target speed in
  `AckermannDriveStamped`; it integrates MPC's speed-slew output using measured
  source dt and records the actually applied, ceiling-clamped slew for the next
  solve. It leaves direct throttle/brake to the established actuator
  interface, which ignores the zero acceleration metadata.

This is core/math and message-contract validation, not track acceptance. MPC
remains disabled by default. The last capped batchmode run did not move because
the source-age check rejected delayed headers despite regular packet cadence.
Also, lateral velocity is an explicit state but is still held constant by the
current vehicle model; this recode does not claim its lateral dynamics are
solved.

## Latest validation snapshot — 2026-09-17

Run `sdu_apex_autodrive/artifacts/simulator_trace/pp_local_cluster070_sensor_only_4mps_20260917` is a failed, partial diagnostic capture, not model-promotion evidence. The ROS Humble build passed and package tests reported 48 tests, 0 failures, 14 skipped. The batchmode run used Pure Pursuit with a 4 m/s command cap; observed speed peaked at 2.44 m/s.

- The live runtime boundary was verified: collision reset/stop parameters were
  both false and `/collision_count` had zero subscribers.
- 1,065 consecutive bridge packets were accepted before the fault. Source
  interval mean was 24.993 ms (p95 26.001 ms, max 28.000 ms); there were no
  packet-sequence gaps and command lag was one packet.
- The next packet advanced simulator source time by 37.004 ms
  (telemetry sequence 1088 to 1089). The strict bridge timing watchdog latched
  neutral. This is a source-cadence fault despite consecutive packet sequence
  numbers; the run must not be treated as a complete 40 Hz validation.
- The two accepted responses immediately before the fault show an emerging
  scheduling/response stall: request periods were 25.0 and 29.1 ms, arrival
  periods 31.7 and 35.3 ms, and request-to-response times 9.9 and 16.1 ms.
  Render-frame increments fell from the usual roughly 50--78 per packet to 17
  and then 3. This localizes the onset to the Unity/host/Socket.IO update path,
  but does not yet distinguish which part caused it.
- Offline-only scoring of the accepted prefix found `/current_map_pose`
  position error p95 1.221 m, cross-track estimator error p95 0.537 m, and
  along-track error p95 1.219 m. Simulator-truth distance from the vehicle to
  the raceline was p95 0.917 m. `/odom` position p95 grew from 0.043 m in the
  first 5 s to 1.183 m during seconds 20–25. AMCL accepted 94.7% of updates,
  but did not materially outperform odometry in this prefix.
- A separate control symptom occurs for 13 consecutive accepted packets over
  0.30 s: `/cmd/speed` requested 1.126--1.179 m/s, while offline simulator
  truth speed was at most 0.059 m/s and the applied throttle remained zero.
  `/odom` speed fell from about 0.49 m/s to zero during this interval. This is
  a confirmed target/actuation mismatch in the partial run, not yet a proven
  root cause; inspect the speed-controller state transitions using only its
  legal command, odometry, and IMU inputs.
- Therefore the 0.70 cluster threshold is not promoted, raceline tracking is
  not accepted, and there is no new production MPC/model result. The next
  useful step is to diagnose the speed-command/throttle mismatch and source-
  time update jitter, then address the rapid odometry/map-position drift before
  repeating a full sensor-only batchmode run.

The earlier same-day `pp_local_cluster070_4mps_20260917` run is excluded from
this evidence because its actuator still subscribed to collision telemetry;
the launch/config inconsistency found there has since been fixed.

## Full MPC rewrite handoff progress — 2026-09-18

The latest coding handoff is
`SDU_Apex_Autodrive_MPC_Full_Rewrite_Coding_Handoff_2026-09-18.md`. It
supersedes the provisional ten-state design above as the target architecture.
The target is seven nonlinear plant states `[e_y, e_psi, u, v, r, v_c,
delta_c]` plus the two previous rate inputs, for nine QP states total; there
is no effective-steering dummy state. Unity physics, scenes, timing, and
vehicle behavior remain read-only. The ROS node now calls
`mpc_rti_solve_cycle`; MPC command authority remains disabled by default.

Completed and recorded in `artifacts/mpc_rewrite_845e621/`:

- Phase 0 baseline and Phase 1 regression reproduction.
- Phase 2 continuous raceline geometry/projection and N+1 references.
- Phase 3 legal `/odom` + `/current_map_pose` source-time synchronization and
  replay; command-time forward prediction is still pending.
- Phase 4 seven-state nonlinear stage and Phase 5 branch-aware Jacobians.
  Directional-derivative relative error is `0.000342212` maximum across the
  straight, corner, and braking cases; the heading-wrap Jacobian error is
  `0.000165939` at the tested ±π boundary.
- Phase 6 solver parity: generic active-dimension Riccati calculations match
  an independent dense condensed-QP oracle for 9 states to maximum control/
  state errors `5.59e-9` / `7.45e-9`. Nine-state constrained ADMM is also
  covered. Bounded control-Hessian regularization is diagnosed and rejects
  material indefiniteness.
- Phase 7/8 core: the absolute-value 9-state QP cost/bounds builder now encodes
  the exact previous-input change penalty Q/R/N expansion. It also produces a
  shifted warm nominal, deterministic two-pass progress/reference schedule,
  and exact nonlinear candidate rollout with a hard-corridor gate. A
  synthetic steady-circle test builds and solves an N=4 QP, then accepts its
  nonlinear rollout; separate tests cover warm-shift indices and scalar/matrix
  cost equivalence.

The Humble container build and all seven package tests pass. Phase 9 numerical
replay and N=30 core timing are complete; see
`artifacts/mpc_rewrite_845e621/07_offline_replay/README.md` for full results.
The PP development trace and untouched holdout produced 2,630/2,630 accepted
optimal cycles with the solver iteration cap raised from 50 to 100; no model
equations or cost weights were changed. N=30 core p99 was 0.783 ms on the
synthetic-state raceline timing benchmark. Phase 10's ROS adapter and launch
path are now integrated: shadow mode is mutually exclusive with command
authority, subscribes to Pure Pursuit `/cmd/speed`, and creates only the
`/mpc_shadow/diagnostics` publisher. A ROS-only component probe confirmed the
shadow node has no `/cmd/speed` publisher. Launch Python and YAML parse checks,
the Humble build, and seven package tests pass. These remain static/offline
results: no live shadow trace has been collected, so prediction alignment,
runtime timing, and the Phase 10 gate are not yet accepted. A Phase 10
batchmode startup was attempted on 2026-09-18 and is blocked before telemetry;
details follow. No Unity source, scene, physics, timing, or build artifact was
edited.

### Phase 10 batchmode startup attempt — blocked, 2026-09-18

The Humble ROS stack launched Pure Pursuit capped at 4 m/s and loaded the MPC
shadow component. The component reported shadow mode with command authority
disabled. A rosbag recorder subscribed to `/odom`, `/current_map_pose`,
`/cmd/speed`, and `/mpc_shadow/diagnostics`.

The local competition player was started in batchmode, without
`-no-graphics`:

```text
/home/akselmo/Documents/GitHub/AutoDRIVE/Builds/AutoDRIVE-Simulator.x86_64
  -batchmode -ip 127.0.0.1 -port 4567
  -logFile /tmp/mpc_shadow_batchmode_4mps_20260918.log
```

It exited immediately. `strace` and GDB both confirmed `SIGSEGV` in
`GameAssembly.so` during `il2cpp_init()`; the Unity log remained empty and the
bag contains zero messages. This is a simulator-startup failure, not an MPC,
controller, or timing result. The Unity checkout has pre-existing user edits
to vehicle/controller scripts and the competition scene, so rebuilding it
from that dirty source could change vehicle behavior. The shared local
`GameAssembly.so` is newer than the competition player data, but this date
mismatch is only a candidate explanation, not a proven root cause. The older
May simulator image was deliberately not substituted because it is not the
same verified competition build.

Phase 10 live prediction alignment therefore remains untested. Resume only
with a verified matching competition player/data/`GameAssembly.so` bundle, or
after explicit approval to build an isolated player from the current dirty
Unity checkout with physics/behavior equivalence checked. MPC authority stays
disabled.

The handoff-conformance cleanup is complete: the ROS node and offline tools
share the nine-state `mpc_rti_solve_cycle` path. The superseded compatibility
controller, old model API, and unused math-helper path are no longer part of
the package.

## Timing and sensor-reference audit — 2026-09-17

The 40 Hz mean does hide real delivery jitter. In the 9,631-packet 4 m/s
diagnostic capture, Unity source-time steps remained 24–31 ms and request
steps remained about 25 ms, but the bridge's own monotonic response-arrival
interval reached 49.9 ms seven times. In the 7,789-packet anchor capture,
source steps remained 25–28 ms while six bridge-arrival intervals reached
47–48 ms. These are actual host-side response-delivery stalls, not just a CSV
timestamp artifact. They were followed by normal ~25 ms intervals, not a
short next interval that fully repaid the delay. A separate recorder-only
56 ms then 3 ms callback pair occurred while the bridge's own intervals were
regular (24/26 ms), demonstrating additional recorder/executor jitter. AMCL
had still longer publish gaps (up to roughly 190–200 ms) while its source
stamps remained regular, so AMCL callback/compute latency is a distinct issue.

The timing recorder was reducing capture quality: it explicitly flushed the
CSV on every callback and retained all events in Python objects until close.
It now periodically flushes (default 0.5 s), retains only bridge timing rows,
and streams topic partitions from the canonical `events.csv` at shutdown.
This bounds recorder-side I/O and memory overhead; it does not fix the
bridge-side stalls or change simulator behavior. The last partial run also
started recording at simulator time 14.495 s, so its observer replay lacks the
preceding state history and must not be scored as a fresh-start odometry
validation.

Read-only Unity source inspection found an important point-definition issue.
`IMU.FixedUpdate()` derives linear acceleration from the Rigidbody velocity
difference and rotates it into vehicle axes; it does not compute acceleration
at the IMU GameObject's mounted location. The F1TENTH prefab sets the Rigidbody
COM near the root (`z = -0.00468 m`) while the IMU object has a separate
longitudinal offset. The dev odometer currently uses `imu_x_m: 0.08` both for
the published IMU TF and as an acceleration lever-arm correction. The active
test scene's effective COM/IMU override is not yet confirmed, so this is a
source/runtime mismatch candidate, not permission to change the deployed
correction. Position p95 favored 0.08 m over zero in both full diagnostic
captures, but their sensor residuals and run conditions differed; that fit
does not establish that 0.08 m is the physical reference point, so no offset
candidate is promoted.

Other sensor findings from those offline-only captures:

- The Unity wheel encoder component reads the rear WheelCollider RPMs
  independently and accumulates quantized angles. Runtime odometry currently
  averages the two channels. Repeated-angle samples were simultaneous on both
  sides (3.4% in one capture and 18.8% in the other), so a split does not
  recover a fresh measurement when both are stale. In turns, independent
  100 ms wheel-speed estimates differed by median 0.047 m/s and p95 0.110
  m/s: worth retaining as a slip/asymmetry diagnostic, not yet proven as a
  better speed estimator. Differential-wheel yaw alone was materially noisier
  than the direct IMU gyro.
- One full-length 4 m/s trace had sensor-only `/odom` longitudinal error p95
  0.087 m/s and lateral-velocity error p95 0.022 m/s; during active braking,
  these were 0.124 and 0.082 m/s. The other full-length trace was worse
  (overall p95 0.367 and 0.175 m/s, with straight active-braking longitudinal
  p95 0.490 m/s). These captures have different run conditions and do not
  establish a universal controller result. They do support stratifying
  encoder/acceleration calibration by powered drive versus active braking,
  then validating on a clean holdout.
- The custom EKF consumes `/odom` pose deltas and propagates covariance; it
  does not fuse or window IMU measurements. More IMU windows belong in the
  sensor odometer only if source-time replay shows held-out benefit. EKF
  covariance tuning alone cannot repair a biased mean trajectory.
- The 0.70 AMCL cluster threshold remains unpromoted: its partial run accepted
  more scan corrections but did not materially beat odometry position error.
  First measure AMCL scan-to-publish latency and whether accepted corrections
  reduce cross-track error on a fresh synchronized run.

Next validation must start the recorder before connecting/resetting the sim,
capture a fresh 40 Hz source interval trace plus live `/odom`, `/ekf_odom`,
and `/current_map_pose`, and separate powered straight, powered turn, active
braking straight, and active braking turn samples. First establish clean
multi-lap Pure Pursuit at the conservative cap, then repeat at the requested
full-speed operating point. Do not change Unity physics, scene geometry, or
the unverified acceleration-reference offset during this process.

## Batchmode validation launch blocker — 2026-09-17

This launch blocker was superseded later on 2026-09-17 by the fresh PP run
recorded below. The earlier failure remains useful history but is not the
current player status.

The next controlled PP run was preflighted with the Humble controller stack,
global AMCL, `/cmd/speed` capture, collision safety disabled, a 16 m/s PP cap,
and the configured 1.5 m/s first-lap ramp. The Unity track player was started
with `-batchmode` and without `-no-graphics`, as required. It exited with
SIGSEGV before connecting or producing any simulator packets. GDB places the
fault in `GameAssembly.so` during `il2cpp_init`, before the managed
batchmode/socket startup code; this is not a PP crash and is not evidence about
localization or vehicle behavior.

The player artifacts currently present are from different build dates:
`AutoDRIVE-Simulator.x86_64` and `UnityPlayer.so` are dated Sep 10,
`AutoDRIVE-Simulator_Data` metadata Sep 15, and the shared `GameAssembly.so`
Sep 16. This is consistent with a mixed/partially replaced player, but does
not by itself prove binary incompatibility. No rebuild was attempted: the
external Unity checkout is dirty and its competition-scene diff replaces a
track `MeshCollider` mesh. Rebuilding from that state could change the
collision environment, so a matching known-good build or explicit confirmation
of the intended collider asset is required before rebuilding. No Unity source,
scene, physics, or build artifact was edited in this turn.

The diagnostics recorder now flushes the event stream every 0.5 s instead of
on every callback, retains only bridge-timing rows in memory, and writes a
streamed `controller_trace.csv` containing `/cmd/speed` and Pure Pursuit's
diagnostics (including validity, CTE, target speed, and trajectory indices).
The targeted recorder and speed-controller tests pass (34/34), and
`git diff --check` is clean. This reduces recorder load and makes PP output
auditable, but no simulator-side A/B timing result or new lap is available
until the player startup blocker is resolved.

## Fresh PP, timing, odometry and AMCL validation — 2026-09-17

Run `sdu_apex_autodrive/artifacts/simulator_trace/pp_geom_cov_candidate_20260917`
is a valid multi-lap prefix, but not a clean full-run acceptance: the strict
40 Hz source-cadence watchdog stopped it at 121.450 s. The Unity player ran in
`-batchmode` without `-no-graphics`; no Unity source, scene, physics, track,
or vehicle behavior was edited. Runtime odometry parameters were checked live
at `imu_acceleration_reference_x_m = 0.15532` and
`lateral_velocity_reference_forward_offset_m = 0.15532`. The Humble build
passed; all three localization C++ tests and all 108 Python tests passed.

- The corrected estimator completed eight raceline-index wraps with no invalid
  Pure Pursuit samples. Reported CTE p95 was 0.060 m and the commanded speed
  peaked at 8.76 m/s (the raceline did not request the 16 m/s project ceiling).
  It did not crash or leave the track before the cadence watchdog latched
  neutral. This demonstrates a long multi-lap driving segment, not a passing
  end-to-end run because the capture ended on a source timing fault.
- There were 4,855 accepted source packets, no sequence gaps, and a 24.998 ms
  mean source interval. Requests averaged 25.000 ms and bridge response
  arrivals 25.001 ms. Four response-arrival intervals were below 15 ms and
  four above 35 ms, including four short-then-long pairs: the 40 Hz mean does
  hide bursty delivery. Source-header age was not zero: p95 was 34.3 ms for
  IMU, 35.2 ms for `/odom`, 40.6 ms for `/amcl_pose`, and 36.7 ms for
  `/current_map_pose`.
- Recorder callback delay after bridge arrival was only 1.43 ms mean, 1.84 ms
  p95 (3.72 ms maximum). Bridge request-to-response age was 29.5 ms mean and
  30.5 ms p95. This evidence does not support CSV writing as the primary
  source of the delivery delay; the latency is already present before the
  recorder callback. The bridge intentionally allows two in-flight requests
  to preserve 40 Hz when round-trip time exceeds 25 ms; this capture had an
  exact one-packet command lag. That bounded pipeline plus the measured
  irregular arrival spacing confirms delivery jitter. Specifically, four
  short intervals were followed by long intervals, while no long-then-short
  pair occurred; this run therefore does not reproduce the proposed immediate
  delayed-then-fast compensation pattern. There were no sequence gaps or
  evidence of an indefinitely growing queue in this prefix.
- The terminal packet was consecutive (`telemetry_sequence` 4859 to 4860),
  but source time advanced only 13.001 ms (physics step 121438 to 121451,
  render-frame delta 1). It arrived 22.747 ms after the previous response and
  31.408 ms after its request. The bridge correctly latched neutral under its
  strict `[15, 35]` ms source-step contract. This is a source clock/update
  irregularity, not packet loss. The capture does not establish why Unity
  advanced only 13 physics steps during that callback interval; inspect
  request/callback scheduling and repeatability before relaxing the guard or
  changing any simulator timing setting.
- Offline-only `/odom` position p95 was 0.825 m over the accumulated run,
  longitudinal-speed error p95 0.149 m/s, lateral-speed error p95 0.044 m/s,
  and yaw-rate error p95 0. The lateral-speed result meets the 0.10 m/s goal
  after correcting the Rigidbody-COM/rear-axle reference mismatch. `/ekf_odom`
  had the same mean trajectory as `/odom`; the EKF is a covariance trust
  filter, not an IMU-fusing or multi-window motion estimator.
- `/amcl_pose` position error p95 was 0.105 m and `/current_map_pose` p95
  0.107 m, above the 0.05 m target. Yaw error p95 was about 9.7e-6 rad; the
  current-map covariance maximum diagonal p95 was 0.098 m^2 and remained below
  the PP safety gate. Thus yaw/cluster confidence is not the obvious remaining
  error; position still needs improvement. AMCL's normal compute p95 was
  under 5 ms. One startup resampling stage took 190 ms and caused the only
  >50 ms pose callback gap; the scan/odom queue remained empty and reported no
  dropped scans. Do not treat this one startup event as persistent 40 Hz AMCL
  overload.
- The two rear encoder channels repeated together on 10.36% of adjacent
  samples; there were no left-only or right-only fresh increments. Keep both
  channels available for differential turn/slip diagnostics, but independent
  filtering will not create new samples when both wheels repeat.
- Offline source-contract inspection confirms exact zero throttle selects the
  vehicle's configured brake-torque branch (default CAWB applies the fixed
  brake torque to all four wheels). In the candidate, 562 moving samples were
  full-brake/zero-throttle and 4,264 were powered, using `speed > 0.5 m/s` to
  exclude stationary transitions. Their absolute longitudinal-speed p95
  errors were 0.324 versus 0.136 m/s; lateral-speed p95 errors were 0.089
  versus 0.008 m/s. Brake samples were mainly in turns (absolute yaw rate
  median 0.573 rad/s), so this is useful stratification, not a clean straight
  braking identification holdout. Do not call throttle-zero coast-down data.
- The full offline report is saved as
  `full_speed_localization_report.json` and
  `full_speed_localization_errors.csv` beside the raw capture. Every file in
  the run folder remains below the 100 MB repository limit.

Next work is now specific: preserve the corrected point geometry; first
resolve/reproduce the source-clock short step without changing Unity physics;
then collect matched straight powered, straight full-brake, powered-turn and
brake-turn data. Fit/ablate longitudinal encoder and IMU windows in the sensor
odometer (not the EKF), and only change AMCL cluster gates if a matched run
shows that they reduce the 0.107 m map-position error. Require a clean
collision-free multi-lap PP run after timing is stable before calling the
stack recovered.

## Brake-mode signal audit — 2026-09-17

The current captures do not record the actuator's published
`/autodrive/roboracer_1/throttle_command`; the simulator-reported applied
throttle is useful as an offline label but is not an estimator input. I tested
whether the legal `/cmd/speed` target and `/odom` speed alone can identify the
full-brake mode one source packet ahead. Across both the 5-minute run and the
121-second candidate, a simple `odom_speed - target_speed >= 0.3 m/s` rule
reached only 0.67 precision / 0.77 recall (older capture) and 0.67 / 0.76
(candidate). The false-positive rate among powered samples is too high for
selecting different online encoder windows. Target drops alone were weaker.
No brake-window switch is therefore promoted from this proxy.

The model-ID recorder now also captures the existing actuator-published
steering and throttle commands into `actuator_commands.csv`. This is
observation-only instrumentation: it does not change the actuator, estimator,
bridge, or simulator. On the next valid batchmode run, align that command
stream with source packets and compare causal brake-mode detection against the
offline applied-throttle labels. If the exact actuator output reliably
separates modes, test the already measured 75 ms encoder window only during
braking while retaining the 100 ms powered window; score both longitudinal
and lateral error in straight/turn regimes before considering runtime use.
Keep `/cmd/speed` and `/odom` as the only fallback if the actuator command
signal is not allowed at deployment. The simulator timing fault and clean
collision-free multi-lap acceptance remain unresolved.

## Bridge request-window correction — 2026-09-17

The short source-step / short-arrival fault was reproduced with the current
rebuilt localization binary in `pp_rebuilt_localization_20260917`. The last
two completed requests before the fault had response ages of 51.506 ms and
46.187 ms. With the former two-request cap, both slots were occupied; the
next scheduled send was skipped, yielding a 41.32 ms request gap. The next
source step was 38.000 ms, then that response arrived only 13.125 ms after
the previous response (request response age 17.991 ms). Sequence remained
consecutive. This directly reproduces the user's delayed-then-fast arrival
pattern as bounded bridge backpressure, rather than a lost packet or recorder
callback delay.

The dev-only bridge now sizes its bounded in-flight window as
`ceil(75 ms * 40 Hz) = 3`. This preserves the existing 75 ms transport timeout,
strict 15–35 ms source-step contract, and no-interpolation rule; it only avoids
skipping a scheduled request while the full response deadline is represented
by two slow replies. The dedicated sizing test and bridge/recorder/report
targeted tests pass (11/11). This is a diagnosis-driven fix, not yet a proven
recovery: a fresh batchmode PP run using this exact bridge and the rebuilt
localizer is required to establish whether source jitter and multi-lap driving
improve.

The earlier `pp_actuator_mode_trace_20260917` remains useful for transport,
AMCL, and actuator-command alignment only. Its odometry-accuracy numbers are
excluded because the live C++ executable was stale relative to the checked-out
source. The subsequent short `pp_rebuilt_localization_20260917` run verified
the rebuilt executable and runtime parameters but ended at 7.4 simulated
seconds, so it provides no meaningful odometry-accuracy or lap validation.
No Unity project file, scene, track, physics, or vehicle behavior was changed.

## Request pacing catch-up correction — 2026-09-17

The first PP run with the three-slot window, saved as
`sdu_apex_autodrive/artifacts/simulator_trace/pp_three_slot_current_20260917`,
had no request-sequence or telemetry-sequence gaps. Its valid prefix contained
690 packets at a 24.980 ms mean source interval; recorder callback delay was
1.74 ms mean / 2.46 ms p95. However, the request scheduler preserved its old
periodic phase after a late send. Near the end, consecutive request intervals
were 34.715 ms then 20.786 ms; the faulting request was inferred at about
15.96 ms after its predecessor from the recorded response age and arrival
interval. That request produced an 11 ms source-time step (11 physics steps),
despite consecutive telemetry sequence numbers. This is the same late-then-
early compensation pattern the user suspected. The captured prefix was not a
multi-lap or clean timing acceptance.

The initial pacing guard moved the next request deadline to a full 25 ms after
the actual emission. It prevented catch-up bursts, but the follow-up run
measured 25.80 ms median request spacing (about 38.8 Hz), so that strict floor
also let timer oversleep accumulate as a slower cadence. The bridge has now
been refined to preserve the nominal 25 ms phase grid while enforcing a 24 ms
minimum between actual sends. This blocks the observed 16–20 ms bursts while
allowing small phase correction; this latest setting is not yet simulator-
validated. The source 15–35 ms contract, response deadline, three-slot cap,
and sequence checks remain unchanged. Targeted bridge/recorder/report tests
pass (13/13). The next batchmode PP capture must verify near-40 Hz mean
request/source cadence, no short source steps, and multiple laps. No simulator
project, scene, physics, track, or vehicle behavior was modified.

## Collision-triggered checkpoint respawns — 2026-09-17

The `pp_paced_three_slot_20260917` source-truth trace contains eight
multi-meter position snaps between otherwise consecutive 25–27 ms packets
during the first 15.2 simulated seconds. Each snap is accompanied by velocity
falling nearly to zero and yaw returning to a fixed checkpoint heading; every
`sent_reset` value is false. The source timing and telemetry sequence continue
through these events, so they are not message gaps or bridge reset commands.

Read-only inspection of the Unity source identifies the matching existing
path: `LapTimer.OnCollisionEnter` calls `Respawn()` when the vehicle collides
with the racetrack object; `Respawn()` zeros linear/angular velocity, assigns
the last passed checkpoint pose, and increments `CollisionCount`. This
checkpoint-respawn signature matches the recorded jumps. No Unity file was
edited. The current capture did not preserve the collision counter itself, so
the bridge now retains the existing `V1 Collisions` telemetry as
`simulator_collision_count` in the offline packet trace only; it is not
published on ROS or consumed by any runtime node.

The first such respawn happened while the preceding estimates were still
accurate: `/current_map_pose` was about 0.03 m from truth and PP CTE was
−0.18 m. On the very next source packet, truth had moved to the checkpoint
while `/odom` and AMCL remained at the pre-respawn pose, about 1.66 m away.
Across later snaps, the map-pose error reached 5–7 m and PP CTE grew above
2 m. Therefore those large offline odometry/model errors are dominated by
unannounced teleport discontinuities, not smooth vehicle dynamics; such
intervals must be segmented out of continuous-motion identification.

The active AMCL YAML previously disabled `enable_recovery_injection`, despite
an existing guarded recovery path that waits for repeated local-cluster
rejections, then requires a strong, stable map/raceline cluster and the
existing pose-jump gates. That path is now enabled as a candidate, with its
10% recovery particle ratio and confirmation thresholds unchanged. This does
not make simulator collision data a runtime input.

The first candidate run,
`pp_amcl_checkpoint_recovery_20260917`, verified the live parameter as true
and recorded `simulator_collision_count = 0` for all 565 packets, with no
truth-position snaps. PP CTE p95 was 0.063 m (max 0.166 m), and commanded
speed reached 6.47 m/s by about 0.80 raceline fraction. Offline map-position
error p95 was 0.107 m; `/odom` position p95 was 0.310 m, longitudinal-speed
error p95 0.106 m/s, lateral-speed error p95 0.018 m/s, and yaw error p95
effectively zero. These estimates and path tracking are substantially better
than the collision-contaminated run, but the recovery branch was not observed
to activate, so this is not evidence that it fixed checkpoint recovery.

The run still failed the hard timing gate before completing one lap: request
spacing had no sub-25 ms bursts (minimum 25.249 ms), but Unity source time
advanced 38 ms / 38 physics steps in one telemetry step, with consecutive
sequence numbers and one render-frame increment. All earlier accepted source
steps were at most 30 ms. No watchdog was relaxed. A full collision-free
multi-lap run, and a separate deliberate observation of the guarded AMCL
recovery path in an unaltered simulator run, remain required.

The 25 ms floor above was subsequently identified as too conservative: its
request median was 25.80 ms. The revised 24 ms floor on the 25 ms phase grid
is now validated for mean-rate and catch-up behavior in the follow-up run
below; the separate strict source-step gate still fails on an isolated
37 ms step.

## Three-wrap PP run and timing-stage separation — 2026-09-17

Run `pp_amcl_recovery_24ms_20260917` used the updated three-request bridge
window and 24 ms minimum send spacing on the 25 ms request phase. Unity ran in
`-batchmode`, not `-no-graphics`; no Unity source, scene, physics, track, or
vehicle behavior was changed. The run completed three raceline-index wraps
(then began a fourth) at up to 8.5 m/s. There were no invalid PP samples, no
collision-counter increments, and no truth-position discontinuities. PP
cross-track error was 0.057 m p95 / 0.166 m maximum. This restores a clean
multi-lap PP segment, though the run is not strict timing acceptance.

The trace contains 1,851 consecutive source packets over 46.34 s of simulator
time. Mean source interval was 25.049 ms (39.92 Hz); request spacing averaged
25.056 ms and bridge-arrival spacing 25.057 ms. There were no request or
telemetry sequence gaps, no request intervals below 24 ms, and no bridge
arrival intervals outside 15–35 ms. The final requested packet advanced
source time by 36.997 ms (37 physics steps) while telemetry sequence advanced
by one and render frame by seven. The strict source watchdog correctly
stopped the run. Earlier accepted intervals included 15 intervals above
30 ms, so the mean being 40 Hz does not mean every step was exactly 25 ms.

Timing stages distinguish the suspected recorder backlog from transport
latency. Bridge-to-recorder callback latency was 1.93 ms mean / 2.56 ms p95;
the `/odom` callback interval mean was 25.055 ms and p99 28.684 ms. This does
not support excessive CSV writing as the cause of burst delivery in this
capture. Conversely, request-to-bridge response age was 33.19 ms mean / 36.29
ms p95 (40.89 ms max), and `/odom` source-header age on callback was 42.86 ms
p95. Responses were consistently delayed relative to each request but
arrived at a steady cadence; this is latency, not evidence of a growing
backlog. A second run reproduced the source-step failure in the opposite
direction (10 ms); read-only Unity inspection then localized the bad interval
to its fixed-time telemetry sampling path, as detailed below. Do not relax
the 15–35 ms watchdog.

Offline state comparison over this run gives `/current_map_pose` position
error p95 0.131 m, AMCL pose p95 0.129 m, and `/odom`/`/ekf_odom` position
p95 0.350 m. `/odom` longitudinal-velocity error p95 was 0.145 m/s and
lateral-velocity error p95 0.046 m/s. These are whole-run offline metrics;
they do not contradict low PP CTE, because raceline-relative tracking error
and global truth-relative localization error are different measures. AMCL's
guarded recovery parameter was enabled, but this collision-free run did not
exercise the recovery branch, so no improvement is attributed to that
candidate. The full report and joined errors are saved beside the raw trace;
each artifact remains below 100 MB. The offline report now includes
`position_error_truth_frame` absolute and signed components for future runs.

Do not read the Euclidean map-pose position p95 in isolation. An offline
projection of source-aligned position residuals onto the truth-yaw frame
shows repeatable component separation in both clean captures:
`/current_map_pose` body-lateral p95 was 0.0334 m and 0.0314 m, while its
body-longitudinal p95 was 0.1275 m and 0.1282 m. `/odom` and `/ekf_odom`
body-lateral p95 was 0.296 m and 0.264 m, and their body-longitudinal p95 was
0.266 m and 0.260 m. EKF mean pose error is effectively identical to `/odom`
in both runs; the EKF currently changes trust/covariance, not the biased mean
trajectory. This is offline truth analysis only. Combined with live PP CTE
p95 of 0.057/0.062 m, it indicates that the current AMCL/map-pose path has
already corrected lateral position to about 3 cm p95; the remaining 0.13 m
Euclidean error is mostly along the vehicle heading. Avoid AMCL parameter
changes aimed only at lowering that Euclidean metric.

The source-aligned odometry velocity errors also reproduce a braking-specific
problem. Using simulator-applied throttle only as an offline regime label,
longitudinal-speed error p95 in full-brake samples was 0.20–0.32 m/s across
these captures, compared with 0.094–0.140 m/s in powered straight/turn
samples. Powered-turn lateral-speed error was 0.007–0.008 m/s; full-brake
turn lateral-speed error was 0.074–0.112 m/s. These regime scores are
diagnostic, not deployable labels or a fitted braking correction. Next
odometry work should determine whether encoder quantization/dropout and the
existing causal IMU deceleration path explain the braking residual, using
runtime-available signals only; do not feed the simulator throttle field to
the estimator.

## Timing-fault reproduction and Unity source-stage diagnosis — 2026-09-17

The repeat run, `pp_timing_repeat_20260917`, used the same batchmode player,
PP settings, and strict source watchdog while sampling the Unity process with
`pidstat`. PP completed one raceline-index wrap and continued about 27% into
the next before the watchdog stopped it; all 781 recorded PP diagnostics were
valid, CTE was 0.062 m p95 / 0.166 m maximum, speed command peaked at 7.95
m/s, and the simulator collision counter remained zero. Its `/current_map_pose`
and AMCL position-error p95 were 0.130 m; `/odom` and `/ekf_odom` p95 were
0.307 m.

This second capture reproduces source-step jitter independently of recorder
load: 791 packets had consecutive request and telemetry sequences, request
spacing averaged 25.056 ms, bridge response-arrival spacing 25.058 ms, and
bridge-to-recorder callback latency was 1.93 ms mean / 2.63 ms p95. Yet at
source time 31.1150017 -> 31.1250019 s, the next packet advanced by only
10 ms / 10 physics steps while its bridge arrival interval was 25.724 ms and
request response age 32.674 ms. That violates the lower 15 ms source gate;
the bridge stopped and latched neutral. The PID sample showed Unity using
about 280–296% CPU with roughly 794 MB resident memory near the stop, without
a corresponding whole-process CPU spike. No collision occurred.

Read-only Unity code inspection narrows the faulty stage. `Socket.OnBridge`
calls `EmitTelemetry` synchronously; that response stamps `Simulation Time`
from `Time.fixedTime` and includes the vehicle physics-step counter. The
Socket.IO plugin drains its WebSocket event queue from Unity `FixedUpdate` on
the main thread. Thus the 10 ms interval is already present in Unity's
fixed-time/physics-step state when telemetry is formed; neither host message
arrival timing nor recorder callback spacing created it. The configured
fixed step is 1 ms and project time scale is 1. This identifies the Unity
fixed-loop/request-service path as the stage that needs further diagnosis,
but the capture does not distinguish main-thread scheduling from delayed or
uneven FixedUpdate/event-queue service. Do not change the fixed step, time
scale, physics, or the source watchdog to hide the fault. A future diagnostic
pass should correlate Unity main-thread scheduling with source steps; the
clean 3-wrap run establishes PP capability but not continuous 40 Hz source
acceptance.

## Promotion rule

An MPC model is not promoted because it resembles a car, has a low one-step
fit, or uses unavailable Unity state.  Promotion requires: source-contract
review, clean 40 Hz causal input timing, blind legal-state recursive scores at
the four horizons, controller watchdog/failsafe tests, and a collision-free
batchmode lap using the same legal runtime inputs.

## Open-ground braking and encoder validation — 2026-09-17

Three fresh-player, batchmode `speed_steps` traces exercised straight-line
acceleration and zero-throttle full braking. Unity physics and project files
were not edited. The configured project speed ceiling remains 16 m/s; these
runs targeted 12, 14, and 15.3 m/s and do not validate any turning behavior.

| Target | Run directory | Peak truth speed | Brake distance to confirmed rest | Source timing |
| --- | --- | ---: | ---: | --- |
| 12 m/s | `brake_speedsteps_openplane_12mps_train_20260917` | 11.995 m/s | 10.83 m | 672 packets; mean 25.070 ms, range 13–41 ms; no sequence gaps |
| 14 m/s | `brake_speedsteps_openplane_14mps_train2_20260917` | 14.02 m/s | 13.65 m | 717 packets; mean 25.068 ms, range 14–32 ms; no sequence gaps |
| 15.3 m/s | `brake_speedsteps_openplane_15p3mps_train_20260917` | 15.293 m/s | 16.88 m | 747 packets; mean 25.068 ms, range 15–35 ms; no sequence gaps |

All three runs confirmed rest between/after target phases and recorded zero
collisions. Distances are offline truth measurements from the end of the target
phase to the confirmed stop; they are speed-dependent observations, not a
constant-deceleration fit and not a direct conversion of the user-provided
428 N. On the 10 Hz calibration snapshots, deployed observer longitudinal
body-speed absolute p95 error was 0.030/0.028 m/s in powered/braking samples at
12 m/s, 0.048/0.017 m/s at 14 m/s, and 0.098/0.094 m/s at 15.3 m/s. These are
straight open-ground results scored offline against the simulator's direct
body-forward speed; lateral/yaw or track-localization accuracy is not tested
by them. Relative longitudinal `/odom` position error ended at -0.049 m,
0.008 m, and 0.191 m respectively; lateral error stayed below 3 mm.

The left/right ablations are saved as `encoder_side_ablation.json` in each
run directory. Repeated-angle counts were simultaneous on both sides (65/233,
74/277, and 79/308 moving intervals); no left-only or right-only fresh
increments occurred. Paired mean, left-only, and right-only errors were
effectively identical on these straight runs. This gives no basis for choosing
one encoder over the other; retain both for differential turn/slip diagnostics
and keep the existing fused odometer. The ablation also shows that standalone
mapped wheel speed is not a reliable body-speed substitute in these high-speed
conditions, while the live observer's innovation gates/IMU propagation kept
its speed residual much smaller.

An initial 14 m/s attempt stopped before motion when host packet-arrival spacing
reached 77.032 ms. The bridge now permits up to 150 ms host arrival/response
delay while preserving each packet's actual source stamp, physics step, and
sequence; a unit test explicitly verifies that a valid 77 ms packet is
accepted. The successful 14 m/s retry itself had no arrival gap above 32 ms,
so the new bound is code-tested but has not yet been exercised by a live
77 ms packet. Source timing remains measured separately: the 12 m/s run's
13–41 ms source intervals are real variable intervals, not fabricated 25 ms
samples.

Unity's `-logFile` output grew to about 0.7–1.2 GB per run after the ROS bridge
had shut down, because the still-running player logged a full stack trace for
each refused reconnect. This was teardown-time retry spam, not CSV recorder
load during the captured runs. The four assistant-created temporary logs were
removed after inspection; all raw run CSVs and reports remain. Stop the
batchmode player immediately after a finite ROS capture to prevent recurrence.

## AMCL high-speed along-track gain A/B — 2026-09-18

**Question and decision rule.** Test whether raising
`local_scan_correction_fast_along_track_gain` from `0.25` to `1.0` at the
existing `3.0 m/s` threshold improves the live controller pose without trading
away lateral localization or raceline tracking. Promote only if along-track
error improves without a meaningful cross-track regression. This is one
parameter A/B, not a gain sweep.

Baseline `pp_amcl_along_base_20260917` and candidate
`pp_amcl_along_gain1p0_20260918` both used the saved raceline, Pure Pursuit,
the same installed base configuration, and the batchmode
`racelineSpeedSweep` player. The candidate override changed only the fast
along-track gain. Both runs used the diagnostics-only ground-truth monitor;
the high-volume model-ID and telemetry event recorders were off. No simulator
source, scene, physics, or vehicle settings were changed. Because no map
provenance transform was supplied, truth comparison is relative to the
monitor's first-pair alignment, not absolute map-pose accuracy.

To compare equal exposure, monitor rows were deduplicated by source timestamp,
then scored through the first `13 x 51.738 m = 672.6 m` of moving ground-truth
path. Position residuals are estimate-minus-truth projected onto the local
raceline tangent (along) and normal (cross); p95 is absolute. The controller
uses `/current_map_pose`:

| Metric over matched path | Gain 0.25 | Gain 1.0 | Change |
| --- | ---: | ---: | ---: |
| `/current_map_pose` along p95 | 1.551 m | 1.459 m | -5.9% |
| `/current_map_pose` cross p95 | 1.395 m | 1.458 m | +4.5% |
| AMCL along p95 | 1.567 m | 1.475 m | -5.9% |
| AMCL cross p95 | 1.389 m | 1.456 m | +4.8% |
| Actual vehicle raceline cross-track p95 | 1.251 m | 1.292 m | +3.3% |

The candidate completed more than 15 raceline-lengths by position-path
distance without a collision; peak speed was 8.65 m/s versus 8.63 m/s in the
baseline. In the matched 672.6 m window, source-time means were 25.06 ms
(candidate) and 25.09 ms (baseline), with a 49 ms maximum in both. There were
5 and 37 intervals over 35 ms, respectively. These are source-stamp intervals
from the monitor, not packet-sequence or host-arrival proofs; the monitor does
not record sequence IDs. The candidate remained live for the full capture and
the player and launch were then stopped cleanly.

**Disposition: do not promote gain 1.0 as a blanket fast-speed default.** It
slightly improves along-track p95 but worsens cross-track localization and the
vehicle's actual raceline cross-track p95. The strong improvement seen in the
first 96 seconds did not hold as a non-regressing result over the matched
multi-lap window. Keep the production value at `0.25`; this A/B is complete and
must not be repeated unchanged. Next use the existing directional `/odom`,
EKF, and AMCL residuals, stratified by braking/turning and track distance, to
identify the source of cross-track growth before proposing any speed- or
curvature-conditioned correction. Do not change EKF covariance to address the
mean trajectory.

## Braking-turn encoder observability — 2026-09-18

This analysis uses existing collision-free, source-sequenced track captures
`pp_amcl_recovery_24ms_20260917` and `pp_timing_repeat_20260917`; simulator
applied throttle and truth speed are offline labels only. The target is to
decide whether braking-turn speed should use a separated encoder or a
brake-specific observer term, not to collect another general speed run.

In the full-brake/turn subset (`abs(applied_throttle) <= 1e-4`, truth speed
above 0.5 m/s, `abs(truth yaw rate) >= 0.1 rad/s`), every joined `/odom`
diagnostic sample rejected the wheel update: 0/170 and 0/62, respectively.
Median paired wheel rates were only 0.041 and 0.044 m/s, and median current
packet rates were zero, while median truth speeds were 3.748 and 3.233 m/s.
The median IMU longitudinal accelerations were -6.474 and -6.316 m/s^2. The
current observer is therefore correctly avoiding wheel-derived body speed
when the braked wheels have effectively stopped; splitting or loosening the
encoder gate would substitute a near-zero wheel rate for a moving chassis.

In those same groups, signed longitudinal `/odom` speed error (estimate minus
truth) averaged -0.100 and -0.110 m/s, with absolute p95 0.316 and 0.202 m/s.
Lateral-speed absolute p95 was 0.109 and 0.072 m/s. By contrast, existing
open-ground speed-step traces put the deployed observer below 0.1 m/s p95 in
both powered and braking straight samples through 15.3 m/s. The brake-turn
residual is repeatable but modest at the five-step (125 ms) control horizon;
it does not justify changing the global deceleration calibration or treating
the simulator-applied throttle as a runtime input.

**Disposition:** encoder separation is not a braking-turn speed fix. The
targeted train/held-out observer candidates using these traces are evaluated
in the following section; none justified an additional runtime encoder or
deceleration term. Simulator truth/applied throttle remain offline labels,
never runtime inputs.

## Existing-trace observer and AMCL decisions — 2026-09-18

This section closes the immediate candidate loop using the same two clean
captures above. It is not a new simulator run. The question was whether the
repeatable brake-turn residual can be reduced with a causal turn/encoder term
without worsening the state actually consumed by localization and control.
The development replay on `pp_amcl_recovery_24ms_20260917` reproduces live
`/odom` body velocities and integrated position to floating-point precision;
`pp_timing_repeat_20260917` remains the untouched holdout. Future sensor
samples and simulator truth are never candidate inputs.

The already-active, runtime-legal bounded turn correction
(`turn_speed_bias_constant_mps = -0.03`, inputs: mapped wheel speed and IMU
yaw rate) was ablated. Keeping it reduced holdout all-moving longitudinal
speed p95 from 0.157 to 0.133 m/s and powered-turn p95 from 0.148 to
0.118 m/s; truth-heading along/cross position p95 also fell from 0.351/0.384 m
to 0.260/0.264 m. Its small braking-turn-only speed cost does not outweigh the
track-wide held-out gain. **Retain the existing -0.03 m/s bounded correction**;
do not refit or duplicate it in another model.

Two additions were evaluated and rejected. A single coefficient on separate
left/right encoder differential plus IMU yaw rate reduced held-out powered-turn
lateral-speed p95 only from 0.00698 to 0.00667 m/s (all-sample lateral-speed
MAE 0.00654 to 0.00636 m/s), while held-out integrated cross-position p95
regressed from 0.26379 to 0.26423 m. This is below a useful position gain and
does not justify another runtime state. A brake-turn deceleration-scale
candidate (`0.90`, fitted on the recovery trace) improved held-out brake-turn
longitudinal-speed p95 from 0.202 to 0.153 m/s, but worsened full-trace
longitudinal/cross position p95 from 0.260/0.264 to 0.304/0.306 m. Reject it;
keep `decel_ax_scale = 1.005` and `decel_ax_offset_mps2 = 0.020`.

A causal brake-turn lateral-velocity override was also replayed recursively:
only with preceding commanded zero throttle, a turn, encoder dropout and
negative IMU longitudinal acceleration, replace dynamic lateral integration
with the existing yaw-rate/forward-speed kinematic lateral estimate. It
substantially reduced brake-turn lateral-speed p95 (recovery 0.111 to
0.0168 m/s; holdout 0.0742 to 0.00753 m/s), but integrated along/cross
position p95 regressed on both traces: recovery 0.266/0.296 to 0.279/0.316 m;
holdout 0.260/0.264 to 0.269/0.276 m. **Do not promote this twist-only win**:
the objective is the future pose/control state, and the integrated result is
worse. No observer source or YAML change follows from these rejected
candidates.

The same source-stamped traces were stratified by speed and raceline curvature
to determine whether AMCL is the next tuning target. `/current_map_pose`
lateral p95 is 0.0341/0.0312 m and along-track p95 is 0.1273/0.1281 m across
the two traces. In the 6–8 m/s bin, along p95 rises to 0.181 m and lateral to
0.052/0.049 m; above 8 m/s there are too few samples for a conclusion. The
highest-curvature bin (`abs(kappa) >= 0.4 rad/m`) is not the worst along-track
bin (0.056/0.053 m); the 0.05–0.10 rad/m bin is worse (0.180/0.162 m).
Speed and curvature bins are marginal, not a matched joint analysis, so they
do not establish a speed- or curvature-conditioned gain.

AMCL correction health does not indicate a transport backlog in these traces:
correction acceptance was 97.6%/94.4%, correction-age p95 27/26 ms, scan to
matched-odometry source error was 0 ms, and scan queues/drops were zero. Here
`correction_age` means time since the last accepted map correction, not pose
message delivery latency. The stable ~3 cm lateral map-pose error, recent
scan corrections, and rejected live gain-1.0 A/B (which worsened cross-track
localization and vehicle tracking) provide no basis for another blanket AMCL
gain change or EKF covariance adjustment. Keep the accepted AMCL along gain
at `0.25`.

**Use decision:** no more generic braking or AMCL gain sweeps. Keep the
validated turn-speed correction and existing observer/AMCL settings. The next
implementation candidate is the shared source-command response model used by
odometry and the future MPC, scored recursively with legal `/current_map_pose`,
`/odom`, IMU/encoder state and the command sequence only. First verify exact
command-to-source transition alignment and encode the target-speed/braking
response at N5; retain N10/N20/N30 as regression checks. Simulator truth is
only the offline target. Promote a response term only if the untouched
`pp_timing_repeat_20260917` holdout improves signed/absolute along, cross and
yaw N5 errors without a material N30 or integrated-pose regression. If the
recorded closed-loop command sequence cannot be treated as a planned exogenous
input for that question, do not score it as closed-loop MPC evidence; define
the missing excitation before collecting anything else.

## Held-out yaw-response identification — 2026-09-18

**Question and intended use:** does the source-geometry model's instantaneous
steering-to-yaw-rate response explain a material part of raceline prediction
error over the control-relevant N5 horizon? Use the existing
`pp_amcl_recovery_24ms_20260917` capture for identification and the untouched
`pp_timing_repeat_20260917` capture for recursive holdout scoring. No new
simulator run was made. Inputs at each rollout origin are legal
`/current_map_pose` and `/odom`; the recorded `/cmd/speed` steering sequence is
the known exogenous input. Simulator packet pose, velocity and yaw-rate are
used as offline regression data and endpoint scores; no simulator-truth value
is fed into the recursive held-out rollout. This is an open-loop replay of a
recorded Pure Pursuit command sequence, not closed-loop MPC acceptance.

The identification set contains 1,850 valid source transitions. The held-out
run contains 790 transitions; 756 legal-state origins have complete N30
rollouts and source-stamped map pose. For this isolation test both models hold
the legal initial forward and lateral velocities constant. The deployed
geometric baseline sets yaw rate immediately to `u * tan(delta) / 0.324` and
holds lateral velocity. The identified no-bias response is
`r_dot = (-r + 3.011897 * u * tan(delta)) / 0.087735`, recursively propagated
from the legal initial yaw rate. Its 87.7 ms time constant and 3.012 1/m steady
gain correspond to an effective wheelbase of 0.332 m, close to the source
geometry's 0.324 m. The fitted intercept was only 0.00246 rad/s at steady
state; removing it changed training/holdout yaw-acceleration residual RMSE by
only 0.00013/0.000095 rad/s^2, so it was omitted as an unsupported extra bias
parameter.

This no-bias response is now implemented in the shared C `vehicle_model` used
by the MPC; its model signature was incremented so stale warm starts are not
reused. I replayed that exact compiled C model on the untouched holdout. At
each rollout origin the predictor received only legal `/current_map_pose`,
`/odom` velocities/yaw rate and the recorded steering command sequence. Truth
pose/yaw were used only as offline endpoint targets. The table reports
truth-frame along-track and cross-track absolute p95 separately, plus absolute
yaw p95; source-stamped variable time steps were preserved and no 40 Hz samples
were synthesized.

| Horizon / subset | Along p95 (m), baseline → C model | Cross p95 (m), baseline → C model | Absolute yaw p95 (deg), baseline → C model |
| --- | ---: | ---: | ---: |
| N4, all 756 origins | 0.1326 → 0.1326 | 0.0291 → 0.0285 | 1.379 → 0.731 |
| N5, all 756 origins | 0.1340 → 0.1340 | 0.0303 → 0.0293 | 1.774 → 0.917 |
| N10, all 756 origins | 0.1792 → 0.1790 | 0.0388 → 0.0329 | 4.260 → 1.921 |
| N20, all 756 origins | 0.4772 → 0.4769 | 0.0802 → 0.0487 | 11.870 → 6.375 |
| N30, all 756 origins | 0.9714 → 0.9680 | 0.1750 → 0.0816 | 26.497 → 15.613 |
| N5, 400 turning origins | 0.0704 → 0.0704 | 0.0298 → 0.0292 | 1.974 → 0.710 |
| N30, 514 turning origins | 0.8976 → 0.8836 | 0.1841 → 0.0693 | 28.611 → 17.392 |

At N5 the along bias/MAE/p95 are unchanged (-0.0094 m / 0.0432 m /
0.1340 m). Cross bias improves from -0.0069 to -0.0067 m and cross p95 from
0.0303 to 0.0293 m, while cross MAE changes slightly worse (0.0122 to
0.0123 m). Thus the gain is clear in yaw and the cross-error tail, but it is
not a broad N5 mean-position improvement. N10/N20/N30 cross MAE also improves
(0.0147→0.0138, 0.0328→0.0205, 0.0686→0.0351 m respectively).

The separate lateral-velocity regression
(`v_dot = 0.011986 - 8.653151*v - 0.030998*u*r + 4.213381*u*tan(delta)`)
was rejected: by itself it worsened held-out cross p95 from 0.0303 to 0.0369 m
at N5 and from 0.1750 to 0.3091 m at N30. Combining it with the yaw response
also regressed cross p95 (0.0349 m at N5; 0.1965 m at N30). Do not add that
lateral term to odometry or MPC.

**Disposition:** keep the yaw pole in the MPC model candidate and reject the
fitted lateral-velocity term. This is a real, held-out prediction improvement,
but not live-race acceptance: the holdout exercises recorded Pure Pursuit
commands at up to about 8 m/s, forward speed was held fixed for isolation, and
no closed-loop lap has been run with the updated MPC. Do not recollect this
same maneuver. Next evaluate the identified speed/braking response in the
shared model, then build/run the updated controller in batchmode and measure
collision-free raceline laps. No Unity/simulator behavior was edited.

## MPC delayed-state / catch-up diagnostic — 2026-09-18

Run `sdu_apex_autodrive/artifacts/simulator_trace/mpc_yawpole_live_8mps_20260918`
is retained as a failed MPC acceptance run and a timing/control diagnostic.
The Unity player ran in batchmode. No Unity project, scene, physics, track,
vehicle behavior, or build artifact was changed. The capture plus reports is
51 MB; every individual file remains below the 100 MB repository limit. Its offline
directional localization report and matched error rows are saved beside the
source trace.

- The bridge delivered 4,759 consecutive packets with no sequence gaps. The
  simulator source interval was 25.000 ms median / 25.058 ms mean, with a
  50.005 ms maximum; every source interval remained inside the bridge's
  supported 1--250 ms range. The nominal 15--35 ms band is only a cadence
  diagnostic, not a valid hard-reset boundary.
- Downstream callbacks were not consistently current despite the 40 Hz mean:
  `/odom` host-arrival interval reached 334.6 ms and its source-header age
  reached 317.1 ms; `/current_map_pose` reached 340.5 ms / 323.7 ms. The
  `/cmd/speed` source-stamp interval reached 335.9 ms and was followed by
  2--7 ms command bursts. This is a delayed-then-catch-up pattern in the
  control path, not missing bridge sequence numbers. Bridge-to-recorder
  callback latency was 1.63 ms mean / 2.37 ms p95, so CSV callback handling is
  not established as the cause; a no-recorder A/B remains the direct test of
  recorder load.
- The MPC subscriber queues were depth 20 for odometry and 10 for map pose,
  while pose freshness used callback receipt time with a 300 ms timeout. Thus
  queued source-old states could be replayed and called fresh. MPC's hard
  source-step gate was also narrower than the bridge contract at 15--35 ms.
  The runtime now uses depth-one latest-state queues, checks both source
  stamps against a 75 ms age limit, and accepts source intervals only within
  the already-supported 1--250 ms range. Stale data still commands a safe
  stop; nominal 40 Hz timing remains measured rather than fabricated.
- The targeted recorder A/B is now complete with the same updated controller
  and batchmode player. In the recorder-off check, 120-message `ros2 topic hz`
  windows stayed around 39.9--40.3 Hz with observed maxima of 31 ms; one
  80 ms map-pose age was rejected. With the recorder enabled, 2,205 source
  packets had no sequence gaps, a 25.060 ms mean source interval, and a 48 ms
  maximum. `/odom`, `/current_map_pose`, and `/cmd/speed` host callback rates
  were 39.89--39.91 Hz; their maximum callback intervals were 55.4, 49.2, and
  52.8 ms. Source-header age p95 was 11.0 ms for odometry and 12.9 ms for map
  pose (maxima 73.5/75.2 ms). Bridge-to-recorder callback delay was 1.69 ms
  mean / 2.36 ms p95 / 5.95 ms maximum. The 300+ ms stall did not recur in
  either condition; this does not support CSV/LiDAR recording as its sole
  cause, and it does not identify the intermittent stall's original cause.
- The recorder's LiDAR callback stores range-count/min/max summaries, not the
  full range array. Both timing checks are useful evidence; do not collect
  another timing-only run unless a code or host change needs independent
  verification.
- This still is not a driving pass. In the matched recorder-on run, target
  speed fell to zero at source time 13.185 s while the raceline requested
  3.23 m/s. Immediately before that, steering was saturated at -0.5236 rad,
  odometry speed was 1.10 m/s, path CTE was 0.55 m, and heading error was
  -0.51 rad; there were no collisions. The controller remained stopped for
  the rest of the 55.3 s capture. This makes the next problem a
  model/controller failure under a tight turn and growing tracking error, not
  AMCL startup or transport timing. Do not use this closed-loop failure as a
  clean open-loop identification holdout.

**Next use of existing data:** stratify the already accepted collision-free
Pure Pursuit track traces jointly by source speed and raceline curvature, then
score the current yaw-response model at N5 (with N10/N20/N30 regressions) using
legal state/command inputs and truth only as the offline target. This directly
tests whether the constant yaw gain explains the MPC's tight-turn saturation
before changing that model. Promote a speed/curvature-dependent term only if
the untouched track holdout improves N5 along/cross/yaw errors without an
unacceptable N30 or integrated-pose regression. Use the failed MPC capture to
set the failure envelope and compare the resulting prediction there; do not
recollect the same run or add a speed floor to hide the optimizer's response.

### Existing-trace speed-dependent yaw-gain candidate — 2026-09-18

This analysis consumes the same two accepted PP traces; no simulator data was
collected. The explicit decision was whether the current constant yaw gain
should become speed dependent. The training trace supplied 1,663 excited,
valid source transitions (`speed > 1 m/s`, `abs(speed*tan(steer)) >= 0.04`);
legal `/odom` speed and `/cmd/speed` steering were used as model inputs, while
packet yaw rate was only the offline training target. The untouched
`pp_timing_repeat_20260917` trace was recursively replayed from legal
`/current_map_pose` and `/odom` states with its recorded command sequence;
packet pose/yaw were endpoint scores only. Source-stamped variable steps were
preserved. Forward and lateral speed were held at each legal rollout origin
to isolate the yaw response, matching the previous yaw-model evaluation.

The constant-gain fit was `tau=0.078184 s`, `gain=2.943070 /m`; its one-step
training yaw-rate RMSE was `0.06234 rad/s`. The speed-dependent candidate was
`tau=0.080190 s`, `gain(u)=3.055374 + 0.075620*(u-4) /m`; it only reduced that
RMSE to `0.06170 rad/s`. On the untouched track holdout:

| Model | N5 along p95 (m) | N5 cross p95 (m) | N5 yaw p95 (deg) | N30 cross p95 (m) | N30 yaw p95 (deg) |
|---|---:|---:|---:|---:|---:|
| Current constant (`tau=.087735`, `gain=3.011897`) | 0.13410 | 0.02936 | 0.910 | 0.08550 | 15.141 |
| Refit constant | 0.13410 | 0.02940 | 0.864 | 0.07280 | 14.958 |
| Speed-dependent candidate | 0.13411 | 0.02951 | 0.837 | 0.09996 | 16.209 |

Each horizon used the same eligible origins across the three models (750 at
N5, 725 at N30); the sample count differs by horizon because complete future
endpoints are required. The speed-dependent term slightly lowers N5 yaw error
but worsens N5 cross error and materially regresses both N30 cross and yaw.
The constant refit leaves N5 along unchanged and changes N5 cross by only
0.04 mm, with a 0.05-degree yaw-p95 reduction; its N30 cross score is better,
but that does not address the primary N5 position error. **Reject the
speed-dependent migration and retain the already validated shared C yaw
parameters for now; do not churn the production candidate for a negligible
N5 position change.** This is a completed model-selection step, not a request
for a repeat yaw test.

The joint N5 speed/curvature strata put the largest along error in the 6--8
m/s, `abs(curvature)=0.05--0.10 rad/m` group (39 origins; along p95 0.193 m,
cross p95 0.045 m). Across the eligible holdout, N5 along p95 was 0.134 m
versus cross p95 0.029 m. Before attributing this to vehicle speed, the same
750 legal rollout origins were scored at their start: `/current_map_pose`
already had 0.13129 m along p95 (0.04323 m MAE; -0.00929 m signed bias) and
0.03249 m cross p95. Paired origin/N5 along errors have correlation 0.947 and
the same sign on 89.5% of origins; their MAEs are 0.04323/0.04366 m and the
signed N5-minus-origin bias is only -0.00027 m. The p95 absolute along-error
increment is 0.04348 m. Thus the dominant N5 along-error tail is inherited
from the initial localized pose, with a smaller prediction increment; a
speed-response change cannot legitimately be credited with fixing the
dominant error. The speed/curvature bins describe where the combined error is
largest, not its causal source.

On those same 750 origins, raw `/amcl_pose` along p95 was 0.1295 m (MAE
0.0425 m) and cross p95 was 0.0329 m, versus 0.1313/0.0325 m for
`/current_map_pose`. The tiny along improvement and tiny cross regression do
not support replacing the controller's map-pose input with raw AMCL pose.

**Next use existing track data to separate localization from prediction:**
evaluate the causal per-scan AMCL along-track correction against the current
map-pose estimate on a development trace and untouched track holdout. The
previous live gain-1.0 A/B regressed, so do not repeat a blanket gain increase;
test a correction only if a named scan-observable feature predicts a signed
along innovation and a causal implementation improves held-out AMCL mean/
along error without worsening cross/yaw or raceline tracking. Score the
predictor again only after its start-pose contribution is separated. Use the
already recorded open-plane powered/braking speed-step traces for the later
vehicle-speed response question, and promote such a term only if its own
held-out transition and integrated-pose residuals show a gain. No generic
collection is justified by the current evidence.

### Raw-scan AMCL observability experiment — preparation, 2026-09-18

The existing accepted runs retained only LiDAR range summaries, so they
cannot answer whether the map scan likelihood contains additional useful
along-track information. An opt-in recorder sidecar now preserves each full
`LaserScan.ranges` vector as little-endian float32, indexed by the canonical
event index and source stamp. It is disabled by default; ordinary runs keep
their current recording and I/O behavior. The packer and invalid-range
preservation have a passing unit test. This instrumentation does not change
Unity or any simulator physics.

This is not authorization for another generic lap capture. The dedicated
offline consumer, `tools/model_id/analyze_amcl_scan_observability.py`, now
reproduces the configured AMCL likelihood-field against the production map.
The one planned capture has a specific decision: estimate a bounded
along-track scan-likelihood correction from the first chronological half,
then evaluate the same causal correction on the untouched second half against
`/current_map_pose`. Simulator pose is an offline score target only. Report
signed/absolute along and cross errors, yaw, source cadence, and scan-score
ambiguity that explains when the correction is trustworthy.

Use the captured data only to select a candidate if the held-out segment
improves along-track error and does not materially regress cross-track or yaw
error. Only such a candidate proceeds to a matched live AMCL/PP A/B; no
offline-only result is a production improvement. Recorder and offline-consumer
tests, including a synthetic end-to-end held-out correction case, pass.

### Raw-scan capture result and AMCL validity fix — 2026-09-18

The one targeted `amcl_scan_observability_pp_clean_20260918` batchmode capture
contains 2,221 source-stamped scans across 55.7 simulator seconds. Source
cadence was 25.094 ms mean / 25.000 ms median, with 10 intervals over 35 ms;
there were no sequence gaps. The scan sidecar and the recorder's independent
summary both show **0 finite ranges in every 1,081-range scan** (0 of 271
subsampled beams usable). This is a sensor-data failure, not evidence that the
map likelihood has no along-track information.

The first offline report misleadingly returned a zero-gain rejection because
all candidate likelihood scores were `-inf`/`NaN`; that result is withdrawn.
The analyzer now reports `invalid_no_valid_lidar_returns`, records sampled
valid/total beam counts, and does not fit or score a correction unless enough
valid scans exist. The current capture therefore has **no AMCL correction
candidate result** and must not be used to tune scan gains.

The same trace exposed a runtime confidence bug: AMCL's GPU update accepted an
all-invalid scan with uniform particle weights, then could mark an odometry-
predicted cluster as a successful map correction. `/amcl_localization_health`
reported 98.15% accepted and zero rejected despite every scan having no valid
returns. AMCL now skips particle weighting/clustering for scans with zero
usable sampled returns, marks the scan unaccepted, does not refresh the last
map-correction time, and appends sampled-valid/total counts to
`/amcl_scan_alignment`. Odom propagation remains live; the next valid scan
propagates the accumulated source-time odometry delta. This is an observability
and estimator-integrity fix, not a claim that it repairs the absent LiDAR data.

Other decisions from the same run: left/right encoder separation is rejected;
both sides freeze together under the full-brake branch, and deployed odometry
already rejects wheel updates in every labelled brake sample. Keep the current
braking coefficients; this short PP trace is not a holdout for refitting them.
The offline scan correction is not evaluated, and no generic AMCL/PP lap should
be repeated. A narrowly targeted boundary diagnostic,
`lidar_range_boundary_diag_20260918`, answered the next question: over 105
source packets (104 intervals), cadence was 25.038 ms mean, 24--26 ms, with no
sequence gaps. The decoded range payload already contained 0 finite returns
out of 1,081 at the Unity-packet boundary; the bridge then published the same
0/1,081 values to ROS. This rules out bridge conversion/publication and CSV
recording as the source of the absent returns in this run; it does not alone
prove a LiDAR code defect. The Unity log identifies the player data path as
`AutoDRIVE-ModelIdentification-racelineSpeedSweep_Data`, and its packed scene
contains `SRL 2026 ICRA Track`, so this was not the default open-plane build.
AMCL's new guard reported 0/271 usable sampled beams, and PP correctly waited
for `/current_map_pose` rather than treating odometry-only propagation as a
global localization fix.

The current checked-out `LIDAR.cs` was last modified on 2026-09-17, after this
player’s data was built on 2026-09-16, so its source cannot be assumed to match
the running binary. Current read-only scene metadata shows the track root is
active on Unity's Default layer with an enabled MeshCollider, consistent with
the current source's Default-only ray mask; this does not establish the
collider/raycast behavior in the older binary. The capture therefore answers
the bridge question, but not the exact Unity-side reason for zero hits. Before
another drive or any sensor change, rebuild the same competition-scene
diagnostic player from a recorded source revision, then make one short,
purpose-limited check that establishes whether its Unity source payload
contains any finite track returns. If it does, immediately use that capture
in the existing scan-observability analysis; only a held-out causal gain can
justify an AMCL change or live A/B. If not, inspect the matching build's
raycast/collider path next. Do not modify Unity physics, scene geometry, or
sensor behavior as part of this diagnosis. No model/encoder/braking
coefficients are changed based on the invalid scan data.

## MPC speed-response replay and focus transition — 2026-09-18

This closes the planned reuse of the two clean track traces. The
`pp_amcl_recovery_24ms_20260917` development capture fitted the causal
longitudinal response; `pp_timing_repeat_20260917` remained untouched during
fitting and is the holdout. The replay script is
`tools/model_id/evaluate_mpc_longitudinal_response.py`; it reads the current
runtime coefficients from `f1tenth_mpc/include/mpc_types.h`, so the score stays
aligned with the implementation. Its state is initialized only from
source-stamped `/current_map_pose` and `/odom`; the known recorded
`/cmd/speed` sequence is supplied as the planned input. Simulator packet pose,
heading and body speed are offline targets only. No future sensor or truth
measurement is used recursively. All horizons preserve actual Unity source
intervals. The `/odom` to packet source-time join covered 791 samples with a
maximum offset residual of 0.24 microseconds; source-step median/p95/max were
25.000/26.002/33.002 ms.

| Trace / horizon | Body-speed MAE, old → candidate (m/s) | Along MAE (m) | Cross MAE (m) | Yaw MAE (rad) | Euclidean position MAE (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development N5, 0.125 s | 0.1381 → 0.0740 | 0.0801 → 0.0793 | 0.0143 → 0.0143 | 0.0183 → 0.0183 | 0.0834 → 0.0827 |
| Holdout N5, 0.125 s | 0.1250 → 0.0731 | 0.0634 → 0.0630 | 0.0123 → 0.0122 | 0.01084 → 0.01074 | 0.0669 → 0.0667 |
| Development N30, 0.752 s | 0.2483 → 0.0841 | 0.1426 → 0.0882 | 0.0358 → 0.0354 | 0.0403 → 0.0254 | 0.1542 → 0.1019 |
| Holdout N30, 0.752 s | 0.2221 → 0.0818 | 0.1266 → 0.0783 | 0.0345 → 0.0323 | 0.0407 → 0.0263 | 0.1388 → 0.0926 |

N5 is effectively neutral for position and improves speed error by 41.5% on
the independent holdout. By N30, holdout speed MAE is 63.2% lower and
Euclidean position MAE is 33.3% lower; longitudinal error improves 38.2%,
lateral error improves 6.3%, and yaw error improves 35.3%. The small N5 pose
change is expected to be bounded by the same initial map-pose error supplied
to both models. These results support using the candidate in MPC development;
they do not establish closed-loop stability or live racing acceptance. The
reported Python recursive evaluator mirrors the C transition equations and
reads their constants directly from the C header; C unit tests separately
cover the response, target integration, braking envelope and linearization.

**Project focus decision:** pause broad odometry/AMCL tuning. Keep the
currently validated runtime settings and reject the encoder-side selector and
blanket AMCL along-gain increase. Reopen localization only if a fresh MPC
closed-loop trace shows a specific estimator error preceding controller
failure. This is a development priority change, not a claim that localization
has zero residual error.

The planned capped batchmode run is complete but did not move the vehicle;
see **Capped MPC live input-age diagnostic — 2026-09-18** below. Do not repeat
it unchanged. First implement an MPC-specific source-time state handoff that
handles the measured sensor-to-controller age without blindly relaxing
freshness or fabricating 25 ms intervals. Preserve source ordering and measured
dt; propagate a legal map-pose anchor to the latest odometry timestamp using
causal odometry/command history, and reject actual ordering faults or age
beyond a justified bound. Then rerun the same 4 m/s batchmode check. Only a
clean first lap permits a three-lap run and later increase toward 8 m/s. Keep
the default MPC disabled and do not edit Unity physics, scene, sensor behavior,
or timing.

## Capped MPC live input-age diagnostic — 2026-09-18

Run `sdu_apex_autodrive/artifacts/simulator_trace/mpc_target_response_candidate_4mps_20260918`
tested the updated MPC in batchmode with explicit test-only enable, a 4 m/s
cap and the configured first-lap ramp. It used the same read-only temporary
player build as the valid-lidar PP traces; no Unity source, scene, physics,
timing or vehicle behavior was edited. The experiment is a failed drive
attempt, but a useful input-freshness diagnosis; it is not evidence against the
identified vehicle model or localization accuracy.

- The recorder has 2,306 consecutive bridge packets and no request or telemetry
  sequence gaps. Source dt median/p95/max was 25.0/29.0/52.0 ms (mean 25.11
  ms). LiDAR returned 1,031 finite ranges in all 1,081-beam scans. The car did
  not build speed: `/cmd/speed` stayed at 0 m/s, `/odom` stayed at 0 m/s, and
  simulator applied throttle and collision count both remained zero.
- Delivery itself was regular: `/odom` arrival-gap median/p95/max was
  25.1/28.1/63.8 ms; `/current_map_pose` was 25.1/28.4/73.2 ms. However,
  their source-header ages against ROS arrival epoch were high: odometry
  78.1/85.4/207.1 ms and map pose 80.4/88.1/208.0 ms (median/p95/max).
  The MPC's configured `input_timeout_s=0.075` therefore rejected ordinary
  source-stamped states even though packet sequence and arrival cadence were
  intact. Logs explicitly report stale odometry/map-pose ages; the controller
  consequently published only zero target speed.
- This separates the immediate failure from AMCL scan validity, map loading,
  sensor cadence and vehicle-model fit. AMCL had valid scans and published map
  pose, but the MPC refuses those states under the present absolute-age gate.
  Blindly increasing the timeout would accept a state up to several control
  steps old and is not yet justified. The source-stamp-to-host offset and
  source-age behavior must be reconciled first; if necessary, propagate the
  global pose anchor to the newest odometry source time using legal causal
  odometry and command history. Keep measured source dt and never synthesize a
  nominal 25 ms step.
- The canonical `events.csv` has been retained and the derived CSV partitions
  were finalized from that stream. All individual files are below 100 MB. The
  simulator and ROS stack started for this experiment have both been stopped.

**Next use:** implement and unit-test the MPC-only timestamp handoff, replay
this exact trace through it to verify that fresh-in-order 40 Hz inputs are
accepted while genuinely delayed/reordered data still fail safe, then repeat
the 4 m/s batchmode test. Do not touch localization tuning unless that rerun
shows a specific estimator residual after the timestamp problem is fixed.
