# Practice-track minimum-time optimizer

This directory contains the IPOPT raceline optimizer for
`autodrive_practice_20260924_b.pgm`. Its job is to produce a fast, dynamically
feasible full-lap starting line. Local/sector micro-adjustments are a separate
second stage and must start from a feasible IPOPT candidate; they are not a
replacement for the global solve.

## Current state

The runtime target CSV is geometry plus requested body-forward speed and path
curvature; it contains no steering or throttle commands. The MPC loader reads
curvature from column 5 and speed from column 6 (one-based), and
`mpc_trajectory_sample` interpolates both along path progress. Export now keeps
the optimizer's curvature computed from the optimized XY path and rejects a
candidate if that curvature disagrees with an independent geometric check.
Thus the target exported to MPC cannot silently receive the older, separately
approximated curvature estimate.

> **Critical model correction — 2026-10-22:** the exact pinned practice
> `UnityPlayer.so` calls `setSubStepCount(5.0f, 5, 1)`. The previous 128 m/s
> threshold was wrong. Existing v247-v250 Phase-I residuals used that wrong
> generated wheel helper and are retained only as historical solver
> diagnostics; they are not valid feasibility measurements for the practice
> vehicle. The production Odom/MPC plant and the active IPOPT generator now
> use the verified 5 m/s branch. The corrected elastic solver is generated in
> `_working_solver_v251_<mode>`; older `_working_solver_v244`–`v248` caches and
> their provenance are frozen historical artifacts and must not be executed or
> treated as current-model results. A new corrected-model solve is recorded
> separately under `optimized_practice_v251_*_correct5_20261022/`.

> **Current source-equation audit — 2026-10-22:** the active elastic entry in
> `recover.py` builds a revised v251 candidate. It distinguishes yaw-rate
> acceleration from Euler yaw-angle rate, includes the source-derived 3-D
> tire/suspension wrench and normalized tire-load filter, and consistently
> uses tilt-projected horizontal velocity for path progression and optional
> lateral trackability checks. Candidate generation fails closed unless
> Unity's `Fixed Timestep`, production `kDt`, and damping source agree with the
> constants used in the transcription. Wheel ray geometry is cross-checked
> against the active practice prefab; this removed an inherited 10° front
> quaternion reconstruction where the prefab specifies 5°. Seven focused
> equation/source checks pass. This
> remains continuous spatial collocation, not the Unity 1 ms / 25 ms
> command-held plant. The latest completed 1,200-iteration solve used the
> earlier v249 equations and failed with a 0.237 maximum dynamics defect. No
> new solver or simulator run has been started while transcription gaps remain.

- The active entry point is `run_ipopt.py`; setup and model provenance are
  recorded per run in `provenance.json` and `ipopt_report_pass_*.json`.
- The IPOPT problem is restored from commit
  `ca120e0cde7ae60e70082d38f86c5931f79f148f`. Historical source is kept
  unchanged; source-matched modifications are generated in the versioned
  `_working_solver_v251_<mode>` cache. Phase-I prints a flushed diagnostic
  checkpoint every 10 IPOPT iterations: candidate acceptance, step size, best
  elastic merit, and best raw dynamic defect. Retained checkpoints prioritize
  the maximum raw dynamic defect, matching the strict feasibility gate; merit
  is only a tie-breaker. The log cadence is intentionally
  bounded; it exposes stagnation without flooding the terminal.
- The generated model verifies roll/pitch contribution to tire contact-point
  velocity and now uses the pinned practice player's `setSubStepCount(5.0f,
  5, 1)` schedule: five substeps below 5 m/s, one at or above 5 m/s. It remains
  a spatial collocation model, not a 1:1 copy of Unity's complete native
  WheelCollider/contact solver.
- The optimizer contact-ray geometry is bound to production wheel-center,
  transform, COM, and suspension constants, then checked against the active
  serialized practice prefab. This caught a 0.095 mm stale front transform
  height copied from the old reset snapshot; the shared plant and reset state
  now use the prefab's `0.060 - 0.06434 = -0.00434 m` value. The front force
  point is derived from that transform height minus wheel radius. No Unity
  simulator physics or controller behavior was changed.
- The practice scene explicitly overrides each WheelCollider's forward-tire
  stiffness (`0.8`) and extremum value (`0.9`). They currently equal the
  prefab values, but the previous source guard rejected rather than applied
  them. The loader now constructs scene-effective tire curves and fails closed
  on unhandled wheel-physics overrides. Eight focused equation/source tests
  cover this path; they do not validate optimizer feasibility.
- Torque provenance is checked against the serialized practice scene, not the
  prefab default: the F1TENTH prefab defaults to 85.6 Nm, while the practice
  scene overrides `MotorTorque` to 428 Nm. With `CAWD`/`CAWB`, full positive
  command gives 107 Nm motor torque per wheel; exactly zero command applies
  428 Nm brake torque to each wheel. `source_vehicle.py` now binds this scene
  override to the C++ plant constant, verifies the `VehicleController.cs`
  torque branches, and includes controller/scene/prefab sources in the model
  fingerprint, so a future source mismatch fails closed.
- The loader also pins the active controller path (`DrivingMode=1`, CAWD,
  CAWB, front-wheel Ackermann), 30° steering limit, 183.346°/s steering rate,
  324/236 mm Ackermann geometry, and Rigidbody mass/drag/angular drag/gravity/
  dynamic state against the production constants. The serialized controller
  `WheelRadius=0.0325 m` is explicitly recorded as inactive for this CAWD
  branch; actual tire radius comes from the WheelColliders (`0.059 m`).
- A candidate is rejected unless IPOPT passes the zero-elasticity source-
  dynamics gate. If that passes, the time-minimization stage runs and exports
  a geometry/speed/curvature target after the swept-map footprint check. The
  optional source-native replay exercises only the optimizer's internal
  control witness; it does not drive or score the target with MPC, and a
  failed witness replay does not erase the target. `--skip-native` still
  exports the target, explicitly marked as unvalidated in MPC or Unity.
- The reusable native seed has 143 nodes at 0.1987 m spacing. When the target
  mesh differs, it is periodically resampled only after its map/centerline
  hashes and lap length are checked. A 0.12 m diagnostic mesh reduced the
  initial roll-rate, pitch-rate, and yaw-rate collocation maxima by about
  39%; that is seed-discretization evidence, not yet a feasible IPOPT lap.
  Algebraic force/load projection does not make the recursive trajectory
  dynamically consistent.
- There is not yet a verified IPOPT-produced sub-5-second line. Existing
  `v292`/`v300` search outputs are explicitly marked not promoted; they remain
  evidence/comparators, not production refiners. Preserve the refinement stage
  for use after IPOPT has a valid full-lap baseline.
- The observable `.12 m`, three-brake-zone limited-memory repeat
  (`optimized_practice_v247_phase1_mesh12_limited_3brakes`) completed 1,200
  Phase-I iterations in 218.1 s and still failed the strict feasibility gate.
  It accepted only six checkpoint improvements: merit moved from 7.68850 to
  7.45287, but maximum dynamic defect worsened from 0.46576 to 1.37123; the
  worst final residual is lateral force balance at node 122 (1.371 m/s²).
  Thus IPOPT was iterating, but its accepted feasible checkpoints plateaued
  early and did not produce a valid line. The run had 18 injected full-brake
  nodes absent from its native seed, so the next controlled diagnostic removes
  those nodes to isolate drive-only consistency. Two exact-Hessian attempts
  produced no completed-iteration callback (one ran 37m40s), making that mode
  impractical at this problem size; it is not counted as a solve result.
- The controlled zero-brake diagnostic
  (`optimized_practice_v248_phase1_mesh12_no_brakes`) completed 1,200
  iterations but also failed strict feasibility. Removing the injected brakes
  reduced seed longitudinal mismatch from 5.385 to 0.566 m/s² and removed the
  41.17 rad/s wheel-lock discontinuity. However, L2-only checkpoint selection
  still accepted candidates whose maximum defect rose from 0.46576 to 0.70821
  (worst: lateral force balance, node 36), despite merit falling 7.10822 to
  6.96890. Zero-brake runs are diagnostic, not racing lines. This falsifies
  “the artificial brake seed is the sole cause” and motivates max-defect-first
  retention plus restarts from that best checkpoint.
- The first observable max-defect-first continuation
  (`optimized_practice_v249_phase1_mesh12_no_brakes_maxfirst`) ran three
  bounded Phase-I attempts, each for 1,200 IPOPT iterations. The retained
  maximum raw defect moved from 0.46576 to 0.43785, 0.43716, then 0.43619;
  this is a real but small 6.35% reduction, not feasibility. The final worst
  residual is roll-rate collocation (0.43619), closely followed by lateral
  force balance (0.43188). The third attempt accepted one backtracked
  checkpoint near iteration 150, then had no accepted checkpoint for its
  remaining 1,050 iterations. Its live 10-iteration status output made that
  stagnation visible while the solve was active. No time-optimal stage, native
  replay, raceline export, or simulator run occurred. The retained-checkpoint
  metric now matches the strict maximum-defect gate, but IPOPT still minimizes
  its original elastic sum-of-squares objective; objective/search alignment
  remains the next diagnosed limitation.

## Reproduce the current practice setup

Activate this directory's `.venv`, then check imports, source parameters,
map/raceline pairing, and generated model without running either solver or
simulator:

```bash
.venv/bin/python run_ipopt.py --setup-check --skip-native \
  --output optimized_practice_setup_check
```

The optimizer's default extra wall-clearance margin is read from the active
`f1tenth_mpc/config/mpc_competition.yaml` profile, so a change to the runtime
corridor is reflected in the target search and recorded in `provenance.json`.
An explicit `--wall-margin` overrides that value and is called out in the
report. The checked-in practice profile uses 0.05 m; do not silently substitute
the older 0.15 m default.

Run initial-seed diagnostics only (no IPOPT solve or native replay):

```bash
.venv/bin/python run_ipopt.py --diagnose-initial-only --skip-native \
  --seed-file optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --seed-backend native --mesh-spacing .10 \
  --output optimized_practice_seed_diagnostic
```

The optimizer's command witness is restricted to the production MPC input
envelope: `MPC_THROTTLE_MIN_DRIVE` through full throttle. Unity's exact-zero
full-brake mode is deliberately excluded because the production MPC cannot
command it. The optimizer therefore cannot rely on an unavailable brake
command when constructing its speed/curvature target. This is still finite
spatial collocation against the source-derived plant, not a guarantee of
continuous-time or Unity-contact-solver feasibility; steering/throttle in the
witness CSV are feasibility variables, not runtime commands. At the default
0.10 m knot spacing, source feasibility is evaluated at knots and midpoints,
so the largest spatial gap between those checks is 0.05 m; the report records
that gap and explicitly marks continuous feasibility as unproven.

Warm-start `ax_mps2` and `ay_mps2` are the active v251 OCP's body-coordinate
velocity derivatives `du/dt,dv/dt`; its force balances include rotating-frame
transport terms and its collocation propagates these state derivatives
directly. New files carry the `body_velocity_derivative_v1` marker. Historical
unmarked seeds stored inertial acceleration resolved in body axes and are
converted once on load using `du/dt=ax+r*v`, `dv/dt=ay-r*u`.

Run the full IPOPT feasibility and time solve, followed by the source-native
witness replay if desired. That replay is diagnostic only, not proof of Unity
parity or MPC tracking; final validation requires the practice simulator/MPC:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python run_ipopt.py \
  --seed-file optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --seed-backend native --mesh-spacing .10 \
  --feasibility-iterations 1200 --iterations 1200 \
  --native-correction-passes 0 --output optimized_practice_candidate
```

## Acceptance and next stages

1. Get a full-lap Phase-I solution with the source-dynamics gate passing.
2. Run IPOPT's minimum-time stage and inspect time, swept corridor clearance,
   speed, curvature, steering/throttle feasibility witness, and constraint
   residuals.
3. Use source-native replay as a model diagnostic, then validate the target
   with the actual practice simulator/MPC. A model-predicted lap time alone is
   not a driving result.
4. Only after a feasible global candidate exists, run the separate
   micro-adjustment/sector search. Compare its changes against that candidate;
   retain only changes that pass the same native and simulator gates.

Competition rule compliance is enforced by the runtime/controller topic-policy
checks, not by this offline optimizer. Ground truth and simulator-only signals
may be used for offline model validation and must not enter the competition
controller path.

### Opt-in native contact diagnostics

The retained low-speed practice seed did not preserve the native per-wheel
contact callback or the returned solver/body-math snapshots, so the remaining
force residual cannot be split reliably from that seed. If a later, specifically
approved practice-only capture is needed, this explicit option stores those
channels plus the exact 25-step steering-rack and per-wheel command schedule
(steer angle, drive torque, brake torque), packet index, command, state, and
backend fingerprint:

```bash
.venv/bin/python run_ipopt.py --capture-native-diagnostics \
  --output optimized_practice_contact_diagnostic
```

This option runs the native warm-start trial and then exits before IPOPT or the
native witness replay. It is not offline-only, and it was not run while preparing
this change. Output is JSONL under `warmstart_physx/`, with a manifest describing
field semantics and counts. Contact callback values and returned post-stage
body state share a 25 ms packet association, but the native protocol does not
expose the callback's exact 1 ms substep timestamp; the manifest explicitly
prohibits treating them as simultaneous. Empty capture files are retained when
a trial fails before producing samples, so absence of records is distinguishable
from an unrequested capture.

## Focused code checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/pytest -q tests/test_v231_projection.py \
  tests/test_v2316_contact_velocity.py \
  tests/test_v236_wheel_substep_index.py \
  tests/test_v237_dual_monotone_and_breakdown.py \
  tests/test_v240_phase1_campaign.py
```

These tests cover solver-source transformation, seed validation/projection,
wheel/contact equation wiring, and fail-closed acceptance. They do not certify
minimum lap time, native replay, or Unity driving.
