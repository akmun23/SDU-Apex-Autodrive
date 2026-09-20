# MPC implementation continuation note — updated 2026-09-19

## Current objective and constraints

Continue the 2026-09-18 MPC coding-agent handoff and finish implementation before attempting to make the car drive. The current goal is a complete, internally consistent MPC with evidence-backed offline readiness; live driving is not yet assumed safe or required.

- Do not edit Unity, simulator physics, scenes, or simulator behavior.
- Do not use timing jitter as a rejection or launch blocker. Keep timing as diagnostic evidence. Investigate delay if a collision occurs.
- Keep runtime inputs within the competition-legal interface; simulator truth remains offline-only.
- Preserve 40 Hz / 25 ms prediction, N=30 (0.75 s), adaptive rho, and the existing 100-iteration cap. Do not tune weights before the model/diagnostic work is complete.
- When a later live test is justified, use the ICRA image in batchmode (not `-no-graphics`). Do not start a GUI.
- The active controller-facing raceline is the dense export
  `f1tenth_planning/trajectories/autodrive_mintime_exact_dense/autodrive_mintime_raceline.csv`.
  It contains 2,661 points at approximately 0.02 m spacing. The sparse
  optimizer output remains the source solution, not the active runtime file.

## Correction: prior multi-lap result and raceline baseline — 2026-09-20

The earlier claim that MPC had completed multiple laps at approximately
12.5 s is not supported by the preserved authority artifacts. The documented
12.20/12.09/12.26 s lap splits belong to the Pure Pursuit run and that run had
four collision-count increments. The separate 4.0 m/s MPC authority probe
cannot be a 12.5 s lap: a 53 m lap at that cap would require an average speed
above the cap. Its preserved `controller_trace.csv` contains
`/mpc_shadow/diagnostics`, not `/mpc/diagnostics`, so it is shadow/PP evidence,
not MPC command-authority evidence. The later 12.21 raceline-traversal claim
also has no matching raw `/mpc/diagnostics` authority artifact and must remain
unverified until reproduced with an authority trace.

The current accepted raceline is also not the same file used by the earlier
tests:

| baseline | samples | length | critical-corner speed at s=34--36 m |
|---|---:|---:|---:|
| older FTG mintime | 2,587 | 51.718 m | 4.57 to 2.82 m/s |
| current exact mintime | 313 | 53.106 m | 6.34 to 3.75 m/s |

The current exact profile enforces an approximately 8.0 m/s²
`abs(kappa)*vx^2` lateral-acceleration proxy through most of the lap and asks
for substantially more entry speed in the corner where current MPC authority
fails. This is a real controller-operating-point change, not merely a file
sampling change. A current-code authority A/B with the older raceline and a
4.0 m/s diagnostic cap reached s=52.38 m before its terminal collision; the
current exact-raceline run reached s=38.14 m. That A/B is diagnostic only—the
older path was not a clean full-speed acceptance—but it confirms that the
raceline/speed profile materially changes the failure point.

From this point onward, only `/mpc/diagnostics` plus `/cmd/speed` with
`controller:=mpc` counts as MPC driving evidence. `/mpc_shadow/diagnostics`
and PP lap times are kept as separate baselines and must not be reported as
MPC authority results.

## Work completed in this session

This note has been updated after the repository cleanup pass. The active
runtime path is now the only MPC implementation; the legacy controller,
compatibility library, and utility-math path described by the original note
have been removed.

1. Read the supplied coding-agent handoff and carried forward its ordered priorities: isolate the obsolete solver, build rejection diagnostics, verify CT2 semantics, analyze existing applied-command replay before gathering more data, implement adaptive same-sample RTI, instrument adaptive rho, and only then optimize/tune.
2. Added an MPC-only AMCL warm-up delay in `sdu_apex_autodrive/launch/controller.launch.py`. Default is 2 seconds from the AMCL process-start event; AMCL keeps processing scans during the delay. `mpc_start_delay_sec:=0` disables it. Pure Pursuit/FTG do not use this delay. If localization is explicitly disabled, MPC launches without waiting for AMCL. Launch behavior has only had Python syntax validation, not a live launch test.
3. Removed the superseded `f1tenth_mpc/src/mpc.c`, `include/mpc.h`,
   `src/util_math.c/.h`, and their old regression test/benchmark entirely.
   The CMake package now builds and installs only the active RTI MPC path.
4. Began the RTI implementation in `f1tenth_mpc/include/mpc_rti.h`, `src/mpc_rti.c`, `include/riccati_solver.h`, and `src/riccati_solver.c`:
   - Added R1/R2/adaptive mode and candidate/trigger/solver telemetry structures.
   - Added an exact nonlinear candidate objective/corridor-slack scorer, R1 pass helper, and adaptive-trigger calculations.
   - Added rho start/change-count fields and enabled Riccati profiling in the core build.

## Current implementation state

R1/R2/adaptive RTI is wired into `mpc_rti_solve_cycle()`, including
same-sample R2 seeding, feasible-candidate selection, fallback to R1, and
per-pass telemetry. The active 7-state nonlinear model and 9-state augmented
Riccati path are the only controller/model implementation. The ordinary model
rollout uses a scalar path; automatic differentiation is retained only for
Jacobian construction. The active Riccati backward/RHS passes exploit the
known 7+2 augmented layout, while generic dense factorization remains the
reference factorization because an experimental specialized factorization did
not preserve live ADMM convergence.

## Validation performed

- `git diff --check`: passed.
- `python3 -m py_compile sdu_apex_autodrive/launch/controller.launch.py`: passed.
- Humble container build: `colcon build --packages-select f1tenth_mpc --merge-install`: passed. The old unused-helper warning no longer applies after RTI integration.
- `ctest --test-dir build/f1tenth_mpc --output-on-failure`: 9/9 passed.
- No simulator or controller launch was started for this session. The only active container observed was the development container `sdu_apex_autodrive`.
- No current-session ROS 2/Python test suite beyond the MPC CTest suite was run.

## Current cleanup validation — 2026-09-19

- Humble container build completed without compiler warnings for the active
  package; `ctest` passed 8/8.
- A 5,000-cycle N30/40 Hz analytic-Jacobian benchmark with prefactorization
  completed with 5,000/5,000 optimal cycles; solve time was approximately
  0.321 ms p50, 0.334 ms p95, 0.338 ms p99, and 0.801 ms maximum on the
  development container.
- The existing live event replay remained 484/484 accepted with zero
  rejection, zero regularization, and 30 steps at 25 ms. This is offline
  evidence only; no new simulator behavior or physics was changed.
- `cppcheck` found no correctness warnings in the production sources. Its
  remaining notes are parameter-name style differences in public C function
  declarations, not runtime defects.

## Live authority investigation — 2026-09-19

- Collision handling is now terminal for authority tests: the actuator sends
  one neutral command, latches the first nonzero collision count, exits with a
  failure status, and the launch shuts down the remaining stack. Samples after
  the first collision are invalid for controller analysis.
- An ICRA batchmode authority run with the independent 40 Hz bridge cadence
  received 347 packets in 9.49 s. The valid pre-collision stream had a 25.96
  ms mean interval and a 25.07 ms median; one delayed interval reached about
  340 ms. This confirms a near-40 Hz source stream with transport jitter, not
  a sustained 10 Hz source.
- The command-history delay experiment (`command_actuation_delay_s=0.225`)
  is rejected. It produced 67 actuator-command steps above the configured
  0.08 rad/25 ms MPC steering step and sign-reversing jumps up to about 0.85
  rad in `/cmd/speed`. The normal no-delay authority trace stayed within the
  0.08 rad step at the controller boundary. The runtime configuration is
  therefore back to `command_actuation_delay_s: 0.0`; the predictor option and
  regression remain available for offline experiments only.
- The remaining live root-cause target is the transport/application queue:
  simulator feedback followed an older steering command by approximately two
  bridge packets near the collision, while the legal runtime has no applied
  command sequence. The next change must model that causal uncertainty without
  shifting the current MPC state discontinuously. AMCL was within roughly
  0.01–0.04 m of simulator truth immediately before the collision in the
  recorded run, so this evidence does not point to AMCL as the cause.

## Earlier evidence to retain (not new validation)

- Pure Pursuit ran the accepted mintime raceline in ICRA batchmode. Initial complete lap splits were about 12.20, 12.09, and 12.26 seconds; later laps slowed. Four collision-count increments occurred. One was close to a long response-arrival interval; the other three had ordinary intervals. Existing packet logs cannot prove which applied command/physics frame caused each collision. Do not describe the PP run as collision-free acceptance.
- An earlier MPC authority probe remained stationary: RTI rejected nonlinear rollout at stage 24, followed by residual rejections. This was not a timing rejection. No further MPC live run has been made after the instruction to finish implementation/offline readiness first.

## Remaining actions

1. Keep the generic dense factorization as the numerical reference until a
   replacement is proven equivalent over live ADMM cycles, not just matrix
   coefficient parity.
2. Keep production diagnostics bounded; replay/benchmark CSV writers remain
   test tools and are not part of the ROS 40 Hz node.
3. Treat a new batchmode simulator run and collision-free multi-lap result as
   separate live acceptance evidence. The cleanup/build/replay results above
   do not claim that acceptance.

## Worktree handling

The worktree was already broadly modified before this session; retain all user
changes. In particular, preserve the user's deletion of
`f1tenth_planning/trajectories/autodrive_mintime_exact_v13.zip` and all
simulator-trace artifacts. Do not reset/clean the repository. This cleanup
changed only MPC source/tests/docs; Unity/simulator files and physics were not
modified.

## AMCL correction validation and MPC rejection isolation — 2026-09-19

The earlier AMCL failure path was isolated from the first clean slow-pole
authority run. After about 255.5 m, AMCL rejected a 0.201 m local scan
correction against the 0.200 m post-phase limit, fell back to odometry, and
then lost the local cluster. The correction implementation was changed to
bound the correction actually applied to the published pose while retaining
the scan update. The existing association, raceline-alias, yaw, and recovery
gates remain active. This does not loosen the localization acceptance gates.

The localization package was rebuilt in the Humble container and its three
tests passed. A fresh valid MPC-authority run used the ICRA image in
batchmode, Xvfb for Unity's graphics support, the mintime raceline, the 4.0
m/s test cap, MPC enabled as the authority controller, and terminal collision
safety. It ran for approximately 187 s / 649.5 m / 12.21 raceline traversals
with zero collision-count increments. Source command rates were approximately
39.96 Hz median for speed and steering. The recorder showed no finite Pure
Pursuit command fields, so this was not a shadow-only run.

That run measured AMCL error p95 0.178 m (maximum 0.359 m), current-map error
p95 0.114 m (maximum 0.293 m), and odometry/EKF error p95 about 3.635 m. The
old AMCL rejection/no-cluster chain did not occur: its two diagnostic counts
were both zero. The new bounded-correction log count was also zero, meaning
the new clamp branch was not directly exercised in that particular run; the
result validates the absence of the previously observed failure chain, not
every possible correction case. Two local raceline-alias rejections were
recorded and were handled by causal odometry as designed.

The same run contained 66 MPC rejected cycles while still remaining
collision-free: 27 residual rejections and 39 nonlinear-rollout rejections.
The rejection logger was then extended with the missing cause fields and a
targeted fresh authority capture was performed. Its first 15 rejections show
the decisive split:

- nonlinear rejections have `reason=5`, which is
  `MPC_RTI_ROLLOUT_CORRIDOR`; R1 and R2 both fail the exact corridor check,
  typically at stages 18–29 of the 30-step horizon;
- residual rejections have `r1=status=4`, which is
  `MPC_RTI_CYCLE_REJECTED_RESIDUAL`, with no nonlinear failure reason;
- the residual rejections are therefore not caused by AMCL uncertainty or a
  simulator timestamp gate, while the nonlinear rejections are not generic
  solver crashes or invalid model values.

This is the current root-cause boundary. The next MPC change must address the
feasible-corridor/model mismatch and the separate ADMM residual convergence
cases. Corridor limits and residual tolerances have not been loosened, and no
fallback result is being counted as an MPC solution. The latest targeted run
ran for about 85.4 s / 284.1 m / 5.34 track lengths with zero collisions and
was stopped after diagnostic capture. It is useful rejection evidence, but is
not being promoted as full-speed acceptance. The compact evidence file is
`sdu_apex_autodrive/artifacts/mpc_authority_debug_20260919/ground_truth_reject_diag_4mps.csv`.
The simulator and all ROS nodes started for that capture were stopped.

The oversized high-rate sensor CSV from the earlier long run remains outside
the repository at `/tmp/autodrive_oversized_sensor_record_20260919.csv`
inside the development container; it was not restored because it exceeded the
repository's 100 MB file limit. The compact ground-truth CSVs remain the
reproducible evidence files.

## RTI iteration, fallback, and initial-state isolation — 2026-09-19

The current live authority investigation tested the solver and failure path
without changing Unity or simulator physics:

- On the same 800-solve offline event replay, the accepted counts were 0/800
  at 5 iterations, 10/800 at 10 iterations, 472/800 at 25 iterations,
  552/800 at 50 iterations, and 738/800 at 100 iterations. The 100-iteration
  cap is therefore necessary, but it does not by itself solve the residual or
  nonlinear failures.
- Strict R1 at 100 iterations accepted 329/800. Same-sample R2 and adaptive
  R1-to-R2 both accepted 738/800 on this replay. Adaptive R2 remains enabled;
  it is not being removed for a lower-iteration fallback.
- The earlier raceline-curvature/error feedback fallback was removed. A
  rejected candidate is not an MPC command and is no longer replaced by an
  unmodelled steering law or held stale authority command.
- An authority rejection now publishes neutral through `publish_stop()` and
  resets the RTI memory. The rejection remains logged and is not hidden. The
  separate collision terminal latch is unchanged. A short ICRA batchmode
  verification produced residual and nonlinear rejection logs, no collision,
  and no continued stale-steering authority before controlled shutdown.
- The current source already caps the complete kinematic raceline reference
  (`u` and reference yaw rate) by the active speed ceiling; no additional
  speed-reference patch was needed.
- The standalone Riccati default was aligned from 50 to the production 100
  iteration cap in `f1tenth_mpc/include/mpc_types.h`, removing the remaining
  misleading solver default.

The first live reject occurs at the off-raceline start state, around
`e_y=-0.425 m`, `e_psi=-0.269 rad`, and near-zero measured speed. Later
rejections can repeat while the vehicle is decelerating or stationary. This
means the immediate blocker is not a 10 Hz source stream, high-speed vehicle
response, or an acceptable fallback command. The remaining work is to make
the exact virtual-car rollout, initial-state/corridor linearization, and
ADMM residual convergence agree. Weight tuning is deferred until that
feasibility boundary is fixed; otherwise it would only select a different
failure branch.

## Dense raceline update — 2026-09-20

- Added a controller-facing periodic linear export at 0.02 m spacing. The
  active file has 2,661 rows, approximately 53.21 m geometric length, and
  remains below 100 MB. It preserves the solved optimizer path fields without
  fitting a new spline or inventing curvature.
- Switched the controller, localization heading prior, MPC fixtures, and
  README defaults to that dense file. Humble CTest passed for MPC (8/8) and
  localization (3/3).
- A fresh batchmode ICRA MPC-authority run loaded all 2,661 points, then
  collided during the first corner. Collision termination worked correctly.
  Therefore the dense export is mechanically valid but not yet a closed-loop
  acceptance result; the active MPC still needs correction before claiming a
  usable full-speed raceline.
- A direct 0.14 m solved-mesh attempt ran beyond 2,200 IPOPT iterations and
  oscillated around 9.4618 s without convergence. It was not promoted. This
  indicates that simply increasing the optimizer mesh density is not, by
  itself, a validated solution to the live tracking failure.

The dense export was then audited at the lap seam. Its coordinates and 0.02 m
spacing were continuous, but the first heading value had been interpolated on
the wrong angular branch during the final closing segment. That produced a
false multi-radian heading jump over the last 0.10 m. The exporter now closes
the unwrapped heading on the nearest equivalent 2*pi branch, and the active
2,661-row CSV was repaired accordingly: segment-heading error is now p95
0.0337 rad and maximum 0.0974 rad, with no seam discontinuity. This is a
raceline/export fix only; no simulator or vehicle physics was changed.

The corrected exact dense line was also tested under live batchmode Pure
Pursuit authority. It loaded all 2,661 points, but collided during the first
lap at about 28% progress; the older 2,586-point dense track raceline likewise
collided at about 8% progress in a separate clean run. These are terminal
collision results, not successful-raceline claims. The two traces are kept as
`pp_exact_dense_raceline_validation_20260920` and
`pp_old_dense_raceline_validation_20260920`. The new dense export therefore
solves the point-spacing/seam problem, but it has not yet established a live
usable raceline; the next raceline work must use the actual collision location
and commanded trajectory, not only offline wall clearance.
