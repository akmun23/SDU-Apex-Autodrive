# IPOPT structural investigation — 2026-10-28

## Current objective

Keep the 51.030 s ten-scored-lap MPC setup frozen while finding a substantially
better, source-feasible and command-realizable raceline. More iterations alone
are not the proposed solution. A max-iteration exit proves neither optimality
nor a physical lap-time floor. The frozen setup is specified in
`RACING_51P030_FREEZE_20261028.md`.

## Work in progress, in order

1. Progress-aligned accounting of the roughly 155 ms/lap difference between
   the 4.9483 s target integral and 5.1030 s achieved mean. Separate speed,
   outside-path distance, heading/lateral motion, and unexplained timing/
   localization residual. Accounting is not itself causal identification.
2. Audit v354 active constraints, duals, stationarity and line-search behavior.
   Distinguish artificial trust bounds from physical limits and poor scaling.
3. Evaluate a source-plant witness held at 25 ms command epochs. Spatial
   collocation feasibility does not certify Unity/actuator executability.
4. Make smooth geometry, steering and source dynamics consistent inside the
   OCP, not by smoothing a purportedly certified result afterward.
5. Compare bounded-budget starts from different geometry basins, using real
   feasibility and progress evidence; not another thousands-iteration sweep.
6. Feed a validated executable witness into MPC initialization/feedforward
   only after the witness passes; never treat it as an unconditional command.
7. Controlled repeated ten-scored-lap trials only after a justified candidate.

## Evidence already established

- v354 uses 474 nodes, 16 physical/collocation states, steering-rate and
  throttle controls: 8,532 decisions and 33,654 constraint multipliers in the
  saved time-stage checkpoint. Factor-4 best saved feasible sample is
  4.922087 s, not an IPOPT KKT-converged or Unity-validated racing result.
- Current componentwise trust bounds divide the aggregate budget by the
  number of normalized coordinates before taking a square root. Thus a fixed
  factor is mesh-dependent and does not mean the geometry is unrestricted.
  Whether these rows are actually active must be measured before attributing
  the plateau to them.
- Solver-coordinate scaling exists but is disabled by default through
  `SDU_IPOPT_VARIABLE_SCALING=0`. Prior v356 short screening saved a feasible
  4.923962 s sample in 45 iterations with scaling; this motivates controlled
  auditing, not claiming that scaling alone solves the racing objective.
- Existing snapshots save primal decisions and constraint duals but not all
  row labels, bounds, residual values or stationarity. Add reproducible audit
  data rather than guessing active limitations from one residual number.

## Cleanup and verification status

Speed experiments are stopped; no simulator/controller is currently running.
v354 continues independently while its checkpoint and numerical behavior are
inspected. Do not interrupt it or delete its seed/current generated root casually.

429 obsolete generated directories were moved to recoverable desktop Trash:
old colcon logs, retired handoff builds, two scratch MPC builds, and unused
limited-memory generated source copies. Retained current log targets, current
build/install, venv, oracle dependencies, all record bags and live optimizer
checkpoints. Root colcon logs fell from about 4.5 GB to 127 MB. Trash must be
emptied separately to recover physical free space. Retired generated sources
also have a local tar backup in `live_runs/repo_cleanup_20261028/`.

364 generated output/cache files were removed from Git's index, not deleted
from disk. These include live checkpoints: their ongoing writes remain intact.
Six required optimizer source modules were wrongly ignored; explicit source
exceptions now make them exportable. Added tracked `COLCON_IGNORE` boundaries
for optimizer/data/vendor/scratch trees: clean discovery now reports exactly
the five real ROS packages, with no accidental NumPy/Cython builds.

Verification so far: MPC 12/12 CTest targets; localization/AMCL 8/8; tools
141 unit tests. Full historical optimizer suite: 571 passed, 17 failed, four
warnings. Failures must be diagnosed as current regressions versus retired
native/Phase-I contracts; do not delete a failing test merely to get green.
Detailed logs: `live_runs/repo_cleanup_20261028/`.

GitHub HTTPS lacked credentials, but existing SSH authentication succeeds.
The repo's push URL now uses SSH to the same repository; dry-run passed.
Final source commits/push and post-push verification remain pending.

## Implemented investigation and structural changes

### Progress-aligned record accounting

`tools/analyze_racing_time_loss.py` accounts for all ten scored laps of the
51.030 s record on one-metre progress sectors. Its identity separates forward
speed, outside-path distance, heading/lateral motion and an explicit residual.
It never silently discards negative progress jumps or receipt-clock offsets.
Five deterministic tests passed. This is controller-state kinematic accounting,
not independent ground-truth causal attribution.

Per-lap gap to the 4.94827949 s reference is 154.72 ms. Contributions: 14.56 ms
forward-speed shortfall, 125.86 ms outside-path travel, 3.89 ms heading/lateral
motion, 10.52 ms timing/projection remainder, +1.13 ms boundary/quadrature and
-1.23 ms receipt-versus-Unity clock. Closure error is below 1e-15 s. Worst
one-metre sectors start at progress 1, 14, 13, 2, 15 and 0 m.

### Exact saved-NLP audit

Added `ipopt_nlp_audit.py` and `run_ipopt.py --audit-time-checkpoint`.
It rebuilds the original feasible-seed graph without solving, checks saved
primal dimensions, objective and violation, evaluates objective/Lagrangian
gradients, groups constraint activity/duals/complementarity, and exports all
rows, bounds and exact graph time densities. Audit rejects a non-feasible
starting seed instead of launching Phase I. Old checkpoints reconstruct their
old trust geometry explicitly; new snapshots record that geometry.

v354 factor-4 best point reproduces 4.92208702091 s and violation
0.0007316668301. Physical-coordinate stationarity infinity norm is 0.135814;
this is NOT IPOPT's internally scaled `inf_du`. Trust radius is 0.0216523
normalized, allowing only 5.41 mm in lateral offset at factor 4. None of those
rows is strictly active at 1e-5 tolerance, although the nearest is 4.71e-5 from
its bound. Do not attribute the whole plateau to an active trust bound or call
this a physical floor. Signed-dual/complementarity diagnostics are nonzero too.

Implemented mesh-independent per-node trust budgets, preserving the old
aggregate definition solely for explicit checkpoint reconstruction/comparison.
Added adaptive barrier selection and retained coordinate-scaling control.
These change numerical search, not physics/feasibility thresholds. The first
50-iteration comparisons did NOT produce a better feasible export: adaptive
search reached 4.899696 s with violation 0.003814, still above 0.001. A subsequent
60-iteration sampled run also failed; the original feasible seed was correctly
retained. A wider box alone is not the solution.

CLI solves now checkpoint intermediate feasible samples by default; short
screening samples every five iterations. Previously only the unattended shell
wrapper enabled that behavior, so a normal CLI solve could lose a useful
intermediate point at max-iteration termination. Failed/infeasible points remain
separate and are never promoted.

### Source-command realizability and adapter correction

Added `command_realizability.py`: independent 25/100/250/750 ms fully recursive
windows over every 25 ms lap phase, with actual positive throttle and held rack
targets. It requires exact saved graph time densities; rounding/rebuilding the
mesh selected different sides of coordinate knots and changed the time integral
by 0.278 ms. That difference is not fitted away.

For the 4.922087 s point, simple target-at-start/middle/end conversions give
750 ms XY RMSE 12.10/6.07/13.61 cm and maxima 55.63/17.95/42.62 cm. They fail
the 5 cm requirement. Source-derived rack-mean inversion was investigated;
parity tests check the actual asymmetric steering branch and exact floating
operation grouping. It did not remove the coupled rollout problem, so no such
witness was fed into MPC.

Fixed `physics.SourcePlant.step`: it now uses the production plant's substep-
integrated rear-axle displacement and yaw, rather than re-integrating endpoint
COM twist. The latter discarded accurate integration and used the wrong lateral
origin. This changes the offline adapter, NOT production odometry/plant physics.
Added 1 ms replay support for independent perimeter checking.

Normal spatial exports now run the production command-realizability diagnostic
and carry its result explicitly, rather than silently equating spatial
collocation feasibility with an executable command witness.

### New production-code command OCP: actual structural progress

`command_ocp.py` + `command_ocp_kernel.cpp` provide a separate periodic
command-feasibility polishing problem. They use the SAME C++ transition and
forward-mode derivatives as production, linked directly into a CasADi external
kernel. No native PhysX executable, learned dynamics or fitted constants.

- Twenty-one shooting states: rear-axle XY/yaw plus the first eighteen source
  state entries, including all four suspension histories. Eight sticky timers
  are an exact zero invariant with strictly positive throttle; they are not
  discarded dynamically relevant states in another mode.
- Two decisions per 25 ms command interval: actual held steering target and
  normalized positive torque. No zero-throttle/full-brake mode.
- Independent pose and physical-state periodicity; a free phase cross-section,
  not a pinned starting pose. Two-metre ordered seed basin is an explicit
  numerical/topological guard, not a physical/global-optimum claim.
- Full mapped vehicle perimeter and 5 cm margin against the real PGM distance
  field. Independent full-lap replay starts once, with no future resets.
- Fixed trial duration in integer 25 ms intervals. It is a realizability
  polisher, NOT a claim to solve unrestricted continuous minimum time.

First 197-interval trial (4.925 s) converged in 36 iterations / 5.06 s. Full-lap
recursive XY maximum was 4.00 mm, with positive node clearance. That initial
test did not yet check every 1 ms perimeter, so its early passing label is
superseded by the stronger validation below.

195 intervals (4.875 s) converged in 100 iterations / 14.47 s, residual
8.84e-6. Recursive XY maximum 1.76 mm; yaw-rate max 0.00169 rad/s; forward speed
max 0.000491 m/s. BUT independent 1 ms perimeter replay found the true 5 cm
margin short by 3.08 mm, so this was NOT accepted. The 192-interval (4.8 s)
start failed after 100 iterations, residual 0.1177; not accepted either.

The actual between-node clearance fault is being fixed with source-evaluated
6/12/18/24 ms prefix poses INSIDE the NLP constraints, not by moving the
exported path afterward or weakening the required margin. Those prefixes bind
integer durations as constants; no false derivative of variable step duration
is exposed. Kernel value and branch-local Jacobian parity tests passed before
this extension; extension tests and the corrected 4.875 s solve are in progress.

Artifacts are grouped under `optimized_practice_command_ocp_20261028/`, not
offered as runtime references. No Unity/MPC ten-lap validation or new racing
record is claimed.

## Additional cleanup

Removed the retired odometry-executable steering-constant fallback from the
source parameter reader. Current production definitions/owned headers and
Unity-prefab cross-checks remain authoritative; unresolved aliases now fail
closed. No numerical vehicle constants changed. Updated the affected tests,
including stale 0.01 versus actual 0.001 throttle expectations and an incomplete
coordinate-frame fixture. Historical unsupported-mode tests still need separate
retirement/repair; they are not hidden to claim a green full suite.

79 additional untracked historical root ZIP bundles (about 106 MiB) moved to
recoverable Trash. No source, checkpoint or recording removed. Trash is not
emptied, so this is workspace decluttering, not reclaimed physical disk space.

## Still outstanding

Independent GT check of the time-loss attribution; fully passing source-tick
wall validation of the shorter witness; geometry-basin multistart; source-
validated witness export/nominal integration; Unity agreement and controlled
repeated ten-scored-lap trials. The frozen 51.030 s setup remains untouched.
The old v354 service was not interrupted by these bounded, separate tests.

## Source-validated sub-4.9 s result (continued 2026-10-29)

The new command formulation has produced an actual usable **source-model
candidate**, not just another infeasible fast iterate. It is not yet a driven
Unity/MPC lap.

`period195_refined/pass_01` converged in 21 iterations / 28.29 s after adding
two exact wall cuts: command interval 57 at physics tick 21, and interval 58
at tick 2. Time is **4.875 s**, maximum scaled NLP violation **1.558e-7**.
The independent recursive source replay has maximum full-lap XY disagreement
**0.0187 mm**, maximum forward-speed disagreement **0.00000520 m/s** and yaw-rate
disagreement **0.0000223 rad/s**. These tiny values measure consistency between
the optimized plan and the SAME production dynamics, not error against Unity.

All 4,875 source ticks pass the complete mapped perimeter + 5 cm requirement.
Minimum conservative source-tick clearance is 0.0500083 m (5 cm + 8.25 microns).
No collision/contact, Unity agreement, robustness-to-disturbance or racing
performance claim follows automatically. Raw source poses are kept in the
reference—no export smoothing or shortening invalidates the checked geometry.

`certified_4p875` independently rebuilt/replayed the saved witness again with
current code, without IPOPT. It confirms periodicity, local transition defect
and every source-tick clearance. Distributed artifacts:

- `f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p875_candidate.csv`
- `.../source_command_4p875_witness.npz`
- `.../source_command_4p875_validation.json`

The target has 4,875 raw source points, largest spacing 8.76 mm, SHA-256
`8e8f9278a9231db31a8658513ed45fea4f01389df99b41d7c15a4c076e39e093`.
Its body-forward speed integral is 4.875352 s, distinct from the actual command
period 4.875000 s. Forward speed spans 2.55–8.76 m/s; throttle 0.07094–0.3870;
peak COM lateral velocity 0.492 m/s; peak yaw rate 3.441 rad/s. No hard brake.
Peak COM sideslip is about 9.77 degrees, not an unconstrained sideways teleport.

Parallelization was benchmarked BEFORE enabling it: 195-node source Jacobian
map 32.87 ms serial, 17.71 ms / four workers, 12.58 ms / eight workers, bitwise
identical values and derivatives. Those are map microbenchmarks, NOT online
MPC timings. The command OCP now defaults to at most eight available workers.

Direct 4.8/4.7 s starts failed their source gates within their bounded budgets;
they are not retained as racing candidates. Resampling fast suspension history
from 25 ms shooting nodes was identified as another poor initializer. The
continuation now uses its genuine saved 1 ms source history when available;
tests explicitly cover fast motion absent at the 25 ms nodes. Subsequent 4.85
and 4.8 s screens still failed; do not claim a floor or add thousands of
iterations. Preserve 4.875 s while investigating a better basin/formulation.

No production MPC/odom equations or the frozen 51.030 s config were changed.
The new target is separate. At that checkpoint, runtime reference validation
and witness nominal integration were the next gates; results follow below.

## First real practice validation and source-reference coupling

Run `live_runs/mpc_source_command4p875_20261029_01/` used the separate 9-column
4.875 s source target with the UNCHANGED frozen MPC weight/trust profile. It
completed warmup + ten scored laps and the post-target collision watch with
zero collisions, controller faults or bridge faults. Ten scored laps:
**50.85199976 s**, mean **5.08519998 s**, population standard deviation
**0.04670 s**, fastest **5.013800 s**. This is a real 0.178 s improvement over
51.030 s, but NOT sub-50 s, sub-4.9 s driving or repeated validation.

Gate median/p95/max were **9.199/17.318/20.675 ms**. One deadline-deferred
cycle occurred; therefore do not claim a reliable maximum below 20 ms.
Cross-track RMSE was **7.63 cm**, speed-error RMSE **0.559 m/s**. Those are
controller tracking diagnostics, not ground-truth rollout errors. Source model
self-consistency numbers above must never be substituted for Unity agreement.

The live target still exposed a concrete representation mismatch: the existing
MPC reference builder requested zero lateral velocity/body-path angle,
`r=curvature*u` and bicycle steering even though a validated source witness was
available. Implemented an explicit schema-marked 15-column reference carrying
the planned rear-axle lateral velocity, body heading error, gyro yaw rate,
physical rack, held steering target and torque. These are precomputed reference
data, NOT new optimizer states or prohibited live sensors.

- State references now use that witness. Cold and unsolved-tail controls use
  its current-application-epoch steering/torque, not next-state or bicycle
  values. Issued-command warm starts still take precedence.
- The old yaw-reference alignment pass must NOT overwrite source yaw rate;
  a new regression test caught that exact issue and it was fixed.
- Physical states interpolate; commands remain zero-order-held across source
  packet edges. Mixed schemas, zero-throttle witnesses, malformed rows and
  capped-speed reuse fail closed. No extra solver dimensions or plant calls.
- Retired the loader's implicit one-metre corridor fallback and partial CSV
  acceptance. Maintained files are explicit 9-column corridors or schema-marked
  15-column witnesses. The 9-column record path/weights/math remain unchanged.
- Avoid bicycle `atan` work when the source rack reference is already present.

Candidate: `source_command_4p875_witness_candidate.csv`, SHA-256
`e9dffcd11a8b3004b16bd28e38955612912307e666a83bbb8e7a496994fe23d7`.
Same 4,875 raw source XY samples and speed profile as the tested line. No
geometry adjustment or dynamics fit. Full MPC CTest suite **12/12** passes,
including source-state/held-input interpolation, exact nominal initialization,
speed-cap rejection and nonlinear objective/condensed-direction parity in both
turn directions with and without the witness. Focused Python export tests
**7/7** pass. Runtime-topic static policy passes; organizer approval of the
existing development 40 Hz bridge remains unresolved.

Source-reference online validation is the next test. No result for it is yet
claimed. Commit `fb76cfc46` preserving the optimizer, candidate and source
validation has been pushed and remote SHA verified. New coupling changes are
being verified/committed separately.
