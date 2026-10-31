# IROS competition MPC: runtime and validation progress

## Current gate

The optimizer's best source-certified reference is **7.125 s**, witness
`7a9e6afbf50d666f`. This is a source-model ranking, not a Unity lap. Latest
IROS validation a28 completed two scored laps (7.4596 s and 7.3487 s), then
rejected at 204.708 m. a26 remains the best recent lap count at four; neither
reached the 10 scored-lap gate. No 10-lap validation has succeeded. The
practice records are not IROS competition results.
Source certification, QP convergence, accepted nonlinear horizons and actual
completed driving are separate claims.

Single dashboard: <http://127.0.0.1:8765/>. Active optimizer workspace:
`live_runs/optimizer_iros_multi_incumbent_20261029`. The persistent validator
imports the prior 28 attempt records from `optimizer_iros_conditioned_20261029`,
pauses the active optimizer during MPC builds/runs/replay, then resumes the
same iterate. The dashboard and validator now use this same workspace.

## Implemented corrections

1. **Track-specific configuration.** The launch file had unconditionally
   applied the practice fast-line profile to IROS. Non-practice maps now get
   the competition base plus an explicit optional overlay. Parameter-file
   precedence is logged and tested.
2. **Replay ordering.** Diagnostics put rack angle after queued steering
   targets; C++ replay expects it before them. The canonical exporter now
   explicitly reorders them. Prefix replay uses that same header/exporter.
   Old results from the swapped exporter are not current solver evidence.
3. **Steering objective consistency.** Source-witness steering increments
   use commanded-target changes, not slower physical-rack changes. The first
   target error uses the actual prior command. QP and nonlinear merit match.
   Geometric-reference behavior is unchanged. Consistency does not prove the
   chosen weights or witness command pattern are optimal for tracking.
4. **Capped warmup.** Capped samples use geometry-only reference targets,
   not lateral/yaw/throttle states from a different-speed native witness.
   The same full physical plant predicts motion. Uncapped samples restore
   native references without modifying the trajectory file.
5. **Feedback-stabilized trials.** Optional `feedback_rollout` reuses final
   augmented Riccati gains already calculated by ADMM, transformed to physical
   units. It generates an explicit corrected command sequence during nonlinear
   trial evaluation. Its linear reference trajectory uses projected controls,
   not unconstrained primal states paired with different inputs. Commands keep
   the same actuator/trust boxes and exact plant/corridor/deadline checks.
   Backtracking starts from original QP controls, not an already corrected
   sequence. No future measurements, new solver coordinates or factorization.
   These are augmented Riccati gains, not constrained-optimum sensitivities.
   Default off; these IROS trials explicitly enable it.
6. **Soft clearance reference.** Optional `reference_clearance_reserve_m`
   moves the lateral target into the footprint corridor where zero offset
   leaves insufficient reserve. Uses desired body heading and rear-axle bumper
   anchors. Hard **5 cm** clearance is unchanged. Default zero; a08/a09 use
   3 cm. This does not certify a replacement raceline.
7. **Native input prior.** `weight_steering_target` was a zero-centered input
   effort cost even for source witnesses. It now tracks the actual native
   commanded target when available, separately from physical-rack tracking.
   Geometry-only paths retain zero-effort regularization. QP linear terms and
   exact nonlinear scoring both changed; regression checks cover their match.
   Same a09 states/weights improved acceptance **512/536 → 523/536**. Raising
   this weight from 0.05 to 1 or 10 instead achieved 502/536 and 516/536, so the
   unchanged 0.05 value was retained. A third 7.225 s Unity trial is queued.
8. **Recovery eligibility guard.** Exact revalidation of a retained safe prefix
   is no longer gated on the rejected candidate having a usable ADMM status or
   a full 75 ms candidate sequence. The candidate remains ineligible unless its
   status and prefix qualify; the retained sequence is independently checked
   against the same exact plant/corridor/deadline gates. The change compiled;
   its online effect remains unverified.

No physical source-plant equations or parameters were fitted/retuned here.
MPC still cannot choose zero throttle; odom/replay retain its hard-brake model.

## Unity evidence

All trials use production MPC, odom and AMCL on the IROS competition map.
Complete artifacts are in `live_runs/apex_mpc_f8c5b636134d8b64_aNN/`.
Progress is unwrapped along the approximately 42.97 m circuit; spawn is near
39.1 m. Until a09, no warmup lap completed.

| Attempt | Change | Terminal progress | Full-horizon accepts | Publication gate max |
| --- | --- | ---: | ---: | ---: |
| a01 | Leaked practice profile | 62.295 m | 165 | 18.514 ms |
| a02 | Correct map profile | 65.574 m | 155 | 23.928 ms |
| a03 | Native-target increment correction | 64.842 m | 158 | 18.479 ms |
| a04 | Cost scaling and warmup ramp | 65.007 m | 202 | 19.132 ms |
| a05 | More overspeed/yaw authority, less throttle damping | 76.574 m | 261 | 20.897 ms |
| a06 | Stronger heading/yaw bundle; reverted | 64.836 m | 210 | 18.904 ms |
| a07 | a05 plus feedback trial rollout | 62.937 m | 218 | See retained report |
| a08 | a07 plus soft clearance reserve | 77.284 m | 309 | 19.079 ms |
| a09 | Stronger path tracking, less native-increment damping | 107.140 m | 511 | 19.350 ms |

a07 used no recovery prefix or retained-plan actions, then rejected at a
tight wall with +2.3 cm lateral error. a08 passed there but accumulated -9 cm
error near the next turn; its last three-step prediction crossed its permitted
interval by about 1.4 cm. It used five retained-plan and two prefix actions.
a09 passed that turn and warmup, then developed heading/lateral error during a
chain of retained/prefix plans in lap two (20 retained, four prefixes overall).
Do not count those as full-horizon accepts.

The watcher aborts on the first MPC fault. Some bags also contain a collision
during shutdown roughly 0.1 s later. Metrics cut at the **first** fault; the
dashboard shows later collision evidence separately. These are not successful
or universally collision-free runs.

## Fixed-issued-command physics replay

Each 750 ms recursive prediction uses a logged production observer state and
actual subsequent issued commands, with **no future state correction**. GT is
scoring only. This is not an unexecuted MPC plan compared to Unity replanning.
Overlapping windows are not independent. Windows end before the first fault.

| Attempt | Windows | Endpoint XY RMSE | XY max anywhere in horizon | Paths over 5 cm |
| --- | ---: | ---: | ---: | ---: |
| a03 | 170 | 0.986 cm | 2.990 cm | 0 |
| a05 | 287 | 0.761 cm | 5.278 cm | 1 |
| a06 | 214 | 0.902 cm | 3.982 cm | 0 |
| a07 | 195 | 0.559 cm | 1.479 cm | 0 |
| a08 | 293 | 1.333 cm | 3.582 cm | 0 |
| a09 | 512 | 0.772 cm | 2.175 cm | 0 |

a09 forward/lateral speed RMSE: 0.00805/0.00143 m/s; yaw-rate RMSE:
0.0106 rad/s. Its loss of tracking is not explained by huge fixed-command
prediction error on this recording. This does not prove accuracy for arbitrary
unobserved sequences/regimes. The freshly compiled evaluator SHA was
`7de79b79a487fbcdc954c0719f5fcec11e3f16b7af0e2111d64a9efd2dccb1f8`.

## Offline alternatives already tried

- Feedback rollout improved same-state full-horizon acceptance **226/236 →
  233/236** on a04 and **281/311 → 287/311** on a05, with identical weights and
  two passes. Generated explicit controls reproduce the policy's plant rollout.
- Three/four SQP passes did not improve both corpora consistently; the trial
  extension was removed. Production retains at most two passes.
- Wider/disabled steering trust regions did not transfer consistently; kept
  0.08 rad. Cold/reference/warm initialization and input-penalty variants did
  not justify globally replacing initialization. a06 regressed online distance.
- Saved-state replay cannot restore all live ADMM/certified-plan memory.
  Counterfactual acceptance counts are not online driving proof.
- On the complete a09 corpus, warm initialization accepted 512/536 horizons;
  reference-seeded and fully cold initialization each accepted 514/536.
  Raising heading weight from 30 to 200 (and lateral from 200 to 300) reduced
  acceptance to 507/536. These comparisons used the same current physical
  plant, feedback trial rollout and 3 cm soft reserve. They do not justify
  another blind online heading-weight increase or a global cold-start switch.

## Automation and next gate

### Additional objective audit (after a09)

The original source-reference gradient regression used constant yaw/rack/target
references. Adding a spatially varying source-reference case exposed omitted
reference-progress derivatives: three tested input directions disagreed with
the exact nonlinear objective beyond the 0.5% test tolerance. The QP already
propagated progress for geometry, but treated these reference costs as fixed
while its nonlinear merit resampled them.

The tested correction stored cheap reference-only progress tangents
(centered 1 mm interpolation secants) and includes their cost gradient in the
existing progress-deviation coordinate. It includes tracking-reference terms,
curvature-scaled weights and both adjacent steering-increment costs. Speed
targets remain fixed over an SQP horizon, as in the current exact scorer. No
plant finite differences, additional optimizer coordinates or physical model
changes. At interpolation knots a secant is not proof of differentiability.

All 47 C++ tests passed after this correction, including the newly varying
case. However, the same a09 fixed-state replay accepted only **483/536**
horizons versus **512/536** before it. A positive diagonal-only progress
Gauss-Newton augmentation was worse (**463/536**) and was removed; it omits
cross curvature and must not be portrayed as an exact Hessian. The retained
correction used the existing convex frozen-reference Hessian approximation
with a corrected first derivative and exact nonlinear line search. It is not
a proven driving improvement. **a10 failed online at progress 64.383 m, before
warmup finished, versus a09's 107.140 m. The standalone correction was removed
from production.** Its full reproducible patch remains at
`live_runs/iros_mpc_convergence_20261029/reference_progress_gradient_trial.patch`;
the new varying-reference test is part of that unpromoted patch. All report and
replay artifacts remain available. A proper coupled reference-cost curvature
or coherent reference formulation remains an unresolved solver issue; do not
repeat this first-gradient-only change as an untested fix. Do not conflate
objective consistency with improved control.

- One enabled GUI/optimizer; old reverse campaign stopped and disabled, files
  recoverable. Retry is same-origin JSON into a serialized queue, not arbitrary
  command execution. No automatic weight search or infinite retries.
- Per-attempt overlays/SHA, map, trajectory and controller image are retained.
  Fresh production build; monitored Unity; failed bags decoded too.
- Automatic report builds/fingerprints the source-only plant evaluator and
  shows fixed-command prediction metrics beside timing/tracking/lap results.
- Missing recorded images allow an explicitly documented decoder-only image
  substitution (`analysis_environment.json`), not silent binary equivalence.
- Unknown optional metadata becomes JSON `null`, not NaN; regression-tested.
- Warmup completion remains visible even when no scored lap finishes.
- Pause leases check owner PID and process start time. Supervisor restart does
  not thaw a still-live offline replay. Normal SIGTERM releases owned work.

Reproduction: a09's immutable overlay is
`mpc_validation/inputs/f8c5b636134d8b64/attempt_09_mpc.yaml` in the optimizer
workspace; it is also copied to the trial's `mpc_overlay.yaml`. Relative
priorities: lateral/heading 200/30, underspeed/overspeed 0.35/2, yaw 2, wheel
slip/lateral speed 0.05/0.05, steering-command/target 0.03/0.05, throttle command
0.001, steering/throttle increments 1/0.1. Feedback enabled, soft reserve 0.03 m,
startup 3 m/s ramping over one warmup lap. Base solver: two SQP passes,
150 ADMM iterations/pass, eight Jacobian workers, 20 ms publication budget.
This incomplete racing profile is not promoted over the practice baseline.

Verification: 120 focused Python regressions passed; 47 C++ plant/solver tests
passed, including gain scaling/unchanged QP solution, explicit generated-control
replay and clearance geometry. Shell/JavaScript syntax and runtime-topic-policy
checks passed. Source model fidelity is measured by the replay table above,
not inferred from those unit tests.

Next: use a09's states **before** the recovery chain to distinguish a bad
nominal basin from weights/ADMM convergence. Re-test a demonstrated change at
full reference speed. Keep hard clearance intact. The required result remains
repeated scored laps, not just a warmup or accepted QPs.

## 2026-10-30: competition-line recovery and timing work

The active source-model optimizer's best remains **7.125 s** for the current
competition image (witness `7a9e6afbf50d666f`). This is a source-model ranking,
not an actual lap time. MPC attempts a01–a06 did not complete a scored lap;
the last attempts terminated on exact nonlinear corridor rejection. Thus the
faster line is not yet a usable/race-proven line.

Three 7.125 s MPC tests were compared with unchanged base dynamics and the
fixed-command replay, then failed experiments were discarded:

- Raising lateral/heading weights to 500/100 regressed the opening hairpin;
  these weights were restored to 200/30.
- Raising only `admm_rho_ey` to 16 regressed progress and drove the penalty to
  its maximum. This overlay change was removed.
- Steering-target weight 5 remains the best tested of 2.5/5/10 on the newer
  source-witness line. The separate native steering-target QP correction is
  unit-tested; it does not imply full-lap success.

The a06 trace exposed a recovery-policy defect: after six consecutive prior
plan actions, both 75 ms prefixes passed exact plant/corridor validation and
had ample wall clearance (candidate 0.2401 m vs retained 0.1979 m), but the
candidate's exact short-horizon objective was 108.75 vs 86.96. Existing policy
kept replaying the lower-cost prior, despite the 4.22 cm clearance gain. The
related a03 trace had a 3.57 cm gain with the prior already below the 5 cm
reserve. This is not a claim that clearance alone optimizes racing; it is a
narrowly gated way to avoid indefinite stale-plan replay.

Production selector change: after three consecutive retained-plan actions, an
iteration-limited candidate may refresh only if both equal-duration 75 ms
prefixes passed exact source-plant and corridor checks, both retain at least
5 cm slack, the candidate improves slack by at least 5 mm, and its exact
prefix objective is no more than 30% worse. The existing below-reserve
clearance override requires a separate 2 cm material gain. Normal residual,
objective and hard nonlinear checks remain; no solver-tail controls are
published. Tests cover the captured a06 selection, a large objective
regression, and the prior 9.9 mm case. This adds no plant or solver work on the
nominal converged path.

The first unit run exposed that the prior 5 mm override would admit the
captured 9.9 mm poor-residual change. The threshold was raised to 2 cm; this
still admits the captured 3.57/4.22 cm cases. Final package verification in the
IROS ROS image: **12/12 selected CTest tests passed, 145 tests/assertions, 0
failures**. The optional QP-oracle Python test was excluded because this image
does not contain SciPy; the first unfiltered run otherwise passed package tests
and failed only that import. Both the production config and trial overlay parse
as YAML. The ADMM-penalty experiment was removed from the overlay.

Attempt a07 rebuilt the then-current selector and completed warmup, then stopped at
progress 62.65 m. The stale-chain rule was not involved (`consecutive prior
actions=0`). At the first predicted sample, the candidate state exceeded the
effective corridor by 1.4 mm after the configured 5 mm nonlinear tolerance;
the current body-to-wall clearance estimate was about 4.36 cm. There were 219
converged solver cycles and two retained-plan cycles before the terminal
rejection. Full-gate timing was p95 17.20 ms, max 19.20 ms; no deadline fault
or collision was recorded. Scored laps: zero. Fixed-command 750 ms replay was
1.23 cm endpoint RMSE, 3.90 cm maximum path error, and no windows above 5 cm.
This separates a barely missed online corridor condition from the much smaller
open-loop replay error.

For the next controlled retry only, the attempt overlay raises
`nonlinear_corridor_tolerance_m` from 5 to 7 mm. This grants 2 mm to the exact
nonlinear acceptance check only; QP corridor faces, source vehicle geometry,
trajectory, and 5 cm nominal reserve are unchanged. It is intended to test
whether the 1.4 mm first-step miss is a tolerance-edge artifact while the
remaining horizon remains exact-checked. It does not establish a new safety
margin or certify the raceline. Judge the retry on progress, scored laps,
corridor events and worst timing; do not promote the 7.125 s source line unless
it completes repeated full-speed laps.

Attempt a08 used 7 mm nonlinear tolerance. It passed the a07 location, but
then accumulated 21 consecutive prior-plan actions from progress 74.16 m.
At 74.43 m the exact-validated candidate had 7.46 mm more 75 ms slack, with
29.8% higher exact prefix cost; the 2 cm stale-chain threshold rejected it.
Tracking error grew from -5.3 cm to -21.4 cm by the eventual failure at
76.18 m, where both three-command prefixes failed corridor validation at
stage 2. No collision or scored lap; p95/max gate 17.24/18.89 ms. The
2 mm tolerance change therefore avoided the first tiny boundary miss but did
not solve the developing stale-plan/trajectory tracking failure.

Follow-up selector rule separates ordinary near-wall overrides (2 cm) from
stale-plan refresh: after at least three prior actions, an exactly validated
candidate may refresh at a 5 mm clearance gain, still requiring both prefixes
to have >=5 cm slack and candidate exact cost no more than 30% worse. This
would have allowed the a08 candidate at 74.43 m but not its later 5.5x-cost
steering-reversal examples. Unit coverage was added for that exact sample and
the excessive-cost rejection. Rebuild/tests must pass before attempt a09.

Attempt a09 exercised that selector and the 7 mm exact-check tolerance. It
reached progress 76.57 m; in the 70–75 m sector it had only two retained-plan
actions instead of a08's long chain, and terminal cross-track was 12.7 cm
instead of 21.4 cm. It still had no completed lap and stopped at `corridor@2`
with no collision. At terminal state the solver hit its 150-iteration cap at
11.48 ms, with primal/dual residuals 0.096/0.026 and zero accepted SQP steps.
Whole-run gate timing was p95 16.59 ms, max 18.67 ms. The candidate and prior
three-command exact prefixes both violated the lower corridor bound by the
third command, while current estimated cross-track was already -12.7 cm. This
is now an iteration-limited constrained solve, not the previous stale-plan
chain or the 1.4 mm tolerance-edge event.

Next trial uses 220 ADMM iterations instead of 150, holding all weights,
vehicle model, candidate and corridor policy fixed. The RTI wall deadline stays
20 ms and terminates work if reached; the point is to use unused budget on
states that demonstrably hit the iteration cap, not to enlarge the deadline.
Success means lower residual and an exactly valid first-three-command prefix
without a deadline overrun, then continued lap progress. A changed iteration
cap alone is not considered a racing result.

### 2026-10-30 follow-up: a10 diagnosis and a11 gate

Attempt a10 configured 220 ADMM iterations, but the compiled hard ceiling was
still 200. The controller clamps the ROS parameter to `MAX_ITERATIONS`, so a
configured value above that macro silently has no effect. a10 completed
warmup and one further lap-distance, reaching 118.87 m, then failed
`rejected_nonlinear_rollout; corridor@2` in the repeated tight section. It
completed 584 converged cycles, with 7 time-budget terminations, 4 pre-ADMM
deadline skips, and 7 iteration-capped cycles. Maximum publication gate time
was 19.174 ms (p95 17.896 ms); no collision or recorded deadline fault. No
scored lap was completed. Fixed-command 750 ms replay was 1.47 cm endpoint XY
RMSE, 6.81 cm maximum, with 6/548 windows over 5 cm and yaw-rate RMSE
0.0231 rad/s. This replay is not the unexecuted MPC horizon and does not
explain away the tracking/corridor failure.

The a10 trace supports two separate next experiments, not a claim that either
is solved. First, a repeated 17-action prior-plan chain persisted despite
candidate refreshes with similar clearance; at progress 118.76 m both exact
three-command prefixes had less than 5 cm clearance reserve and were invalid.
The selector is now changed to permit a stale-chain refresh after three prior
actions when both exact-validated prefixes retain the 5 cm reserve, candidate
objective is at most 30% worse, and candidate slack loses no more than 2 cm.
It remains narrowly gated to stale chains and does not weaken exact validation
or the ordinary under-reserve override. Regression tests cover measured a10
examples and reject a 4.7x objective increase.

Second, the compile-time solver ceiling is raised to 300 so the experiment's
250-iteration configuration is actually honored. The 20 ms RTI deadline stays
hard; this is not permission for a cycle to exceed it. Attempt a11 is the
single next serialized test using the same 7.125 s source-certified line and
weights, updated stale-chain selector, and 250 configured iterations. Judge
the measured gate-time distribution, exact corridor decisions, progress, and
scored laps together. The target remains repeated full-speed scored laps; a
longer iteration cap or a completed warmup alone is not success.

### 2026-10-30 follow-up: active workspace, dashboard and a13

The always-on validator was moved to the active multi-incumbent workspace and
the prior 28 attempt records were imported without re-running already seen
witnesses. The active dashboard serves this workspace; it reports one latest
attempt while the full history remains in `mpc_validation/state.json`. A13 is
the first trial from this migrated watcher.

A13 held the a11 weight bundle and reduced the minimum clearance-gain trigger
from 20 mm to 10 mm, with the exact 5 cm corridor reserve and 20 ms gate
unchanged. It did **not** complete a scored lap. It ran 307 controller cycles:
303 ADMM converged, four hit the time budget, and one terminal nonlinear
rejection occurred. The terminal observer state was at progress 76.588 m,
`e_y=-0.1003 m`, `e_psi=0.0128 rad`, forward speed 4.809 m/s, yaw rate
2.090 rad/s; the line requested 5.181 m/s at curvature 0.363 1/m. Both the
candidate and retained-plan 75 ms exact prefixes failed at stage 2. The
candidate's third predicted sample exceeded the effective lower corridor bound
by 1.18 mm **after** the configured 7 mm nonlinear tolerance. The failed QP
was time-budget limited (342 iterations across two SQP passes; primal/dual
residuals 0.159/0.188); that cycle took 14.71 ms end-to-end. Across the run,
publication-gate p95/max were 16.50/18.92 ms with no gate overruns. The
10 mm selector threshold is not implicated in this terminal event: at the
last two pre-failure cycles, the candidate had only 0.7–2.0 mm extra slack
and a 41–130% worse exact prefix cost, so retaining the prior was consistent
with policy; at the terminal cycle both options were invalid.

The independently decoded fixed-issued-command 750 ms replay on this bag had
6.99 mm endpoint XY RMSE, 18.59 mm maximum path error, and no windows above
5 cm. This supports good propagation for the logged state/commands; it is not
evidence that the optimizer's unexecuted horizon was accurate or feasible.
The failure is therefore currently classified as an online planning/tracking
feasibility failure, not established as an odometry/plant-model mismatch.
Do not respond with another broad weight sweep or loosen the physical corridor.
Next diagnosis should replay the approach into 76.588 m with the exact captured
state, queued inputs, and trajectory, then compare feasible first-command
sets/solver nominal against the logged issued sequence. Any recovery must keep
the delayed-command and exact corridor checks; if no safe first command exists,
the run must still terminate rather than publish an unchecked action.

The dashboard's failure-card decoder reads the bounded tail of the per-cycle
diagnostic sidecar and reports state, line demand, solver termination, and
candidate/prior validation. The active dashboard process must be restarted
after Python/HTML code changes so it does not continue serving an old decoder.

Attempt a14 restored the a11 20 mm selector threshold as a controlled repeat.
It also failed before scoring a lap, at the separate tight section near
canonical progress 37.39 m (unwrapped progress 80.082 m). State was
`e_y=+0.0296 m`, `e_psi=-0.0527 rad`, `u=2.738 m/s`, `r=2.230 rad/s`,
`kappa=0.881 1/m`; the reference requested 2.844 m/s and -6.325 m/s^2.
Both candidate and retained-plan prefixes failed on their first predicted
sample. Candidate slack was -9.84 mm after the corridor tolerance; steering
targets already in the two-packet queue were +0.260 and +0.325 rad, while the
current physical rack was +0.325 rad. The ADMM iterate hit its 250-iteration
cap (primal/dual 0.162/0.164); the failed cycle took 10.49 ms, and run-wide
publication-gate p95/max were 16.58/18.95 ms with no deadline violation.
Fixed-issued-command 750 ms replay was 14.82 mm endpoint RMSE, 56.55 mm max,
with two of 311 windows over 5 cm. This single worst window warns against
calling the rollout uniformly below 5 cm, although it is not evidence that the
whole prediction model is generally inaccurate.

At the same track section, a11's observed laps were on the opposite lateral
side or near the line (roughly -4 cm to +2 cm), whereas a14 arrived about 3 cm
toward the tight wall. The command queue and heading error leave too little
time for a reaction at the failing sample; the useful intervention must begin
earlier in the approach. A15 tested one variable: increase only the MPC's soft
`reference_clearance_reserve_m` from 6 cm to 10 cm, moving its lateral target
farther from a pinched wall. The source raceline, speed profile, hard 5 cm
clearance constraint, solver weights and timing gate were unchanged. The
measured result and decision follow below.

A15 did not validate the larger reserve. It stopped in the earlier tight-turn
sector at progress 76.620 m with `e_y=-0.1111 m`, `e_psi=+0.0422 rad`, speed
4.517 m/s, yaw rate 1.920 rad/s, and curvature 0.358 1/m. Its stage-3 prefix
state crossed the lower corridor bound by 6.24 mm; candidate and retained
prefixes both failed. This is about 1 cm farther outside than a13 at the same
sector, although the observer state and commands differed, so the reserve
change cannot be isolated as the sole cause. Revert to the 6 cm reserve; do
not promote the 10 cm setting.

At this sector the speed reference was 5.20 m/s and desired heading error
about +0.066 rad while the terminal state was +0.027 rad. a16 tested only
`weight_e_psi=100` (from 30), restoring the 6 cm reserve. It regressed badly:
the high-curvature section at progress 79.4–80.9 m switched from a11's
repeated current-solver line-search steps (typically 7–29 ADMM iterations,
cross-track within about 1 cm) to repeated 250-iteration caps and retained
prior-plan actions. Cross-track grew from +1.8 cm at 79.44 m to -37.9 cm at
80.86 m; heading error reached -0.40 rad, steering saturated at +0.52 rad,
and both exact three-command prefixes failed at stage 2. No lap was scored.
The terminal QP had primal residual 1.03, dual 0.025, and rho reached its cap
of 32. The higher heading penalty is therefore reverted; it is not a useful
anticipation fix. The failure was preceded by solver nonconvergence and a
long sequence of lower-cost retained-plan choices, not a corridor-tolerance
edge.

Two cycles in a16 also exceeded the 20 ms gate (26.22 and 39.73 ms), both
deferred before ADMM because Jacobian/QP construction alone took 25.04 and
37.27 ms. The 39 ms event occurred around progress 76.73 m, before the final
loss of tracking. This is a real worst-case timing issue, distinct from the
curvature-sector convergence failure; it needs profiling under serialized
trial load before changing worker count or the hard deadline.

The a17 retry restored the a11 heading weight (30), 6 cm reserve, and 20 mm
selector threshold, but did not reproduce its progress. a17 stopped at
76.652 m, with `e_y=-0.0972 m`, `e_psi=+0.0161 rad`, speed 4.927 m/s, yaw
2.065 rad/s, reference 5.22 m/s and curvature 0.35 1/m. The final QP hit the
time budget (342 iterations across SQP passes; primal/dual residuals
0.052/0.010); both candidate and retained prefixes failed corridor validation
at stage 2. It completed zero scored laps. Gate p95/max were 16.79/19.20 ms,
with no over-budget cycle. Unlike a11, this approach repeatedly selected the
retained plan as cross-track drift grew; the a11 run therefore remains a
successful but non-repeatable observation so far.

At 76.534 m, the retained prefix had only 0.25 mm minimum predicted slack,
while an exact-valid candidate had 5.91 mm. The 20 mm under-reserve override
kept the prior. Candidate objective was 1.45x the prior, but the first three
steering commands were essentially identical; the candidate reduced throttle
instead of introducing the large steering reversal in the known harmful v300
case. The a18 experiment configured a separate narrowly gated choice: when
the prior is below its 5 cm reserve, allow a candidate with at least 5 mm more
exact-validated slack, at most 1.5x short-prefix cost, and per-step steering /
throttle changes bounded to 0.10 rad / 0.20. It does not relax the hard
corridor, residual rules outside this near-wall case, or timing deadline. The
known 9.9 mm poor-residual steering reversal is explicitly regression-tested
to remain rejected by the command-continuity gate. The dashboard now shows the
last recovery choice, both slack values and relative objective before a
terminal nonlinear rejection, so this behavior is visible rather than hidden.

a18 completed one warmup lap in 9.007 s, then faulted during the next lap at
progress 119.354 m (`corridor@2`). It completed zero scored laps. Warmup-lap
cross-track RMSE/max were 7.05/28.45 cm; whole scored-through-fault RMSE was
5.68 cm. Gate p95/max were 17.47/19.43 ms with no deadline violations. The
terminal QP stopped at its time budget (primal/dual 0.0153/0.0200), and both
the candidate and retained three-command prefixes failed exact corridor
validation at stage 2. Thus the near-boundary selector could not rescue the
terminal event: neither action was certified for the full 75 ms command-effect
window.

Reviewing the implementation against a17 exposed a selector control-flow bug:
the converged-solver branch returned through ordinary objective ordering before
the configured near-boundary rule. The rule was therefore unreachable for the
normal converged case that motivated it. a18 does not validate that policy.
The selector now checks the strictly gated near-boundary case before that
converged early return; the material 20 mm clearance override remains first,
and hard exact rollout, objective ratio, command-continuity, and timing gates
are unchanged. A captured converged a17 regression test covers the missed path,
alongside the existing steering-reversal exclusion. Next is a serialized a19
repeat on the same witness and settings. It must demonstrate that this branch
actually fires in diagnostics and improves repeat progress/tracking before it
can be retained as a production improvement. If both prefixes again fail at
stage 2, investigate the local corridor/initial-state feasibility separately;
do not loosen the exact safety validator to manufacture acceptance.

a19 used the corrected selector and fired the near-boundary override twice
(62.231 m and 64.484 m), but did not improve the result: it stopped at
64.613 m with zero completed laps. Cross-track RMSE through the fault was
8.99 cm; max timing was 18.96 ms, no deadline miss. The terminal state had
`e_y=-0.196 m`, `e_psi=-0.103 rad`, `u=5.30 m/s` against a 5.85 m/s target;
the terminal exact rollout exceeded the adjusted corridor by 7.6 mm at stage
2, and both candidate and retained plan failed there. The final ADMM solve
timed out with primal/dual residual 0.325/0.179. The two selector overrides
are not evidence of a gain: a19 regressed from a18's completed warmup lap and
still suffered long retained-plan chains. The branch stays disabled in the
next test while preserving its code/test for controlled future use.

The 750 ms replay of the *actual commands issued in a19* scored 7.0 mm XY
endpoint RMSE (19.6 mm max), 7.3 mm/s forward-speed RMSE and 0.0105 rad/s yaw
rate RMSE. This strongly supports that the source plant predicts those logged
commands accurately; it does not certify an unexecuted optimized horizon.
The observed failure is therefore not explained by a large open-loop plant
replay error. a20 tested heading weight 45 with the near-boundary selector
disabled. It stopped before a lap at 76.287 m, with `e_y=-0.120 m`,
`e_psi=-0.034 rad`, speed 5.28 m/s versus 5.00 m/s target, and stage-2
corridor failure. The iteration-capped terminal residuals were 0.316/0.052;
timing max was 19.15 ms with no deadline miss. Its actual-command replay was
9.9 mm XY RMSE / 32 mm max, 9.4 mm/s forward-speed RMSE and 0.0158 rad/s yaw
RMSE. This did not show that increasing heading weight improves full-run
tracking, so the next trial restores heading weight 30.

The a20 terminal plan snapshot requested a sharp steering reversal
(`-0.142 rad`) while the source witness at the same region commands about
`+0.133 rad`; the reference speed was already being undershot earlier in the
turn. a21 isolated stronger source-steering feedforward (`weight_steering_target`
10 instead of 5), with heading 30 and the near-boundary selector disabled. It
again failed near the same sector at 76.481 m before a lap. The accepted plan's
first target at the terminal cycle was `+0.141 rad`, but the rejected candidate
was `+0.061 rad`; the witness target is about `+0.133 rad`. That candidate's
exact delayed rollout fell from current yaw rate 2.09 to 1.02 rad/s at stage 2
while the reference required about 1.59 rad/s, and crossed the corridor by
about 27 mm. Residuals were 0.098/0.037 at iteration cap; max timing was
18.85 ms. The logged-command replay was 7.2 mm XY RMSE / 23.6 mm endpoint max.
Increasing the source-steering penalty to 10 did not prevent the candidate
from shedding roughly half the feedforward in this state.

a22 tested `weight_steering_target=50`, heading 30 and no near-boundary override.
It completed a 9.029 s warmup and one scored lap in 7.500 s, then failed on the
next pass at 161.728 m. Tracking RMSE through rejection was 5.50 cm; gate
p95/max 16.43/19.37 ms, with no deadline miss. The same-control 750 ms replay
was 10.0 mm XY RMSE / 28.5 mm endpoint max and 0.0097 rad/s yaw-rate RMSE.
At terminal rejection, the current state was already `e_y=-0.160 m`; the
candidate requested steering 0.059 rad and throttle 0.365, versus retained
0.139 rad / 0.265 and source-witness steering/throttle about 0.133 / 0.286.
Its stage-2 rollout reached `e_y=-0.180 m`, `r=0.896 rad/s` against about
1.75 rad/s path yaw demand, and exceeded the corridor by about 54 mm. ADMM
reached 250 iterations with 0.255/0.031 residuals. Thus the stronger target
penalty improved lap progress but did not prevent an unsafe, high-throttle /
low-steering iterate; it is not a success profile yet.

a23 raised longitudinal-slip cost from 0.05 to 0.5 while retaining the a22
steering/heading profile. It regressed: no lap, terminal at 57.087 m, tracking
RMSE 10.76 cm, terminal residual 0.686/0.153, and actual-command 750 ms XY
RMSE 11.28 cm (28.4 cm max). Revert the slip penalty; it changed the chosen
startup controls enough to enter a regime where the live replay itself was no
longer accurate, so it is not a viable fix.

a22's rejected first control was exactly at the negative steering trust edge:
nominal `+0.141 rad` to candidate `+0.061 rad` is the configured `-0.080 rad`
step. a24 kept the a22 weights (`weight_steering_target=50`, heading 30,
slip 0.05) but halved the per-SQP steering trust radius to 0.04 rad. It
completed one warmup lap (9.236 s) and one scored lap (7.517 s), then failed at
165.341 m in the tight hairpin. The terminal current state had `e_y=+3.35 cm`,
`e_psi=-0.039 rad`, `u=2.886 m/s`; at stage 2 the upper corridor bound was
`+1.57 cm`, and predicted `e_y=+2.16 cm` (5.9 mm outside). ADMM hit its
iteration cap with residuals 0.0796/0.0329. Gate p95/max were 16.16/19.38 ms;
no deadline miss. Actual-command replay remained 10.3 mm XY RMSE / 34.5 mm
max. The smaller step extended one scored lap but did not make the plan
repeatable, and the terminal error moved to the opposite corridor edge.

a25 doubled only `weight_e_y` from 200 to 400. It regressed: no scored lap,
tracking RMSE 10.90 cm, 750 ms actual-command XY RMSE 7.46 mm, but the
terminal state reached `e_y=-0.715 m`, `e_psi=-0.685 rad` and failed at stage
5; ADMM residual was 0.869/0.026. One cycle crossed the deadline at 20.006 ms.
Revert to 200. a26 repeated the a24 profile and completed four scored laps:
7.4066, 7.3986, 7.3666 and 7.6386 s (mean 7.453 s), then rejected at
270.647 m on a tight section. Tracking RMSE through rejection was 4.07 cm;
gate p95/max 16.03/19.42 ms, no deadline miss. Its terminal ADMM stopped at
250 with 0.050/0.044 residuals; the failed predicted prefix exceeded its
upper corridor bound by about 0.8 mm. The actual-command rollout was 13.1 mm
XY RMSE / 65.2 mm max. This is the best recent lap count, not a pass.

a27 repeated the exact a26 profile but rejected before its first lap, at
122.642 m. Tracking RMSE was 6.87 cm; actual-command replay was 8.8 mm XY
RMSE / 32.2 mm max. Terminal ADMM hit 250 with 0.0635/0.0494 residuals; the
prefix missed the narrow corridor at stage 2. Gate p95/max were 16.30/23.49
ms, including one deadline violation. Thus a26 is not repeatable yet.

a28 raised only the ADMM cap from 250 to 300. It completed a 9.153 s warmup and
two scored laps (7.4596, 7.3487 s), then rejected at progress 204.708 m. Across
1,199 controller cycles, 1,183 converged, 7 hit the iteration cap, 8 exhausted
the ADMM time budget and one missed the 20 ms publication gate. Gate p50/p95/
p99/max were 9.43/16.44/18.38/20.042 ms; one cycle was over budget. Tracking
RMSE through rejection was 4.93 cm cross-track, 0.0447 rad heading and
0.411 m/s speed error. The final full solve hit 300 iterations with primal/
dual residual 0.107/0.026; candidate and retained-plan 75 ms prefixes both
failed exact corridor validation at stage 2. At the current state, `e_y` was
about -0.100 m. More iterations did not produce a safe action at that late
boundary state and reduced timing headroom; restore the 250 cap.
Actual-issued-command replay scored 13.8 mm endpoint XY RMSE, 13.4 mm/s forward
speed RMSE and 0.0137 rad/s yaw-rate RMSE, but reached 95.1 mm maximum path
error in a 750 ms window (37 windows exceeded 5 cm). This is not a test of an
unexecuted MPC plan.

Offline fixed-state replay of the a28 terminal row compared `weight_e_y=200`
and 250, steering-slew weight 1 and 0.5, `rho_max=32/64`, `rho_u=7/14/32`,
and reference-seeded initialization. None changed the conclusion: each
250-iteration solve remained iteration-limited with primal residual around
0.105–0.111 and failed the exact nonlinear corridor check at stage 2. A second
SQP pass also failed at stage 2, used about 17.1 ms in offline cycle timing and
did not improve the ADMM residual. These replays cannot restore live solver or
certified-plan memory, but show that simply pushing the same local solve
harder or making small penalty changes does not repair this terminal state.
The a29 online profile raised underspeed cost (`weight_u`) from 0.35 to 0.70,
keeping the fastest witness and all hard validation gates unchanged. It scored
three laps (7.4886, 7.3866, 7.2967 s; mean 7.3906 s), then rejected at 250.814 m
at corridor stage 1. Cross-track RMSE through rejection was 4.16 cm, heading
RMSE 0.0379 rad and speed error RMSE 0.361 m/s. Timing p50/p95/p99/max was
9.67/16.10/18.09/19.52 ms with no gate violation. Actual-command 750 ms replay
was 12.58 mm endpoint XY RMSE and 100.44 mm endpoint maximum; this is not an
unexecuted-plan score. The result was a small two/three-lap gain, not a stable
finish.

a30 combined `weight_u=0.70` with a modest `weight_e_y=250` increase. It made
one 9.1231 s warmup lap, then failed before a scored lap at progress 119.659 m.
The full horizon reached 250 ADMM iterations (0.0492/0.0216 residuals); exact
candidate and retained-plan 75 ms prefixes both failed at stage 2, only about
2 mm beyond the hard corridor. Tracking RMSE through failure was 6.16 cm,
0.0558 rad heading and 0.485 m/s speed error. The 20 ms cycle gate had a
21.675 ms outlier, while the scored-lap callback p95/max were 16.48/19.58 ms.
Thus the combined weight change did not improve robustness and regressed lap
completion; revert both weights to the known a26 baseline for comparison.

Implemented an exact-validated shorter recovery-prefix search in production:
on nonlinear rejection, check equal candidate/prior prefixes at 75 ms, then
50 ms, then 25 ms. Select only the longest duration for which at least one
prefix passes the unchanged nonlinear source-plant, actuator-envelope and
hard-corridor checks. Do not certify or publish the unvalidated tail; replan
at the next 25 ms tick. This directly addresses a30's failure mode, where the
first two predicted steps were safe but the third crossed the corridor. The
production image compiled successfully; unit-test binary and another online
trial are pending. It is a recovery mechanism, not proof of multi-lap success.

a31 isolated that code path using the prior a26 weights (`weight_e_y=200`,
`weight_u=0.35`) and the same 7.125 s source witness. It completed a 9.3461 s
warmup and one 7.4396 s scored lap, then reached a state at 142.645 m where
even the next 25 ms candidate and retained command both violated the hard
footprint corridor: predicted cross-track was +2.18 cm against a +1.57 cm
upper bound. The full solve hit its 250 cap (0.097/0.081 residuals). Tracking
RMSE through rejection was 6.03 cm / 0.0523 rad heading / 0.514 m/s speed
error; scored-lap timing p95/max was 18.14/18.79 ms, no budget violation.
The adaptive horizon correctly did not publish an action because even its
minimum 25 ms check failed. This identifies a real-time infeasible state, not
a prefix-length bug. The 39 prefix-selection unit tests pass and the current
production image builds. Next, repeat the a30 profile with the new code to
check whether shorter exact-valid prefixes actually carry the car beyond its
previous failure point; no ten-lap result exists yet.

a32 reran the a30 profile with adaptive validation and completed four scored
laps (7.3507, 7.2827, 7.3616, 7.3257 s; mean 7.3302 s), then rejected near
290.081 m. Cross-track/heading/speed RMSE through failure was 4.09 cm,
0.0381 rad and 0.295 m/s. Gate p50/p95/p99/max was 9.71/16.23/18.42/19.33 ms,
with no deadline violation. The terminal state had `e_y=-0.179 m`,
`v=-0.249 m/s`, `r=2.283 rad/s`; neither candidate nor retained action passed
even the one-step hard corridor check, so no action was issued.

The immediately preceding exact-validated 50 ms recovery decision exposed a
specific selection issue: candidate and prior were both safe, but candidate
had 6.03 mm minimum slack versus prior 2.63 mm, 3.40 mm better, and its first
two controls stayed within configured steering/throttle-change bounds. The
near-boundary selector still declined it because `minimum_clearance_gain_m=0`
intentionally disables that override (it requires a positive gain). It kept
the lower-clearance retained sequence; the next two cycles' slacks collapsed
to zero and then the next action became infeasible. Set the enabled minimum
gain to 2 mm and rerun this same profile. Candidate objective was about 20%
higher, within the configured 1.5 ratio; exact rollout, hard corridor, and
control-change bounds still gate selection. This is a targeted recovery-choice
change supported by the captured terminal chain, not a relaxed safety limit.

a33 used that 2 mm near-boundary threshold and reached four scored laps, with
the same 7.3302 s mean as a32, before its next rejection at 290.149 m. The
failure was not a timing miss: the final 25 ms source-plant prediction crossed
the lower footprint corridor by 2.3 mm with `e_y=-0.1338 m`, `v=-0.1269 m/s`,
and `r=2.0919 rad/s`; the local path reference was about 5.23 m/s and
`kappa=0.289 1/m` (reference yaw rate about 1.51 rad/s). Thus the car reached
the curvature-unwind section with excess yaw and lateral motion. The previous
exact-valid plan chain had reached seven actions.

An offline weight sweep at the a33 terminal states raised yaw-rate cost from 2
to 4, then to 8 while heading cost rose from 30 to 60. It did not restore a
feasible full candidate horizon; terminal rejection still advanced to stage
1. This does not justify a broad yaw-weight increase.

a34 tested a two-action stale-plan refresh and the temporary convergence-path
change on the same 7.125 s source witness. It rejected at 65.17 m before any
scored lap. Around 63.57 and 64.07 m, the new policy selected converged
candidates with only 1.2–1.5 mm more 75 ms clearance and 4–9% worse exact-prefix
objective; near 64.47 m it refreshed an iteration-limited candidate with
22.8 mm more local clearance. The car then reached `e_y=-0.166 m` at 65.17 m,
where the next one-step prediction crossed the corridor. The corresponding a33
passes at this map location stayed near `|e_y|<2 cm`. This is evidence that
the experimental stale-refresh expansion did not transfer safely; it is not
promoted.

Restore stale refresh to its iteration-limited-only behavior and three-action
threshold. a35 tested only the 1 mm near-wall selector preference (with stale
policy restored) and completed one scored lap at 7.2877 s before rejecting at
142.773 m. The final predicted point exceeded the upper corridor bound by
0.72 mm; timing p50/p95/p99/max was 9.36/15.98/18.16/19.48 ms, with no timing
gate miss. The 1 mm threshold did not prevent the failure and is not retained.

At 142.645 m, the previous accepted rollout predicted `e_y=0.0089 m`,
`u=2.529 m/s`, `r=2.796 rad/s` at the next sample; the measured state at
142.713 m was `e_y=0.0169 m`, `u=2.603 m/s`, `r=2.708 rad/s`. The roughly 8 mm
lateral discrepancy is large relative to the local remaining clearance, even
though speed/yaw-rate errors were modest. Offline fixed-state replay raised
`weight_e_y` from 250 to 500/750/1000/2000. At the 142.274 m state this moved
the first nonlinear failure from stage 26 to 29 or removed it for one tested
profile; at 142.581 m it moved stage 22 to 23–25, but iteration-limited cases
reached ~20.6–24.5 ms replay cycles and did not become fully feasible. This is
promising but timing-risky evidence, not a promotion. a36 tests `weight_e_y=500`
online with the a33 stale/clearance-selection settings restored; all hard
corridor and validation gates remain unchanged.

The fastest IROS line still ranks at 7.125 s in the source-only optimizer and
is not Unity-validated. No current MPC profile has completed the required ten
scored laps on it. a35 was faster on its single scored lap than a33's mean, but
it is not a repeatable result.

a36 tested `weight_e_y=500` against a33's 250 and failed before a scored lap at
80.120 m. Its final exact rollout crossed the upper footprint corridor by
1.12 mm at the first 25 ms step; timing remained below the 20 ms gate
(p50/p95/p99/max 8.56/15.51/16.65/18.38 ms). The car entered the preceding
high-curvature section with `e_y≈+2.8 cm`, while a32/a33/a35 traversals at the
same progress were generally closer to the path. The heavier lateral penalty
therefore did not improve robust tracking and is reverted.

An exact production-source reachability grid at the a36 terminal state found
0 safe constant-action sequences out of 112 over 75 ms, and none over 150,
300, or 500 ms. This is a terminal state with no sampled safe action, not a
reason to relax the footprint constraint. Yaw-rate penalties 2/4/8 all hit the
offline iteration cap on the final bend states and did not restore feasibility.
Reducing the source-steering-target cost from 50 to 5 or 15 increased exact
75 ms clearance at the immediately preceding fixed state (about 0.26 mm to
2.19/2.46 mm), but those isolated cold replays hit the iteration cap; lower
steering-slew weights 0.1/0.3/0.6 barely changed the command or clearance.

a37 restored `weight_e_y=250` and tested a moderate
source-steering-target cost of 15 (from 50), with the same fast 7.125 s source
witness, strict nonlinear footprint/corridor gate, and iteration/time limits.
The rationale was to let state feedback depart from witness steering when the
measured path error demands it, without applying the more extreme reduction
that produced capped offline solves. Results follow below.

a37 completed three scored laps (7.4676, 7.3457, 7.6376 s; mean 7.4836 s)
before rejection at 233.433 m. It did not improve on a33's four-lap/7.3302 s
partial result. The final 25 ms prediction crossed the upper footprint bound
by 5.38 mm; the exact reachability grid found no safe constant-action choice
from that state. The target at that point requested 4.494 m/s and +3.39 m/s²,
while measured speed was 4.226 m/s and the applied throttle was 0.131. The
source witness throttle at the same path station is about 0.31. Timing
p50/p95/p99/max was 9.91/16.48/18.11/20.12 ms, with one cycle at or above the
20 ms gate. The source-steering weight reduction is not promoted.

Code review found a real asymmetry in source-witness use: source steering is
an explicit QP/line-search target, but source throttle only initializes the
nominal sequence. The QP then chooses throttle using speed error, zero-centered
throttle effort and slew costs, without a soft penalty for departing from the
known source-witness throttle. Added `weight_throttle_target` to production
configuration, QP construction, scalar/exact objective evaluation, and cycle
diagnostics; its default is zero, so ordinary geometry references and current
production behavior remain unchanged. Unit coverage checks the QP Hessian,
linear term and scalar target cost. Build and all 13 MPC package tests pass.

The next serialized a38 trial restores a33's `weight_e_y=250` and source
steering weight 50, then enables only a conservative 0.10 source-throttle
target weight against the same 7.125 s source witness. The online exact
nonlinear footprint/corridor validator is unchanged. This directly tests
whether using the optimizer's source command as feedforward reduces the
observed speed/control mismatch without sacrificing convergence or the 20 ms
gate; no result is promoted until measured.

a38 did not produce a scored lap. It rejected at 62.650 m after the preceding
cycle reached the 250-iteration cap; the next predicted point exceeded the
upper footprint bound by about 3.7 mm. Timing p50/p95/p99/max was
8.40/16.78/18.68/18.90 ms, with no 20 ms miss. The experimental throttle-
target term did not demonstrate improvement and was removed from production
code, configuration, tests, and diagnostics after this result.

At the same location, a33 passed with `e_y≈+0.2 mm`, while a38 entered at
`e_y≈+2.3 cm`; the available next-step upper bound was about +1.94 cm. That
means the fatal rejection was the consequence of arriving on the wrong side
of a narrowing corridor, not a timing miss. a39 therefore keeps a33's source
steering/throttle objective (source throttle target remains zero) and raises
only `weight_e_y` from 250 to 1000. The line, speed targets, corridor and
exact rollout checks are unchanged. This is a targeted test of earlier
lateral correction, not a claim that higher weight is already validated.

a39 rejected at 76.985 m after 336 scored cycles and no completed lap. Its
cross-track error grew from about -1.6 cm at 74.61 m to -11.0 cm at 76.73 m;
the final solve hit the 250-iteration cap with primal residual 0.692 and neither
candidate nor retained-plan prefix passed exact rollout/corridor validation.
Gate timing remained below 20 ms (max 18.28 ms). Raising `weight_e_y` to 1000
was harmful and is reverted. The optional source-throttle target added for
a38 also failed to show benefit and has been removed from production code,
configuration, tests, and diagnostics. The next attempt restores the better
supported a33 lateral profile (`weight_e_y=250`, steering-target 50); the
fast source-only 7.125 s witness and hard corridor remain unchanged.

a40 restored `weight_e_y=250` but still failed before one lap at 57.636 m.
This run began in the same reproducible start state as a32/a33; unlike those
runs, speed rose to 8.94 m/s while the reference was 6.97 m/s, then remained
5.01 m/s as the target fell to 2.45 m/s in a 1.24 m^-1 curvature section.
The QP kept requesting roughly 0.49 throttle through the overspeed buildup.
Timing remained inside the gate (p50/p95/p99/max 12.22/17.11/18.10/19.02 ms;
zero misses). This points to insufficient speed-error priority in the active
overlay, not an iteration-count or deadline problem. For a41 the overlay
restores the production `weight_u=35` and `weight_u_overspeed=20`, retaining
the a33 lateral/steering profile and all hard validation. This tests early
speed correction without adding zero throttle or weakening the corridor.

a41 used the production speed penalties and did correct the a40 overspeed:
near 50 m it was 6.01 m/s for a 6.39 m/s target (a40 was 8.24 m/s for 6.38).
It still rejected at 76.889 m after 311 cycles and no completed lap.
Cross-track error grew from roughly -1 cm near 74.6 m to -10.6 cm by the fault;
lateral velocity stayed around -0.04 to -0.06 m/s while yaw rate was about
1.8-2.0 rad/s. Timing p50/p95/p99/max was 7.70/17.49/18.64/19.59 ms, zero
gate misses. For a42 the overlay keeps a41 speed/path/steering values and
restores production longitudinal-slip, lateral-velocity and yaw-rate costs
(5.0, 5.0, 7.8125). This tests whether reducing persistent sideslip prevents
the repeated lateral divergence; hard footprint/corridor gates remain intact.

a42 rejected at 76.340 m after 311 cycles; lateral error reached -18.6 cm,
worse than a41's -10.6 cm, with no completed lap. Its solver repeatedly timed
out or hit the iteration cap; exact-prefix recovery kept the prior plan until
the first-step corridor check failed. Raising longitudinal-slip, lateral-
velocity and yaw-rate costs to production values did not help and is reverted.
For a43, retain a41's production speed weights and lower slip/yaw costs, but
raise only `weight_e_psi` tenfold (30 to 300). This targets the observed
negative heading lag during the growing negative cross-track error. The
previous 1500 heading weight is deliberately not reused because it destabilized
the high-curvature solve; all nonlinear acceptance/corridor constraints remain
unchanged.

a43 rejected at 62.492 m after 225 scored cycles. At the terminal sample,
`e_y=+3.30 cm`, while the exact hard upper corridor was +1.57 cm; the current
6 cm soft clearance reserve had shifted the lateral target to about -5.12 cm.
The controller was being asked to move roughly 8.4 cm across track into a
section whose physically safe corridor had only about 1.6 cm of positive-side
room. It reached that point with speed near target (4.72 vs 4.37 m/s) and
timing p95/max 18.03/19.10 ms. The increased heading penalty did not prevent
the failure and is reverted. For a44 only the *soft* reference reserve is set
to zero: the hard 5 cm vehicle footprint/corridor validation stays enabled.
This avoids double-reserving space already accounted for by the raceline and
hard footprint margin, while allowing the solver to use the full safe corridor.

a44 reached 77.104 m (farther than a43's 62.492 m) with the 6 cm soft reserve
removed, and reduced the observed lateral error to about -5.2 cm at rejection
(a41 was -10.6 cm). It still ended with no exact-valid command as the lower
hard corridor bound tightened: at the last accepted state the next-state
lower bound was about -2.6 cm, while the predicted lateral position was about
-4.1 cm. Timing stayed below budget (p95 16.73 ms, max 18.80 ms).

a45 also failed at the same late transition (77.18 m, no lap completed), with
cross-track error reaching 10.75 cm earlier in the run and about 6.3 cm at the
terminal state. The final predicted state exceeded its hard lower corridor by
about 8 mm; timing stayed below 20 ms (p95 16.21 ms, max 18.92 ms). Raising
`weight_e_y` to 1000 did not fix the failure and is reverted. The time series
shows the source steering target stepping from about +0.123 rad to near zero
over roughly 0.1 m while yaw rate remains high; the controller also hit
iteration/time limits there.

a46 tested a global 0.08-rad steering trust region. It failed much earlier,
at 73.27 m, when the exact candidate's next-step corridor slack was -1.3 mm;
the accepted-plan recovery also had no valid prefix. Timing was p95 16.9 ms,
max 19.0 ms. This confirms that widening every SQP stage is not a safe remedy.
The source-steering jump is localized in path distance, so production now has
an opt-in stage-local expansion: keep the 0.04-rad base and expand only where
the source-witness steering target delta exceeds that base, capped by an
explicit 0.08-rad maximum. No expansion occurs for geometry-only references,
and every result still must pass the same exact nonlinear corridor check.

a47's stage-local expansion reached one warmup lap and then failed at 115.82 m
with a 3.6 cm exact-corridor violation. Timing p95/max was 16.59/19.21 ms.
This was farther than a44-a46 but not a multi-lap result, so it is not yet
promoted. For a48, reproduce a33 on the same witness: `weight_e_y=250`,
`weight_e_psi=30`, speed costs 0.70/2.0, 0.04-rad global trust region, 6 cm
soft reference reserve, and transition expansion disabled. This checks
whether the prior four-scored-lap result is repeatable with the current build;
the hard nonlinear/corridor checks and 20 ms gate remain unchanged.

a48 reproduced a33's control profile on the identical witness and completed
four scored laps at 7.4646, 7.3686, 7.4226, and 7.5366 s (7.4481 s mean),
with no collision. It then rejected at 293.56 m: the next predicted lateral
position was +2.19 cm against a +1.72 cm hard upper bound, a 4.6 mm excess.
The terminal QP hit its 250-iteration cap with primal/dual residuals 0.110/0.094;
its complete cycle took 10.1 ms, so the cap—not the 20 ms cycle budget—was
binding there. Across the full run, timing p95/max was 16.29/22.08 ms with two
publication-gate misses while the long optimizer was active. a49 requested
500 iterations, but startup diagnostics showed an effective cap of 300 because
the node silently clamped against compile-time `MAX_ITERATIONS=300`. It stopped
at 104.90 m (one warmup lap, no scored lap) with a corridor rejection; timing
p95/max was 16.24/21.32 ms and one publication-gate miss. The requested 500
was therefore not actually tested. Added an explicit startup warning and a
`max_iterations_requested` diagnostic field so requested/effective caps cannot
be conflated. a50 then tested the same profile with a real 500 cap: it
completed four scored laps at 7.4316, 7.2977, 7.3806, and 7.4056 s (7.3789 s
mean), then failed at 289.86 m. The final solve reached 500 iterations with
primal/dual residuals 1.261/0.0092 and failed exact corridor validation; the
car was already at `e_y=-0.217 m`. Timing reached 22.62 ms, over the 20 ms gate.
More iterations neither made the solve feasible nor improved the lap count, so
the cap is returned to 300; the requested/effective-cap diagnostics remain.

The a50 prefix diagnostics identified a concrete recovery gate mismatch. At
289.751 m, candidate and retained 75 ms prefixes both passed exact plant and
hard-corridor validation, with 6.77 mm and 13.63 mm of slack respectively.
Candidate objective was 50.33 versus 41.70 (within the configured 30% stale
refresh limit), and the candidate's clearance loss was only 6.86 mm (within
the 20 mm limit). Yet the controller retained the prior sequence for 15
consecutive actions because `quality_prefix_clearance_reserve_m=0.05` required
an additional 5 cm of slack on top of the hard corridor's existing 5 cm
vehicle-to-wall margin. This was a double-counted reserve, not a failure of the
hard safety gate. The competition profile and next validation overlay now set
that selector-only reserve to zero. The 5 cm hard corridor, exact nonlinear
validation, 20 mm maximum candidate clearance loss, 30% objective limit, and
bounded command-delta checks remain unchanged. A regression test captures the
a50 candidate/prior values.

The failed stage-local steering-trust expansion from a47 has been removed from
the runtime interface and implementation; no production or test profile used
it, and its only live trial did not complete a scored lap. On the extracted 33
terminal-state IROS replay rows, cold ADMM initialization reduced iteration-cap
rows from 15 to 13 but did not increase exact-accepted actions (6/33); fixed rho,
rho-max 64, `weight_e_y=500`, lower steering-target weight, and lower steering
slew weight likewise did not improve exact-accepted actions. None of those
weight/penalty changes is promoted. a51 below is that requested closed-loop
check; the online and offline results did not justify promoting a new weight.

### a51 — selector reserve correction and competition-lap replay

a51 rebuilt the production controller and ran the 7.125 s source-certified IROS
witness with the 300-iteration cap and `quality_prefix_clearance_reserve_m=0`.
The cap and zero reserve were confirmed in the live parameter overlay/controller
logs. The 9.151 s lap was the warm-up lap; **zero scored laps** completed before
the run stopped at 100.03 m
with `rejected_nonlinear_rollout` (corridor at stage 0); it did not collide
before the run supervisor stopped it. Thus a51 is not a successful multi-lap
validation. It issued 462 commands over 463 scored cycles; gate p50/p95/max was
9.11/16.45/19.82 ms, with no cycle at or above the 20 ms gate.

The zero selector reserve did not cure this distinct fault: exact validation
found neither the new candidate nor retained plan valid at the failing state.
Predicted stage-0 `e_y=+0.017931 m` exceeded its already tolerance-adjusted
upper corridor bound `+0.015788 m` by 2.14 mm. The diagnostic's `margin_m` is
the projected body envelope, not a measured wall clearance. The data therefore
do not justify loosening the 5 cm hard wall-clearance constraint. The lateral
error had grown from near zero to about +2 cm as curvature rose from near zero
to about 1.2 m^-1, while heading error reached about -0.07 rad; the failure is
late steering/path tracking at this tight section, not rejection caused by the
extra selector reserve.

An offline sequence replay rebuilt 463 solver states from a51 diagnostics. It
does not restore the live ADMM iterates or certified-plan memory and is useful
only for comparative screening. Baseline replay had 6 corridor-rejected rows.
Increasing `weight_e_psi` from 30 to 60 produced 36 rejects; increasing
`weight_e_y` from 250 to 500 produced 70. Neither is a viable weight change.
Lowering `weight_steering_target` to 25 alone reduced this reconstructed count
to 5, but the final two first-stage failures remained; raising it to 100 gave
12 rejects. Reducing steering-slew weight to 0.25 alone left the count at 6.
Changing only the soft reference-clearance reserve from 0.06 to 0.10 m yielded
5 rejects; 0.15 m yielded 42. A combined reserve/weight/slew variant yielded
8. These small offline count changes do not establish closed-loop improvement.

### a52 — steering-target weight A/B

a52 changed only `weight_steering_target` from 50 to 25 while retaining the
zero selector reserve and 0.06 m reference reserve. It stopped at 62.61 m,
before completing the warm-up lap, with one stage-0 corridor rejection after
227 controller cycles. Gate p50/p95/max was 6.29/16.18/18.85 ms, with no
20 ms violations. At rejection, predicted `e_y=+0.020391 m` exceeded the
tolerance-adjusted upper bound `+0.019632 m` by 0.76 mm; neither candidate nor
prior prefix was exact-valid. Thus the offline 5/463 versus 6/463 difference
did not transfer and the lower steering-target weight is rejected.

### a53 — soft interior-reference target A/B

a53 used steering-target weight 50 and a 0.10 m soft reference-clearance
reserve. It completed the warm-up lap (9.160 s) and two scored laps (7.537 s,
7.467 s), then stopped at the next narrowing section at 208.18 m. At rejection,
cross-track error was +26.0 mm and the exact first-step rollout reached +22.5 mm
against a +17.0 mm allowable bound. There was no collision; the exact gate
withheld the command because the predicted body envelope no longer maintained
the requested wall reserve. Scored-lap cross-track RMSE was 23.9 mm and 43.6
mm, with maxima 73.3 mm and 167.0 mm. Speed-error RMSE was 0.401 m/s, mostly
underspeed; the laps remained 0.34–0.41 s above the 7.125 s source-witness
time. Timing stayed under the 20 ms gate: p50/p95/max 9.72/16.41/19.92 ms, no
cycles at or above 20 ms.

The 0.10 m soft reserve remains validation-only: a53 outlasted a51's earlier
rejection, but it did not complete the target laps.

### a54 — underspeed-cost A/B

a54 changed only underspeed `weight_u` from 0.70 to 1.40; steering-target
weight remained 50, soft reserve 0.10 m, iteration cap 300, and hard
corridor/collision checks were unchanged. It completed six scored laps at
7.327, 7.496, 7.427, 7.472, 7.355, and 7.462 s (mean 7.423 s, range 0.169 s)
before a later stage-0 exact-corridor rejection at progress 356.15 m. It did not
collide and did not finish ten laps. Relative to a53's two scored laps (mean
7.502 s), a54's speed-error RMSE fell from 0.401 to 0.304 m/s and cross-track
RMSE from 0.0513 to 0.0432 m. This is a promising but not yet repeatable gain.
The rejection was again a narrow-boundary event: predicted `e_y=+20.3 mm`
against an allowable `+16.7 mm`; both candidate and prior exact validation
failed at stage 0. Timing p50/p95/max was 9.92/16.66/19.57 ms, with no cycle
at or above 20 ms.

The next test repeats a54 unchanged on the same certified 7.125 s witness.
Only after repeatability is established should this underspeed weight be
promoted beyond the serialized validation profile.

### a55 — repeated global speed-weight trial

a55 repeated a54's `weight_u=1.40` profile unchanged. It failed before the
warm-up lap completed, at 76.83 m, with no scored laps. Timing p50/p95/max was
7.42/16.78/19.37 ms (zero cycles at/above 20 ms). The terminal predicted state
was 5.07 m/s, `v=-0.292 m/s`, `r=1.92 rad/s`, giving `|u*r|≈9.75 m/s²`; its
cross-track state was -95.7 mm against a -90.8 mm bound. The +0.0049 m
corridor miss is small, but this was a real hard-gate failure. a54's six-lap
result therefore did not repeat, and globally raising speed pressure is not
robust.

### a56 — curvature-scheduled speed priority

The next test keeps `weight_u=1.40` but sets the already-supported
`underspeed_curvature_gain_m=2.0`. The implemented schedule is
`weight_u/(1 + gain*abs(curvature))`: it preserves stronger speed catch-up on
straights while reducing it as curvature and lateral demand rise. Steering,
throttle limits, iteration cap, path, and exact hard-corridor validation are
otherwise unchanged. This is a test overlay only; promotion requires completed
scored laps with better speed and no earlier safety rejection.

a56 failed before its warm-up lap completed, at 102.20 m. It had one timing
cycle above 20 ms (max 20.405 ms); p50/p95 was 9.65/17.37 ms. The predicted
state was `e_y=-66.8 mm` against a `-53.9 mm` lower bound, with
`u=4.08 m/s`, `r=-1.86 rad/s`, and curvature `-0.190 m^-1`. The path rate at
that speed is about `-0.77 rad/s`, so yaw rate was substantially more negative
than the path demand. The model also failed to converge (327 reported
iterations; primal residual 1.56). The single-run data do not establish that
curvature scheduling caused the failure, but identify yaw-rate mismatch as a
direct issue. a56 is not promoted.

### a57 — yaw-rate tracking A/B

a57 held the a56 speed schedule (`weight_u=1.40`, curvature gain 2.0) and
changed only `weight_r` from 2 to 6. It completed one scored lap at 7.505 s,
then rejected at 165.51 m with predicted `e_y=+27.6 mm` versus a `+16.6 mm`
upper bound. There was no collision. Timing p50/p95/max was 10.10/16.81/19.51
ms, with no over-budget cycles. The terminal solver was iteration-limited at
300 with primal residual 0.173. This does not show a reliable gain from the
yaw-rate weight change; a57 is not promoted.

### a58 — stronger curvature attenuation

a58 resets `weight_r` to 2 and changes only the a56 curvature gain from 2 to
4. With `weight_u=1.40`, this returns the underspeed cost close to the old
0.70 baseline already around `|curvature|=0.25 m^-1`, and attenuates it further
in tight bends, while preserving the higher cost on straights. This is an
isolated test overlay; no hard corridor or safety rule changes.

a58 completed one scored lap at 7.483 s after a 9.176 s warm-up, then failed at
162.14 m with predicted `e_y=-150.1 mm` against a `-118.4 mm` lower bound.
There was no collision. Timing p50/p95/max was 9.68/16.33/19.43 ms, all under
20 ms. At rejection, the iteration-limited QP residuals were 0.523 primal
and 0.117 dual. The proposed first action increased throttle by 0.10 and
reduced steering by 0.04 from nominal, despite the state already being outside
the narrow-side corridor. Curvature attenuation alone did not produce a
feasible correction.

### a59 — lateral-velocity tracking A/B

a59 changes only `weight_v` from 0.05 to 1.0 relative to a58. This gives the
source-witness lateral velocity a meaningful objective contribution while
retaining the same speed schedule, yaw weight, path and hard checks. It tests
whether the plant's lateral state is being underweighted during the measured
drift; it remains validation-only.

a59 completed two scored laps at 7.367 s and 7.421 s (mean 7.394 s) after a
9.290 s warm-up, then rejected at 187.72 m. Scored-lap cross-track RMSE was
15.8 mm and 23.5 mm; speed-error RMSE was 0.215 and 0.263 m/s. Combined scored
RMSE was 49.5 mm cross-track and 0.411 m/s speed. Timing p50/p95/max was
9.53/16.62/19.57 ms, with no over-budget cycles. At rejection,
`e_y=-67.0 mm` exceeded the `-55.5 mm` bound by 11.5 mm, with `e_psi=-0.125
rad`; the solver reached its iteration cap with primal residual 0.277. This is
better than a58's single scored lap and tracking/speed errors, but not enough
to establish repeatability or safe ten-lap use.

### a60 — unchanged a59 repeat

a60 repeated the exact a59 profile. It completed five scored laps at 7.497,
7.477, 7.471, 7.389, and 7.510 s (mean 7.468 s, standard deviation 0.042 s)
before a hard-corridor rejection at 332.56 m. There was no collision. Timing
p50/p95/max was 9.82/16.64/19.70 ms, with no cycle at or above 20 ms. Scored
cross-track RMSE was 30.5, 37.1, 18.4, 20.5, and 36.7 mm; speed-error RMSE was
0.366, 0.260, 0.369, 0.235, and 0.329 m/s. At rejection `e_y=-257.7 mm`
exceeded the `-219.9 mm` bound by 37.8 mm; yaw error was small but lateral
velocity was `-0.096 m/s`, and the solver primal residual was 0.836 at the
300-iteration cap. The improved lateral-velocity profile extended from two
scored laps in a59 to five in a60, but this still is not a ten-lap pass.

### a61 — unchanged a59/a60 profile repeat

a61 repeated the same `weight_v=1.0`, `weight_u=1.40`, curvature gain 4.0
profile. It completed one scored lap at 7.375 s, then rejected at 142.62 m.
Timing p50/p95/max was 9.61/16.70/19.27 ms, all under 20 ms. The next exact
state had `e_y=+19.8 mm` against a `+16.2 mm` bound; no collision occurred.
This confirms that the a60 five-lap result is not yet robust across starts and
the limiting failure is still a narrow-turn tracking error.

### a62 — stronger soft interior target (rejected: startup timing)

a62 changed only `reference_clearance_reserve_m` from 0.10 to 0.15 m relative
to the a59/a60 profile. It ended before the first scored lap, while stationary
at startup, because both MPC cycles exceeded the strict 20 ms publication gate:
24.49 ms and 20.91 ms. They reached only 21 and 42 ADMM iterations and were
stopped by the time budget. The first solve reported 22.46 ms total solve time;
the second reported 17.28 ms, with 5.84 ms Jacobian wall time, 5.94 ms QP
build, 8.09 ms ADMM, and 3.13 ms exact rollout (component timers are not
strictly additive). There was no corridor rejection or collision. Because no
lap was scored, this does not establish that the larger soft reserve caused
the timing spike or changed driving. The profile has been returned to 0.10 m
for the next repeat; the hard physical corridor remains unchanged.

### a63 — unchanged a59/a60 profile repeat

a63 restored the 0.10 m soft reserve after a62 and repeated the same 7.125 s
source-certified witness. It completed two scored laps (7.544 s, 7.410 s;
mean 7.477 s), then rejected at 204.585 m. Timing p50/p95/max was
10.00/17.30/25.79 ms; two cycles exceeded 20 ms. Near the terminal point,
the prior sequence had been consumed. Both exact-valid 75 ms prefixes existed
at 204.286 m, but the safer/lower-cost candidate (66.95 vs 73.11 objective;
37.1 vs 22.8 mm slack) was rejected solely because one throttle difference was
0.2145, just above the 0.20 near-boundary selector cap. The following retained
actions consumed the plan, and at 204.585 m neither new prefix was corridor
valid. This directly motivated the bounded selector change below.

### a64 — allow a measured safe deceleration prefix

The near-boundary throttle-difference limit increased from 0.20 to 0.22. This
is only a selector eligibility bound: both candidate and retained prefixes
still require exact nonlinear source-plant and hard-corridor validation. A
regression test reproduces the a63 controls and confirms the 0.2145 difference
remains bounded at 0.22. a64 used the same path, weights, 0.10 m reserve and
other settings as a63. It passed the former 204 m failure and completed four
scored laps at 7.463, 7.333, 7.470 and 7.520 s (mean 7.446 s, standard
deviation 0.069 s). Timing p50/p95/max was 9.67/16.46/19.52 ms, with no
cycle at or over 20 ms. It then failed at 270.771 m in the next tight turn:
the 300-iteration candidate and retained prefix both failed exact stage-0
corridor validation. The state had `e_y=+23.7 mm`, `e_psi=-0.068 rad`,
`u=2.578 m/s`, `r=2.776 rad/s`; the next predicted candidate remained about
3.2 mm beyond the current upper corridor bound. The QP hit its iteration cap
with 0.069 primal residual. This is a later physical/tracking failure, not a
regression in the selector; a64 is still not a ten-lap pass.

The default near-boundary throttle-difference limit is now 0.22, with the hard
5 cm corridor unchanged. a65 changes only the soft clearance reserve from
0.10 to 0.15 m to test whether the controller begins correcting earlier for
the later tight turn. This remains a validation-only trial.

a65 completed three scored laps (7.438, 7.505, 7.354 s; mean 7.432 s,
standard deviation 0.062 s), then rejected at 209.427 m. Timing remained below
the 20 ms gate (p50 10.17, p95 17.03, max 19.62 ms). The terminal QP hit 300
iterations with primal/dual residuals 0.698/0.425, and neither candidate nor
retained prefix passed the exact corridor check.

The trace exposed an earlier selector mismatch. At 209.062 m both three-command
prefixes passed exact validation. The candidate had lower exact rollout cost
(94.81 vs 96.73), 12.5 mm more corridor slack (127.9 vs 115.4 mm), and bounded
first controls, but was not selected. The path-target reserve was 0.15 m while
the separate quality-selector reserve was 0.0 m; that disabled the existing
near-boundary override. The selector reserve is now set to 0.15 m for the next
serialized validation, with a captured-case unit test. This changes only the
preference between two independently exact-valid prefixes; it does not relax
the hard corridor check or certify an unvalidated horizon. Whether it delays
the eventual failure remains to be measured.

a66 is that selector-reserve test. It passed the former 209 m failure and
completed six scored laps (7.438, 7.544, 7.520, 7.692, 7.610, 7.626 s; mean
7.572 s) before rejecting at 375.548 m. Timing remained within the gate
(p50 10.35, p95 17.06, max 19.85 ms; zero cycles at/over 20 ms). At the new
failure, cross-track error had grown to -0.150 m against a -0.131 m bound. The
source witness requested forward/lateral velocity 5.28/-0.210 m/s and yaw
2.17 rad/s; measured state was 4.49/-0.069 m/s and 2.08 rad/s. Yaw was close,
while lateral velocity lagged by about 0.14 m/s and heading error was 0.015 rad
versus a source heading-error target near 0.035 rad. The terminal 300-iteration
QP had 0.567 primal and 0.172 dual residual, with neither exact prefix valid.
The a65 selector correction passed the earlier section but did not resolve
this later tracking/convergence problem.

a67 rejected much earlier, at 161.774 m after one scored lap. Near that
section, the a66 profile had `e_y=-0.055 m` at 161.44 m; a67 with `weight_v=5`
had reached `e_y=-0.212 m` at 161.36 m and then `-0.248 m`, beyond its
`-0.219 m` hard bound. The final QP hit 300 iterations with 0.844 primal
residual. This is a clear regression for the 5x lateral-velocity weight; do
not promote it. Timing stayed under 20 ms (max 19.44 ms), so the failure was
tracking/solve quality, not compute deadline.

a68 restored `weight_v=1` and tested `weight_e_psi=60`. It completed three
scored laps (7.407, 7.589, 7.321 s) before rejecting at 250.947 m. The next
predicted e_y was 0.0183 m against a 0.0177 m upper bound (about 3.4 mm beyond
the bound); both candidate and prior exact-prefix checks failed. The terminal
QP reached 300 iterations with 0.069 primal residual. Timing stayed below
20 ms (max 19.45 ms). Heading error was less negative than in a66, but that did
not prevent the bound miss; revert `weight_e_psi` to 30 rather than promote
the 2x value.

a69 repeated the a66 profile (`weight_v=1`, `weight_e_psi=30`, both soft
reserves 0.15 m) on the same fastest witness, but failed during warm-up at
119.148 m and scored no laps. It entered a retained-plan chain at 116.413 m;
at 117.876 m the candidate and retained 75 ms prefixes were both exact-valid.
The candidate's exact cost was 19.50 versus 17.01, but its minimum corridor
slack was 0.2499 versus 0.2414 m. The 0.15 m selector reserve was inactive
because both prefixes still had more than 0.15 m wall slack. As the chain grew,
the vehicle's `e_y` moved from -0.087 m at 117.876 m to -0.200 m at 119.148 m;
the final 300-iteration solve had 0.553 primal residual and neither prefix
passed the hard corridor gate. This run shows that matching the lower-cost
retained prefix is not sufficient for repeatable tracking, even when both
prefixes pass the short exact check.

For a70, increase only the validation-overlay selector reserve to 0.30 m.
This lets the bounded near-boundary rule compare the same a69 captured case
earlier; add a unit regression for the captured objective/slack values. It
does not relax the hard corridor gate, and it is not production-promoted.

### Current gate

The a50 stale-plan selector fix remains in production: its default
selector-only reserve is zero while exact plant/corridor checks and the
configured physical corridor remain intact. a66 explicitly enabled a 0.15 m
soft selector reserve in its validation overlay; that moved the failure later
but was not sufficient for completion.
The earlier a51 result also showed this alone is insufficient for repeatable
racing. No tested weight
change has passed the ten-lap gate. The next work should target earlier
steering/path-error buildup before the narrow section and replay the actual
saved trajectory states with control-history caveats. a54 gives the first
clear speed-weight gain (six scored laps, mean 7.423 s) but still stops at a
hard-clearance boundary, and a55 did not repeat that gain. a56 failed early
with one timing miss; a57's higher yaw-rate weight did not prevent another
corridor rejection. a58 also failed at a bend with a high QP residual and an
inward correction not selected; a59 improved two laps after increasing
lateral-velocity tracking and a60 completed five with low variance, but a61
did not reproduce that distance. a62 ended at startup on two timing-gate
misses and scored no lap. The bounded near-boundary selector correction
allowed a64 to pass the old 204 m failure and complete four scored laps, but
it rejected later at a tight bend. a65 tested a larger path-target reserve;
a66 tested the selector reserve and advanced to six scored laps. a67 rejected
early with `weight_v=5`, and a68 rejected with `weight_e_psi=60`; both weights
were restored. a69 repeats the best-performing a66 profile. No profile is
production-promoted until it
repeatedly completes ten scored laps within the timing gate. The independent
IPOPT search remains running; do not stop it.
