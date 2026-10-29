# IROS competition MPC: runtime and validation progress

## Current gate

The optimizer's best source-certified reference is now **7.225 s**, witness
`b40ae5a6097efcb0`; its first automatic Unity test failed without completing
warmup (209 full-horizon accepts, 18.879 ms maximum gate time, 2.334 cm maximum
fixed-command prediction error). Its second test with the improved profile
completed a **9.130 s warmup**, then failed in lap two at progress 107.475 m:
512 full-horizon accepts, 19.172 ms maximum gate time, 3.687 cm maximum
fixed-command prediction error over 513 windows. Controller comparisons below
use the earlier **7.400 s** witness
`f8c5b636134d8b64`. **a09 completed one warmup lap (9.314 s), then failed in
lap two. No scored lap or repeatable racing success yet.** The practice records
are not IROS competition results. Source certification, QP convergence,
accepted nonlinear horizons and actual completed driving are separate claims.

Single dashboard: <http://127.0.0.1:8765/>. Single optimizer workspace:
`live_runs/optimizer_iros_conditioned_20261029`. The validator pauses that
optimizer during builds, driving and replay, then resumes the same iterate.

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
