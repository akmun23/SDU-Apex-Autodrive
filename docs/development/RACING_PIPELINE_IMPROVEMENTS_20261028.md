# Racing pipeline improvements — 2026-10-28

## Goal and starting point

Ten consecutive scored practice laps below 50 s, excluding warmup; repeated
success, not a reference integral or one fast lap. Best previously verified
ten scored laps: 52.239 s. Current warm-start gate fix is `61539ce98`; its
verification and failed weight/SQP experiments are documented in
`MPC_REPAIRABLE_WARM_START_20261028.md`. No physics or acceptance gate is
relaxed here. Existing IPOPT continuation remains running independently.

## Attribution before edits

The last slip-1 capture's 1,371 command-time predictions were compared with
interpolated later Unity truth, excluding samples after the first fault. This
is offline analysis only; truth is never an autonomy input.

| Command-time state | RMSE | Maximum absolute error |
| --- | ---: | ---: |
| XY position | .0759 m | .1485 m |
| Forward position component | .0706 m | .1483 m |
| Lateral position component | .0278 m | .0746 m |
| Yaw | .0036 rad | .0247 rad |
| Forward speed | .0508 m/s | .7503 m/s |
| Lateral speed | .0058 m/s | .0709 m/s |
| Yaw rate | .0702 rad/s | .8074 rad/s |

In the final 500 ms, lateral pose error was only .0236 m RMSE (max .0333 m),
while MPC's estimated cross-track error grew beyond .49 m. Localization is
not perfect, particularly along-track, but it does not explain away that
large tracking excursion. Tail velocity/yaw errors remain relevant and are
not declared solved.

## Concrete reference problem

The v343 tracking-reserve line's speed integral is 4.948605 s, but that is not
proof that its geometry, speed and actuator response are jointly feasible.
Its source OCP certificate does not transfer after geometric editing.

At the failed turn, recorded sampled curvature jumped between approximately
-.28, -.91, -.46, -.63, -.25, -.81 and -1.20 per metre. The active MPC's
steering-slew objective (weight 2000) is centered on changes in its
curvature-derived steering reference, not zero command change. This makes
reference jitter a direct source of demanded command changes.

Independent closed-loop geometry audit of 474 original nodes:

- Curvature-step p95 .4774 per metre.
- Curvature-derived geometric steering reference rate p95 10.11 rad/s,
  maximum 21.64 rad/s; 83 segments exceed source rack slew 3.2 rad/s.
- This rate is a reference diagnostic using `atan(.324*kappa)`, not a proof
  that the true high-speed inverse steering must equal a kinematic formula.
- The chatter is already in the original OCP-exported XY geometry. The
  tracking-reserve conversion did not create it; Menger curvature agrees
  with stored curvature to about 2.3e-7 RMSE.

## Implemented geometry correction

Extended the maintained `tools/make_tracking_clearance_target.py` with optional
periodic, arc-weighted XY regularization. It solves one sparse positive
definite system for both coordinates on the original nonuniform arc grid:
least displacement plus a second-derivative geometry penalty. It does not
fit tire forces, alter vehicle physics, change MPC weights or filter runtime
sensor data. No per-point clipping recreates curvature spikes. Original
geometry remains unchanged unless explicitly requested.

Candidate: `ipopt_v343_smooth_tracking_candidate.csv`. A .1 m regularization
length moves any point at most **.00194753 m**. Recomputed XY arc lengths,
heading, Menger curvature, acceleration and wall corridors; preserved all
474 speed samples. Steering-reference rate p95 becomes 1.19 rad/s, maximum
2.06 rad/s. Speed integral remains **4.94827949 s**.

The independent swept-footprint certificate retains the source-size body and
5 cm wall margin, with .03051 m additional certified clearance. This is not
a new vehicle-dynamics certificate. Seven unit tests cover the sparse solve
against an analytic circle eigenvalue, periodic seam, nonuniform spacing,
rigid transforms, noise reduction, zero/invalid smoothing and displacement
limits. The default no-smoothing path stays supported for reproduction.

## Live verification

`mpc_v343_smooth_geometry_20261028_01`: unchanged production MPC weights,
full candidate speed profile, eleven total laps requested. Full state,
commands, solver diagnostics and permitted-sensor odometry are recorded;
supervisor terminates on collision or controller fault. Completed eleven laps,
zero collisions: ten scored total **51.37400055 s**, mean 5.13740005 s,
population SD .05189 s, fastest 5.0548 s. Cross-track RMSE .06416 m versus
.0803 m for the prior ten-lap baseline. Twenty-eight prior-plan actions and
three candidate prefixes were used. This is a real one-run gain, not yet
repeatable: repeat `_02` failed after warmup on a corridor check.

Timing: gate median/p95 9.680/17.292 ms. One 20.423 ms cycle was deferred.
Two accepted commands passed the pre-publish gate but the publication wrapper
returned after 20 ms (maximum 21.489 ms). That does not isolate middleware
cost from scheduling/history/logging overhead. Do not claim strict full-cycle
maximum below 20 ms from the gate alone.

A coordinated ey/u/slew trial (8000/70/1000/20, other settings unchanged)
also failed a corridor check after one scored lap and is not promoted.
The candidate geometry remains separate from the installed production default.

## Corrected QP footprint formulation

The former QP froze the rectangle's support at nominal heading and bounded
only lateral position. Exact validation instead rotated the rectangle at the
candidate heading. A unit regression demonstrates an ey=.1 m, epsi=.1 rad
point that passed the former .3 m corridor box but violates the actual body
clearance. Added all eight left/right corner half-planes linearized in
(e_y,e_psi), including both active corners at heading zero. Removed the old
heading-frozen QP path; exact nonlinear validation and wall margins are unchanged.

Riccati-ADMM projects its existing first two split coordinates jointly onto
that polygon and broad physical boxes, in the correct per-coordinate penalty
metric. The interior and one-face cases exit early; only multi-face corners
enumerate vertices. No dynamic/optimizer states, separate constraint duals,
Riccati dimensions or heap allocations were added. Solver scaling transforms
the plane normals as well as states. Frozen-QP schema v4, C++/Python decoding,
source fingerprints and independent oracle constraints were updated together;
old schemas are rejected. Box-relaxation diagnostics never relax coupled rows.

Verification: 12/12 MPC CTest targets; four new geometric/projection/ADMM
tests, including 360 independently computed face projections and unequal-scale
converged parity. Nine Python fixture/oracle tests pass. Three saved full QPs
(rows 100/600/1500) solve independently with OSQP at strict tolerances.

Matched 2,307-state sequence replay: old frozen-heading core from commit
`f35effb9` with the same pre-edit indexed replay driver versus the new core,
identical inputs/profile/target. Full horizons accepted **2,287 -> 2,294**;
converged QPs **2,210 -> 2,223**. Recorded offline medians 7.598 -> 7.017 ms,
but isolated run timing and compiler-build differences prevent treating that
as a guaranteed speedup. The replay reconstructs memory; it does not restore
the original bag's unrecorded ADMM iterate. Results and three oracle witnesses:
`live_runs/mpc_v343_smooth_geometry_20261028_01/coupled_footprint_replay/`.

Supervised production trial: `mpc_v343_coupled_footprint_20261028_01`, smooth
target, unchanged production weights. Completed eleven laps, zero collisions:
ten scored **51.33699989 s**, mean **5.1337 s**, population SD .05604 s.
Cross-track RMSE .06403 m. Gate median/p95/max 9.818/17.432/19.370 ms.
626 optimal, 18 feasible, 1,627 line-search, 32 prior-plan actions and one
candidate prefix. This remains one successful run, not a repeatability claim.

## Publication-path instrumentation and cleanup

Commit `40aeb5e05` removes a redundant post-publication lateness warning and
its unused expected-stamp argument. Full diagnostics retain lateness. Added
separate middleware publication and command-history timings, exported with
finite/null regression tests. The subsequent pace trial measured middleware
median/p95/max 53/76/161.5 microseconds; history median/max .647/4.406 us.
Therefore it would be incorrect to attribute the earlier wrapper-return
outliers specifically to DDS. No physics or command timing was changed.

## Negative speed-profile experiment

A separate `ipopt_v343_smooth_pace103_candidate.csv` keeps the smoothed
geometry but requests 3% more speed, with acceleration recomputed consistently.
Its 4.80415 s integral is NOT a new dynamic feasibility certificate. Trial
`mpc_v343_coupled_pace103_20261028_01` completed warmup 5.8218 s and two
scored laps 5.1688/5.3068 s before a corridor failure. It did not improve actual
speed and is not promoted. Increasing the target alone is not the correction.

## Rejected SQP proposal selection — in progress

Inspection found R2 blindly replaced R1 whenever neither had a valid full
horizon. The replacement could discard a usable first-pass action without
comparing it. The production core now compares equal-duration exact 75 ms
prefixes when budget permits, keeps the lower-cost valid prefix, and restores
the corresponding ADMM memory. Ties or two invalid prefixes retain R1. Both
full horizons remain rejected; the controller independently validates any
published prefix. No clearance relaxation or braking fallback was added.
Added explicit diagnostics and report fields for both prefix costs, selected
pass, and validation time. Twelve CTest targets and report unit tests pass.
Initial trial `mpc_v343_compared_refinement_20261028_01` completed eleven laps,
zero collisions: ten scored **51.21600008 s**, mean 5.12160001 s. Cross-track
RMSE .06183 m. Of 22 rejected-pair comparisons, R1 was retained eleven times
and R2 chosen eleven times; additional comparison time was at most 47.5 us.
640 optimal, ten feasible, 1,623 line-search, 23 prior-plan and three prefix
actions. This first comparison used 25 ms rather than the intended 75 ms:
the in-horizon delay constant is zero because the initial state is already
advanced to the application epoch. Replaced that derivation with a shared
three-step constant used by both core comparison and controller validation;
the latter had continued enforcing 75 ms independently throughout. Do not
attribute the first trial to the final three-step comparison. One 59.993 ms
deadline miss was deferred; ordinary gate median/p95
9.816/17.170 ms. This is not a strict maximum-timing pass. Among line-search
acceptances: alpha=.5/.25/.125/.0625/0 counts 181/384/364/369/325. The heavy
backtracking motivates testing a tighter local SQP input box rather than
claiming convergence has been solved. Prefix-only selection does not promote
any invalid full horizon.

## Next work

1. Score full-horizon acceptance, prefix-chain lengths, speed/path error,
   timing and laps versus the recorded baseline. Do not equate offline
   curvature reduction with a racing gain.
2. If the smooth candidate improves racing, repeat it before promotion and
   tune response against its consistent geometry instead of noisy feedforward.
3. Investigate remaining nonlinear corridor failures. The heading-frozen
   footprint QP has been replaced, not left as an alternative runtime path.
4. Keep only measured useful runtime changes; retain negative evidence in
   run reports rather than leaving rejected experimental code enabled.

## SQP locality and coordinated response trials

`mpc_v343_local_step_20261028_02` reduced only local SQP input radii from
.08/.10 to .04/.05 rad/normalized throttle. Physical actuator bounds and
weights stayed unchanged. Eleven laps, zero collisions: ten scored
**51.13700008 s**, best scored 5.0478 s. Cross-track RMSE .06062 m;
685 optimal, 22 feasible, 1,581 line-search, eight prior-plan and one prefix
action; one deferred deadline miss. Gate median/p95/max
9.816/17.190/20.302 ms. Thus neither the timing maximum nor sub-50 s goal is
claimed solved. Like the preceding trial, core R1/R2 proposal comparison was
still 25 ms; the controller's final publication gate remained 75 ms.

The following trial, `mpc_v343_tracking_response_20261028_01`, uses the
corrected shared 75 ms comparison (saved as `24d7a1d5`), the smaller local
radii, and coordinated ey/epsi/u/overspeed/steer-slew/throttle-slew weights
8000/1000/70/40/1500/25. Completed eleven laps without collision, but scored
**51.38700056 s**, worse than the simpler locality trial. Cross-track RMSE
.06472 m. Three deferred misses, gate max 24.798 ms. Not promoted; recorded
overlay is retained with the run, not as an enabled production alternative.

The single 59.993 ms outlier in the earlier refinement trial was localized to
the OpenMP Jacobian region: 59.171 ms wall versus 4.714 ms summed worker CPU;
all worker stage totals were below .663 ms. No ADMM ran in that cycle. This
is a scheduling/team-launch or barrier stall, not proof that plant arithmetic
took 59 ms or that middleware publication was responsible.

`mpc_v343_local_step75_20261028_01` tests the smaller radii with unchanged
production weights and the final 75 ms comparison: eleven laps, no collision,
ten scored **51.03000021 s**, best scored 5.0338 s. Cross-track RMSE .06075 m.
662 optimal, ten feasible, 1,612 line-search, five prior-plan and one prefix
action. One deferred miss, max 20.024 ms; median/p95 9.938/17.233 ms.
Repeated ten-lap completions now exist, but not yet ten laps under 50 s.

The production QP cost directional derivative has also been checked against
independent central differences of recursively propagated exact nonlinear
objectives, on straight and varying-curvature ellipse references, steering
and throttle perturbations at stages 0/8/20. Includes source slip loss and
all production stage weights. Twelve comparisons agree within .5% at 1e-4
input perturbations; larger finite steps exposed curvature of the recursive
response, not a demonstrated sign error. This does not certify arbitrary
high-speed states; it prevents claiming an unmeasured gradient bug as the
reason for every line-search backtrack.

`mpc_v343_local_step_pace103_20261028_01` retries the previously failed 3%
higher requested speed with the smaller local steps and final prefix logic.
The maintained target generator now has explicit speed scaling and tests
that acceleration scales quadratically; no dynamic certificate transfers.
Odom and AMCL equations
remain unchanged; no new evidence yet justifies altering their physics to
compensate for controller errors. Runtime-topic policy still passes. Custom
40 Hz bridge competition approval remains unresolved, not certified here.
