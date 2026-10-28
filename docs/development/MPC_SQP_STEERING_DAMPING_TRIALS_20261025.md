# MPC second-SQP and steering-damping trials — 2026-10-25

## Question

Does a selectively-triggered second SQP pass, combined with a lower steering-
slew penalty, improve repeatable tracking of the 5.367 s nominal IPOPT line?
The success criterion is real practice-track lap performance and consistency
after the warmup lap—not a lower optimizer objective or more full-step QP
acceptances by themselves.

## Fixed reference and baseline

All valid comparisons below used
`f1tenth_planning/trajectories/autodrive_practice_20260924_b/ipopt_v1_baseline_raceline.csv`
on the practice simulator. SHA-256:
`35bed562a8b0611c48cbf2df8520aff091a3f212adf0eacd5f59e6b9f527fb73`.
Its 5.367 s value is the optimizer's nominal integral, not a measured runtime
lap.

The six-lap baseline capture is
`live_runs/mpc_ipoptv1_sqp2_practice_20261025_01/run` (not the earlier v300
capture). It had no collision. Lap 1 was warmup; laps 2–6 were
`6.3597, 6.3957, 6.1597, 6.0827, 5.6608 s`: mean `6.132 s`, population
standard deviation `0.263 s`. It is a baseline, not proof of consistent or
sub-5-second performance. Tracking RMSE was `0.3289 m` cross-track,
`0.1863 rad` heading, and `0.8677 m/s` speed. Publication-gate p50/p95/max
was `5.03/14.06/20.288 ms` (one sample at/over 20 ms; no callback exceeded
25 ms).

## Offline screen

Saved-state bag replay was used to screen damping and refinement-policy
changes before live trials. Reducing steering-slew weight alone from 2000 to
1000 changed tight-turn mean absolute steering only from about `0.253` to
`0.257 rad`; throttle changed `0.135` to `0.140`, with essentially unchanged
line-search alpha counts and corridor slack. A 500 weight gave about
`0.264 rad` mean absolute steering but reduced fifth-percentile corridor
slack from `0.115` to `0.087 m`. The expected large steering response was not
seen; lower damping alone was not a useful fix.

Second-pass screens also exposed a timing/acceptance tradeoff. Refining every
accepted damped result (alpha < 1) exceeded the 20 ms max gate in replay
(max `25.09 ms`). Restricting refinement to alpha <= 0.25 stayed under the
gate in replay (max `17.86 ms`) and raised the full-step acceptance count,
but slightly reduced fifth-percentile corridor slack (`0.052 m`). Combining
that trigger with steering-slew 1500 reached `23.04 ms` max; with 1000 it
stayed at `18.61 ms` max. These are replay screens, not closed-loop success.

## Live trials

### Selective second pass + steering-slew 1000

Capture: `live_runs/mpc_ipoptv1_selective_refine_slew1000_20261025_01/run`.
The lower weight was applied only through a trial YAML overlay; production
`mpc_competition.yaml` was not changed. Six laps completed without collision.
Warmup lap was `7.238 s`; laps 2–6 were
`8.9860, 6.3537, 6.2977, 6.6377, 6.3117 s`: mean `6.917 s`, population
standard deviation `1.042 s`. Against the baseline, the scored mean regressed
by `0.786 s` and variability increased substantially. Cross-track, heading,
and speed RMSE were `0.5070 m`, `0.2210 rad`, and `1.0668 m/s`, all worse than
baseline. Gate p50/p95/max was `5.372/15.890/16.932 ms`, with no cycle at/over
20 ms. This timing success does not offset the worse driving result.

### Selective second pass + original steering-slew 2000

Capture: `live_runs/mpc_ipoptv1_selective_refine_slew2000_20261025_01/run`.
This isolated the refinement trigger from the lower-weight change. The run
was stopped by the supervisor after an unpublished nonlinear corridor
rejection at `38.511 s` (`lap_count=5`); it completed no collision and did not
finish the requested six-lap trial. The four scored laps before the fault
were `6.0797, 6.2077, 7.5786, 6.2827 s`: mean `6.537 s`, population standard
deviation `0.606 s`. Cross-track, heading, and speed RMSE were
`0.4158 m`, `0.2058 rad`, and `0.9432 m/s`, worse than baseline. Gate
p50/p95/max was `5.178/15.798/16.514 ms`, with no cycle at/over 20 ms.

At the fault the state was already far outside desired tracking
(`e_y=1.242 m`, `e_psi=0.149 rad`, `u=2.743 m/s`, `r=-2.596 rad/s`), near
`-0.812 1/m` curvature. The exact candidate rollout exceeded the corridor
threshold at stage 18 by only about `0.8 mm` beyond the configured 5 mm
tolerance. The second pass had consumed its solve budget; diagnostics show
`line_search_skipped_budget=true`, no accepted second-pass candidate, and no
new command published. This is evidence that the refinement policy can
consume time needed for nonlinear backtracking, while failing to recover the
large tracking error. It does not establish that the small corridor
overrun alone caused the preceding loss of tracking.

The decoded bag's 750 ms fixed-command source-plant replay passed its 5 cm
full-path gate for `1515/1515` windows (maximum error `4.20 cm`). This is a
separate open-loop plant check on the recorded commands; it does not mean the
closed-loop MPC followed the reference accurately. In the closed-loop
replanning horizon comparison, 750 ms XY RMSE was `0.385 m` and p95 absolute
error `0.883 m`; those future commands were replanned and are not a fixed
input prediction experiment.

## Code/config disposition

Neither live variant improved the lap outcome. The selective refinement
trigger was reverted to the previous policy: refine only after a rejected
full-step corridor candidate or an alpha-zero line-search result. The source
was rebuilt (`mpc_bag_solver_replay` target) and `git diff --check` passed.
Production steering-slew weight remains `2000`; the lower weight exists only
in the first trial's run-local overlay. No measured evidence supports
promoting either change. The default raceline, map, plant, and production
weights were not altered by these trials.

## Next diagnostic direction

Do not lower steering damping further by guesswork or keep the experimental
second pass enabled. The logs identify an important failure interaction:
refinement may spend the time reserve needed for line search, while large
cross-track/heading errors accumulate before the terminal corridor rejection.
The next change should target the source of those tracking errors (especially
the high-curvature negative-turn segment), preserve a reserved exact nonlinear
backtracking budget, and be screened on saved terminal states before another
live multi-lap trial. Any new setting must be compared against the exact
baseline reference and reported using post-warm lap mean and variance.

## Parallel IPOPT job

The long IPOPT job was deliberately left running, as requested. PID `257154`
was still active at the time this entry was written, using about 11.2 GB RSS
and one CPU core. Its output directory
`SDU_Apex_IPOPT_PhysX_MinTime_v2/ipopt_user_sketch_coarse_20261025_resume03`
contained only `provenance.json` and `optimizer_reference_track_mesh.csv` at
the last check; no candidate/checkpoint was present yet. Continue monitoring
that process and evaluate any newly completed candidate before calling it a
faster usable raceline.
