# IROS equal-time search and MPC correction — 2026-10-30

## Measured starting point

- Fastest independently checked source witness: `7a9e6afbf50d666f`, 7.125 s.
- Prior equal-time archive attempted eight solves and accepted no alternatives.
- Last 7.100 s solve reached 1,200 iterations, residual 5.39e-5; its 1 ms
  footprint check missed the requested buffer by 0.339 mm at worst.
- Clearance slack is **additional to the 5 cm body footprint buffer**. A
  -0.339 mm slack means approximately 49.661 mm body-to-wall clearance, not
  a wall collision.
- MPC a66 completed six scored laps; the repeated profile a69 failed during
  warmup. a70's 0.30 m soft selector reserve also failed, at 122.715 m after
  warmup. a70 publication timing p50/p95/max: 9.99/16.93/19.28 ms.

## Identified implementation problems

1. `track_optimizer.publish_best` returned before exporting every equal-time
   witness. The Unity validator could not see these alternatives.
2. The fixed-duration OCP minimized distance from the old state and command
   sequence. It restored a seed but did not directly search for better
   clearance or smoother equal-time controls.
3. Checkpoints and stagnation detection ranked constraint residual alone.
   A feasible quality improvement could be discarded or stopped.
4. Equal-time refinement discarded candidates with missing 1 ms clearance
   cuts instead of repairing the same primal as the shortening campaign does.
5. Periodic command regularization omitted the last-to-first input change.
6. After reboot, the validation service watched the old conditioned workspace,
   while the dashboard watched the multi-incumbent workspace.
7. MPC's QP uses frozen tracking targets, but its nonlinear merit resamples
   those targets at candidate progress. Native steering-target jumps can
   change the objective the QP thought it was minimizing. The prior
   first-gradient-only trial was unsuccessful; do not repeat it.

## Implemented

- Equal-time immutable exports and quality ranking by clearance, steering
  increments and throttle increments. Exported alternatives remain available
  to independent Unity trials even when the current best pointer is retained.
- New fixed-duration objectives: maximize minimum footprint reserve, or
  minimize periodic input changes with steering prioritized. The old trajectory
  is only a weak numerical prior in these quality searches.
- Feasibility-first, then objective-based checkpoints and stagnation detection.
  Solver feasibility threshold now matches independent certification at 1e-5.
- Equal-time sub-packet wall-cut restoration and immediate accepted-candidate
  export. Each accepted alternative gets its own shorter-duration branch.
- Explicit search-only migration of saved campaigns, retaining map/source/seed
  contracts. New objective archives use a separate versioned directory.
- Single validation service now watches the same workspace as the dashboard.
  Updated search runs as `sdu-iros-quality-search-20261030.service`; the
  validator can freeze this verified optimizer service during builds/trials.
- MPC QP and nonlinear merit now use the same tracking references and nominal
  speed-weight branch for each SQP pass. Tracking schedules refresh with the
  next pass/cycle. Physical propagation and hard wall bounds still follow
  candidate progress. This changes the control objective, not the plant.
- Pass-to-pass and candidate/prior comparisons use the common first-pass
  schedule. Re-scoring uses already propagated plans and adds no plant steps.

## Verification and remaining work

- Targeted optimizer regressions: **65 passed**, including equal-time export,
  objective checkpoint ordering, periodic smoothing and sub-packet cut repair.
- MPC package test results: **172 reported tests, zero failures**. Condensed
  objective-gradient checks now use spatially varying source fields with the
  same frozen pass schedule. A separate test confirms that frozen cost targets
  cannot widen candidate-progress wall bounds.
- a71 queued on the unchanged 7.125 s witness with the a66 weight profile and
  both soft reserves restored to 0.15 m. It tests the objective correction;
  no new weights or altered raceline are included in this comparison.
- Replay difficult recorded states, then run serialized full-speed Unity trials.
  Completion requires repeated ten-scored-lap runs; no new racing claim yet.

## Lateral-target shift investigation and controlled verification

The practice 4.800 s source witness requested a 5 cm full-body buffer. Its
saved 1 ms certificate reports 0.0358 mm extra slack: 5.00358 cm minimum
conservative map clearance. This is planned clearance, not measured execution
clearance. Practice already defaulted to a zero soft lateral-reference reserve.

The IROS test overlay instead sets `reference_clearance_reserve_m=0.15`.
`reference_at_progress` shifts only `e_y`, leaving source heading, yaw rate,
lateral velocity and steering references unchanged. Recomputing this exact
reference rule on a72's 7,125 reference points gives a maximum 0.141926 m
target displacement and 1,289 points displaced by more than 1 cm. The source
witness itself has no violation of the MPC corridor rule, including its
0.10 m preview: minimum zero-error slack before tolerance is 0.008074 m.
An additional lateral-target shift is therefore not needed to make this
native witness satisfy that corridor rule.

a71 failed without completing a scored lap. a72 also failed; its 103 accepted
line-search cycles include 43 alpha-zero nominal retentions. Thirteen
consecutive alpha-zero cycles immediately precede the prior-plan chain around
59.58--61.20 m. Converged QPs did not produce accepted corrective steps.
Recorded-command 750 ms a72 replay is nevertheless accurate: 209 overlapping
windows, 0.005220 m endpoint RMSE, 0.012463 m maximum path error,
0.007745 m/s forward-speed RMSE, and 0.007756 rad/s yaw-rate RMSE. This is
executed-command replay, not execution of each unselected MPC plan, and does
not establish independent odometry accuracy.

User requested a controlled removal test, practice before competition:

- No weight sweep, physics change, raceline replacement or hard-margin change.
- Toggle only `reference_clearance_reserve_m`; the separate prefix-selection
  reserve remains unchanged to avoid conflating two interventions.
- Practice uses the production fastline profile and existing 4.800 s witness.
  The launcher now accepts and seals an explicit MPC overlay, so paired
  zero/0.15 m trials cannot silently discard the requested setting.
- First current-code zero-shift practice trial:
  `live_runs/practice_target_zero_20261030_01`, one warmup plus ten scored laps.
  Results below. Every simulator trial is supervised and stops on collision.
- Competition trial is conditional on the practice verification; no claim
  that removal is sufficient or that every failure has this cause.

### Controlled practice results: removal not yet verified

All three trials used the same freshly compiled current runtime, 4.800 s
reference (SHA256 `03433092578e36f2dc143b3272c6bce507bef10675c1a2b96989c7757653e03b`),
practice weights, localization parameters, 5 cm hard corridor and 20 ms gate.
Only the requested lateral-reference reserve changed between arms. These
trials include the newer objective/steering-rate implementation, so they are
not an unchanged reproduction of the October 29 successful runtime.

| Run under `live_runs/` | Reference reserve | Official laps completed | Outcome |
| --- | ---: | ---: | --- |
| `practice_target_zero_20261030_01` | 0 m | 0 | 40.754 ms publication-gate overrun, stale-state event, then candidate/prior rejection. |
| `practice_target_shift15_20261030_01` | 0.15 m | 0 | Timing failure: final Jacobian phase 47.145 ms; no accepted plan remained. |
| `practice_target_zero_20261030_02` | 0 m | 0 | No deadline overrun (max 18.602 ms); retained-plan/prefix chain followed by nonlinear corridor rejection. |

There were no reported collisions. The first zero arm's Jacobian phase took
38.076 ms wall time despite the longest worker's summed stage wall time being
only 2.125 ms. The shifted arm shows the same distinction: 47.145 ms versus
2.117 ms. This identifies a dispatch/barrier/scheduling gap to investigate,
not proof that tire equations or ADMM arithmetic took that entire duration.

The zero repeat issued 17 retained-plan actions and eight candidate prefixes.
Terminal reported heading error was 1.455 rad and lateral body velocity
3.468 m/s. Path projection then switched branches (progress 31.00 to 24.37 m),
so the terminal -2.323 m Frenet error must not be read as smooth accumulated
cross-track drift. Exact-stamp map-pose/truth comparison peaked at 5.84 cm.
Its recorded-command 750 ms replay scored 114 overlapping windows: endpoint
RMSE 0.008604 m and maximum path error 0.020200 m. The first zero arm's replay
had larger errors (4.30 cm endpoint RMSE, 26.51 cm maximum) around its timing/
state-epoch disruption; it must not be silently pooled with the repeat.

Conclusion: the structural target inconsistency remains real, but these runs
do not establish that the reserve causes every failure or that removing it
restores lap completion. A higher-clearance path is not yet justified as the
sole fix. No competition trial launched because the user's practice-pass gate
was not met. No weight sweep, margin relaxation or physics change was made.
The existing IROS validation overlay remains untouched pending that gate.

Regression coverage now checks that disabling the target shift restores
`e_y=0` and preserves native heading/lateral/yaw/steering targets, identical
corridor bounds and the hard 5 cm margin. MPC package results: 172 reported
tests, zero failures. All simulator processes from these trials are stopped;
bags, sealed overlays, full diagnostics and reports are retained per run.
