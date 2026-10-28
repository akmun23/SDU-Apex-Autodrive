# MPC throttle-slew and deadline-recovery update — 2026-10-24

## Scope and lap accounting

The practice MPC run targets six completed simulator laps. **Lap 1 is warmup;
only laps 2–6 are scored.** Lap time comes from the simulator's completed-lap
topic, not the optimizer's estimated time.

The production MPC configuration now uses
`weight_throttle_slew: 100.0`, promoted after the separate throttle-slew
screen and its full-speed practice run. The current update addresses an
observed publication-deadline failure that ended a production-config attempt
after its warmup lap.

## Failure evidence that motivated the change

In `live_runs/mpc_direct_input_practice_20261024_slew100_prod1`, an MPC
candidate exceeded the 20 ms end-to-end publication gate (27.288 ms). The
controller latched a fault and published no replacement output, even though a
previously accepted plan existed. The simulator kept applying the last command
until the collision watcher stopped it. This is a concrete missed-cycle
failure, not evidence that a lower-level command should be substituted.

## Implementation

- Added `MPC_RTI_CYCLE_DEFERRED_DEADLINE` to identify an unpublished cycle that
  missed its end-to-end deadline.
- On one isolated miss, the controller leaves the actuator command untouched
  for that tick. At the next sensor update it revalidates the next action from
  the prior certified plan against the current state and delayed-command
  rollout, then publishes that action if it still passes.
- An unpublished candidate does not replace the prior certified plan or seed
  the next nominal sequence. If a prior action is unavailable or fails
  revalidation for a non-timing reason, the solver is seeded from the shifted
  prior sequence when possible; otherwise it resets to a clean nominal and
  solves again.
- A second consecutive deadline miss still latches a controller fault. No
  active-brake or zero-throttle racing fallback was added.
- Deadline misses are counted as deferred, not accepted MPC outputs. The
  supervised-run watcher recognizes the deferred status and continues
  monitoring; it still stops on a latched fault or collision.
- The existing `weight_throttle_slew: 100.0` setting and practice trajectory
  geometry were retained. No physics or track assets were changed.

## Build and supervised practice run

The mounted production runtime build completed all five required packages:
`sdu_apex_msgs`, `f1tenth_planning`, `f1tenth_localization`, `f1tenth_mpc`,
and `sdu_apex_autodrive`. Colcon emitted the existing nested vendored-NumPy
package-discovery warnings about missing Cython; they did not prevent the
build. The runtime topic-policy check passed.

The simulator ran through the established practice batch-mode launcher. MPC,
bag recorder, and collision/lap watcher were ready before the simulator
started. The watcher target was lap count 6, which means lap 1 warmup plus five
scored laps.

Capture and full evaluator output:

- [`run/`](../../live_runs/mpc_direct_input_practice_20261024_slew100_recovery1/run/)
- [`analysis_current_code.txt`](../../live_runs/mpc_direct_input_practice_20261024_slew100_recovery1/analysis_current_code.txt)

Outcome:

| Item | Result |
|---|---:|
| Completed laps | 6 total |
| Lap 1 warmup | 6.886 s |
| Scored laps 2–6 | 6.044, 6.138, 6.101, 6.121, 6.180 s |
| Scored-lap mean | **6.117 s** |
| Scored-lap best | **6.044 s** |
| Scored-lap range | 6.044–6.180 s |
| Collisions | 0 |
| Publication-gate p50 / p95 / max | 4.443 / 8.588 / 15.576 ms |
| Cycles at or over 20 ms | 0 / 1518 |
| Solver-reported p50 / p95 / max | 4.271 / 8.189 / 15.072 ms |
| MPC status counts | 1485 optimal, 15 feasible, 18 candidate-prefix; 5 startup waits |

Production odometry on this run measured forward-speed RMSE 0.0085 m/s and
lateral-speed RMSE 0.0048 m/s against simulator truth. Relative XY error ended
at 4.3 cm and reached 5.4 cm maximum. Closed-loop MPC diagnostics show the
750 ms horizon comparison at 12.66 cm XY RMSE and 66.51 cm maximum; this is a
replanning closed-loop diagnostic, not a fixed-command plant-rollout accuracy
claim.

## What this run does and does not establish

This is a second full-speed, collision-free run using the throttle-slew-100
configuration. The prior valid run in
`live_runs/mpc_direct_input_practice_20261024_throttle_slew100_2/` had five
scored laps averaging 6.153 s and no collisions. This run averaged 6.117 s.
That supports repeatable operation over these two runs and is a modest time
improvement; it is not a statistically established gain or a sub-5-second
result.

No publication deadline was missed in the recovery run. Therefore the normal
production path, lap accounting, collision supervision, bag capture, and
under-20-ms timing were exercised, but the new deferred-deadline recovery path
was **not triggered online**. The prior 27.288 ms collision run remains the
direct evidence for the bug; the fix for that exact missed-cycle path is
compile-verified but still awaits a naturally occurring deadline miss in a
supervised run. No artificial timing injection was used.

The full-horizon rollout still diverges materially by 750 ms, and the result
does not demonstrate the requested sub-5-second racing target. Those remain
open items for continued model and controller work.

## Next action

Continue repeated practice trials with lap 1 excluded from all score summaries.
If a deadline event occurs, confirm the bag shows
`deferred_deadline_recovery` followed by an accepted revalidated prior action;
stop on a second consecutive miss, latched MPC fault, or collision. Keep the
existing trajectory and physics unchanged during that check.

## 2026-10-24 — positive throttle floor and rejected-turn prefix selection

The minimum MPC drive command is now `q=0.001` instead of `0.01`. The bridge
serializes to three decimal places, so `0.001` remains `0.001`; it is the
smallest positive command that is unambiguously distinct from Unity's exact
`q=0` full-brake branch. At the practice motor setting (`428 N m` total),
`q=0.001` is `0.428 N m` total drive torque (`0.107 N m` per wheel in the
four-wheel drive path). It is a positive drive command, not a braking command.
No q=0 decision was added to MPC; q=0 remains in the plant to reproduce Unity
braking and is allowed only in the explicit development path follower.

The saved v323 practice event at diagnostic index 108 was replayed with the
rebuilt production source-plant evaluator. Both the three-step candidate and
retained-plan prefixes pass exact source-plant and corridor validation, but
the candidate costs `24374.69` versus `10510.14` for the retained plan over the
same 75 ms. The candidate's first command reverses steering from the current
approximately `-0.322 rad` to `+0.112 rad` and requests `q=0.416`; therefore
the `0.01` throttle floor did not cause this event. The candidate was
iteration-limited with large solver residuals, and the existing policy had
published it after only checking short-horizon feasibility. Production now
enables quality-aware selection on rejected solves: when both equal-length
prefixes pass exact validation, it selects the lower runtime objective. The
retained action wins for this event. If neither branch validates, the
controller reports a fault instead of publishing the unchecked candidate.

The next-epoch offline MPC replay still fails to converge from either branch
(both hit 100 iterations and fail nonlinear validation at stage 1). The
quality selector prevents this demonstrated worse prefix; it does not claim
to fix the underlying high-curvature convergence or prove that the rollout
matches Unity there. The selector and new throttle floor compile in the local
ROS build; the saved practice bag is offline evidence only. A supervised
practice run with the same target line is required to validate the behavior
online.

## 2026-10-24 follow-up — minimum drive command and steering/yaw residual

The MPC command floor is `q=0.001`; exact `q=0` remains excluded from its
decision bounds. The bridge rounds to three decimal places, so `0.001` is the
smallest positive wire value that cannot round into Unity's separate full
brake branch. A smaller requested value is not a useful MPC setting: values
that serialize to `0.000` select full braking. `q=0.001` is only a very small
positive drive torque (0.428 N m total, 0.107 N m per wheel for the active
four-wheel drive configuration), not negative torque or a brake command.

One boundary issue in the offline search tool was corrected: its minimum was
stored as a C `float`, whose rounded representation is slightly greater than
the decimal CSV value `0.001`; an exact `0.001` seed was therefore rejected.
The tool now uses the exact decimal bound while continuing to accept the
production float representation. Focused Python verification passes: 26 tests.

The existing full-speed practice capture
[`recovery2`](../../live_runs/mpc_direct_input_practice_20261024_slew100_recovery2/run/)
is useful for diagnosis, but does **not** test live `q=0.001`: its lowest
published MPC throttle command was `0.010`. It completed six laps (lap 1
warmup, scored laps 2–6 mean `6.254 s`) with zero collisions. Replaying its
recorded commands through the current source plant gives 1,519 valid 750 ms
windows; 1,516 remain within 5 cm at every 25 ms sample. Maximum path-error
RMSE/p95/max are `1.27/2.34/10.72 cm`. The three failures are concentrated in
one high-steering transition around `2.5 m/s`; they are not general rollout
divergence across the lap.

At the worst transition, commanded throttle is positive (`q=0.111`), and
steering feedback agrees with the replay target to about `0.0001 rad`. Yet the
one-step yaw-rate prediction is `-1.9398 rad/s` versus Unity's `-2.1124 rad/s`
(error `+0.1726 rad/s`); the predicted/observed yaw acceleration signs also
disagree (`+1.7` versus `-4.4 rad/s^2`). Using next steering feedback as an
offline oracle leaves the same three 750 ms windows over 5 cm (worst
`10.86 cm`). This points to a localized yaw/tire-contact response mismatch,
not a need for the hard-brake mode and not a general one-packet steering
association fix. Do not tune the model from these three correlated windows as
if they were independent samples.

Two additional offline isolations on this same bag sharpen that diagnosis:
replacing logged throttle with the next recorded throttle feedback leaves the
same three windows over 5 cm (worst `10.79 cm`), while re-anchoring predicted
yaw rate to Unity truth at every 25 ms epoch eliminates all three (`0/1519`;
future truth, diagnostic only). Re-anchoring only `u` leaves two failures;
re-anchoring only `v` also eliminates all three. The `v`/yaw-rate coupling
means this does not uniquely assign the defect to either state. It does show
that the threshold failures arise from accumulated lateral/yaw motion error,
not throttle association or pose integration alone. In the one-step
partition, yaw-rate RMSE is `0.0047 rad/s` below `0.10 rad` steering and
`0.0252 rad/s` at or above it; the latter has p95 `0.0624` and max
`0.1730 rad/s`. The highest-error turn also has a wheel/body speed mismatch
proxy of about `2.17 m/s`. Steering-sign counts are imbalanced (1,205 negative,
334 positive), so that comparison is descriptive rather than proof of a
left/right vehicle asymmetry.
Full diagnostics: [throttle-feedback oracle](../../live_runs/mpc_direct_input_practice_20261024_slew100_recovery2/analysis_source_rollout_throttle_oracle.txt)
and [state re-anchoring](../../live_runs/mpc_direct_input_practice_20261024_slew100_recovery2/analysis_source_rollout_state_reanchoring.txt).

The saved rejected-solve event is distinct: its iteration-limited candidate
reverses steering and requests `q=0.416`; exact source-plant comparison scores
it worse than the retained-plan prefix (`24374.69` versus `10510.14`). The
quality-aware prefix selector now keeps the lower-cost exact-valid action for
that reproduced state. This prevents that demonstrated bad publication, but
does not repair the remaining yaw-response residual above or establish a new
live `.001` run. The next model work should isolate the high-steering,
low-speed yaw-force discrepancy using the existing practice capture and
source equations; keep the MPC brake mode disabled.

## 2026-10-25 — v300 rejected-turn recovery failure and correction

The current production MPC was run on the practice map with the V300 trajectory
under a six-lap watcher (warmup plus five scored laps). This diagnostic attempt
ended after 5.53 s with zero completed laps and one collision. It is a failed
trial, not evidence of improved lap completion. The full bag is
[`run/`](../../live_runs/mpc_v300_recovery_prefix_20261025_01/run/).

The new diagnostics isolated a concrete bad branch at progress `40.787 m`:

| Quantity | Iteration-limited candidate | Retained exact-valid plan |
|---|---:|---:|
| ADMM iterations / status | 100 / iteration cap | — |
| Primal residual | 0.233 (over 2× 0.02 tolerance) | — |
| 75 ms exact source-plant objective | 7,121 | 7,733 |
| Minimum corridor slack | 2.62 cm | 1.64 cm |
| Requested first steering | **-0.524 rad** | about **-0.050 rad** |

Both prefixes passed their 75 ms corridor validation. The selector nevertheless
published the candidate because it had a lower short-prefix objective and a
small clearance gain, despite the grossly unacceptable ADMM residual. At the
next cycle neither candidate nor retained plan passed exact validation
(candidate failure at stage 1); the simulation then collided. Thus this was
not simply an unavailable prior plan: a poor-quality iterate displaced a
valid continuation immediately before the failure.

The selector now receives solver-convergence and residual-quality as separate
facts. A nonconverged candidate whose residuals exceed `2 × tolerance` cannot
replace an exact-valid prior based on objective or clearance alone. A
residual-qualified iteration-limited candidate still needs the configured
objective gain or meaningful clearance override; a converged candidate keeps
the normal objective/clearance comparison. This preserves valid prior
behavior instead of adding a fixed braking command or changing the trajectory.
The captured terminal decision is now a regression test.

Verification so far: MPC controller and prefix-selection test targets compile;
all 10 MPC CTest cases pass. **This correction has not yet passed a new live
practice run.** The next test is the same V300 practice setup, retaining full
diagnostics and the collision/fault watcher. The independent IPOPT run on the
user sketch remains running in the background; it is intentionally unaffected
by this MPC change.
