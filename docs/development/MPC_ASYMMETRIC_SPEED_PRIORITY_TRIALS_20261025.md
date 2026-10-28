# MPC asymmetric speed-priority trials — 2026-10-25

## Question

Test the bundled hypothesis that underspeed should cost more than modest
overspeed, while throttle slew remains free. These are practice-track
closed-loop trials on the same `ipopt_v1_baseline_raceline.csv`; all weight
changes were run-local overlays. No candidate was promoted to the production
profile.

The development reset was held across Unity startup for the valid runs. The
first `_01` attempt for the U60 bundle is invalid: its bag has no MPC
diagnostics or commands and starts with a nonzero collision counter. It is
retained only as a startup/reset diagnostic, not scored as a controller run.

## Results

| Run | Bundled weights `(e_y, e_psi, U, O, curvature gain)` | Throttle slew / effort costs | Recorded laps (s) | End condition | Cross-track / heading / speed RMSE |
|---|---|---|---|---|---|
| `mpc_iterations150_repeat_20261025_01` reference | `(4000, 1500, 7.5, 60, 1)` | `100 / 0.1` | warmup `6.639`; scored `5.724, 5.800, 5.651, 5.941, 5.808` | 5 scored laps | `0.171 m / 0.112 rad / 0.769 m/s` |
| `mpc_asymmetric_speed_priority_u60_o15_g5_noslew_20261025_02` | `(4000, 1500, 60, 15, 5)` | `0 / 0` | `7.484, 6.344` | collision after an unpublished rejection at 15.91 s | `0.445 m / 0.230 rad / 1.502 m/s` |
| `mpc_asymmetric_speed_priority_u100_o10_g4_trackx4_noslew_20261025_01` | `(16000, 6000, 100, 10, 4)` | `0 / 0` | `7.474, 7.129, 7.285` | solver rejection at 27.36 s; no collision before stop | `0.426 m / 0.221 rad / 1.293 m/s` |
| `mpc_asymmetric_speed_priority_u30_o10_g2_noslew_20261025_01` | `(4000, 1500, 30, 10, 2)` | `0 / 0` | `7.062` | corridor rejection at 9.12 s; collision followed | `0.356 m / 0.196 rad / 1.560 m/s` |
| `mpc_straight_speed_priority_u30_o10_g10_noslew_20261025_01` | `(4000, 1500, 30, 10, 10)` | `0 / 0` | `6.913, 6.086` | corridor rejection at 18.28 s; watcher stopped before collision | `0.302 m / 0.174 rad / 1.255 m/s` |

The reference’s five scored laps average **5.785 s** (population SD **0.097 s**).
None of the asymmetric candidates completed the six-lap test or improved that
reference. Their path and speed errors also regressed materially. The two
U60/U100 runs each had one publication-gate sample above 20 ms (maxima 21.24
and 21.76 ms); the U30 runs stayed below 20 ms (maxima 17.59 and 17.93 ms).

## What the data establishes

- Setting both throttle penalties to zero does remove the soft slew/effort
  costs: in the U60 capture, issued throttle reached `1.0` and changed by as
  much as `0.862` in one cycle. Thus throttle slew was not being held to a
  narrow rate by the objective.
- That freedom did **not** increase mean throttle use: the U60 capture averaged
  `0.208`, versus `0.206` in the reference. It did permit larger steps, but
  speed RMSE rose from `0.769` to `1.502 m/s` and path RMSE from `0.171` to
  `0.445 m`.
- The recurring loss is at the tight negative-curvature section, around
  `k=-0.85` to `-1.12 1/m`. Cross-track error grows beyond `0.7 m` and reaches
  `1.13 m`; the final U30/gain10 sample is `u=2.70 m/s` versus `3.03 m/s`
  requested. That last failure is not explained by overspeed alone. The U30/
  gain2 run reaches the same section while about `0.62 m/s` over target, so
  speed/turn interaction remains a concern.
- Curvature-scheduling overspeed more strongly did not make the line reliable:
  even gain `10` ended in a nonlinear corridor rejection. These tests do not
  establish that globally tolerating overspeed is safe or faster.

## Decision / next engineering step

Keep the proven production weights unchanged. Do not keep sweeping scalar
speed weights: the common failure is lateral/path divergence in the hairpin,
not throttle-rate saturation. Next diagnose the first divergence through that
turn—requested versus applied steering, yaw-rate response, lateral velocity,
and the exact-corridor rollout versus measured state—then change the demonstrated
steering/rollout cause. Preserve the existing baseline as the controlled
comparison. Keep the 20 ms publication gate; the U60/U100 timing outliers also
need attention before any candidate can be promoted.

## Artifacts

- Per-run sealed manifests, bags, and exported reports are under the matching
  `live_runs/<run-id>/` directories listed above.
- The candidate weights remain only in their run-local `mpc_overlay.yaml`
  files. `f1tenth_mpc/config/mpc_competition.yaml` was not changed by these
  experiments.

## Continued work: isolated-CPU repeat and terminal yaw diagnosis

The long v324 IPOPT run was kept active while the MPC was tested. To prevent
the optimizer from starving the real-time stack, PID `689833` was moved to
nice level 10 and pinned to logical CPUs `0-3`. It remains running. The first
time stage finished at its 1,200-iteration cap with an objective of `5.2074 s`,
maximum non-trust constraint violation `2.66e-5`, and dual infeasibility
`0.0689`; the optimizer continued into trust factor 4. This is a feasible
intermediate, not a converged raceline. No candidate/checkpoint was present at
the last file check.

With IPOPT isolated, a production-weight v323 control run completed six laps
without collision. Excluding warmup, lap times were `5.940, 5.957, 5.686,
6.016, 5.774 s` (mean `5.875 s`, population SD `0.124 s`). All `1,463` command
cycles were accepted. Publication gate p50/p95/max was `4.402/14.125/19.969
ms`, with no cycle at or above 20 ms. This is a valid control reference, but
it is not yet at the `5.5 s` target.

The previously screened `e_y=5000`, `e_psi=1800`, steering-slew `1000` overlay
was then repeated under the same CPU isolation. It stopped after two counted
laps on an unpublished nonlinear corridor rejection (no collision was
recorded before the supervisor stopped Unity). The failure was at cumulative
progress `91.91 m`, `e_y=-0.353 m`, `e_psi=0.072 rad`, `u=6.34 m/s`,
`r=+1.31 rad/s`, against `k=-0.095 1/m` and a `6.61 m/s` target. The solver
hit its time-budget termination; the candidate and retained three-command
prefix both failed exact validation at stage 2. One of 691 measured gate
cycles exceeded 20 ms (`21.281 ms`). The run is a failed weight trial, not a
successful speed-up.

The captured full practice baseline was re-exported and replayed with a
moderate combined speed/yaw bundle (`weight_u=20`,
`weight_u_overspeed=30`, longitudinal slip `40`, yaw-rate `25`, steering-slew
`1500`, throttle-slew `0`; other path/steering weights unchanged). On the same
1,463 saved states, exact acceptance was unchanged at `1,461/1,463` versus
production. Mean replay throttle moved only `0.203` to `0.209`; mean speed
error remained `-0.271 m/s`. Replay p95/max cycle cost was `13.42/23.57 ms`
versus `10.80/18.87 ms` for production. On the 92-state terminal window,
acceptance rose `82/92` to `85/92`, but fifth-percentile corridor slack fell
from `2.24 cm` to `0.86 cm`; yaw-weight variants did not reliably reverse the
first steering command. Saved-state replay does not contain the live
certified-plan memory and cannot substitute for a full live validation.

**Disposition:** neither the tracking overlay nor the moderate speed/yaw
bundle is promoted. The terminal episode is a rapid curvature-sign transition
with significant positive yaw rate while the path requests negative
curvature; simply increasing global speed or tracking weights has not yet
shown a repeatable live improvement. The next correction must prevent this
state from developing upstream or improve the transition response itself.
Keep the production v323 profile as the controlled reference while v324 IPOPT
continues.

Artifacts for this continuation are under
`live_runs/mpc_v323_baseline_cpu_isolated_20261025_01/analysis/` and
`live_runs/mpc_v323_trackbundle_cpu_isolated_20261025_01/analysis/`; the two
run manifests and raw bags are retained alongside them.

### Additional coupled speed/yaw screens

To test the user's combined speed-priority proposal without reusing the failed
path-weight change, two direct-weight bundles were replayed on the same full
production-baseline states and the isolated terminal window:

| Bundle | Full baseline-state replay | Terminal-window replay | Decision |
|---|---|---|---|
| `U20/O30/slip40/r25/steer-slew1500/throttle-slew0` | `1461/1463` exact accepts (same as baseline); throttle mean `0.209` vs `0.203`; p95/max cycle `13.42/23.57 ms` vs baseline `10.80/18.87 ms` | `85/92` vs baseline `82/92`; p05 slack `0.86 cm` vs `2.24 cm`; longest reject streak `4` vs `5` | No material full-run gain; do not live-test/promote. |
| `U40/O15/slip40/r50/steer-command6/steer-slew1250/throttle-slew0` | `1457/1463` exact accepts; p05 slack `7.64 cm`; throttle mean `0.210`, with a command reaching `1.0` | `85/92` exact accepts, reject streak `5`, mean replay throttle `0.183`, max `0.89` | More extreme throttle commands without an acceptance/clearance gain; reject. |

Both direct replay screens reuse the recorded vehicle states, so their speed
error is necessarily identical sample-for-sample; neither shows an actual
closed-loop speed or lap-time gain. They only establish that removing the
throttle-slew penalty is possible and that this alone does not make the
controller use materially more throttle. No production weights were changed.

The paired live bags isolate the effect of the failed tracking overlay near
the low-curvature transition. At cumulative progress `91.56 m`, production
had `e_y=-0.065 m`, `r=-0.109 rad/s`, `steering=-0.006 rad`, and command
`-0.026 rad`; the overlay run at `91.61 m` had `e_y=-0.354 m`,
`r=+1.257 rad/s`, `steering=+0.177 rad`, and command `+0.148 rad`. Both were
near zero reference curvature and about 6.5 m/s target speed. This supports
rejecting the overlay: the increased path/heading costs and halved steering
slew left a large positive yaw transient through the transition. The
production-weight baseline remains the only valid live profile from these
comparisons.

The v324 IPOPT process is PID `689833`, still running from the v323 optimizer
seed at 6 cm mesh, with source-equality Phase I and trust factors `2,4,8`,
1,200 iterations per time stage. Its first time stage reached the iteration
cap at objective `5.2074 s` with a feasible iterate but did not converge; it
continued to factor 4. While no simulator was active, its CPU affinity was
expanded from CPUs `0-3` to `0-7` for the offline replay/build work. It is
currently pinned to `0-7`; before another live MPC run, re-pin it to CPUs
`0-3` and recheck the 20 ms maximum gate.

### Curvature-scheduled speed-priority follow-up

The live `U40/O15/throttle-slew=0` trial completed one scored lap at `5.900 s`
then faulted at progress `83.818 m`: cross-track error reached `0.750 m` while
the car was `0.282 m/s` above the local speed reference in a `|k|≈1.0 1/m`
turn. It was faster than the baseline state at that point, but too far off the
line. This supports separating straight-line speed catch-up from turn speed
management rather than applying a globally high under-speed weight.

Implemented `underspeed_curvature_gain_m` in the MPC objective. The effective
under-speed weight is

`weight_u / (1 + underspeed_curvature_gain_m * abs(reference_curvature))`.

The parameter is applied consistently to the QP and exact nonlinear objective,
defaults to zero (preserving the established profile), and is exposed to ROS
and saved-state replay. The full `f1tenth_mpc` build and all 10 package tests
pass.

The same 1,463 saved states were replayed with the new exact implementation:

| Profile | Exact accepts | Optimal ADMM fraction | p05 clearance | Replay max | First-throttle p95 |
|---|---:|---:|---:|---:|---:|
| Production reference | `1461/1463` | `0.856` | `9.10 cm` | `23.99 ms` | `0.113` |
| `U40/O15`, no throttle slew, under-gain `4`, over-gain `1` | `1459/1463` | `0.815` | `9.04 cm` | `24.99 ms` | `0.191` |
| `U40/O60`, no throttle slew, under-gain `10`, over-gain `1` | `1461/1463` | `0.817` | `9.08 cm` | `24.85 ms` | `0.191` |
| `U40/O15`, no throttle slew, under-gain `10`, over-gain `5` | `1457/1463` | `0.818` | `9.11 cm` | `23.83 ms` | `0.200` |

On the `U40/O60` candidate, average first throttle changed from `0.270` to
`0.282` in `|k|<0.25` sections, while the `0.5<=|k|<1.0` bin changed only
`0.136` to `0.137`; exact acceptance matched the baseline. This is a small,
promising distributional shift rather than evidence of a faster lap. The
`U40/O15` high-curvature-penalty candidate reduced exact acceptance and is not
selected.

The first attempted live check used the `U40/O60` overlay but is **invalid as
an A/B test**: its first MPC state already reported `u=0.282 m/s`, whereas the
baseline began at zero. The configured 20 s MPC delay had elapsed before the
simulator was reset, so the car moved before the reset pulse; the eventual
rejection/collision cannot be attributed to the candidate weights. Keep the
bag and its report for the startup-order diagnosis, but exclude its lap and
tracking results from candidate scoring. Corrected procedure: start Unity and
the recorder first, start MPC against the connected simulator, arm the watcher,
reset and confirm near-zero speed/collision count before the MPC delay ends.

The saved-state screen selected a milder follow-up, `U20/O60`, no throttle
slew, under-gain `10`, over-gain `1`: it produced `1463/1463` exact accepts,
`0.831` optimal-ADMM fraction, and higher straight-bin mean throttle (`0.281`
vs `0.270`) with no high-curvature throttle increase beyond `0.004`. This is
still only a fixed-state screen. Its run-only overlay is
`f1tenth_mpc/config/mpc_moderate_straight_speed_priority_candidate.yaml` and
requires a correctly initialized live run. All replay and invalid-run
artifacts are retained under
`live_runs/mpc_underpriority_curvature_screen_20261025_01/analysis/` and
`live_runs/mpc_v323_straight_speed_priority_20261025_01/`.

### Final v324 result and new geometry-seeded IPOPT run — 2026-10-25

This section supersedes the earlier interim notes saying v324 IPOPT was still
running. PID `689833` completed. Its exported target has predicted integral
`5.1872 s`, but the time stages each reached the 1,200-iteration cap; it is a
feasible incumbent, not a converged or Unity-validated raceline. Compared to
the v323 seed, it changes the path only slightly and raises peak sampled
lateral-acceleration demand from `11.013` to `11.464 m/s²`.

The production-weight v324 line was then driven for six laps in the practice
simulator after a verified reset. There were no collisions, no MPC rejections,
and all `1469/1469` commands were published. Ignoring the startup/warmup lap,
the five lap times were `5.817, 6.358, 6.027, 5.603, 5.934 s` (mean `5.948 s`,
population SD `0.249 s`). Against the contemporaneous v323 production run
(`5.874 s` mean, `0.124 s` SD), v324 is slower and more variable. Cross-track
RMSE increased `0.197 -> 0.270 m`, heading RMSE `0.124 -> 0.151 rad`, while
speed RMSE modestly improved `0.824 -> 0.716 m/s`. The largest path/heading
degradation is around path progress `13-16 m`, where steering reaches its
limit and the car falls below the requested speed. **Do not promote v324.**
Timing p50/p95/p99/p999/max was `5.94/14.57/16.07/17.47/18.36 ms`; none of
the `1469` cycles reached 20 ms. The bag is complete, but its sealed manifest
listed the wrong recorder container name; the correct recorder was manually
finalized after the six-lap stop. This is a run-orchestration defect to avoid
in the next manifest, not a vehicle-data loss.

Geometry audits of the available user-shaped connector candidates found a
useful IPOPT seed direction. In the problematic `13-16 m` section, v323 has
peak `|kappa|=0.93 1/m` and minimum footprint-corridor slack `11.3 cm`; the
15%-toward-outer-wall tradeoff line has peak `|kappa|=0.68 1/m` in that
section and at least `25.4 cm` slack. Its full path is `28.301 m` versus
`27.492 m`, so IPOPT must recover that extra length while retaining the
gentler bend. The separate entry-closer/exit-original candidate has `16.50
m/s²` peak lateral-demand and is not selected as the next seed. These are
static geometry comparisons only, not lap-time or driveability claims.

The first attempt to create a new IPOPT seed exposed an ABI mismatch in the
Python source-plant initializer: `physics.py` declared 18 state values and
omitted struct members, while `unity_source_plant_c.h` defines a 26-value
state and complete input/output structs. This caused source-warm-start calls
to be rejected before any samples existed. The Python `ctypes` definitions
now match the C header, wheel-angle state is preserved across calls, and only
the OCP's 18 optimized physical states are exported into the seed. A header-
dimension/order/offset regression suite plus a compiled C ABI transition test
passes (5 tests); the transition test advances four 25 ms steps successfully.

A fresh solve is now running in
`SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v325b_middle_connector_tradeoff15_20261025/`.
It uses the 15%-tradeoff user-shaped practice line as the reference geometry,
the source-plant warm-start, `0.06 m` mesh, source-equalities Phase I, and
`2,4,8` trust-region continuation. The source warm-start completed at 2.4
m/s (479 flying-lap samples over `23.95 s`, `56.48 m` progress; minimum
clearance beyond the configured margin `0.132 m`); IPOPT is building/solving
Phase I. This source warm-start and optimizer output are not Unity validation.
Only a later supervised practice MPC run can establish whether its output is
faster and easier to follow.

The next run-only MPC bundle is prepared at
`f1tenth_mpc/config/mpc_asymmetric_speed_priority_fast_line_candidate.yaml`:
under-speed weight `40`, overspeed weight `15`, both curvature schedules zero,
throttle-slew penalty zero, and production steering/path/slip weights. This
corrects the earlier test direction that attenuated underspeed priority in
tight turns. It is queued for a live trial only after v325 produces a line
with a faster predicted integral and passing geometry checks.

Before the IPOPT candidate is ready, that bundle and a stronger `U100/O5/slip5`
variant were screened against the same 1,469 saved v324 states, using the same
replay binary and sequence initialization as production. The `U40/O15` bundle
changed exact acceptance only `1461 -> 1462`, and mean first throttle only
`0.206 -> 0.212`; in `|k| >= 0.8 1/m`, it changed `0.110 -> 0.117`, while
fifth-percentile corridor slack fell `10.3 -> 7.8 cm`. The stronger variant
changed high-curvature throttle to `0.119`, lowered exact acceptance to
`1455/1469`, and increased the longest rejection streak from `1` to `6`.
Neither is a demonstrated gain on the old line.

The optimizer's proposed throttle movement grows substantially (the U40
high-curvature p95 plan change is `0.58` versus `0.33` on production), but
the accepted first command moves only slightly. The exact nonlinear line
search reduces many of those large proposals to `0.25x` or `0.125x` and
sometimes rejects them. That means a weight-only attack is not reaching the
vehicle as a large command under the old geometry; the lower-curvature route
must be tested before attributing the deficit to just speed weights.

#### Run `_03` rejection and narrow yaw-direction viability guard — 2026-10-25

The watched `mpc_v323_backtracking_budget_live_20261025_03` practice run used
the v323 baseline trajectory with the `U40/O15`, zero-throttle-slew profile
and the production two-SQP setup. It completed two scored laps at `5.9698` and
`6.2617 s` (mean `6.1158 s`), then stopped after an unpublished nonlinear
rejection followed by a Unity bridge/collision fault. Tracking RMSE was
`0.3697 m` cross-track, `0.1992 rad` heading, and `1.0601 m/s` speed. The
publication gate had no misses (`p95 15.484 ms`, maximum `18.805 ms`). At the
terminal failure, both the new candidate and retained 75 ms prefix failed
exact source-plant/corridor validation at stage 2; this was not a deadline
failure.

The additional fourth line-search backtrack did not rescue any of the final
three recorded rejected states on deterministic replay. It stays available
for comparison, but is not credited as the fix.

The paired exact-prefix replay found one actionable selector error upstream of
that terminal failure. At progress `110.403 m`, the selector published a
candidate with lower 75 ms scalar cost (`2167.8` vs `2936.2`) and nearly equal
minimum corridor slack (`0.2679` vs `0.2730 m`). Its endpoint yaw rate was
`+0.453 rad/s` against a strong negative-turn reference of about
`-2.75 rad/s`; the retained endpoint was `-1.907 rad/s`. Re-solving from the
candidate endpoint rejected nonlinear validation at stage 16, while the
retained endpoint produced an accepted line-search plan. This is evidence
that the short-horizon scalar objective can select a prefix that points the
body rotation against a strong turn, even though its own 75 ms path is
corridor-valid.

Added a narrowly scoped terminal yaw-direction gate to quality-aware prefix
selection. It only overrides a candidate that the existing objective would
otherwise choose when (a) the sampled path reference requests at least
`1.0 rad/s` yaw rate, (b) candidate terminal yaw rate opposes it by at least
`0.25 rad/s`, and (c) the exact-valid retained prefix has yaw rate in the
reference direction. The physics rollout, corridor test, objective, ordinary
optimal plans, and one-valid-prefix cases are unchanged. The guard is
diagnostically recorded as `yaw_direction_guard` and the choice is
`retained_plan_yaw_direction_guard`.

This threshold is deliberately a prefix-selection safeguard, not a claim
that `r=u*kappa` is the exact vehicle yaw dynamics. In the 24 paired branch
states, this strong-opposition pattern occurred five times; only the
`110.403 m` branch had actually selected the candidate in the live run. The
other four were already retained-plan selections. Counterfactual next-solve
replay shows the guard would change that one selected branch from rejected to
accepted; it did not change any previously accepted candidate branch into a
rejection in this small corpus. This remains a single-run signal and requires
the watched live retest below.

The full MPC CMake build succeeded and all `10/10` package tests passed,
including new guard tests for a strong wrong-direction candidate and
non-trigger cases. The rejected `U60/O5/slip5` speed-first candidate config
was removed because its saved-state replay had lower exact acceptance and
less corridor reserve than the moderate candidate.

Next live test: use the saved-state-screened `U20/O30`, zero-throttle-slew
overlay on the same v323 practice reference, with the yaw-direction guard
enabled. This is the best remaining modest speed-priority screen in the saved
state replay (`1462/1463` exact-valid actions; mean first throttle `0.210` vs
`0.203` production; higher straight and turn throttle) and is not yet a lap
time claim. Start from a verified reset, record all signals, and run one
warm-up plus five scored laps; stop at the first collision or controller
rejection. Compare with the five-scored-lap production reference (`5.874 s`
mean); promotion requires a faster mean without material tracking-error
regression, no collision/rejection, and no cycle at/over the `20 ms`
publication gate.

The independent v335 IPOPT continuation finished after about `1 h 30 min`.
Its two time stages at trust factors `1024` and `2048` each reached 2,400
iterations. The first stage reached `5.8994 s` but had maximum non-trust
constraint violation `0.1674`; the second worsened to `6.3433 s` and `0.4616`
violation. The exported `6.6062 s` target is explicitly the captured-motion
feasibility fallback from the input seed, not either time-stage iterate. It
is not a faster or feasible new raceline and must not be used for driving.

To avoid repeating the failed oversized trust jumps, v336 has been started
from the same verified v334 source checkpoint with incremental factors
`64,128,256,512`, 1,800 iterations per stage, pinned to CPU 11 at nice `+10`.
Its console is captured in
`SDU_Apex_IPOPT_PhysX_MinTime_v2/optimized_practice_v336_outer_wall_small_trust_continue_20261025_console.log`.
This is still a hypothesis-driven continuation, not an optimizer result;
only a source/geometry-feasible time improvement can advance to a Unity run.

### v325b outcome and v325c continuation — 2026-10-25

The v325b source-equality Phase I reached its 500-iteration budget with
`Maximum_Iterations_Exceeded`, but the continuation wrapper retained its best
strict-feasible checkpoint (maximum dynamic defect `7.20e-7`, no hard
constraint violation). All three minimum-time IPOPT stages converged: trust
factors 2/4/8 returned `11.8379 / 11.7783 / 11.6950 s`. The target is
geometrically/source-collocation feasible at the reported samples, but is not
a fast candidate: its speed is only `2.136-2.413 m/s` (mean `2.388 m/s`), and
peak lateral acceleration is `0.410 m/s^2`. It is not being sent to Unity.
The exported CSV/report say MPC- and Unity-unvalidated. `run_ipopt.py` exited
with code 3 by design because `--skip-native` exports a target without running
the separate source-native witness; this is not an IPOPT failure.

The slow basin came from initializing the time NLP from a low-speed 2.4 m/s
source warm-start. v325c therefore starts from v325b's source-feasible IPOPT
checkpoint and replaces only its initial speed profile using v323's existing
`2.752-7.895 m/s` profile, time-scaling the associated dynamic rates. This is
an initialization change only; Phase I must re-certify the full source
equations and the minimum-time solve still optimizes all states and controls.
v325c is running in
`optimized_practice_v325c_fast_speed_seed_20261025/`; it has not produced an
IPOPT result yet.

This continuation exposed a CLI validation mismatch: speed replacement on a
source-feasible `--optimizer-seed-file` was implemented downstream but rejected
unless `--seed-backend synthetic` was also supplied. The running v325c uses
that accepted invocation while retaining the optimizer checkpoint as its
actual seed. `run_ipopt.py` now permits the explicit checkpoint path to select
the speed-profile replacement without that misleading backend flag; a
regression test covers that case and the remaining invalid combinations.
Validation: 10 targeted Python tests passed, including the five ABI tests.

Next gate: inspect v325c speed/time/clearance and all phase constraint reports.
Only if its predicted integral is faster than v323 (`5.231 s`) and its static
footprint/feasibility checks pass will it receive the combined live profile
(`weight_u=40`, overspeed weight `15`, throttle-slew cost `0`, unchanged
steering/path/slip terms). A live result still needs the supervised practice
Unity batch run; no optimizer target alone establishes driveability.

### v325c result and v325d curvature-aware continuation — 2026-10-25

v325c completed Phase I without an accepted candidate. It started from the
v325b checkpoint with the v323 speed profile time-scaled over that geometry;
at the checkpoint geometry this implied peak `u^2*|kappa|=16.68 m/s^2`,
above v323's own peak `11.3925 m/s^2`. Phase I began at maximum equality
defect `0.136617`, reached trial defect `0.420`, and exported no candidate
(exit 2). The attempted speed initialization was too aggressive for that
geometry; it is not evidence that the source-constrained track itself has no
fast solution.

The next continuation is v325d. `run_ipopt.py` now reconstructs the actual
candidate XY path from checkpoint `ey_m`, computes discrete Menger curvature,
and can cap an imported speed profile so its initial `u^2*|kappa|` does not
exceed the reference CSV's measured maximum. This operation is explicitly an
initializer only: it adds no OCP/vehicle bound, and Phase I still has to
re-establish the source dynamics. Regression coverage includes circle-offset
curvature, the cap calculation, and invalid CLI combinations; `py_compile`
and the focused speed-seed plus source-plant ABI tests pass (13 tests, 19
subtests).

v325d setup/provenance checks pass. It uses the same v325b source-feasible
checkpoint, same centerline/mesh/model, and v323's speed/curvature CSV. Its
transferred profile starts at `min/mean/max=2.752/5.603/7.895 m/s`; 49 nodes
are capped and the candidate path peak is `11.392 m/s^2`. This remains only a
speed initializer. At the latest check it is in Phase I iteration 120/500;
there are zero accepted feasible iterates, and trial maximum dynamics defects
have oscillated (best observed near `0.047` at iteration 20, then mostly
`0.45-1.25` in later callbacks). The source-feasible v325b checkpoint remains
the retained certified state. Do not count v325d as a raceline or improvement
unless the solve returns strict feasibility and a faster minimum-time result.

v325d is now complete: after 500 iterations IPOPT returned
`Maximum_Iterations_Exceeded`, with no accepted checkpoint and strict
maximum dynamics violation `0.1247618`, again the periodic lateral-velocity
collocation row. No candidate or restart checkpoint was exported. The
callback diagnostics oscillated through iteration 490; low individual
backtracked residual samples did not imply whole-lap feasibility. The run
consumed about 636 seconds in IPOPT. The broad speed replacement is rejected
as an effective initialization strategy for this checkpoint.

Next continuation v325e adds `--checkpoint-speed-seed-fraction` so only a
fraction of the target-profile change is applied to the source-feasible
checkpoint before rate/time scaling. It is still only a starting-guess
homotopy; exact source dynamics remain enforced, and it introduces no new
limit. The CLI validates that a fractional change has both a profile and a
checkpoint. Unit tests now cover blending and invalid use; focused tests pass
(15 tests, 19 subtests). v325e will start at fraction `0.25` (roughly
`3.19 m/s` mean rather than jumping to `5.60 m/s`) to test whether staged
continuation can retain feasibility before IPOPT increases speed itself.

v325e has now passed Phase I. Its initial defect was `0.0355193`; the retained
checkpoint at Phase-I completion has maximum dynamics defect `8.7888e-4`,
with 37 accepted iterates/backtracks over 500 iterations. Intermediate
checkpoints reached `3.39e-6` and `1.80e-7`; final checkpoint selection is
still governed by the objective and strict feasibility gate. The run has
entered the first minimum-time trust-region stage (factor 2). That stage has
not produced a lap-time report yet, so v325e is not yet a faster raceline.

### v326 outer-wall speed homotopy and v323 speed-priority live attempt — 2026-10-25

The v326 IPOPT solve is running from the v325e source-feasible checkpoint,
using the same 15%-tradeoff middle-connector geometry, but a 50% speed-profile
homotopy toward the v323 profile. At the latest check it was still in Phase I
(iteration 446/500): no strict-feasible checkpoint had yet been accepted.
The per-iteration callbacks continue, but the best reported trial defects
remain far above the strict dynamics gate. Do not count v326 as a raceline
unless it passes Phase I, all min-time stages, and the exact export checks.

The first live v323 run with global underspeed priority (`U=40`, `O=15`),
zero throttle-slew cost, and unchanged path/steering weights did not complete
the supervised six-lap run. It published 360 commands, completed one lap in
`34.751 s`, and then collided after both the new candidate and retained plan
failed exact short-prefix corridor validation at stage 2. This was not a
deadline failure: maximum publication-gate elapsed time was `18.751 ms`,
below the 20 ms gate. There were 25 consecutive retained-plan actions before
the final rejection.

The captured onset shows why the global speed weighting is not enough. From
`s=52.418 m` to `54.920 m`, path curvature steepened from `-0.157` to
`-0.614 1/m`; reference speed fell from `6.825` to `4.215 m/s`. Measured
forward speed fell from `6.83` to `5.02 m/s`, while cross-track error grew
from `0.020` to `0.220 m` and heading error from `0.003` to `0.108 rad`.
At `s=54.920 m`, the car remained `0.81 m/s` above reference. The model
predicted one-step XY RMSE `5.7 mm`, but closed-loop 750 ms XY RMSE was
`0.283 m`; these are distinct measures. The failure starts in the
curvature/speed transition, not in the cycle-time gate.

A follow-up saved-state replay tested a curvature-dependent overspeed guard
while retaining free throttle slew and high straight-line underspeed weight:
`U=40`, `O=15`, overspeed-curvature gain `3`, underspeed-curvature gain `0`.
On the same 1,463 baseline states it exact-accepted `1455/1463`, versus
`1461/1463` for the same-state production profile; p05 corridor slack was
`9.16 cm` versus `9.10 cm`, and longest rejection streak was `3` versus `1`.
The 41-row failure-terminal replay improved exact acceptance from `14/41` to
`16/41` and p05 slack from `0.42 cm` to `1.86 cm`, but its longest rejection
streak rose from `20` to `23`. This is a mixed fixed-state screen, not a
closed-loop gain; no speed-priority weights are promoted from it.

**Next at the time of that entry:** finish v326 Phase I and, if it fails,
retry from v325e with a smaller speed increment. The completed follow-up
below records the outcomes and replaces that stale next-step note.

### 2026-10-25 continuation: failure-capture screen and wider IPOPT trust region

The three larger profile-transfer trials are terminal and exported no
candidate: v326 (50%) began with max dynamics defect `0.06294`, v327 (35%)
with `0.04781`, and v328 (30%) with `0.04218`. All hit the 500-iteration
Phase-I cap without a strict-feasible checkpoint. Roll rate was the largest
initial equality residual in each. They are rejected as racelines; the
source-feasible v325e checkpoint remains intact.

I also found why the v325b/v325e minimum-time stages barely raised speed. The
time-stage trust region is converted into one uniform per-variable box by
`sqrt(reference * factor / decision_variable_count)`. At 472 nodes, that
divides one radius across more than ten thousand state, control, load, and
acceleration variables. With factors only `2,4,8`, the allowed per-node speed
move is small. In v325e, the speed mean moved only `3.192 -> 3.307 m/s`; its
time stages converged, but the exported target was still `8.488 s`. This is a
trust-region exploration limit, not evidence that a faster feasible minimum
does not exist. For v325e's feasibility reference `13.53` and 11,329 trust
components, factor 8 allows about `0.098` normalized speed units (speed scale
2 m/s, or about `0.20 m/s` per node). Factor 1024 allows about `1.11`
normalized units (about `2.21 m/s` per node), sufficient to explore the
current 3.3-to-5.5 m/s gap while still enforcing the original nonlinear
constraints.

Two new IPOPT continuations are running against the same practice map,
5 cm margin, 15%-tradeoff user-sketch geometry, and source-derived positive
drive-only model:

| Run | Seed/change | Time-stage trust factors | Current state |
|---|---|---:|---|
| v329 `outer_wall_speed_homotopy_010` | v325e source-feasible output; 10% speed-profile increment toward v323 | `2,4,8` | Exported feasible incumbent at 7.881 s (not converged; `Error_In_Step_Computation`, 697 iterations). Source balances were zero at OCP nodes, sampled footprint clearance 5.38 cm, max speed 4.34 m/s. This is slower and substantially slower-speed than v323, so it is rejected; no simulator run. |
| v330 `outer_wall_trust_expand` | unchanged v325e source-feasible output; isolates wider time-stage search | `16,64,256,1024` | Factor 16 hit its 1,200-iteration cap at a feasible 8.272 s; factor 64 converged at 8.067 s with zero non-trust violation. Factor 256 is now running. Still slower than v323; no final output candidate yet. |
| v331 `outer_wall_incumbent_continuation` | v329’s reported feasible 7.881 s incumbent; source-equality gate may skip Phase I only after a fresh full-constraint check; `64,256,1024` | `64,256,1024` | Started on CPUs 0–7. The new code rechecked the saved incumbent and did **not** take the skip branch, so it entered source-equality Phase I rather than assuming the exported checkpoint remained strictly feasible. At iteration 20, raw max defect was `7.77e-13` and hard violations were zero; the reason the full gate declined reuse remains to be identified from its complete diagnostics. A gate-reason log was added for subsequent runs and its 11 tests pass. |

The 10% increment was selected because the 30% attempt still failed
immediately from a much larger initial residual. The v330 run tests the newly
identified decision-dimension/trust-radius restriction without changing
vehicle equations, constraints, path geometry, or speed seed. Both still
require exact source-feasibility and export validation; predicted time alone
does not qualify a line.

#### Coupled MPC speed-priority screen

With throttle-slew cost set to zero, four profiles were replayed against the
same complete 1,463-state v323 reference and the 93-state turn-guard capture.
This is fixed-state solver/acceptance replay, not a closed-loop lap test.

| Profile | Full capture exact accepts | Turn capture exact accepts | Turn reject streak | Turn p05 clearance | Turn p95/max cycle |
|---|---:|---:|---:|---:|---:|
| Production | `1462/1463` | `86/93` | 7 | `12.57 cm` | `16.16/18.68 ms` |
| `U40/O15`, turn overspeed gain 3 | `1455/1463` | `73/93` | 19 | `15.90 cm` | `17.20/17.98 ms` |
| `U60/O20`, turn overspeed gain 2, modestly higher path costs | `1454/1463` | `86/93` | 7 | `8.86 cm` | `16.74/20.24 ms` |
| `U80/O20`, stronger path costs | `1457/1463` | `73/93` | 19 | `13.61 cm` | `19.42/24.77 ms` |

The higher-priority profiles did increase command-step p95 (`0.119` baseline
to `0.328` for U60 and `0.378` for U80) but did not improve exact acceptance
in the critical turn. U60 matched the baseline count but lost clearance and
crossed the 20 ms stress-replay limit; U40/U80 increased rejection streaks.
Do not promote or run these as a live profile. This supports the user's
direction that throttle slew must not be the speed limiter, but changing
speed weights alone is not a fix for the path/turn failure.

The earlier live test with the turn-guard candidate remains invalid as a
supervised lap: the run watcher was started after the configured MPC delay,
so movement began before monitoring. The resulting bag is retained for
stress replay only. The next run sequence is controller (with a 60 s MPC
startup delay) -> watcher armed and `WATCHER_READY` confirmed -> recorder ->
practice Unity; this ensures monitoring is live before Unity can move. No
candidate or closed-loop lap gain is claimed from that attempt.

**Current disposition:** production weights and the 5 cm corridor remain
unchanged. The fixed-state screens do not support a speed-weight promotion.
Continue the two IPOPT continuations; then take only a strict-feasible target
with a predicted lap time faster than v323 to a watched, full-speed practice
run. For MPC, focus on the first high-curvature divergence and command
chronology rather than another blind scalar sweep. A candidate counts only
after repeated multi-lap practice runs with low lap-time variance.

#### Source-feasible Phase-I short-circuit

While v330 was running, I found that `initial_is_feasible` in the
source-equality optimizer was hardcoded false. Consequently, even an exact
source-feasible checkpoint entered all 500 Phase-I IPOPT iterations. The v330
log confirms this waste: it started from 2.81e-8 maximum dynamics defect,
found no Phase-I descent, and spent about 750 s in that pass before entering
the time stages.

The generated source-equality solver now enables its existing zero-iteration
seed-reuse branch only when both conditions pass: the original source dynamic
defect is within `NLP_FEASIBILITY_TOLERANCE`, and the complete original NLP
constraint audit (including source equalities, actuator bounds, and corridor
constraints) is within the same tolerance. All other seeds still run Phase I.
This changes only startup work; it does not relax constraints or promote a
candidate. The predicate and generated source are covered by the
`test_source_equalities_phase1_uses_exact_dynamics_and_seed_distance` test;
the complete `test_v251_phase1_reduction.py` file passes (11 tests).

This optimization does not affect the two already-running Python processes:
their modules were loaded before the source edit. It will apply to future
continuations and retries.

#### Hairpin failure and paired solver-policy replays — 2026-10-25

The reset-confirmed run06 bag was re-decoded to locate the first loss of
track, not just the terminal rejection. Cross-track error grew from `0.03 m`
at progress `25.25 m` to `0.36 m` at `27.90 m`, then to `0.83 m` by
`29.28 m`. Across the same section, speed target fell from about `6.56` to
`3.45 m/s`; actual speed stayed above target and peaked about `0.75 m/s` over
it near turn-in. Yaw response lagged the increasing negative curvature. Thus
this bag does **not** support removing turn-speed protection: the late turn
was already overspeeding while lateral error accumulated.

The first retained-plan action began at `e_y=0.365 m`, `e_psi=0.268 rad`,
`u=4.24 m/s`, and `k=-0.605 1/m`. The old certified sequence then supplied 23
successive actions while repeated new candidates failed. At the terminal
cycle, the state was already just outside the exact corridor (`e_y=0.834 m`
vs. upper bound `0.8246 m`); the QP's first action reversed steering to
`+0.524 rad` and dropped throttle to its `0.001` floor. Its exact rollout
missed the corridor by `8 mm` at stage 2. The solve used eight Jacobian
workers, about `5.92 ms` for QP/Jacobian construction and `4.89 ms` in ADMM,
with primal/dual residuals `2.30/149.19`. This is a failed solve after the
vehicle had already left the feasible corridor, not evidence that relaxing
the corridor would help.

The post-solve reserve was temporarily screened at `2 ms` against `3 ms` on
the *same* 1,463 saved states and the same `U40/O15`, zero-throttle-slew
profile. All 1,463 first actions and all acceptance statuses were identical.
Replay timing was noisier at 2 ms (p95 `17.13 ms`, max `22.55 ms`) than the
3 ms replay (p95 `17.57 ms`, max `29.23 ms`); these offline runs have no live
20 ms deadline gate, so neither max is a live timing claim. No behavior gain
was demonstrated; retain the production 3 ms reserve.

The extra SQP relinearization was tested rather than removed. On those same
states/configuration, one pass accepted `1,437/1,463` actions (`98.22%`),
while two passes accepted `1,452/1,463` (`99.25%`): the second pass adds 15
exact-valid actions and is therefore retained behind its deadline gate. With
the per-pass cap reduced from 150 to 100, two-pass replay accepted
`1,453/1,463` (`99.32%`), p95 fell from `17.57` to `15.57 ms`, and max from
`29.23` to `27.05 ms`. This fixed-state experiment is not enough to lower the
production iteration cap; the offline tool does not enforce the live cycle
deadline and these are not lap-time results.

A more extreme simultaneous speed bundle (`U100/O1`, throttle-slew cost zero,
100 iterations) was also screened. It accepted `1,403/1,463` actions
(`95.90%`), with mean first throttle `0.213` (versus `0.217` for U40/O15);
in the tracked hairpin its acceptance was `221/247` and mean first throttle
`0.131` (versus `241/247` and `0.141` for U40/O15/100). It did not deliver
more throttle and is rejected. The most defensible live candidate remains the
milder `U20/O60`, under-speed curvature gain `10`, over-speed curvature gain
`1`, throttle-slew cost zero: its saved-state screen was exact-valid on all
1,463 points and raised straight-bin mean first throttle from `0.270` to
`0.281`, without extra throttle in the high-curvature bins. A correctly
reset, watched practice run is required to establish whether that is a real
closed-loop speed gain.

#### Supervised five-lap speed-priority trial and IPOPT continuation — 2026-10-25

The candidate `U20/O60`, under-speed curvature gain `10`, over-speed
curvature gain `1`, throttle-slew cost `0` was run on practice with the v323
trajectory (SHA-256 `9827e5f2f8f4e4dd416fde19e28ed70f37d1516571cd0f6f2c334b62a923b8ad`).
The collision/lap watcher stopped the run at five lap-count increments with
no collision. The first lap was warm-up and is excluded: the four scored laps
were `6.1767, 6.0507, 6.1427, 6.5197 s` (mean `6.2225 s`, range `0.4690 s`).
The same v323 line's prior production-profile run completed five scored laps
at mean `5.8744 s` (range `5.6858–6.0157 s`). The candidate is therefore
`0.3481 s/lap` slower on these runs, not a gain.

The corresponding closed-loop diagnostics also regressed: cross-track RMSE
`0.3223 m` vs `0.1967 m`, heading RMSE `0.1794 rad` vs `0.1235 rad`, and
speed-error RMSE `0.8923` vs `0.8240 m/s`. Its cycle-gate p95/max was
`14.525/22.200 ms`, with two samples at/over the 20 ms gate; the baseline
was `14.125/19.969 ms`, with none over. It published 1,303 commands across
1,305 cycles, used the retained certified plan 30 times, and had two isolated
deadline recoveries. No collision occurred, but this is not a successful
speed-priority promotion.

The sector data explains the loss. In the `10–15 m` sector, mean speed was
`3.91 m/s` vs `4.19 m/s` baseline, with cross-track RMSE `0.51 m` vs
`0.31 m`; in `15–20 m`, it was `4.56` vs `4.94 m/s`. The tested schedule
mathematically reduces under-speed weight in bends:
`weight_u / (1 + 10*|curvature|)`. At `|curvature|=0.3 1/m`, its effective
under-speed weight is only `5`, below the baseline `7.5`. Zero throttle-slew
cost also yielded almost no mean-throttle gain (`0.209` vs `0.207`) while
raising p95 throttle step from `0.102` to `0.194` and allowing commands to
reach `1.0` (baseline max `0.58`). This directly rejects the tested bundle;
the next bundle must not attenuate under-speed pressure in turns, and any
reduced throttle damping must be checked against this observed command
chatter.

Two further saved-state replays used the same 1,463 baseline states and v323
line. `U40/O15`, under-speed gain `0`, over-speed gain `1`, throttle-slew `0`
reported `1,301/1,463` accepted; `U20/O60` with the same curvature gains
reported `1,336/1,463`. These counts are **not comparable** to the stored
baseline replay: the new invocations accidentally disabled nonlinear line
search, while the baseline used the production three-backtrack setting. They
are retained only as full-step diagnostics and must not be used to reject a
profile. The corrected `U20/O60`, under-speed gain `0`, over-speed gain `1`,
throttle-slew `0` replay with line search enabled is reported below.

**Correction:** a true production-weight replay (`U7.5/O60`, overspeed
curvature gain `1`, underspeed gain `0`, throttle-slew `100`) accepted
`1,461/1,463` states, with mean first throttle `0.203`, mean turn throttle
`0.133`, and 701 line-search-accepted actions. `U20/O60`, no throttle-slew
cost, under-speed gain `0`, over-speed gain `1` also accepted `1,461/1,463`;
it raised mean throttle to `0.208` and turn throttle to `0.136`, with 758
line-search-accepted actions. A lower overspeed-cost variant, `U20/O30` with
the same curvature schedule and zero throttle-slew cost, accepted
`1,462/1,463`, raised mean throttle to `0.210` and turn throttle to `0.141`,
and had 801 line-search-accepted actions. Offline cycle p95/max was
`11.86/23.83 ms` for U20/O30 versus `12.35/30.28 ms` for the production-weight
replay; these are not live timing guarantees. U20/O30 is the strongest
corrected saved-state candidate and is queued for a watched six-lap-count
practice test (one warm-up plus five scored laps). The prior U20/O60
curvature-gain-10 live bundle remains rejected on its actual closed-loop
outcome.

IPOPT v334 completed its source-equality continuation from a strict-feasible
seed. Factor `64` converged at `7.5119 s`; factor `256` reached 1,200
iterations at `7.1785 s` with maximum non-trust violation `4.66e-5`; factor
`1024` reached 1,200 iterations at `6.6062 s` with maximum non-trust
violation `1.71e-4`. Its report says the native witness was skipped, and the
candidate is not MPC/Unity validated. It is slower than v323's `5.2310 s`
optimizer target and cannot be promoted. Since the final stage was still
improving when it hit its iteration cap, a v335 continuation was started from
v334's saved source seed, using factors `1024,2048`, 2,400 iterations per
stage, and CPU 11. It is a background search only; its output must still pass
source/geometry audits and an actual watched practice run before it counts.

### Follow-up: direct speed-priority trials and trajectory provenance correction — 2026-10-25

The saved-state screening was followed by three watched practice trials. A
trajectory-resolution audit found that the manifests initially named the
intended `ipopt_v1_baseline_raceline.csv`, but the launch had no paired
`SDU_APEX_MAP_YAML` / `SDU_APEX_TRAJECTORY_FILE` override and therefore loaded
the package default `autodrive_practice_20260924_b_mintime_raceline.csv`
(SHA-256 `09b0c3e5b36b781b8709eb08a98cd8c36eb7951391f269ed761409a64938a253`).
The reports and manifests have been corrected to the actual resolved line.
Consequently these runs characterize the controller on that mintime line;
they do **not** measure its ability to follow the intended IPOPT line. The MPC
now logs the resolved trajectory path at startup so this discrepancy is
visible in future runs. Future supervised tests must set map and trajectory
overrides together and compare the startup-resolved path/hash with the sealed
manifest before scoring.

The `U20/O30`, no-throttle-slew trial completed 12 scored laps after its
warm-up, with no collision: mean `6.0415 s`, population SD `0.0211 s`, range
`6.0057–6.0857 s`. Cross-track and heading RMSE were `0.0288 m` and
`0.0281 rad`; speed-error RMSE was `0.439 m/s` including startup. Cycle-gate
median/p95/max was `4.36/8.11/29.15 ms` with one sample at or over the 20 ms
budget. This is repeatable, safe operation on the resolved default line, but
not a speed improvement over the previously reported `5.785 s` run.

The more aggressive `U40/O0`, no-throttle-slew trial did not complete a lap.
At progress `43.925 m`, the car had `1.220 m` cross-track error and `0.390 rad`
heading error while travelling `2.752 m/s` against a `3.560 m/s` target in a
`-0.642 1/m` turn. Speed had been above target shortly before the departure;
the controller then chained 27 retained prior-plan actions after corridor
failures, and the next candidate and retained plan were both invalid. The
collision watcher stopped the run. This rejects globally unpenalized
overspeed: its failure was a high-curvature tracking loss, not evidence that
throttle slew should be constrained.

The `U40/O15` profile with no throttle-slew cost, curvature-aware slip and
overspeed terms, and stale-plan refresh completed six lap-count increments
without collision. Four recorded scored laps were `6.0477, 6.0557, 6.0717,
6.0607 s` (mean `6.0589 s`, population SD `0.0087 s`); the fifth scored lap
time was not recorded before the supervised stop. Cross-track/heading RMSE
were `0.0295 m` / `0.0288 rad`, and speed-error RMSE `0.6275 m/s` including
startup. Gate median/p95/max was `4.45/8.27/20.58 ms`, with one sample at or
over budget. The stale-refresh and yaw-direction guard did not activate. This
profile is clean but slower than the `U20/O30` run; it is not promoted.

Taken together, the trials do not support another global weight increase as a
path to lower lap time. The safe profile tracks its requested speed closely
after startup, yet remains around `6.04–6.06 s`; higher global speed pressure
causes a turn departure. The next meaningful opportunity is a genuinely
faster, feasible speed/geometry target—not further weakening bend overspeed
cost or limiting throttle slew. Keep the latter at zero unless new evidence
shows command chatter itself is the cause.

#### IPOPT continuation v336 status

The v336 run is continuing from v334's source-feasible full-state checkpoint,
with the edited outer-wall/middle-connector seed, 6 cm mesh, 5 cm wall margin,
and trust factors `64, 128, 256, 512`. The seed passes the source-equality
reuse gate (`max dynamic defect = 0.000171`, tolerance `0.001`), so Phase-I
was correctly skipped. At 23:44 local time the process was still running on
one CPU core after about 20 minutes, with about 3.27 GB RSS, but the log had
not yet emitted its first IPOPT iteration or saved a candidate checkpoint.
This establishes active CPU work, not optimizer progress or a raceline
improvement. Leave it running and inspect its first stage result; any exported
line remains an unvalidated target until geometry/source audits and a
supervised MPC run with an explicitly resolved trajectory path pass.

## Follow-up: reject poor-residual steering reversals — 2026-10-26

The new approach is to stop changing speed weights and first prevent the MPC
from publishing a control direction that its own QP has not solved adequately.
The `mpc_iterations150_repeat_20261025_01` run remains the record-profile
reference: the original `ipopt_v1_baseline_raceline.csv`, 150 iterations,
overspeed-curvature gain 1, and the production weight profile. Its five scored
laps averaged about `5.785 s` (best `5.651 s`). The failed `v323` fast-line
trial used a different trajectory plus the run-only `U20/O30/no-throttle-slew`
overlay, so it is not a controlled comparison of raceline alone.

At progress `121.331 m` on the failed trial, the vehicle was at
`u=6.429 m/s`, `r=-1.339 rad/s`, in `k=-0.255 1/m`, with a `5.928 m/s`
reference. The nominal steering was `-0.089 rad`; the selected line-search
command was `+0.111 rad`. Its exact 30-step source-plant rollout predicted
cross-track error growing to about `1.0 m`, and the recorded vehicle followed
that excursion. The QP had stopped on its time budget at 143/150 iterations,
with primal/dual residuals `0.1029/0.0353` against tolerance `0.02`. The code
accepted any positive-alpha backtrack without checking those residuals.
Therefore this event does not support blaming the plant rollout first: the
candidate was already visibly poor in the model, and the publication gate
allowed it through because it remained inside the track corridor.

Using the same established quality rule as rejected-horizon prefixes, the
new `mpc_rti_solver_residual_quality_ok` gate requires both residuals of an
iteration-limited result to be at most `2 × tolerance` (`0.04` for the current
`0.02` setting). It runs before nonlinear backtracking; a poor QP direction
is rejected and the controller attempts to publish the next action of its
retained plan only after exact current-state source-plant validation. No
emergency-brake command or weight change was introduced. A cycle with no
exact-valid retained action remains an explicit controller fault.

The gate was applied to saved diagnostics with the actual `0.04` threshold:

| Capture | Accepted cycles | Poor-residual accepts blocked by the rule | In progress 115–125 m | Median `|e_y|`, poor vs other |
|---|---:|---:|---:|---:|
| Known-good record-profile run | 771 | 93 (12.1%) | 2 | `0.210 m` vs `0.055 m` |
| Failed v323 fast-line/U20-O30 run | 639 | 107 (16.7%) | 33 | `0.321 m` vs `0.066 m` |

The record run also had poor-residual cycles, so the new gate is not yet
declared a lap-time improvement. The strong discriminator is concentration:
the failed hairpin has 33 such accepted cycles in the 115–125 m region versus
two in the record run. Longest poor-residual run was 10 cycles in the record
capture and 17 cycles in the failed hairpin; both fit inside a fresh 30-action
retained horizon, subject to exact revalidation at each publication.

Implementation and checks so far:

- The residual gate now applies before both full-step acceptance and
  line-search acceptance, including second SQP passes.
- The controller reports `solver_residual_quality_rejected` distinctly.
- A regression test uses the hairpin residuals and verifies that a feasible
  positive-alpha step cannot bypass the gate; an integration test verifies
  rejection before line search. The MPC target builds and all 36 source-rollout
  tests pass.
- The exact historical controller image is not present locally. Its run
  manifest still establishes the practice map, original trajectory hash,
  loaded parameter hashes, and measured lap results; a new run will validate
  the current source against that profile rather than claim binary identity.
- No simulator has been launched for this change yet. Next: rerun the original
  record trajectory/profile under the new gate, score only laps after warmup,
  and compare lap mean/variance, steering direction through the hairpin,
  retained-plan reuse, path/speed errors, collisions, and the 20 ms gate.
- Keep IPOPT processes `1381634` (v336) and `1435495` (v337) running in the
  background; neither is part of this MPC diagnosis.
