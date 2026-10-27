# High speed MPC tracking — 2026-10-27

Goal: repeated practice laps averaging at most 5.5 s after the warmup lap,
using the requested trajectory speed and competition runtime inputs.

## Starting evidence

- Saved best repeated profile: mpc_throttle_slew75_20261025_01, five scored
  laps averaging 5.7790 s. Its nominal ipopt_v1_baseline_raceline.csv integral
  is 5.367 s. All 1,443 MPC cycles passed full-horizon validation; no recovery
  prefix was used. Thus prefix recovery does not explain its whole time gap.
- Large speed-weight changes and reduced steering damping have already failed
  live trials. Focus this work on the first divergence and solver/model
  consistency before further weight searches.
- Two background IPOPT jobs remain active. They were pinned to logical CPUs
  14 and 15 at nice 10 for live tests. The controller uses CPUs 0–11 and Unity
  uses CPUs 12–13, avoiding direct competition for those logical CPUs.

## Fresh restored baseline

Run: live_runs/mpc_restored_baseline_20261027_01.
Source commit a6f035956 plus the sealed worktree manifest.
Controller rebuilt mounted production sources. Practice stock image,
batchmode, full trajectory speed, eight Jacobian workers, production weights.
Recorder and watcher armed before Unity, reset held through Unity startup.

Result: stopped before the first lap after an unpublished corridor rejection.
The terminal nominal and candidate contain saturated steering reversals.
The run, controller log, reset history, and full typed state are retained for
first-divergence analysis. No improvement is claimed from this trial.

## Active investigation

1. Compare predicted and measured yaw/steering at the first bad command.
2. Check the QP objective and exact objective for consistency.
3. Check conversion of physical displacement into moving path coordinates.
4. Make and verify production corrections, then repeat supervised full-speed
   runs with the first lap excluded from timing scores.
5. Report fixed-input plant error separately from closed-loop tracking, and
   retain the maximum 20 ms command-publication budget.

## Implemented and checked

- Replaced the second-order path-coordinate update with exact projection onto
  its local reference circle, including the matching analytic sensitivity.
  The same Unity body displacement is used. Cached the transform geometry
  across sensitivity lanes and removed redundant yaw-wrap trigonometry.
- Added configurable SQP input trust boxes centered on each nominal command.
  They restrict the local QP step without limiting total throttle or imposing
  an actuator slew limit across cycles. Physical bounds remain in force.
- Exposed both radii in ROS parameters, diagnostics and the offline replay.
- Added a reusable supervised practice launcher that seals provenance,
  rebuilds mounted code, arms recording/supervision, and holds reset through
  startup. No controller or simulator state is read from ground truth.
- Four relevant CTest targets passed, including exact-circle and QP bounds
  checks, source-plant derivatives and command-time state propagation.

Fresh baseline fixed-command replay: 212/212 valid 750 ms windows stay within
5 cm at every sample; maximum position error 1.98 cm. Forward-speed RMSE is
0.00916 m/s and yaw-rate RMSE 0.00908 rad/s at 750 ms. These measurements
point to optimization/control decisions rather than a broad physical model
error as the cause of the baseline's tracking failure.

## Local SQP live trial

Run: live_runs/mpc_local_sqp_20261027_01.
Input radii: steering 0.08 rad, throttle 0.10. Production weights.
One warmup plus five scored laps completed, no collision or recovery prefix.
Scored laps: 5.6778, 5.8498, 5.7978, 5.8668, 5.8448 s.
Mean 5.8074 s, best 5.6778 s. All 1,439 commands passed full-horizon
validation. Publication p50/p95/max: 4.15/9.48/12.34 ms.
Tracking cross-track RMSE: 0.144 m; speed error RMSE: 0.718 m/s.
This is improved operation versus the fresh failed baseline; it does not
beat the historical 5.779 s average or meet the 5.5 s goal.

Next: a coupled speed-tracking trial with under-speed weight 25, throttle
slew weight 25, steering radius 0.08, and throttle radius 0.15. Keep the
measured plant and path for the comparison. Promote only measured gains.

## Stronger speed bundle

Run: live_runs/mpc_local_sqp_speed_20261027_01. Completed six laps.
Scored times: 5.6788, 5.7358, 5.9658, 6.0197, 6.1497 s.
Mean 5.9100 s, population SD 0.1769 s. Publication maximum 19.39 ms.
This bundle regressed and was not promoted.

## Located and corrected loaded-path geometry mismatch

The replay now checks a plan's integrated physical displacement against the
world pose reconstructed from its own progress/cross-track/heading states.
Even with the exact local circle correction, changing reference curvature
produced median/p95/maximum position inconsistency of 3.90/10.93/12.55 cm,
and maximum heading inconsistency 0.0602 rad. This is a coordinate/model
transcription error; it is separate from Unity physical prediction accuracy.

Implemented a shared inverse of P=C(s)+e_y*N(s) using the actual loaded path
interpolation. Projection, nominal rollout, candidate validation and
command-time prediction now use the same frame. The QP chains the analytic
frame derivative with source-plant integrated-body-motion sensitivities.
Physical motion does not require an approximate curvature map.

On the same 232-state corpus the corrected maximum closure error is
0.000000322 m and 0.000000224 rad. Replay cycle p50/p95/max:
4.09/8.87/16.46 ms. This is internal geometric consistency, not a new claim
of that accuracy against Unity truth.

The independent cross-knot geometry and derivative tests pass. Removed the
old first-reference/second-projection nominal pass: a single physical rollout
now determines final progress and reference sampling, with no repeated plant
propagation. No optimizer states or physics calls were added.

Live trial in progress:
live_runs/mpc_loaded_frame_sqp_20261027_01, original production weights plus
the 0.08 rad / 0.10 local SQP radii.

Result: six completed laps, no collision, and all 1,418 scored controller
cycles accepted through full-horizon validation without recovery prefixes.
Scored laps: 5.7108, 5.8098, 5.7238, 5.5878, 5.6438 s.
Mean 5.6952 s, population SD 0.0753 s. Publication p50/p95/max:
4.12/9.39/17.82 ms; no sample at or over 20 ms.
Cross-track/heading/speed RMSE: 0.0946 m / 0.0750 rad / 0.6913 m/s.
The saved 5.779 s record capture had 0.1291 m / 0.0968 rad / 0.7114 m/s.
This is a measured live improvement, short of the 5.5 s average goal.

Promoted the local SQP radii to the production profile. The temporary
identical candidate YAML was removed; its original contents are saved as
mpc_overlay.yaml in both relevant run directories for reproduction.

The next live trial uses the same production controller against the v337
5.1465 s nominal target:
live_runs/mpc_v337_loaded_frame_20261027_01.

Result: six laps, scored mean 5.7478 s, SD 0.0597 s, max publication
16.88 ms. This did not outperform the baseline trajectory. Fixed-command
750 ms replay: XY RMSE 1.11 cm, maximum 5.46 cm; 9/1406 windows exceeded
5 cm somewhere along the horizon. Speed/yaw-rate RMSE 0.0160 m/s /
0.0111 rad/s. No claim of a universally exact Unity replica is justified.

Coupled weight trial live_runs/mpc_loaded_frame_balanced_20261027_01:
under/overspeed 25/30, steering slew 1500, throttle slew 25, throttle local
radius 0.15. Six laps; scored mean 5.7596 s, SD 0.1239 s, max 15.54 ms.
One candidate prefix. Not promoted: the unchanged weight profile remains
the measured best at 5.6952 s. Trial snapshots are retained for reproduction.

Next correctness fix: ensure optional SQP refinement starts from the accepted
sequence (without an artificial temporal shift), compares the same nonlinear
objective against the first pass, and accounts both passes' computation once.
Then test actual speed tracking, not just solver acceptance.

## SQP and merit acceptance corrections

Implemented: same-cycle refinement no longer shifts its seed as if time had
advanced (the previous shift also duplicated the penultimate tail command).
It starts from the accepted first pass when available, shares the first
pass's speed targets, and selects refinement only if exact cost improves.
Pass timing/counters now include each pass once regardless of which wins.
Optional refinement also covers damped alpha <= 0.5 steps within the deadline.

Run live_runs/mpc_monotone_sqp_20261027_01 tested these changes before the
line-search correction below. Six laps, scored mean 5.7028 s, SD 0.0810 s,
publication maximum 18.02 ms; all commands full-horizon valid. No measured
lap-time improvement versus 5.6952 s is claimed from refinement alone.

Further diagnosis: 249/1418 published cycles of the 5.6952 s run detected
merit regression, then accepted alpha=0.5 even though exact cost still
exceeded nominal. The backtrack loop checked feasibility, not descent.
It now requires decreasing exact cost when nominal itself is valid; otherwise
the independently validated alpha=0 sequence remains available. Helpers and
source-rollout tests pass. Trial live_runs/mpc_merit_sqp_20261027_01 now tests
the complete correction with original production weights.

Result: six complete laps, no collision or recovery command. Five scored:
5.6008, 5.5208, 5.5688, 5.6078, 5.6168 s. Mean 5.5830 s, SD 0.0351 s.
Publication median/p95/max 5.19/17.20/18.89 ms, zero budget violations.
All 1,386 issued commands passed full-horizon validation; zero published
line-search actions increased exact cost above their valid nominal.
The original weights are unchanged. The geometry + SQP/merit corrections
improve the saved 5.7790 s mean by 0.1960 s (3.39%); 5.5 s remains open.

Candidate-only next experiment: symmetric speed cost 25/25, no curvature
overspeed multiplier, steering/throttle slew 1500/25. This tests removing
the speed-cost branch inconsistency, not arbitrary extra throttle forcing.

Symmetric trial live_runs/mpc_merit_symmetric_20261027_01 completed six laps.
Scored 5.5278, 5.5818, 5.5008, 5.5218, 5.7008 s: mean 5.5666 s, SD
0.0722 s. A startup cycle deferred publication at 20.53 ms; no invalid command
was published. This is not a timing-qualified promoted profile. The startup
Jacobian took 8.78 ms and exact rollout 5.08 ms; host compilation briefly
overlapped early startup before being pinned off the controller CPUs.
All later tests must avoid host compilation during live operation.

The trial also wasted 92 second-QP builds with no time left for ADMM.
Refinement now requires its measured nominal/Jacobian preparation cost,
1 ms solver work and the existing validation reserve before it starts.
Next coupled candidate: speed 50/50, steering slew 1000, throttle slew 25.
The 25/25 result justifies testing stronger speed tracking with monotone
acceptance; neither profile is promoted without measured transfer.

Fixed-command replay of the 5.5830 s run: 750 ms speed RMSE 0.00783 m/s,
lateral speed 0.00109 m/s, yaw rate 0.00819 rad/s, yaw 0.00231 rad,
XY 0.00989 m. Maximum XY 0.0603 m; 2/1361 windows exceed 5 cm. Low average
error does not erase these remaining tails, and is not a 1:1 guarantee.
All ten package CTest targets pass after the SQP/merit correction.

The 50/50 + steering slew 1000 trial aborted before lap one: 28 consecutive
certified-plan actions exhausted the plan and both fresh/retained trajectories
failed the corridor. No collision; no time claim. This is not promoted.
Run: live_runs/mpc_speed_tracking_20261027_01. Max publication 18.31 ms.

Retaining steering/throttle slew 2000/75, speed weights 15/40 and overspeed
curvature gain 0.5 completed six laps in mpc_speed_balance_20261027_01.
Scored 5.4808, 5.5988, 5.5418, 5.5528, 5.5608 s: mean 5.5470 s, SD
0.0383 s. All 1376 published commands full-horizon valid, no recovery.
Publication median/p95/max 5.01/16.77/19.05 ms, no budget violations.
Snapshot saved in that run's mpc_overlay.yaml before testing 25/25 speed
weights with the same 2000/75 damping and curvature gain 0.5.
