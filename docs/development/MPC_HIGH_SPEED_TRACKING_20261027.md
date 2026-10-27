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
