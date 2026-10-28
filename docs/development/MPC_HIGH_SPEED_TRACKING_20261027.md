# High speed MPC tracking — 2026-10-27

Goal: match the selected raceline's theoretical time, using its requested
speed and competition runtime inputs. Average <=5.5 s is an achieved milestone,
not the final target (user clarified on Oct 27). Current v337 target: 5.1465 s.

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

## Initial investigation

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

Run mpc_speed_balance_20261027_02 (25/25 speed, curvature gain 0.5) completed
six laps with no recovery. Correct scored mean 5.5214 s, SD 0.0189 s:
5.4948, 5.5028, 5.5338, 5.5338, 5.5418 s. Publication maximum 18.59 ms.

Fixed a report bug discovered here: value-change deduplication silently
discarded one of two identical consecutive quantized lap durations. Reports
now use rising lap-count transitions and the corresponding duration topic;
fault cutoffs still apply, stale durations/counter resets are rejected.
13 report tests pass, including identical-lap and cutoff regressions. When
lap counters are absent, reports explicitly label duration-change-only data.

The next controlled profile changes only speed weights to 35/20, keeping
steering/throttle damping and physical constraints unchanged.

The 35/20 trial mpc_speed_balance_20261027_03 aborted at the first hairpin,
with corridor failure at stage 2; not promoted. The 25/25 run is the current
best qualified profile (mean 5.5214 s, not yet <=5.5).

Scored-only diagnosis of the 25/25 run (1131 issued samples after first lap):
speed error bias -0.00175 m/s, RMSE 0.0811 m/s; actual/reference mean speed
5.11785/5.11960 m/s. Cross-track bias +0.0369 m, RMSE 0.0601 m; heading
RMSE 0.0415 rad. Positive lateral error is outward on the clockwise corners,
so hitting body speed alone does not yield the nominal line's lap time.
Next trial keeps 25/25 and 2000/75 actuator damping but doubles cross-track
and heading weights to 8000/3000, targeting that measured path-length loss.

Cross-check over 1103 samples strictly between the first and sixth lap-count
transitions: using ds/dt=(u*cos(e_psi)-v*sin(e_psi))/(1-kappa*e_y), the
approximate per-lap time loss is speed 0.00061 s, outward offset 0.14008 s,
heading 0.00465 s and lateral velocity -0.00042 s; total 0.14491 s versus
the measured ~0.154 s gap to the nominal 5.367 s line. This is an attribution
using the path-coordinate metric, not an independent simulator measurement.
It explains why more throttle alone was not closing this final gap.

Doubling lateral/heading costs (mpc_path_tracking_20261027_01) did not help:
scored mean 5.5400 s, SD 0.0541 s, max publication 18.63 ms. The post-first-
lap cross-track RMSE rose to about 6.6 cm instead of falling. Not promoted.

Promoted the measured-best 25/25 speed weights, curvature gain 0.5, original
4000/1500 path weights and 2000/75 actuator damping to mpc_competition.yaml.
Removed the five temporary candidate files created during this investigation;
exact overlays remain with their run evidence, not in production config.
Existing unrelated worktree edits and earlier profiles were not overwritten.

The corrected controller now tests the v337 5.1465 s target again, with the
new SQP/merit implementation and best qualified weights:
live_runs/mpc_v337_merit_20261027_01. The default reference CSV is unchanged.

## Goal reached on the faster v337 target

First full-speed run, mpc_v337_merit_20261027_01: five scored laps
5.4118, 5.3858, 5.4218, 5.4318, 5.3678 s. Mean 5.4038 s, SD 0.0236 s.
Publication median/p95/max 5.88/16.88/18.55 ms; no >=20 ms cycles. One
exact-revalidated prior-plan action; all other published actions full-horizon
valid. Six complete laps, no collision.

Unchanged repeat mpc_v337_merit_20261027_02: five scored laps
5.3608, 5.4218, 5.3968, 5.4698, 5.3668 s. Mean 5.4032 s, SD 0.0399 s.
Publication median/p95/max 5.74/17.06/18.64 ms; no >=20 ms cycles. All 1343
issued actions full-horizon valid, no recovery. Six complete laps, no collision.
All ten scored laps across the two independent trials are below 5.5 s.

Saved the numerically identical v337 CSV in the planning package as
trajectories/autodrive_practice_20260924_b/ipopt_v337_raceline.csv (LF-normalized
copy; canonical SHA256 036fd894383aee11674f77db945911ee21b1940dd2701a78c4c50226ec8d372a).
The original solver output used CRLF and SHA256 1bd0b8470128853c86eaacd6984f5aff38a4cc3007d4c4cba727c49bc98b63d5;
all parsed rows match. Controller loads 474 points, 27.41 m; nominal time
5.1465 s, not a claim that the controller has reached that time.

After these two online passes, selected v337 as the default launch/trial
raceline. The previous reference CSV files and runtime physics are untouched.
Final check: mpc_v337_production_20261027_01, one warmup plus ten scored laps,
using production weights without a trial overlay.

Reproduction:
`SDU_APEX_TRIAL_TOTAL_LAPS=11 tools/run_mpc_practice_trial.sh <new-run-id>`
Reports are made with tools/export_mpc_run_report.py; count lap completions
using lap-counter transitions and exclude the first warmup lap.

Physics qualification on mpc_speed_balance_20261027_02: 750 ms fixed-command
XY RMSE 1.03 cm, speed 0.0103 m/s, yaw-rate 0.00964 rad/s. Raw maximum path
error is 8.82 cm; 31/1348 windows exceed 5 cm. All 31 overlap an independently
flagged GT pose/twist closure inconsistency (>5 cm in one 25 ms interval);
zero of the 1317 remaining windows exceed 5 cm. This diagnostic screen is
reported alongside the unscreened result, not used to hide failures or claim
an exact native-Unity replica. Further model claims must preserve that caveat.

## Longer-run failure and retained-suffix correction

mpc_v337_production_20261027_01 stopped after four laps, no collision. A chain
of 29 prior-plan actions ended in a nominal-build failure, before ADMM ran.
The terminal state was ~7.15 m/s, e_y=-0.200 m, e_psi=-0.250 rad. This means
the two five-lap successes did not establish sustained ten-lap reliability.

Code diagnosis: prior-plan warm starts padded the finite remaining suffix to
30 by repeating its last command. Rejections progressively flattened the
entire horizon to stale steering/torque. Candidate-prefix stitching had the
same padding; the deadline-specific reseed also skipped the next command by
starting at cursor+1 before the nominal builder shifted again.

Implemented a shared suffix copier with the true remaining length and the
already-issued action at index zero. Removed fabricated padding in direct
prior recovery, quality-selected recovery, candidate stitching and deadline
reseed. Missing nominal tail steering is initialized from future path
curvature, carrying the previous positive throttle only as an initial guess;
this is not published fallback control and must still pass QP/exact validation.
No physics equations or optimizer dimensions changed. New tests verify exact
indexing/counts, expired suffixes and path-based tail initialization. Nominal
failures now report the failing stage instead of -1.

Testing the correction with unchanged best weights/line, one warmup plus ten
scored laps: live_runs/mpc_v337_suffix_fix_20261027_01.

Result: eleven complete laps, no collision, all 2429 issued commands accepted
through full-horizon validation, no recovery. Ten scored laps:
5.4538, 5.3698, 5.4378, 5.3888, 5.4608, 5.4488, 5.4328, 5.3538,
5.3508, 5.3958 s. Mean 5.4093 s, SD 0.0403 s; max lap 5.4608 s.
Publication median/p95/max 5.93/17.14/18.71 ms, zero >=20 ms cycles.
Fixed-command 750 ms replay: 2405/2405 windows stay below 5 cm at every
sample, no GT-closure exclusions; endpoint XY RMSE 0.00981 m, max path
0.0444 m. Speed/lateral/yaw-rate RMSE 0.00841/0.00130 m/s / 0.00891 rad/s.
Production odom speed/lateral RMSE 0.0083/0.0011 m/s; gyro yaw-rate error zero
against the simulator's same-axis truth. Eleven-lap uncorrected odom drift
is not zero (relative XY endpoint 6.1 cm, maximum 9.8 cm); AMCL remains active.
All ten package test targets and 13 report tests pass.

Unchanged long repeat mpc_v337_suffix_fix_20261027_02: eleven complete laps,
no collision/recovery, all 2425 issued actions full-horizon valid. Ten scored
mean 5.4153 s, SD 0.0384 s, maximum lap 5.4618 s. Publication maximum
19.22 ms; none >=20 ms. Thus twenty scored laps with this final code were
all below 5.5 s. Repeat rollout evaluation is saved alongside this run.

## Continuing toward theoretical time and IPOPT parity

The 5.5 s milestone is not completion of the revised goal. Current measured
gap to v337 is ~0.263 s/lap. A post-warmup path-coordinate estimate attributes
~0.080 s/lap to speed tracking, 0.159 s to outward offset, 0.006 s to heading.
Corresponding speed/cross-track/heading RMSE: 0.131 m/s, 0.0773 m, 0.0498 rad.

Both existing IPOPT jobs remain active (PIDs 55467/55468, CPUs14/15). They
have emitted native iteration messages around 3636–3656, so they are advancing,
not merely consuming CPU. They were not stopped or modified.

Audit found the same *class* of geometric inconsistency in IPOPT: independent
linear interpolation of C(s), sin/cos heading, and curvature assumes a unit
normal, C'(s)=T(s), and heading'=curvature. On its actual mesh, max discrepancy
between the heading derivative and interpolated curvature is 0.11439 /m
(RMS 0.00567 /m); normal-length error up to 0.000368. These assumptions need
correction in the symbolic path equations and numeric export. Running jobs
retain their already-constructed graphs; any fix applies to new content-
addressed solver instances, and their old-model outputs must be revalidated.
No claim that the IPOPT model is already equivalent to the corrected MPC.

## Continue toward a mean below 4.9 s

Both v338 processes finished. Wide trust selected the original 5.146502 s
seed, not a faster solution (its time-stage final iterate was 5.290012 s).
Factor 1024 retained a 4.915787 s iterate with maximum scaled constraint
violation 0.00028627, then failed the exporter's *coarse* swept-clearance
certificate. This was not evidence of wall penetration: on interval 114–115,
three subdivisions yielded +4.24 mm sampled residual clearance above the
requested 5 cm, minus a 9.91 mm unsampled bound. Twelve subdivisions certify
+1.76 mm conservative residual clearance. The exporter now adaptively refines
inconclusive intervals and still rejects measured or unresolved violations.
The requested wall margin is unchanged. Tests cover refinement and rejection.

Implemented `path_frame.py` for IPOPT. It uses linear XY and unwrapped
heading exactly as production MPC. The spatial dynamics now include the
actual chord tangent metric, chord normal drift and heading derivative.
Node/midpoint kinematics, numeric geometry and native-witness time scheduling
use this definition. Vehicle force equations are unchanged. Independent
finite-difference world-motion/seam tests and Phase-I tests pass.

Two new solves use separate content-addressed graphs and print level 5:

- `optimized_practice_v339_exact_frame_20261027`: v337 checkpoint, L-BFGS,
  CPU 14, 500 feasibility iterations, then 1000 per trust factor
  16/64/256/1024. Initial corrected-equation defect 0.0203852, so the old
  seed is correctly rejected for immediate reuse and must pass Phase I.
- `optimized_practice_v339_fast_full_state_20261027`: complete saved
  4.915787 s differential state plus controls, reprojected algebraic seed,
  exact Hessian, CPU 15, 500/1200 iterations, trust 64/256/1024. It remains
  explicitly uncertified until Phase I passes. Added full-state reseeding
  to avoid mixing a fast path with the previous seed's wheel/body states.
  The initial exact-Hessian launch exposed a previously hardcoded L-BFGS
  options matcher; corrected it to preserve either selected Hessian method.

MPC investigation confirmed that the prior LTV model dropped changes in
progress between stages even though exact nonlinear validation resampled
the path at actual progress. Implemented one deterministic progress-deviation
coordinate, analytically propagated through the same loaded-path transform.
It has no independent initial value or tracking weight. The plant remains
20 physical coordinates; the QP now has those plus progress and two input
history coordinates (23). No physical source-plant calls were added.
Updated all solver scaling layouts. Independent six-stage derivative checks
and all ten CTest targets pass. Supervised trial:
`live_runs/mpc_v337_progress_coupling_20261027_01`.
This change has not yet demonstrated a live lap-time improvement.

The progress-coupling trial completed six laps: five scored mean 5.4390 s,
full-horizon accepts 1352/1352, no recovery/collision, max publication
18.538 ms. Ey 6000 / steering slew 1000 then completed six laps at 5.4404 s,
max 18.641 ms; not promoted. Both are slower than the prior 5.409/5.415 s
ten-lap results. The coordinate derivative correction is independently
verified, but no lap-time benefit is claimed from it alone.

## Position-origin audit and current optimizer

Verified directly from the exported **practice player** scene:
GPS component 2117 uses Transform 536, `Rear Axle Center`, at root z=-0.16 m.
LiDAR Head 517 -> LIDAR 561 at root z=+0.1133 -> Sensors 552 -> Roboracer 550.
Therefore the configured rear-axle-to-LiDAR offset +0.2733 m is correct.
Do not change it to +0.1133 m; that is the root-origin coordinate.

Found a genuine IPOPT origin mismatch: spatial position propagated COM
velocity, whereas runtime trajectory positions / GPS are rear axle. The OCP
now transports velocity using omega cross [-0.15532,0,0] before horizontal
projection (v_rear=v_com-r*L, w_rear=w_com+q*L). Tire forces still use COM
velocities. The lever is checked against production kComForwardFromRearAxleM.
Map exporter footprint bounds were also root-based [-0.24,+0.27]; corrected
to rear-based [-0.08,+0.43], matching MPC. Earlier theoretical time/clearance
reports are not recertified automatically by these corrections.

Fixed-mesh implementation: the sole drive phase is a known-length parameter,
not an optimization variable. The first v340 launch exposed a primal-cache
count assumption; removed the parameter from flat decision slices while
preserving it in report/checkpoint snapshots. All relevant tests pass.

Current run: `optimized_practice_v341_rear_axle_frame_20261027`, console
`SDU_Apex_IPOPT_PhysX_MinTime_v2/v341_rear_axle_frame_console.log`.
Seed converts the complete saved v338 COM state to the rear-axle path and
resamples all physical states/controls at the projected stations. This is a
source-derived coordinate conversion, not a fitted correction. Initial
defect 0.173157 requires Phase I; no source-feasibility claim. 8532 decision
variables / 25122 constraints. CPU 14, Phase I 300 iterations, then 1200 per
trust factor 16/64/256/1024. Wall margin remains 5 cm.

The two v339 COM-origin processes were retired after this source audit.
Their SIGINT was ignored; SIGTERM stopped only PIDs 374375 and 379572.
Logs/provenance/seed files remain. No useful faster candidate from those
graphs is claimed; they must not compete with or replace the corrected run.

## Localization contribution and current live trial

On the prior ten-lap run, current-map-pose error after warmup was mainly
longitudinal: -0.0594 m bias / 0.0916 m RMS, lateral -0.0050 / 0.0267 m.
The apparent 10.5 ms lag is a diagnostic, not a known sensor delay or an
implemented time shift. Old full-data captures contain 40 Hz LiDAR: 10/1581
duplicate scans; lack of scans in the recent performance profile prevented
checking their freshness. Added raw LiDAR to that profile and sealed AMCL
config/snapshots alongside MPC config in future trial manifests.

Candidate legal-sensor fusion test: along-track AMCL correction gains 0.10
(ordinary and fast), with cross-track/yaw handling unchanged. Run
`mpc_v337_along_pose_gain010_20261027_01`: six laps, five scored mean
5.3906 s, all 1342 commands exact-full-horizon valid, no recovery/collision,
max 18.987 ms. Map-pose XY RMSE 0.093 -> 0.079 m, max 0.255 -> 0.197 m.
This is a first measured improvement, not yet repeated ten-lap evidence.

Next live candidate `mpc_v337_pose_speed_bundle_20261027_01` retains gain
0.10 and tests under/overspeed weights 35/20, throttle slew 50. These are
candidate values, not promoted production MPC weights. The final target
remains mean <4.9 s; neither the current reference nor achieved live laps
meet it yet.

## Ten-lap localization / speed-objective comparison

All rows use v337 geometry at original requested speeds. Warmup is excluded
from the averages; no reference-speed multiplier was used.

| Capture | Along-scan gain | Speed under/over; throttle slew | Scored laps | Mean | SD | Publication max |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mpc_v337_pose_speed_bundle_20261027_01` | 0.10 | 35/20; 50 | 5 | 5.3656 s | see report | 19.287 ms |
| `mpc_v337_pose_speed_bundle_20261027_02` | 0.10 | 35/20; 50 | 10 | 5.3924 s | see report | 18.609 ms |
| `mpc_v337_cross_track_scan_20261027_01` | 0.00 | 35/20; 50 | 5, then fault | 5.2410 s | not a completed trial | 18.306 ms |
| `mpc_v337_along_scan_gain003_20261027_01` | 0.03 | 35/20; 50 | 10 | 5.3589 s | 0.02734 s | 18.697 ms |

Zero along-track correction was **not** promoted: after six total laps it
failed the exact corridor check at stage zero. The supervisor stopped it
without collision. Its decreasing lap times are not evidence of consistency.
The 0.03 trial completed eleven total laps, all commands full-horizon valid,
no recovery or collision. Current-map-pose XY RMSE was 0.064 m, p95/max
0.121/0.157 m. Gain 0.10 repeat was 0.080 m RMSE, max 0.204 m. Gain 0.03
is still being repeated; no zero-error localization claim is made.

For speed-bundle 01, the recorded-command 750 ms replay (no future state
correction) gave forward/lateral/yaw-rate RMSE 0.01224 m/s / 0.00123 m/s /
0.00980 rad/s and endpoint XY RMSE 0.01166 m. Unscreened maximum path error
was 0.1254 m, with 33/1304 windows exceeding 5 cm somewhere. Independent
GT pose/twist closure identifies a common pose jump in 31 of these windows;
two clean failures remain. The raw result must not be replaced by the
closure-screened score or described as all windows within 5 cm.

## Peak-slip objective correction

Found a substantive QP/nonlinear-objective mismatch: the QP penalized
instantaneous wheel-only slip, whereas nonlinear acceptance penalized the
maximum slip excess during the incoming held-input interval. The former
omitted body-speed/contact-state and direct torque effects on that cost.

The existing forward-AD plant now returns the same four interval peaks and
their body/wheel/suspension/steering/throttle sensitivities. QP assembly uses
these to preserve the peak cost's first derivative. Its positive diagonal
Gauss-Newton Hessian is an optimization approximation, not a change to
vehicle dynamics; no extra physical rollout or optimizer state is required.
Removed the obsolete instantaneous wheel-only QP penalty. Retained peak
diagnostics and the exact nonlinear acceptance check.

Also corrected the terminal multiplier of the last command-feedforward
penalty after eliminating the logged command channel from the QP, and
recomputed yaw-rate targets after applying frozen speed targets in exact
candidate scoring. This aligns those terms with the optimization objective.

Independent finite differences verify peak derivatives for body speed, yaw,
wheel speed, rack, height, roll, steering input and throttle. A QP-gradient
test verifies the previously missing body-speed cost derivative. The finite
difference denominator uses the actual float perturbation, not a rounded
nominal epsilon. All ten CTest targets pass. Live full-speed repeat:
`mpc_v337_peak_slip_objective_20261027_01` (gain 0.03, same 35/20/50 profile).
No lap-time improvement from the new objective is claimed before scoring.

Peak-slip trial result: eleven total / ten scored laps, mean 5.3703 s,
population SD 0.03550 s, max publication 19.343 ms. It used 2398 full-horizon
commands, three retained prefixes and one candidate prefix. Raw ADMM
termination was optimal in 2277 cycles versus 2328 in the prior gain-0.03
trial. The objective correction is independently checked, but this first
trial is not faster and does not establish a convergence improvement.
Next controlled trial keeps the same profile and changes only the slip
weight 25 -> 5, because the corrected peak cost is not the old instantaneous
residual. This changes an objective weight, never tire physics or slip limits.

Development builds now discover only the five current runtime packages;
analysis/vendor setup.py files are no longer mistaken for ROS packages.
This does not remove any package or alter runtime topic policy.

## Corrected IPOPT feasibility progress

The v341 L-BFGS Phase I exhausted 300 iterations without certifying the
corrected seed. The parallel v342 exact-Hessian solve reached a fully
feasible iterate at iteration 20: maximum constraint violation 1.6688e-7.
It retained a later, lower seed-displacement feasible checkpoint (iteration
30, maximum 0.000273317), and is continuing its configured Phase-I/time
stages. These are feasibility results, not a new faster exported raceline.
Do not replace the running graph or treat an infeasible raw time objective
as usable. Console: `SDU_Apex_IPOPT_PhysX_MinTime_v2/v342_rear_axle_exact_console.log`.

Added independent circular COM-to-rear-axle/seam conversion tests; 26 focused
IPOPT tests pass. Fixed resume provenance to include imported frame helper
equations, not merely the byte-identical generated importer. Existing running
processes retain their sealed graph; this strengthens subsequent resumes.

## Overnight result recovery and restart — 2026-10-28

The v343 run completed and exported its retained Phase-I feasible point:
predicted lap 4.941441705748207 s, maximum scaled constraint violation
6.182244142493784e-5 against the 0.001 acceptance limit. This is not a
stationary minimum-time solution or a Unity/MPC-validated lap.

| Time trust factor | Final objective | Maximum violation | Result |
| --- | ---: | ---: | --- |
| 64 | 4.923914 s | 0.014697 | 1200 iterations; infeasible |
| 256 | 4.922632 s | 0.007334 | 1200 iterations; infeasible |
| 1024 | 5.704074 s | 0.431163 | 1200 iterations; infeasible |

Recovered files, all under `optimized_practice_v343_certified_restart_20261027/`:

- `optimizer_target_raceline_pass_00_not_mpc_validated.csv`: 474 finite
  nine-column rows; SHA256
  `9fcda13ff57e0ed9fac968519fe9bfaf19f226be3130fdd9e4d075f1a12bd534`.
- `ipopt_candidate_pass_00/optimizer_seed.npz`: complete exported state,
  controls and algebraic quantities; all numeric entries finite; SHA256
  `8d814076e65c407a49cd46c0877eb29dbace0a4ef092c878e386eec70e49776a`.
- `ipopt_report_pass_00.json`, `report.json`, source witness state/midpoint
  CSVs, and all three raw time-stage checkpoints remain available. The export
  report records the map footprint check and geometry/curvature consistency
  check as passed. Native witness replay was skipped.

The initial v344 handoff did not start overnight. The subsequent v345 launch
had valid CPU/wall caps, but unnecessarily reprojected a raw-derived seed;
the initializer then recomputed throttle and raised its initial residual to
0.09423. It was stopped before any minimum-time stage. Preserve that log for
provenance; it is not the active run.

The bounded v346 verification run used v343's
complete exported checkpoint directly, without forced reprojection or
changes to its saved states and controls. Map, reference mesh and model
provenance checks remain enforced, followed by an independent source
constraint evaluation before checkpoint reuse. Its actual startup log
confirmed maximum residual 6.182244142708196e-5 and `reuse=True`; Phase I was
correctly skipped and a complete feasible checkpoint was saved before time
optimization. That run was stopped when the user replaced the four-hour
shutdown plan with an unrestricted all-day run.

`tools/run_ipopt_unattended.sh` passes the CPU limits as explicit CLI options
because the main CLI clears environment-only CPU limits when its arguments
remain zero. Budgets: Phase I 600 CPU seconds; each time stage 2400 CPU / 2700
wall seconds; outer process guard 11700 wall seconds; service guard 11790
seconds. Factors remain 64, 256, 1024, maximum 1200 iterations each. Native
IPOPT limits return through normal checkpoint/result export; the outer guard
preserves earlier completed checkpoints but cannot guarantee a final export
from an interrupted stage.

The wrapper now records the seed path/hash in `status.json` and writes an
interrupted status on TERM/INT. Run output and logs persist independently of
Codex. No new trajectory is automatically promoted to the production MPC.
Inspect `console.log`, `status.json`, `time_stage_iterates/`, and final
`report.json` in the active output directory after completion. A wrapper `finished`
status alone does not establish a new feasible or Unity-validated raceline.

### Active all-day run

User service: `sdu-apex-ipopt-all-day-20261028.service`.
Output: `optimized_practice_v347_all_day_20261028/`.
Launcher: `tools/run_ipopt_unattended.sh OUTPUT_DIR SEED.npz --all-day`.
Input: v343's complete exported `ipopt_candidate_pass_00/optimizer_seed.npz`,
with the checksum recorded above. An exact launcher copy is saved as
`launcher.sh` inside the output directory.

All CPU, stage-wall, outer-wall and service runtime limits are disabled.
Time stages each allow 6000 iterations, with factors 2, 4, 16, 64, 256, 1024.
Factors were checked using the actual CLI parser: the first factor must be
strictly greater than one. A narrow initial trust region starts near the
retained feasible 4.9414 s solution before widening, rather than jumping
immediately to factor 64. The physics, constraints and acceptance tolerance
are unchanged. Stage completion saves its raw iterate independently; the
selected feasible result is exported after continuation finishes. There is
no automatic endless relaunch loop or automatic production promotion.

User lingering is enabled so the service survives logout. A companion
`sdu-apex-ipopt-awake-20261028.service` blocks idle/sleep while the optimizer
service lives and releases that block when the optimizer stops. No scheduled
system shutdown was present at launch. The PC remains user-controlled.

Inspect progress:

```bash
python3 tools/watch_ipopt.py optimized_practice_v347_all_day_20261028 --unit sdu-apex-ipopt-all-day-20261028.service
tail -f optimized_practice_v347_all_day_20261028/console.log
systemctl --user status sdu-apex-ipopt-all-day-20261028.service
```

The read-only dashboard refreshes every two seconds, following the log
incrementally and showing the current trust stage, iteration/objective,
primal/dual residuals, actual service status and latest checkpoint on disk.
Ctrl+C exits only the viewer. `--once` prints one snapshot; this was checked
against the active run (stage 1/6, iteration 223, one saved checkpoint).

Last night's recovered CSV and complete checkpoint remain available under
v343 even if any subsequent stage fails or is interrupted. A faster objective
with violation above 0.001 remains rejected; more iterations are not evidence
that a result has converged or that Unity can drive it.

Startup verification completed: the actual all-day process independently
accepted the unchanged seed at residual 6.182244142708196e-5, skipped Phase I,
saved its feasible initializer and entered the exact-Hessian time solve at
factor 2. At iteration 17 its objective was 4.9400668 s and primal violation
7.20e-4; dual residual was 1.46e-2, so it was not yet converged. Approximately
12 GiB RSS, one active CPU core and no swap usage were observed. The service
has `RuntimeMaxUSec=infinity`; its companion sleep inhibitor was visible and
active. Only this one IPOPT process was running. Subsequent progress belongs
to its durable console and stage reports rather than this startup snapshot.
