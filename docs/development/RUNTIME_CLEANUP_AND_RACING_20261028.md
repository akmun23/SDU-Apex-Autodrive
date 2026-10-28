# Runtime cleanup and racing improvements — 2026-10-28

Target: ten consecutive scored laps below 49.8 s total, excluding warmup.
IPOPT all-day service v347 remains running on CPU 15; do not restart it for
runtime development. Runtime trials use the stock practice simulator, allowed
sensor topics only, with ground truth reserved for recording/scoring.

## Baseline and evidence

- Production repeat `mpc_v337_production_peak_slip_20261027_01`: eleven total
  laps, ten scored averaging 5.3654 s (53.654 s total), no collision; maximum
  publication time 19.111 ms. Target v337 integral is 5.1465 s.
- Recorded production odometry errors: forward velocity RMSE 0.0088 m/s,
  lateral velocity 0.0011 m/s, yaw rate effectively zero against gyro/truth;
  relative endpoint drift 0.134 m over the complete eleven-lap capture.
- v343 supplies a source-OCP-feasible 4.9414417 s target and complete saved
  seed. It has not been driven/validated in Unity/MPC. Do not claim that target
  as an achieved lap time.
- Last night's lower-damping `mpc_v337_local_correction_authority_20261027_01`
  completed eleven total laps; its full report is now being generated.

## Work queue

1. Audit active build/launch paths; remove confirmed dead code and redundant
   hot-path work, preserving unrelated pre-existing worktree edits.
2. Fix source-time AMCL matching: endpoint clamping currently makes the
   existing future-scan queue unreachable. Existing two production captures
   have zero measured scan/odom epoch offsets, so this is a jitter-path
   correctness fix, not an established explanation for their pose error.
3. Review 3D velocity-to-world pose projection and 25 ms integration in odom;
   compare with source plant's full attitude kinematics before changing it.
4. Remove redundant CUDA synchronization only when same-stream ordering
   proves equivalence. Keep synchronization before host reads.
5. Check CPU tests, allowed-topic guard, GPU build/runtime, replay accuracy
   and solver timings. Re-score the existing candidate before further tuning.
6. Run supervised practice comparisons, then test faster feasible targets.
   Record full state/command histories, stop immediately on collision, score
   ten laps after warmup, and promote only measured improvements.

## Findings and implemented corrections

- `physx_glibc_compat.cpp` is not built by current odom. Its remaining offline
  native validator dependency was found and retained under
  `tools/native_physx_support/`, with the validator/build paths updated. The
  compatibility code is removed from the production localization package.
- `Resampler::resample()` and its extra device particle buffer are unused;
  the live filter exclusively calls `resample_to()` with double buffering.
  `resample_to()` also blocked the host despite its same-stream API contract.
  Removed that unused API/buffer and redundant same-stream synchronization;
  removed two unused CPU estimate vectors. Host-read synchronization remains.
- Odom observer assigns corrected rear-wheel values twice and retains
  conditionals that cannot be reached after its mandatory source-state gate.
  Removed these, their unused history fields and a redundant pose-origin flag.
- Odom world-position integration still treats COM velocities and body yaw
  rate as planar quantities despite available roll/pitch/heave state. Any
  correction was evaluated on recorded data: attitude projection reduced
  aligned-replay position RMSE 0.08482 -> 0.06771 m, endpoint 0.11753 ->
  0.09975 m, without fitting. Implemented full body-to-horizontal rear-axle
  projection with cached previous projected velocity; body velocity outputs
  and measured IMU yaw are unchanged.

## Verification and live results

- Source-time history uses a shared binary bracket lookup instead of a
  linear scan and endpoint substitution. Added tests for future/past/exact,
  adjacent and empty/single-sample histories. Deferred duplicates cannot
  propagate the filter backwards. Removed unused `max_scan_age` parameter;
  structurally valid scans remain governed by source-time history.
- Three focused CPU CTest targets pass: history, observer and source runtime.
  New independent rigid-transform test checks tilt projection and preservation
  of body velocities. Runtime allowed-topic guard passes. GPU localization
  and actual practice launch/build were exercised successfully.
- `mpc_v337_runtime_cleanup_20261028_01`: 11 total / 10 scored laps,
  53.646999 s total, mean 5.364700 s, SD 0.024139 s, no collision. Production
  odom endpoint drift 0.064 m (old repeat 0.134 m), max 0.092 m; AMCL map-pose
  RMSE 0.070 m (old 0.079 m). Recorded u/v/r RMSE 0.0060 / 0.0010 / 0.0000.
  These live repeats differ in motion; do not attribute the velocity error
  change to the projection, which does not change velocity outputs.
- MPC now owns its control callback group on a dedicated single-thread
  executor. The four-thread ROS container keeps sensor callbacks; one
  persistent control owner reuses eight OpenMP workers. Destructor cancels
  timers/executor and joins before destroying solver state.
- `mpc_v337_persistent_control_20261028_01`: 11 total / 10 scored, 53.565001 s
  total, mean 5.356500 s, SD 0.039968 s, no collision. Observed Jacobian worker
  IDs 32 -> 8; max Jacobian wall 6.953 -> 5.996 ms. Full publication median /
  p95 / max 8.873 / 17.493 / 19.439 ms (old 7.902 / 17.074 / 19.111), so no
  full-cycle timing improvement is established from this repeat.
- Removed 14 obsolete MPC experiment overlays; production profile, one active
  candidate and dedicated accuracy/open-loop test profiles remain. Historical
  live-run parameter copies remain attached to their measurements. Deleted
  tracked overlays are recoverable through Git; untracked overlays were
  discarded as requested.

## Faster target and current next trial

`mpc_v343_fast_target_20261028_01` used the recovered v343 4.941442 s target
with actual MPC. Warmup 5.918800 s; scored laps 5.134800, 5.129800, 5.185800 s.
It stopped without collision on an exact corridor failure after four total
laps. At turn exit, e_y grew about 0.067 -> 0.120 m while full plans were
still accepted, then roughly 25 retained-plan actions exhausted the useful
suffix before the corridor fault. This is faster, not yet consistent.

Active candidate `mpc_sqp_tracking_candidate.yaml` now tests a coordinated
tracking/authority change: ey/epsi/u weights 10000/3000/80, steering/throttle
slew 500/20, SQP radii 0.06 rad / 0.12 throttle, ADMM cap 250. Source physics,
positive-drive envelope, two SQP passes and 20 ms total budget remain in
force. `mpc_v343_tracking_bundle_20261028_01` stopped after warmup on another
exact corridor failure; the larger weight bundle was not promoted. The
ten-lap 49.8 s target has not been achieved.

## Corridor geometry correction

The old controller used `half_width*abs(cos(epsi)) + front*abs(sin(epsi))`
on BOTH sides of the rear-axle origin. The actual rectangle reaches 0.430 m
forward and only 0.080 m backward; that symmetric bound invents
`0.350*abs(sin(epsi))` metres of vehicle on one side (4.4 cm at 0.126 rad).

Implemented signed support of all four actual body corners, separately for
left and right, in QP bounds, exact nonlinear acceptance and reported prefix
slack. Retained configured 5 cm clearance and 5 mm numerical tolerance.
Removed duplicated trigonometry/corridor checking in nonlinear scoring.
Independent corner-transform tests cover both turn directions and backwards
orientations; all ten MPC CTest targets pass. The old 0.150 m *default*
clearance constant was misleadingly named a hard minimum: production already
overrode it to 0.05 m. Renamed/defaulted it consistently to 0.05 m.

The plotting tool also retained actor-centred anchors [-0.240, 0.270] despite
plotting rear-axle CSV positions. It now reads [-0.080, 0.430] and width 0.273
from the production header, preventing that 16 cm footprint-origin mistake.

Verification: `mpc_v343_asymmetric_footprint_20261028_01`, using the
unchanged production weight profile, new corridor geometry and recovered
4.9414 s target. No physical tire/body equation or IPOPT graph was altered.
It stopped after two total laps, no collision. The first rejected prediction
was only 0.8 mm outside its tolerance-adjusted corridor; that reference leaves
about 1.35 cm of lateral tracking room with the aligned actual footprint and
5 cm clearance. Correct footprint geometry does not solve tracking by itself.

## Wall approach and further GPU cleanup

- Added an optional squared-hinge objective on remaining physical rectangle
  clearance. Its reserve does not change the hard margin or tolerance. Exact
  nonlinear scoring, stage/terminal QP gradients and scalar scoring share the
  same implementation; diagonal positive Gauss-Newton curvature adds no new
  states. Independent finite-difference/mirrored geometry and QP/nonlinear
  consistency tests pass; all ten MPC CTest targets and 21 audit tests pass.
- `mpc_v343_corridor_reserve_20261028_01` tested weight 20000/reserve 0.05 m
  with baseline tracking/damping. It failed before warmup: one 75.5 ms cycle
  was followed by a long retained-plan chain and growing lateral error.
  This trial does not establish a wall-cost gain and is NOT promoted.
- Fixed controller default weights/trust radii/startup seed that contradicted
  the comment promising equivalence with the production YAML. Missing config
  must not silently switch to the old objective.
- Removed the failed, disabled along-track likelihood correction completely:
  header, runtime branch, parameter, unused diagnostic fields. It previously
  accumulated metre-scale bias. Retained the eleven active source-alignment
  fields; updated the bag evaluator to accept that active layout.
- Replaced four per-update profiling-only CUDA stream waits (predict, scan
  upload, sensor update, normalization) with reusable stream timing events.
  The confidence host-read boundary remains; when confidence is disabled,
  explicitly await normalization before reusing pinned upload storage. Kernel
  order and particle/filter math are unchanged. Event times measure GPU work,
  not the former host wait duration; do not compare those columns as identical
  metrics. Full callback wall time remains comparable.

Current verification is `mpc_v337_gpu_stream_cleanup_20261028_01`, reference
weights, no wall-cost overlay, eleven total laps requested. Its production
build/live accuracy and timing must pass before calling GPU changes verified.

Result: eleven total laps, ten scored in 53.821 s, no collision. Mean 5.3821 s,
SD 0.04558 s. One 21.899 ms cycle was deferred; max is not a clean <20 ms pass.
AMCL steady PF wall median/p95 2.858/3.521 ms versus 2.918/3.550 before stream
events. Map-pose RMSE 0.071 m versus 0.070; no accuracy improvement is claimed
from GPU ordering alone. After warmup, odom u/v/r RMSE .0115/.0018/0, with
u max .2524 m/s (tail remains to diagnose). Relative pose endpoint/max were
.019/.045 m. Map-pose errors are mostly along-track: bias -.0543 m, RMSE
.0686 m; cross-track RMSE .0264 m, max .0559 m.

The lighter soft-wall trial `mpc_v343_soft_wall_20261028_01` also failed before
warmup. Removed the entire unsuccessful wall-cost implementation, tests and
configuration rather than retaining a disabled experimental runtime path.
Archived run config/diagnostics remain the evidence for the failed attempts.

## Direct pace objective and current verification

Added optional terminal progress reward using the existing propagated station
deviation: `-weight_progress*(s_N-s_0)`. It adds no states, Hessian terms or
physical approximation. QP terminal linear cost and exact nonlinear ranking
use the same objective. Recovery improvement tests now support signed costs
without the old `prior_objective > 0` assumption. Added QP identity, nonlinear
reward, station-origin invariance and signed-selection tests; ten CTests pass.
Production weight remains zero. The 400-weight candidate failed after warmup
in `mpc_v337_progress_20261028_01`, no collision, max cycle 18.490 ms. It is
not promoted; a progress incentive alone does not repair failed QP tracking.

Further AMCL audit found genuinely discarded work: successful local scans
resample/KLD-count/roughen/inject the old cloud and immediately replace every
particle/weight with a new cloud at the fused pose. Moved the existing pose
gate before that decision (still called exactly once), and skip resampling
only when that same accepted scan will replace the entire local cloud.
Rejected/global hypotheses preserve their existing PF updates. Random draws
for discarded clouds disappear, so Monte Carlo repeatability changes even
though the replacement distribution is unchanged. Reinitialization also
reuses pinned storage and two same-stream uploads instead of per-scan pageable
vectors/two global copies; its completion boundary is retained.

Verifying this production change with reference weights and eleven total laps:
`mpc_v337_amcl_no_discarded_resample_20261028_01`. IPOPT v347 remains running.

Result: eleven total laps, no collision; ten scored in 53.701 s (mean 5.3701 s,
SD .03947). Max cycle 19.345 ms, no deadline deferrals. Current-map-pose RMSE
.067 m versus .071 in the stream-only repeat. The speed difference between
these repeats is small and does not establish a racing-speed gain.

## Fixed-QP matrix cache and tooling consistency

ADMM recomputed `P[k+1]*d[k]` at all thirty stages on every iteration even
though both terms are unchanged between penalty refactorizations. Cached it
only inside one ADMM QP and recompute after every refactorization. Public
quadratic-factor reuse still allows a new affine `d`; a regression test
changes `d` and compares against the independent unfactored pass.

The saved-state replay target exposed a stale hard-coded 22-state/29-stage
fixture ABI; production is 20 physical + progress + two previous commands,
with all thirty stages. Updated the frozen fixture schema to v2, 23/2/30, and
added a compiled C++ round-trip test so a Python-only schema test cannot mask
this mismatch again. Retired dimensions are rejected, not silently padded.
All eleven MPC CTest targets and 26 focused Python tests pass.

Benchmark corpus: 421 recorded racing states from
`mpc_v337_gpu_stream_cleanup_20261028_01/solver_inputs.csv`. Reference binary
uses the solver at HEAD `62a9eb84` and otherwise the same current core; no
stale executable was used. Reports and six frozen QPs are under that run's
`affine_cache_benchmark/`. Sequence replay accepted all 421 states for both;
ADMM median .722 -> .664 ms, p95 2.859 -> 2.801 ms. Floating-point grouping
changes warm-start/SQP branches across the sequence, so these are NOT identical
QP timing comparisons or proof of unchanged live control. On six identical
frozen QPs, statuses and iteration counts match; max complete-state numerical
delta across fixtures is .00459 (mixed units, predominantly internal states),
not a pose error. Source dynamics are unchanged.

Current live verification: `mpc_v337_affine_cache_20261028_01`, unchanged
production weights and geometry. Direct progress reward remains experimental,
not enabled in production. The 49.8 s ten-lap goal remains unmet.
