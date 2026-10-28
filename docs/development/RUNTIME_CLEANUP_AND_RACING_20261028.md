# Runtime cleanup and racing improvements — 2026-10-28

Target: ten consecutive scored laps below 49.8 s total, excluding warmup.
Latest MPC warm-start gate correction and its supervised tests are recorded in
[MPC repairable warm-start correction](MPC_REPAIRABLE_WARM_START_20261028.md).
The correction is implemented; the ten-lap time target remains unmet.
IPOPT v347 was interrupted by power loss during its second time stage. Optimized
recovery v354 runs on CPU 15; do not restart it for runtime development.
See [IPOPT recovery and performance](IPOPT_RECOVERY_AND_PERFORMANCE_20261028.md)
for the current service, checkpoints, correctness tests and optimization work.
Runtime trials use the stock practice simulator, allowed
sensor topics only, with ground truth reserved for recording/scoring.

## Baseline and evidence

### Power-loss recovery, 2026-10-28

- The old run **did advance** after its first time stage (trust factor 2).
  At iteration 6000 it saved a 4.9400679968 s iterate with maximum constraint
  violation 0.0003611644, below the 0.001 feasibility gate. IPOPT returned
  `Maximum_Iterations_Exceeded`; this is not a stationarity/convergence claim.
  The continuation code accepted that feasible restart and entered factor 4.
- Factor 4 reached iteration 1048: console objective 4.9388517 s, reported
  primal residual 0.000793. Power loss interrupted this stage. Its primal
  vectors were not saved, so the printed objective/residual cannot reconstruct
  that iterate. The previous complete factor-2 NPZ is intact, SHA256
  `bb9fc985db4ec3eacf0a2cb94fbcef4f011a7274a0839d0e39b3583b89c1418c`.
- Initial recovery directory: `optimized_practice_v348_power_recovery_20261028`.
  Transferred every archived differential state and both control arrays at
  fraction 1.0 with the existing provenance-checking seed converter; zero
  speed samples clipped. Algebraic quantities are recomputed by the current
  source equations, and the ordinary strict Phase-I gate must pass again.
  This is a primal reseed, **not** restoration of IPOPT duals, barrier state,
  or the interrupted iteration counter. The time-stage trust box is centered
  on the newly certified initializer, as in the existing restart formulation.
- Initially started `sdu-apex-ipopt-recovery-20261028.service`: factors
  **4,16,64,256,1024**, 6000 iterations each, exact Hessian, original 0.06 m
  mesh / 0.05 m margin / positive throttle bounds, no CPU/wall caps. No new
  simulation, no runtime trajectory promotion, no physics/model changes.
  Sleep inhibitor is tied to the service; a forced shutdown still stops it.
  Old directory and its console/checkpoints remain untouched.
- Launcher now accepts explicit remaining trust factors with fail-fast
  validation. Dashboard now labels a stale `running` status as interrupted
  when the named service is inactive, rather than displaying it as running.
  Three recovery-contract unit tests and three existing raw-seed tests pass;
  shell syntax check passes. Pytest plugin auto-loading was disabled because
  the host ROS launch-testing plugin lacks `lark` in the optimizer venv.
- Observe recovery:

  ```bash
  python3 tools/watch_ipopt.py optimized_practice_v348_power_recovery_20261028 --unit sdu-apex-ipopt-recovery-20261028.service
  ```

The initial v348 attempt exposed a throttle re-seeding bug and was superseded.
The linked IPOPT document describes its fix, the successful exact recheck, and
the v354 continuation with durable mid-stage checkpoints. This section records
the initial recovery investigation, not the latest active service.

**Follow-up work in progress:** same-epoch ADMM refinement fix and competition
bridge isolation are detailed in the 2026-10-28 follow-up section below. The
52.239 s result predates that solver correction; no ten-lap improvement is
claimed for the follow-up until repeated online verification passes.

**Latest production status (read first):** the practice launch/trial default
is now `ipopt_v343_tracking_reserve_raceline.csv`, SHA256
`21117242ebc292d666b0781e7e10fed6c4b0b4433bdbc5a1544fa4b6eda9724d`.
Reference v337 remains unchanged for comparisons. Three supervised repeats
completed eleven total laps each, ten scored: **52.468 / 52.347 / 52.239 s**,
zero collisions. Latest mean **5.2239 s**, SD .07905 s; max cycle **19.395 ms**.
One earlier repeat deferred one 20.656 ms cycle; no hard worst-case timing
guarantee is claimed. The **49.8 s goal remains unmet** (2.439 s gap).

Latest complete-run odometry u/v/r RMSE: **.0091 m/s / .0017 m/s / 0 rad/s**;
u max .0492 m/s, v max .0245 m/s. Relative eleven-lap position drift .080 m,
max .105 m. AMCL map-pose RMSE **.065 m**, p95 .119 m, max .151 m. AMCL PF wall
median/p95 **2.548 / 3.160 ms**, versus 2.918 / 3.550 in the earlier cleanup
repeat; startup maximum 40.832 ms is reported separately, not hidden.
These different live trajectories do not isolate every estimator change.

Current remaining driving issue: controller-estimated cross-track RMSE .0803 m
and speed-error RMSE .5895 m/s; it still misses the 4.9486 s speed-profile
integral substantially. Seventy-eight retained certified-plan actions and seven
exact candidate prefixes were used in the last run. No new simulation data is
needed merely to investigate these already-recorded states.

- Production repeat `mpc_v337_production_peak_slip_20261027_01`: eleven total
  laps, ten scored averaging 5.3654 s (53.654 s total), no collision; maximum
  publication time 19.111 ms. Target v337 integral is 5.1465 s.
- Recorded production odometry errors: forward velocity RMSE 0.0088 m/s,
  lateral velocity 0.0011 m/s, yaw rate effectively zero against gyro/truth;
  relative endpoint drift 0.134 m over the complete eleven-lap capture.
- v343 supplies a source-OCP-feasible 4.9414417 s target and complete saved
  seed. The original geometry failed repeated driving; the promoted separate
  tracking-reserve geometry completed the repeats above. Neither profile's
  theoretical integral is an achieved lap time.

## Original work queue (implementation/results below)

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

## Faster target experiments (historical, not active settings)

`mpc_v343_fast_target_20261028_01` used the recovered v343 4.941442 s target
with actual MPC. Warmup 5.918800 s; scored laps 5.134800, 5.129800, 5.185800 s.
It stopped without collision on an exact corridor failure after four total
laps. At turn exit, e_y grew about 0.067 -> 0.120 m while full plans were
still accepted, then roughly 25 retained-plan actions exhausted the useful
suffix before the corridor fault. This is faster, not yet consistent.

The subsequently retired `mpc_sqp_tracking_candidate.yaml` tested a coordinated
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

That cache trial stopped after eight total laps on a corridor fault, no
collision; max cycle 19.522 ms. This does not prove repeatable tracking, and
the cache is not yet considered a live robustness gain.

## Distinct scan evidence

Source `LIDAR.FixedUpdate()`/`LaserScan()` writes `CurrentRangeArray` only at
capture; Socket republishes that held array. Offline full-array byte
comparisons on two practice captures found 13/2254 and 18/2246 moving scans
identical to the preceding scan at a new ROS timestamp (0.58% / 0.80%). The
reported scan period is 25 ms; this rules out an every-other-packet 20 Hz hold
as the main localization-bias explanation. Do NOT introduce an assumed fixed
scan delay from the observed along-track error.

Implemented exact duplicate-measurement suppression only after local lock.
Compare full ranges and used scan geometry/frame, not a fuzzy threshold/hash.
Keep the odom prediction baseline unchanged until the next distinct scan; do
not reweight the cloud or refresh map-correction health at a new pose using
the held scene. Global bootstrap keeps its repeated-evidence search. Added
CPU tests for fresh stamps, one-ULP range changes, NaNs, different geometry /
frames and empty scans. Four focused localization CTests and topic policy pass.

Current test `mpc_v337_progress100_distinct_scans_20261028_01` retains baseline
tracking/damping, reduces the failed 400 progress incentive to 100, and
includes the new distinct-scan check plus matrix cache. This is a candidate
trial, not a promoted profile or an attribution-isolated speed comparison.

Result: eleven total laps, no collision, no deadline miss; ten scored 53.686 s
(5.3686 mean, SD .04736), max cycle 19.407 ms. This establishes operation of
the integrated candidate, NOT a meaningful speed gain over the reference.

## Fast target with explicit tracking allowance

The v343 target's minimum 0.20 m center-to-wall distance leaves only 1.35 cm
for lateral tracking with a .273 m-wide aligned car and .05 m hard wall margin.
v337's minimum is .25 m. Rather than relax the hard wall margin or keep adding
controller penalties, created a separate tracking-reserve target from v343:

- New reusable `tools/make_tracking_clearance_target.py`: convex periodic
  least-displacement/second-difference smoothing under corridor bounds; no
  vehicle-force fitting. Resample exact map corridors after each geometry
  update, enforce an input-path displacement limit, and certify swept map
  footprint clearance with the same rear-axle dimensions as production.
- `tools/vehicle_footprint.py` is the shared production-header geometry reader
  for this tool and plotting; no ROS/plot dependency needed for dimensions.
- Four tests pass: no-op on sufficient clearance, correct-wall motion,
  periodic symmetry, impossible reserve and analytic circle curvature.
- Candidate `optimized_practice_v343_certified_restart_20261027/
  tracking_reserve_target_not_dynamics_certified.csv`: 474 points, maximum
  displacement 4.809 cm, minimum center-to-wall distance .25 m; swept-map
  clearance certificate .03022 m BEYOND the retained .05 m hard wall margin.
  Speed samples are unchanged; geometric speed-profile integral 4.948605 s.
  This is NOT a newly certified source-OCP trajectory, not a proven lap time.
  Its changed geometry does not inherit the original dynamic certificate.

Actual supervised verification `mpc_v343_tracking_reserve_20261028_01` requests
eleven total laps, baseline production weights (progress reward zero), current
source plant, distinct-scan check and matrix cache. Original reference and
IPOPT seed/run are unchanged. Stop immediately on collision or controller fault.

Result: eleven total laps, ten scored in **52.468 s** (mean **5.2468 s**, SD
.05147 s), no collision/deadline miss, max cycle **19.318 ms**. The geometry
change gave a usable faster multi-lap target versus ~53.7 s v337 repeats, but
not the requested 49.8 s. The source-OCP dynamic certificate still does not
transfer; this recorded MPC/Unity run is its actual driving evidence.

## Ordered matrix loops instead of the affine cache

Retired the experimental private `P*d` cache. Its mathematically equivalent
float grouping changed warm-start/SQP branches across sequence replay, making
it harder to attribute live behavior. It is removed from the source, not left
as a runtime switch.

Reordered independent coordinate loops in the existing factored Riccati
products `B'P`, `B'PA`, `P A` and backward `A'p`. Each individual coordinate
still accumulates terms in the original order; inner loops now traverse
contiguous rows and permit compiler SIMD. No states, dynamics, weights,
factorization reuse assumptions or arithmetic reductions are removed. GCC
vectorization diagnostics confirm 16-byte vector loops. The independent
unfactored pass remains unchanged as the test oracle.

Evidence in `affine_cache_benchmark/ordered_simd.{json,csv}`:

- Six frozen QPs: **bitwise exact recorded result parity**, zero state/control
  delta, identical statuses and iteration counts.
- All 421 racing-state replay rows: **zero differences** in status, iteration
  count, first steering, first throttle and nonlinear objective.
- ADMM median **.722 -> .520 ms** (-28.0%), p95 **2.859 -> 1.916 ms** (-33.0%).
  Whole offline cycle median 5.220 -> 4.929 ms, p95 15.377 -> 14.824 ms.
  Maximum 27.641 -> 19.766 ms is one observed comparison, not a worst-case
  timing guarantee. ROS/publication timing still needs the actual run.
- All eleven MPC CTest targets pass.

Current live check: `mpc_v343_tracking_reserve_ordered_simd_20261028_01`, same
faster reserve target and baseline weights (progress reward zero), eleven
total laps requested. This isolates the new matrix-loop implementation from
another weight change. IPOPT v347 remains untouched in the background.

## Final promotion and retirement

- Ordered-loop repeat completed ten scored laps in **52.347 s**, mean 5.2347 s,
  SD .04165 s, no collision. One cycle was deferred at 20.656 ms. The profile
  never intentionally publishes an over-budget replacement command.
- A stronger pace/damping bundle on the reserve target failed before warmup
  (`mpc_v343_reserve_pace_bundle_20261028_01`). Removed the progress reward
  implementation, parameters and failed candidate overlay entirely. The
  existing propagated progress-deviation coordinate remains dynamically
  necessary; it is not that retired optional reward. Production weights stay
  at the validated reference values. Generic signed-objective comparison
  remains an actively used deduplicated prefix-selection helper.
- Copied the validated target into installed planning assets and changed
  `controller.launch.py` / `run_mpc_practice_trial.sh` defaults. Explicit map /
  trajectory overrides and the previous v337 reference remain available.
- `mpc_v343_promoted_defaults_20261028_01` tested the actual new defaults with
  no overlay/trajectory argument: ten scored **52.239 s**, no collision or
  deadline deferral; max **19.395 ms**. This is the implemented default, not
  an offline-only candidate. No runtime ground truth inputs were introduced.
- Removed the duplicate old solver-input exporter (with obsolete startup-ramp
  assumptions), stale hard-coded weight sweep, and unused legacy accuracy
  profile. Canonical saved-state exporter, benchmark/frozen-QP tools and the
  actual production profile remain. The one-shot horizon overlay now changes
  ONLY its measurement protocol, not old weights/tolerances/worker settings;
  it latches at >=6.5 m/s. It must overlay production configuration.
- Fixed reporting that compared local `/ekf_pose` coordinates directly with
  absolute Unity world coordinates and printed a spurious ~8 m error. EKF is
  still scored via body velocities and origin-aligned relative odometry;
  global map poses are scored separately. Added a regression test. Added
  actual AMCL PF wall statistics to the normal report. Final focused Python
  tests: **35 pass**; MPC **11 CTests pass**, localization **4 focused pass**;
  allowed-topic guard passes.

## Fixed-command rollout diagnostic — completed, tail gate not passed

`fixed_command_rollout.txt` / `fixed_command_rollout_cases.json` in the promoted
run score the shared production C++ source plant against the captured **actual
issued command sequence**, without future state re-anchoring. The rebuilt
`libunity_source_plant_eval.so` contains our own online module, NOT an external
PhysX engine. Bridge debug stamps/actual sent inputs are used offline only to
remove command-association ambiguity; they are not allowed runtime estimator
inputs. Ground truth is scoring only. No yaw-residual gain or oracle future
IMU/actuator input is enabled. Existing full-speed bags cover this diagnosis.

On 2,321 complete rolling 750 ms windows, endpoint XY RMSE is **.01936 m**,
p95 **.01948 m**, maximum **.74028 m**. Both full-path and endpoint errors
stay below .05 m in **2,317 / 2,321** windows. These windows overlap; they are
not independent trials. Four adjacent failures are one localized transient,
not four independent incidents. No window was excluded by the truth pose /
twist closure validity screen. The maximum-error gate therefore remains
**FAILED**, despite the low pooled RMSE.

| Horizon | u RMSE m/s | v RMSE m/s | r RMSE rad/s | XY RMSE m | XY max m |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25 ms | .00906 | .00167 | .01020 | .00485 | .03245 |
| 750 ms | .00801 | .00345 | .01566 | .01936 | .74028 |

Failure start stamps: 1793172431477950617, 1793172431502967602,
1793172431527660770, 1793172431552604866 ns. All start around 6.43–6.66 m/s,
.079–.092 rad steering. Yaw-rate error first exceeds .1 rad/s after 7–10
steps (175–250 ms), during the same subsequent steering reversal; position
first exceeds .05 m at steps 20–21 (500–525 ms). Worst transient yaw-rate
error is **2.410 rad/s**, even though the final-step error is only .393 rad/s.
Endpoint-only motion metrics must not conceal that intermediate error.

Targeted, diagnostic-only initial-state substitutions on the worst case:
production initial u/v 6.58811/.137995 m/s gives .65538 rad integrated yaw;
substituting only same-epoch truth u=6.5779 gives -.06390 rad; substituting
only truth v=.1408 gives -.06056 rad. Actual yaw change is about -.117 rad.
These are **not deployable fixes or future-state oracles**: they show strong
initialization sensitivity of this nonlinear unwind. Recomputing initial
jounce alone gives .38110 rad and does not resolve it. Do not promote a
jounce reset, extra damping, timing shift or tire-parameter change from this
diagnostic. Source-equation/contact-boundary and observer-initialization
attribution remains open; no native-solver equivalence claim is justified.

The separate `horizon comparison` section at the end of the text report
compares a controller plan with subsequently **replanned** controls; its
15 cm score is not this fixed-command plant accuracy measurement.

## Savepoints and current verification

Scoped commits: `71cb23faf` localization cleanup/projection/GPU ordering;
`e33ed8e9` exact-parity ordered Riccati SIMD and active MPC configuration;
`b39c90b2` held-scan suppression and reporting; `b7b6009b` faster installed
target; `726e33cf` retired analysis/runtime paths removed. Existing unrelated
staged work was not blanket-committed. Recovery snapshot of the pre-existing
staged work: `6a39b5be6ef534946b926b266beaa78264c9810f` (stored in the stash
list); do not apply it over this worktree without reviewing it first.

Final focused verification now includes 47 Python tests, 11 MPC CTests and
four localization CTests, plus the allowed-topic guard. The asymmetric
rear-axle footprint correction is also applied to the trajectory auditor,
so offline validation does not reintroduce the removed symmetric-overhang
assumption.

One subsequent coordinated pace/tracking experiment was supervised:
`mpc_v343_tracking_pace_20261028_02`, e_y/e_psi/u weights 6000/2000/50,
steering/throttle slew weights 1500/30, same plant, trust regions, profile and
acceptance gates. It completed five total laps, four scored averaging
5.20205 s, then stopped without collision on an exact corridor failure after
a long retained-plan chain. One deadline was deferred. The small speed gain
does not meet the repeatability gate, so **not promoted**; the temporary
overlay was deleted, with its exact copy retained in the run evidence.
Attempt `_01` was stopped during startup, before simulator launch: its YAML
selected the wrong node name, so it is **not a weight-comparison run**.
Added tested preflight selector validation in the manifest sealer and direct
development launcher: mismatched/empty overlays now fail before driving
instead of silently testing unchanged defaults. Commit `898ce502` saves this
guard and the auditor's shared-dimension/asymmetric-footprint correction,
without folding unrelated staged changes into the commit. Independent corner
rotation tests cover every integer heading from -180 to +180 degrees.

IPOPT remains running (stage 1/6, factor 2, iteration 3,585 at the last check;
current objective 4.940069 s, primal residual .00018, optimality residual
.00366). No optimizer restart/source edit was made for runtime tests. No
simulator or controller is left running after the supervised trials.

## Remaining work, not a completion claim

1. Attribute the localized, initialization-sensitive unwind rollout failure
   using the already-recorded full-speed evidence. No tire/physics change
   was made merely to hide its tail; the .05 m maximum criterion is unmet.
2. Improve turn-exit tracking/convergence without long retained-plan chains;
   the stronger weight bundle did not solve this. Use the actual command /
   state history, not another blind parameter sweep.
3. Reach ten scored laps below 49.8 s. Best verified repeat is 52.239 s;
   neither the target integral nor IPOPT's current iterate counts as a lap.
4. Leave the all-day IPOPT service running; evaluate completed checkpoints
   against source feasibility and measured driving before promoting them.

## Follow-up: convergence, precision and competition boundary

The objective remains ten scored laps below 49.8 s (also meets the user's
latest <50 s requirement), excluding warmup. A new IPOPT path is not the only
route to improvement: the current target's integral is 4.9486 s, whereas
closed-loop repeats average around 5.22 s. Tracking/convergence and estimator
uncertainty still cost time on this already-loaded target.

### Measured solver defect and correction

On the latest 2,346-cycle repeat, only 637 cycles selected a full nonlinear
step; 1,382 selected alpha .5/.25/.125/.0625 and 326 had alpha zero. Fourteen
consecutive retained-plan actions occurred twice, with |e_y| growing to
.137–.151 m at high-speed turn exit. Simply changing weights or waiting for
IPOPT does not fix those controller-side limitations.

Found a concrete chronology bug: `riccati_admm_solve_core` shifted projected
variables, duals and per-stage penalties on **every call**, including the
second SQP solve at the same command epoch. RTI correctly kept its nominal
sequence unshifted, so the solver warm start was one stage ahead of it.
Added explicit `reuse_same_epoch` configuration for refinement only; the next
control cycle retains the normal single shift. A one-iteration analytic
zero-cost QP regression test with distinct stage markers proves stage indices
are preserved for refinement and shifted for the next epoch. No vehicle
physics, horizon, state dimensions, weights or safety margins were changed.

Frozen-QP format is now **v3** in C++ and Python, including this explicit
chronology bit; unsupported older versions are rejected, not silently decoded
with the wrong layout. Compiled roundtrip and Python parsing tests cover it.
Removed stale 22-state documentation and the old comment claiming two unused
future control slots despite the direct-input 30-stage production horizon.

Saved-state comparison: 600 consecutive full-speed states, baseline production
weights, v343 tracking-reserve target, eight workers. Baseline vs corrected:
ADMM convergence 516 -> 520/600; total iteration p50 56 -> 52, p95 300 -> 287;
cycle p95 17.373 -> 16.517 ms. Exact full-horizon acceptance 592 -> 590/600,
so this is **not** proof of a racing/robustness gain. These replays reconstruct
solver memory, not the unrecorded original live ADMM state.

Tighter fixed steering/throttle trust-region trials did not establish a win;
an adaptive refinement-radius prototype also worsened convergence/timing.
That prototype was removed entirely. Reports remain in
`live_runs/mpc_v343_promoted_defaults_20261028_01/trust_region_replay/` as
negative evidence; do not reintroduce it as an untested improvement.

Live verification `mpc_v343_same_epoch_sqp_20261028_01` completed ten total
laps, then stopped without collision on a corridor failure at stage 2; ten
**scored** laps were not completed. Full command/state history was recorded.
This corrects a solver defect but has not yet passed the racing goal.

### Competition inspection: allowed topics are not blanket certification

Checked the current official [2026 technical guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/#23-data-streams)
and [rules](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-rules-2026/#24-inspection-rules).
Permitted runtime inputs are LiDAR, camera, IMU, encoders and actuator
feedback; controls are steering/throttle ROS topics. Simulator pose/odometry,
race telemetry, reset and TF are restricted during racing. Team estimators /
MPC continue to use permitted sensors and their own derived messages only.

The guide also forbids modifying `autodrive_roboracer`. Our wrapper leaves
installed files intact but changes that bridge's behavior in memory. This
is a **submission-compliance uncertainty**, not resolved by a topic grep.
The user confirms no organizer approval is known. Keep it for development;
organizer clarification is required before claiming submission compliance.
Suggested question: may a separate team package wrap the unmodified installed
bridge to pace unchanged WebSocket packets at 40 Hz, synchronize ROS headers
and isolate legal actuator callbacks, without using restricted telemetry in
the autonomy algorithm? No message has been sent to organizers.

Implemented conservative isolation regardless of that outstanding question:

- Competition launch explicitly disables development packet diagnostics and
  reset. Default diagnostics flag is off even outside that launch.
- Restricted packet details are not inspected or parsed on that path; no
  per-packet timing JSON publisher/work is run. Development launch explicitly
  enables the existing complete recording path for offline diagnosis.
- Removed nonstandard camera/intensity-control fields from outbound packets;
  retained only the official actuator/reset fields. Competition reset is
  always false and has no subscriber.
- Deleted nine always-null legacy metadata globals, repeated resets/reads
  and their empty JSON fields. Diagnostic schema v5 retains actual request,
  receipt and issued-command evidence for development bags.
- Four new dependency-free boundary tests prove disabled diagnostics never
  access packet values or ROS/clock/publisher state, verify the official
  command-field set and enforce the competition launch overrides.

Remaining attribution of the four localized 750 ms rollout tails is unchanged;
none of these solver/bridge edits changes the physical plant equations.
