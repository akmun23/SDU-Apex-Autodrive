# MPC baseline recovery — 2026-10-30

## Scope and acceptance

Recover the MPC used by the successful practice runs before attempting new
cross-track robustness changes. Preserve the current draft. Keep the source
plant, hard 5 cm body clearance, practice reference, weights and permitted
sensor inputs unchanged. No weight sweep. Required first gate: one warmup
plus ten scored practice laps, then repeatability. A source hash or successful
build is not a driving certificate.

## Recovery evidence

The sealed manifest in
`live_runs/mpc_amcl_age_motion4p800_20261029_02/run_manifest.json` records a dirty
runtime, so its base commit and Docker base image are insufficient to identify
the compiled MPC. Commit `d309fa153744d9c8a4fa23219d2d648c8387c94d` contains
exact SHA-256 matches for all seven modified MPC files from that manifest:

- `include/mpc_reference.h`, `include/mpc_rti.h`, `include/mpc_trajectory_io.hpp`
- `src/mpc_reference.c`, `src/mpc_rti.c`
- `test/test_mpc_trajectory_io.cpp`, `test/test_vehicle_model_source_rollout.cpp`

The remaining MPC files are recovered from the same package revision, whose
changes from the successful run's base commit are exactly those seven files.
Three historical candidate YAMLs were recovered only inside the isolated
worktree; they are not new production modes.

The recovery worktree is
`live_runs/mpc_reference_reproduction_20261030/source`. It uses the current
committed non-MPC runtime and launcher with the recovered MPC package. This is
an exact MPC-source reproduction, not a claim that every localization source
file from the historical dirty runtime has been recovered. Effective saved
MPC weights, trust radii, solver cap and hard corridor were compared to the
successful run and agree. Current practice localization settings match the
saved scan-age/gain settings; code identity is a separate question.

## Known differences to investigate, not bundle together

1. The successful MPC penalized steering-command changes relative to required
   reference increments. The draft penalizes raw changes while retaining the
   practice steering-slew weight of 1500.
2. Draft nonlinear merit uses a frozen per-SQP target/weight schedule and
   double objective values; the successful implementation predates this.
3. Draft nonlinear rollout applies Riccati feedback. The successful package
   predates that added policy and its gain scaling.
4. Rejected-plan/prefix selection and warm-start logic changed substantially.

None is yet identified as the sole cause. Preserve full reproduction as a
control, then test one coherent algorithm change at a time on the same states
and references. Retain practice validation as a regression gate for changes
intended to improve IROS; do not introduce track-name branches in dynamics or
controller logic.

## Current execution

- `practice_reference_reproduced_20261030_01` is launched from the recovery
  worktree using a freshly compiled package, original 4.800 s source witness,
  current practice fastline profile, and one warmup plus ten scored laps.
- Main worktree is not overwritten. No other simulation or weight sweep runs
  concurrently. Result pending.

## Reproduction findings and first isolated restoration

- Reproduction attempt 01 stopped during launch, before the simulator: the
  historical CMake install list omitted the newer name of the identical
  practice fastline YAML. Added that config install entry only; no runtime
  equations changed. This is a harness failure, not a driving failure.
- Attempt 02 completed warmup, then failed a first-stage corridor check by
  1.37 mm after 411 control cycles. Maximum gate time 19.012 ms, no deadline
  miss. Recorded-command replay: 387 windows, endpoint RMSE 1.348 cm, maximum
  path error 3.972 cm.
- Attempt 03 completed warmup and eight scored laps, mean 4.971 s, before
  rejection on the next lap. Three gate overruns, maximum 20.610 ms. Its final
  17 consecutive retained-plan actions coincide with heading error increasing
  toward -0.964 rad. Recorded-command replay: 1,802 windows, endpoint RMSE
  1.476 cm, maximum path error 4.953 cm. No collision reported. Thus the old
  MPC's ability to drive quickly is reproduced, but ten-lap repeatability is
  not yet re-established.

The first main-tree restoration changes only the steering-rate objective to
the successful v1/geometry rule:
`w_delta * ((delta[k]-delta[k-1])-(delta_ref[k+1]-delta_ref[k]))^2`.
QP linear terms and exact nonlinear merit use the same reference increments.
Throttle increments stay zero-centered; weights, source plant and constraints
are unchanged. The newer double-precision merit and frozen target schedules
remain, so this is an isolated restoration, not falsely labelled an exact
copy of the entire historical MPC. The source-command input prior remains a
separate cost; this rate reference uses physical rack/feedforward steering.

Before editing, the complete current MPC diff was preserved in
`live_runs/mpc_reference_reproduction_20261030/draft_before_steering_cost_restore.patch`.
The exact recovered MPC remains independently available in its worktree.
Updated objective/gradient regressions pass: 172 reported package tests,
zero errors/failures. Main-tree trial `practice_rate_restore_20261030_01` is
was run with the same reference and zero soft target shift.

## Main-tree online result and remaining constraint issue

- `practice_rate_restore_20261030_01`: warmup plus seven scored laps, mean
  4.94709 s, then corridor rejection. Gate p50/p95/max 9.848/17.230/20.562 ms
  (two over-budget cycles). No collision reported.
- Unchanged repeat `practice_rate_restore_20261030_02`: warmup plus five
  scored laps, mean 4.97500 s, then corridor rejection. Gate max 32.535 ms
  (one over-budget cycle). No collision reported.

This restores actual fast driving from the recent zero-lap failures without
changing weights or dynamics, but is NOT a repeatable ten-lap pass. The
historical package also failed repeats, so the remaining fault cannot all be
attributed to the draft changes. No competition trial has been launched.

Both native lines satisfy the current MPC corridor rule at all source samples:
minimum practice slack 3.605 mm and IROS slack 8.074 mm before tolerance,
including the +/-0.10 m width preview. Command-time prediction already
advances to the actuation epoch; `MPC_COMMAND_DELAY_STEPS` is zero. Do not
introduce another two-stage shift into source commands based on the stale
interpretation that the optimizer still includes a two-stage input queue.

There is a distinct boundary-model counterexample in restored repeat 02.
At failed predicted progress 188.570376 m, the rectangular Frenet corridor
rejects `e_y=-0.0107450 m` against lower bound `-0.0048070 m`. Reconstructing
that SAME predicted pose from the reference and evaluating the map-footprint
perimeter gives 0.052571 m conservative body-to-wall distance, even after the
existing 22.75 mm between-probe correction. Thus this rejection is stricter
than the optimizer's own 5 cm map-footprint certificate. This is not a ground-
truth clearance claim and cannot be generalized to every rejected state:
repeat 01's failing pose has only 0.028508 m conservative map clearance.

Two separate questions must remain explicit:

1. Is the user's 5 cm buffer nominal planned clearance that tracking may
   consume, or a strict minimum actual predicted-body clearance? The present
   code enforces the latter. An asynchronous clarification was requested;
   no hard clearance has been relaxed without that decision.
2. Independently of that choice, the local normal-width corridor can reject
   a pose accepted by the source optimizer's full-map footprint check. A
   track-independent correction should make those geometric predicates
   consistent, not add a map-name offset or resample a different target line.

All new simulation processes are stopped. Current recovered objective and
its 172 passing package tests are retained; the exact historical worktree and
pre-restoration draft remain available for comparison. Further validation
must report actual lap completion, not promote on unit tests alone.
