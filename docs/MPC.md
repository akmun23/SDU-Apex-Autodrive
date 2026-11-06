# The MPC

One controller: a 30-stage, 40 Hz sequential quadratic program over steering
target and normalized throttle. Every candidate is integrated through the
source-derived Unity plant (`f1tenth_mpc/src/unity_source_plant.cpp`); the QP is
only a local model that proposes the next candidate.

## Problem

Decision: 30 pairs (steering target, throttle in [0.001, 1]). Cost: quadratic
tracking residuals (lateral/heading error, speed, lateral velocity, yaw rate,
steering state/queue), steering/throttle effort and slew, and the longitudinal
tire-slip cost `w * max(0, r)^2` per wheel and stage, where `r` is the signed
peak of |wheel surface speed - ground speed| minus the tire curve's first
extremum over the stage's 25 plant substeps. Hard constraints: the eight body
corners must stay inside the corridor at every stage, and the command box.
Exact zero throttle is Unity's brake branch and is not a decision.

## One control cycle (`mpc_rti_solve_cycle`)

1. Build the warm start by shifting the previous plan by the elapsed command
   stages and integrating it from the freshly measured state.
2. `source_sqp::solve` (`src/mpc_source_sqp.cpp`) iterates until the deadline:
   exact forward sensitivities of the stage maps (`linearize`, parallel over
   stages) -> local model (`build_model`) -> QP (`src/mpc_source_qp.cpp`) ->
   exact rollout of the candidate -> accept only if the exact result improved.
   The best exactly verified iterate is always kept (anytime).
3. Independent re-validation of the plan, then publication.

The local model: Gauss-Newton curvature of the quadratic residuals; each wheel's
slip cost as an exact convex epigraph of the first-order residual (so a slip
limit the step would cross is visible *before* the step); first-order wall rows
of all eight corners; the legal command interval. While the iterate violates a
wall the rows enter as an exact L1 penalty, so the QP is always feasible and
restoration is the same code path as optimization. Row generation is exact: rows
enter the QP lazily and the solve repeats until every omitted row is satisfied.

The step is limited to a radius in scaled control units. The radius is adapted
only from the ratio of exact to predicted improvement and is carried from cycle
to cycle. There is no curvature shift, regularization or fallback controller.

## What was measured about accuracy

* First-order sensitivities of the plant are valid over very small steps in the
  stiff wheel/throttle range: a 0.0025 throttle change moves the next-stage
  speed by 2x its first-order prediction, and the response is asymmetric around
  the nominal point.
* A finite-difference Hessian of the exact gradient (the best case for an
  exact-second-order method) still gave full Newton steps that were wrong by
  about 3x at the median, because the cost landscape has kinks and negative
  curvature, not merely curvature. Exact second derivatives with this plant would
  cost on the order of a second per iteration.
* Hence the step bound. Without it the undamped GN step is accepted at about 6-9%
  of its length by the exact check; with it, 65-95% of steps are taken whole.

## Development workflow

    tools/run_trial.sh practice RUN_ID                 # closed loop in Unity
    tools/summarize_trial.py live_runs/RUN_ID          # deferrals, timing, QP accuracy
    tools/evaluate_mpc_bag.py live_runs/RUN_ID/run     # lap times

Frozen-problem benchmarking (`f1tenth_mpc/util/`): run a trial with the overlay
`capture_source_benchmark: true`, then

    mpc_export_case DIAGNOSTICS_RECORD.json RACELINE.csv OUT.case
    mpc_sqp_bench solve OUT.case [ITERATIONS] [RADIUS]  # model vs exact, per iteration
    mpc_sqp_bench drive OUT.case CYCLES                 # closed loop on the source plant

`drive` reproduces the controller's deferral/collision behaviour without the
simulator; `solve` prints the predicted and exact cost change of every step.

## Results (practice track, 11 laps = warm-up + 10 scored, default profile)

Every one of 2,189 cycles published; no deferred or degraded cycle. Scored lap
times 4.860-4.893 s (mean 4.878 s); the raceline reference is 4.799 s. Iteration
time: median 4.0 ms (linearization about 2 ms, QP 0.6 ms, rollout 0.9 ms);
cycle gate p99 18.7 ms of the 22 ms budget.

## Known limits

* Where the plant switches branches (wheel spin onset, tire saturation) the
  model is only valid over a short step; a few percent of trials in the sharp
  hairpins are rejected and re-solved at a smaller radius. The tail of
  |predicted - exact| cost is dominated by those rejected trials (P99 about 200
  against costs of 10^4 in the hairpin).
* Not yet exercised: the competition track after the solver rewrite (the old
  controller lost control there after a single missed publication).
