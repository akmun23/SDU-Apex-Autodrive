# Nonlinear SQP and finite-difference second-order experiment — 2026-11-03

## Engineering status

This checkout contains source-derived sensitivities, a stagewise PSD steering/throttle
secant-curvature correction applied to the **second** SQP pass, exact source
relinearization, progress-coupled hard footprint polygon experiments, a
source-viability/recourse experiment, and a strict nonlinear full-horizon acceptance
validator. The source optimizer / source-gradient and Phase-I paths remain optional
and not qualified for automatic replacement of the normal solver or fallback.
**This is not a complete second-order SQP dynamics formulation or a robust racing
qualification.**

A convex QP is intentionally retained at each SQP iteration; the nonlinear
problem is the original, exact source-model multiple-shooting problem and
physical footprint objective/constraints. The two local subproblems must not
be confused with exact finite-step tire/contact dynamics.

## What finite differences demonstrate

The earlier 18-corner-state, 2,772-component-per-scale audit evaluated the
**local first derivative** of the 25ms source model. At the smallest perturbation
the maximum relative scaled discrepancy was 0.277%, with no tested relative
error greater than 1%. This does not test whether a candidate SQP step is
accurate or source dynamics match Unity at the same actuator epoch.

The additional, reproducible experiment is:

```bash
python3 tools/audit_mpc_source_corner_second_order.py \
  live_runs/comp_gradient_nearwall_capture_20261102_01/run \
  --max-rows 18 --output /tmp/mpc_corner_second_order_vs_linear_20261103.csv

for scale in 2 4 8; do
  python3 tools/audit_mpc_source_corner_second_order.py \
    live_runs/comp_gradient_nearwall_capture_20261102_01/run \
    --max-rows 18 --h-scale "$scale" \
    --output "/tmp/mpc_corner_second_order_scale_${scale}_20261103.csv"
done
```

This uses the **full source plant** and its forward sensitivities at 18 recorded
high-steer states. At each state it compares first-order Taylor prediction
against a 3-evaluation directional second-order central finite difference
and the actual exact 25ms source step after a *finite* change. It evaluates
steering command, throttle command, physical steering, lateral velocity,
yaw rate, pitch, and rear-left jounce. There were 720 output comparisons
across six physical output channels at the default perturbation.

| Direction (across all output channels) | Percentage where quadratic was more accurate, nominal FD h | Range across h×2, h×4, h×8 |
|---|---:|---:|
| yaw rate | 68% | 69–74% |
| lateral velocity | 27% | 24–31% |
| physical steering | 8% | 8–22% |
| rear-left jounce | 6% | 25–28% |
| pitch | 3% | 6–35% |
| steering command | 36% | 35–40% |
| throttle command | 14% | 10–29% |

**Interpretation:** A local second-order Taylor model is often less accurate
than the first-order model over these relatively large perturbations. The
piecewise slip-force law, suspension/contact transitions, actuator bounds,
and the numerical finite-difference scale are plausible causes. This audit
does not isolate a unique cause. Some channels are insensitive to current
physical steering perturbations because of command queue dynamics. It uses
representative basis-direction changes, **not the actual multistage QP
solution direction**; a follow-up candidate-direction sweep is needed before
designing a live Hessian or deciding how many nonlinear source rollouts to
remove.

Do not turn on a full numeric Hessian based on first-derivative parity
alone. Preserve the source-exact acceptance and R1→R2 relinearization.

## Corrected mathematical QP defect in this working tree

A separate post-transform QP defect was exposed by the full ROS test suite.
Independently rounded float32 Gauss–Newton source-slip/suspension Hessian
components can leave a small negative eigenvalue, notably after
converting absolute to incremental steering/throttle coordinates. The old
regularizer applied only to a subset of body/suspension state diagonals and
rejected a valid source-slip test with an approximately -0.0022 eigenvalue.

Updated `include/mpc_qp_convexity.hpp` to use a minimal positive spectral
correction plus a **roundoff-bounded** diagonal pad, re-quantizing the actual
25-coordinate Q/N/R Hessian. The realization of the correction is removed
from q/r at the **state and input nominal**, maintaining the original source
first-order gradient. A genuinely indefinite physical Hessian remains a
hard error, not a disguised trust-region relaxation.

The actual hard 3D corridor QP also had an ADMM state-split bug: a polygon
coupled to progress could be projected but ignored when the independent
progress bound was treated as nonconstraining. Both ADMM split and Riccati
augmentation now recognize polygon-constrained state axes, including the
progress coordinate. A 160-case weighted projection comparison against
OSQP passed; the native ADMM on a difficult saved QP can still have poor
raw primal convergence within 300 iterations. A zero projected wall
violation does **not** prove a feasible raw solution. This is not fixed merely
by increasing iteration counts.

Rebuild/test after the spectral correction and regression updates:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select f1tenth_mpc --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select f1tenth_mpc
colcon test-result --verbose
git diff --check
```

Result: **336 tests, zero errors/failures**, with all in-development changes
still uncommitted. This is a numerical test result, **not Unity qualification**.

## Release blockers

1. Capture *actual* QP proposal state/control directions at failing corner
   epochs, compare exact source `f(x+dx,u+du)` with first and directional
   second-order predictions for each stage in physical units and footprint.
2. Derive an SQP acceptance / corrected-subproblem strategy from exact-source
   agreement. Avoid global second-order approximations across slip/contact
   transitions, and keep two relinearized SQP passes.
3. Compare synchronized Unity states against all source channels and actuator
   queues, including rear wheel rates, steering, yaw and jounce, at each
   high-slip failure.
4. Improve successor-state viability in the *normal* constrained problem
   before removing fallback and source-search pathways. A source-local QP
   optimum is not a recursive-feasibility certificate.
5. Qualify on strict practice and competition Unity runs, then five
   independent ten-lap runs. No timing claim or production promotion before
   nonlinear correctness/feasibility.
