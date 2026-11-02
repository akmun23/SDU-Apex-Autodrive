# Normal SQP convergence and terminal continuation, 2026-11-01

Status: numerical repairs implemented; source curvature and terminal continuation
remain experimental. Neither fallback removal nor cross-map qualification is
established. Production parameter files, plant equations, margins and actuator
limits were not changed by this work. All changes were made in the existing
dirty checkout; unrelated changes were preserved.

## Numerical repairs

The former hybrid branch forced `sqp_limit=1`, contradicting both the configured
two passes and its public comment. All QP modes now honor the configured limit.
R1 receives an earlier deadline when the available cycle budget can reserve a
complete R2 build/solve/verification opportunity. R2 remains conditional on the
remaining time and retains the better physically verified frozen objective.

The ADMM progress coordinate is a displacement from the current nominal, not
an absolute physical state. Warm split variables now shift once at a new epoch
and rebase that coordinate to the new nominal, in the solver's actual scaling.
R2 retains the command epoch. The generic solver also shifts channel penalties
before normalizing shifted duals, rather than pairing a shifted dual with the
old stage's penalty. Tests check station invariance across both epoch modes,
scaling, command alignment and penalty ordering.

A concrete source of catastrophic residuals was found in feasibility
restoration. It multiplied the complete speed Hessian diagonal and linear
coefficient by 0.1, leaving the speed/wheel/input cross terms of the source-slip
Gauss-Newton cost intact. That operation can turn a convex QP indefinite.
Restoration now preserves the coupled objective and adds only PSD wall
penalties. Physical input bounds and final nonlinear acceptance are unchanged.

Two archived phase18 states were retained under
`live_runs/mpc_normal_sqp_unification_20261101/phase18_*` with their complete
physical state, source internal state and nominal command sequence. Replaying
captured corrected QPs, then reintroducing only the old diagonal scaling:

| Source state | Legacy primal / dual | Corrected primal / dual |
| --- | --- | --- |
| progress 23.24495 m, run `_04` | 0.182271 / 0.636292 | 0.000014305 / 0.000273705 |
| progress 136.16758 m, run `_07` | 1.97592e27 / 1024.512 | 0.0573559 / 0.0212708 |

The corresponding legacy condensed Hessian minimum eigenvalues are -4293.19
and -20818.18. This reproduces the divergence mechanism; it is not a claim of
bit-identical live replay, because the live z/y/rho snapshot was not saved.
The second corrected QP still reaches its iteration cap: a finite residual is
not complete convergence.

Independent LPs find all six saved corner LTV problems feasible. Their normal
condensed Hessians are positive definite, with condition numbers approximately
4.60e4 to 9.09e6. Large local source sensitivities and finite-step nonlinear
error remain separate from QP infeasibility. Artifacts: `ltv_feasibility.json`,
`conditioning.json`, `restoration_convexity.json`, and
`extreme_restoration_replay.json` in the same experiment directory.

## One main SQP

The runtime source-primary branch and the independent post-QP source optimizer
were removed. The source nonlinear optimizer remains available as an offline
oracle and derivative regression tool. Nonzero
`experimental_source_nonlinear_iterations` is explicitly rejected. The old
`experimental_qp_source_polish` parameter is a compatibility alias for
`source_informed_sqp`; it no longer launches another controller.

The existing source adjoint now has a shared operation on an already-built QP.
Normal SQP uses the exact physical nominal's source Jacobians, frozen objective
and this adjoint without another horizon rollout or Jacobian build. R2 can
compare the new source gradient with the old quadratic model's gradient at the
accepted control sequence. Additional nonnegative input curvature is centered
on the new nominal, so it preserves the current first derivative. Absolute
command and increment-coordinate formulations both include all induced
state/input cross terms.

The first experimental correction used the norm of gradient discrepancy. It
improved exact saved-state cost in 5/6 cases but regressed the sixth and failed
live qualification: 50.718 s for ten scored laps, three viability recoveries,
one short prefix, cross-track RMSE 0.17073 m. Its higher full-step acceptance
count was not evidence of better control. That norm rule was replaced with
missing **positive directional curvature** only; negative and perpendicular
gradient changes no longer automatically cause extra damping. The current
saved-state results are in `directional_secant/corner_comparison.csv`:

| Progress | Normal two-pass exact cost | Directional source curvature |
| --- | --- | --- |
| 23.8618 m | 227.0029 | 172.1638 |
| 39.8147 m | 264.8583 | 264.8583 |
| 50.8604 m | 343.2089 | 322.6918 |
| 66.3103 m | 55804.8170 | 55804.8170 |
| 293.2570 m | 1983.5162 | 1983.5162 |
| 303.9130 m | 9798.1500 | 9798.1500 |

These compare the same 750ms frozen objective and original physical corridor.
The correction is a secant estimate, not a proven global curvature bound.
`source_informed_sqp` defaults off pending live repeatability.

The directional correction's first live trial,
`normal_sqp_source_directional_20261101_01`, completed ten scored laps in
**48.9610 s**, with one viability recovery and one retained-plan action,
no reported collision or publication-gate violation. Maximum gate 18.2597 ms,
cross-track RMSE 0.02948 m, maximum QP primal residual 0.06798. R2 completed in
2199/2202 solver cycles; 1928 cycles had a positive selected exact R2 gain.
This is promising single-run evidence, not a paired causal performance claim
or proof that the remaining recovery paths can be removed.


## Terminal constraints and costs inside the QP

`terminal_viability` defaults off. It represents a particular legal backup:
hold the final optimized command for three further 25ms source transitions.
Its terminal value includes three future tracking costs and optional wall
reserve costs. Up to ten affine inequalities represent both future footprint sides
at all three steps, physical speed upper bounds at each step and nonnegative
forward progress through the suffix. Stationary startup permits zero progress.
The first prototype also imposed an unqualified heading/lateral-velocity
funnel; it failed and those artificial hard limits were removed. A satisfied
constant constraint on a stopped source branch is omitted rather than sending
a zero-normal plane to ADMM.

The terminal model uses the production source sensitivities and exact path
frame differentials. It propagates sensitivity through all 20 physical states,
progress and the final two commands. Only inexpensive footprint/reference
geometry is differenced. A PSD diagonal majorizer of the dense continuation
Gauss-Newton cost preserves its gradient. General terminal halfspaces get
independent ADMM split variables and enter the Riccati terminal quadratic and
linear terms; they are not approximated by independent state boxes. Scaling
transforms their normals consistently. Terminal multipliers are initialized
anew when the planes change.

Both SQP passes share frozen continuation targets. Every accepted 750ms plan
also receives a full nonlinear continuation check, including the existing
world-sweep callback and command/state validity gates. Terminal failure gets
ordinary backtracking and relinearization; it is not mislabeled as a plant
model failure. The former terminal tie-break is redundant and skipped only
when this stronger optimization mode is enabled. Full-horizon validation
without a supplied objective builds a local schedule rather than silently
rejecting every recovery plan for missing terminal targets.

Regression tests check analytic continuation gradients against independent
exact-source perturbations, a new wall just beyond the 750ms horizon affecting
both QP constraints and cost, authoritative world rejection, a coupled terminal
constraint's optimum with and without scaling, and terminal globalization.
Fixture captures preserve terminal planes using an optional tagged `TRM1`
extension; old v5 captures still round-trip without the extension. Python LP
and OSQP tools include the planes instead of silently dropping them.

Replacing repeated source finite differences with analytic propagation reduced
the observed three-repeat 20ms corpus from 3/18 to 15/18 completed R2 attempts
in the initial terminal model. The final physical-constraint corpus has 17/18
completed R2 attempts in terminal-only mode and 18/18 with directional source
curvature; every replay accepted a verified plan. Observed maximum cycle times
were 16.2624 and 16.2797 ms respectively. See
`physical_terminal_deadline20ms/corner_comparison.csv`. These are sample
measurements, not timing guarantees.
The initial funnel version failed before lap one
(`normal_sqp_terminal_20261101_01`, corridor at stage 0). The corrected physical
continuation model certifies all six saved corners, including corner303, in
unbounded replay (`physical_terminal/corner_comparison.csv`). Its live trial
`normal_sqp_physical_terminal_20261101_01` reached three lap boundaries then
failed near progress 105.961 m: invalid-model classification, one completed
QP, no accepted full plan. Therefore the terminal mode is **not qualified**.
A 75ms held-command backup is not a controlled invariant set or an
infinite-horizon safety proof. Jointly optimized successor controls and more
robust nonlinear feasibility restoration remain work to do.

## Actual convergence and live validation

Diagnostics and replay CSV now expose R1 cost, R2 cost, selected exact cost
gain, source directional derivative and applied source curvature. Infeasible
costs are not counted as convergence. Frozen-QP replay emits valid JSON even
when numerical divergence produces a nonfinite objective.

Normal repairs with both optional modes off:
`live_runs/normal_sqp_stability_20261101_01` completed one warmup plus ten scored
laps, ten-lap sum **49.5610 s**, zero recovery/prefix actions, no collision or
20ms gate violation reported, gate maximum **18.2000 ms**. Cross-track RMSE was
0.03832 m. R2 ran in 2223/2226 solver cycles and actually lowered the selected
frozen exact cost in **1884/2223 (84.75%)**. Median selected R2 cost gain was
1.33497. There were 304 selected zero-alpha line searches. Maximum reported
primal/dual residuals were 0.22677/0.22626. This is one successful race, not
repeatability or elimination of all convergence problems.

To reproduce the saved-state comparisons:

```sh
cmake --build build/f1tenth_mpc --target mpc_bag_solver_replay mpc_qp_fixture_replay -j 6
python3 -m tools.benchmark_mpc_normal_sqp --output live_runs/normal_sqp_check --repeats 3
python3 -m tools.benchmark_mpc_normal_sqp --output live_runs/normal_sqp_deadline_check --cycle-budget-us 20000 --repeats 3
ctest --test-dir build/f1tenth_mpc --output-on-failure -j 4
python3 -m pytest -q tools/test_mpc_qp_fixture.py tools/test_mpc_qp_oracle.py
```

Costs with terminal viability include the extra suffix and must not be
compared numerically with the original 750ms-only objective.

## Recovery audit and removal gates

| Existing condition / path | Cause primary optimization should address | Decision |
| --- | --- | --- |
| Auxiliary zero-rho Riccati seed fails | Numerical initialization, not physical infeasibility | Preserve positive-rho numerical retry and its regression. |
| Restoration residual explodes | Broken coupled Hessian | Fix diagonal scaling; do not remove physical recovery based on that fix alone. |
| Invalid full source step / steering or slip branch crossing | Finite-step model error | Shared adjoint and source curvature in normal SQP; keep exact backtracking. |
| Zero-alpha accepted nominal | Stagnation despite a safe incumbent | Measure exact gains; 304 occurrences in the normal race prohibit declaring this redundant. |
| Valid 750ms endpoint has no useful successor | Missing viability in finite horizon | Terminal value/constraints implemented experimentally; failed live trial prohibits recovery removal. |
| Viability early warning, minimal-nearwall, blended geometry, stage-targeted and optional witness seeds | Poor direction or threatened future clearance | Keep enabled policies; source mode's recoveries show normal formulation does not yet subsume them. |
| Current-candidate prefix / quality-aware prefix | Full plan rejected while a short action is still certified | Keep source/collision/residual checks and episode bound; short safety does not certify a successor horizon. |
| Retained-plan prefix / full remaining suffix | Deadline miss or failed new solve | Keep command-epoch alignment, fresh physical revalidation and consumed-plan cursor. |
| Full reacquisition after short-prefix episode | Avoid indefinite stale replay | Keep full-horizon requirement and solver/physical reseeding. |
| Stationary startup retry | No accepted plan and no remaining solver time | Keep stationary-only restriction; faster solves cannot remove scheduling faults. |
| Publication deadline, current-state validity, physical commands, body/world collision checks | Execution and physical validity | Essential; retained in all optimization modes. |

Only the separate source controllers and the terminal-mode duplicate tie-break
have become structurally redundant. No emergency collision/physical protection
was removed. The simulation-only continue-after-rejection diagnostic is not
used by any qualification run here.


Final validation: all 22 MPC CTest targets passed, including the eight new
SQP/terminal/warm-start regressions and coupled-slip restoration regression.
All 30 selected Python fixture/oracle/extraction/report tests passed. Full
build and `git diff --check` passed. All isolated trial containers were stopped.
No optional optimization mode was enabled in a production profile. The
physical-terminal live rejection was saved with the complete causal physical
state and nominal controls in
`live_runs/mpc_normal_sqp_unification_20261101/terminal_live_failure`.
