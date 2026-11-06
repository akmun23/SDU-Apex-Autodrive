# Source SQP continuation after a63e30608 — 5 November 2026

## Scope and baseline

Continues commit `a63e30608f13da0948e9d7b8f5a95c6d0517d5db` on
`mpc/next-engineering-handoff-20261021`. No original model, objective,
actuator limits, track profiles, hard physical inequalities, real-time
publication policy or fallback architecture was changed.

The recorded competition corpus is
`f1tenth_mpc/test/data/competition_corner_20261105/`. The unchanged
original-source replay scores all 25 frozen feasible witnesses, including
record 775. Feasibility of a witness does **not** mean the native SQP can find
it, prove recursive viability, satisfy KKT, meet 22 ms, or match Unity.

## Correctness guard for persistent SQP curvature and duals

The exact compact solver had guarded only `program->rows.size()` when carrying
hard-constraint dual multipliers and Lagrangian BFGS memory from one accepted
iterate to the next. Equal row count does not guarantee equal physical
constraint identity at a switched source/reference branch.

Added `same_original_source_constraint_layout` to compare every row's
`type, stage, index, source_fixed_tick, source_vehicle_substep`.
Changing **only** a numerical margin preserves identity. On any identity
change the solver resets carried duals and Hessian memory without modifying
the source-integrated primal. A dedicated C++ test exercises same-size branch
changes and one-sided terminal cut insertions/removals. Offline output now
reports `source_constraint_layout_resets` and each iteration's reset.

On all 22 frozen recorded competition states (8 SQP iterations), numerical
outcomes matched pre-guard replay, **12/22 returned source-feasible**,
and 0 identity resets were needed. This is a correctness guard, not a new
convergence claim.

## Research-only rejected source-gradient BFGS update

Opt in only with `MPC_SOURCE_NLP_REJECTED_GRADIENT_BFGS=1` in the offline
executable. `OfflineOptions.rejected_source_gradient_bfgs` defaults to
**false**, and no ROS parameter enables it.

On an original-source-integrated **rejected full SQP step**, reuse that exact
primal and compute its own source adjoint/reduced original racing-cost and
physical-inequality gradients. Freeze the existing SQP curvature multipliers,
use the same ordered physical constraints, and form a scaled gradient secant
`y=grad(L,U+d)-grad(L,U)`, with `s=(U+d-U)/control_scale`.
Update the existing 60x60 positive-definite Lagrangian BFGS metric with
Powell damping; reject numerical non-finites, vanishing/ill-conditioned
secants and any recorded original source branch-flag change. The normal
source-AD and original source rollout remain authoritative. No step-size
grid, extra nonlinear physical candidate, input trust box, second controller
or alternative NLP is introduced. The earlier scalar merit secant is not
also fitted to an accepted gradient update.

The original source-branch flags did not change during any of the successful
gradient updates on the two recovered late corner states. This is narrower
than a proof of smoothness across every tire/contact switching surface.
Other nonsmooth branches and float-wrapper gradient noise still require
further source verification.

## Independent competition results

All rows below are offline, full-precision *source* replays of the same
recorded original NLP. These are **not live SQP settings**. None met the
original constrained NLP KKT termination test.

| Epoch | 20-iter default SQP | 20-iter rejected gradient BFGS | Verified gradient result |
| --- | --- | --- | --- |
| 757 | source-feasible | source-feasible | 0.03024 m minimum clearance |
| 765 | infeasible, -0.32577 m clearance | **source-feasible** | +0.120074 m, cost 12368.681 |
| 769 | infeasible, -0.20350 m clearance | **source-feasible** | +0.001567 m, cost 10376.627 |
| 775 | infeasible | **still infeasible** | -0.24613 m; feasible witness exists independently |

Epoch 765 first reaches source feasibility at SQP iteration index 8; epoch
769 at index 17, well beyond the live request of 3 SQP passes.
The two recovered controls were passed to the independently built
`mpc_source_nlp_oracle`: original racing cost parity and nonnegative original
body, rate and input-box margins all passed. Epoch 769's 1.57 mm clearance
is too small to claim robust Unity safety.

Individual rejected-gradient AD evaluations cost roughly 3.4–4.1 ms
in these **offline** experiments; cumulative extra AD was around 23 ms
for 765 and 32 ms for 769. Do not enable the research option in the
22 ms runtime on the basis of these results.

**Combined elastic SQP + source-gradient BFGS** was also tested at 20 and
60 iterations. It remained infeasible for 775; 769 required 60 iterations
to recover, while 765 improved in cost. The combination is not uniformly
better and stays offline only.

The original frozen 22-state benchmark passes replay/independent witness
checks. A new native regression in `tools/test_competition_failure_benchmark.py`
reconstructs the recorded problems, runs BFGS recovery on epochs 765/769,
and independently scores the returned control arrays with the source oracle.
These tests guard correctness, not production time.

## Validation

- Source/controller and replay tools build.
- **14/14 CTest suites pass**, including the new constraint-identity C++ unit
  test, source oracle replay, native capture, and gradient-recovery regression.
- The original 22-state replay is unchanged with research mode disabled.
- No Unity/ROS races, 22 ms scheduling stress tests, source/Unity fidelity
  requalification, or source-NLP KKT qualification have been performed after
  the latest GitHub commit.

## Next mathematical engineering work

1. **Infeasible linearizations at record 775**: the convex subproblem's
   primal and dual residuals can explode while an independent source-feasible
   witness exists (only +0.0855 mm margin). Investigate constraint scaling,
   active hard-row degeneracy and primal-dual Newton conditioning. Do not
   count a source witness as proof of a numerically feasible QP.
2. **Cost-effective true curvature**: compare the rejected-gradient secant
   with stage-local analytic source second-order directional derivatives,
   including tire force and active body-corner constraints. Validate
   semismooth branch selection, true frozen-multiplier KKT curvature and
   float-wrapper precision on the recorded corpus.
3. **Few-iteration feasibility**: 765/769 currently recover only after
   >3 SQP iterations, so improve the normal first/second SQP directions and
   terminal/source viability, not only total iteration allowance.
4. **Real-time profiling**: measure exact source AD, QP factorization,
   accepted/rejected source rollouts and complete ROS publication against
   the 22 ms budget. Do not trade away original-source physics fidelity.
5. **Race qualification**: repeat full Unity practice and competition races
   only once source-corpus recovery and deadline gating pass. Maintain
   competition and practice weights/profiles independently.

## Reproduction

```bash
cmake --build build/f1tenth_mpc --target \
  mpc_source_exact_offline_optimize mpc_source_nlp_oracle \
  mpc_source_live_fixture mpc_source_feasibility_witness_probe \
  mpc_source_benchmark_capture_check mpc_controller_node -j4
ctest --test-dir build/f1tenth_mpc --output-on-failure

python3 tools/competition_failure_benchmark.py replay \
  --output artifacts/competition_post_a63e_replay --iterations 8 --repeat 1

MPC_SOURCE_NLP_REJECTED_GRADIENT_BFGS=1 \
build/f1tenth_mpc/mpc_source_exact_offline_optimize \
  artifacts/competition_post_a63e_replay/epoch_769 - 20 \
  artifacts/competition_post_a63e_replay/gradient769.controls
```
