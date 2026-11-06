# Source-only MPC development architecture (2026-11-05)

## Mandatory design
The runtime executes exactly one 30-stage original-source nonlinear racing
program per 40 Hz epoch. Its 60 physical steering/throttle controls are
optimized via iterative SQP. The QP is a local constrained direction generator;
all reported accepted primal trajectories are reintegrated and scored by the
original source physics, racing objective and physical inequalities.

An SQP iteration has exactly one **full** physical correction evaluation
(alpha = 1). Accepted iterates are relinearized. Rejected steps increase
positive-semidefinite Hessian regularization by 3x, keeping the SAME
original constrained nonlinear program and reusing its same-iterate source AD
Jacobian. Each rejected direction is counted as an SQP iteration; no alpha
grid or alternate fallback controller is introduced. Rejections are recorded
in the nonlinear iteration history; final failure is exposed explicitly. The previous-iterate steering/throttle trust box,
fixed-alpha sample searches, alternate optimization directions, reduced
Riccati/ADMM controllers, retained-plan publication, recovery/fallback
controllers and multiple-shooting experiment are not part of the build.
The numerical QP itself still runs multiple internal Newton iterations.

Production C++ entrypoint: `mpc_rti_solve_cycle` in `mpc_rti.cpp`.
Single nonlinear solver: `mpc_source_exact_offline_solver.cpp`.
Original dynamics and sensitivities: `vehicle_model.cpp`,
`unity_source_plant*.cpp`, `mpc_source_sensitivity.cpp`,
`mpc_source_exact_nlp*.cpp`.
Local constrained QP: `mpc_source_exact_qp.cpp`.
ROS control owner: `mpc_controller_node_impl.hpp`.
The source-only architecture is asserted by
`tools/test_source_only_architecture.py` (part of CTest).

Practice and competition profiles contain their own original racing weights,
plus 3 cm additional source clearance and the 22 ms full-cycle budget.
Profiles are **development candidates**, not race-qualified controllers.

## Reproducible qualification

```bash
cmake -S f1tenth_mpc -B build/f1tenth_mpc
cmake --build build/f1tenth_mpc -j 6
ctest --test-dir build/f1tenth_mpc --output-on-failure

# Same source evaluator, one exact full SQP step per iteration
MPC_SOURCE_NLP_QP_REGULARIZATION=1000 \
MPC_SOURCE_NLP_QP_SOFT_WEIGHT=1000000 \
MPC_SOURCE_NLP_MIN_CLEARANCE_M=0.03 \
build/f1tenth_mpc/mpc_source_exact_offline_optimize \
  live_runs/mpc_source_second_order_20261104_01/case_16 \
  tools/source_exact_fixture_seeds/case_16.controls 6

# Actual mpc_rti_solve_cycle, strict source replay; not Unity
MPC_SOURCE_NLP_QP_REGULARIZATION=1000 \
MPC_SOURCE_NLP_QP_SOFT_WEIGHT=1000000 \
build/f1tenth_mpc/mpc_source_exact_live_receding_qualifier \
  live_runs/mpc_source_second_order_20261104_01/case_live_fast_17 \
  tools/source_exact_fixture_seeds/case_live_fast_17.controls \
  40 3 1 22 0.03 0
```

## Verified single-problem SQP status (2026-11-05)

- The local QP ALWAYS carries the original racing objective gradient, even
  when the original trajectory violates physical constraints.
- Original source-model dynamics are differentiated and condensed; body
  footprint, actuator-rate and input-box rows belong to one nonlinear
  optimization problem. Input boxes are hard; footprint/rate rows currently
  use an augmented-Lagrangian local QP and exact nonlinear qualification.
- Original physical inequality multipliers are carried between SQP iterations
  and updated using the newly integrated ORIGINAL source margins.
- Every SQP iteration evaluates at most one full nonlinear correction.
  No alpha grid, separate feasibility-first optimization, recovery planner or
  controller fallback remains in the live runtime.
- New original-NLP stationarity, inequality-feasibility and complementarity
  diagnostics distinguish a feasible iterate from a converged KKT point.
  **The recorded corner fixtures have not demonstrated KKT convergence.**
- The ROS controller now reports original NLP KKT convergence and
  full-step stagnation separately from physical feasibility.
- 11/11 retained CTest tests pass, including two original-source corner
  regressions and a check guarding against reintroduction of retired solvers.

The last newly built 40-cycle offline source-model receding replay showed
case_16: 5/40 source-validated publication cycles, maximum observed
6.467 ms; case_live_fast_17: 40/40, maximum observed 20.27 ms. These
latencies are observations, not an operating-system worst-case guarantee,
and include no Unity races.

## Important open blockers
- The live-fast-17 source replay can publish all 40/40 epochs within 22 ms
  on a quiet pinned CPU, but this is **not an absolute worst-case guarantee**.
- With 1,000 initial regularization, three same-NLP full SQP iterations and
  3 cm original-source clearance, **both known fixtures completed 40/40
  source-model receding epochs within 22 ms in an initial pinned-CPU run**.
  Five repetitions and Unity race tests are still required.
- Source-feasible is NOT KKT-converged. Source-original stationarity and
  complementarity are reported and must be reduced without sacrificing
  physical feasibility or hard real-time timing. A terminal successor
  viability formulation is not yet certified.
- Physical rate and corridor constraints currently use a penalized elastic
  formulation in the local QP; original hard source feasibility remains
  authoritative only at nonlinear acceptance and publication.
- The original Unity/source high-slip/contact mismatch is unresolved.
- No post-cleanup sustained Unity racing qualification has been performed.

The numerical/runtime public interfaces no longer contain Riccati/ADMM
solver types or alternative solver settings. A small number of older diagnostic
POD fields remain only because historical source replay fixtures use
their exact binary layout. Migrate those fixtures to a source-only versioned
schema before deleting the last diagnostic compatibility fields.

The pre-cleanup uncommitted worktree is backed up outside the repository at:
`/home/akselmo/Documents/autodrive_repository_snapshots/`.
