# Fast exact-source SQP audit — 2026-11-03

**Status: offline implementation, not integrated into production ROS MPC.**
Scope is exclusively the mathematical QP, source-model derivatives, exact nonlinear objective, and convergence. No Unity race or weight tuning was performed.

## Main finding

The earlier so-called full second-order SQP required 120 perturbed full-source adjoint evaluations to form a 60x60 finite-difference source-cost Hessian for *every* iteration. This Hessian is not used by the infeasible-start Phase-I subproblem: Phase I minimizes correction/constraint slack before trying to improve racing cost. The source FD *constraint Jacobian* is essential to obtaining the observed good recovery; calculating the full cost Hessian during Phase I is redundant.

The optimized oracle still calls the exact checked-in source-derived plant, carries wheel speeds and all 22 hidden source-plant states, retains original physical constraints, uses exact-source merit for every accepted step, and never treats a locally feasible affine QP as source-feasible.

## Implemented

- Persistent C++ full-source merit evaluator in tools/mpc_offline_source_merit.cpp. Its G request evaluates cost+exact adjoint+strict source validation; Q evaluates cost+strict validation but skips adjoint; D produces accurate source cost/wall/rate for derivative-only probes with **strict_checked=false**, never eligible for publication. Baseline source cost, steering rate and all 60 future hard footprint margins matched exactly between request modes.
- Persistent C++ original-world geometry + 30-stage sensitivity/QP builder in tools/mpc_offline_exact_source_relinearize.cpp. Context loaded once. Binary QP output is SHA256-identical to one-shot reference.
- One fast offline optimization in tools/mpc_offline_fast_source_sqp.py. Exact finite-difference constraint Jacobian perturbations are distributed across eight *independent* persistent C++ processes; no unsafe shared physics state. The 60-input feasibility QP includes 60 actual source corridor margins, 120 actuator/rack and throttle slew margins, physical command bounds and an explicit Phase-I slack minimization. Every SQP output is still validated by the original exact source rollout, with second-order constraint correction.
- Source-linearized 60x60 control-space Gauss-Newton cost Hessian and damped BFGS updates for **already source-feasible** iterations. Full finite-difference source-cost Hessian is retained as an offline mathematical reference, not called in the fast Phase-I path.
- The exact source-mode constraint Jacobian includes all four actuator-rate faces per stage. An earlier missing/input indexing error was corrected. Full rate Jacobian comparison against source finite differences produced maximum errors around 1.3e-5 in tested controls.
- Monotone source filter first-accepted step globalization (with exact nonlinear merit) replaces exhaustive 11-step alpha enumeration when an acceptable step is found. Original strict source feasibility remains the decision authority, and a feasible incumbent is retained even when no subsequent cost improvement is accepted.
- A separate **tangent-only** source sensitivity option exists for comparison but is *not qualified*: high-slip constraint derivatives disagree strongly with direct full-source finite differences; it failed the main reference case after eight outer SQP iterations. A selective hybrid also underperformed. Neither is a valid faster production replacement.

## Measured speed and equality on the same computer (16 CPUs reported)

| Experiment | Frozen reference | Elapsed | Strict source feasible |
| --- | --- | ---: | --- |
| Original full finite-difference Hessian, 2 SQP iterations | epoch_1793692917333601071_case_6 | 3.26 seconds | Yes, iteration 2 |
| Optimized exact-source FD constraints (8 workers), 2 iterations | same | **0.123–0.156 seconds** over tests | **Yes, iteration 2** |
| Original full FD Hessian, 6 iterations | epoch_1793641756087085412_case_6 | 6.48 seconds | Yes, iteration 6 |
| Optimized same source-constraint reference, 6 iterations | same | **0.354 seconds** in fixed-run test | **Yes, iteration 6** |

The exact source objectives in the 2-iteration reference were **6531.9060 → 4739.1884 → 4478.4627**, with squared physical source violation **0.1530985 → 0.0072823 → 0**. All accepted reference outputs were reproduced by the faster code; in Phase I, the source-cost Hessian is not part of the QP.

Source evaluation using the persistent process: G (strict + gradient) median ~5.3 ms, Q (strict without gradient) ~2 ms, D (derivative-only, no strict validation) ~1.23 ms. The C++ QP relinearization server reduced the benchmark from ~12.0 ms one-shot to ~6.9 ms per fixture request. These are isolated overhead measurements, not guaranteed native cycle times.

Eight workers outperformed sixteen in the tested full-FD source constraint runs (more workers do not automatically help). The optimized solver remains far above the 20–25 ms live control deadline. The next speed improvements require a native source-sensitive 60-input solver and direct derivatives of *all* physical source-plant internal states, or a mathematically equivalent reduced full-state derivative method. Simply dropping derivative columns from the QP is not acceptable: this was measured to worsen nonlinear convergence substantially.

## Six frozen failure cases tested with eight source SQP iterations at most

| Case suffix | First feasible SQP step | Final squared original source violation |
| --- | ---: | ---: |
| 1793641756087085412_case_6 | 6 | 0 |
| 1793692853295529252_case_6 | 1 | 0 |
| 1793692917333601071_case_6 | 2 | 0 |
| 1793693111767116537_case_6 | none | 2.42838 m² |
| 1793693111641647593_case_2 | 6 | 0 |
| 1793693111867025977_case_10 | none | 0.47540 m² |

Total **4/6 cases source-feasible** in the saved source-only corpus, and 2/6 not recovered. The last two have much larger initial trajectory violations; the result is not an excuse to publish an invalid iterate or to bypass collision checking. They need further source-accurate nonlinear formulation work and proper feasibility classification.

## Required next correctness and performance work

1. Keep the hard physical model and original cost frozen. Profile and implement a **native in-process** 60-command constrained QP with exact primal feasibility and deterministic allocation. Remove repeated Python/SciPy/OSQP orchestration only after source and QP result parity is demonstrated.
2. Correct the *source dynamics sensitivity* discrepancy, rather than just changing the margin. The 23-state QP currently omits 22 hidden PhysX/source-plant state coordinates, even though the exact rollout carries them. High-slip tangent derivatives can diverge from full-source finite differences. Implement a full-source-state forward sensitivity/adjoint or other validated differentiation of wheel/suspension states; test all 60 command-direction derivatives on captured corners.
3. Once the numerical/local source Jacobian is correct, use one or two source-exact SQP iterations with a deterministic feasibility-first filter and optional second-order constraint correction. The fast reference proves this can be effective; it does not establish a uniform two-iteration bound. Implement cost Hessian as source-derived GN with conditioned BFGS/secant updates only after source feasibility, and compare predicted-versus-actual cost on all source-valid trajectories.
4. Before live promotion, prove 30-stage exact-source feasibility and objective prediction agreement on the complete frozen corpus, including unrecovered failures. Then verify terminal successor viability and Unity/source state discrepancy. **Only afterward** benchmark the live 20–25 ms deadline.

## Reproduction

The two source-only C++ tools are standalone offline development utilities compiled against the current repository's libmpc_core.a. The experimental Python needs scipy, numpy and an OSQP vendor. No production controller behavior is changed by these tools.

- Fast correctness regression: python3 tools/test_mpc_offline_fast_source_sqp.py
- Original synthetic quadratic/QP regression: python3 tools/test_mpc_offline_second_order_source_sqp.py
- Frozen source corpus results: /tmp/mpc_fast_source_sqp_20261103/full_corpus_summary.json
- Regression pass includes G/Q/D exact-value parity, source/geometry nominal parity, true 120-row actuator Jacobian, identical C++ one-shot/persistent QP binary, and two source-feasible SQP steps with exact objective parity.

Existing working tree includes substantial earlier uncommitted work, which must not be overwritten or mixed blindly into a commit.

## Frozen-replay interpretation and stagewise prefix audit (2026-11-03)

A frozen context starts from a state produced by historic controller commands. It cannot model what an improved controller would have done at earlier control epochs. Therefore, a long-horizon suffix collision in a frozen replay is not, by itself, proof of the improved MPC's closed-loop failure. Equally, a source-feasible frozen full-horizon command plan does not prove recursive feasibility or Unity/source agreement.

The new tools/audit_mpc_source_sqp_prefix.py independently reevaluates saved optimized command iterates using the full C++ source model and reports the first future stage with an original physical footprint corridor violation. Across the six tested frozen corners, the source-feasible prefix improved in all six:

| Captured case | Original feasible stages / 30 | Optimized stages / 30 | Final full horizon source-valid |
| --- | ---: | ---: | --- |
| 1793641756087085412_case_6 | 9 | 30 | Yes |
| 1793692853295529252_case_6 | 8 | 30 | Yes |
| 1793692917333601071_case_6 | 22 | 30 | Yes |
| 1793693111641647593_case_2 | 25 | 30 | Yes |
| 1793693111767116537_case_6 | 20 | 21 | No |
| 1793693111867025977_case_10 | 16 | 22 | No |

The two still-infeasible full horizons reduced squared source wall violation by approximately 72% and 98%, respectively. These are meaningful same-epoch candidate improvements, not proof that an incomplete plan is safe to publish.

Next correctness gate: measure *per-stage QP-predicted versus exact-source* state, cost and physical constraint errors on the prefix **before the original first failure**. Separately evaluate whether the first applied command preserves a feasible successor state. A longer feasible prefix is necessary, but not a replacement for first-action safety, recursive feasibility or exact nonlinear validation. Closed-loop simulation should be reserved until QP/source consistency is qualified.

Audit file: /tmp/mpc_fast_source_sqp_20261103/prefix_analysis.json
