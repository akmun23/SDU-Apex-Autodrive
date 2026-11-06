# Source-model receding-horizon QP counterfactual — 2026-11-03

## Motivation

A frozen QP is an exact local solver test, not proof that online control cannot prevent the associated failure. Earlier commands in a logged sequence can move the vehicle into a state that is difficult or impossible to recover at a later fixed epoch. A decreasing exact-source constraint-violation metric is a genuine optimization improvement even when zero violation is not reached for a frozen historical state.

## Implementation

- tools/mpc_offline_advance_source_epoch.cpp: validates ONLY the selected first command against original source constraints, propagates precisely one 25ms source stage (including full wheel/suspension/actuator state and route-relative progress), and generates a fresh context, objective schedule and hard QP from the resulting state. It never claims the full horizon is feasible based on the one-step check.
- tools/mpc_offline_counterfactual_closed_loop.py: iterates exact-source SQP, applies a single selected steering/throttle action, advances the source vehicle, shifts the planned inputs, rebuilds the next epoch and repeats. A matched comparison instead shifts the original seed without optimizing. All later states are actual source-model counterfactual outcomes, not historical replay states.
- tools/test_mpc_offline_counterfactual_closed_loop.py: two-epoch causal regression, checking different controls and progress, new source contexts/QPs, preserved full-horizon strict validator results and first-step source validation.

## Numerical evidence

From late failure snapshot epoch_1793692917333601071_case_6:
- Shifted original controls: 0/5 full-horizon source-feasible, squared violation 0.153098 -> 0.237424 m².
- Optimized two-SQP controller: 5/5 full-horizon source-feasible, zero squared corridor violation at every selected predicted horizon.
- Both branches had first-step source validity for all five steps. Thus individually valid first outputs alone do not guarantee feasible future horizons.

From earlier snapshot epoch_1793692917233236533_case_2 (about 100ms / four control epochs earlier):
- 5-step matched counterfactual: optimized SQP 5/5 source-feasible full horizons; shifted controls 0/5; both 5/5 individually valid first steps.
- 20-step matched counterfactual (0.500 s modeled motion): optimized SQP 20/20 source-feasible full horizons and 20/20 legal first steps; shifted original controls 0/20 feasible full horizons despite 20/20 legal first steps. The shifted plan's largest squared violation was approximately 0.000105674 m². Each branch recomputed physical source state from its own earlier controls.
- The optimized branch is NOT a one-shot repair of the same logged state. It demonstrates the prevention mechanism in the source model over successive replans.

Local diagnostics: /tmp/mpc_counterfactual_prior_optimized_20/results.json and /tmp/mpc_counterfactual_prior_shift_20/results.json. Outputs are local development diagnostics, not committed artifacts.

## Qualification limits and next work

1. These are source-model-only experiments, not Unity/PhysX WheelCollider verification, a real-time 40Hz controller, map swept-volume certification, a terminal recursive-feasibility proof or complete races. Full source predictions do not guarantee Unity agreement.
2. A source-infeasible 30-step intermediate is allowed to seed the next offline SQP iteration, and its reduced violation is logged. It is NOT automatically publishable. The first applied command is separately source-validated, and a feasible incumbent is retained when available.
3. Preventability of late failure should be measured with causal pre-failure counterfactuals across multiple initial states and both track geometries; judge the solver separately by mathematical QP parity, source constraint Jacobian accuracy, exact-source cost and hard feasibility.
4. The major performance/correctness task remains a native in-process 60-control source-sensitive QP with accurate derivatives through all wheel/suspension/internal states and exact-source globalization, preserving the 1–2 SQP iteration successes without 120 command-direction finite-difference probes per SQP step.
5. No online control path was changed or race run in this task. The remote Git working tree contains prior uncommitted work; do not overwrite it.

## Reproduction

- Run: python3 tools/test_mpc_offline_counterfactual_closed_loop.py
- Run: python3 tools/test_mpc_offline_fast_source_sqp.py
- For full comparison: run tools/mpc_offline_counterfactual_closed_loop.py twice from the earlier case above with --cycles 20 and --iterations 2; choose default optimize versus --strategy shift_seed. Each run must use the current built matching source-merit, QP-server and one-step-advance C++ binaries.
