# MPC solver, rollout, recursive-feasibility work — October 31 2026

## Nonnegotiable constraints
Keep the full 30-stage 750ms native/source plant, both SQP passes, exact actuator command queue, hard corridor and body constraints, and the 20ms publication deadline. Do not make an unchecked fallback the primary controller, loosen physical margins, or alter protected production YAML. Preserve the independent five-run practice winner (~49.371s mean for ten scored laps).

## Measured computation hotspots
- Full-Riccati 150-iteration fixed-QP benchmark: factorization ~1-2ms, factored RHS ~1.5-3ms, projection ~0.4-0.7ms. Native compiler ISA and exact-zero indirect sparse indexing did not consistently speed up recorded cases. Avoid blind sparse loops.
- OpenMP Jacobian worker sweep on 18 recorded IROS states and three repetitions: 1 worker 14.24ms, 2 workers 7.91ms, 4 workers 4.46ms, 6 workers 3.45ms, 8 workers 3.72ms average. Keep 6 on the tested i5-13400F, unless further p99 evidence warrants another number.
- Private fixed-QP affine Riccati caching P[k+1]*d[k] mathematically valid, ~7.1% median solver speedup (~0.27ms) across 8 recorded QPs; 28/28 saved-state nonlinear MPC decisions unchanged. Public API must continue accepting changes to d under quadratic-factor reuse. Opt-in CMake flag SDU_APEX_MPC_FIXED_QP_AFFINE_CACHE=ON (default OFF), because two practice Unity runs completed but took 50.091 and 49.901s, above strict 49.5s. Do not confuse numerical correctness with racing performance.
- Adaptive rho remains needed for numerical residual performance. Global skipping second SQP gave 50.139s practice and is rejected.

## R1 vs R2 is essential
Use the new refinement_full_horizon_selected_pass field, not the old short-prefix-only refinement_prefix_selected_pass counter. Identical saved-state full MPC replay with two passes: 7.050 IROS failures showed 7/10 accepted full horizons selected from R2, 2/10 R1, 1 no full horizon. 7.075 hairpin failures found no full horizon in 18/18, explaining need for nonlinear candidate/terminal feasibility improvements. Logs in live_runs/mpc_collision_margin_experiment_20261030/full_horizon_selection_*_current.csv.

## Implementation priorities
1. Instrument per-stage Riccati factorization and forward/backward RHS and per-pass R1/R2 convergence, objective, controls, feasibility, iterations and wall/CPU time. Optimize contiguous dense matrix access, invariant products with exact valid scope, and avoid temporary copies; preserve convergence and validate the real full SQP and rollout, not bit parity.
2. Audit Jacobian source-plant step evaluation, duplicate reference lookups and state serialization. Remove redundant evaluations only if every physical state, tracking Jacobian and exact corridor check remains correct.
3. Replace emergency recovery's heuristic station prediction s0+0.55*u0*t with **causal source-model-propagated station** under the candidate throttle/steering and queued actuator commands. In real rollout velocity is already simulated; only seed generation uses the inaccurate heuristic.
4. Design recursively viable nominal/terminal behavior. An exact-valid 750ms plan can end at a physically clear but unrecoverable state. Verify at least another 750ms viable successor in stored failure states, then integrate a computationally bounded viability indicator/terminal condition into normal SQP rather than triggering hundreds of last-minute recovery plans.
5. Measure p99 and maximum end-to-end callback, independently repeat five complete ten-scored-lap practice races and multiple complete ten-scored-lap IROS races without collision, fatal rejection, or 20 ms publication misses. Lap time is diagnostic only until the runtime and robustness work is finished. One previous IROS run completed in 83.938s but had one 21.865ms deadline miss and 386 recoveries. Target is consistent 20ms-compliant execution, ideally <=17ms worst measured, and reduce recovery frequency, not hide numerical failure.

## Safety/guarantee limits
An MPC cannot guarantee noncollision from arbitrary irrecoverable states or under unrestricted OS preemption. Correct approach is proactive viability and predictable bounded compute, with a well-defined fail-closed policy when no physically safe command exists. Avoid claiming that accepting unchecked commands removes full shutdowns safely.

## Latest check
Current repository HEAD 43312ea19. MPC builds and complete test suite passes 209 tests, zero errors/failures. Independent optimizer remains running; do not disturb its files or background process.
