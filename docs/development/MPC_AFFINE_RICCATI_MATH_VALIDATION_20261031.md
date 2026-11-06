# Riccati affine cache: correctness and performance, October 31 2026

## Mathematical identity

For each horizon stage k, the factored Riccati RHS calculates:

    p_shift(k) = p(k+1) + P(k+1) * d(k)

Quadratic factor P depends on the fixed QP matrices and the ADMM penalty rho. Affine dynamics d is also fixed for a given QP. We now calculate P*d once per quadratic factorization, then reuse it during the repeated ADMM RHS updates. This changes float addition grouping and permits small rounding differences; it does NOT change the mathematical optimization problem, penalties, constraints or source-plant dynamics.

CRITICAL PUBLIC API BUG CAUGHT AND FIXED: The first attempt cached the affine term unconditionally. This was mathematically wrong for public riccati_solver_pass_factored(), which explicitly supports changing the d(k) vector and reusing unchanged quadratic factors. Two existing tests failed. Corrected implementation sets private_fixed_qp_affine_cache_enabled ONLY within the private riccati_solver_primal_pass() during one ADMM QP. Public factored calls continue computing the latest P*d, using the original accumulation order. Within private ADMM, penalty changes invalidate/rebuild P and refresh the affine cache. All public reuse tests then passed unchanged.

Source: f1tenth_mpc/src/riccati_solver.c and f1tenth_mpc/include/riccati_solver.h.

## Numerical validation

- Complete f1tenth_mpc test suite: **207 tests, zero failures** following the public API fix; do not weaken any tolerance.
- Eight true recorded IROS QP fixtures, 150 ADMM iterations: same solver status and iterations for every fixture, max individual solution-state delta 0.000629425 in mixed-scaled units, negligible primal residual differences (<=1e-6) and small dual residual changes (max around 0.000107). Input bounds and feasibility checks unchanged.
- 28 complete MPC saved-state replays with genuine vehicle internal state, full 750ms source-plant nonlinear acceptance, and two SQP passes: zero changed solver statuses, zero changed nonlinear acceptance/rejection decisions, zero changed SQP pass outcomes; first steering differs by at most 0.00000835 rad and first throttle by at most 0.0000191 command units. Source: live_runs/mpc_collision_margin_experiment_20261030/qp_affine_fullreplay_parity_20261031.json.
- Eight QP isolated p50 solve medians: baseline 3.83744ms, cache 3.566845ms, a 0.271ms or approximately 7.1% reduction, but NOT a 3ms worst-case ROS callback improvement. Source: riccati_affine_cache_revisit_20261031.json in the same experiment directory.

## Recorded Unity regression

- Uncapped 7.075s native IROS witness: live_runs/iros_7075_affinecache_20261031_01 completed two 8.855/8.994s laps then collided in the third; no missed deadlines, max callback 18.537ms. This does not establish repeatability and does not prove the cache caused the collision; the pre-cache vehicle controller was already inconsistent across independent IROS trials.
- Practice first run: live_runs/practice_guarded_affinecache_20261031_01 completed ten scored laps in 50.091s, zero collision or deadline miss, max callback 18.041ms, but failed the 49.5s practice time target. A separate identical practice repeat is recorded in live_runs/practice_guarded_affinecache_20261031_02 and affinecache_practice_repeat_20261031.json.
- A correct numerical solver is necessary but insufficient evidence for a robust closed-loop racing controller. Do not claim a new production 10-lap IROS result or complete five-run practice gate on this evidence.

## Worker scaling and other optimizations

Three repeat sweeps of 18 identical saved states gave 30-stage Jacobian wall times (mean): 1 worker 14.24ms, 2 workers 7.91ms, 4 workers 4.46ms, **6 workers 3.45ms**, 8 workers 3.72ms. Six was the fastest average for i5-13400F, so adding more is not automatically beneficial. See jacobian_worker_sweep_20261031.json.

Profile of difficult fixed-QP solves: dense Riccati factorization roughly 1-2ms, RHS forward/backward recursions roughly 1.5-3ms, projection 0.4-0.7ms, adaptive rho bookkeeping about 0.01ms. Exact-zero sparse indirect-index loops preserved bit parity but were much slower, and were reverted. Compiler -march=native and projection span caching also lacked consistent gains. Retain both SQP passes: globally skipping the second reduced timing but worsened practice to 50.139s.

No production YAML profiles were changed. The affine cache is retained in the solver source with correct private-path limits; this is a measured kernel optimization, NOT yet a fully racing-promoted configuration. Deadline target remains <=17ms worst observed callback, hard publication gate <20ms, full feasibility without shortcuts.

## Second practice repeat and optional-build decision
- practice_guarded_affinecache_20261031_02: independently completed ten scored practice laps in **49.9010s**, no collision or missed deadline, max callback 17.9427ms, six recovery actions. Like the first 50.091s run, this fails the strict <=49.5s practice gate. These runs do not show that the cache is mathematically wrong, but do not establish race-time equivalence either.
- Preserved the optimization behind the explicit CMake option SDU_APEX_MPC_FIXED_QP_AFFINE_CACHE (default OFF) in f1tenth_mpc/CMakeLists.txt, without altering any production YAML or any physical constraint. The option enables fixed-QP caching solely in private ADMM; the public factored Riccati API remains correct when d changes in either build.
- To test the optimized code in a separate ROS build: colcon build --packages-select f1tenth_mpc --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DSDU_APEX_MPC_FIXED_QP_AFFINE_CACHE=ON. To restore the original validated default: same command with option OFF.
- OFF full C++ suite: 207 tests, 0 failures; ON isolated Riccati public reuse/feedback suite: 5 tests, 0 failures. Earlier pre-switch source with the same optimized arithmetic and the same fixed public-API behavior also passed all 207 tests.
- This is a **retained useful experimental solver optimization**, not a reason to sacrifice the validated repeatable practice controller. Promotion still requires full lap-time and timing gates.
