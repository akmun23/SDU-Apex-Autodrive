# Unified MPC Riccati rewrite — 31 October 2026

## Result

The production Riccati/ADMM numerics now use **one** fixed-size, stack-only C++/Eigen implementation for:
- 23x23 Riccati quadratic update P[k] = Q[k] + A^T P[k+1] A + G^T K (symmetric upper triangle mirrored).
- Full 30-stage factored Riccati backward RHS and forward state/control rollout (one C ABI call per full horizon, **not** one per stage).
- Fixed-QP affine cache P[k+1]*d[k] computed once per factorization and reused each ADMM RHS iteration, invalidated on rho refactorization.
- Public one-shot Riccati/cold start calls the same factorization/RHS pathway instead of a separate 202-line C implementation. Public factored callers may change d while reusing quadratic factors: private cache flag is never activated for them.

Old C quadratic and factored RHS loops and C/Eigen/affine-cache build switches were removed. There is no selectable old numerical backend remaining in the production target. The C ABI, source-plant simulation, all state/control channels, constraints, convergence checks, adaptive penalties and **both R1 and R2 SQP passes** remain intact.

Source files:
- f1tenth_mpc/src/riccati_eigen_kernels.cpp — sole fixed-size dense kernel implementation, Eigen::Map on contiguous C arrays, no heap, no C++ object crosses the ABI.
- f1tenth_mpc/src/riccati_solver.c — ADMM/QP orchestration, rho updates, constraints, factorization and public C wrapper.
- f1tenth_mpc/CMakeLists.txt — links Eigen3 directly; Dockerfile already installs libeigen3-dev, no new Docker dependency.
- A deliberate C API remains because it is also used by the source-plant/QP replay tools and ROS integration. C++ itself does not make arbitrary code faster; measured SIMD and data layout gains motivated this migration.

## Controlled numerical benchmarks

Eight true saved IROS QPs; 23 states, 2 controls, 30 stages, up to 150 ADMM iterations; same pinned CPU8 and controlled background optimizer, three interleaved repetitions for each of the eight fixtures.

| Code | Median per-QP p50 (24 comparisons) |
| --- | ---: |
| Frozen original pure C solver (separate /tmp snapshot) | 3.751 ms |
| Consolidated full Eigen solver, affine cache off | 2.805 ms |
| Consolidated full Eigen solver + fixed-QP affine cache | 2.515 ms |

The independently measured improvement is approximately **0.95ms per difficult QP** without the cache and a further **0.17ms per QP** from caching; comparisons were made in separate controlled sessions. Total improvement relative to frozen original C is approximately 1.2ms/QP, around one third of QP time. The second SQP pass is retained, so multiple expensive QP solves per callback may benefit. This is NOT a guaranteed 3ms worst-case callback improvement.

Detailed artifacts:
- live_runs/mpc_collision_margin_experiment_20261030/consolidated_eigen_matched_qp_benchmark_20261031.json
- live_runs/mpc_collision_margin_experiment_20261030/eigen_affine_matched_qp_benchmark_20261031.json
- Earlier matrix/RHS microbenchmark raw logs in same directory; obsolete standalone benchmark .cpp files removed to keep repository focused on full-QP and full-state replay.

## Correctness gates

- Entire f1tenth_mpc suite: **209 tests passed, zero failures** after unifying all public/factored/cold-start calls.
- For original C vs combined Eigen cached version on 8 frozen QPs: identical status, 150 iterations, essentially unchanged primal residual and small dual/input residual floating differences; no relaxation of any tolerances.
- Separate original C vs unified **cold** and **recorded warm** starts on 8 frozen QPs: unchanged status and iteration counts, primal residual difference <=1e-6, dual residual differences <=7.06e-5; input bound differences <=1e-6. Existing public changing-affine factorization tests pass.
- 28 complete saved-state RTI/SQP nonlinear source-plant replays: when comparing the Eigen implementation before/after affine caching, zero changes in nonlinear acceptance/rejection or SQP outcome, max first control differences ~2.2e-7rad and 2.83e-7 throttle units. Comparing original C with full Eigen without caching, 28/28 nonlinear decisions likewise unchanged.

## Live Unity results (R1/R2 retained, 11 target lap crossings)

Two full-configuration experimental trials were run before removal of the legacy cold-start C method; removal was then validated by unit and frozen cold-start QP tests.

**Full Eigen (cache off)**:
- IROS native 7.075s witness: 9 lap crossings, then exact nonlinear horizon rejection (corridor at stage11), **zero collision**, 0 missed >=20ms deadlines, max callback **19.207ms**. Not a complete competition 10-scored-lap race.
- Practice: 11 crossings, ten scored laps **50.339s**, 0 collision, 0 deadline misses, max callback **19.317ms**. 13 recovery actions. Fails strict <=49.5s race performance gate.

**Full Eigen plus cache (now standard)**:
- IROS native 7.075s witness: 7 lap crossings, then exact nonlinear horizon rejection (corridor at stage20), **zero collision**, 0 missed >=20ms deadlines, max callback **18.909ms**. Not full competition distance.
- Practice: 11 crossings, ten scored laps **50.559s**, zero collision, zero missed deadlines, max callback **19.321ms**, 14 recovery actions. Fails strict <=49.5s target.

These racing variations existed with the previous C implementation as well. Neither numerical backend demonstrates repeated full-distance IROS reliability, and weight tuning and source-map-constrained SQP are still required. Do not claim that the numerical rewrite fixes wall feasibility, lap time or guarantees 17ms.

## Remaining work (separate changes, do not undo R2)

1. Audit duplicated source-model work in Jacobian/nominal/candidate rollouts. Keep all per-stage dynamics and physical checks. Six OpenMP Jacobian workers were already fastest mean on this CPU (3.45ms vs eight 3.72ms).
2. Upgrade SQP corridor constraints to reflect conservative **real occupancy/world-body geometry**, rather than optimizing Frenet halfplanes and discovering world-body overlap after optimization. Preserve full independent map validation.
3. Add terminal viability to normal SQP: exact-valid 750ms trajectory need not be safe at 775ms; avoid entering states with no feasible next horizon instead of masking them with unchecked fallback commands.
4. Separately tune racing weights and reduce 10-lap runtime while maintaining hard 20ms deadline. Reference practice five-run winner averaged 49.371s. Required eventual promotion gate: 5/5 independent practice races <49.5s and multiple clean ten-scored-lap IROS runs with no physical collision, no fatal rejection, and no deadline misses.

## No claimed hard guarantee

An unbounded OS scheduler load can delay any userspace process, and from physically unrecoverable high-speed states no controller can guarantee collision-free driving. The appropriate improvement is more accurate recursive feasibility and lower bounded per-cycle work, not disabling physical validation or forcing unchecked commands.

The autonomous IPOPT optimizer was protected throughout simulator tests and resumed afterward. User production racing YAML was not replaced. The source code numerical solver migration is complete, but closed-loop behavior remains to be tuned and repeatedly validated.

## Final integrated cold-start live regression and source cleanup

After the first combined Unity trials, removed the 202-line public one-shot C Riccati backward/forward pass. Public one-shot and ADMM cold start now call the same factorization and fixed-size Eigen RHS as warm ADMM. The public API remains safe when d changes between calls; the private affine cache is enabled only for a fixed-QP solve.

- Post-removal C++ full suite: **209 tests passed, zero failures**.
- Eight frozen real QPs with both recorded and freshly cold-initialized solver states: identical status and iteration count against original C, <=1e-6 primal residual change, <=7.06e-5 dual residual change, <=1e-6 input bound difference.
- **Final integrated practice Unity trial:** live_runs/practice_guarded_unifiedeigen_coldstart_20261031_01, 1 warmup + ten scored laps, total **49.905s**, 6 accepted recovery actions, zero collisions, zero full-horizon shutdown, zero >=20ms missed publication deadlines, maximum callback **18.111ms**. Still above strict <49.5s racing target; NOT a five-run performance promotion.
- There has NOT been a new full-length IROS Unity run with the final cold-start wrapper (previous combined build reached seven lap crossings before a genuine corridor@20 nonlinear horizon rejection). Future IROS validation must use the final consolidated code.
- The total diff to f1tenth_mpc/src/riccati_solver.c is a net reduction of about 300 lines of duplicate C code; the maintained fixed-size Eigen source is 111 lines. There are no C/Eigen solver selection flags nor fixed-QP affine cache flags left in CMake or the numerical solver. This is a direct replacement, not an experimental parallel implementation.
