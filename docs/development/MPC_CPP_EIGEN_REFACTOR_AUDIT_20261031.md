# C++ / Eigen refactor audit — 31 October 2026

## Objective and constraints

Keep full 30-stage (750ms) nonlinear PhysX/source-plant prediction, both SQP passes (R1 and R2), 23-state/2-input Riccati OCP-QP, hard 5cm/vehicle-body constraints, 40Hz actuator queue, and <20ms command-publication deadline. Seek <=17ms worst observed under representative load. Keep protected five-run practice baseline and production YAML unchanged.

## Footprint state

The existing C++ map checker already precomputes EDT and a conservative 64x64 subcell lookup, applies a whole-body bounding-circle early certificate, then the full perimeter if necessary. On 317 recorded IROS source-step swept pose transitions: mean 19.7us -> 3.8us, median 17.3us -> 0.62us, p95 37.2us -> 15.3us; 314 pass / 3 reject remain identical. This certifies sampled step-endpoint interpolation, NOT all intermediate unknown PhysX substeps. Map authority is still primarily a post-SQP physical validator, not a constraint in the QP.

## New controlled numerical experiments

Created tools/benchmark_mpc_matrix_kernels.cpp and tools/benchmark_mpc_rhs_kernels.cpp; use actual 23x23 stage A and 23x2 B plus affine d from eight saved IROS QP fixtures, representative P and RHS vectors. These are standalone kernel benchmarks with malloc-disabled fixed-size row-major Eigen maps and original source-equivalent C loops, pinned CPU8, five interleaved periods.

1. Matrix products P*A then A^T*(P*A): C 97–107us/30 stages; Eigen 55–63us/30 stages, around 1.70–1.78x speedup, max relative numeric delta <=0.0002. P is representative SPD, not recorded P.
2. RHS A^T*p and A*x+B*u+d: C 8.5–9.5us/30 stages; Eigen 5.9–6.5us/30 stages, around 1.40–1.46x speedup; max absolute numeric delta <=0.00002.
3. Actual optional Riccati C++ backend: source f1tenth_mpc/src/riccati_eigen_kernels.cpp, C ABI flat pointer to stack-only fixed Eigen 23x23/2x23, computes Pnew=Q + A^T*P*A + G^T*K, mirrors upper triangle for exact symmetric storage. Added CMake flag SDU_APEX_MPC_EIGEN_RICCATI_QUADRATIC (default OFF), so this does not change production.
4. Eight difficult real QPs, 150 ADMM iterations each: C p50 solver us = 3259.79,3376.24,3729.26,3293.91,3710.05,3490.13,4093.91,3751.02. Eigen us = 3039.84,3140.32,3399.59,3007.09,3566.82,3157.64,3735.09,3383.43. Around 4–10% faster full solver on these fixtures, about 0.14–0.37ms per QP. Status/iterations unchanged, residuals practically identical; numerical solution deltas ~0.00027–0.00108 in mixed-scale state coordinates.
5. All 28 saved full-state RTI replays, including both SQP passes and full exact 750ms source-rollout validation, kept identical status/feasibility/SQP decisions: 0 critical mismatches. Max first steering delta 8.4192e-6 rad; first throttle delta 1.9058e-5. Data: live_runs/mpc_collision_margin_experiment_20261030/qp_eigen_fullreplay_parity_20261031.json.
6. Experimental ON standalone build: 20/20 CTest targets passed. Production OFF ROS build: 209 tests, zero failures. No Eigen objects or dynamic allocation cross C ABI; no production parameter change, no live Unity claim.

## Is a complete C->C++ rewrite advantageous?

NO blanket rewrite. C and C++ can produce equally good machine instructions; vectorization, data layout, aliasing, workspace ownership and elimination of redundant nonlinear steps are what matter. std::vector is contiguous but may allocate and cannot improve a fixed 23x23 stack buffer just by changing language. Use C++17 RAII for map/state ownership and immutable snapshots, preallocated std::array / fixed-size Eigen matrices for measured numerical kernels, Eigen::Map with no copy to bridge C memory, and explicit cached-value invalidation. Avoid per-cycle new/delete, dynamic matrix allocation, temporary state copies, virtual dispatch, hidden thread coordination. Keep stable C ABI for source plant, ADMM/QP and replay tools; test optional backends against identical frozen QPs.

HPIPM + BLASFEO can be evaluated as a separate OCP-QP backend because they target embedded small-matrix optimization, but that replaces the QP algorithm (ADMM->IPM); it requires full inequalities/boxes mapping and nonlinear source-validity A/B, not simply C++ conversion. Not installed or benchmarked here.

## Larger remaining convergence and physical safety issue

Current map EDT body geometry is highly optimized but remains principally a rejection checker after the SQP solves Frenet corridor constraints. True world-wall constraints must enter local QP linearization and trust region via conservative differentiable body clearance, including progress, heading and body corners. The eight present halfplanes are already used; carefully expand constraint representation instead of dropping safety constraints. A valid current 750ms horizon can still end with no safe next horizon. Provide a model-validated viable terminal condition or continuation in primary MPC. Emergency braking-aware steering seed 0.55*current_speed lookahead is heuristic; improve it to causally integrate station from the exact model with steering/throttle and queued commands. Maintain two SQP passes: previous saved states selected R2 in 7 of 10 valid solutions.

## Prioritized work

1. Expand actual RHS and control-Hessian Eigen/SIMD A/B, measure full QP and SQP (not only microloops), compare residuals and 30-stage nonlinear feasibility and live 10-lap repeatability.
2. Audit duplicates in source-model Jacobian, stage state initialization, reference sampling, map clearance, QP scratch arrays; cache immutable per-cycle values only when valid under all perturbed state/control candidates.
3. Integrate conservative map footprint as *solver constraints*, preserve independent exact nonlinear validation, then add terminal recursive viability. Avoid substituting repeated emergency actions for earlier feasible SQP steering.
4. Keep six Jacobian workers on this i5-13400F (mean 3.45ms vs eight workers 3.72ms on identical 18-state replay), and retain full R1/R2. No unproven 3ms worst-case reduction or real-time guarantee.
5. Repeated Unity five-run practice under 49.5s and full ten-scored-lap IROS tests before any experimental promotion. Active optimizer campaign and existing production overlays untouched.

## Reproduce

Default: colcon build --packages-select f1tenth_mpc --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DSDU_APEX_MPC_EIGEN_RICCATI_QUADRATIC=OFF
Experimental standalone: cmake -S f1tenth_mpc -B /tmp/sdu_mpc_eigen_quadratic_20261031 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DSDU_APEX_MPC_EIGEN_RICCATI_QUADRATIC=ON
Then: cmake --build /tmp/sdu_mpc_eigen_quadratic_20261031 -j6 && ctest --test-dir /tmp/sdu_mpc_eigen_quadratic_20261031 --output-on-failure -j6
Bench raw logs: live_runs/mpc_collision_margin_experiment_20261030/eigen_vs_c_matrixbench_20261031.txt and eigen_vs_c_rhsbench_20261031.txt.
