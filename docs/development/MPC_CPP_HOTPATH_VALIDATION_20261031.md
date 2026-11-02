# MPC C++ hot-path continuation — 31 October 2026

## Priority (updated by user)

Do **not** gate numerical rewrites, legacy-code deletion or controller hardening on a 49.5 s race time. Work in this order:

1. Maintain one C++20 source-plant and Riccati implementation. Remove an old path only after verified equivalence; do not carry dual legacy and new implementations.
2. Reduce repeated computation, benchmark real QPs and source-plant rollouts, and retain only changes with demonstrated non-regressing performance.
3. Reach repeatable complete closed-loop races under the **20 ms hard publication deadline** with headroom (preferred observed maximum 15–17 ms), no collisions, no faults, and no unchecked safety shortcuts. Use a five-run practice qualification when a candidate is ready, and independent IROS races.
4. **After** repeatable timing and robustness are established, independently adjust lateral/heading and other weights, improve convergence/fallback policy and recover lap time. Lap time remains an observation now, not an acceptance gate.

## Changes and outcomes

- **Rejected and reverted:** consolidated ADMM per-state bound pointers and rho-balance span calculation. 211 ROS tests and 16 frozen QPs (48 pair repetitions) remained numerically identical, but median per-fixture QP runtime increased **2306.1 → 2453.5 µs (+6.4%)** under a pinned CPU8 interleaved benchmark. Preserved the faster old C++ ADMM loop; no experimental duplicate implementation remains in source. Evidence: `live_runs/mpc_collision_margin_experiment_20261030/cpp_admm_bound_dedup_qp_ab_20261031.json`. This is one noisy benchmark; no reason to promote a measured regression.
- **Accepted:** `mpc_trajectory_advance_tracking_with_progress_sampled()` exposes the already computed final Newton projection path sample, while `mpc_trajectory_advance_tracking_with_progress()` remains a thin, compatible wrapper. `rollout_candidate_impl()` passes the exact returned geometry to reference/corridor construction instead of issuing an identical second `mpc_trajectory_sample()` for every reached future stage. Existing hard body/world-map checks, source-model integration, output commands, and nonlinear validation remain unchanged.
- **Equivalence:** 212 ROS tests passed, including `MpcReference.ProjectedTerminalSampleExactlyMatchesResampling` comparing the sampled final geometry, tracking states, and 3x6 Jacobians at multiple progress points. Archived-vs-new full nonlinear IROS saved-state replays: **28/28** unchanged critical decision statuses and unchanged first steering/throttle outputs at reported precision. Evidence: `live_runs/mpc_collision_margin_experiment_20261030/cpp_presampled_fullreplay_parity_20261031.json`.
- **Focused timing:** Standalone `mpc_reference.cpp` projection+sampling microbenchmark (500-point periodic curve, pinned CPU8, 12 interleaved batches, 40,000 calls per batch) measured median **0.2984 → 0.2666 µs/call**, about **11% faster for this narrow operation**, with the same checksum and zero projection failures. A full 30-stage candidate saves roughly 1 µs at that microbenchmark rate; do **not** present this as a millisecond-level callback improvement. Logs and benchmark source under `/tmp/sdu_mpc_terminal_sample_bench*`.
- **Unity practice regression:** `practice_cpp_projection_sample_reuse_20261031_01` completed warmup plus **ten scored laps**, no collisions, no publication misses, no accepted recovery cycles, max callback **18.108 ms**. Total 49.352 s is diagnostic only, **not** a clean-up gate. Over 2,228 callback rows: callback p50 8.606 ms, p95 16.658 ms, p99 17.278 ms; exact rollout p99 9.072 ms (max 10.192 ms), ADMM p99 5.108 ms (max 13.513 ms). One successful practice run does not certify repeatability or sub-17 ms maximum.

## Remaining numerical work

The hot-path timing concentration is unchanged: exact physical nonlinear rollouts/backtracks (up to about 10 ms per callback) and exceptional cold/recovery ADMM episodes (up to about 13 ms). Optimize those measured sections without suppressing physical checks, shortening the 750 ms prediction horizon, turning off R2, or changing objective weights. For each candidate, keep a frozen previous binary, check the full 212-test suite, replay difficult source-state/command QPs for exact decision parity, perform pinned interleaved speed comparisons, and keep only validated improvements. The `f1tenth_mpc/src` production tree has **zero `.c` files**; C++ implementation and public ABI are not duplicated.

Work remains uncommitted, alongside independent existing IPOPT changes. No optimizer source, production race YAML, or fallback-control policy was edited here.

## Continued structural cleanup

The 31 October cleanup additionally split the ROS diagnostic worker and the Riccati factorization/ADMM/scaling responsibilities into dedicated C++ units, removed redundant RTI timing copies, and eliminated 773 unchanged-penalty comparisons per ADMM iteration via exact rho-change invalidation. Numerical and benchmark records are in `docs/development/MPC_CPP_STRUCTURAL_CLEANUP_AND_RHO_INVALIDATION_20261031.md`.
