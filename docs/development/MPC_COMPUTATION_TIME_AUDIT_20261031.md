# MPC computation-time audit — 2026-10-31

## Objectives and hard limits
- Achieve <20ms from callback entry to ACTUAL command publication, ideally max <=17ms to retain >=3ms headroom, without sacrificing 750ms source/PhysX physical feasibility, nonlinear corridor checks, solver correctness, or the validated fast practice overlay.
- A genuine guaranteed deadline cannot be asserted solely from a finite sample under non-RT Linux. Require a bounded deadline-aware work budget, safe independently validated fallback if a deadline is at risk, runtime scheduling/resource isolation, and independent repeated live validation.
- The configured high-speed recovery is NOT a constant-speed physical rollout: only the initial *candidate steering reference seed* used an inappropriate v_current × t future-progress prediction. After constructing a candidate, the full source-plant rollout propagates per-stage steering/throttle, actuator delays, speed, yaw/slip. The new emergency 0.55 scaling is a heuristic; a future improvement should generate seeds from dynamically propagated causal progress, not assume constant speed.

## Baselines
- Prior 7.075-second native source witness full-distance Unity success: one warmup + ten scored laps **83.938s**, zero collision, **one deadline miss**, maximum 21.865ms at publication; ADMM 15.5467ms, QP build 2.6879ms, nominal build 1.686ms. This outlier consumed 22 ADMM iterations with 3 Riccati factorizations.
- Practice verified 5/5 independent 10-scored-lap races under frozen prefix-seed overlay: 49.340,49.374,49.486,49.396,49.258s, no deadline misses/collisions. Do not promote incompatible tuning.

## Exact QP fixtures, microbenchmarks
- Captured **8 independent full QP binary fixtures** from the saved IROS hairpin state under `live_runs/mpc_collision_margin_experiment_20261030/qp_hairpin_20261031/qp_*.fqp` using production replay. Preserve fixture hashes and state configuration.
- Added optional `MPC_QP_BENCH_REPEATS` and benchmark-only `MPC_QP_BENCH_RHO_PERIOD` to `tools/mpc_qp_fixture_replay.cpp`, with per-run exact solver parity checks and p50/p95/max time.
- Current portable build -O3: 150-iteration saved QP medians ~3.1–4.7ms isolated; `-march=native -ffp-contract=off` is inconsistent or slower (median of medians ~3.91ms vs portable ~3.62ms). **Do not enable native compilation on this basis.**
- Increasing adaptive-rho update period 8,16,32 gave no reliable speed advantage and modified residuals. Disabling adaptive rho speeds kernels to ~2.9–4.3ms but significantly worsens primal residual, often from ~0.03–0.16 to ~0.05–0.31. **Do not disable rho adaptivity or weaken convergence.**
- Tested precomputed projection bound spans in `riccati_solver.c` with exact QP replay parity across 8/8 fixtures and 207 passing unit tests; improvement was not consistently measurable, so reverted only that micro-optimization via saved original source; preserve the benchmark results in `admm_cached_span_benchmark_20261031.json`.

## Thread CPU vs wall-clock telemetry
- Added `MpcRtiCycleResult_t.admm_thread_cpu_us` and instrumentation of first ADMM solve and optional restoration solve to `f1tenth_mpc/include/mpc_rti.h`, `f1tenth_mpc/src/mpc_rti.c`, `f1tenth_mpc/src/mpc_controller_node.cpp`. No solver arithmetic changed.
- One instrumented live IROS attempt (`live_runs/iros_7075_cpuwall_profile_20261031_01`) suffered a simulator collision in lap 2, **not** a clean race benchmark. Max callback 18.929ms, no deadline miss; 547 measured ADMM solves had mean wall-minus-thread-CPU gap only ~3.8 microseconds; max ~502.6 microseconds; worst ADMM wall 9.151ms vs thread CPU 9.129ms with 262 iterations. This shows most ADMM time in this run was actual CPU, rather than OS descheduling. The historical 15.55ms/22-iteration outlier cannot be attributed conclusively without its own CPU telemetry. OS contention/frequency effects remain plausible but unproven.

## Selective second SQP pass experiment
- Opt-in flag `skip_sqp_refinement_after_first_valid_horizon` (default **false**) skips a second expensive SQP on an already accepted damped full-horizon candidate, while retaining second pass on first-pass nonlinear rollout rejection.
- A single full-speed IROS attempt (`iros_7075_fastrefine_20261031_01`) completed 2 laps before a genuine rejected nonlinear horizon, no collision or missed deadlines; callback p50 **7.237ms**, p95 **15.226ms**, p99 **17.785ms**, max **18.714ms**. Compared with the historical successful 10-lap run p50 ~9.7, p95 ~17.0, max 21.865ms; the observed max difference of ~3.15ms is NOT a controlled same-trajectory or guaranteed improvement.
- Practice regression (`practice_guarded_fastrefine_20261031_01`) completed ten scored laps with no collision or missed deadlines and max callback 18.285ms, but **10-lap 50.139s**, recovery actions 23, an anomalously slow ~5.457s lap; **FAIL practice <49.5s gate.** Do not promote selective skip or use it to replace five-run practice profile.
- CRITICAL INTERPRETATION CORRECTION: `refinement_prefix_selected_pass` in saved solver diagnostics counts equal short-prefix comparison only in cases where neither full 750ms pass is certified, NOT general full-horizon R2 improvement selection. Historical "54 useful IROS vs zero useful practice refinements" inferred from that field is NOT a valid count. Second pass can choose an improved complete horizon without this field. Review `mpc_rti.c` logic `mpc_rti_refinement_improves_action(first_comparison_valid,...)` and add direct full-horizon R2-selection instrumentation before designing selective refinement.
- The node restores `memory_after_first` when R2 isn't chosen; avoid claiming R2 always poisons the next cycle when it isn't selected. The live practice regression could differ because R2 was actually improving the selected full horizon, not just its warm state.

## Source changes and test gate
- Changes made in this task:
  - Optional benchmark-only tool modes in `tools/mpc_qp_fixture_replay.cpp`.
  - ADMM thread CPU telemetry in `mpc_rti.h`, `mpc_rti.c`, controller diagnostics.
  - Optional selective R2 skip flag, disabled by default, in those same files.
  - No production YAML changed by this task; no proven numerical/physics shortcuts or tolerance relaxation.
  - `riccati_solver.c` restored to pre-task bytes after speed precomputation provided no reliable improvement.
- Complete C++ suite at finish: **207 tests, zero failures**.

## Next technical optimization priorities
1. Add explicit `refinement_full_horizon_selected` and `first_pass_nonlinear_valid` diagnostics, with per-pass QP+ADMM+Jacobian timings, and benchmark R2 selection on identical saved states (not just prefix-selected-pass).
2. Optimize Riccati factorizations and RHS only after profiling with `SDU_APEX_MPC_RICCATI_PROFILE=ON` in an isolated test build, including adaptive-rho factorization count. Audit numeric parity on all frozen QPs plus complete physical rollout.
3. Explore a deterministic bounded stop condition/upper work reserve **without publishing an unchecked plan**; tune control budget to 17ms only as a separate live experiment. Do NOT confuse early termination with a 3ms CPU speedup. Reject any change that increases hard 750ms rejections or slowdown.
4. For hard worst-case guarantees, provision dedicated real-time scheduling/CPU isolation and assess latency/jitter under bounded worst-case load; absolute wall-clock hard bounds cannot be promised in ordinary Linux userspace. Keep an independently validated command fallback and fail closed on missing feasibility.
5. For promotion require no missed 20ms publication deadlines, five separate ten-lap practice runs under 49.5s **and** multiple complete full-speed IROS ten-lap runs, no collision/rejection, with WORST-CASE measured callback <=17ms under representative load. Retain the protected practice five-run winner until then.
