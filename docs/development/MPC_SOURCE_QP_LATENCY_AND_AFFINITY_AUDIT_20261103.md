# Source-exact MPC runtime optimization: structured QP, sensitivities and CPU scheduling

Date: 2026-11-03. Status: implemented and independently source-tested; still experimental and OFF in production ROS defaults. **Not yet qualified for 25 ms deadline in every cycle, Unity agreement, or racing.**

## Implemented and retained

1. **QP memory / exact algebra**: reuse dense Eigen normal matrix, weighted inequality Jacobian, residual, Newton direction and LDLT decomposition buffers throughout the primal-dual IPM. When 30/30 controls are optimized, assemble the hard wall/rate/box constraints only once; stop copying a second full 300x60 matrix. With 20/30 mode, exact elimination remains. Keep original strict 1e-7 primal/relative-dual/duality-gap certificate and independent source-feasibility test. 40 IPM maximum (previous audit: 60/60 frozen-state parity versus 90; 25 fails one case). After these cleanup changes, the **same 60 recorded normal QPs retained identical acceptance, objective, physical violation and QP statuses**. Nominal standalone solver median ~16–17 ms on a variable-load workstation; these are neither guaranteed deadline times nor one-to-one CPU measurements.
2. **Sensitivity source and workspace**: retain full 26-coordinate source plant with 20 dual lanes, exact wheel/IMU/actuator history and path reference coupling. For a command at stage k, no earlier trajectory stage can depend on it. Tangent scoring now updates only stages k+1...30; initial batch lane zero and actuator-rate Jacobian for stages <k are explicitly initialized to zero. Stop re-zeroing the full ~200KB trace each batch. Source derivative, perturbation nominal, and trace scratch buffers are thread-local and reused across sequential MPC solves. Optional full finite-difference prefix/suffix/terminal nominals are lazily allocated only when used. Skip zeroing arrays only when *every element is assigned*; inactive dual control seeds remain explicitly zero-initialized, after a regression caught an unsafe draft omission. Six independent corner fixtures show **bitwise-identical source cost gradients, full physical wall and actuator-rate Jacobians, and QP objective** before/after this optimization. Measured tangent scoring median ~2.01 -> 1.73 ms on six archived hard-corner cases; source-plant differentiation still dominates.
3. **SOC fast path**: when the current full 30-stage source trajectory is already strictly source-feasible AND exactly one SQP iteration is requested, skip the optional second-order-correction QP. It is NOT required to publish an already strictly certified plan. SOC still runs on source-infeasible starts and multi-iteration SQP. All 60 normal source-frozen cases retained exact acceptance, objective, wall and QP statuses, while this avoided **21 QP solves and 21 exact source evaluations**. The independent source validator is unchanged, and any truly infeasible control remains nonpublishable. Added focused GTest of normal feasible one-pass SOC omission.
4. **ROS control scheduling**: Linux-only optional integer parameter `control_thread_cpu_affinity` (default -1, no pinning). When enabled, validates against process allowed CPU affinity and pins only the existing persistent SingleThreadedExecutor control thread via `pthread_setaffinity_np` before its callback spin. Other ROS/Unity/sensor callbacks are NOT pinned or moved. On this specific i5-13400F, CPU2 is a P-core and CPU15 an E-core; another machine's CPU layout may differ.

## Profiling and measured results

On the authorized workstation (i5-13400F) with 120 frozen-case runs per CPU, pinned via taskset and identical source solver: CPU2 P-core median 15.52 ms, P95 24.22 ms, max 26.87 ms, median source-dual 7.12 ms. CPU15 E-core median 29.35 ms, P95 46.39 ms, max 50.26 ms, source-dual 13.96 ms. Speed ratio approximately 1.9x, **no physics/weight change**. These are saved-state solver-only timing, not ROS publish end-to-end. CPU frequency, cache warmth and other desktop workloads still affect variance.

Final consolidated normal one-iteration full-30 forward-source SQP: 60/60 recorded frame QP status, acceptance, racing objective and physical wall-violation parity. Independent *causal* 60-cycle source-model run with the actual native C++ controller entry point, exact source-model step advancement and persistent nominal/actuator memory: **60/60 strict 30-stage feasible**, final source racing objective 21.09581327 (same as earlier reference), maximum exact next-state error ~4.98e-11.

Three post-SOC repeats of 60 independently saved states each, restricted to CPU2 with a **strict 25 ms solver deadline**: 58/60, 59/60, 58/60 certified outputs within deadline, median ~15.2 ms; P95 ~21.7-22.9 ms; maximum solver runtime including deadline rejection ~25.6-26.0 ms. These ~97% results are NOT enough to claim hard realtime and do not include ROS publication or Unity state synchronization. Deadline-expired candidates remain rejected.

Full CMake build and 22/22 CTest suites pass; 7 targeted source SQP tests pass including a new exact-source normal single-iteration SOC skip test. `git diff --check` clean.

## Discarded experiments (not retained as active legacy branches)

- Structured `A^T D A` shortcut recognizing 120 ±unit box rows: faster on many frozen cases but a numerically sensitive QP lost a useful accepted cost reduction (59/60 only); increasing IPM cap to 120 did not cure it. Not promoted; experimental solver code and flags removed.
- Stage-shifted, epoch-guarded IPM warm starts: full primal+dual, primal-only and dual-only all tested over 60 genuinely evolving exact-source MPC epochs with strict validation. All were source-feasible but had worse tail latency or less favorable objective than cold initialization (e.g. full warm max 157 ms vs cold max ~41 ms in one comparison). All implementation fields, CLI flags and added RTI-memory ABI bytes removed.
- `-march=native` compilation: about 10% speedup in paired CPU-pinned tests, but 1 of 60 floating-point sensitive QP outputs changed. Remains build-time opt-in, not a production default.
- Strict-validator-first nominal reuse: six exact Jacobian/trajectory parity checks and 60-frame normal output parity passed, but on already infeasible saved corners it could cause >100 ms extra work and normal speed gain was inconsistent. Removed from active code.
- Additional array zeroing changes caused a derivative-regression failure in development, fixed before final validation: inactive control dual lanes MUST remain zero; all current tests and six exact corner derivative snapshots pass.
- No deletion of legacy Riccati/ADMM/R1/R2 production implementation: it is bypassed by the opt-in native source-SQP but remains the established production controller. Removing it before full replacement certification would break current usage, not reduce the active native SQP CPU.

## What remains

Still cannot meet 25 ms in 100% of normal P-core frozen frames, nor is CPU2 necessarily isolated from interrupts/system noise. For real-time eligibility: isolate/priority control CPU if the host permits, eliminate remaining source rollout duplication *without* losing swept-wall/contact checks or high-slip derivative fidelity, check race conditions with Unity actuator queues, and benchmark full ROS callback->publish with monotonic absolute deadline plus several complete independent races on both tracks. The main remaining normal-case cost is ~7 ms full source forward sensitivities on a P-core and a few ms exact nonlinear validation / trial updates; the QP median is ~1 ms on normal states, so another optimizer rewrite is less likely to help than those components. Keep normal source model + strict validator, avoid broad recovery/fallback additions.

Reproduction (from repository root):

```sh
cmake --build build/f1tenth_mpc -j6
(cd build/f1tenth_mpc && ctest -j6 --output-on-failure)
python3 tools/mpc_native_source_closed_loop.py \
  live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917233236533_case_2 \
  --cycles 60 --iterations 1 --forward-source \
  --out /tmp/mpc_source_recheck
# For live experimental ROS only, after validation: set
# native_source_reduced_sqp:=true
# native_source_sqp_full_source_forward:=true
# native_source_reduced_sqp_iterations:=1
# control_thread_cpu_affinity:=2   # ONLY if CPU2 is suitable and allowed
```

No checkout reset, broad code deletion or commit; pre-existing uncommitted changes preserved.
