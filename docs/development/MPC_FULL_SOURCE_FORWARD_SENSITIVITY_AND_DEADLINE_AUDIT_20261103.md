# Full-source forward sensitivities and hard deadline qualification — 2026-11-03

## Scope and release status

- **Implemented and tested:** all 26 physical Unity-derived source coordinates in the 20-lane source-plant tangent (including four wheel rates, physical steering, body height/roll/pitch, four suspension jounces and eight sticky-tire timers), integrated body forward/lateral/yaw displacement, exact actuator queue shifts, exact stored raceline projection Jacobian (including progress), nonlinear source objective and physical left/right corridor sensitivity. The primal vehicle trajectory always uses the original source integrator. The QP still requires independent strict nonlinear source validation.
- **Not released:** `native_source_reduced_sqp` remains disabled by default, as does `native_source_sqp_full_source_forward`. This local test does **not** establish 25 ms deterministic realtime operation, Unity/PhysX agreement, or actual race completion.
- All modifications preserve uncommitted original user changes; no branch reset, merge, or commit performed.

## Source-sensitivity implementation

Files:
- `f1tenth_mpc/include/vehicle_model.h`, `f1tenth_mpc/src/vehicle_model.cpp`: `vehicle_model_full_source_interval_sensitivities` maps all 26 C source-state tangent coordinates and direct steering/throttle commands into the existing real source-plant dual20 model and exposes physical next state, body displacements and longitudinal slip derivatives. The public source plant, its exact nonlinear step and existing 23-state QP code are not substituted.
- `f1tenth_mpc/include/mpc_source_sensitivity.hpp`, `src/mpc_source_sensitivity.cpp`: batch 20 normalized command sensitivities; propagate all 26-source coordinates, exact 3x6 measured projection Jacobian including progress, exact shift/append of the four actuator history slots. Score the resulting physical-state tangent at +/- scaled finite increments using the **same** full source racing objective and left/right footprint checker as the normal SQP. Derive all 60 physical corridor constraint rows, 120 rate rows, cost gradient and convexified diagonal curvature. For full 30-command horizons there are 60 dual-plant 25ms sensitivity intervals per SQP pass; for 20-command/30-physical there are 50. Earlier unperturbed stages are mathematically independent of later commands and are skipped.
- `f1tenth_mpc/src/mpc_source_reduced_sqp.cpp`: source forward derivatives optionally replace 120 (full 30 controls) or 80 (20 controls) full source finite difference perturbations per pass, while preserving all QP constraints, the exact nonlinear acceptance/filter, SOC, deadline and strict independent final physical certificate.
- `f1tenth_mpc/include/mpc_rti.h`, `src/mpc_rti.cpp`, `src/mpc_controller_node_impl.hpp`: opt-in `native_source_sqp_full_source_forward` only. The existing source and legacy QP selection defaults are unchanged. Progress and rate/timing metrics exposed.
- `tools/mpc_native_source_sqp_replay.cpp`: `--full-source-forward`, `--opt20` and derivative audit using `MPC_SOURCE_SQP_DUMP_JACOBIAN=1`. `tools/mpc_native_source_sqp_controller_epoch.cpp` and `tools/mpc_native_source_closed_loop.py` accept `--full-source-forward` and an offline `--deadline-us=25000` condition for saved-state deadline audit.
- `f1tenth_mpc/src/mpc_rti.cpp`: correct only the optional 20-command fixed warm-start tail. A newly fabricated terminal witness changed steering +0.0852599 -> -0.0143138 rad, creating a physically invalid rate step that the fixed 20-command QP could not alter. Hold the real last command for that unsolved tail stage instead. Every legacy and full-30 control path keeps its previous seed policy.

## Numerical correctness caveat

These are **branch-local derivatives**. Sticky/contact/force saturation and source-state floating-point projections are not differentiable across every finite perturbation. This may make central finite differences and the primal-local source dual derivative disagree on large/high-slip finite steps. The 23-state source adjoint alone does not replace this 26-source-state derivative.

- A new C++ unit regression `MpcSourceReducedSqp.FullSourcePhysicalForwardDerivativesMatchFiniteDifference` compares the exact 40-command-variable source FD oracle against full forward cost, 60 wall and 120 actuator-rate Jacobians on a fully source-initialized synthetic case; passes, with independent 30-stage source certification. `MpcSourceReducedSqp.ForwardSourceDeadlineDoesNotPublish` verifies expired-deadline rejection and no candidate publication.
- Six recorded high-slip corner cases: maximum absolute actuator rate J mismatch ~5e-6, max physical wall-J mismatch 0.04154 m per normalized command, worst scaled cost-gradient discrepancy 1.697 in extremely nonlinear corner/branch regions (in one reference corner 0.021). Derivative equality is **NOT** qualified in these regions. The exact full-horizon source rollout still rejects unsafe proposed commands; however extra SQP updates / bounded numerical recovery are needed to prevent a forced unrecoverable state.
- Six-case actual nonlinear SQP comparison preserved source-feasible status for cases which fully recovered, but exact objective and remaining wall violation differ in some cases; never describe the forward approximation as a bit-exact equivalent to the FD source oracle.
- Saved derivative comparison: `/tmp/mpc_full_source_forward_sixcase_derivative_parity.json`; two-pass actual nonlinear comparison: `/tmp/mpc_forward_source_sixcorner_comparison.json`.

## Receding-horizon source-plant results (Core i5-13400F, Release -O3)

With the same recorded competition pre-corner source context and retained real native controller memory, apply first command to full plant every 25ms and recompute all 30 stages:

| Native forward-source configuration | Fully strict source validated | Median cycle | Max cycle |
|---|---:|---:|---:|
| 20 optimized commands, 1 SQP iteration (20-epoch run) | 20/20 | 19.67 ms | 29.73 ms |
| 20 optimized commands, 2 SQP iterations (20-epoch run) | 20/20 | 32.66 ms | 49.19 ms |
| 30 optimized commands, 1 SQP iteration (60-epoch run) | **60/60** | **17.59 ms** | **34.92 ms** |
| 30 optimized commands, adaptive 1–2 SQP iterations (60-epoch run) | **60/60** | 20.99 ms | 60.96 ms |

All accepted controls and complete source trajectories passed **independent** nonlinear certification, and exact stage-1 physical next-state plus queue comparison differed by at most ~5e-11. These are SOURCE closed-loop trials; not Unity closed loop, not race laps. The highest runtimes violate live 25 ms and the true ROS publication reserve. A 20-variable adaptive source trial terminated when a later frozen terminal step could not be repaired; keep the 30-command solver as the stronger new candidate.

Profiling an unbounded reference full-forward 20-control one-pass run: ~19.95ms total, ~12.17ms source dual-plant differentiation, ~1.94ms cheap tangent objective/footprint scoring, ~1.19ms QP, ~4.38ms exact source model evaluation/validation. These vary with contact regimes and workstation load.

## Absolute deadline test: UNQUALIFIED for realtime

Re-run the native controller independently on each of 60 saved normal-source states, supplying the stored matching previous-step optimizer memory, and injecting a true monotonic absolute deadline immediately before calling the controller:

- **25,000 µs:** **34/60 accepted** with strict 30-stage source certificate. **26/60 correctly rejected due to elapsed deadline.** Median observed controller runtime 20.87ms, worst 40.64ms. This is NOT a runnable closed loop after misses, simply an honest per-frame budget audit.
- **20,000 µs:** 10/60 accepted; 50/60 deadline expired.
- Median split in 25ms budget: ~10.62ms forward derivatives; ~3.86ms QP; ~3.86ms exact source rollout/validation. Longest QP solve consumed ~29.76ms despite a deadline (the IPM checks between iterations, but one expensive linear-algebra operation can still overrun). The full algorithm correctly checks the deadline after all work and never certifies an overdue candidate for publication.
- Measurements: `/tmp/mpc_source_forward_deadline_60frame_audit.json`.
- Future release requires measured worst-case < publication budget (prefer <= 15-20ms MPC solve depending on ROS margin), minimal timing spikes and real Unity/source agreement on both maps. It is **not acceptable** to simply increase the 25ms callback deadline or publish unvalidated QP candidates.

## QP numerical exploration

- The existing dense Mehrotra IPM already factors the same KKT matrix once per predictor/corrector pair and retains strict primal/dual residual plus independent nonlinear source certificates.
- Tested `-march=native -ffp-contract=off` in a *separate* build directory (`build/f1tenth_mpc_native`), with no stable speedup compared with portable Release -O3, so deployment defaults remain unchanged.
- Tested a smaller-primal/slack/dual balanced interior-point initial guess on 15 normal frozen frames. It worsened median QP compute time (from ~29ms to ~42ms in that noisy measurement), sometimes stalled at 99 iterations and changed approximate QP iterates. **Entire experimental initializer removed**, not left as a second solver path.
- No reliable persistent primal/dual warm start is implemented yet. That is the next potential source of significant QP tail-latency improvements, but needs proof on real candidate epochs rather than hot-starting a stale or incompatible active set.

## Required next acceptance gates

1. Preserve full 30 optimized controls and 30-stage collision prediction, use true 26-source-state sensitivities. Increase derivative parity coverage for high-slip, constraint-boundary and source branch transitions. Where they cannot be certified, check exact nonlinear merit and never publish a bad candidate. Do not quietly replace it with an incompatible 23-state linearization.
2. Improve dense QP tail latency with a demonstrably well-conditioned warm-start or alternate factorization, maintaining all primal/dual and strict nonlinear checks, then repeat the 60-frame *absolute-deadline* audit.
3. Qualify many independent consecutive source-model closed-loop episodes with varying speeds and corner conditions, including startup, queue restart and terminal reachability, rather than only frozen historical states.
4. Resolve true source/Unity physical model discrepancies and run Unity closed loop before attempting full-speed qualification/races on practice/competition maps.
5. Only after 25ms budget and source/Unity agreements pass, enable opt-in live mode; do not delete the proven full-reference solver or simulator emergency stop before release acceptance.

## Reproduction

C++ build and regression: `cmake --build build/f1tenth_mpc -j8`; `(cd build/f1tenth_mpc && ctest -j8 --output-on-failure)`

Full 30-stage forward source QP: `build/f1tenth_mpc/mpc_native_source_sqp_replay live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917333601071_case_6 - 2 --full-source-forward`

60-step source closed-loop: `python3 tools/mpc_native_source_closed_loop.py live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/epoch_1793692917233236533_case_2 --cycles 60 --iterations 1 --forward-source --out /tmp/mpc_fullsource_forward_reproduction`

All source-forward and native-source-SQP flags remain opt-in; **the production controller is not changed to use this experimental solver**.
