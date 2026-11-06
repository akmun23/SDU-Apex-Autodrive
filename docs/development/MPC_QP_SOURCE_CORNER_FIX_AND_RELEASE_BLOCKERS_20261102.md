# Source-corner QP mathematical repair and strict Unity validation — 2026-11-02

## Status: CORNER CONTROLLER NOT YET QUALIFIED

This document records **actual changed files, independently reproducible numerical evidence, and two strict failed Unity trials**, not a proposal or a successful controller handoff. There is no justified claim that the complex high-slip finite-step QP mismatch or recursive feasibility is solved. Previously qualified practice configurations are unchanged.

## Verified defect 1: missing progress gradient in source Phase-I

The source-plant infeasible-nominal repair differentiated the exact physical footprint violation only with respect to lateral position and heading. The same source rollout sampled left/right corridor bounds at the predicted **actual path progress** at each stage, but its adjoint's progress coordinate was always zero. This is a real first-order derivative omission, especially harmful near narrowing corners.

Fix: in `f1tenth_mpc/src/mpc_source_nonlinear_optimizer.cpp`, add the candidate-progress partial derivative to the same Phase-I footprint merit, evaluated with the original `mpc_rti_reference_at_progress` and `mpc_rti_compute_corridor_slack` routines at s±0.002m. Both tire model and exact acceptance remain unchanged. Shared tested helper `mpc_rti_source_corridor_progress_derivative` is declared in `f1tenth_mpc/src/mpc_rti_internal.hpp`.

Added deterministic regression `MpcSqpSourcePhaseOne.ProgressGradientMatchesExactChangingWidth`: compares against independent finite difference on a contracting wall, verifies nonzero derivative, passes. The normal controller still has its original strict rollout and default-disabled Phase-I initializer.

Direct **same-snapshot A/B**, corpus `live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures/`, exact `--source-phase1` replay case 4:

- Before: first invalid stage 13; original footprint violation 1.30031e-6 m²; Phase-I attempted one iteration, accepted zero, no strictly valid full-horizon plan.
- After: identical saved physical state and original controls; one Phase-I iteration accepted; final zero corridor violation, strictly valid full horizon with 1.3355 mm physical minimum clearance.
- All 19 available R1/R2 saved nominals: **2 full-horizon repairs after one Phase-I iteration**; raising the offline cap to 4 iterations increased to **6 / 19**, but the newly repaired hard states commonly required 14–32 ms just for Phase-I, outside the usable real-time budget. Therefore it would be incorrect to enable a four-iteration repair in 20ms live control.

The source gradient is locally corrected, but cannot rescue states whose queued first step is already outside the hard corridor. Fitting local derivatives is not the same as matching nonlinear finite steps at high slip.

## Verified defect 2: false QP bound-violation diagnostics

`tools/mpc_qp_fixture_replay.cpp` incorrectly treated `-BIG_BOUND / BIG_BOUND` as real ±50 physical state bounds, although both native Riccati and the independent OSQP formulation treat them as unconstrained sentinels. In a captured solved R2 QP at case 0, this falsely reported **52.0511** state-bound violation due to wheel angular speeds >50rad/s.

Fix: only compute lower/upper physical violations when the corresponding bound lies strictly within the sentinel. The identical saved R2 solution now reports **0** primal and projected state-bound violation. All QP controls and solver iterates are unchanged by this diagnostics correction.

## Independent mathematical QP/physics audit

Installed OSQP 1.0.4 to `/tmp/sdu-mpc-osqp-v104` only; no global Python or ROS modifications. Executed `tools/audit_mpc_osqp_source_fidelity.py` on captured near-wall states; results `/tmp/mpc_osqp_source_current_20261102.jsonl`. Multiple solved, small-residual OSQP QPs still violated the **original source** full-horizon physical footprint constraints when their complete control sequence was integrated. Example: R1 on case 2, 625 OSQP iterations then fails exact source at stage 16, while the R2 QP in that case has an independently solved source-feasible control sequence. This distinguishes mathematically solved local affine problems from globally valid nonlinear trajectories. It is not evidence of a broken core ADMM factorization.

The 18-state high-slip C++ source forward sensitivities were independently finite differenced 2,772 times at each perturbation scale. At 100% perturbation, 51 comparisons had >10% scaled-relative error; at 10% or 1% scale, zero. At 1% scale max relative error 0.277%. Thus the local derivatives are plausible while larger source dynamics steps are strongly nonlinear, particularly rear-left suspension jounce / pitch with high steering.

A separate 178-epoch true Unity-versus-exact-source *one-step* replay, `/tmp/mpc_20261102_high_slip_source.csv`, found up to **0.06762 rad/s yaw-rate error** around 2.55m/s at significant steering, and up to 0.152m/s longitudinal speed error at >9m/s. This does not justify attributing error to any single PhysX tire coefficient or changing observer state without further causal tests.

## Rejected coupling experiment, NOT kept

An experimental full (e_y, e_psi, path-progress) PSD Gauss–Newton cross-Hessian was added to opt-in soft-reserve QP and passed unit tests. With otherwise the same 20k soft-corridor source-informed experimental overlay, a strict Unity competition trial failed earlier (near 59.607m), compared to the previous no-cross version's failure near 80.357m. Independent simulator starts are not matched-state causal counterfactuals; the data do **not** justify promotion. The cross-Hessian experiment was **reverted** from the final working tree. Existing progress-only diagonal soft-reserve mathematics is preserved.

## Strict Unity tests (neither completed a scored lap)

- `live_runs/comp_qp_phase1_progress_20261102_01/`: Phase-I progress derivative correction and opt-in experimental 20k reserve; **0 laps, stage-0 strict corridor rejection** at progress ~80.357m, next predicted footprint shortfall ~1.325mm. No collision; no final-cycle deadline miss (~11.60ms).
- `live_runs/comp_qp_fullgn_progress_20261102_01/`: same plus unqualified 3-variable GN cross-Hessian; **0 laps, stage-0 strict corridor rejection** at progress 59.607m, shortfall ~0.892mm. Reverted GN experiment.

These are both real same-track strict Unity executions with identical general overlay type but distinct fresh starts. A few millimetres of model/epoch/corridor error is enough to invalidate a nominal when the full-horizon candidate or retained plan approaches the physical wall envelope. A solved QP cannot retroactively change an already-applied or queued actuator command.

## Remaining P0 mathematical/physical blockers

1. Derive a **feasible/robust successor set** from the exact 33-state source plant, real command FIFO and measured Unity/source discrepancy **before** the state arrives at the point of no return. Current 30-stage source and terminal trial methods demonstrably do not guarantee recursive feasibility.
2. Use true nonlinear source step acceptance or constrained repeated source-based SQP iterations in high-slip corners, not an assumption that an affine Riccati QP can be physically exact at large steering/throttle perturbations. Validate per-channel full-horizon QP prediction error and objective reduction, not only derivative limits.
3. Causally match 25ms Unity state/command epochs and wheel contact/suspension dynamics at the failed corner, including map-based full-footprint bounds and reference progress. The one-step model is accurate enough to support diagnostics but **not certified to reproduce Unity within the sub-centimetre corner clearance currently being used**.
4. Repair runtime budget (20ms end-to-end strict) while allowing physically valid source SQP iterates to be found. Four Phase-I iterations are not an acceptable simple production switch.
5. Retest strict competition full race and five independent 10-scored-lap qualifications before any promotion; preserve prior practice qualified binaries/overlays. Keep every existing strict nonlinear validator until those gates pass.

No speed, steering, throttle, or QP penalty was silently tuned. The audit changes should not be represented as a fully repaired controller.

## Audit warning: Python first-step feasibility is not production source-frame geometry

A new diagnostic inconsistency was reproduced from the second strict Unity failure's complete saved state (`/tmp/comp_qp_fullgn_final_rejection.json/failure.json`). The C++ live controller's authoritative exact-source validation reported first-stage hard footprint **-0.892mm** at progress 59.6073m. The older independent `tools/audit_mpc_trajectory_feasibility.py::_source_rollout` Python calculation, for the same recorded control and sourced internal wheel/suspension state, reported **+18.823mm** corridor clearance. A 55-steering/throttle one-step grid in that Python implementation had 55/55 nominally feasible steps. These are **not** 55 physically verified production-MPC plans: the Python path uses an approximate curvature-based Frenet update and its own footprint/reference handling, whereas the production C++ MPC uses `mpc_trajectory_advance_tracking_with_progress_sampled` and the hard polygon checker. The difference is ~19.7mm in this case. Do **not** use that independent Python checker to certify sub-centimetre feasibility, declare a live stage-0 violation avoidable, or validate a modified QP. Use the actual C++ source validator from matching recorded physical/command state. The standalone Python diagnostic was not changed because a faithful rewrite of its full geometry path requires a separate parity harness.
