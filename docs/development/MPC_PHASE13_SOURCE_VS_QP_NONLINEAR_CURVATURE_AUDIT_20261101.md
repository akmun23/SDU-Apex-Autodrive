# MPC Phase 1/3 — source-model parity and QP curvature overhaul audit

Date: 2026-11-01. Status: diagnostic investigation and engineering design; NO qualified production solver change.

## Evidence sources and reproducibility

Practice race bag: live_runs/phase13_practice_increment_gain025_tenlap_20261101_01/run
Practice source trajectory: f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv

Two distinct comparisons:
1. Unity versus MPC's exact nonlinear source model: python3 tools/evaluate_mpc_bag.py BAG --trajectory TRAJECTORY --source-plant-rollout. Original console output on development machine at /tmp/mpc_phase13_practice_simulator_parity_20261101.log.
2. Current frozen LTV/QP versus its own nonlinear source model: tools/mpc_bag_solver_replay.cpp diagnostic and saved-state source in live_runs/mpc_phase13_solver_consistency_audit_20261101/. Exact case logs: gradient_case_0.log, gradient_case_4.log, gradient_case_7.log. Structured summary: curvature_summary.csv.

The offline replay reconstructs initial plant state but NOT the full original live nominal/warm-start and control epoch history. Its results are mechanism evidence, not bit-identical live-cycle reproducers. Its wide-corridor diagnostic is NEVER used to certify safe publication; hard corridor status was separately computed.

## 1. Unity physics mismatch is not the same as QP approximation mismatch

- Correct source/Unity 25ms command phase is recorded queue slot 2: n=2220, one-step u RMSE 0.01359m/s, lateral velocity RMSE 0.00197m/s, yaw-rate RMSE 0.01185rad/s, steering-angle RMSE 0.00114rad.
- Using previous queue slot 1 incorrectly inflates yaw-rate RMSE to 0.17407rad/s. The solver must have the command-application epoch and queue state exactly aligned.
- 2191 no-reanchoring 750ms windows: source-model endpoint XY RMSE 0.0133m and p95 0.0231m. Full 750ms path: 16/2191 windows exceed 50mm deviation (max 0.0935m), disproportionately high steering; unacceptable to declare universal exact Unity parity.
- Real Unity clearance and published 50mm physical footprint margins remain authoritative.

## 2. Frozen LTV/QP optimization model loses finite-step accuracy

- At nominal, LTV state mismatch against nonlinear source <1e-7 (ey/epsi/u). Local first derivative generally agrees with source finite differences. QP ADMM low residual does NOT certify nonlinear model fidelity.
- Full raw QP command direction: lateral prediction error 0.158m at progress 23.8618m and 0.359m at 39.8147m, with yaw-rate mismatch 1.79rad/s and 0.78rad/s.
- The same direction at alpha=.04 has lateral error 0.000347m and 0.000067m. This is locally higher-order, rather than a constant nominal-state affine offset.
- All three measured raw directions had zero source-plant brake branch steps and zero physical command clamp stages. Nonlinear behavior is real even when input/plant branch switches do not happen.
- Post-solve Riccati feedback alters raw QP commands; steering difference reaches 0.088rad / 0.034rad in these two cases. The physical candidate rollout cannot be judged as if it were the raw QP primal.

### Frozen direction objective mismatch

Cost values below are changes from alpha=0. Model/exact absolute objective constants differ.

| Progress | alpha | LTV QP cost delta | Exact source cost delta | Actual corridor valid |
|---|---:|---:|---:|---|
| 23.8618m | .125 | -71.1 | -57.3 | yes |
| 23.8618m | .25 | -132.6 | -70.5 | yes |
| 23.8618m | 1.0 | -303.2 | +1526.0 | yes |
| 39.8147m | .125 | -96.6 | -90.6 | yes |
| 39.8147m | .25 | -180.3 | -133.3 | yes |
| 39.8147m | .5 | -309.1 | +561.3 | yes |
| 39.8147m | 1.0 | -412.2 | +5281.8 | yes |
| 50.8604m | .04 | -20.7 | -17.6 | yes |
| 50.8604m | .125 | -61.7 | +202.4 | yes |
| 50.8604m | 1.0 | -263.5 | +62757.5 | no, stage 29 |

Hence more or smaller line-search halvings mask a badly optimistic direction. Previous complete practice runs: four backtracks 49.210s / ey-RMSE .0396m / alpha=0 10.5%; six backtracks 49.664s / .0457m / 2.4%; eight backtracks 49.864s / .1209m / 1.2%. Lower alpha=0 frequency is not performance.

### Hessian directional/steering/throttle interaction

Offline channel-decomposed finite differences at nominal around alpha=.02 (rounded; numerical values depend on step size, and are NOT universal curvature multipliers):

| Progress | Hessian component along source QP direction | QP | exact nonlinear |
|---|---|---:|---:|
| 23.8618m | steering-only Hss | 991 | 1836 |
| 23.8618m | throttle-only Htt | 2163 | 4524 |
| 23.8618m | mixed Hst | -1274 | -1922 |
| 39.8147m | steering-only Hss | 76 | 1339 |
| 39.8147m | throttle-only Htt | 659 | 568 |
| 39.8147m | mixed Hst | +45 | -261 |
| 50.8604m | steering-only Hss | 4170 | 11261 |
| 50.8604m | throttle-only Htt | 4612 | 16443 |
| 50.8604m | mixed Hst | -4128 | -8232 |

These are 2D projections along the fixed absolute steering-only/throttle-only QP proposals, NOT individual stage partial derivatives. The sign reversal of mixed curvature at 39.81m and the strong nonlinear second-order steering sensitivity justify testing a richer coupled Hessian. Large finite-size corrections still exceed what a local Hessian can predict.

## 3. Current structural limitations and intended architecture

Current physical source dynamics: full held-command nonlinear source plant, 1ms physics internal integration, 25ms prediction stages. Optimizer: 20 physical state coordinates (ey, epsi, u,v,r, physical steering, wheel speeds, body pose/suspension), progress deviation and two previous-command coordinates (23 total), 30 stages, 2 command decisions. Omitted command-old samples are history and the low-speed source timers remain internal to source-plant rather than independent optimizer coordinates; audit their influence in low-speed regions, but do NOT assume they cause high-speed corner divergence.

QP dynamic equality uses stage-specific first derivative A/B and affine d at an exact nominal. Current stage cost supports diagonal Q and R, restricted N, and diagonalized per-wheel transition slip Gauss-Newton. The Riccati P matrix is full due to dynamics, but the stage Hessian representation is not general. Path/corridor projection and future nominal progress also enter linearly. Nonlinear exact validation can reject a QP solution which is locally feasible but has badly changed steering/yaw/slip.

### Recommended implementation sequence

1. **Establish exact cycle reproduction and stagewise model-error attribution.** Capture actual live warm-start, measured x0, queued command stamp, QP A/B/d, cost weights/objective schedule, raw primal, ADMM state, selected first action and exact full-trajectory validation in one frozen corner fixture. Replay same epoch without guessing the previous warm-start. Measure per-stage mismatch in position, ey/epsi, u/v/r, wheel omega and rack, and exact body footprint slack. Confirm numerical source Jacobian/AD against finite differences in corner regimes and check x0/command slot handling.
2. **Improve globalization by physically meaningful relinearization, NOT smaller step backtracking.** Implement a full nonlinear shooting/SQP update: start from a source-valid nominal with command queue aligned, build sparse QP, solve, integrate exact source, and relinearize around the new physically realized trajectory as soon as model mismatch grows. Track predicted vs actual objective and stagewise constraint residual. Existing second RTI pass is deadline-gated and not a general multi-iteration globalization method; extend/restructure only after profiling fast, safe full-trajectory iteration under hard 20ms.
3. **Represent the right second derivatives.** Provide a richer symmetric PSD stage Hessian in the Riccati QP: state-state, both-input full 2x2, and general state-input cross terms for tracking and slip. Compute source-derived Gauss-Newton Jacobian-product terms instead of independent diagonals, and update incremental-coordinate transformation for general cross terms. Then test second-order *dynamics* terms (costate-weighted Hessian of source transition) via AD second derivatives or targeted Hessian-vector products, with appropriate regularization. Cost Hessian alone CANNOT make first-order dynamics predict .36m lateral deviations accurately at a full step.
4. **Handle hard corridor feasibility within the optimization.** Linearize actual signed body-footprint/world-wall slack at the moving physical trajectory, with terminal viability, anticipatory braking and steering saturation. A unified feasibility-aware SQP objective may allow elastic state slack *during optimization*, but the published action requires exact source rollout and strictly nonnegative hard-clearance slack. Do not preserve separate wall-relaxation/fallback implementations once superseded and qualified.
5. **Optimize execution cost before expanding iteration count.** Cache unchanged source transitions, stage Jacobians and factorization terms; exploit the two-input dimension and block sparse stage structure. Profile P50/P95/max ADMM, Jacobian, nonlinear rollout, refactorization and acceptance. A general dense 23x23 stage Hessian is possible but must be justified by real corner curvature and measured compute.
6. **Qualify before deleting safety nets.** Freeze current qualified practice baseline, run saved-corner reproduction, unit finite-difference tests and hard-safety test corpora. Require five independent ten-scored-lap practice races with <20ms worst publication, zero collisions, low max ey, no persistent alpha=0/throttle stagnation and <=1-2% baseline-time regression. Then transfer to competition geometry and require one full safe lap before ten-lap repeated qualification. Exact footprint/physical collision checks cannot be removed as solver 'fluff'.

### Success is not literal elimination of all globalization

A genuinely infeasible control state can occur after entering a tight corner too quickly, regardless of numerical solver accuracy. No SQP, DDP or NMPC implementation can guarantee safe feasible future controls for arbitrary physical states. Goal is robust predictable control for *reachable* states, with unified feasibility treatment and an authoritative safety gate; not a false universal guarantee.

Code scope: offline replay auditing in tools/mpc_bag_solver_replay.cpp. Production source and config had unrelated preexisting dirty-tree edits; they are intentionally NOT promoted by this audit.

## Offline A/B: existing second SQP relinearization versus single pass

At the same reconstructed cold-state, original weights and plant, with no runtime budget limit, increment-QP and feedback gain .25:

| Saved progress | One-pass exact merit | Two-pass exact merit | One-pass time (offline) | Two-pass time (offline) | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 23.8618m | 308.416 | 229.962 | 3.55ms | 8.29ms | second pass improved |
| 39.8147m | 356.613 | 328.462 | 5.23ms | 11.14ms | second pass improved |
| 50.8604m | 357.603 | 357.603 | 5.76ms | 11.50ms | first pass retained; second did NOT fix |

All cases passed the reconstructed replay's exact corridor check at the selected action; cost comparisons use the same frozen objective within each replay. This shows existing relinearization can help, but is NOT by itself a universal curvature solution. Offline replay timing excludes real ROS, Unity and publication-gate load, so 11.5ms offline is not proof of a <20ms controller.

## Speed error / corner-entry comparison in complete practice races

Below are averages excluding first warmup lap, by position modulo 26.593m. Sign of the exported speed error is actual speed minus reference (positive is overspeed).

| Practice sector | Qualified single canonical run | Trust-region-free 4-step run |
| --- | ---: | ---: |
| 10-12m | +0.097 m/s | +0.007 m/s |
| 20-22m | -0.176 m/s | -0.053 m/s |
| 24-26m | +0.072 m/s | +0.109 m/s |
| 26-28m (wraparound) | +0.132 m/s | +0.197 m/s |

The trust-region-free version improved average tracking in some sectors but had MORE speed overshoot near the late corner/lap seam. Coupled steering and longitudinal control plus anticipatory terminal feasibility are therefore necessary. Do not hide this with a per-corner speed-cap hack without physics-based evidence.
