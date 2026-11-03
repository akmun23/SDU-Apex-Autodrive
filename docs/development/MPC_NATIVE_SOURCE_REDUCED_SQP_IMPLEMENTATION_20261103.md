# Native full-source 60-control SQP integration — 2026-11-03

**Status:** Implemented and linked into the native ROS MPC as an **explicitly opt-in accuracy-first backend**. Source-only frozen corpus qualifies the mathematical path, but **not yet approved for live 25 ms control or for removal of the established RTI solver**. No Unity run or actual racing test is claimed.

## Goal and distinction from existing RTI

The original RTI constructs a 23-state affine multiple-shooting approximation with source A/B matrices and two optional R1/R2 iterations. Its local QP can be solved accurately while the complete full-source control trajectory leaves the physical corridor by centimetres or metres, especially with nonlinear tire/slip, actuator and reference-progress changes.

The new backend eliminates the vehicle state trajectory **exactly**, rather than eliminating physical states. Its optimization variables are the **60 absolute steering/throttle commands**, with normalized coordinates steering/.15 and throttle/.25. Every evaluation integrates the complete current checked-in source plant from the same measured command epoch, including source internal wheel/suspension and queue states, and samples the original progress-dependent physical corridor.

For commands z, the actual constrained problem is:

- x[k+1] = F_source(x[k], u[k])
- original left and right physical footprint margin c(x[k], progress[k]) >= 0 for all 30 predicted stages
- actual source-dependent steering rack slew and throttle command slew >= 0
- original steering and drive-only throttle absolute bounds, steering within +/- max_steering and throttle in [0.001,1]
- same frozen source racing objective J(z) as the controller, never a separately updated R2 schedule.

The local SQP is still **quadratic**, but its 60x60 wall Jacobian and 120x60 slew Jacobian are obtained from perturbing the complete nonlinear source rollout in all 60 command directions, not from a reduced tangent model that omits 22 physical hidden source-state coordinates.

## Native algorithm

1. **Source evaluation.** Build one same-epoch source nominal and the exact original 30-stage source cost, 60 wall constraints and 120 source rack/throttle rate constraints. Strictly source-validate any candidate intended for publication.
2. **Finite-difference derivatives.** 120 full-source perturbed rollouts for 60 normalized controls. Use central finite differences and retain state/corridor progress dependence. The cost gradient is measured directly from the exact objective.
3. **Infeasible-start Phase I.** First solve a hard-constrained minimum command-correction QP using the full constraint Jacobian. If the **linearized QP** is infeasible or uncertified, solve a nonnegative-wall-slack minimization with rate/command constraints still hard; fix the minimum summed slack (within numerical tolerance) and minimize correction norm. A tiny strictly positive regularizer makes the native slack LP numerically solvable by the same interior-point method. This is *not* a weighted racing-cost-versus-wall-penetration compromise. A negative exact wall margin remains invalid.
4. **Feasible-start racing SQP.** Use the original racing gradient and a convexified source-derived **diagonal** cost Hessian; apply Powell-damped BFGS using later accepted source gradients. This is not yet a full exact Lagrangian Hessian. The existing Python full-Hessian reference remains available for comparison.
5. **Native QP solver.** In-process double-precision Eigen dense primal-dual predictor-corrector interior-point method, deterministic bounded iteration count and independent affine primal/KKT certificate. Every constraint is checked in native coordinates before any direction can be used. No Python or OSQP execution in the live path.
6. **Globalization and source SOC.** Backtracking accepts the first genuinely source-improving step. Exact-source second-order constraint correction solves a new minimum-norm QP around the *actual nonlinear full step* using its measured margin defect. The already evaluated full-step source rollout is reused; no redundant second full-source integration just to start SOC. All SOC steps must be independently source evaluated again. Nonfeasible decreasing-V intermediate steps only seed the next SQP iteration; **they are never eligible for command publication**.
7. **Incumbent and publication.** Keep the last accepted source candidate. Only an exact full-horizon source-valid result with source-compliant rates and original corridor constraints sets the controller's accepted status and returns a candidate control sequence. Infeasible, timed-out or numerically failed results preserve diagnostic violation/prefix data but restore controller memory and expose **zero candidate controls**.
8. **Deadline.** The absolute command deadline is checked before every expensive source oracle evaluation. A missed deadline cannot certify a late candidate. Dense QP solve and source inner calls are not yet hard real-time; consequently this mode is **not live-qualified**.

The local SQP's artificial 0.25 mm wall *tightening* is only numerical conditioning, never a relaxation of the physical validation.

## Files implemented

| File | Responsibility |
| --- | --- |
| f1tenth_mpc/include/mpc_source_reduced_sqp.hpp | In-process native SQP API, settings, stats and derivative diagnostic snapshot |
| f1tenth_mpc/src/mpc_source_reduced_sqp.cpp | Full-source rollout/FD derivatives, Phase I, native IPM, SOC, strict acceptance |
| f1tenth_mpc/src/mpc_rti.cpp | Select new backend after freezing the cycle objective; when selected bypass old 23-state Riccati R1/R2 and restoration paths |
| f1tenth_mpc/include/mpc_rti.h | New backend selection and result telemetry |
| f1tenth_mpc/src/mpc_controller_node_impl.hpp | ROS params for backend and SQP iteration count; warning that deadline is not qualified |
| f1tenth_mpc/src/mpc_controller_diagnostics.cpp | Identifies optimizer backend and emits new constraint and timing diagnostics |
| f1tenth_mpc/CMakeLists.txt | Links the new solver into mpc_core, dev-only frozen replay target |
| tools/mpc_native_source_sqp_replay.cpp | Exact frozen source-context command-epoch standalone reference driver |
| tools/mpc_live_failure_replay.cpp | Native backend replay through the **actual** mpc_rti_solve_cycle controller entrypoint, independent of old captured R1 QP parity |
| tools/test_mpc_native_source_sqp_corpus.py | Six-case source-only prefix/horizon regression |
| tools/test_mpc_native_source_sqp_controller_entry.py | Six-case actual controller entrypoint, no unsafe published candidates |
| f1tenth_mpc/test/test_mpc_sqp_convergence.cpp | Three new C++ tests for strict full-source plus rate validation, impossible walls, expired-deadline nonpublication |

## Configuration (development only)

- ROS parameter native_source_reduced_sqp: **false by default**.
- ROS parameter native_source_reduced_sqp_iterations: 8 by default in this mode.
- The currently qualified/default original racing RTI code is unchanged whenever the new option is false.
- The flag is deliberately not enabled on the practice or competition race profiles. It will cause ordinary 20–25ms control-cycle deadline rejection in the current source-FD implementation, so should **not** be enabled in a race.
- Even when the new backend is selected, the ROS publication layer retains its explicit emergency/command-continuity safety policies; these are NOT the normal QP search path and must not be deleted without a qualified replacement.

## Numerical evidence

**Exact control-space source derivatives:**
- On the saved near-wall reference, maximum 60x60 physical wall Jacobian discrepancy between C++ and the qualified Python finite-difference source reference: ~2.34e-11.
- Maximum 120x60 actuator-rate Jacobian discrepancy: ~4.91e-6, consistent with source float command quantization.
- Source/QP both start at exact nonlinear cost 6531.9060046 and physical squared wall violation 0.1530984859.

**Reference nonlinear corner:**
- Both native and Python source-exact methods recover a fully strict source-valid 30-stage horizon after **two accepted SQP corrections**.
- Native exact source objective = 4478.3877327 and wall V = 0.
- Python reference objective = 4478.4627275 and wall V = 0.
- Difference in the final 60 absolute command coordinates at most ~2.7e-6.
- Uncorrected local QP's first full step predicted V=0 but the source world model reported V~0.01222 m2 and a maximum 0.216m constraint-margin discrepancy. Source-exact SOC corrects the finite-step mismatch, with verified final V=0. This is proof of the distinction between accurately solving an affine QP and accurate nonlinear control prediction.

**Six fixed competition failures** using eight attempted native iterations:

| Recorded case suffix | Initial squared wall V (m2) | Final source wall V (m2) | Initial first violated stage | Final first violated stage | Full source-valid |
| --- | ---: | ---: | ---: | ---: | --- |
| 1793641756087085412_case_6 | .05329 | 0 | 10 | none | Yes |
| 1793692853295529252_case_6 | .16147 | 0 | 9 | none | Yes |
| 1793692917333601071_case_6 | .15310 | 0 | 23 | none | Yes |
| 1793693111767116537_case_6 | 8.67354 | 1.25214 | 21 | 23 | No |
| 1793693111641647593_case_2 | 1.21615 | 0 | 26 | none | Yes |
| 1793693111867025977_case_10 | 23.60754 | 1.61950 | 17 | 22 | No |

- **4/6 full-horizon source-valid**, and **6/6 longer feasible prefixes**; two not recovered remain explicitly nonpublishable. The two difficult frozen failures still advance physical feasibility despite QP convergence limitations.
- This is strictly conditional on the already-recorded source state. An infeasible frozen suffix is **not** proof that a closed-loop controller would have driven into that state. Conversely, valid frozen horizons do **not** establish recursive feasibility.
- Native runtime is hundreds of milliseconds to more than a second in difficult cases, with schedule-dependent variance; this is not close to the 20–25ms deadline. The offline 8-process Python constraint-FD implementation was faster than the sequential native FD oracle in some reference runs. **Timing optimization is deliberately not the current correctness criterion.**

## Verification and remaining blockers

- New native controller entry synthetic tests: independent full 30-stage rollout, all actuator-rate inequalities, same exact-source objective and first control; impossible future corridor cannot certify; an expired deadline exposes no candidate.
- Source-only six-case regression: 4/6 strict full horizons, 6/6 longer prefixes.
- Original legacy replay parity: 59/59 frozen replayable cases unchanged when new mode disabled.
- Complete native ROS package build + 22/22 CTest suites passed, plus previously implemented Python exact-source FD/second-order source regression.
- The 60-command solver does not rely on an old 23-state QP model for acceptance; however, **its own local quadratic model remains an approximation** and can predict incorrect full finite steps. SOC and exact re-evaluation prevent treating it as a source certificate.

**Still required before removing the original Riccati and fallback implementation:**
1. Improve the new dense IPM's scaling, hot-start/regularization and failure handling on the 2 still-infeasible frozen examples; do not silently treat an iteration-limited QP or soft wall slack as a solution.
2. Implement validated full-source internal-state automatic/analytic constraint sensitivities instead of 120 sequential rollout finite differences per SQP iteration; benchmark parity in high-slip, contact-branch transitions and near actual steering/throttle bounds.
3. Upgrade/validate reduced Hessian approximation against source objective and feasibility across full nonlinear step sizes; test independent SOC QP feasibility/KKT and safe source-cost globalization.
4. Establish first-command successor feasibility across reoptimized epochs, rather than only a feasible frozen prefix. Source/Unity yaw and wheel-speed discrepancies in earlier recorded failures remain unqualified.
5. Only after 1–4, do timed synchronized Unity closed loop on both tracks and enforce 20–25ms hard deadline. Then remove remaining superseded R1/R2, restoration/fallback *solver search* code. Keep external hard-collision and publication-safety guards.

**No simulator races were run; the new backend is not yet promoted to competition or practice.**
