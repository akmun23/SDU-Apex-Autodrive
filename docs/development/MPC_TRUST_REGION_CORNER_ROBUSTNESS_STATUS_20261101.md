# MPC status: trust-region removal, corner robustness and current blockers
Date: 2026-11-01. Based on live code inspection and recorded local Unity practice/competition races. Not a promotion or deployment authorization.

## Original requirements and verified status

1. Remove the artificial SQP input trust region. **Implemented**: there is no active trust-region limiter in f1tenth_mpc/{src,include,config,test}. The objective and physical steering/throttle command limits remain. Line search is still used as nonlinear globalization; that is not a trust region.
2. Avoid complete shutdown for a merely *predicted* corridor excursion. **Implemented for diagnostics only**: SDU_APEX_IROS_CONTINUE_ON_MPC_FAULT=1 permits explicitly tagged simulator-only continuation; strict supervisor still aborts on an MPC rejection. Both tested competition runs eventually collided physically. There is no basis to deploy this policy to a physical vehicle.
3. Improve normal high-speed corner handling and convergence, minimizing fallback reliance. **Incomplete**: exact source constraints, QP fidelity and feedback improved; tight corners still exhibit 150-iteration ADMM stops, large residuals, horizon corridor rejections, and recovery use.
4. Revisit weights after solving structural issues. **Defer**: changing lateral/heading/slip weights now can mask a poorly posed QP or missing terminal viability.

## Implemented changes

### Production-path QP and trajectory
- Complete Unity/PhysX-inspired source plant for nominal physical rollout and stage Jacobians, with causal steering/throttle input history and synchronized odometry/Bridge command epochs.
- 30-stage / 750ms source model, direct steering and throttle commands, state-dependent physical footprint/corridor constraints; hard original source validation remains mandatory for certified full-horizon plans.
- Full coupled stage Hessian including state-state, steering/throttle and wheel-slip Gauss–Newton terms, rather than incomplete diagonal surrogates; frozen reference and weighting schedule held consistent across SQP passes while physical plant and corridor geometry are relinearized.
- Command-increment QP with physically bounded absolute and incremental commands. Existing Riccati feedback projection and nonlinear merit/line search remain because removing them has not qualified.
- Optional R2 nonlinear relinearization when the 20ms budget permits; ADMM (150-iteration cap) remains an important numerical limiter in corners. Feasibility restoration and prior/candidate prefix/viability recovery remain enabled in the current practice profile.
- QP matrix fixture version 5 and independent OSQP feasibility/convergence audits capture coupled Hessian terms, permitting objective-equivalent solver debugging.

### Experimental nonlinear and source accuracy
- New f1tenth_mpc/src/mpc_source_adjoint.cpp: 60-control exact-source adjoint using source-Jacobian linearization, one frozen objective; independent tight-step FD audits highlight source tire branch sensitivity.
- New f1tenth_mpc/src/mpc_source_nonlinear_optimizer.cpp: bounded iterative single-shooting physical source optimization with original corridor validation and exact merit acceptance; independent short terminal continuation probe.
- Source-state reuse: exact-validated nominal is reused for adjoint Jacobian construction instead of reintegrating it; regression-tested equivalence. Expensive offline derivative checks are disabled in live polishing mode.
- Isolated experimental ROS mode experimental_qp_source_polish, OFF by default; from source-validated QP trajectory, 1 iteration with up to 3 candidate trial rollouts, deadline-gated, source-validated and terminal-probed. No production profile promotion.
- Alternate experimental source-primary policy attempted earlier: failed before completing a lap due invalid model/corridor/collision; do not promote it.

### Failure/shutdown and experiment policy
- The normal controller retains strict reject/fault on inability to certify useful full-horizon/short-prefix actions; it can still stop a race on an MPC fault, which is distinct from an *observed* Unity collision.
- The simulation diagnostic policy can continue after predictive corridor rejection to study real physical outcomes; it is explicitly not certified safety behavior.

## Results and important limitations

**Qualified practice baseline**: five complete independent 10-scored-lap races at 49.162,49.091,49.175,49.291,49.092s, mean **49.1622s**, no controller/collision failures and no hard publication deadline misses. Corresponding track profile: f1tenth_mpc/config/mpc_practice.yaml. This is the benchmark to preserve.

**Experiment hybrid without terminal probe**: one complete 49.416s race; next full race failed near progress 200m after growing lateral/heading error. Full-horizon objective improvement does not guarantee future viability.

**Hybrid with terminal probe, SQP R2 disabled**: initial 49.264s full practice run, lateral RMSE 2.84cm, peak 12cm, 1 recovery, cycle max 18.615ms; two more consecutive runs completed, fourth failed near progress 136.735m with physical corridor rejection; ADMM recurrent 150-iteration caps and occasional enormous residuals. Does NOT meet 5-run criterion.

**Last claimed 'R2 restored' hybrid**: practice race completed 49.407s, peak lateral error 48.9cm, 3 viability recoveries, max 18.744ms, no deadline misses. **Important code-state discrepancy**: as of the final inspection, mpc_rti.cpp STILL contains
`const int sqp_limit = configuration->experimental_qp_source_polish ? 1 : ...`.
The recorded experiment's `sqp_iterations_attempted>=2` count is ZERO. Thus the latest race is NOT evidence that actual R1/R2 SQP plus source polishing works. A documentation comment in mpc_rti.h asserts R2 preservation, but the code does not agree. Resolve and add regression tests; do not claim the hybrid preserved R2.

**40Hz feasibility**: one live diagnostic run attempted source refinement 816/827 controller cycles and selected 182; median refinement 4.19ms, but 58 timeout flags and no proven overall speed improvement. New terminal-aware run selected 437 refinements and rejected 62 short terminal continuations. Full cycles of experimental races were <20ms, but repeated race viability remains unresolved.

**Competition**: no qualified complete race. An explicit simulator-only continuation experiment drove beyond many predicted corridor errors but ultimately physically collided. The 6.925s IPOPT trajectory is an OFFLINE source witness, not Unity closed-loop race performance.

**Model fidelity**: causal command epochs are independently confirmed exact in multiple runs. A few 750ms fast/high-slip source-to-Unity pose differences still exceed 5cm; some discrepancies are linked to Unity pose/twist reference incoherence and should not be fitted as tire constants without coherent ground truth.

**Test status**: local CMake and 21/21 CTests passed repeatedly for recorded modifications. Working tree contains many staged, unstaged and untracked edits from ongoing project development; no commit made as part of this status document.

## Next implementation steps (strict priority)

1. **Restore a coherent, single QP-SQP baseline**: reconcile the experimental R2-disable flag with documentation and tests; preserve the established 2-pass opportunity and remove contradictory dead code/comments. Add a deterministic test that the selected experimental switch cannot silently suppress R2 when claiming otherwise.
2. **Diagnose and fix bad corner QP numerics** using saved progress~136.735m and ~200m failure states. Independent exact full-Hessian OSQP comparisons, residual-by-coordinate and state/constraint scaling; no arbitrary rho sweep, low-quality QP acceptance, or weight-based masking. Track 150-iteration caps, primal/dual divergence and nonlinear valid direction rates.
3. **Integrate terminal viability as a meaningful optimization condition** in the QP/SQP (terminal penalty/continuation or tightened feasible terminal set derived from reachable source trajectories), not merely a post hoc heuristic gate on source polishing. Keep the FULL original physical corridor check for acceptance; an unviable 750ms iterate should be improved inside the normal optimization.
4. **Develop a source-consistent curvature-aware QP direction**: reuse integrated nominal, source A/B and shared cost Gauss–Newton curvature; score exact nonlinear merit and integrate at least the feasible-source direction into normal SQP, not a separate autonomous gradient follower. Minimize repeated model/Jacobian and validation calls under timing budget.
5. **Remove reliance on fallback by obtaining viable normal actions** in difficult corners. Until proven, keep only bounded certified prior/short-prefix recovery and no fabricated full-brake output. Distinguish a deadline miss from numerical infeasibility. Strict race abort after unresolvable rejection remains, while simulation diagnostic continuation may be used to gather causal data.
6. **Requalify**: exact epochs, high-slip source-vs-Unity accuracy, offline hard corner corpus, hard-corridor feasibility, worst complete 40Hz controller cycle <20ms (prefer 15-17ms), 5 consecutive complete practice 10-lap races below performance target, THEN full competition races. No further recovery removal until these pass.
7. **Clean superseded paths only after validated replacement**; isolate experiment toggles and configs, keep production practice and competition profiles distinct and unchanged during development.

Key runtime paths:
- f1tenth_mpc/src/mpc_rti.cpp, mpc_rti_qp.cpp, mpc_source_adjoint.cpp, mpc_source_nonlinear_optimizer.cpp
- f1tenth_mpc/src/mpc_controller_node_impl.hpp
- f1tenth_mpc/config/mpc_practice.yaml and mpc_competition.yaml
- live_runs/phase18_qp_polish_terminalcontinuation_20261101_{04,05,06,07}/
- live_runs/phase18_r2_source_polish_terminal_20261101_01/ (label not proof of R2 actual)
- docs/development/MPC_QP_SOURCE_OBJECTIVE_FIDELITY_20261101.md
- docs/development/MPC_SIMULATION_CONTINUE_AFTER_CORRIDOR_REJECTION_20261101.md
