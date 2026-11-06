# Frozen QP vs exact source: why OSQP feasibility does not establish MPC feasibility

Date: 2026-11-02. Scope: mathematical QP/source fidelity and in-house Riccati/ADMM; no production race profiles or fallbacks changed in this audit.

## Reproducible commands

```bash
# In a Python environment with OSQP 1.0.4 and scipy:
source /opt/ros/jazzy/setup.bash
colcon build --packages-select f1tenth_mpc --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
cmake --build build/f1tenth_mpc --target mpc_live_failure_replay mpc_qp_fixture_replay -j6
python3 tools/audit_mpc_osqp_source_fidelity.py \
  live_runs/sqp_integrated_phase1_capture_20261102_01/captures \
  --cases 2 6 10 \
  --restoration-relaxation-m 0.18 \
  --output /tmp/mpc_osqp_source_fidelity.jsonl
```

The audit reads the *identical recorded* R1/R2 .fqp Hessians, gradients, linearized dynamics, control and state bounds, footprint polygons and command-epoch source context. Offline OSQP solutions are converted using the native C++ absolute/increment command convention, and fully rolled through the Unity-source plant and original strict corridor. The world-map callback is unavailable in these saved contexts; its absence cannot authorize an action. The two QP variants are **as captured** and **the same frozen restoration objective with 0.18 m of wall relaxation undone**. The latter is a strict-linearized feasibility experiment, not a reconstructed racing objective.

## Findings

1. **Riccati/ADMM convergence is insufficient at the recorded budget.** Native replay commonly exits at 150 ADMM iterations with residuals 0.14–7+. OSQP solves some identical QPs in 175–1,375 iterations (different solver, convergence measures and numerical scaling). Increasing the native offline iteration cap to 1,600 made case-6 R1 and R2 converge at 546 and 984 iterations, but case-10 did not converge at 1,600. The experiment's global compile-time iteration cap was reverted to the original 300 after testing; production has not been changed.
2. **The captured R1/R2 QPs deliberately relax physical wall polygons by 0.18 m.** At unchanged source nominal states the recorded polygon margin exceeded exact-source clearance by precisely **0.175 m** (= relaxation 0.18 minus strict-source tolerance 0.005). `mpc_rti_prepare_feasibility_restoration_qp()` applies the relaxation. It is a *search device*, not a proof of strict physical safety. An added GTest checks strict-QP/source margin parity and explicit restoration relaxation. The footprint fidelity diagnostic had incorrectly normalized a lateral plane by the Euclidean norm of metres/radians; that measurement bug is fixed.
3. **Reinstating the strict linearized walls does not fix finite-step accuracy.** OSQP still solves cases 2, 6 and 10 for R1/R2, but all 12 tested optimum source rollouts (6 as captured + 6 strict restored) violate the original full-horizon corridor. Case 16 R1 and case 20 R1/R2 have truly linearized-infeasible stored QPs; their OSQP failure is not a native-solver bug.
4. **The first-order source model is locally meaningful but not globally predictive.** On case-6 R1, at a candidate scale of 0.000244 the maximum source-vs-linearized state error is 0.0000694 in the highest-error native channel; at 0.000976 it is 0.00128 (~18.5x for 4x perturbation), consistent with second-order error in a small neighborhood. At full scale, the max error is ~15.19 **rad/s of front-right wheel angular speed**, not 15.19 metres. On the same full-scale strict OSQP solution, the largest measured errors are: lateral e_y **1.702 m**, heading e_psi **0.529 rad**, progress **0.455 m**, yaw rate **9.27 rad/s**. These are fatal nonlinear prediction errors for an entire 750 ms horizon.
5. **Backtracking fails from a pre-existing infeasible nominal.** In saved case-6, original strict-corridor nominal fails at prediction stage 23; even scaling the solved QP correction down to 0.000244 leaves the strict source violation at stage 23. A line search requiring every intermediate plan to be strictly valid cannot converge from this infeasible starting point.
6. **Adding first-order progress slopes directly to all hard wall planes is not a standalone fix.** Isolated Python OSQP experiments used the original source nominal's progress and native map-width derivatives. It increased OSQP iterations dramatically (e.g. case-6 R1 strict-QP 1,050 vs progress-aware 5,175), and still did not produce a source-valid horizon; in some R2 cases the modified QP became infeasible. Nothing from that experiment was merged into the native projection solver. Note that bounds are piecewise/non-smooth around corners, so local width gradients are often zero despite upcoming sharp geometry changes.

## Solver architecture conclusion

**A QP solve is not a proof that the nonlinear source plant can follow its solution.** Standard SQP only gives a valid local first-order subproblem. The current normal RTI chain lacks a *proven infeasible-start globalization* that repeatedly reduces actual nonlinear corridor violation while relinearizing the complete source state after accepted intermediate steps. A successful QP may be the wrong nonlinear step even if OSQP solves it to high accuracy. Nor does increasing native iterations make the nonlinear model globally affine.

**Next implementation should remain entirely within SQP correctness** (do not tune racing weights, modify racelines, remove fallbacks, or add more emergency policies):

- Build a normal SQP nonlinear feasibility merit/filter that treats strict-source constraint violation and racing objective as separate priorities; accept intermediate **non-publishable** full-source iterates only when the *measured* nonlinear constraint violation decreases, then relinearize and solve again. Never publish until the entire original hard corridor is valid.
- Add mathematically explicit elastic constraints in the QP (nonnegative slack decision variables, with well-scaled penalization) so the infeasible-start subproblem is solvable **without silently shifting physical walls by a fixed 0.18 m**. Preserve the full frozen source objective, actuator queue, hard control boxes and stagewise source Jacobians.
- Globalize finite steps using actual nonlinear source-model agreement, infeasibility reduction and second-order corrections at contact/track active-set transitions; do not mistake local Jacobian accuracy for full-horizon physical validity.
- Repair the native QP convergence/KKT issue only against independent strict-QP OSQP parity: equal objective, bound/footprint residual, physical decision-space trajectory and actual solve time. A 0.02 raw ADMM residual can still yield several metres of dynamic mismatch in projected variables when stage dynamics are poorly conditioned.
- First qualify the recorded failure corpus: fewer infeasible-source iterations, decreasing exact violation under relinearization, at least one **strictly source-valid full horizon** from recoverable cases and correct classification of truly infeasible cases; only then optimize to the 20 ms deadline and attempt repeated races. No current results meet this acceptance standard.

## Changes retained from this audit

- `tools/audit_mpc_osqp_source_fidelity.py` reproducible R1/R2/strict-vs-relaxed OSQP-source experiment, with separate 23-state physical-unit error reporting from `tools/mpc_live_failure_replay.cpp`.
- Correct dimensional convention in `polygon_signed_slack_m()` used for diagnostics (no mixing heading radian slope and lateral metres under an arbitrary Euclidean normalization).
- `MpcSqpConvergence.QpFootprintAgreesWithExactSourceAtNominalAndExplicitRestoration`: regression of exact-source footprint, unrelaxed QP and the separate 0.18 m restoration QP.

All changes remain uncommitted; do not replace the established racing controller or claim successful convergence based on these results.
