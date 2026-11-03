# MPC runtime hot-path cleanup audit — 2026-11-03

## Scope and release safety

This pass removes **proven redundant work and superseded optional algorithms** from the current native 40Hz RTI controller. It does **not** promote the still-offline source-exact 60-control SQP or relax any hard nonlinear collision, actuator, sensor, source or deadline check.

The repository already had a large set of uncommitted edits at HEAD `4bebe49ae`. No mass reset, history rewrite or bulk deletion was used. Historical failure fixtures and the source-exact offline reference were retained.

## Removed from the native control cycle

1. **Repeated exact-source R1/R2 objective scoring.** Both full-source validated pass candidates already carry their exact nonlinear objective and minimum physical clearance. R2 now compares those finite cached values directly. It does not run `mpc_rti_score_candidate_plan` again or copy each complete accepted pass into a duplicate comparison plan.
2. **Repeated state/progress trajectory copies.** Exact-source rollouts now integrate directly into `pass->candidate.states/progress`. Intermediate line-search candidates and rejected-partial source prediction diagnostics reuse the destination buffers. Candidate publication still requires a valid fully checked source rollout.
3. **Always-constructed raw-proposal diagnostics.** The separate raw-QP full-source diagnostic plan and widened-corridor evaluation now exist only when `capture_sqp_prediction_metrics` is enabled. The ordinary source prediction error path still calculates its original comparison data, including partial-rollout stage errors.
4. **Unqualified rank-one directional source-cost curvature branch.** Removed the second-order-only opt-in runtime overlay, its five full 30-stage source-cost evaluations, configuration switch, old test and A/B variants. The reference accurate reduced 60-control source Hessian is a *different* offline method and remains available for independent mathematical comparison.
5. **Superseded source-informed rank-one secant QP and alternative gradient-search fallback.** Removed both opt-in flags, extra gradient-descent candidate rollouts and runtime `MpcRtiSqpCurvature` full-QP storage. Their departure eliminates the formerly **unconditional** 30-stage condensed QP gradient computation in *every R1/R2 pass*. This is a correctness-preserving removal: both paths were disabled in the established baseline and did not form part of the qualified current controller.
6. Removed matching dead config/telemetry fields and unit tests that exercised only the removed modes. The offline replay retains historical experiment IDs as explicitly skipped, rather than silently switching their meaning.

## Tests / no regressions

- **59/59 replayable frozen competition failures** match the saved pre-cleanup baseline (same status, QP iteration counts, relevant numerical residuals and exact nonlinear objective within test tolerance), no changed cases. Other 26 archived captures were historically unreplayable with the modified working tree and must not be counted as passes.
- Full native build succeeds; **22/22 CTest suites pass**. The targeted `test_mpc_sqp_convergence` contains 36 passing tests after deleting three tests dedicated to retired algorithms.
- Rebuilt the offline source-model C++ servers against the updated library after changing public C++ configuration structure layout. Exact G/Q/D source-merit parity, source physical rate/footprint Jacobians, one-shot versus persistent exact-QP binary fixture and the two-iteration strict-source recovery pass. The frozen reference still reaches **exact physical source feasibility** with final objective **4478.462727498** in two SQP iterations (~124ms offline in this particular regression).
- Original full-Hessian source-SQP mathematical tests pass, providing a comparison oracle. No racing/Unity feedback test was claimed.

## Remaining code categories — deliberate

| Category | Defaults / condition | Decision |
| --- | --- | --- |
| Exact source/full-horizon collision, actuator limits, safety and publication | Required | **Keep.** Not redundant fallback. |
| R1 / R2 QP and source relinearization | Normal solver | **Keep** until the single new 60-input source SQP replaces it with independently tested equivalent behavior. |
| Four-step exact-source line search | Bounded, gated, optional configurable | **Keep temporarily**: it remains the only qualified globalization when the existing finite-step QP is wrong. Remove when native feasibility-first globalization and exact objective acceptance are certified. |
| Nominal-source feasibility repair and secondary QP restoration (some with wall relaxation for search) | Opt-in, disabled by default | **Not currently on baseline hot path.** They are redundant long-term with the new source SQP; remove after replacement has corresponding recovery tests rather than deleting the only currently available optional recovery before the replacement exists. No relaxed wall may authorize publication. |
| `source_violation_filter_sqp`, elastic corridor QP | Opt-in, disabled by default | **Retain as independent mathematical QP parity/Phase-I reference only for now.** Its linearized elastic cost can buy wall violations and cannot replace source-exact phase I. Remove from runtime once native source-first feasibility QP meets parity tests. |
| ROS fallback episode, revalidation of previously certified command, emergency stop and viability recovery | Mostly disabled or deadline/fault gated; includes required safety | **Keep** until live successor feasibility and a superior command-continuity path are qualified. Cannot remove collision safety merely to save compute. |
| QP capture/replay and diagnostic output | Explicit flag / async | **Keep gated** for debugging and correctness; no file I/O in production solve. |
| Offline exact-source finite-difference and full-cost-Hessian experiments | Not in native ROS MPC executable's control path | **Keep as regression/oracle utilities**; they have no direct per-cycle compute cost. The fast source-SQP prototype still imports the full-Hessian reference module. |

## Next priority (not completed in this cleanup pass)

Implement the **single native 60-control source-derived SQP** with accurate full-source vehicle-state sensitivities, hard rate/bound constraints and exact-source feasibility-first acceptance. Then prove local QP predicted-versus-source state/cost/clearance parity, horizon prefix improvement and recursive feasibility on a common corpus. Only then delete the remaining fallback solver implementations and auxiliary QP search variants, while retaining hard safety protections. The 20–25ms deadline remains unmet by the current ~124ms offline reference, so no release/drive claim is justified.
