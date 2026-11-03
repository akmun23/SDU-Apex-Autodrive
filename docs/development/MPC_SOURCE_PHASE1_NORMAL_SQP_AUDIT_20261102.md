# Source-exact Phase-I initialization inside normal SQP — November 2, 2026

## Scope

The existing live-failure QP/plant capture and original racing SQP were retained. Added a **disabled-by-default** `source_nominal_feasibility_repair` parameter which, after the normal physical nominal and frozen objective are built, uses the already existing exact-source Phase-I corridor optimizer to try to find a strictly valid nominal **before** R1. A successful source-valid repair is used solely as the normal R1 linearization seed. No Phase-I output can bypass the normal SQP, strict nonlinear checks or command publication deadline, and neither canonical racing YAML has been promoted.

The Phase-I routine is now a shared source-model helper, not a second controller. It optimizes physical footprint violation using source sensitivities and respects the configured steering/throttle direction norms instead of hard-coding those values. Only a complete 30-stage source-certified solution may replace an infeasible nominal. A failed repair leaves R1's original nominal and objective unchanged. The successful Phase-I path now carries its complete source-certified trajectory directly into R1, rebuilding reference samples but not recomputing the 30-stage source rollout or running a duplicate strict validator. The caller reserves at least 10.5 ms for subsequent SQP/ADMM/validation and bounds nominal repair to an intended 4.5 ms (source sensitivity/rollout kernels are not preemptible mid-call; measured repairs sometimes took 6–9 ms). Repaired seeds reset stale ADMM duals, but don't alter the frozen objective schedule or the source wall contract.

## Exact captured failure corpus

Source: `live_runs/sqp_integrated_phase1_capture_20261102_01/captures/` (30 saved cycles at progress 50.202–57.494 m). The live QP capture is not itself a feasible control trajectory. Exact QP/ADMM replay parity still fails for R2 cases 6, 16 and 20 (max absolute deltas 2.03, 1.95, 1.36). These remain separate reproducibility blockers; do not call the entire corpus bitwise-exact.

On the 27 cases with passing R1/R2 fixture parity, running a fresh-memory normal R1 offline with the same frozen objective and physical state gave:

| Experiment | Full exact-source horizon accepted | Offline worst/failure case treatment |
|---|---:|---|
| Original normal SQP | 0 / 27 | Failed full-horizon validation in every case |
| Original SQP + source Phase-I nominal repair | 1 / 27 | Case 0 at 50.202 m converted from rejected horizon to accepted, strictly validated line-search horizon |
| Phase-I + existing feasibility-biased first QP | 1 / 27 | No additional full-horizon recoveries; do not promote |

In case 0, one source Phase-I iteration produced a certified nominal. Before redundant-integration removal, the complete offline R1 comparison took ~6.2 ms versus ~4.3 ms without it. After reuse, 8 repeated offline comparisons measured a mean **5.70 ms** repaired cycle and **4.26 ms** unrepaired cycle; the repaired Phase-I operation itself averaged **1.82 ms**. The unrepaired cycle still rejected the exact source horizon while the repaired cycle accepted it. Later corpus states typically required 3.8–4.2 ms for Phase-I and were not made fully feasible. Even stronger steering/throttle directions or four Phase-I iterations reduced the scalar corridor violation without producing additional valid full horizons. Do not conflate decrease of this soft violation metric with physical feasibility. Offline fresh-memory SQP comparisons do not exactly reconstruct the live warm-start state, although separate live-QP captures retain those states.

The formerly reported generic maximum state difference (e.g. 1.09 or 1.29) combines physically different state coordinates; inspect its index and units before calling it a metre-valued tracking error. Footprint and clearance errors are measured separately in metres.

## Strict Unity/ROS closed-loop trials

All competition trials used the 6.925 s source-witness raceline, strict zero nonlinear corridor tolerance, source-informed SQP and 40 mm soft near-wall reserve. Source Phase-I was opt-in through local overlays; the canonical competition profile was not modified. No collision occurred in these trials.

| Trial | Extra features | Progress | Outcome |
|---|---|---:|---|
| `comp_phase1_control_20261102_10lap_01` | Phase-I disabled; control A/B | 100.532 m | 1 total lap, then strict source corridor rejection at stage 0 |
| `comp_phase1_strict_20261102_01` | Phase-I enabled | At least two total laps | Completed requested two-total-lap trial, no strict rejection |
| `comp_phase1_strict_20261102_10lap_01` | Phase-I enabled | 124.793 m | 2 total laps, then stage-0 corridor violation **8.097 mm**; 31 accepted prior-plan cycles, one accepted prefix; R1 ADMM 150-iteration cap, primal residual 0.297, Phase-I attempted but no feasible 750 ms horizon |
| `comp_phase1_terminal_20261102_10lap_01` | Phase-I + existing terminal viability | Before first lap | Strict source corridor rejection; no qualification |

In the long Phase-I trial, the feature was attempted 42 times and produced nine full-horizon repaired nominals before the eventual failure. Around progress 122.689 m, repeated prior-plan publications began while the invalid R1 horizon's first failing stage advanced from about 21 toward 0. By progress 124.793 m the queued steering/throttle and source state were outside the strict next-step envelope. This is a **recursive-feasibility failure developed across many cycles**, not merely a failure of the final ADMM solve. The final cycle took ~13.81 ms and did not miss its deadline, although the run included a 20.52 ms measured end-to-end outlier and remains unqualified for a hard 20 ms bound.

Machine-readable records: `live_runs/comp_phase1_strict_20261102_10lap_01/{mpc_cycle_diagnostics.jsonl,mpc_run_report.json,mpc_run_report.csv}`, corresponding control run and two-lap run. `live_runs/` is gitignored; this document is the tracked evidence summary, **not** a replacement for the local datasets.

## Validation and disposition

- Local Release MPC build passed.
- New `MpcSqpSourcePhaseOne.RepairsTinyLateHorizonViolationWithStrictPlant` test passes; it constructs a late 3 mm physical corridor violation and requires a source-certified repaired full horizon.
- The final local Release MPC build and the full `colcon test --packages-select f1tenth_mpc` run passed **249 tests, 0 failed**, including the new cached-certified-vs-independently-reintegrated-plan parity assertions. `git diff --check` passed.
- Parameter `source_nominal_feasibility_repair` stays **false** in both tracked practice/competition profiles.
- The strict Unity trials in the table were run on the earlier source implementation before reuse optimization and the configurable Phase-I direction fix. The subsequent optimized binary has passed offline tests but **has not completed a fresh live 10-lap qualification**.
- No five-run ten-scored-lap qualification was established. Do not call the solver or practice/competition profiles qualified by these experiments.

## Remaining high-priority engineering

1. Correct the high-slip, finite-step SQP prediction model: source-derived directional curvature alone is not a substitute for a useful nonlinear Hessian/step; improve first-pass physical accuracy and test per-pass cost/footprint agreement on all captured states.
2. Build **proactive recursive feasibility** into the normal MPC formulation. Source-exact Phase-I only repairs a full-horizon infeasible nominal where one can still be found; it cannot undo already queued corner-entry commands. A shiftable strictly feasible plan with certified terminal controlled continuation is needed, including proper actuator FIFO and source/Unity uncertainty.
3. Remove repeated retained-plan reliance: a long train of accepted 75 ms prefixes/certified prior actions progressively approaches the boundary even while a complete new horizon cannot be obtained. A valid normal full-horizon solution must be found before the frontier reaches stage 0.
4. Investigate three non-bitwise-identical captured R2 replays; validate exact source rollout and physical one-step/750 ms source-versus-Unity agreement at the critical command epochs. Avoid conflating mixed-unit state maxima with position error.
5. Bound source Phase-I kernel execution and overall P99.99 command publication within 20 ms under real CPU load; then qualify 5 independent runs × 10 scored laps on both maps.

## Globalization and runtime follow-up (same day)

- Removed the optional **persistent post-QP trust-radius multiplier** from the normal SQP. The original raw QP proposal now always goes through exact-source acceptance without pre-shrinking. The legacy radius memory/metric fields remain as no-op placeholders solely to preserve the captured `source_context.mctx` POD layout. A positive `minimum_sqp_model_agreement_ratio` is now a *pure exact-source merit filter*; zero disables it in canonical race profiles. Eliminated a duplicate QP objective evaluation made redundant by dropping the post-solve radius. `MpcSqpGlobalization.AgreementFilterDoesNotReintroduceScalarTrustRegion` covers this behavior.
- Retained the four exact-validated nonlinear backtracking alpha trials (`0.5, 0.25, 0.125, 0.0625`) for now. In the previous control/repair-enabled strict competition bags, the nonlinear line-search P95 wall time was approximately 4.15/3.34 ms, versus 2.78/2.71 ms median QP/Jacobian construction. Backtracking selected physically valid damped solutions in dozens of cycles; blindly removing it would reduce the solver's ability to publish full-horizon legal actions, not improve nonlinear convergence.
- Diagnosed all **three** formerly non-identical recorded R2 QP replays: live ADMM hit its wall-clock budget at 122, 142 and 137 iterations, while the offline solver ran 150 iterations. `mpc_live_failure_replay` now detects deadline-interrupted captures and reproduces the recorded ADMM iterate at its exact iteration count with a disabled offline clock gate. Cases 6, 16, 20 now replay with zero solution delta, as do the other sampled captures. This is exact mathematical-iterate parity; it does not reproduce original operating-system scheduling.
- Found that source Phase-I used all eight trial scales even when `max_exact_trials_per_iteration=1` was requested. It now obeys the trial cap. On the saved source-failure corpus, one attempted trial retained the early case-0 full-horizon recovery and reduced difficult-case Phase-I cost from roughly 4 ms to about 2 ms in isolated offline calls. It is **not yet a race-qualified runtime saving**; a stricter trial cap may also reduce recoverability on unseen live states.
- A separate offline 6×6 search over independently interpolated steering and throttle alphas found no feasible 750 ms horizon among 60 recorded R1/R2 candidate directions. Because it produced no recoveries, this experimental search was not added to the production solver.

## November 2 late-phase: trust-radius removal, alpha-zero fixtures, and source-gradient globalization

### Solver decisions

The persistent post-QP trust-radius multiplier (`sqp_trust_region_scale` in historical fixtures) has been **removed from active control**. The raw ADMM control proposal is now passed directly into source-exact evaluation. A positive `minimum_sqp_model_agreement_ratio` remains an optional exact-source cost agreement *filter*, not a post-solve trust region; zero remains default. Legacy capture fields retain their physical layout but are no longer read/updated for control. A second QP objective evaluation that was mathematically redundant without radius scaling was eliminated.

The four source-exact backtracking step sizes (0.5, 0.25, 0.125, 0.0625) were **retained**. They select numerous actual full-horizon legal trajectories, and deleting them would hide incorrect SQP directions, not cure those directions. On recent strict competition bag replays, p95 measured `line_search_us` was ~3.3–4.2 ms; that time is significant enough to justify better SQP directions, but not unsafe command acceptance. An offline 6x6 steering/throttle interpolation grid on 60 old infeasible QP proposals found zero source-feasible full horizons; this experimental search was removed, rather than becoming another solver path.

The three previously divergent R2 `.fqp` replays were caused by recorded live **wall-clock ADMM cuts** at 122, 142, and 137 iterations, versus 150 iterations in faster offline replay. The offline tool now truncates precisely at the recorded iteration count (not the wall time); all three obtain **zero raw solution delta**. This establishes mathematical QP-iterate parity, not identical source/Unity physics or scheduler timing.

Phase-I feasibility optimization now obeys its existing `max_exact_trials_per_iteration` and direction norms. The number of Phase-I exact trials is exposed in `source_nominal_feasibility_max_exact_trials`, default 8 for the experimental enabled mode; `source_nominal_feasibility_repair` stays false in both canonical YAMLs. The source-certified trajectory is reused for normal SQP linearization rather than re-integrated. Individual source kernels remain non-preemptible, so requested time budgets are approximate rather than hard per-call bounds.

### Closed-loop A/B for Phase-I cost

| Strict experiment | Full-lap outcome | Max progress | Successful full-source nominal repairs | Worst end-to-end measurement |
|---|---|---:|---:|---:|
| `comp_phase1_control_20261102_10lap_01` | 1 completed lap; stage-0 corridor stop | 100.532 m | disabled | 19.42 ms |
| `comp_phase1_strict_20261102_10lap_01` (older Phase-I) | 2 completed laps; stage-0 stop | 124.793 m | 9 / 42 attempts | 20.52 ms |
| `comp_phase1_capped_notrust_20261102_10lap_01` (1 Phase-I trial) | 0 completed laps; stage-0 stop | 80.355 m | 5 / 19 attempts | 18.86 ms |
| `comp_phase1_repair8_notrust_20261102_10lap_01` (8 Phase-I trials) | 0 completed laps; stage-0 stop | 62.390 m | 4 / 34 attempts | 17.90 ms |

These are separate deterministic-config but non-identical Unity live starts, **not paired identical-state counterfactuals**. Differences in maximum progress cannot be attributed solely to the number of Phase-I trial scales. None qualifies the controller.

### Early alpha-zero stagnation and source-adjoint alternative

Added optional `forensic_qp_capture_zero_alpha` (disabled by default). It records the first alpha-zero accepted cycle and then every 8th alpha-zero cycle, up to 12, through the existing asynchronous forensic recorder, in addition to normal rejected-cycle captures. Strict `comp_alpha0_probe_20261102_01` recorded 12 accepted alpha-zero fixtures at progress 38.953–44.449 m and 26 later rejected-cycle fixtures; it stopped at ~64.55 m. These fixtures contain the exact QP, ADMM warm state, complete source plant and actuator timeline, not just a fresh-memory guess.

On the 12 **still-valid** alpha-zero source-state fixtures, `mpc_rti_source_nonlinear_optimize` with full original corridor gave:

| Initial source gradient scale | Exact trial count | Strictly valid objective improvements | Median isolated runtime |
|---|---:|---:|---:|
| 1.0 | 4 | 0 / 12 | 2–7 ms range |
| 1.0 | 8 | 7 / 12 | variable, up to 9.8 ms |
| 0.0625 | 1 | 7 / 12 | 2.13 ms |
| 0.0625 | 2 | **9 / 12** | **2.55 ms**, max 5.1 ms |
| 0.0625 | 4 | 11 / 12 | 3.02 ms |

This is an isolated local descent from the *captured nominal*, not a guarantee that these commands improve the Unity lap or preserve future feasibility. The 0.0625 / two-trial mode was therefore integrated only as an **opt-in normal-SQP alternative source-gradient direction** (`source_gradient_globalization: true`) when the QP full step is invalid or regresses cost, and only when the nominal is already source-certified and there is time budget. Any selected source-gradient direction is subject to exact full-horizon original-corridor validation and cost comparison; normal source exact alpha backtracking remains in place if it does not improve. It is not a separate fallback controller, and both race YAMLs default it off.

The first Unity run with this feature aborted on **invalid diagnostics JSON**, because an accepted non-QP source-gradient step was assigned `alpha = NaN` but `line_search_selected_alpha` was emitted as raw JSON. This was fixed by using the JSON writer's finite-number/null handling; local build and 250 tests passed. No source-driven safety behavior was inferred from that malformed-diagnostic abort.

The corrected strict Unity trial `comp_source_gradient_sqp_fixedjson_20261102_10lap_01` had **157 source-gradient attempts, 49 selected strictly valid full-horizon improvements, 10 Phase-I attempts, 6 repaired nominals, and only 2 retained-prior-plan acceptances**. However, it still stopped before one lap at progress **62.79 m** on a stage-0 corridor violation; worst reported end-to-end control time was **19.63 ms**. The earlier cap8 trial without this feature had 29 retained prior-plan acceptances and stopped at 62.39 m, but these are not identical state/seed comparisons. The gradient mode reduced alpha-zero/preexisting-plan dependence locally but did **not** solve corner successor feasibility; it is not qualified for production.

The source-gradient path initially re-integrated the exact same physical horizon after the source optimizer had already certified it. The latest change now lets the optimizer return its cached, certified `MpcRtiNominal_t`, which is reused by the normal SQP without that redundant source integration. The strict numerical parity test in `test_vehicle_model_source_rollout.cpp` confirms cached progress, lateral error and yaw rate match an independent full source re-integration; 250 package tests passed. This final cache optimization has **not been separately live-race qualified**.

### Next actual engineering blockers

1. **Recursive feasibility, not merely better alpha:** prevent the 30-stage source corridor violation from marching toward stage zero across repeated valid but non-renewing trajectories. Candidate actions must provide a physically achievable successor/terminal continuation under steering/throttle queues, and source/Unity uncertainty. The existing simplistic terminal-viability trial failed before one lap; it is not sufficient.
2. **Fewer expensive gradient calls:** live runs attempted the adjoint 157 times to select 49 improvements, often for 3–6 ms. Trigger from predicted SQP quality or test only after selected alpha schedules fail, then benchmark under 20 ms end-to-end. Do not remove hard original-corridor validation.
3. **Better source/QP finite-step prediction:** measured high-slip SQP curvature disagreements remain, even though ADMM iterate parity is repaired. Improve model-consistent globalization (possibly true second-order curvature or nonlinear constrained shooting) against saved initial *recoverable* states.
4. **PhysX/Unity agreement and qualification:** verify source/Unity residuals on corner entries, then five independent full ten-scored-lap strict race runs on both tracks; preserve baseline profiles until both performance and execution-time gates pass.

## Fast QP-gradient source-globalization experiment (November 2)

**Result: saved-state improvement and improved strict Unity progress, but no ten-lap qualification.** After introducing the separate source-adjoint globalization, 12 accepted-alpha-zero snapshots were captured at stations 38.953–44.449 m (38 total captured cycles including 26 later rejects) in `live_runs/comp_alpha0_probe_20261102_01/qp_captures`. These represent physically valid nominal plans where the normal SQP could not select a useful finite step.

- Full source-adjoint gradient with normalized per-channel directions and an initial .0625 multiplier (two exact trials) generated strict full-horizon cost improvements in 9/12 alpha-zero cases; median isolated cost about 2.55 ms. Starting at 1.0 and searching 4 scales recovered zero; starting .0625 with 4 trials recovered 11/12 but consumed more time. These are exact-plant replay findings, not Unity.
- The already-built `mpc_rti_qp_source_gradient` matches the useful full-horizon source-gradient *directions* in these snapshots. Reusing the existing QP gradient and testing just physical scales 0.0625 and 0.03125 produced 9/12 exact-source improvements with no second source-LTV/adjoint rebuild. The normal-SQP replay of all 12 captures selected 9 new full-valid horizons. Its median additional gradient work was about **0.74 ms** (wall-time, offline), and source-based convergence was usually faster than completing four unsuccessful alpha backtracks. The original 4-step nonlinear backtracking and full hard constraints remain available when this option fails.
- The single optional `source_gradient_globalization` code path in `mpc_rti.cpp` now reuses the QP's original adjoint gradient, tests at most two strict source-validated trajectories, compares the same frozen racing cost, and respects the 20ms absolute deadline with a 2.3ms final reserve. The superseded extra adjoint construction was removed from the live-path experiment. `tools/mpc_live_failure_replay --qp-gradient` and `--compare-normal-sqp` reproduce the offline measurements from the local corpus. The setting remains OFF in tracked production practice/competition YAML.

Strict closed-loop comparisons using the same nominal geometry-only competition raceline and strict zero nonlinear corridor tolerance (run-to-run variability means distances are NOT guaranteed causal improvements):

| Trial | Feature | Total laps observed | Maximum station | Gradient attempts / selected | Median gradient time | P95 / max end-to-end |
|---|---|---:|---:|---:|---:|---:|
| `comp_phase1_repair8_notrust_20261102_10lap_01` | Source Phase-I; no gradient | 0 | 62.390 m | 0 / 0 | — | 16.32 / 17.90 ms |
| `comp_gradient_repair8_20261102_10lap_01` | Separate source-adjoint globalization | 0 | 77.987 m | 251 / 62 | 4.03 ms | 15.76 / 19.55 ms |
| `comp_fastqpgrad_repair8_20261102_10lap_01` | Reused QP gradient + two exact trials | **1** | **123.197 m** | **503 / 131** | **1.68 ms** | **15.31 / 19.67 ms** |
| `comp_fastqpgrad_terminal_20261102_10lap_01` | Fast gradient plus existing soft R1/R2 75ms continuation preference | 0 | 62.722 m | Selected 27 gradient alternatives; made 232 terminal extension probes, but terminal preference never distinguished either R1/R2 | — | max 19.05 ms |

Every trial ended with a strict `corridor@0` source rejection; no collision was recorded. The one-lap result is NOT a complete race, and the maximal 19.67ms measured cycle does NOT prove a statistical hard WCET bound. The first full source-invalid state remains uncorrectable by a normal correction with the queued actuator commands; recursive viability is still missing.

## Disposition and engineering next steps

1. Keep the fast *QP-gradient* globalization as an opt-in experimental direction-quality improvement. The more expensive duplicate-adjoint live version was removed; do not enable the old adaptive post-QP trust radius or delete exact backtracking.
2. Address recursive feasibility **inside the normal optimization problem**, before the 30-stage tail expires. Existing 75ms held-command continuation preference tested two possible terminal commands and provided no useful discrimination in the 232-probe live run; simply requiring its result as another hard constraint is not justified. Next work should form a source-consistent successor-viability cost/certificate with a realistic *optimized* steering/throttle continuation, including full queue/hidden state, and test the prediction on saved pre-failure states. Do not count arbitrary short-prefix reuse as a full viable plan.
3. Make per-cycle cost of the source-exact gradient and Jacobians more predictable (the fast path still took 3.68ms at P95 for gradient work and 19.67ms at worst end-to-end during the observed run). Avoid adding more source rollouts on every 40Hz cycle without a measurable feasibility benefit.
4. A/B validate at least two maps using the same immutable source/raceline/image/overlay fingerprints, then require 5 independent runs with 10 scored laps per map, zero collisions, all published commands strictly validated and no 20ms deadline misses.

## Consistent in-SQP full-footprint clearance reserve study (Nov 2, evening)

### Problem reconstructed

The strict `comp_selective_gradient_20261102_10lap_01` live trace converged R1 normally (primal residual typically ~0.0006–0.0010) through the last 20+ control cycles. Its minimum full-source footprint slack declined through 14.5, 12.7, 10.2, 7.3, 2.2 and eventually **0.2 mm**; the very next measured/publication epoch then had no strictly legal stage-1 plan (`corridor@0`). This is NOT evidence that more ADMM iterations, a smaller trust region, or a larger alpha grid can recover it. The underlying problem is future/successor feasibility with real delayed-command/hidden-state uncertainty.

Implemented a bounded, asynchronous forensic mode `forensic_qp_capture_near_wall` (defaults OFF) that samples exact-valid SQP decisions in distinct strictly positive clearance bands `<10mm`, `<5mm`, `<2mm`, `<0.5mm`, each capped at three snapshots and at most every fourth qualifying cycle. The `comp_gradient_nearwall_capture_20261102_01` run collected 4 early valid pre-failure contexts near 57.57–58.46m and 13 later rejection contexts. These were captured into local ignored `live_runs/.../qp_captures/` without blocking control callback serialization.

### Exact physical local clearance derivative

Added `mpc_rti_qp_clearance_gradient` in `mpc_source_adjoint.cpp` (declaration `mpc_rti_internal.hpp`). It finds the limiting source-footprint polygon plane and backpropagates its local physical gradient through the QP's *existing source-exact* A/B and command-increment chain rule; no extra plant Jacobian. `MpcSqpClearanceAdjoint.MatchesExactSourceFootprintMarginOnStraight` compares derivative against independently integrated exact source rollouts (passed). The offline `mpc_live_failure_replay --clearance-gradient` research tool uses the SAME helper and original hard source validator.

In one captured R2 nominal at 57.571m, minimum physical clearance improved **6.2086mm -> 27.6958mm** with initial correction scale 0.03125, at **1769.279 -> 1777.584** frozen racing objective (+0.47%). This demonstrates the availability of a better-margined finite-horizon candidate, **not** recursive feasibility or guaranteed Unity gain.

An opt-in **post-QP clearance tie-break** was tried, selected one exact legal plan with ~4.97mm -> 26.96mm slack in 1.536ms, but its strict run `comp_clearance_merit_20261102_10lap_01` stopped at 59.019m after 17 retained-plan publications. **Removed this unqualified tie-break from production code completely**; no alternate fallback code remains.

### New consistent soft reserve merit inside source SQP

Added opt-in `weight_source_corridor_reserve` (default 0) to `MpcRtiConfiguration_t`, ROS parameter declaration, and both normal QP and exact nonlinear merit. Exact cost at every k>0 is `w * max(0, near_wall_tracking_reserve_m - minimum_source_footprint_slack_k)^2` (world-sweep slack when authoritative). The QP adds the active 8-corner footprint hinge's source-linearized PSD Gauss-Newton stage-state Hessian and the identical exact first derivative; terminal Riccati diagonal uses a PSD diagonal majorizer preserving that derivative. This **does not** tighten the hard physical corridor and does not add another rollout or fallback controller. Cost=0 exactly leaves the former objective. The nominal 40mm reserve used by these trials is the already configured soft near-wall target; 20k and 50k are *experimental weights only*.

Added `MpcSqpClearanceMerit.SourceAndQpShareFirstDerivative`: reconstructs the physical source trajectory and independently finite-differences ONLY the additional margin cost, comparing it to the difference in normal QP source-adjoint gradients with reserve weight on vs off. PASSED. Full ROS MPC package: **252 tests, 0 failures**, Release build succeeds, `git diff --check` clean.

Strict competition A/B runs (one warmup plus ten scored laps requested, but **none completed**):

| Configuration | Full laps before stop | Maximum progress | p95 / max end-to-end | Outcome |
|---|---:|---:|---:|---|
| Baseline selective source-gradient, no extra margin cost (`comp_selective_gradient_20261102_10lap_01`) | 1 | 105.621m | 14.70 / 18.99ms | `corridor@0` |
| Consistent physical reserve weight **20,000**, run 1 (`comp_corridor_merit_20k_20261102_10lap_01`) | **2** | **148.419m** | **15.17 / 18.45ms** | `corridor@0` after 25 retained plans and 7 prefixes |
| Same weight **20,000**, repeat (`comp_corridor_merit_20k_20261102_10lap_02`) | **1** | **123.195m** | **14.34 / 19.24ms** | `corridor@0` after 3 retained plans |
| Stronger physical reserve weight **50,000** (`comp_corridor_merit_50k_20261102_10lap_01`) | **0** | **80.321m** | **14.51 / 18.80ms** | `corridor@0` |

All four runs had no reported collision and no measured publication deadline miss. The longer first 20k run is promising but NOT repeatable; the 50k run disproves any assumption that simply increasing a wall-clearance penalty fixes successor feasibility. Qualified 5x10-lap trials on BOTH tracks remain outstanding. All canonical practice and competition YAMLs are unchanged, and `weight_source_corridor_reserve` remains **disabled by default**. Run artifacts and bags are ignored, retained on this machine and not contained in this tracked report.

### Exact replay binary-ABI safety

New model float originally inserted into the raw POD between existing fields: an ABI hazard even when the overall struct size remained 160 bytes. Corrected: appended it into the model's previous final 4-byte alignment padding so all old model offsets stay unchanged, bumped the `.mctx` context version **2 -> 3**, and explicitly migrate old version-1/2 captures with `weight_source_corridor_reserve=0` (formerly undefined padding). The binary reader accepts v1/v2/v3, requires the recorded raw sizes, and keeps strict existing field validation. The `AsyncFailureRecorderPreservesBothPassesAndPhysicalEpoch` regression now tests nonzero v3 weight roundtrip and intentional conversion of that same file to v2, requiring a zero migrated weight and correct unrelated physical state. Old local v2 alpha-zero captured contexts still replay with QP exact parity after the migration. Do not silently reinterpret arbitrary old padding as a new nonzero physical cost.

### Immediate next step (not a weight sweep)

Build a horizon-successor continuation or controlled-invariant *approximation* in the normal source SQP with actual delayed command FIFO/hidden tire state, including physically optimized steering/throttle for the post-horizon states, and validate recursively across successive measured command epochs. A short held-command continuation with two arbitrary actions had zero discrimination in 232 previous live probes and is not enough. Inspect source-to-Unity one-step mismatches near 0–5mm clearance and ensure the publication-age/state synchronization error envelope is included. Keep tested finite-step backtracking and real 20ms publication bound; do not declare a vehicle robust until repeated 10-scored-lap races pass on both tracks.
