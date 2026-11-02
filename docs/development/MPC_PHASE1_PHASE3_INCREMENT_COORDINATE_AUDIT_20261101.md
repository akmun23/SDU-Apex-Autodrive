# Phase 1 / Phase 3: increment-QP and nonlinear-feasibility audit (2026-11-01)

## Scope, safety and baseline

- Working branch: `mpc/next-engineering-handoff-20261021`. Existing staged and unstaged work preserved, no forced checkout or commit.
- Established practice ten-scored-lap benchmark: approximately 49.162 s. Practice profile and competition production YAML are **not modified by this audit**.
- The original native Unity/PhysX plant, actual vehicle footprint and 50 mm competition clearance are authoritative. The 20 ms publication deadline remains a hard constraint.
- This work repairs previously implemented experimental *increment-QP* support. It does not promote the experimental feature to production, or claim a completed competition lap.

## Verified implementation faults and repairs

1. **Inconsistent QP input coordinates in Riccati feedback rollout.** `mpc_rti_transform_qp_to_command_increments` transforms `u = p + Δu` in cost and dynamics. The feedback rollout's linear reference erroneously propagated *absolute* commands through the transformed `B Δu`; after the transform `A[:,p] += B`, this double-counted the previous command. `initialize_rollout_policy` now converts each projected physical command into the delta relative to the carried linear previous-command state before evaluating the transformed model.
2. **Feasibility-restoration QP bounds changed input meaning.** Restoring absolute steering/throttle bounds to a transformed increment-QP would prohibit negative throttle increments and allow QP steps larger than the normal delta bounds. Restoration now preserves signed input increment bounds and absolute-command state boxes in delta mode. The former widening to the full physical envelope remains confined to direct-input QPs.
3. **Restoration solution decoded incorrectly.** Restoration previously treated `restore_solution.u[k]` as physical absolute controls. It now reconstructs absolute commands sequentially from the command-time x0 and each optimized delta, using the same stage input bounds as the normal QP. Source-plant nonlinear validation still checks the resulting physical sequence.
4. A single pair of inline conversion helpers is shared by feedback propagation and restoration decoding, avoiding more inconsistent ad-hoc formulas.

### Added numerical tests

- Restored delta-input bounds preserve signed throttle and steering changes.
- Eight-stage direct-command versus increment-QP affine-state propagation parity.
- An existing QP versus exact nonlinear finite-difference objective-gradient test now runs in **both direct and increment coordinates**, across straight/curved/source-witness scenarios with 3 stages × 2 commands each.
- Existing local quadratic objective and pseudo-Huber steering-variation gradient parity retained.

Local build and `ctest --test-dir build/f1tenth_mpc --output-on-failure -j 8`: **21/21 test targets passed**. This is numerical validation, not a nonlinear racing guarantee.

## Competition experiments (full source speed, geometry-only IPOPT reference)

| Experiment | Safety mode | Result | Observation |
|---|---|---|---|
| `comp_sim_nonfatal_delta_20261101_01` (before fixes) | Explicit simulation-only continuation | 0 laps, eventually collided | 4,145 cycles, 3,894 α=0 (93.9%), median issued throttle 0.001, median speed error −3.12 m/s |
| `comp_phase13_increment_coords_20261101_01` | Strict rejection stop | 0 laps, stopped at corridor@19 | At failure nominal and full candidate both exact-invalid; ADMM converged in 1 iteration (primal 3.8e−6); callback 12.93 ms |
| `comp_phase13_increment_nonfatal_20261101_01` | Explicit simulation-only continuation | 0 laps, actual collision | 292 cycles; 114 α=0 (39.0%); median published throttle 0.229; median speed error −0.75 m/s; 34 nonlinear rejections |
| `comp_phase13_restoration_strict_20261101_01` | Strict rejection stop; feasibility restoration enabled | 0 laps, corridor@25 | Nominal invalid, four backtracks failed; restoration attempted without achieving a certified horizon; speed 7.12 m/s, callback 13.55 ms |

**Do not compare the 292-cycle diagnostic proportion directly to the 4,145-cycle reference as a controlled closed-loop speed/reliability test.** They covered different driven states and distances. The coordinate corrections improved numerical integrity and yielded useful throttle in the short diagnostic, but no competition lap passed. Simulation-only continuation allowed physically unsafe first-step proposals after MPC corridor rejection and is never a qualification run.

## Remaining root cause: recursive feasibility, not merely ADMM convergence

At the strict failure on the patched increment QP, both the warm-start nominal and the full QP candidate were physically corridor-invalid. The QP converged to tiny residuals, yet four exact nonlinear line-search alternatives did not meet the full-horizon wall corridor. At another strict screening with restoration, 117 aggregate iterations and restoration attempts likewise failed to certify a viable full horizon. Trust-region deletion, improved numerical coordinates, and restoration alone do **not** solve the safety viability problem.

The normal SQP currently backtracks with one scalar α multiplying both steering and throttle. That couples beneficial longitudinal decisions to lateral infeasibility. It also accepts a physically valid α=0 nominal after all unsuccessful trials, which was responsible for prolonged low-throttle behavior in the older diagnostic. Blindly accepting higher-cost or future-wall-invalid commands is not acceptable as a racing solution.

### Engineering sequence to complete

1. **Exact saved-state replay with the same command epoch:** for failed stages 19/25 and recurring early corners, compare x0, actual queued commands, nominal warm-start shift, time-alignment, QP predicted states, and source nonlinear rollout at each 25 ms stage. A 5 ms callback cannot automatically justify shifting a 25 ms model stage. Reuse stored evidence instead of repeated arbitrary weight sweeps.
2. **Local feasibility direction in normal SQP:** add stage-specific soft corridor *search* residuals/feasibility-oriented merit when the nominal is invalid, while retaining hard real footprint limits at candidate acceptance. Validate the recovery direction and its successors through the unchanged nonlinear source model. A one-step feasible prefix alone is not recursive-feasibility proof.
3. **Globalization of combined commands:** investigate decoupled steering/throttle backtracking or a block-coordinate exact-valid merit check when steering correction invalidates an otherwise valuable throttle direction. Every candidate must pass the existing physical validation and command-time deadline. Add separate α diagnostics rather than mislabeling anisotropic steps with a single scalar.
4. **Corner dynamic anticipation:** compare predicted lateral speed, yaw rate, rack state, tire slip and source actuator response to Unity during curvature transitions. Use spatial path curvature/speed/wall geometry from IPOPT, never its prerecorded dynamic state or commands.
5. **Qualification gates:** direct/increment exact objective-gradient and finite-difference regressions, hard model parity, solver residuals, <20 ms end-to-end callbacks, a strict full-speed competition lap, then ten scored laps, then multiple independent repeatability trials. Check practice against the 49.162 s baseline with only ~1–2% permitted regression. Do not promote unsuccessful experiments into the production YAML.

All original raw source telemetry remains in the `live_runs/` folders. This document is a decision record; it is not another implementation branch or obsolete runtime fallback.
