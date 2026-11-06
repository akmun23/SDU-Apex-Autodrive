# MPC recursive-feasibility architecture — controller-only robustness, 2026-10-29

## Objective and non-negotiable boundaries

**Do not slow or replace the offline raceline to mask an MPC deficiency.** Retain the native optimized geometry and speed/witness as the nominal target. Develop a controller that can deviate within physically safe track boundaries, select steering and throttle directly, recover from solver nonconvergence and modeled disturbances, then return to the fast line when feasible. A source-certified minimum-time trajectory is feasible from its nominal initial state but need not be viable from *every* perturbed closed-loop state; no controller can guarantee tracking an arbitrarily unrealizable trajectory.

Preserve all of: production full-vehicle footprint wall constraints (>=5 cm), input and actuator bounds, observed model/source queues and state timestamps, exact nonlinear source-plant validator, <20 ms end-to-end publication gate, collision/stop watcher, no truth-state online policy, and no silent unsafe fallback. The nominal line is never altered by the recovery logic.

## Inspected current implementation

- `f1tenth_mpc/src/mpc_rti.c`: 30x25 ms RTI-SQP, 8-thread source-Jacobian, Riccati/ADMM up to 150 iter/pass, optional second SQP, damping line search (alpha=1/2,...,1/16), exact nonlinear corridor check and retained seed.
- `f1tenth_mpc/src/mpc_controller_node.cpp`: correction of state to command application epoch, full-plan acceptance when exact-valid, and rejection fallback via retained-plan/candidate **75 ms** prefixes with quality/residual gating. On a rejected plan, both prefixes can be invalid and the system latches a fault.
- `f1tenth_mpc/include/mpc_types.h`: **MPC_COMMAND_DELAY_STEPS=0**. Command-time predictor already accounts for external latency. Historical command queues remain in state for physical causal propagation and actuator memory, not two additional optimized delay stages.
- The current max-speed setting is a **target-speed ceiling**, not a hard physical-state speed bound or automatic braking policy. Physical speed can exceed it.

## Exact replay of five recorded failure states

The offline tool `tools/mpc_fault_reachability_probe.cpp` loads the actual 15-column source witness, decodes the last watcher rejection snapshot, reconstructs MPC physical/hidden states, and calls the **production** `mpc_rti_validate_candidate_plan` at alternative steering/throttle inputs. No physics or hard corridor gates are changed. Compilation on host uses the already-built `build/f1tenth_mpc/libmpc_core.a`.

| Trial limit | Track station at failure (diagnostic) | Actual e_y / e_psi | First failed predicted stage | Candidate shortfall | 75 ms constant-action grid | 300 ms / 500 ms valid |
|---|---|---|---:|---:|---:|---:|
| 4.0 m/s | 37.67 m | +0.030 m / -0.044 rad | 2 | -4.0 mm | 0/105 | 0 / 0 |
| 5.0 m/s repeat | 38.50 m | -0.402 m / -0.481 rad | 1 | -20.9 mm | 0/98 | 0 / 0 |
| 5.5 m/s | 31.45 m | +0.330 m / +0.081 rad | 2 | -15.1 mm | **unresolved reproduction discrepancy** (the offline reconstructed snapshot returned invalid before the expected prediction stage); do not treat sampled count as conclusive | unverified |
| 6.0 m/s | 19.86 m | +0.029 m / -0.023 rad | 2 | -3.8 mm | 0/112 | 0 / 0 |
| 7.0 m/s | 38.32 m | -0.636 m / -0.242 rad | 7 | -26.8 mm | **91/91** | **15 / 7** |

The exact-grid search holds each tested control constant and samples roughly 90–120 distinct steering/throttle combinations; this **does not prove reachability or unreachability**. Passing an exact 500 ms open-loop rollout also does not certify viability under future uncertainty or closed-loop performance. Importantly, the 7 m/s state admits multiple full-physics short and medium-horizon solutions despite the rejected SQP; a candidate-independent safety controller is justified. The 4 and 6 m/s snapshots lack a sampled rescue at the fault epoch; anticipatory feasibility is therefore essential.

**Correct indexing:** logged `state[0]` is cross-track error `e_y`; `state[1]` is heading error `e_psi`. Old notes mistakenly cited `state[1]` as lateral error. Source CSV length is re-read by the production trajectory loader; minor printed modulo-station differences versus an approximate 42.9655 m manual length are expected.

## Implementation sequence

### P0 — Establish counterfactual test fixtures first

- Export the recorded **pre-fault** trajectories (at least 0.25–1.0 s before each rejection), complete state and committed command queues, source physics internal state, track station, reference checksum, and cycle configuration. Capture solver QP/ADMM memory using the already-supported offline `MpcRtiQpCapture_t` path where possible.
- The recovery solver must be evaluated against **pre-fault states**, not just the last irrecoverable state. The first rejected tick is too late for a controller whose physical response needs multiple ticks.
- Match the production validator's status/stage for each fixture; investigate the 5.5 m/s snapshot mismatch before accepting any associated counterfactual.

### P1 — Separate performance and feasibility controllers

- **Performance MPC**: unchanged fast speed/geometry witness; pursue minimum tracking/actuation cost with exact hard wall constraints.
- **Feasibility-restoration MPC**: distinct objective that first maximizes safe footprint/corridor slack and the ability to remain viable over a useful horizon; penalizes steering and yaw/speed transients and avoids zero-progress/retreat. Tracking the exact racing line and attaining its speed become secondary objectives *during recovery only*. Use the same source plant and steering/throttle interface.
- Restore feasibility using a dedicated SQP globalization strategy (adaptive trust-region enlargement/recentering, merit/filter globalization and/or elastic **linearization-only** slacks). Never soften final physical collision validation. Allow alternate nominal seeds: previously certified plan, source feedforward witness, current stable steering, and saturated-steer repair candidates. Multi-start is **bounded deterministic feasibility restoration**, not MPPI.
- Start with a simple deterministic candidate-independent **source-validated recovery oracle** to establish that a safe alternative exists and its CPU cost. Do not publish a 75 ms-only escape unless a longer safe successor is also demonstrated. Only after benchmarking should it become a real-time safety filter.

### P2 — Anticipatory recursive-feasibility guard

- Before publishing the primary command, check that predicted next-step states have a backup that remains safe **well past the first 75 ms**. Use a robustly defined terminal safe set or feasible backup certificate; otherwise reject the tempting near-wall action *early*, while steering/speed authority remains.
- Include actuator rate/physical steering, possible control-epoch timing uncertainty, IMU/odom/model residuals in the reachable-set construction. Use stage-dependent uncertainty tightening inside the optimizer, **not** an assertion that a future 5 cm footprint clearance measured exactly is guaranteed under perturbations.
- Prefer a higher-clearance candidate only when its **predicted recoverability** is stronger, not simply because the path-tracking weight is different. Track intervention rate, minimum physical clearance and feasibility horizon.

### P3 — Speed adaptation only as on-demand control authority

- Keep the original race's CSV speed and geometry unchanged. An online speed governor may command less throttle **temporarily** if the current perturbed state cannot make the next turn safely; it should recover to the nominal pace without needing map-specific tuning or a new slow raceline.
- Current controller cannot ask for simulator q=0 full brake under its normal positive throttle envelope. Calibrate/validate any exceptional brake command in the same source model and actuator interface before adding it. Do not fake a speed limit: q<0.001 and hard-brake semantics are discrete and physically significant.

### P4 — Solver computational reliability

- Prioritize the **feasibility residual** over repeated cost-only refinements near hard-wall active sets. The recorded failed passes include ADMM primal residuals ~0.058, 0.077, 0.53, 1.33 and 1.43 versus the configured 0.02 tolerance.
- Measure failed QP KKT/constraint residuals and factorization/ADMM time on saved fixed QP captures. Try adaptive regularization, scaling, penalty continuation, feasibility-polishing and smarter warm/cold start selection offline. Do not add unnecessary O(N^3) runtime or unconditional second passes.
- Keep a strict <20 ms command-publication gate. Time budget includes data synchronization, state propagation, search, exact validator, recovery publication and actuator-history update, not just ADMM.

## Acceptance criteria, in order

1. All five fault snapshots are faithfully replayed and their causes classified as infeasible candidate, insufficient search, or truly no sampled current-state safe control. Capture early points to demonstrate recovery before the fault.
2. Offline recovery search finds legal, source-certified solutions from relevant **pre-fault** states; show improvement in safe successor horizon and steering/throttle commands, not just a 75 ms rescue.
3. Production C++ changes pass existing physical rollout, QP, line search and control epoch tests; the practice-map 4.800 s production witness remains unchanged and is revalidated against baseline.
4. At the **unchanged 7.400 s** competition source reference, run **at least three independent Unity supervised warmup+10 scored-lap trials** with no collision, no fatal MPC rejection, no command publication at/over 20 ms. Only then revalidate the 7.125 s source witness and faster candidates.
5. Report achieved Unity scored-lap time/variance separately from the source witness's nominal duration, plus minimum corridor slack, recovery count, highest solve gate, solver residuals and time of true model-induced faults.

No 'run completed' claim is justified by a single 10-lap pass. If robustness cannot be demonstrated at one optimized line, the controller work is not finished; reducing the offline raceline's speed is not a substitute for this plan.
