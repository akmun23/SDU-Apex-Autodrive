# MPC fallback-episode investigation and feasibility restoration: 2026-10-30

## Problem and precise failure mechanism

Native practice witness is 4.8s. Fast MPC candidate `weight_u:45`, `underspeed_curvature_gain_m:0.6`, no speed cap, unchanged lateral/heading tracking weights and wall constraints achieved **49.397s / 10 scored laps, zero publication deadlines** in its second independent run. Its third independent run FAILED with body/corridor failure near lap station 17.560m. Telemetry: at command time e_y=+0.33139m, forward speed 6.422m/s and stage-0 predicted e_y=+0.33465m outside permitted upper bound +0.31663m. Occupied-map body footprint audit yielded map clearance -0.02272m. Actual collision followed ~34ms after MPC fault. Max callback-to-gate was only 19.167ms, zero >20ms misses, so the crash was NOT a publication timing failure.

In the final 1.5s, 28 of 60 published cycles used a *short* candidate/retained-plan recovery prefix, which passes only 25-75ms of exact PhysX-source rollout. The remaining cycles were new accepted full solves. The short-prefix fraction 46.7% is dramatically higher than in 49.397s good run (two total short-prefix actions in all ten scored laps). Under short-prefix replay, lateral error grew from ~0.028m to ~0.331m, while acceleration/positive throttle continued. The ADMM primal residual grew from ~0.03 to ~0.65; input/steering constraints and 150-iteration cap became binding. These prefixes were locally certified; **none proved a 750ms viable successor**. A single accepted full-horizon cycle between short prefixes does not prove repeated replay safe; a receding-horizon hard corridor alone does not imply recursive feasibility.

## Cold start: what was already implemented

Before today's experiment, each short-prefix publish invoked `mpc_rti_memory_reset_solver_state(&rti_memory_)`: ADMM primal/dual variables were ALREADY reset. However, it kept a shifted or composed **physical-control warm start**, frequently seeded from the stale prior plan. Resetting the solver is not enough if its next local linearization point is still being recycled. A complete trajectory/reference re-seed is a distinct intervention.

## Experimental code (opt-in, never promoted)

- `f1tenth_mpc/include/mpc_fallback_episode.hpp`: combined episode for both retained-plan/candidate prefix, 16-action sliding history at nominal 40Hz and explicit consecutive count. A single full solution clears *consecutive*, not recent history. Tracking is constant-time and zero-allocation.
- `fallback_max_short_prefix_actions`: integer ROS parameter, defaults **zero (OFF)** to preserve protected practice baseline. Value `2` permits two individual 25-75ms fallback actions before full-horizon-only recovery. Also limits >2 short-prefix actions within the 16-action history, even if interleaved with occasional accepted full solutions.
- After the first short fallback: invalidate stale nominal and physical-control warm start, cold rebuild from current measured state/native source witness. Preserve physical actuator-delay queue in measured state. ADMM state also reset.
- After the episode exceeds the threshold: try a fully source-certified 750ms plan including 300-500ms tapered steering/torque corrections, with no arbitrary speed cap. Require valid full source rollout, exact physical corridor and stable terminal motion. On success publish it, seed next optimizer and reset episode. On failure, run fresh MPC; disallow any further 25-75ms replay until a full-horizon accepted result exists. **This gate may still latch a controller fault; it does not manufacture a physical safe action from an infeasible state.**
- Search was reduced to <=12 candidates and a 5ms per-callback evaluation allowance (must leave >=15ms until publication gate for normal QP), because an earlier 24-trial search allowed only five ADMM iterations, residual 3.44, before its time budget expired.
- `f1tenth_mpc/test/test_mpc_fallback_episode.cpp`: isolated failure, alternating candidate/prior sources, decaying 16-action count, disabled-mode compatibility. `colcon test` **190 tests, zero failures**.
- Existing related CPU work: nonzero-tolerance tracking loss optional and default disabled (2cm tolerance ran slower, 50.964s), safe redundant-accepted-horizon validation skip and identical prefix comparison dedupe; default behavior is unchanged where possible.

## Real Unity practice results (one warmup plus ten scored unless failure)

| Trial | Total scored [s] | Deadline misses | Collisions/abort | Interpretation |
|---|---:|---:|---|---|
| Fast no episode protection `_02` | **49.397** | 0 | none | one good speed success |
| Same fast no protection `_03` | incomplete | 0 | physical collision, 17.560m | serious repeated prefix problem |
| Bounded fallback, cold source only `_01` | incomplete | 0 | early MPC fault near p=81.475m | hard cap alone stops bad behavior but is NOT a usable recovery |
| Bounded + 24 multiseed + rejection-only physical rescue `_01` | 49.928 | 2 | none | recovery proved, but deadline overrun; build contention overlapped |
| Bounded + 24 multiseed + 16-action history `_02` | incomplete | 0 | MPC fault near p=148.523m | QP only five ADMM iterations, primal residual 3.44 due compute starvation |
| Bounded + 12 multiseed, 15ms solve reserve, 16-action history `_03` | **49.670** | **0** | **none** | first clean within 49.8s, fails stricter 49.5s threshold and needs repeats |
| Same as `_03`, independent `_04` | incomplete | 0 before fault | MPC fault near p=81.488m, stage24 | ADMM 150-iteration limit, primal residual 0.0796, no certified full successor; not repeatable |

Real named run folders use `live_runs/practice_fast_bounded_multiseed_recovery_20261030_XX`, and configs/logs are in `live_runs/mpc_collision_margin_experiment_20261030`.

## Two distinct QP failures that a fallback policy cannot solve alone

1. **Numerical deadline starvation**: after expensive multiseed search, one cycle hit time budget at five ADMM iterations (primal 3.4386, dual 1.1322), rejected at future stage24. Fixed partway by reserving 15ms for the normal MPC solve, but not a proof of WCET.
2. **Local feasibility and iteration cap**: later 15ms-reserved repeat failed near the same problematic corner, 150 iterations, primal residual 0.0796; exact native rollout at stage24 left wall corridor, despite deadline being met. Physically recovering from e_y~0.18m/r~-2.73rad/s at u~3.38m/s requires a different feasible rollout. Cold nominal and bounded tapered controls were not sufficient.

## Engineering next steps (NOT YET IMPLEMENTED OR VALIDATED)

1. **Direct feasibility-restoration SQP**: introduce Phase-I softened corridor/terminal constraints *only inside the QP* with exact source-plant hard acceptance untouched. The new optimization should minimize constraint violation and maximize recovery of safe margin before racing objective, rather than letting hard linearized state bounds make the QP incompatible with its own short-horizon nominal. A hard nonlinear source certificate remains mandatory for publishing a full plan.
2. **Adaptive trust-region/regularization**: when residuals stay high, examine the exact stage-specific Jacobian, steering and torque trust region, `rho_u` saturation at 32, and candidate nonlinear validation. Try targeted widening with line search and physical certificates, not blindly larger error weights. Reproduce failure-state QP offline with unbounded iterations and identical captured causal command queue before choosing an ADMM change.
3. **Recursive safe set or terminal viability backup**: source-validated 750ms rollout is necessary but not sufficient for indefinite safety. Offline characterize the recoverable set under real actuator limits, braking/positive-only torque and map clearance, then require terminal controls to enter that set with a stored backup policy. Avoid claiming any MPC can always find a safe action after the state is physically unrecoverable.
4. **Runtime timing**: account for source-seed/restoration cost BEFORE allocating ADMM work, cap total QP + validation + ROS publication and measure worst case under CPU contention. 20ms deadline must be verified by independent full races. Non-RT Linux has no unconditional WCET guarantee.
5. **Cross-map:** validate unchanged source-speed 7.075s IROS raceline with this controller separately; no practice promotion until 3 independent full 10-scored-lap runs are safe AND under both 49.8s total and 4.95s mean (under49.5s total). A race must also pass IROS safely without artificial speed caps.

## Uncapped IROS cross-map result (13:03 local)

Tested same optional bounded policy with `live_runs/mpc_collision_margin_experiment_20261030/iros_uncapped_bounded_fallback2.yaml`, native source witness `live_runs/optimizer_iros_adaptive_regions_v6_20261030/exports/e75d1f49f552ec98/source_command_raceline_with_witness.csv`, nominal source 7.075s, no artificial cap. **FAILED before first lap** near progress 57.5869m. Current e_y=-0.0719m, e_psi=-0.0628rad, u=2.717m/s, v=-0.0103m/s, r=2.619rad/s; source prediction first violated corridor at stage20, with e_y=-0.2986m and e_psi=-0.1732rad. ADMM 150 iterations, primal residual 0.05366, dual 0.00438, iteration cap, callback time 18.0317ms. After permitted two short-prefix actions, no fully certified reacquisition/normal solve existed; bounded policy faulted. Thus this particular code is **NOT** a cross-map recovery solution. Do not promote to practice, competition, or release defaults.

## Handoff status

**Actual implementation status: experimental and failing repeatability**, not a complete fix. Retain isolated fallback with existing exact physical prefix validation. Current optional `fallback_max_short_prefix_actions=2` removes indefinite replay, cold-reseeds the nominal, and attempts full source-validated steering/torque reacquisition, but it can still run out of physically feasible options and latch a fault. Immediate priority is a dedicated full-state Phase-I feasibility restoration SQP/ADMM strategy with hard source-plant validation. If the real state is outside the recoverable kernel, no optimizer can honestly guarantee safe progress. Baseline practice YAML and uncapped witness remain protected. The optimizer source/dashboard and autonomous optimizer process were not modified.

## Phase-I same-Jacobian feasibility restoration, later 30 Oct evaluation

An experimental feasibility-restoration QP is now implemented in `f1tenth_mpc/src/mpc_rti.c` and declared in `f1tenth_mpc/include/mpc_rti.h`. It reuses the already computed 30-stage physical Jacobians, temporarily relaxes QP-only polygon/box bounds with convex feasibility-biased local costs and wider local control trust regions; **native full-horizon source plant and hard wall clearance remain mandatory** for publishing actions. The `enable_feasibility_restoration` ROS parameter defaults false. After repeated short-prefix actions the controller may prioritize restoration before racing QP. Unit tests include `f1tenth_mpc/test/test_mpc_feasibility_restoration.cpp` and 193 package tests were passing when checked.

Early source-speed uncapped IROS tests (`iros_uncapped_phase1_same_jacobian_20261030_01`, `iros_uncapped_phase1_priority_20261030_02`) **FAILED** before completing a lap near track progress ~57m, predicting collision at stage 22. One failure was 150 ADMM iterations with primal residual 0.0712 under 20ms callback. Thus this QP relaxation alone is not a general cross-map solution.

Actual Unity **practice** test `practice_phase1_restoration_20261030_01`: protected source trajectory with weight_u=45, underspeed_curvature_gain_m=0.6, bounded fallback 2, rejection-only 8-trial full source recovery, QP restoration relaxation .18m and wall weight 8000. **Ten scored laps, 49.859s total / 4.9859s mean, no collisions, 1 publication deadline miss**, 73 accepted_feasible, 1463 accepted_line_search, 693 accepted_optimal, 6 accepted_viability_recovery, and only 3 accepted_prior_plan. This demonstrates a low-fallback full race and successful feasible search, but fails both practice speed and real-time release criteria. Requires independent repeat before inferring robustness. Report: `live_runs/practice_phase1_restoration_20261030_01/mpc_run_report.json`.

**Do not promote**: current work is experimental, separately opt-in, and fails the full requirements. Further work: profile the one deadline overrun; offline reproduce problematic IROS QP; Phase-I slack with provable terminal viability, not merely QP wall relaxation; adaptive solver trust/penalty handling; independent full-ten-lap cross-map validation, no arbitrary speed cap, preserve protected practice configuration.
