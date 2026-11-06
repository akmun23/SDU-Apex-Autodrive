# Controlled source recourse integrated in normal SQP — 2026-11-02

## Implementation and invariants

The opt-in ROS parameter `successor_recourse_weight` (default **0**) now
augments the **normal R1/R2 QP** and the **same immutable exact-source racing
merit** with a controlled, 8-step (200ms) terminal recourse cost. The existing
750ms physical corridor, actuator limits, command queue, safety-gated
publications, and recovery behaviour are unchanged. No recourse command is
published as a fallback. Neither canonical practice nor competition YAML was
changed.

- `f1tenth_mpc/src/mpc_successor_viability.cpp`: freezes an absolute-control
  continuation chosen by a 3-policy source-validated steering/throttle sample
  (held, intermediate, controlled slow-down), converted to deltas from the
  horizon's final command. The source rollout starts from the complete final
  plant state, including wheel, body, suspension, tire/latent states, and
  delayed actuator queue. If the 3-policy probe has no feasible continuation,
  a held-command policy is still frozen so future corridor violations
  contribute soft SQP merit rather than becoming an automatic shutdown.
- Exact recourse cost: for the SAME frozen policy and source-integrated
  8-stage tail, sum `w * max(0, reserve - physical_left_slack)^2 +
  w * max(0, reserve - physical_right_slack)^2` at each tail state. The tail
  uses the existing conservative 30-step physical-margin convention (the
  original `mpc_rti_compute_corridor_slack` API forbids stage indices >30).
  Candidate future-tail integration failure receives a finite soft penalty;
  it never invalidates a separately source-legal 750ms main-horizon plan.
- QP recourse derivative: differentiate loaded-path-frame 8-step source
  transitions through native full hidden-state `A`, physical actuator `B`,
  and exact progress/reference geometry sensitivities. Differentiate only
  corridor **geometry** numerically. Preserve the exact first derivative and
  add a PSD diagonal Gauss-Newton majorizer to the normal terminal Riccati QP.
  The same **frozen** 8-step policy is reused during R1, R2, source-gradient
  globalization, nonlinear backtracking, and exact candidate acceptance.
- Deadline protection: freeze the experimental policy only when more than
  14.5ms remains before the absolute callback deadline; otherwise skip its
  whole extra objective for that SQP cycle. Telemetry records the initializer
  and terminal-QP derivative time independently. When exact nominal successor
  cost is zero throughout the 8 future stages, skip **all 8** expensive
  native source derivative evaluations; that cost and its gradient are exactly
  zero, so this shortcut does not change the mathematics.
- `f1tenth_mpc/include/mpc_rti.h` and binary `.mctx` format: the frozen
  recourse controls and cost are **appended** to the objective schedule.
  Capture format bumped to v4. Reader migrates v1-v3 objective payloads with
  recourse disabled, retaining old R1/R2 source/QP replay. Roundtrip tests
  check nonzero v4 recourse controls/weight and v2/v3 migration explicitly.

This is a soft, local controlled-recourse **approximation**. The three sampled
post-horizon policies are NOT a controlled-invariant set or proof of global
recursive feasibility. More terminal model sophistication, robust uncertainty
handling, and source-to-Unity synchronization are still required before
removing any fallback or changing the physical collision response.

## Verified numerical behavior

- New `MpcSqpSuccessorRecourse.FrozenPolicyHasMatchingExactSourceQpGradient`
  test perturbs nine independent physical terminal state/progress/command
  coordinates and finite-differences the **same exact 8-step source merit**.
  The added terminal QP first derivative agrees within numerical tolerance.
- New `MpcSqpSuccessorRecourse.FutureNarrowingChangesNormalQpTerminalDirection`
  test narrows ONLY the LEFT wall shortly beyond the original 750ms horizon.
  The positive source recourse cost produces a QP terminal gradient toward
  the opposite side, before the wall enters the ordinary physical horizon.
- All **257 MPC tests pass**, no warnings requiring code rollback, Release
  build succeeds, `git diff --check` clean.

## Strict Unity competition experiments (NOT qualified)

Each experiment retained strict zero nonlinear corridor tolerance, source
informed SQP, existing Phase-I, the 20k near-wall reserve cost, and otherwise
used the earlier immutable competition raceline. These are independent
closed-loop runs, NOT deterministic solver parity checks.

| Run | New recourse | Full laps | Max progress | P95 / observed max e2e | Outcome |
|---|---|---:|---:|---:|---|
| `comp_corridor_merit_20k_20261102_10lap_01` | disabled | 2 | 148.42m | 15.17 / 18.45ms | stage-0 corridor failure |
| `comp_corridor_merit_20k_20261102_10lap_02` | disabled | 1 | 123.20m | 14.34 / 19.24ms | stage-0 corridor failure |
| `comp_terminal_recourse_12k_20261102_10lap_01` | 12k; 9-policy seed, unoptimized derivatives | 1 | 123.29m | 18.45 / 19.90ms | stage-0 corridor failure |
| `comp_terminal_recourse_fast3_20261102_10lap_01` | 12k; **3-policy seed**, skip zero-cost sensitivity | **2** | **166.02m** | 16.04 / **21.39ms** | stage-0 corridor failure, at least one over-budget cycle |
| `comp_terminal_recourse_fast3_4k_20261102_10lap_01` | 4k; 3-policy seed | 0 | not yet exported | — | stage-0 corridor failure |

The fast3 run scored two lap-counter transitions of ~7.89s and ~7.14s; still
far from the 5s/lap ambition. It produced 850 strictly accepted full horizons
and then reverted to five retained-plan actions and two accepted prefixes
before the stage-0 stop. Its strict run report recorded one deferred deadline
recovery and `gate_ms` 20.93ms maximum; diagnostic `end_to_end_control_ms`
reached 21.39ms. This **fails** the target of zero deadline misses. We do NOT
promote the experimental recourse cost or remove any existing fallback.

### Measured runtime progress

| Terminal recourse work per live cycle | First full nine-policy implementation | 3-policy + skip-zero derivative |
|---|---:|---:|
| Initial policy selection, median | ~1114 us | **~400 us** |
| Terminal QP derivative work, median | ~2704 us | **~465 us** |
| Terminal QP derivative work, 95th percentile | ~4705 us | ~3210 us |

Offline captured near-wall states yielded the **same feasible/infeasible
classification** with 3 sampled controls as the complete 9-grid in this
small corpus, but not guaranteed elsewhere. The 200ms source continuation is
more discriminating than the old held-command 75ms probe in some states; its
sampled `no feasible options` finding still does not prove mathematical
infeasibility.

## Next steps (do not replace with more weight tuning)

1. The new terminal direction is now *real SQP mathematics*, but its recourse
   policy is too restricted. Develop a **constrained optimized 8–16-step
   future control policy** or certified local predecessor-set approximation,
   incorporating the actual full hidden state, queue and future footprint.
   Evaluate successor feasibility **across consecutive state/command epochs**,
   not one terminal sample. Preserve QP/source gradient agreement.
2. Address the 21.39ms deadline outlier and the 3.2ms P95 recourse QP cost.
   The worst observed over-budget cycle was labeled
   `deferred_deadline_recovery` at station 99.502m, with the regular QP source
   Jacobian already taking ~5.38ms. The recourse update itself took only
   ~0.74ms there, so reducing *only* recourse weight cannot eliminate timing
   spikes: optimize native 30-stage Jacobians, Riccati iterations and callback
   scheduling with strict cutoff and publication margin.
3. Prevent the accepted-horizon/next-stage corridor-collapse sequence by
   measuring actual successor-state error and latency vs Unity at the corner
   frontier. A source margin of 0–5mm is not robust to the measured 5cm-scale
   longer-horizon source-to-Unity error. Add uncertainty-aware constraints or
   a source-grounded margin that is *state/velocity dependent*, not a uniform
   overly restrictive 50k clearance weight.
4. Qualify both production-like maps only after measured deadline and full
   10-scored-lap success, then five consecutive repetitions per map. No
   fallback removal until normal SQP can achieve those requirements.
