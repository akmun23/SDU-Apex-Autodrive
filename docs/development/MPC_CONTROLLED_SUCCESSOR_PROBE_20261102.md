# Controlled successor feasibility — first source-exact implementation (2026-11-02)

## Engineering scope

Added `f1tenth_mpc/src/mpc_successor_viability.cpp` and the private
`mpc_rti_source_successor_probe` interface in `src/mpc_rti_internal.hpp`.
This performs a **strict source-exact**, full-hidden-state, actuator-queue-aware
control search *after* an already-valid 750ms horizon. A hard collision or
corridor constraint is never relaxed. The tested successor duration is 8 or
16 control steps (200/400ms at 40Hz), rather than the earlier 3-step held
command check (75ms).

The bounded source optimizer evaluates nine different steering/throttle control
continuations: three blends towards progress-indexed source-witness steering
and three throttle schedules (hold, blend towards witness, controlled reduction).
The candidate controls obey configured steering/throttle command rate limits
when those are enabled and are reintegrated with the native source plant, all
22 internal hidden channels, the in-flight steering and throttle queue, and
full original body footprint. The output includes the number of source-valid
candidates, progress, best minimum corridor clearance, and physical terminal
heading/lateral speed. This is a **bounded sampled recourse search**, NOT a
continuous globally optimized successor, terminal controlled-invariant set,
or proof that failure of nine tested candidates implies physical impossibility.

`tools/mpc_live_failure_replay CASE_DIRECTORY --successor-probe` compares the
new source successor with the historical held-command 75ms probe, using
`MPC_SUCCESSOR_OFFLINE_STEPS=8` (default) or `=16`. This tool operates only
on existing saved native source/QP failure fixtures and does not publish
controls. Neither canonical race profile is changed.

## First saved near-wall results

From the still-valid R1/R2 nominals in
`live_runs/comp_gradient_nearwall_capture_20261102_01/qp_captures`:

- Five exact-valid starting horizon/branch samples were available.
- Original 75ms held-command terminal check succeeded in **3/5**.
- The 200ms nine-candidate controlled continuation found at least one
  source-valid successor in **2/5**; it evaluated all nine candidates when
  possible (at most ~0.89ms per probe, ~0.39ms median in this sample).
- The 400ms variant also found at least one source-valid successor in **2/5**;
  fully feasible probes took up to ~1.87ms. Neither of the additional 200ms
  of lookahead improved discrimination in this five-sample fixture.
- One 75ms-accepted branch had **zero** feasible *sampled* 200ms/400ms
  continuations. This illustrates that a three-step held-command continuation
  is an inadequate future feasibility indicator; it does NOT prove that all
  physically admissible future control policies fail.

The `MpcSqpSuccessor` unit tests independently reintegrate selected 8-step
and 16-step controls through the original source corridor validator and
compare reported physical progress and minimum clearance. An out-of-corridor
source starting state has no accepted successor.

## Experimental normal SQP tie-break (rejected after actual closed-loop trial)

Briefly integrated as opt-in `prefer_source_successor_refinement` R1/R2
selection, only when BOTH R1 and R2 had full source-valid horizons and a
near-wall/dynamically difficult terminal, with a 1–2% racing-cost cap and
remaining deadline checks. In strict Unity competition trial
`live_runs/comp_successor200_20k_20261102_10lap_01`:

- MPC reached **80.320m**, zero full laps, then `corridor@0` stop.
- 210 cycles attempted terminal comparison; 199 compared pairs were both
  sampled-successor feasible, 10 were both sampled-successor infeasible, and
  only one was asymmetric. The tie-break changed selection 16 times, but did
  not recover full-race reliability.
- Maximum observed end-to-end cycle **19.877ms**, leaving effectively no
  margin inside the nominal 20ms cycle; P95 ~16.96ms.
- Therefore this is NOT a qualifying improvement. The entire opt-in runtime
  tie-break (parameter, selection, and telemetry) was removed, rather than
  leaving an ineffective parallel legacy path in the controller.

## Next implementation — into the optimization, not a fallback

1. Use a controlled source successor to construct a **frozen local terminal
   recourse merit/certificate**: evaluate terminal e_y, e_psi, u, v, yaw,
   hidden slip and in-flight command queue, plus the future track corridor
   through the extended source rollout. Derive its sensitivity with respect
   to the 23-dimensional optimized terminal state via the native source A/B.
2. Add its positive-semidefinite Gauss-Newton terminal contribution to the
   **normal R1/R2 QP**, and score the SAME frozen objective in every exact
   nonlinear rollout. Guard against violating a previously valid horizon by
   an overly restrictive terminal objective or poorly predicted control.
3. Optimize a richer time-varying tail control policy than nine seeded
   schedules, or create a genuinely constrained predecessor/viability set.
   Validate gradient parity, successor feasibility over successive real
   command epochs, and prediction errors against Unity; no 200ms finite
   prefix alone can certify recursive feasibility.
4. Budget the terminal-source work under the **20ms total cycle deadline**
   and benchmark in Unity. Preserve hard collision/physical limits and the
   qualified practice baseline. Only after repeatable 10-lap races should
   repeated prior-plan/prefix fallbacks be retired.
