# Yaw residual MPC implementation — 2026-10-07

## Bottom line

There is still **no demonstrated lap-time improvement** and no completed new
raceline. The controlled P0 reference remains 20 scored laps, best 5.7378 s,
mean 5.7904 s. An earlier report separately cites a 5.7018 s reference best;
no run in this work has beaten it, and sub-5 s has not been achieved. The
2026-10-07 throttle-overlay attempt stopped after five scored laps and two
collisions; its best 5.7658 s is neither a valid full-run result nor progress.

The existing high-speed/high-steering yaw-residual model had measurable
held-out open-plane prediction gains, but it was stranded in Python. This
change ports that exact fitted residual into the production C MPC transition,
its analytic Jacobian, and the exact-discrete offline speed optimizer. The
residual remains **opt-in**; the frozen default MPC and P0 trajectory are
unchanged.

## Implemented

- Added an optional 13-feature yaw-acceleration residual and support gate to
  `f1tenth_mpc/src/vehicle_model.c`. Inputs are only the current MPC state,
  current command, and reference curvature. No future measurement or simulator
  truth is consumed.
- Added the matching analytic derivative path, including the fitted feature
  interactions, correction clipping, and smooth target-speed/tracking/steering
  gate.
- Added parameter loading and validation in the ROS MPC node. The experimental
  profile is
  `f1tenth_mpc/config/mpc_practice_yaw_residual_support_gated.yaml`; the
  production `mpc_competition.yaml` keeps this model disabled.
- Extended the C/Python/CasADi transition verifier and exact 25 ms discrete
  speed-profile optimizer to load the same fitted model. This allows a future
  optimizer candidate to use the same discrete yaw transition as runtime MPC.

The coefficient source is the already-held-out-tested artifact
`live_runs/racing_model_diagnostics_20261006/yaw_residual_support_gated_v3/yaw_residual_candidate.json`.
It is trained on P0 r01 plus open-plane training captures and scored on
independent open-plane captures, including 7.5 m/s high-steering data. Its gate
is zero in the trusted P0 practice support, so it does not alter the baseline
there.

## Verification and outcome

- `f1tenth_mpc` built successfully with the ROS 2 Jazzy toolchain. The targeted
  build is preserved under
  `live_runs/racing_model_diagnostics_20261007/yaw_residual_mpc_integration/`.
- On 1,000 seeded model states, C, Python, and CasADi agree with the residual
  active: maximum state difference `4.77e-5` (entirely yaw rate), maximum
  distance difference `5.04e-8 m`, maximum longitudinal-acceleration
  difference `1.38e-5 m/s²`; maximum C/CasADi Jacobian difference on smooth
  cases was `0.00152`. Results:
  `live_runs/racing_model_diagnostics_20261007/yaw_residual_mpc_integration/c_python_casadi_parity.json`.
- Existing held-out open-plane results for the same fitted model: at 7.5 m/s
  high steering and 750 ms, yaw-rate RMSE fell `5.293 → 1.764 rad/s`, and
  position RMSE fell `5.827 → 3.239 m`. This is a useful relative gain but
  still far too much absolute error to call it an accurate lap simulator.
- One exact-discrete speed optimization was attempted on the frozen P0
  geometry with the learned correction and existing optimizer constraints.
  IPOPT did not converge at 650 iterations (primal infeasibility `0.0538`,
  dual `224,882`); therefore it produced **no raceline**, passed no recursive
  replay, and is rejected. No iteration-only retry was made. Details are in
  `live_runs/raceline_candidates/p0_yaw_residual_discrete_speed_20261007/`.
  Its rejected iterate ranged up to 8.17 m/s and 0.46 rad steering, but had
  zero stages inside the candidate's joint speed/steering/tracking support
  gate. Thus this fixed-geometry speed solve never used the new correction;
  changing speed alone on P0 does not create the required high-speed,
  high-steering combinations.

No simulator was launched in this implementation step. The fit remains a
development candidate, not an approved runtime behavior change or a lap-time
claim. The failed solve shows the learned yaw residual alone does not make the
current speed optimizer converge. The remaining optimizer constraints also
include an envelope that previously failed practice transfer; it must not be
used to justify a faster raceline. The next implementation work should first
change path geometry as well as speed using the learned region, and replace the
failed envelope constraint with measured, practice-transferable support. Then
the exact-discrete optimizer must converge and pass recursive production-C
replay before one resulting trajectory is validated in the established
batch-mode simulator with immediate collision abort.
