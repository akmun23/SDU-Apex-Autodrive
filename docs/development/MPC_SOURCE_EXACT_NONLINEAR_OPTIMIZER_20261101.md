# MPC source-exact nonlinear shooting accuracy audit (2026-11-01)

**EXPERIMENTAL ONLY — not a qualified or selected ROS control policy.**

## Physics and state alignment

Recorded strict practice race: live_runs/phase13_practice_increment_gain025_tenlap_20261101_01/run.
Indexed 25ms/750ms Unity/source comparisons: live_runs/mpc_phase13_prediction_fidelity_20261101/unity_source_outliers.json.

Sixteen overlapping 750ms windows exceed 5cm error at a single strong steering reversal around unwrapped progress 160–161.5m, worst 9.35cm. The source state starts at rack around +0.4rad and newest queue command around -0.4rad. These are not sixteen independent events.

Causal steering-boundary reconstruction: 16 outliers, worst 9.35cm. Suspension jounce reconstruction: 16, worst 9.44cm. Initializing source propagation from unavailable same-epoch Unity motion truth (oracle diagnostic ONLY) reduces to 9 outliers, worst 5.35cm. The error-window starting speed mismatch RMSE is 0.0511m/s, compared with 0.0084m/s for clean path-error-under-2cm windows. Suspect causal starting state estimation and alignment, but the remaining discrepancy is not solved.

## Frozen QP versus physical source

At progress 23.86,39.81,50.86m the current 30-stage QP can mispredict full-direction lateral motion by .153,.448,1.080m versus the exact source. At small command perturbations its local source Jacobian is accurate; at full-size changes yaw, tire-slip and longitudinal/lateral dynamics invalidate the surrogate.

An optional offline 4-pass full-step source-exact relinearized QP (MPC_REPLAY_NONLINEAR_SHOOTING_ITERATIONS=4 in tools/mpc_bag_solver_replay.cpp) did not find any cost improvement over the starting source trajectory across the three cases. Repeated full-step QP solves alone are not the solution. Its deliberately widened test-corridor polygon projections were disabled; this does not apply to the production safety checks.

## Direct-source control optimizer and faster adjoint

A 30-steering/30-throttle direct nonlinear source benchmark in tools/mpc_bag_solver_replay.cpp uses 120 finite-difference full-source rollouts and a handful of full-source candidate evaluations per iteration. It improves exact source cost for all three states and passes the ORIGINAL physical corridor.

New reusable C++ core derivative, f1tenth_mpc/src/mpc_source_adjoint.cpp, API mpc_rti_source_adjoint_gradient in f1tenth_mpc/include/mpc_rti.h, linked into mpc_core by CMake:

- Physical nonlinear nominal at the SAME command epoch for the complete 30-stage horizon.
- Source-derived A/B evaluated exactly at its dynamically simulated stage states.
- Backward full-horizon costate with stage Q/R/N terms; convert signed-command-increment gradient back to 60 absolute steering/throttle gradients.
- No inaccurate full-step QP state prediction used for choosing the source-exact candidate.

The offline replay now delegates to this one C++ core implementation, with no duplicate adjoint solver. One experimental adjoint iteration uses one source Jacobian build plus five full source candidate evaluations, rather than about 125 expensive full source rollouts. Exact candidate merit and original body-footprint corridor remain authoritative. This still uses offline bounded candidate trials, not a real-time production policy.

New test MpcSourceAdjoint.PhysicalSteeringAndThrottleGradientAgreesWithExactPlant checks ten finite-difference gradients from independent exact source rollouts. All 21 MPC CTests pass. At the three initial difficult states, full-gradient relative error with small finite differences was 0.72%, 0.82%, and 7.22%; some subsequent tire/slip regimes exceeded 10%. Therefore exact objective-derivative parity is still a blocker.

## Reproducible physical accuracy corpus

Run from repo root after building target mpc_bag_solver_replay:

    python3 tools/run_mpc_source_accuracy_corpus.py --finite-difference

Outputs: live_runs/mpc_phase13_prediction_fidelity_20261101/source_accuracy_corpus_summary.csv and case logs, using saved states in live_runs/mpc_phase13_solver_consistency_audit_20261101/.

Progress 23.8618m: start cost 310.53, source-adjoint 4-iteration cost 106.66, finite-difference 4-iteration cost 103.27, source-adjoint hard clearance 0.0663m.
Progress 39.8147m: start cost 466.72, source-adjoint cost 192.94, finite-difference cost 192.45, source-adjoint hard clearance 0.0395m.
Progress 50.8604m: start cost 372.90, source-adjoint cost 288.93, finite-difference cost 280.12, source-adjoint hard clearance 0.0890m.

All six final control horizons passed original hard corridor checks. This demonstrates a better optimization direction exists, but it DOES NOT establish exact Unity physics, real-time suitability or global optimization.

## Strict next gates

1. Explain and reduce the nine remaining 5cm Unity/source path outlier windows in the strong steering reversal using only causal state and control timing; confirm odometry u/v/r, hidden wheel/contact and command stamp alignment.
2. Eliminate significant mismatch between source-adjoint objective gradient and true exact-source finite differences in hard corner/slip/progress regimes. Test missing reference/progress effects and nonsmooth source branches. The core must be mathematically right before relying on it.
3. Integrate source-exact iterative optimization into an isolated *experimental* MPC policy, with physically full-horizon predictions and hard collision checks. Validate a large hard-corner corpus, no hidden fallback reliance, and meaningful convergence; avoid accepting tiny steps as success.
4. Only after actual accuracy gates: optimize bounded worst-case computation under 20ms, run complete repeated practice and competition races, then retire truly superseded fallback and legacy line-search implementations. Do not relax physical safety.

Production race weights and fallback paths have NOT been changed in this source accuracy investigation.


## 2026-11-01 source accuracy follow-up

The source-adjoint now differentiates exactly the frozen target schedule
used by the nonlinear candidate scorer. Earlier candidate-progress reference
resampling caused gradients of a different objective. First iteration
full-gradient parity became 0.50%, 0.70%, 1.24% relative error. The four-pass
source-adjoint costs are 103.06, 192.73, 279.96 vs independent 60D
finite-difference costs 103.27, 192.45, 280.12; all hard-clearance valid.
21 MPC tests passed. Later-iteration gradient errors remain.

The severe 160–162m Unity/source rollout mismatch was traced to a real
odometry-versus-Bridge actuator command epoch discrepancy, rather than
an evidenced PhysX tire-force error. Corrected through request-stamped
Bridge-authoritative commands in source odometry and verified in live races.
See docs/development/MPC_BRIDGE_COMMAND_EPOCH_ACCURACY_20261101.md.
