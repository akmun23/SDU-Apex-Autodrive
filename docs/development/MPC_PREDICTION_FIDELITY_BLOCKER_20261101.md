# MPC prediction fidelity — blocking gate, 2026-11-01

## Focus
Stop work on tuning weights, searching more line-search halvings, removing fallbacks, or qualifying race times until the optimized control direction can be predicted accurately over the full horizon. Production-safe physical collision checks remain. The old qualified baseline remains a historical control.

All corner fixtures below are offline cold-state reconstructions from practice race live_runs/phase13_practice_increment_gain025_tenlap_20261101_01, not exact reproductions of live SQP memory.

## Source/Unity vs QP/source are different accuracy problems
One-step source/Unity 25ms (correct queued slot) yaw-rate RMSE 0.01185rad/s and forward-speed RMSE 0.01359m/s. 2191 measured 750ms rollouts had position endpoint RMSE 0.0133m, but 16 complete paths exceeded 50mm error, worst 0.0935m. Unity parity is good on average, not certified at hardest corners.

QP is locally accurate at its *own* source-derived linearization point, but full-control directional predictions diverge. Same exact source-input sequences were independently rolled out with the authoritative nonlinear source model.

| Progress | Proposed change alpha | Maximum one-stage yaw-rate linearization error | Maximum accumulated 750ms lateral prediction error |
| --- | ---: | ---: | ---: |
| 23.8618m | 0.25 | .02393rad/s | .00806m |
| 23.8618m | 1.00 | .32302rad/s | .15283m |
| 39.8147m | 0.25 | .02342rad/s | .00876m |
| 39.8147m | 1.00 | .56352rad/s | .44797m |
| 50.8604m | 0.25 | .09530rad/s | .14622m |
| 50.8604m | 1.00 | .95334rad/s | 1.07952m |

The small-change stage-yaw defect at 50.86m is 0.000077rad/s at alpha=.005, 0.000319 at alpha=.01, and 0.001320 at alpha=.02, approximately quadratic growth. Steering-rack state defect is near floating-point precision. The initial model's A/B are probably locally consistent, but the nonlinear dynamics of yaw, tire slip and body motions are not adequately approximated for finite corrections.

## Physically exact midpoint source relinearization, same command epoch

| Progress | Linear full-step lateral error | Midpoint-relinearized full-step lateral error | Linear full-step 25ms yaw defect | Midpoint-relinearized 25ms yaw defect |
| --- | ---: | ---: | ---: | ---: |
| 23.8618m | .15283m | .07764m | .32302rad/s | .25452rad/s |
| 39.8147m | .44797m | .16906m | .56352rad/s | .26962rad/s |
| 50.8604m | 1.07952m | .11167m | .95334rad/s | .30150rad/s |

Midpoint relinearization is an offline technique to quantify how much a physically shifted source operating point improves the LTV model. It is exact at the midpoint by construction, but still inaccurate at large changes. It is NOT a completed new optimizer. Full second-order Taylor expansion of the entire trajectory also failed at large steering changes.

An early OFFLINE midpoint diagnostic accidentally called next-epoch warm_start(), which shifts commands by one stage, and consequently produced false nominal discrepancies. Added the explicit same-epoch nominal rebuild function to the MPC API, used by existing production R2 and offline audit, plus an integration regression proving same-epoch nominal states/controls equal independent exact source rollouts. All 21 MPC test targets pass. The old production R2 already used shift=0; this was primarily a reproducibility/interface correction.

Source data files:
- live_runs/mpc_phase13_solver_consistency_audit_20261101/gradient_case_0.log, gradient_case_4.log, gradient_case_7.log
- live_runs/mpc_phase13_prediction_fidelity_20261101/steering_yaw_curvature_benchmark.csv
- tools/mpc_bag_solver_replay.cpp

## Accuracy-first engineering acceptance sequence

1. Verify source-vs-Unity 25ms command epoch, steering, u/v/r, wheel omega/slip, roll, and nonlinear 750ms geometric error specifically for the 16 outlier windows. Unity remains ground truth.
2. Verify source-plant first-order Jacobians vs independent finite differences at corner states and record tire-contact/saturation branch flags, including the physical states that the reduced optimizer omits.
3. Maintain a stagewise-and-full-horizon benchmark for alpha=.005,.01,.02,.04,.125,.25,.5,1, comparing predicted vs actual ey/epsi/u/v/r, physical steering, wheel speeds, progress, and wall-footprint slack.
4. Replace reliance on a frozen linear surrogate by an optimization iteration with nonlinear source-model shooting (or an equivalent DDP/SQP formulation). Build its derivatives around the exact current candidate trajectory, at the SAME physical command epoch. Use the source-model rollout as a fundamental part of optimization; source-exact acceptance only at the END is not sufficient.
5. Test local source-dynamics second derivatives / Hessian-vector products (especially yaw and longitudinal-lateral coupling) as tools to improve each nonlinear iteration. Do not treat a larger cost Hessian, trust region, line search, or fallbacks as the correction.
6. Require predicted-to-actual source trajectory fidelity on the difficult corner corpus before returning to computational budgeting, full-race timing, repeatability and removal of now-unneeded fallback paths. Maintain hard physical collision checks.

An exact nonlinear source model can still disagree with Unity under unusual contact. No single frozen quadratic approximation is globally exact for arbitrarily large changes; the objective is for the nonlinear solver to generate and update physically consistent candidate trajectories and choose actions based on those, not an inaccurate frozen forecast.

## Current status

Completed: 21-test suite, same-epoch rebuild regression, dense 25ms yaw-curvature sweep, physical midpoint relinearization comparison, machine-readable 27-row benchmark.

New experimental work: source-exact 60-control nonlinear optimization and source-derived 30-stage adjoint in MPC C++ core, both offline only. Three hard replay states obtain substantially better exact cost under strict original clearance checks; an independent 10-perturbation gradient GTest and 3-state corpus runner were added. Details in docs/development/MPC_SOURCE_EXACT_NONLINEAR_OPTIMIZER_20261101.md. Outstanding: nine 5cm Unity/source outliers after an oracle initial-motion alignment (not deployable), 7%+ adjoint gradient mismatch at hard tire regimes, reproducible full live SQP memory/command epoch, a trustworthy real-time source-exact optimization policy with a hard 20ms limit, and race qualification. Do not claim these are solved.
