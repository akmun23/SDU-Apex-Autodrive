# Nonlinear vehicle-model discovery update — 2026-09-28

## Stop point

Per the user's request, active GPU training was interrupted. No simulator container was running. The interrupted 24-frame-history GRU left its log and three atomically saved member checkpoints in:

`live_runs/derived_dynamics_learning_20260928/nonlinear_model_tournament_gru_history24_20260928/`

The final log line before interruption was member 3, step 12,000 (`train=0.001928`, `val_nrmse=0.18333`). The process was interrupted during the next rollout, so there is no completed training report or evaluation for this run. Its checkpoints are partial artifacts, not a validated model. The log is at `live_runs/derived_dynamics_learning_20260928/nonlinear_model_tournament_gru_history24_20260928.log`.

## What was done

Following the adaptive-model-discovery handoff, this work trained and compared three sequence predictors on the existing open-plane oracle dataset, rather than doing another MPC-weight sweep:

`live_runs/derived_dynamics_learning_20260928/oracle_dataset/openplane_dynamics.npz`

The tournament used whole-run partitions: 55 fit runs, 8 additional validation runs, and 8 experiment-test runs selected from the original training pool. The dataset's original test partition was not used for this comparison; the final-test run was not used. Since some runs do not contain long enough uninterrupted sequences, per-run reports cover only 6/8 validation runs and 7/8 experiment-test runs at the tested history/horizon lengths. Treat those results as conditional on supported windows, not full-coverage generalization.

Compared three-member ensembles with a 16-frame history and 32-step free-running prediction:

| Model | Validation, 750 ms RMSE (u / v / yaw rate) | Experiment-test pooled, 750 ms | Readout |
| --- | --- | --- | --- |
| GRU | 0.339 m/s / 0.267 m/s / 0.444 rad/s | 0.0161 / 0.00212 / 0.1171 | Best overall validation forward/lateral result; large hard-run failures remain. |
| Four-expert recurrent mixture | 0.373 / 0.345 / 0.377 | 0.0188 / 0.00192 / 0.1224 | Better validation yaw, worse u/v; expert gates did not specialize consistently across splits. Not a global improvement. |
| Finite-history NARX | 0.368 / 0.299 / 0.375 | 0.0163 / 0.00202 / 0.1171 | Better validation yaw than GRU, worse u/v; no clear overall winner. |

The easy pooled experiment-test numbers hide difficult individual cases. In particular, the GRU's 750 ms errors on the full-input validation run were about 0.483 m/s forward, 0.613 m/s lateral, and 0.543 rad/s yaw rate. Do not use the pooled experiment-test score as evidence that the model is ready for MPC.

## Best root-cause evidence so far

A diagnostic-only rollout ablation replaced predicted future actuator and rear-encoder channels with their measured future values. On the difficult full-input run, GRU free-running 750 ms errors were `u/v/yaw = 0.483/0.613/0.543`; with oracle actuator-plus-encoder feedback they were `0.479/0.605/0.540`. The tiny change means prediction drift in those channels is not the main source of the body-state error. The main gap is in the recursive body-state transition, latent/slower dynamics, or state representation. This is evidence against spending the next iteration merely improving actuator/encoder prediction.

The mixture model's expert-gate diagnostics showed some regime correlation in training but unstable occupancy/meaning on the experiment fold. That supports regime-dependent dynamics as a hypothesis, but does not validate the current four-expert implementation.

## Code and artifacts

New/modified research tooling is in `tools/vehicle_dynamics_learning/`:

- `train_nssm.py` — GRU, mixture, and NARX training; fixed whole-run split support; optional channel forcing for diagnostics.
- `evaluate_nssm_runs.py` — per-run validation/experiment-fold evaluation.
- `diagnose_rollout_feedback.py` — oracle-channel error attribution; diagnostic only.

Existing observer work in the same directory was left untouched. Detailed metrics and checkpoints are under each `nonlinear_model_tournament_{gru,mixture,narx}_20260928/` directory; per-run reports and the feedback-attribution JSON are stored beside each training report. These are offline research models only: no runtime controller, odometry, localization, or simulation physics was changed, and none of these candidates has been promoted into the MPC.

## Next useful work when resuming

1. Inspect the saved 24-frame checkpoints and log; evaluate them on exactly the same supported windows and per-run breakdown as the 16-frame GRU. Do not describe the partial run as a completed model.
2. Focus the next model experiment on the demonstrated body-transition error (e.g. a physically structured state transition with a learned nonlinear residual or a carefully specified latent-state model), and compare it against the GRU baseline on the same fixed splits. Avoid broad parameter sweeps.
3. Preserve whole-run separation and report per-run/high-steering errors. Do not select or tune from the experiment fold repeatedly; reserve a fresh independent run set before claiming a final model.
4. Only integrate a candidate after it improves the hard validation cases and passes closed-loop sim evaluation. These open-plane prediction results alone say nothing about lap time or safe MPC performance.

## Continuation — IMU roll, throttle transients, and high-steering training (2026-09-28)

### Sensor provenance and slip evidence

The follow-up analysis is in `live_runs/derived_dynamics_learning_20260928/imu_roll_wheel_slip_20260928/report_roll_and_rate.json`; the read-only bag analysis is implemented in `tools/vehicle_dynamics_learning/analyze_imu_roll_wheel_slip.py`. It processed 57 clean, unique bags (fit plus the fixed validation fold); there were no bag errors. The experiment fold and the original test/final-test data were not used.

Across 401,299 receipt-time-aligned IMU/odometry pairs, roll/pitch/yaw from IMU orientation exactly match odometry pose orientation, and all three IMU angular-velocity components exactly match odometry twist angular velocity (maximum absolute differences are zero in the recorded values). Roll is present and varies, but this bridge does not provide an independent roll measurement. A model cannot learn an independent sensor correction from those duplicate channels. This is a simulator-bridge limitation, not evidence that physical roll has no effect on tire loading.

The encoder/kinematic rear-slip proxy shows an adjusted within-run association between preceding 100 ms throttle-command total variation and common rear-wheel slip. At high steering (|steering| >= 0.42 rad), the fitted coefficient is positive in both fit (27 runs: +0.684 m/s per feature SD; run-bootstrap 95% CI +0.029 to +0.897) and validation (4 runs: +1.487; CI +0.066 to +1.606). At the broader |steering| >= 0.30 threshold, the throttle-variation-by-steering interaction is inconclusive: fit CI -0.874 to +0.055; validation CI -0.171 to +0.285. Thus the current data support a tentative association with common-mode slip, but do not show that throttle transients become more damaging as steering increases. It is observational, based on few independent validation runs, and does not prove wheel overload or causation.

The slip quantity is only a proxy: encoder-derived wheel surface speed calibrated on four steady-straight fit runs, minus kinematic rear-wheel ground speed. The current bags contain no per-wheel vertical load, contact force, or tire-force labels. Encoder bursts/outliers are still present (the proxy's tails are large), so do not interpret these values as direct tire slip or use them to fit a force curve. The nominal 59 mm wheel radius and this training-only scale imply about 57.0 mm effective radius, with wide run-level uncertainty; this is not enough calibration support to change a runtime wheel radius.

### Targeted high-steering model evaluation

`train_nssm.py` now has an opt-in, offline-only high-steering sampler. About half of each batch is drawn from training rollouts whose mean absolute measured steering feedback over the future 32-frame target window is at least 0.30 rad; the other half remains run-balanced random sampling. The validation runs/windows remain unchanged. Future measured steering is only used to choose training/evaluation windows; it is not fed as a future feedback value to the rollout model (the rollout uses planned command features). Three GRU members were trained with the throttle-total-variation feature and evaluated on the same fixed validation IDs and windows as the plain and throttle-feature GRUs. The reserved experiment fold was not scored.

| Candidate | 750 ms aggregate validation RMSE (u / v / yaw rate) | Full-input validation run (u / v / yaw) | High-angle boundary run (u / v / yaw) |
| --- | --- | --- | --- |
| Plain GRU | 0.339 / 0.267 / 0.444 | 0.483 / 0.613 / 0.543 | 0.0434 / 0.00256 / 0.2561 |
| Throttle-variation GRU | 0.398 / 0.282 / 0.420 | 0.5105 / 0.5708 / 0.5516 | 0.0449 / 0.00270 / 0.2121 |
| Throttle-variation + high-steering sampling | 0.426 / 0.314 / 0.368 | 0.6672 / 0.5227 / 0.6575 | 0.0476 / 0.00212 / 0.2525 |

Lower is better. On each run's full operating profile, reweighting is mixed: it reduces aggregate yaw error, but worsens aggregate forward/lateral error; it slightly improves yaw/lateral error on the dedicated high-angle run while worsening forward error there, and worsens forward/yaw error on the full-input run.

A follow-up evaluation scores only the same held-out validation windows whose mean absolute measured steering across the target rollout is >=0.30 rad. It covers 517 matched windows from 5 validation runs. Window-weighted 750 ms RMSE is:

| Candidate | High-steering-only validation RMSE (u / v / yaw rate) |
| --- | --- |
| Plain GRU | 0.410 / 0.379 / 0.397 |
| Throttle-variation GRU | 0.386 / 0.373 / 0.395 |
| Throttle-variation + high-steering sampling | 0.309 / 0.356 / 0.342 |

The specialized candidate therefore reduces the pooled high-steering-window error by about 25% / 6% / 14% for u / v / yaw versus the plain GRU. But this is not yet statistical confirmation: the per-run paired improvement directions are 2/5 for u, 3/5 for v, and 4/5 for yaw; run-cluster bootstrap intervals for median relative change all include zero (u +7.0%, CI -24.9% to +8.3%; v -5.9%, CI -52.6% to +96.3%; yaw -14.9%, CI -88.2% to +5.9%). More importantly, this is the same validation fold used for checkpoint early stopping, not an independent test. The candidate is promising for high-steering specialization, but remains unconfirmed and is not promoted.

The filtered-window reports are `per_run_validation_high_steering_0.30.json` beside each model's per-run validation report. Checkpoints and full-run results are under `live_runs/derived_dynamics_learning_20260928/nonlinear_model_tournament_gru_throttle_rate_highsteer_20260928/`; the preceding throttle-only candidate is under the sibling `nonlinear_model_tournament_gru_throttle_rate_20260928/` directory. The earlier 24-frame-history candidate also remained mixed and had less validation-run coverage than the 16-frame baseline.

### Current decision and next useful discriminator

- Do not add roll or angular-x rate as extra IMU inputs to this simulator model: they duplicate odometry orientation/twist exactly. The physical roll/load-transfer hypothesis remains valid but is unidentifiable from an independent roll sensor in these logs.
- Do not claim throttle overload or add a steering-throttle interaction term: the proxy association is tentative, while the cross-validation interaction interval includes zero.
- Do not promote either learned candidate to MPC or odometry; neither has been assessed in closed-loop driving.
- If another simulator intervention is justified, make it a small paired throttle-slew experiment on the open plane (gentle ramp versus abrupt change, matched initial speed and steering profile, both turn directions, replicated runs), then assess encoder-vs-ground-speed proxy and IMU lateral acceleration. This directly tests causality without repeating a broad speed/steering grid. Before that, check whether IMU linear acceleration contains information beyond odometry-derived acceleration; roll and angular rate are already shown to be exact duplicates.
- Next model work should preserve the plain-GRU baseline and fixed validation set. The high-steering sampling result justifies one independent confirmatory evaluation (fresh whole-run open-plane captures, or a still-unused legitimate holdout if one exists) before any promotion; do not reuse the already-consumed experiment fold. Include per-run high-steering metrics and keep ordinary-regime results as regression checks. No simulator physics or runtime controller/localization code has changed.

## Continuation — throttle-slew experiment and fresh whole-run model check (2026-09-28)

### Randomized throttle-slew experiment

The focused `throttle_slew_pair` profile is implemented in `tools/open_plane_excitation.py` and run with `tools/run_open_plane_experiment.sh` against the Explore image in batchmode. Each run tests one matched starting speed (4.5 or 6.5 m/s), steering magnitudes 0.30/0.42 rad, both turn directions, throttle-command changes of both signs, and three repetitions. Each condition pairs a gradual ramp and a rapid step to the same final command; condition order and ramp/step order are randomized. Each capture records 40 Hz odometry, both encoders, throttle/steering feedback and commands, IMU, collisions, timing, and phase events. The run analyzer checks the measured throttle feedback against the intended profiles, not only the published commands.

Ten confirmatory runs (five per speed; pilots excluded) completed all 24 pairs / 48 probe phases. Every run passed the profile/feedback checks, collision and bridge-fault checks, and the 40 Hz receive gate. Sensor topics were about 39.965 Hz and command topics about 40.000 Hz. Per-run details and uncertainty are in `live_runs/throttle_slew_pair_study_20260928/confirmatory_analysis.json`; bags are the `live_runs/openplane_throttle_slew_holdout_{45,65}_rNN/run/run_0.db3` files.

The primary response is the step-minus-ramp change in mean absolute encoder/kinematic rear-wheel residual (post-window minus pre-window median, 0.35–0.65 s). It is explicitly a slip proxy, not tire slip or tire force. Run-cluster bootstrap results:

| Starting speed | Mean step-minus-ramp effect | 95% run-bootstrap CI | Interpretation |
| --- | ---: | ---: | --- |
| 4.5 m/s | +0.0191 m/s | [+0.0089, +0.0275] | Rapid throttle changes increase this residual on average. |
| 6.5 m/s | −0.0105 m/s | [−0.0357, +0.0146] | Mixed and inconclusive. |
| Pooled | +0.0043 m/s | [−0.0127, +0.0192] | No single speed-independent effect. |

The condition-stratified signed residual and IMU lateral-acceleration results vary with speed, turn direction, steering, and throttle-change sign. For example, at 4.5 m/s and 0.30 rad with a throttle-down step, the 0.35–0.65 s lateral-acceleration difference is +0.865 m/s² in left turns and −0.850 m/s² in right turns; the mirrored signs are consistent across the five runs. This supports a throttle-profile-dependent lateral response under matched steering/speed, but does not identify wheel-load transfer or prove tire overload. Keep the residual labeled as a proxy.

### Fresh whole-run model comparison

Five complete fresh runs (`openplane_model_holdout_45_r01`–`r04` and `r07`) were captured at 4.5 m/s with the 86-phase full-surface profile. Each completed all phases with zero quality failures and about 39.56–39.57 Hz command rate. They were not used for training or checkpoint selection: their IDs do not occur in the training reports. The derived holdout dataset is `live_runs/derived_dynamics_learning_20260928/model_fresh_holdout_20260928/openplane_dynamics.npz` (23,750 samples, 420 sequences, five run groups). The full/high-steering scoring reports and comparisons are under `live_runs/throttle_slew_pair_study_20260928/model_eval/` and `live_runs/throttle_slew_pair_study_20260928/*comparison.json`.

The primary comparison is candidate-minus-plain-GRU 750 ms RMSE (30 steps at 40 Hz); negative is better. It uses 256 identical fixed windows per run and paired bootstrap resampling of the five whole-run IDs. High-steering windows have mean future |steering feedback| >=0.30 rad.

| Scope / predicted state | Relative RMSE change | 95% run-bootstrap CI | Result |
| --- | ---: | ---: | --- |
| Full runs — forward speed u | +37.2% | [+20.2%, +49.3%] | Candidate worse on all five runs. |
| Full runs — lateral speed v | +93.3% | [+22.2%, +135.5%] | Candidate worse. |
| Full runs — yaw rate | +127.5% | [+50.0%, +198.3%] | Candidate worse. |
| High steering — forward speed u | +11.7% | [−3.6%, +19.9%] | No confirmed benefit. |
| High steering — lateral speed v | +107.8% | [+22.9%, +152.3%] | Candidate much worse. |
| High steering — yaw rate | +113.1% | [+77.7%, +131.6%] | Candidate much worse. |

The checkpoint comparison was not allowed to decide from only the combined candidate, so the already-saved throttle-feature-only GRU was also scored on these same runs as a diagnostic ablation. Adding the throttle-rate feature alone worsens full-run 750 ms u/v/yaw by +56.7% / +56.4% / +96.5% relative to the plain GRU. Comparing the high-steering-sampled model against that throttle-feature-only model, sampling improves full-run u by 12.8%, but on the high-steering windows worsens u/v/yaw by +97.5% / +15.6% / +90.5%. At one step the same broad regressions are already present, so this is not just long-horizon recursive drift. The evidence points to poor feature/sampling generalization in the current candidate, not a shortage of nonlinear solver iterations.

Decision: preserve both named checkpoints for offline comparison; keep the plain GRU as the current offline baseline. Do not promote either candidate into MPC or odometry, and do not develop the current combined candidate into a regime-conditioned runtime model. The current evidence does not show a high-steering win without unacceptable losses elsewhere.

### Safety/run-isolation finding and next work

The first four complete model captures ran from zero speed. Run `r05` later hit the existing 8° experiment tilt stop at 11° (48/86 phases); its closed bag showed no collision-count change or bridge timing fault. A replacement attempt without restarting Explore began at 21.321 m/s and 36.4° tilt and immediately aborted before probing. This establishes that an aborted excitation can leave a bad/stale simulator state; do not start another experiment in that world. The sim was restarted using the existing `SDU_APEX_SIM_TRACK=explore SDU_APEX_SIM_MODE=batchmode ./tools/start_simulator.sh` path; `r07` then began at 0 m/s and completed normally. Keep both aborted bags as safety diagnostics, exclude them from model scores, and restart Explore after any abort before another capture. Do not relax the tilt limit.

The next model experiment should isolate the causes instead of combining feature and sampling changes: keep the plain-GRU training split fixed; train a high-steering-sampled model without the throttle-variation feature; compare its one-step and 750 ms per-run residuals with the plain GRU; and only then consider a physics-structured nonlinear residual using the measured throttle slew and signed steering/speed regime. The five fresh runs above have now been used for model comparison and must not be reused for checkpoint selection. Before claiming a new winner, collect a new independent whole-run holdout set, ideally restarting the same Explore simulator for each run. All candidates remain offline-only; no MPC, odometry, localization, competition image, or simulator physics was changed.

For reproducibility, model scoring used a workspace-local CPU PyTorch 2.14.0 environment at `live_runs/derived_dynamics_learning_20260928/model_eval_venv/`; no new container image was pulled.
