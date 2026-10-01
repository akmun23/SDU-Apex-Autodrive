# Full vehicle-modeling reset — implementation progress

**Started:** 2026-10-01  
**Source of requirements:** [`SDU_APEX_FULL_MODELING_RESET_AND_OFFLINE_SIM_HANDOFF_2026-10-01.md`](</home/akselmo/Downloads/SDU_APEX_FULL_MODELING_RESET_AND_OFFLINE_SIM_HANDOFF_2026-10-01.md>)  
**Scope:** Execute the handoff in order. Keep this ledger current with what was
implemented, run, measured, rejected, or is still outstanding. No experimental
teacher is to enter production odometry/MPC before the handoff's evidence gates
are met.

## Current status — 2026-10-01, user-requested stop

- All active operations were stopped at the user's request. Process inspection
  found no remaining model/training/simulator processes; `docker ps` was empty.
- The 512/64 RSSM run was interrupted during its next optimizer update after
  completing evaluations through step 900. Its best saved checkpoint is step
  700 (validation score 0.07034); there is no final training report. The
  checkpoint/history remain in the workspace and are an incomplete candidate.
- The RSSM closed-loop run was interrupted while loading its dataset, before
  any branch executed. Its output directory is empty. No simulator was started.
- The full straight-hold scoring job completed and its report is recorded
  below. The handoff is not complete; no subagent audit was requested.

## Status at start

- Repository HEAD matches the handoff's reviewed commit, `ee07175d`.
- Applicable workspace `AGENTS.md` files: none found.
- Required background docs were read: rigid-body teacher progress, the prior
  external review handoff, vehicle-dynamics data catalog, and throttle-surface
  results.
- Previous work session's training and simulator were confirmed stopped. This
  implementation has not started a new simulation/training job.
- Existing six-lap maximum from the user's instruction is retained. No
  12-lap replay is an acceptance requirement; score shorter horizons first.
- Historical model weights are preserved as comparator-only artifacts; the
  interrupted mixture checkpoint is not resumed. SHA-256 fingerprints:
  - broad mixed GRU `plant_teacher_mixed_gru_5s_20260930/member_00.pt`:
    `f7eaecd65ea3d1b165989bd2aae64c98e803b2a7d78c501d38f856e33203d753`
  - strongest historical planar XY+heading GRU
    `plant_teacher_planar_gru_60s_xy_heading_20261001/member_00.pt`:
    `531f10486a9294936983eb6b7898b5e4843dd7a230c1a2704046774cca4a7b8e`
  - acceleration-supervised rigid-body GRU
    `plant_teacher_rigidbody_gru_acceleration_20261001/best_teacher.pt`:
    `93c7c96b4f0d7bbcbf1dd588fb6118e338aa2517f7fd8b30d8f8da8b6538c07f`

## Ordered implementation checklist

### Phase 1 — data and label audit (handoff §90)

- [x] Re-export only the four bags with the known exporter exception; audit
      showed no clean eligible sequences, so no rows were recovered.
- [x] Freeze and document schema-6 frame/label semantics using packet fields.
- [x] Produce schema-7/equivalent view with separate simulator physical labels
      and production-observer labels, run family, condition, replicate, reset,
      and failure metadata.
- [x] Verify no sequence crosses a recorded reset or invalid/missing packet.
- [x] Verify exact 25 ms packet-time construction.
- [x] Report independent source-run counts at every required horizon.

### Phase 2 — empirical response atlas (§§37–47, 91)

- [x] Derive the handoff's listed state, input, slip-proxy, acceleration,
      curvature, attitude, and command-history quantities.
- [x] Produce conditioned distributions with sample and independent-run
      counts, not only pooled regression scores.
- [x] Measure matched-state response variance for 0, 100, 250, 500, 1,000,
      and 2,000 ms history; determine whether longer history makes response
      approximately single-valued.
- [x] Estimate conditional variance by regime.
- [x] Test left/right mirror symmetry before deciding on augmentation.
- [x] Analyze all fit-usable throttle-surface trajectories by the specified
      time windows, including wheel/body mismatch, response saturation, and
      steering-conditioned longitudinal authority.
- [x] State the evidence gap for coast, throttle reductions, braking, and
      release; do not create another positive-throttle grid.

### Phase 3 — predictive ceiling and high-capacity latent teacher
 (§§14–20, 83, 90–92, 98 steps 7–8)

- [x] Implement and evaluate the direct future-trajectory predictor: tested
      2 s/5 s, 4 s/5 s, and 2 s/10 s context/future configurations.
- [x] Implement an RSSM/state-space teacher with deterministic memory and
      latent state; train with posterior information and use prior-only rollout.
- [x] Compare history lengths 0.4, 1, 2, and 4 s on identical validation
      future windows; the larger-capacity ladder remains outstanding.
- [ ] Complete the prescribed capacity ladder without blind Cartesian sweeps
      (hidden=512/latent=64, 4 s context is still training).
- [ ] Train on fixed run-level splits; use multiple meaningful seeds only for
      surviving configurations.
- [ ] Report local and recursive horizons, support-distance/divergence, and
      per-run uncertainty. Maximum track evaluation remains six laps.

### Phase 4 — global four-wheel grey-box identification
 (§§21–27, 90–93, 98 step 9)

- [x] Implement the handoff's AWD wheel/tire topology with official geometry,
      bounded effective parameters, latent front wheel speeds, and an optional
      effective tire-relaxation state. Suspension remains represented through
      effective load transfer because per-wheel suspension state is unobserved.
- [x] Fit shared physical parameters across runs with bounded differentiable
      dynamics and run-balanced 0.5 s multiple-shooting windows; both relaxation
      ablations were executed, though neither is a useful teacher.
- [ ] Add optimized/inferred per-sequence latent initial conditions only for
      unobserved internal state; current fit initializes these analytically.
- [ ] Report sensitivity, parameter correlations, run-bootstrap uncertainty,
      and non-identifiable parameter combinations (local sensitivity pass done;
      parameter refit/bootstrap uncertainty remains incomplete).

### Phase 5 — physics plus learned residual teacher (§§28–30, 94)

- [ ] Combine only the best-supported grey-box structure with bounded,
      history-conditioned residual dynamics.
- [ ] Compare on the same fixed splits and horizons; preserve earlier local
      accuracy while increasing the horizon.

### Phase 6 — rollout-distribution correction and uncertainty
 (§§52–59, 64–65)

- [ ] Report training-support distance, first divergence time/channel,
      hidden-state drift, and error against support distance.
- [ ] Evaluate physically plausible state perturbations sized from observed
      one-step residuals.
- [ ] Compare pure simulated-state loss with bounded rollout-state
      augmentation; label synthetic pairs as stabilization data, not physical
      transitions.
- [ ] Use multiple shooting/overlapping windows instead of relying solely on
      very long BPTT; apply the prescribed horizon curriculum.
- [ ] Benchmark actual CPU/GPU batches before enabling larger compute jobs.
- [ ] Calibrate ensemble disagreement against errors before exposing it as
      model confidence/support.
- [x] Implement the shared `reset/step/get_state/get_uncertainty/get_support`
      API for historical GRU and RSSM candidates, including rear-axle pose
      integration and COM-to-rear-axle conversion.

### Phase 7 — closed-loop offline simulator (§§31–33, 65–72, 95)

- [x] Create swappable offline plant/sensor/localization/controller/evaluation
      components and plug in the historical GRU first.
- [x] Generate synthetic rear encoders from wheel rotation and the published
      radius/PPR/conversion; generate IMU and actuator feedback causally.
- [x] Feed sensors through production odometry; model localization latency/
      error from practice data without requiring LiDAR simulation initially.
- [ ] Freeze known historical configuration hashes, lap times, collision
      outcomes, and corner metrics; test whether the surrogate ranks cases.
- [x] Add short branched closed-loop scoring from real states (0.5–2 s) before
      any weight search. Do not require or run a 12-lap replay.

### Phase 8 — sensor-only odometry residual path (§§73–77, 96)

- [ ] Build separately labelled production-odom residual targets and a
      high-capacity legal-sensor observer upper bound.
- [ ] Train a gated longitudinal residual first; keep lateral/yaw odometry
      unchanged initially.
- [ ] Balance run families, penalize needless corrections in normal racing,
      and evaluate separately on normal, wheelspin, braking, and high-steering
      data.
- [ ] No runtime integration without normal-regime non-inferiority and hard-
      regime benefit.

### Phase 9 — empirical planner envelope and MPC use (§§78–80, 97)

- [ ] Derive supported yaw-rate/curvature, useful steering, acceleration,
      braking, wheelspin, and stability envelopes from the data/credible plant.
- [ ] Use the empirical envelope in the minimum-time raceline planner before
      embedding a learned model in the NLP.
- [ ] Screen MPC weights only with the closed-loop branch/surrogate gates and
      run-level/ensemble evidence.

### Phase 10 — required final decisions (§§59, 84–88, 99)

- [ ] Answer Q1–Q5 explicitly using unseen-run evidence.
- [ ] Stop model branches that fail the handoff's material-improvement rule;
      record the measured reason.
- [ ] Freeze new blind confirmation sets after architecture selection.
- [ ] Obtain the requested independent subagent audit only after all required
      implementation, evaluation, and documentation are complete.

## Progress log

### 2026-10-01 — kickoff

- Read the complete 3,302-line external handoff and all four documents named in
  its immediate work order.
- Confirmed HEAD is the reviewed commit and worktree is clean.
- Confirmed the prior Explore container and training process are stopped.
- No implementation, training, data mutation, or new simulator capture has
  been performed yet in this phase.

### 2026-10-01 — corrected data audit and schema-7 view

- Re-ran `_extract()` on only the four bags named by the exporter-fix handoff.
  None contained a clean eligible sequence. The three rootless/speed-hold bags
  had zero aligned usable phases; throttle r03 was aborted and failed stream,
  packet-timing, and phase requirements. No additional data was merged.
- Preserved the schema-6 archive and created
  `live_runs/derived_dynamics_learning_20260928/plant_teacher_mixed_dataset_full3d_reset_safe_20261001/`
  as a separate schema-7 view. It retains 850,576 rows / 7,797 sequences and
  adds explicit run-family, condition, replicate, reset, and failure metadata;
  simulator rigid-state/acceleration labels remain separate from legal-sensor
  and production-odom channels.
- Audited reset-command topic presence across all 116 source-run records. It
  occurs in three bags: r04 (771 reset epochs, whole-run quality reject), r05
  (739, clean), and straight throttle r01 (246, whole-run quality reject).
  Against the clean r05 rows, all 739 reset epochs fall between archived
  sequences; there are zero within-sequence reset crossings and zero internal
  packet-ID discontinuities. The source already had 741 reset-separated r05
  sequences. An initial concern that phase coalescing had crossed resets was
  disproved by this direct sample-time audit; prior model results are not
  invalidated on that basis.
- Updated `prepare_dataset.py` so future coalescing also breaks at recorded
  reset-command rising edges, even if packet IDs are consecutive. Added three
  focused tests for reset boundaries, ordinary continuity, and packet gaps;
  all pass. The schema-7 conversion independently verifies its sequence
  invariants and reports zero violations.
- The archive stores fixed `dt_s=0.025` values. Packet sequence is the source
  of continuity; sample receipt timestamps are used only to associate reset
  events. Schema-7 preserves the source’s whole-run split and 116 run records.
- Exact independent source-run counts capable of containing complete
  trajectories with the simulator pose/rigid-state/acceleration labels fully
  present, from the corrected schema-7 archive:

  | Minimum continuous length | Train runs | Validation runs | Test runs | Final-test runs |
  | --- | ---: | ---: | ---: | ---: |
  | 0.75 s | 65 | 3 | 25 | 2 |
  | 2 s | 23 | 3 | 5 | 2 |
  | 5 s | 8 | 2 | 3 | 1 |
  | 6 s | 7 | 2 | 3 | 1 |
  | 12 s | 6 | 2 | 3 | 1 |
  | 18 s | 5 | 2 | 3 | 1 |
  | 36 s (six-lap cap) | 5 | 2 | 3 | 1 |
  | 60 s | 5 | 2 | 3 | 1 |

- Fixed-run split, packet timing, reset, and exporter gates are now explicit.
  The remaining phases—response atlas, new teacher tracks, grey-box fit,
  hybrid, closed-loop scaffold, odometry residual analysis, and planner use—are
  still outstanding; no experimental model has been integrated into runtime.

### 2026-10-01 — response atlas and identification evidence

- Implemented `response_atlas.py` over the frozen 850,576-row schema-7 view:
  792 conditioned distribution rows and 2D maps include row and source-run
  counts, quantiles, and run-mean quantiles. Output is in
  `live_runs/derived_dynamics_learning_20260928/response_atlas_sourcealigned_final_20261001/`.
- Audited the throttle sweep against the exact same-packet simulator encoder
  angles. In the source-stamp-aligned 725,833-row export, both ROS rear encoder
  angles match simulator debug angles exactly; 724,324 consecutive packet
  increments were checked, and 1,439,594 derived 100 ms wheel-speed values
  agree with the independent telemetry at zero RMSE. The atlas now defaults
  to this source-aligned archive and reports that provenance. This validates
  the measured rear-wheel surface-speed path, not hidden tire force or slip
  angle.
- Actuator profile check on all 1,508 controlled trajectories: throttle
  feedback is within 0.05 of target in 29.2% of the first 0–100 ms window and
  100% of each later window from 100 ms through 8 s; steering feedback is
  within 0.02 rad in 100% of every window.
- For straight, zero-start steps from 0.10 to 1.00 throttle, measured body
  acceleration in the 250–500 ms window is about 5.11 to 4.91 m/s² while
  wheel/body speed mismatch grows from about 0.69 to 23.48 m/s. The 100–250 ms
  body-acceleration response is about 5.19 m/s² across targets 0.10–1.00,
  while mismatch grows from about 1.61 to 23.23 m/s. Each condition has two
  repeats from only two source captures. This is strong evidence of body
  acceleration saturation with increasing wheel rotation in this tested
  initial-state regime, not proof of a universal tire-force law.
- Across matched mirrored conditions, 2,905 positive/negative steering pairs
  from two independent source captures were tested; 1,449 pairs have
  `|steering| >= 0.30 rad`. Relative parity RMSE is 0.2–3.5% across the
  measured even/odd responses, with high-steering parity similarly small.
  Symmetry augmentation is supported as a training experiment, but this
  two-capture evidence is not enough to claim population-level symmetry.
- Refined cross-run matched-state results use run-balanced neighbours, at
  least three other source runs, and run-cluster bootstrap intervals. Adding
  500 ms history reduced local yaw-acceleration dispersion by about 22%
  (ratio 0.78, 95% run-bootstrap interval 0.65–0.83); 250 ms and 1–2 s
  histories also reduce yaw dispersion. Longitudinal acceleration dispersion
  did not materially improve (ratio 0.99, interval 0.98–1.11), while lateral
  acceleration dispersion increased (ratio 1.17, interval 1.02–1.31). In the
  high-steering subset, the 500 ms ratio was 0.73 for yaw acceleration and
  0.57 for longitudinal acceleration, but only 13 query runs support that
  subset. History helps some response channels; it does not make the full
  motion response single-valued.
- Negative-throttle coverage is absent: zero samples in the 850,576-row mixed
  archive have a negative command. It contains 3,402 command-drop edges of at
  least 0.05 across 33 run records, but 2,680 occur at 6–10 m/s and none above
  10 m/s; larger drops are concentrated in fast full-input and practice
  captures. Coasting near zero throttle while above 5 m/s totals 13,217 rows
  from 17 records, mostly above 7 m/s and often while turning. These are not
  independent designed braking/release experiments. The handoff's targeted
  descending/coast/braking campaign remains justified after the first model
  round; no further positive-throttle grid is warranted.
- The first non-recursive sequence-to-trajectory ceiling model is implemented
  in `train_direct_sequence_teacher.py`; a separate whole-run scorer is in
  `score_direct_sequence_teacher.py`. The causal input API accepts observed
  context and future steering/throttle commands only. The frozen schema-6
  training source supplies 7 eligible train, 2 validation, 3 test, and 1
  previously examined final-test run for a 2 s context + 5 s prediction; the
  743 training sequences are sampled run-balanced. The direct h2/f5, h4/f5,
  and h2/f10 variants have now completed. RSSM validation context ladder is
  complete, while capacity escalation and fixed-test scoring remain pending.
  The grey-box fit, hybrid teacher, rollout-support evaluation, closed-loop
  simulator, odometry residual, planner, and final decisions remain incomplete.
- Installed a dedicated persistent environment outside the repository with
  official PyTorch 2.9.1 CUDA 12.6 wheels; the existing simulator/dev images
  remain unchanged. GPU is an RTX 3050 Ti with 4 GB VRAM. On real 2 s context /
  5 s target batches (batch 16), the direct model benchmark measured 37.0
  optimizer steps/s on CUDA versus 1.64 steps/s on one CPU thread; peak CUDA
  allocation was 166 MiB. The full training run achieved 11.13 effective
  updates/s including sampling and validation.
- Trained the direct Transformer (2 s context, 5 s future, width 128, 3 layers,
  4 heads, batch 32) for 3,000 updates on the fixed whole-run train split;
  validation selected step 3,000. The source had 7 eligible train runs,
  2 validation, 3 test, and 1 previously examined final-test run. Validation
  checkpoint-selection score was 0.0481 normalized body-state RMSE averaged
  over 0.25/0.75/2/5 s. On the 3 eligible test runs (96 overlapping windows),
  5 s body-state RMSE was 0.16–0.37 m/s in `u`, 0.018–0.021 m/s in `v`, and
  0.088–0.113 rad/s in yaw rate. Integrated 5 s position RMSE was 0.25–0.35 m
  in x and 0.19–0.22 m in y; heading RMSE was 0.042–0.063 rad. These are
  per-run descriptive metrics; 96 overlapping windows are not 96 independent
  trials.
- Scored this frozen checkpoint once on the three high-speed straight holds
  (now consumed diagnostic holds, not a remaining blind confirmation). At
  5 s, forward-speed RMSE was 0.84/1.44/0.97 m/s for the 0.60/0.80/1.00
  throttle runs. Integrated position RMSE was respectively `[1.82, 13.35]`,
  `[3.97, 5.21]`, and `[4.71, 5.22]` m in x/y; heading RMSE was
  0.446/0.138/0.173 rad. Thus Q1 is provisionally **yes for the held-out
  practice distribution, but not across the fresh high-speed straight
  regime**. The direct predictor is a non-recursive ceiling, not a plant;
  these large straight-hold errors prevent treating it as an offline lap
  simulator.
- Artifacts for direct candidate checkpoints, validation histories, and
  fixed-test scores are under
  `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/`.
  The 4 s-context candidate was selected on train/validation only. The old
  high-speed holds remain consumed diagnostics and have not been used for
  checkpoint selection.

### 2026-10-01 — direct predictor context/horizon ladder and shared plant API

- Completed the 4 s-context/5 s-future direct model (width 128, 3 layers,
  3,000 updates), selected step 2,800 on validation only. Its macro normalized
  body-state RMSE on the three fixed test runs was 0.0530 at 2 s and 0.0514
  at 5 s, versus 0.0357 and 0.0497 for the 2 s-context/5 s model. The longer
  context was 48.5% worse at 2 s and 3.3% worse at 5 s on the fixed test; it
  does not meet the handoff's material-improvement rule and is retained only
  as a rejected comparator.
- Trained the 2 s-context/10 s-future direct model (width 128, 3 layers,
  1,500 updates) with six train and two validation practice runs having
  sufficient continuous duration. On the three fixed test runs, macro
  normalized body-state RMSE was 0.0537 at 2 s, 0.0561 at 5 s, and 0.0490 at
  10 s. At 10 s, per-run integrated x/y position RMSE was 0.71–0.92 m /
  0.52–0.57 m and heading RMSE 0.18–0.25 rad. Extending the prediction
  horizon did not fix short-horizon response error or establish a usable
  free-running plant.
- Implemented `offline_plant.py` as the common causal stepping interface. It
  contains adapters for the frozen historical GRU and the RSSM prior-mean
  rollout, a train-only nearest-state support index, and ensemble-spread
  output, and integrates world pose. The RSSM output is converted from COM to
  the rear-axle convention shared with odometry. RSSM latent prior variance is
  deliberately not labeled calibrated state uncertainty. Tests verify
  reset/step behavior, command-only future input, state dimensions, and
  deterministic prior-mean rollout.
- Implemented `score_rssm_teacher.py` for frozen whole-run test scoring.
  Initial shape, finite-output, and causal-prior tests pass. The first RSSM
  candidate (0.4 s context, 2 s rollout, hidden 128, latent 16) trained 1,200
  updates; best validation score is 0.2351, with the worst of three validation
  runs at 0.6174 normalized body-state RMSE at 2 s. Raw context-specific
  validation scores improved to 0.0991 at 1 s, 0.0881 at 2 s, and 0.0577 at
  4 s. Because those context runs differ in eligible windows and normalizers,
  the four frozen checkpoints were rescored on identical future starts across
  the same two validation runs, using one shared body-state scale. The paired
  macro scores were 0.1108 (0.4 s), 0.1028 (1 s), 0.0933 (2 s), and 0.0602
  (4 s): a 45.7% reduction from 0.4 to 4 s. This is a validation result, not
  test-set or closed-loop evidence. The context-4/hidden-128/latent-16 variant
  is the sole capacity-rung survivor so far.
- Implemented a differentiable four-wheel grey-box candidate in
  `four_wheel_greybox.py`. It uses official mass/geometry, Ackermann wheel
  angles, longitudinal slip landmarks (0.15/0.72 to 0.25/0.464), lateral slip
  landmarks (0.01/1.0 to 0.10/0.5), shared bounded effective
  tire/drivetrain/drag/load-transfer parameters, and four wheel-rotation
  states. Rear-wheel rotation is supervised; front-wheel speed remains
  latent. Its first draft conditioned on future measured actuator feedback;
  before fitting, that was corrected to a command-driven plant with measured
  feedback only at initialization/for scoring, a one-packet delay, and shared
  fitted actuator lag/rate. The 25 ms command/feedback lag was observed in the
  capture. The structure includes optional learned longitudinal/lateral tire
  relaxation times and uses 0.5 s independent multiple-shooting windows, with
  2 s free rollout for validation.
- Real-data gradient probes exposed an important numerical result: without
  tire relaxation, gradients are finite at 0.5 s but non-finite by 2 s. With
  relaxation, the 2 s gradients remain elementwise finite but their float32
  global norm overflows. On the same 0.5 s batch, relaxation reduced the
  gradient norm from 1.64e9 to 1.60e4. This motivates testing the handoff's
  relaxation ablation; it does not establish that the model predicts the car.
  A 600-update no-relaxation validation pilot is running at
  `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/greybox_no_relax_h2_shoot0p5/`.
  It uses no test labels for selection; the relaxed pilot follows.
- The paired direct-vs-frozen-GRU test comparison used the same 32 future
  windows in each of three eligible test runs. At 5 s, the independent-run
  bootstrap interval for `GRU error - direct error` is [-0.0274,-0.0148],
  favoring the GRU. The direct model is not a ceiling over that frozen GRU on
  this test split; this updates the earlier provisional conclusion that it
  might explain high-speed generalization. Those test runs are now consumed.
- `experiment_artifacts.py` now writes config, dataset/archive-manifest hashes,
  split hash, git/source hashes, seed, training report, and model report for
  new jobs. The first four RSSM ladder jobs predate the helper; their
  provenance still needs to be backfilled from the recorded run commands.
- No simulator run has been launched in this phase yet. The old straight
  holds are consumed diagnostics; the offline plant/sensor/localization/
  production-odom/MPC harness, hybrid teacher, odometry residual, stability
  and planner screens, fresh blind confirmation, and final Q1–Q5 decisions
  remain outstanding.

### 2026-10-01 — production components, grey-box ablation, and first closed-loop branches

- The no-relaxation grey-box pilot selected step 500 (600 maximum) with a
  validation macro normalized body-state score of 1.0168. Horizon scores were
  0.8595/1.1379/1.0913/0.9783 at 0.25/0.75/1/2 s. The relaxation pilot
  selected step 100 with score 0.9469 and corresponding scores
  0.6890/1.0504/1.0644/0.9836. Relaxation improved 0.25 s by about 19.8%
  but did not improve 2 s (about 0.5% worse); overall change was only about
  6.9%. Both remain rejected as plant teachers. They did not use test labels
  for selection.
- The differentiated four-wheel model fits shared bounded parameters across
  train-run-sampled 0.5 s shooting windows. It does not yet optimize a distinct
  latent initialization for each sequence. The fitted parameters stayed near
  their initialization and the validation errors remained large; the upcoming
  sensitivity/rank analysis is needed before assigning physical meaning to
  any parameter value.
- Built the exact production odometry C++ observer/packet assembler shim and
  replay interface. Production C++ sources compile into the persistent
  workspace library; YAML/config, synchronized packet input, 25 ms update,
  reset, and output finiteness checks pass. Rear-encoder quantization and IMU
  coordinate/COM-offset tests also pass.
- Fit an empirical localization latency/error bank from five train-split
  practice captures only: 14,723 matched samples, median publication delay
  7.5–9.9 ms by run, relative-pose RMSE 7.9–8.9 cm per run and p95 15.2–18.3
  cm. This is a small training-only calibration, not a held-out localization
  generalization claim.
- Corrected the empirical localization surrogate after code review: it now
  anchors the map pose to the branch's measured start, aligns its residual
  trace to that anchor, chooses only origins with enough remaining horizon,
  applies residual at delayed source time, and raises on out-of-trace access
  instead of silently returning zero. Three focused tests pass.
- Built and exercised the exact production MPC RTI C core/config/raceline and
  the exact production speed-controller Python implementation. The MPC API
  requires 12 state channels; an initial 11-channel branch attempt failed
  before producing data. Source inspection of the production ROS node resolved
  the current steering command versus two delayed-command semantics. The
  corrected branch runner passes four test groups covering AMCL, sensors,
  production odometry, MPC, and actuator interfaces (9 tests total).
- The offline branch harness runs the frozen historical mixed GRU through
  causal synthetic encoders/IMU, production odometry, the empirical
  localization surrogate, production MPC, and production actuator. The
  controller receives no future simulator truth. Branch starts come from two
  held-out practice runs (`practice_mpc_12lap_20260925_e/f`), each with a
  straight and turning case; four 2 s branches completed and a repeated run
  produced exactly identical numeric traces.
- In the `e` straight branch the production MPC had 36 optimal steps, 10
  residual rejections, and 34 nonlinear corridor roll-out rejections, with
  three best-effort actions. The other three branches had 80/80 optimal
  cycles and no nonlinear rollout failure. Truth-referenced path lateral RMSE
  was 0.312 m for that rejected straight branch and 0.036–0.129 m for the
  other cases. Estimated production-odom longitudinal RMSE was 0.090–0.115
  m/s and lateral RMSE 0.010–0.015 m/s. Yaw-rate RMSE is zero by construction
  in this harness (gyro is passed through) and is not an independent accuracy
  result. Localization error averaged 5.7–11.3 cm. The GRU support distance
  reached 0.58 in normalized feature space, so these results are diagnostics,
  not evidence of a reliable simulated lap or safe controller ranking.
- The current MPC rejection count was mapped to the production enum rather
  than treating any nonzero failure-stage field as an error; `-1` means no
  nonlinear failure. A progress wrap across the start/finish arc was also
  corrected by unwrapping against production raceline length.
- No real simulator was launched and no collision claim is possible: the
  first branch scaffold has no wall/collision geometry. A zero-graphics mode
  was not used. The full-lap offline simulator, historical configuration
  ranking, hybrid plant, odometry residual, planner envelope integration, and
  fresh blind simulator confirmation remain open.

### 2026-10-01 — capacity rung, high-speed diagnosis, and sensitivity audit

- The context-4 s / rollout-2 s RSSM capacity rung (hidden 512, latent 64,
  batch 16, CUDA, 1,600-update maximum, early stopping) is running in the
  persistent workspace environment. At step 700, validation macro score was
  0.0703 versus 0.1506 at initialization. This is not yet a completed result;
  promotion depends on its selected checkpoint and matched-run scores.
- Completed paired capacity comparisons on the same three practice test runs
  (32 windows/run; these test captures were already consumed by earlier
  experiments): hidden 256/latent 32 versus 128/16 reduced standardized state
  RMSE by 25.1% at 0.25 s, 12.4% at 0.75 s, 22.1% at 1 s, and 20.3% at 2 s.
  Run-bootstrap intervals exclude zero at 0.25, 1, and 2 s, but cross zero at
  0.75 s. This supports a capacity effect on these consumed practice runs,
  not fresh generalization.
- On the three consumed 10 s high-speed straight holds (0.60/0.80/1.00
  throttle), 256/32 improved 128/16 by 54.9%, 35.9%, and 22.9% at
  0.25/0.75/1 s but was 29.3% worse at 2 s. All three runs regressed at 2 s
  (run-bootstrap interval for the error increase: 0.042–0.732 normalized
  units). The larger model therefore has a clear horizon-dependent tradeoff;
  it is not established as a stable plant.
- Added `score_rssm_straight_holds.py` to evaluate a checkpoint with the
  initial 4 s as context and up to 5.975 s of command-only prior rollout on
  each fixed 10 s hold. It reports body/wheel and integrated pose errors at
  0.25/0.75/1/2/4/5 s. These are already-consumed diagnostics, not blind
  confirmation. The first run is pending the capacity job finishing; scorer
  syntax and archive structure have been checked.
- The grey-box local sensitivity analysis uses 22 training runs and
  run-block bootstrap. Without tire-relaxation states its effective Jacobian
  rank is 17/20; with relaxation it is 19/20, with no pairwise sensitivity
  correlation above 0.95. However, the smallest singular-value ratio is zero
  (or 9.5e-18), and one or more sensitivities are identically zero. Specifically
  tire-relaxation time constants are inactive when relaxation is disabled;
  throttle-rate limit is insensitive in both fits. Idle-brake torque and
  motor torque scale lie at their parameter bounds. Thus this is not evidence
  that the physical parameters have been identified: it is a local predictive
  Jacobian audit, not refit-based parameter confidence intervals. Simplifying
  inactive parameter groups, latent-initial fitting, and refit/profile
  uncertainty remain required before interpreting the parameters.
- The response-atlas archive records 850,576 samples across 116 run IDs, but
  only one independent source run contributes samples above 10 m/s. The
  separately captured three straight-hold runs add high-speed full-throttle
  evidence but do not cover high-speed turning/braking. The empirical
  high-speed steering envelope remains unsupported.
- In the cluster-balanced matched-state analysis, increasing history does
  not monotonically collapse response dispersion. Across comparable matching
  radii, median cross-run standard deviation changes only modestly for `ax`
  and `ay`; yaw-acceleration dispersion improves more with 2 s history, but
  remains substantial. This supports hidden/history dependence, but it does
  not establish that the existing measured history is sufficient to make the
  observed state Markovian. Sparse support above 10 m/s is a separate issue.
- Completed a full available-horizon, prior-only RSSM rollout from the first
  4 s of each 10 s straight hold (about 5.975 s forecast; 3 independent runs).
  The 128/16 model had macro-run `u` RMSE 0.79 m/s at 0.25 s, 0.80 at 0.75 s,
  2.32 at 2 s, and 5.57 at 5 s; integrated XY RMSE reached 27.26 m by 5 s.
  The 256/32 model improved the initial 0.25 s (`u` RMSE 0.37 m/s) but was
  worse by 0.75 s (`u` 1.33 m/s) and 2 s (`u` 4.02 m/s), reaching 8.35 m/s
  `u` RMSE at 5 s and 21.83 m integrated XY RMSE. Both also had large yaw-rate
  error (2.22/0.43 rad/s at 0.75 s for 128/16 vs 256/32, respectively).
  The one-pass high-speed benchmark therefore fails the handoff's useful
  straight-hold target by a wide margin; neither RSSM is an offline plant.
  Report: `rssm_capacity_longstraight_consumed_20261001.json`.
- Added the `--plant-checkpoint` route to the short closed-loop evaluator and
  the RSSM shared-API context-length property, so it can evaluate prior-only
  RSSM with the same production sensors/odom/localization/MPC/actuator path.
  Syntax compilation passed. An attempted 0.5 s RSSM branch was stopped while
  loading the dataset at the user's request, before any branch executed; no
  outcome is claimed.
- User then explicitly requested shutdown. No more training or simulator work
  was performed after that instruction. The independent audit is deferred
  because implementation/evaluation is incomplete and the user required it
  only after completion.
