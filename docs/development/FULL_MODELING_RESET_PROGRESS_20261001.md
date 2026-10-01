# Full vehicle-modeling reset — implementation progress

**Started:** 2026-10-01  
**Source of requirements:** [`SDU_APEX_FULL_MODELING_RESET_AND_OFFLINE_SIM_HANDOFF_2026-10-01.md`](</home/akselmo/Downloads/SDU_APEX_FULL_MODELING_RESET_AND_OFFLINE_SIM_HANDOFF_2026-10-01.md>)  
**Scope:** Execute the handoff in order. Keep this ledger current with what was
implemented, run, measured, rejected, or is still outstanding. No experimental
teacher is to enter production odometry/MPC before the handoff's evidence gates
are met.

## Current status — resumed 2026-10-01

- Work has resumed under the original ordered handoff. The user clarified the
  decision domain: raceline traversal from 0–12 m/s with adequate steering and
  longitudinal coverage; extreme-speed evaluation is not a success criterion.
- No model training, simulator, or Docker container was active at resume. The
  available filesystem reported 192 GB free. Repository-level `AGENTS.md`
  search found none.
- The previous stop checkpoint remains below as a historical event. Its
  interrupted model results are retained; they are not treated as a completed
  capacity study.
- Coverage audit found 839,155/850,576 rows in the 0–12 m/s decision domain.
  Only 3,940 rows at 10–12 m/s come from one open-plane training run; practice
  data has no 8–12 m/s samples and very little measured steering at 6–8 m/s.
- Schema-7 loading now validates run/family/condition/reset alignment. The
  existing modeling environment is
  `/home/akselmo/.cache/sdu-apex-modeling-20261001` (PyTorch 2.9.1+cu126,
  GeForce GTX 1080); no package installation was needed.
- Family → run → condition → window sampling is implemented in direct, RSSM,
  NSSM, grey-box, and sensor-observer trainers. For the 4 s context / 2 s
  rollout subset there are five eligible runs per family; 10,000 draws measured
  50.13% open-plane / 49.87% practice before the newer race-family weighting.
- A newer race-domain adjustment handoff was received and read in full. It
  supersedes broad-domain model-selection priorities: 0–9 m/s is core,
  9–12 m/s is the fast boundary, and >12 m/s is OOD/support diagnostics only.
- Its complete 2,081-line revision was read on this resumption. It adds
  race-weighted local/closed-loop scoring, feasible combined-demand coverage,
  race-domain greybox-plus-latent-residual work, and fresh blind captures; these
  are now the active ordered priorities.
- The first family-balanced RSSM attempt was interrupted at optimizer step
  1000 after the adjustment arrived. Best validation was step 900 (0.07747),
  but training used the broad schema-7 view and it is comparator-only, not an
  accepted race-domain model.
- Two immutable schema-8 race views and a separate OOD archive now exist under
  `live_runs/derived_dynamics_learning_20260928/`. Strict existing
  quality/timing gates leave 85 source runs, but only one eligible open-plane
  validation run and no clean practice validation/final-test run. New clean
  race captures are required before blind race-domain confirmation; quality
  gates will not be relaxed to manufacture a holdout.
- The packet-continuity/reset-aware coalesced rebuild is complete: 580,420
  source rows, 3,752 sequences, and 11 runs. The derived 2 s cooldown race view
  has 568,999 rows: 446,131 train, 40,947 validation, 40,958 test, and 40,963
  final-test rows. Validation sequences top out at 2.6 s; test/final-test at
  5.0 s. No valid validation window supports 2 s context + 2 s rollout, and no
  held-out window supports 2 s context + 5 s rollout.
- Race-only coverage/response reports have been rebuilt from this coalesced
  view. All 9–12 m/s samples still come from one training throttle sweep;
  held-out splits have no data above 9 m/s. Matched-state analysis has no
  matched 9–12 m/s queries.
- A continuous 260 s race-domain Explore capture profile is implemented and
  its command-plan tests pass. The first 75 s pilot was stopped because Unity's
  default `-logFile -` flooded the terminal; bag audit found 12.4 MiB, 3,015
  samples per main sensor stream over 75.43 s, zero bridge timing-fault
  messages, and 40.03 Hz command output. It is marked interrupted/aborted and
  excluded from model selection. The same pinned simulator now runs in the
  same batchmode path with Unity logging redirected to `/dev/null`.
- Active operation: `openplane_race_domain_train_20261001_r02`, seed 17002,
  is collecting one uninterrupted 260 s sequence. It is training data, not a
  blind holdout. No model training or simulator physics change has been made.

## Prior stop checkpoint — 2026-10-01

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

## Updated execution focus

- Continue from the first incomplete handoff phase. Use schema-7 run/condition/
  reset metadata and preserve the fixed run-level split.
- Report model quality and coverage over 0–12 m/s, stratified by speed,
  steering direction/magnitude, longitudinal command, and independent run.
- Do not spend further compute on the >12 m/s straight-hold objective. Keep
  those already-completed scores only as out-of-domain diagnostics.
- Continue to defer the independent subagent audit until every required phase
  and evidence gate is complete.

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
      (hidden=512/latent=64, 4 s context was interrupted; re-run only after
      completing the domain-coverage audit and sampler/data-view corrections).
- [x] Correct sampling to family → run → condition → eligible window and log
      its effective family probabilities without changing fixed run splits.
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

## Race-domain adjustment handoff checklist — 2026-10-01

The adjustment in
[`SDU_APEX_RACE_DOMAIN_MODELING_ADJUSTMENT_HANDOFF_2026-10-01.md`](</home/akselmo/Downloads/SDU_APEX_RACE_DOMAIN_MODELING_ADJUSTMENT_HANDOFF_2026-10-01.md>)
supersedes the broad-speed model-selection order where they conflict.

- [x] Read the full adjustment; preserve schema-7 broad data and checkpoints.
- [x] Build schema-8 <=12 m/s views with complete-window eligibility,
      clean source-run gates, preserved run splits/condition/reset metadata,
      and compare 1 s / 2 s post-excursion cooldowns.
- [x] Build separate OOD/support diagnostic archive; retain race-relevant
      <=12 m/s slip in the core view.
- [x] Re-run response atlas and matched-state/history analysis on race views.
- [x] Train and score a 1 s-context/1 s-horizon direct comparator with
      whole-test-run and initial speed/steering/mismatch strata; existing long
      direct checkpoints had no eligible held-out sequences in the phase-level
      view. This is preliminary, with only two independent test runs.
- [ ] Train 2 s and 4 s RSSMs at 128/16 and 256/32 on race views; defer 512/64
      unless in-domain capacity evidence and closed-loop scores justify it.
- [ ] Refit/simplify grey-box on race views; quantify parameter sensitivity.
- [ ] Build and evaluate nominal-physics plus latent-residual teacher.
- [ ] Score plant candidates in the short production odom/localization/MPC
      branch harness and add the prescribed stratified starts.
- [ ] Add track boundary/vehicle collision geometry before any full-lap safety
      claim or broad weight search.
- [ ] Design/collect the missing clean <=12 m/s braking, 9–12 m/s combined,
      corner-exit, and progressive-race data; no >12 m/s sweep.
- [ ] Train/evaluate a race-domain odom residual while production odom remains
      the default expert.
- [ ] Generate/validate the 0–12 m/s empirical planner envelope.
- [ ] Freeze architecture, collect the fresh blind set, answer final decisions,
      then request the subagent audit.

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
  batch 16, CUDA, 1,600-update maximum, early stopping) was interrupted by the
  user's stop request during the optimizer update after step 900. Its best
  checkpoint is step 700 (validation score 0.07034 versus 0.15056 at step
  100). The run has no final training report and is not a completed/promoted
  candidate.
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
  confirmation. The scorer completed; the measured results are summarized
  below and saved in the report.
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

### 2026-10-01 — raceline-domain audit and sampling correction

- The control-domain audit stratifies speed, actual steering, throttle command
  and slew, run family, split, and independent-run support. It counted 839,155
  rows in 0–12 m/s, but only 3,940 rows at 10–12 m/s, all from one open-plane
  training run; practice captures have no 8–12 m/s samples. Measured practice
  steering at 6–8 m/s is also sparse. This is an evidence/coverage gap, not a
  reason to simply increase model size.
- Earlier candidate trainers sampled runs uniformly. The archive contains 105
  open-plane and 11 practice-track runs, and schema-7 condition IDs were not
  used by those samplers. Added `family_condition_sampler.py` and wired the
  hierarchy into direct/RSSM/NSSM/grey-box/sensor-observer training. Sampling
  now chooses family, run, condition, then eligible sequence/window; fixed
  run-level splits and evaluation windows are unchanged. Reports include the
  effective family probabilities and eligible run/condition counts.
- Extended the shared model loader to schema 7 and checked condition-to-run
  references plus each frame's run/reset metadata against sequence metadata.
  The eligible 4 s context + 2 s rollout training subset has five runs from
  each family, 765 open-plane conditions and five practice conditions. A
  seeded 10,000-draw check yielded 50.13% / 49.87% family selection.
- The new sampler tests passed (2), along with the existing RSSM/direct/
  grey-box mathematical tests (9); edited files compile and schema-7 archive
  loading succeeds. An initial test invocation under system Python lacked
  PyTorch; the existing CUDA-enabled modeling venv was then located and used.
  At the time this entry was written, no new training or simulator capture had
  begun; subsequent race-domain work is recorded below.

### 2026-10-01 — race-domain plan adjustment and immutable dataset views

- Read the complete 2,081-line race-domain adjustment handoff. Its 0–12 m/s
  scope supersedes the previous broad-domain RSSM capacity-first plan. Primary
  scoring is now 0–9 m/s plus a separate 9–12 m/s boundary; >12 m/s is only
  OOD/support behavior.
- A family-balanced 256/32 RSSM was started before this adjustment arrived.
  It used the broad schema-7 view, so it was interrupted after evaluation at
  step 1,000. Best validation was step 900, 0.07747; the partial checkpoint and
  history are preserved in
  `rssm_familybalanced_h4_f2_h256_z32_20261001/`. It is not a candidate under
  the new domain rules and its practice-validation metric is not accepted.
- Added `build_race_domain_dataset.py`. It requires exact 25 ms source time,
  schema-7 metadata, the existing fixed splits, zero collisions, no abort,
  zero bridge timing faults, clean stream/collision gates, and empty whole-bag
  quality failures. Every emitted sequence is wholly within 12 m/s and has
  finite state/acceleration labels. It keeps source rows in order and records
  source-frame/sequence indices.
- Created immutable race views at
  `plant_teacher_race_domain_v1/cooldown_1s/openplane_dynamics.npz` and
  `.../cooldown_2s/openplane_dynamics.npz`, plus an OOD-only diagnostic archive
  at `vehicle_dynamics_ood_extreme_v1/openplane_ood_diagnostics.npz`. Both
  cooldown settings currently retain identical rows: every qualifying >12 m/s
  episode in the clean source data ends without re-entering the domain. The
  views contain 809,398 race rows across 7,787 sequence pieces and 85 clean
  runs. OOD diagnostics contain 19,515 frames with overlapping reason flags.
- OOD labels include >12 m/s, the 2 s post-excursion interval, the empirical
  99.5th-percentile rear-wheel/body mismatch tail (25.82 m/s proxy), the
  empirical 99.5th-percentile tilt tail (0.1033 rad; explicitly not a verified
  rollover threshold), and absence from a coarse training support cell. No
  unsupported cell was found at that coarse speed × steering × throttle
  resolution. High-slip <=12 m/s rows remain in the race dataset.
- Strict quality gating exposes a key limitation: no practice-track run
  survives in validation or final-test. The only eligible validation/final
  runs are one open-plane run each; 9 practice runs were rejected for timing
  faults and/or alignment quality. Existing practice validation scores from
  those bags cannot be used as clean confirmation. This is recorded as a data
  gap requiring clean race captures, not as grounds to relax gates.
- Added schema-8 race-domain validation to the model loader so plant training
  refuses OOD-role archives and any race-view frame above its declared cap.
  Long-window race-domain model training remains blocked on usable validation
  windows; a continuity-safe data rebuild is underway.

### 2026-10-01 — race-domain coverage and response analysis

- Reprocessed the 1,508-condition throttle sweep without modifying its source.
  The immutable `throttle_surface_race_domain_v1` keeps all condition records
  and truncates each at its first >12 m/s sample: 701,082/725,833 rows remain;
  88 conditions cross the cap. Two conditions have fewer than 20 usable
  baseline samples, and one phase has a packet discontinuity; the atlas skips
  those conditions/windows rather than treating them as clean responses. The
  expected initial NaNs in the 100 ms encoder-speed derivative are retained
  and do not truncate a condition.
- The capped throttle atlas contains 9,919 complete in-domain response windows
  across 1,469 conditions. Baseline feedback tracking is 100%; target throttle
  is within 0.05 of request in 89.5% overall and 28.9% during the first 100 ms.
  This characterizes transition response, not merely steady-state gain.
- Zero throttle applies active brake torque in this simulator. The current
  throttle wire interface has no separately observable coast command; the
  coverage report does not mislabel zero throttle as coast or infer braking
  merely from deceleration. Coast remains an interface/label gap.
- Race-domain coverage reports rows, contiguous sequences, conditions, and
  independent source runs across the requested speed and steering bins,
  measured throttle, command direction, and wheel/body mismatch. Clean 9–12
  m/s support is only in training: 6,165 rows from one throttle-sweep run.
  Validation, test, and final-test have no samples above 9 m/s. There is not yet
  evidence for high-speed generalization. Training-set absolute wheel/body
  mismatch p50/p90 are 0.574/19.673 m/s; this kinematic proxy is not a tire
  force or direct tire-slip measurement.
- Cross-run matched-state analysis used 1,719 queries from 16 independent
  runs. At the 0.5 IQR-scaled match radius, 500 ms history reduces median
  yaw-acceleration dispersion to 0.746× current-state-only (run-cluster 95% CI
  0.537–0.848); 2 s history gives 0.722× (0.474–0.983). Benefit is at 3–9
  m/s; 0–3 m/s did not improve, and no matched 9–12 m/s queries exist. This is
  observational matched-state evidence for history dependence, not a causal
  or closed-loop result.
- The existing 5–10 s direct checkpoints have no eligible clean held-out
  sequences in the phase-level race view. A 1 s-context/1 s-horizon direct
  comparator was trained without test access and scored once on two independent
  test runs. Normalized body-state RMSE is 0.121/0.139/0.172 at 0.25/0.75/1.0 s
  versus persistence 0.214/0.380/0.348. At 1 s, one run is worse than
  persistence; the two run errors differ sharply. Regime bins have one or two
  runs each, and there are no test examples above 9 m/s or in the high-mismatch
  tail. Keep this only as a short-horizon predictive comparator, not a trusted
  teacher or MPC/odometry integration candidate.
- Updated race-domain run gating to accept a practice bag only through the
  repository's explicit complete-lap-0-to-12 active-interval quality check:
  exactly 13 lap transitions, zero collisions, zero in-interval timing faults,
  and every stream cadence passing. This honors the pre-existing strict active
  interval protocol; all other run-level quality gates remain unchanged.
- An initial throttle derivative/atlas attempt exposed expected encoder
  initialization NaNs and a missing copied debug-field manifest entry. The
  failed derivative and partial atlas outputs were removed and rebuilt; source
  bags and prior datasets were untouched.
- Started a curated rebuild of existing clean bags through the repository's
  `prepare_dataset.py --coalesce-contiguous-phases` path in the cached ROS/API
  image. The coalescer joins only consecutive simulator packet IDs and never
  crosses a recorded reset, preserving fixed-time physics and run splits. The
  goal is to supply valid 2–6 s sequences without relaxing any data-quality
  gate or changing simulator behavior.

### 2026-10-01 — adjustment handoff applied; continuous capture in progress

- Re-read the complete race-domain adjustment. Primary loss/acceptance is
  restricted to 0–12 m/s; >12 m/s remains support/OOD diagnostics. The 0–9
  core and 9–12 boundary remain separate in reporting; no >12 m/s capture is
  planned.
- The race-view builder initially stopped on a source-schema mismatch: the
  coalesced schema-7 archive had `sequence_condition_id` and `condition_labels`,
  but no `condition_run_index`. It now rebuilds run-local condition IDs from
  `(sequence run, condition label)` and preserves original IDs for provenance.
  This matters because 641 source condition labels were reused across runs.
  A focused test verifies that shared labels map to distinct run-local IDs.
- The immutable coalesced race views are under
  `live_runs/derived_dynamics_learning_20260928/plant_teacher_continuous_race_domain_v1/`.
  Both cooldown choices retain 568,999 rows across 3,752 sequence pieces and
  11 runs; no run re-enters <=12 m/s after an excursion, so the 1 s and 2 s
  views are identical. The OOD-only archive contains 17,112 labeled rows.
- The rebuilt coverage report is `race_domain_coverage_continuous_20261001.json`
  plus its compressed CSV. In train, 9–10, 10–11 and 11–12 m/s have 2,225,
  1,263 and 2,677 rows respectively, all from the one 1,508-condition throttle
  sweep. Validation/test/final-test have no rows above 9 m/s. Existing samples
  above 9 m/s are nearly straight (actual steering <=0.1 rad); larger steering
  at that speed is unsupported, not presumed feasible.
- The coalesced response atlas has 568,999 samples, 3,752 sequences, 760
  conditioned bins, and 9,919 complete throttle-response windows. Its frame
  audit confirms the expected body/rear-axle mapping. Zero throttle is active
  braking; the command interface has no separate observable coast command.
- Race-only matched-state analysis has 1,338 queries from 10 runs. At match
  radius 0.5, 500 ms history changes median dispersion ratios to 0.93 for `ax`
  (run-cluster 95% CI 0.85–0.99), 1.00 for `ay` (0.90–1.02), and 0.81 for yaw
  acceleration (0.60–0.90). At 2 s history the ratios are 0.96 (0.86–1.09),
  1.11 (1.05–1.31), and 0.62 (0.24–0.78). History helps yaw, not every output;
  no matched 9–12 m/s queries exist.
- Added `race_domain_experiment_plan.py` and the development-only
  `race_domain_continuous` profile. It records one uninterrupted 260 s phase,
  covering 2–11 m/s, both steering signs, steering reduced with speed,
  corner-exit/unwind and descending speed targets. A measured-speed steering
  limiter and 11.2 m/s command governor are paired with the existing 8 deg
  tilt/collision stops and an 11.9 m/s hard cutoff. Simulator physics are
  unchanged. Seven focused plan/domain-view tests pass; `bash -n` and
  `git diff --check` pass; no broad compilation suite was run.
- The first batchmode pilot was intentionally stopped at 75.4 s to prevent
  Unity's default terminal logger from overwhelming the run. Its bag is 12.4
  MiB; `ros2 bag info` reports 3,015 samples each for odom, IMU, IPS, both
  encoders, steering and throttle over 75.43 s, zero bridge timing-fault
  messages, and 40.03 Hz command output. It is marked interrupted/aborted and
  is not eligible for fitting. The simulator was relaunched with the same
  pinned Explore image and batchmode launcher, changing only
  `SDU_APEX_SIM_LOG_FILE=/dev/null`.
- `openplane_race_domain_train_20261001_r02` (seed 17002) completed all 65
  blocks in 259.999 s with `aborted=false`, zero collisions, zero bridge timing
  faults, 100% packet-sequence alignment and sensor streams at 39.966 Hz.
  The explicitly completed-but-unscored phase had previously been omitted by
  the default phase loader. A narrow admission check now accepts only the exact
  seeded 65-block/260 s profile with a complete phase and experiment-end event;
  the normal whole-run quality gates still apply. Four focused tests cover
  accepted, truncated, mismatched and aborted event records. Re-export recovers
  10,390 samples across two packet-contiguous sequences.
- The first run did not reach the fast boundary: measured speed max was
  9.595 m/s. At target speeds 9.5/10.5/11.0, the inherited feedforward
  produced median throttle commands 0.367/0.388/0.397 and median speeds
  8.83/9.31/9.57 m/s. Throttle feedback tracked commands closely. Thus the
  limitation was the excitation controller, not the bridge cadence or dropped
  actuator feedback.
- A read-only re-analysis of the existing throttle sweep found near-straight,
  stable command plateaus around (8.49 m/s, 0.35), (9.66 m/s, 0.40), and
  (10.82 m/s, 0.45). These are from one source run and only 3/6/2 condition
  sequences respectively, so they are a preliminary capture-calibration
  estimate—not a production model or established uncertainty bound. The
  race-domain experiment uses their linear inverse only above 8.49 m/s and
  keeps the existing 0.5 command cap, 11.2 m/s governor and 11.9 m/s hard
  cutoff. The profile event records those anchors for exact reproducibility.
- `openplane_race_domain_train_20261001_r03` (seed 17003) passed the same
  completed-profile, collision, timing, stream and packet gates. It exported
  10,392 rows in one continuous sequence. Odom ran at 39.969 Hz (p95 gap
  25.75 ms, max 48.88 ms); throttle commands at 40.000 Hz. There were no
  collisions, no timing faults, no packet mismatches, and the measured peak
  speed was 11.007 m/s. No samples exceeded 12 m/s.
- By target, r03 measured medians 9.46, 10.47 and 10.98 m/s for 9.5, 10.5 and
  11.0 m/s commands; throttle-command medians were 0.395, 0.438 and 0.458.
  The 95th-percentile throttle feedback error was below 0.001 norm. Counts by
  measured speed bin were 1,270 (9–10), 2,251 (10–11), and 143 (11–12).
  This validates the capture-only feedforward around 9–11 m/s, but the upper
  bin remains thin and all of these rows are training data from one run.
- r03 also contains five designed descending-speed transitions from positive
  throttle to the interface's zero/active-brake command at measured speeds
  10.99, 10.44, 8.95, 7.01 and 4.93 m/s, with steering feedback magnitudes
  0.02–0.16 rad. In the first event, estimated rear-wheel surface speed fell
  from 11.52 to 3.01 m/s within 100 ms and to zero by 250 ms while body speed
  was still 9.10 m/s; body speed then fell 1.89 m/s in 250 ms. This could be
  severe simulated braking/wheel lock or an encoder-estimation transient; one
  run cannot distinguish them. Other speed-hold cuts rapidly re-engaged
  throttle, so a fixed-duration, matched-state brake pulse is still needed.
  No separately controlled coast or negative-throttle input is available on
  the development command interface.
- Version 2 of the capture schedule moves only the final straight target from
  11.0 to 11.1 m/s, keeping the 11.2 governor, 11.9 hard limit, steering
  envelope and physics unchanged. Its plan version is recorded and admitted
  separately; prior 11.0 m/s captures remain reproducible and loadable.
- `openplane_race_domain_validation_20261001_r04` (seed 17004, version 2)
  passed the whole-run gates and is assigned validation-only. It exported
  10,397 rows in one contiguous sequence; peak body speed 11.109 m/s and no
  samples above 12 m/s. Speed-bin counts were 1,274 (9–10), 1,278 (10–11),
  and 1,115 (11–12). Odom cadence was 39.985 Hz, p95/max gap 25.842/49.733
  ms, with zero collisions, timing faults, or packet-alignment losses. Maximum
  measured roll/pitch magnitudes were 0.1141/0.0592 rad. This verifies the
  excitation profile's boundary coverage, not a learned model's generalization.
- Active real-simulator capture:
  `openplane_race_domain_validation_20261001_r05` (seed 17005, version 2) is
  collecting an independent second validation run after restarting the same
  pinned Explore container through the same batchmode launcher.
- Remaining in handoff order: collect/assign whole-run race-domain splits and
  controlled throttle-down/brake, 9–12 m/s combined steering/braking and
  corner-exit captures; rebuild the race view, atlas and matched-state report;
  train direct/RSSM and nominal-plus-latent residual teachers; compare short
  production-stack branches; add boundary geometry; derive odometry residual
  evidence and the <=12 m/s planner envelope. No model is integrated into
  production. A fresh blind practice run and independent audit remain deferred
  until the model-selection/evidence work is genuinely complete.

### 2026-10-01 — race-domain adjustment applied and focused boundary test prepared

- Read the complete 2,081-line `SDU_APEX_RACE_DOMAIN_MODELING_ADJUSTMENT_HANDOFF_2026-10-01.md`.
  Its scope change is applied: model selection and offline optimization are
  bounded to 0–12 m/s; >12 m/s is diagnostic/OOD only. High wheel/body
  mismatch below 12 m/s remains in-scope.
- Audited the current artifacts rather than treating the handoff's background
  metrics as new evidence. The current 2 s-context/5 s direct checkpoint is
  `race_domain_direct_20261001/c2_f5_w128_seed17007/training_report.json`:
  it has only two eligible validation runs, `openplane_race_domain_validation_20261001_r04/r05`,
  and no eligible `test` or `final_test` sequence for its 5 s horizon. These
  are open-plane prescribed-profile captures, not three practice-track test
  runs. Its macro normalized body-state RMSE is 0.0528/0.0755/0.0807/0.0865
  at 0.25/0.75/2/5 s versus persistence 0.0748/0.1237/0.2807/0.3779.
  This establishes prediction on that excitation distribution only; it does
  not establish practice-race transfer or a recursive plant.
- The corrected cooldown-2 s race view used by this model ladder contains
  610,574 rows, 3,757 sequence pieces, and 15 source runs. Long-window eligible
  training is five runs; validation is only r04/r05; there are no eligible
  held-out test/final-test sequences. The sole clean practice run remains in
  training. Thus the production-MPC branch harness still lacks an independent
  clean practice start set; no racing branch score is claimed.
- Completed a planned RSSM context/capacity ladder on this exact race view,
  without touching test/final-test: 2 s/128-16 (best score 0.1163 at step
  1,400), 2 s/256-32 (0.1040 at step 3,000), and 4 s/128-16 (0.0979 at step
  3,000). For 4 s/128-16, macro normalized body-state RMSE is 0.1009/0.1019/
  0.1361 at 0.25/0.75/2 s; persistence is 0.1124/0.1533/0.2286. Its 2 s
  core-domain run errors span 0.111–0.228 across only two runs, while the
  9–12 m/s boundary score is lower on the straight prescribed profile. Longer
  context helps this validation score, but the two-run interval is too weak to
  establish generalization. The completed 4 s/256-32 comparison scored 0.0871
  at step 3,000; its macro normalized body-state RMSE is 0.0671/0.1244/0.0958
  at 0.25/0.75/2 s versus persistence 0.0614/0.1853/0.2727. The 2 s core
  macro error is 0.1047 and the fast-boundary score is 0.0770 on the same
  straight-heavy two-run validation set. It improves 0.75/2 s persistence but
  not 0.25 s; this is a candidate comparator, not evidence of racing
  generalization. No 512-64 run is justified absent more diverse validation.
- Added an explicit race-speed support flag to the causal GRU/RSSM adapters.
  The 12 m/s cap is checked against the propagated ensemble state and is not a
  state clamp. The closed-loop branch path caps MPC target speed at 12 m/s and
  stops before feeding an out-of-domain predicted state to synthetic sensors.
  This is offline-only; production MPC, odometry, AMCL, EKF and simulator
  physics are unchanged.
- Added the separate seeded `race_domain_brake_boundary` development profile:
  51 four-second blocks (204 s), progressive signed steering sweeps at
  9.5/10.5/11.1 m/s, eight matched throttle-down/braking pairs that hold
  steering, low-speed repositioning between trial groups, then corner
  exit/unwind. The maximum angles rise only 0.01 rad from the previous clean
  profile (to 0.09 rad at 9.5/10.5 and 0.05 rad at 11.1). It retains the
  existing 11.2 m/s governor, 11.9 m/s hard cutoff, collision abort and
  8-degree tilt abort; zero throttle remains active braking. No physics or
  runtime controller code is changed. The new capture-admission gate checks
  the exact seed, serialized plan, duration and completion events before
  allowing this otherwise-unscored run into dataset preparation.
- Focused validation passed: 12 modeling/plan tests, 7 ROS-container capture
  admission tests, `py_compile`, shell syntax, and `git diff --check`. The
  pinned Explore simulator and API images were local. The planned v3 boundary
  train/validation captures and split-correct source/race-view rebuild were
  completed; see the next dated entry for their measurements and the evidence-
  driven v4 follow-up. The direct/RSSM comparisons remain offline-only. Clean
  held-out practice starts, the production-MPC branch comparison, boundary
  geometry, odom residual, and empirical 0–12 planner envelope remain open.
  No candidate has been integrated into production and no independent audit
  has been requested because the handoff is not yet complete.

### 2026-10-01 — new brake-boundary evidence and moderate-steering follow-up

- The v3 `race_domain_brake_boundary` training capture
  (`openplane_race_domain_brake_boundary_train_r01`, seed 17020) completed all
  51 blocks/203.999 s. The exact-plan and whole-run admission gates passed;
  there were zero collisions/timing faults, packet alignment was 99.988%,
  odom/sensor cadence was 39.987 Hz (p95 gap 25.87 ms), commands were 40.000
  Hz, and body speed remained <=11.119 m/s. Steering/throttle feedback p95
  command errors were 0.00030 rad/0.00475 norm. All eight approach/brake
  trials reached the intended target before the pulse; the late-block median
  speeds followed 9.5->7.0, 10.5->8.0, and 11.1->9.5 m/s at fixed steering.
- The independent v3 validation capture
  (`openplane_race_domain_brake_boundary_validation_r01`, seed 17021) also
  passed exact-plan and whole-run gates: zero collisions/timing faults, 100%
  packet alignment, 39.987 Hz sensors, 40.000 Hz commands, and peak speed
  11.264 m/s. Feedback p95 errors were 0.00030 rad/0.00520 norm. Six of eight
  brake-pair approaches matched their target; two left-turn approaches did not
  settle at 9.5/11.1 m/s in their four-second block. Those are retained as
  valid whole-run observations but are explicitly not counted as matched
  braking pairs.
- Rebuilt the complete 17-run source bundle under
  `live_runs/derived_dynamics_learning_20260928/race_domain_brake_boundary_source_20261001_splitfixed/`.
  All source-manifest split assignments were passed explicitly. A first
  intermediate rebuild defaulted earlier race-validation run IDs to train;
  this was caught before model fitting. Only the exact newly created
  split-mismatched derivatives were removed; original bags/manifests and the
  corrected source bundle remain intact. The <=12 m/s race view and OOD-only
  archive are under `plant_teacher_race_domain_brake_boundary_20261001_splitfixed/`
  and `vehicle_dynamics_ood_brake_boundary_20261001_splitfixed/` respectively.
  The race view contains 626,912 rows/3,759 sequences: train 475,094 rows from
  11 runs; validation 69,897 rows from 4 runs; test/final-test are preserved
  and remain unscored. High-speed bins now have 3–4 training runs and 3
  validation runs each, but active-brake samples above 0.1 rad remain sparse
  and no 11–12 m/s moderate-steering brake cell is supported.
- That measured gap prompted a distinct `race_domain_moderate_braking`
  development profile rather than altering legacy captures: 64 six-second
  blocks (384 s) with straight approach, randomized signed steering steps up
  to 0.14 rad at 9.5/10.5 and 0.12 rad at 11.1 m/s, then separated turn-in,
  positive-throttle reduction, zero-throttle active braking, and steering
  release, plus corner-exit/unwind. Existing 8-degree tilt/collision aborts,
  11.2 m/s governor and 11.9 m/s hard cutoff remain. The separate profile is
  versioned; v3 bags remain exactly admissible. Simulator physics and
  production control are unchanged. Zero throttle remains active braking, not
  a coast condition.
- Focused checks passed after implementation: 6 deterministic plan tests, 9
  ROS-container capture-admission tests, targeted syntax checks, shell syntax,
  and `git diff --check`. The v4 training capture
  (`openplane_race_domain_moderate_braking_train_r01`, seed 17022) is currently
  running on the pinned Explore simulator through the same batchmode launcher.
  No GPU training has been run on the new split-correct view yet.
- Next: validate the v4 run's actual speed/feedback/tilt and paired response;
  capture an independent validation run only if it completes cleanly; rebuild
  the split-correct 0–12 view with both v4 runs; then retrain/re-score only the
  direct predictor and leading 4 s/256-32 RSSM comparator. After model
  comparison, proceed to the grey-box/residual and short production-stack
  branch evaluations. The production-MPC harness still lacks independent
  held-out practice starts; no race generalization claim or production
  integration is made.

### 2026-10-01 — braking response replicated; expanded race-view evaluation

- The v4 moderate-steering training capture completed all 64 blocks/384.024 s
  and passed its exact-plan/quality gates: 40.00 Hz commands, 39.987 Hz sensor
  streams, 99.988% packet alignment, no collision/timing failures, peak speed
  11.113 m/s, and peak measured roll/pitch 6.488/3.386 degrees under the
  existing 8-degree abort. All scheduled turn-in states were reached; feedback
  p95 errors were 0.0002 rad steering and 0.000884 throttle norm.
- The independent v4 validation capture completed the same 64-block schedule
  from a fresh simulator start and also passed: 40.00 Hz commands, 39.987 Hz
  sensors, 100% packet alignment, zero collisions/timing failures, peak speed
  11.112 m/s, and peak roll/pitch 6.515/3.377 degrees. No simulator physics or
  production control code was changed.
- Zero-throttle braking produced a repeatable rear-wheel/body-speed transient
  in both independent v4 runs. At the same straight 11.074 m/s brake onset,
  rear wheel surface-speed estimates fell to zero within 0.25 s while body
  speed remained 9.24 m/s; IMU longitudinal acceleration was about -7.9 to
  -8.4 m/s^2. Peak absolute wheel/body mismatch was 10.140 vs 10.142 m/s in
  the two runs, with matching transition and timestamp. Packet timing and
  feedback stayed clean. This supports a rapid wheel-rotation/braking
  decoupling state in the simulator; encoder wheel speed is only a proxy, so
  it does not establish per-tire force or individual-wheel lock.
- Rebuilt the split-preserving source with 19 whole runs and a new <=12 m/s
  schema-8 view at
  `live_runs/derived_dynamics_learning_20260928/plant_teacher_race_domain_moderate_braking_20261001/cooldown_2s/`.
  It contains 657,621 race rows / 3,761 sequence pieces; train has 490,450 rows
  from 12 runs, validation 85,250 rows from 5 runs, and test/final-test remain
  stored but unscored. The parallel OOD archive contains only out-of-domain
  diagnostics. Explicit run split overrides were used for every source run.
- Updated the response atlas and matched-state analysis on this view. Both now
  constrain model-selection diagnostics to train/validation; a regression test
  verifies test runs cannot enter matched-state queries or their neighbor
  pool. The atlas has 575,700 in-scope train/validation samples. Its 9–12 m/s
  response bins now span 8–9 independent runs, but the 11–12 m/s moderate
  steering + active-braking cell still has only six transient samples in each
  of one training and one validation run. Matched states show pronounced
  channel dependence: 250–500 ms history lowers median local longitudinal and
  yaw-acceleration dispersion, while lateral dispersion stays around 1.1 m/s^2
  and rises with 2 s history. These are descriptive local-match results, not
  proof that history reconstructs a sufficient state.
- Retrained the 2 s/5 s direct Transformer (width 128, three layers) on the
  expanded view, without test scoring. It selected step 2,800 at normalized
  validation score 0.11030, with four eligible independent validation runs.
  In the 9–12 m/s macro, normalized body-state RMSE is 0.1150/0.0459/0.0334/
  0.1322 at 0.25/0.75/2/5 s respectively; the 5 s per-run values are
  0.2384/0.2029/0.0347/0.0530. It is not uniformly predictive across runs or
  horizons and remains a non-recursive comparator, not an accepted plant.
- The matched-state, response-atlas, and train-only normalization checks pass.
  A clean updated four-rung RSSM ladder (2/4 s context × 128/16 and 256/32,
  2 s prior rollout) is underway sequentially on the persistent GTX 1080
  environment. No test/final-test scores have been requested. After that,
  proceed only with evidence-supported candidates to grey-box/residual and
  production-stack branch evaluation. New practice-track validation remains
  absent, so neither free-running whole-lap fidelity nor racing generalization
  is established.

### 2026-10-01 — race-domain direct/RSSM rerun and paired comparisons

- Completed the fixed race-domain four-rung RSSM study. All jobs used the same
  schema-8 dataset, run-level splits, 2 s prior rollout, 3,000 maximum updates,
  family/run/condition/window sampling, and validation-only checkpoint
  selection; `score_test` was false for every job. Best selection scores were:
  2 s/128-16 = 0.11746 (step 1,400), 2 s/256-32 = 0.07717 (step 3,000),
  4 s/128-16 = 0.08363 (step 3,000), 4 s/256-32 = 0.09675 (step 2,800).
- On each model's four eligible validation runs, the 2 s/256-32 model's
  fast-domain normalized body-state RMSE is 0.0340/0.0544/0.0876 at
  0.25/0.75/2 s; its core-domain scores are 0.0662/0.0914/0.1294. The
  4 s/128-16 alternative is 0.0284/0.0502/0.0919 fast and
  0.0798/0.1029/0.1486 core. This per-job table is descriptive; it uses
  sampled validation starts that are not identical across configurations.
- Corrected that limitation with paired frozen-model evaluation on common
  future starts from the same four validation runs. At 2 s context, 256/32
  versus 128/16 reduces normalized error 43.1%, 30.0%, 22.4%, and 20.7% at
  0.25/0.75/1/2 s. Run-cluster 95% intervals exclude zero improvement at all
  four horizons, and all four run differences favor 256/32 at 0.25/0.75/1 s;
  at 2 s, three of four favor it and one is slightly worse.
- For 4 s context, 256/32 versus 128/16 improves paired error by 6.8%, 10.0%,
  and 8.0% at 0.25/0.75/1 s with run-cluster intervals excluding zero, but is
  2.2% worse at 2 s (interval includes zero). For 256/32, paired 4 s versus
  2 s context is 25.0% worse on the shared validation metric. For 128/16,
  paired 4 s context is effectively tied (+1.1%); neither supports a material
  history-length gain under the paired metric. These comparisons favor the
  2 s/256-32 RSSM as the lead offline branch candidate; it remains
  comparator-only pending practice-transfer and closed-loop results.
- Updated pairwise comparison artifacts are under
  `live_runs/derived_dynamics_learning_20260928/race_domain_rssm_moderate_braking_20261001/`:
  `paired_validation_c2_capacity.json`, `paired_validation_c4_capacity.json`,
  `paired_validation_context_h128.json`, and
  `paired_validation_context_h256.json`. All compare validation only; the
  held-out test/final-test arrays were not scored.
- The 2 s/5 s direct predictor remains a non-recursive comparator. It selected
  step 2,800 with validation score 0.11030 and four eligible whole validation
  runs. At 9–12 m/s its normalized body-state RMSE is 0.1150/0.0459/0.0334/
  0.1322 at 0.25/0.75/2/5 s. Its 5 s values vary sharply by source run
  (0.238/0.203/0.035/0.053), so it is not a free-running teacher.
- The primary missing evaluation remains actual practice-track transfer.
  Existing practice trajectory data is training-split only; the validation
  captures so far are open-plane. The next action is a fresh, short,
  collision-guarded practice capture with the unchanged production development
  MPC/odometry/localization, pinned practice image, matched map/raceline, and
  debug bag. It will be assigned to validation after capture and used only as
  unseen real-state starts for short branches; no learned candidate is
  integrated into the live stack.
