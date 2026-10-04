# Black-box offline plant progress — 2026-10-03

Scope follows `SDU_APEX_NEXT_BLACK_BOX_PLANT_HANDOFF_AFTER_2129427_2026-10-03.md` from frozen source commit `2129427838101eee277e17e91727810bbe2df687`. This is an evidence record, not a model promotion. No simulator physics, production odometry, or MPC behavior has been changed.

## Current status

WP15–WP26 are implemented and evaluated through the handoff's current stop gate. WP19 freezes body-state increment plus explicit acceleration consistency (B-B), and direct fixed-25-ms encoder angle increment (W-D), as the representation carried into replacement-plant research. WP20 narrowly calibrated run-balanced kNN support; WP21 passed structural tests; WP22 and WP23 failed the material-improvement gate. WP24 reproduced frozen A2 exactly. WP25 localized severe latent-transition/re-encoding inconsistency and weakened the support-damping hypothesis. None of the three authorized WP26 variants passed qualification, so the A2 family stops here and conditional WP27 is not run. No candidate is promoted; the offline simulator remains unsuitable for long-lap simulation, raceline generation, or MPC tuning. WP18 did not support a history-length plateau under its tested run-balanced neighbor estimator, so the 2-second history remains a controlled setting, not a claim of sufficiency. No production odometry, MPC, simulator physics, or active simulator process was changed.

## WP15 — experiment and provenance registry

The registry indexes existing training metadata, comparison artifacts, dataset/checkpoint hashes, whole-run splits, and unresolved document references:

- Script: [`build_experiment_registry.py`](../../tools/vehicle_dynamics_learning/build_experiment_registry.py)
- Registry: `live_runs/derived_dynamics_learning_20260928/experiment_registry_20261003.json`
- Human summary: [`OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md`](OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md)

The refreshed registry finds 115 training metadata files, 93 model runs, and 47 diagnostic artifacts; it checks 55 checkpoint hashes. It records 138 document artifact references, 2 unresolved historical literal references, and 1 unmatched historical glob, with no recorded diagnostic hash mismatches. It indexes active WP19–WP26 artifacts and does not promote a default plant.

## WP16 — signal semantics and wheel-state evidence

Signal definitions, fixed-25-ms encoder semantics, frame conversions, packet/reset validity, and canonical operating-region masks are implemented and unit-tested in `signal_semantics.py` and `operating_regions.py`. The exact current production odometry replay was built from the repository C++ source/config and checked by source hashes. The fixed-40-Hz encoder sidecars were row-matched to their source datasets.

The frozen parent checkpoint is `8f84fa54306492bd9750e4e1e3fb0be014195491eccd8f4fd67f2943402bfe8e`. On 7 OpenPlane validation runs, its one-step rear-wheel prediction RMSE was 0.757 m/s against the unsmoothed fixed-25-ms encoder rate (run-bootstrap 95% CI 0.449–1.284), versus 0.311 m/s against the stored approximately-100-ms filtered wheel proxy (CI 0.226–0.450). On 2 unseen practice runs the corresponding values were 1.001 m/s (0.987–1.016) and 0.405 m/s (0.401–0.408). The fixed-period angle-increment target is the same measurement expressed in radians, not an independent target. This exposes the noise/smoothing difference; it does not establish that encoder rate is a physical plant state.

Canonical artifacts:

- `.../full_throttle_domain_v1/wp16_frozen_parent_one_step_v3.json`
- `.../full_throttle_domain_v1/wp16_wheel_causality_20261003_v4.json`

The wheel audit covers 7 OpenPlane validation and 2 unseen-practice runs. High-steering and high-speed regions show substantial wheel/body proxy mismatch. WP16's signal/math/production-replay gate is complete; WP17 owns the recursive-state decision.

## WP17 — frozen-parent causal interventions

`intervene_effective_race_teacher.py` performed the handoff's I0–I6 diagnostics on the unchanged frozen parent and the exact WP14 starts: 135 dynamic windows across 6 runs and 31 practice windows across 2 runs. At the 10-second horizon only 117 dynamic windows had a complete raw encoder label; no practice 10-second starts existed. The checkpoint hash remained unchanged, no training occurred, and test/final-test captures were not used.

The paired dynamic endpoint results reject measured processed rear-wheel rate as a recursive physical state for the replacement plant. At 5 seconds, giving the parent measured raw or filtered wheel-rate oracles predominantly regressed dynamic body endpoints: for I1, relative improvement was −17% in `u`, −81% in `v`, −28% in yaw rate, −61% in heading, and −46% in radial position; run-level intervals were negative for each measure. I2 had the same broad regression pattern. Practice evidence is weak (two independent runs). Mean-neutralized wheel interventions are off-support and are not interpreted as clean causal effects; the body-oracle branch is explicitly non-predictive by construction.

Frozen decision: **rear processed wheel rate is measurement/output only**. This does not mean wheel physics or latent traction history is irrelevant; a replacement model must infer and predict any needed latent state causally.

Artifact: `.../full_throttle_domain_v1/wp17_frozen_parent_interventions_v4.json`.

## WP18 — black-box observability and history sufficiency

The analyzer matches on current recorded bridge-odometry `u/v/r`, steering/throttle feedback, and steering/throttle command. Simulator rigid-state truth and future encoder values are labels only. It compares the body/actuator vector with the same vector plus current filtered wheel measurements, and causal trailing mean/slope summaries at 0 ms, 100 ms, 250 ms, 500 ms, 1 s, 2 s, and 4 s. The analysis is run-balanced and cross-run: 8 held-out query runs (6 OpenPlane validation, 2 unseen practice) and 18 candidate runs (10 train, 6 validation, 2 practice). Test/final-test runs are excluded. Per-run query rows are capped at 200 per regime and candidate banks at 1,800 base rows plus at most 160 new rows per regime.

There are 39 region strata and 14 history/feature settings. The report retains per-response, per-run conditional variance, neighbour spread p50/p90, independent-run counts, distance, and run-cluster bootstrap intervals. Caps and exact source/fixed-sidecar row alignment were audited after the full run.

The handoff's history rule did **not** pass. Across the 24 primary body-response channels, none had a positive 0-to-4-second conditional-variance reduction under the tested summary/neighbour estimator. Mean normalized conditional variance over all scored responses rose from 0.1336 at 0 ms to 0.2706 at 4 s for body/actuator matching; with filtered wheel observation it rose from 0.1310 to 0.2668. For the 2-second `delta_u` response, body/actuator conditional variance was 0.288 (run-bootstrap CI 0.206–0.384) at 0 ms, 0.246 (0.146–0.355) at 100 ms, and 0.498 (0.356–0.626) at 4 s. No history length was selected.

This is not proof that physical history has no predictive value: the summary representation is only a trailing mean and linear slope, and nearest-neighbour distances change dimension as history is added. Treat the result as “no history plateau supported by this estimator,” not as an irreducible physical uncertainty claim. The PCA diagnostic had 32 response dimensions and required 10 modes for 90% residual variance; the report's provisional `z=16` clue is diagnostic only and does not authorize training.

Important sparse/missing query regimes:

- `7_to_9mps_high_steering`: 2 independent runs.
- `low_speed_high_steering`: 2 independent runs.
- `negative_command_braking` and `braking_release`: no held-out query runs.
- `simultaneous_steering_throttle_transition`: 4 runs; high conditional dispersion for the 2-second forward-speed response (normalized variance 0.746).
- `large_wheel_body_mismatch`: 6 runs.

These counts are reported as weak evidence or unsupported, not interpreted as low uncertainty. In particular, braking and high-steering validation coverage need consideration before any whole-domain model claim.

Artifacts:

- Analyzer: [`analyze_blackbox_observability.py`](../../tools/vehicle_dynamics_learning/analyze_blackbox_observability.py)
- Machine report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/blackbox_observability_report_v1.json`
- Focused tests: [`test_analyze_blackbox_observability.py`](../../tools/vehicle_dynamics_learning/test_analyze_blackbox_observability.py)

## WP19 — target/state representation ablation

The fixed-budget trainer [`run_wp19_target_ablation.py`](../../tools/vehicle_dynamics_learning/run_wp19_target_ablation.py) compared all three body targets against all four wheel targets with one seed, one 300-step budget per candidate, hidden size 32, an 80-sample (2 s) history, and an 80-step (2 s) recursive training/evaluation horizon. The history length was held only to keep the target comparison controlled despite WP18's “no supported plateau” result. Training samples were drawn uniformly by independent run; train-only normalizers gave each run equal weight. Six OpenPlane validation runs (64 starts/run) and two unseen practice runs (64 starts/run) were evaluated independently. No test/final-test split was opened.

Inputs at each start were observable bridge odometry, actuator feedback, and commands. Simulator rigid-body truth supplied labels and the known rollout initial body state only. Future body/actuator/encoder observations were never fed into recursive prediction; future commands were the only known exogenous rollout sequence. The fixed-cadence encoder sidecars were aligned by sequence/packet arrays and verified against the SHA-256 of their actual parent datasets. A small 100-step preferred-W-D smoke completed in 4.19 s on CPU (PyTorch 2.7.1+cpu), 15.3k recursive transitions/s, 0 non-finite gradients, and 0.44 GiB peak RSS. Its 0.98 m/s 2-second forward-speed trajectory RMSE is a runtime/numerical smoke only, not a target-selection result.

The first complete run was superseded after audit found 213 of approximately 306k sampled training transitions where W-C/W-D had a valid next encoder label but W-A/W-B required both adjacent labels. The corrected v2 uses the same adjacent-valid transition set for all four wheel targets. The first run's JSON is explicitly marked superseded and must not be used for selection. Corrected primary report: [`wp19_target_ablation_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask/wp19_target_ablation_report.json). Detailed decision: [`WP19_TARGET_ABLATION_DECISION_20261003.md`](WP19_TARGET_ABLATION_DECISION_20261003.md).

On primary W-D, B-B achieved 2-second OpenPlane validation trajectory RMSEs of 0.680 m/s in `u` (run-bootstrap 95% CI 0.376–1.015), 0.163 m/s in `v` (0.045–0.387), and 0.616 rad/s in yaw rate (0.413–0.792), across six independent runs. Relative to B-A, the macro means were lower by 5%, 49%, and 13%, respectively; B-B won per-run in `u` on 3/6 runs, `v` on 6/6, and yaw rate on 4/6. The intervals overlap, so this is a diagnostic preference, not a statistically established plant-accuracy gain.

The two-run practice-transfer view cautions against a global claim: B-B's 2-second `u` and yaw trajectory errors were 1.483 m/s and 1.510 rad/s versus B-A's 1.323 m/s and 1.346 rad/s (about 12% worse); `v` improved from 0.147 to 0.079 m/s. B-C had lower recursive errors on these two practice runs but very poor one-step forward-speed errors (1.038 m/s on OpenPlane validation and 0.830 m/s on practice, versus B-B's 0.023 and 0.054 m/s) and worse OpenPlane validation `u` rollout. Keep B-C as a comparator; do not select it from two practice runs.

For the chosen B-B/W-D branch, encoder angle-increment RMSE was 0.540 rad/sample (six-run validation CI 0.509–0.578), equivalent to about 1.28 m/s under the canonical 25-ms/radius conversion. All candidates had finite recursive outputs and zero non-finite gradient steps, but finite rollouts are not evidence of accuracy. W-A and W-B, and separately W-C and W-D, are affine rescalings at fixed `dt` and wheel radius; standardized training made each pair nearly equivalent. W-D is retained because it is the direct measurement output required by the WP17 measurement-only decision and downstream encoder path, not because its score alone was superior.

Focused WP19 tests pass: rigid-body A/B/C representation equivalence, acceleration-consistency inversion, target-unit/encoder conversion and validity, command-step alignment, observable-only history, run-balanced normalization, and sequence/window isolation.

## WP20 — joint support calibration

The report [`wp20_support_calibration_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp20_support_calibration_v2_model_train_runs/wp20_support_calibration_report.json) compares only the handoff's two estimators on 384 windows from 6 independent OpenPlane validation runs; 128 windows from 2 practice runs are diagnostics. No future truth is included in support features and no test/final-test data were opened.

Run-balanced kNN passed the predeclared support-calibration gate, narrowly: run-cluster bootstrap 95% CI for per-run Spearman correlation between support distance and normalized 2-second body error was `[0.0037, 0.4273]`; 4/6 within-run correlations were positive; the high-minus-low support-quintile error difference CI was `[0.0104, 0.4309]`. Local Mahalanobis did not pass because its correlation CI crossed zero. The frozen kNN distance thresholds are `0.04170` (supported upper bound) and `0.16416` (weak-support upper bound); larger distances are unsupported. This is a composite body-error support signal, not a per-channel error bound or a general confidence guarantee. It is research-only pending the later validation gates.

## WP21 — replacement plant structure

[`augmented_state_space_plant.py`](../../tools/vehicle_dynamics_learning/augmented_state_space_plant.py) implements the separately testable causal state-space components in the handoff: GRU history encoder, explicit actuator transitions, stable nominal body transition, bounded latent/body residuals, WP19 W-D encoder measurement head, exact pose integration, turn-reflection helpers, and frozen WP20 support adapter. The recursive observable state is `[u, v, yaw_rate, steering_feedback, throttle_feedback]`; encoder values initialize causal history and are generated as measurements, not fed back as a physical body state. Future observations are not required by `step()`.

The architecture starts with latent dimension 8 and an 80-sample/2-second history carried as a fixed comparison setting; this does not resolve WP18's no-history-plateau finding. Focused structural tests cover the requested API, causality, transforms, actuator delays, measurement conventions, determinism, reset isolation, and checkpoint equivalence. All 17 WP21/WP23 plant tests passed.

## WP22 — one-seed recursive training and gate

[`train_augmented_state_space_plant.py`](../../tools/vehicle_dynamics_learning/train_augmented_state_space_plant.py) trained seed 101 only, from the frozen 12 whole-run training split, with run/family-balanced sampling and training-only scales/bounds. Stages A/B/C used 0.5/2/5-second recursive windows and 40 optimizer steps each (120 total); Stage C correctly excluded the two short full-input runs that lack 5-second windows. Six independent OpenPlane validation runs were primary; two practice runs were transfer diagnostics. No test/final-test split was opened, no future truth or feedback was supplied to recursive rollouts, and AutoDRIVE was not launched.

The saved candidate checkpoint is `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/seed101_candidate.pt` (SHA-256 `312a3146e0f8a9ec9e860bdd80dd24de5a80883efb23fcd9ac8c17c44abcbde2`). It is an unpromoted research artifact, not a usable offline simulator.

The preregistered primary 2-second radial-position trajectory RMSE was `2.212 m` for the candidate versus `2.055 m` for the WP19 parent: **7.6% worse**, with only 4/6 run wins and paired run-bootstrap 95% CI for parent-minus-candidate `[-0.457, 0.121] m`. The required improvement was at least 20%, with the interval favoring the candidate. At 2 seconds, OpenPlane `u` RMSE was `0.397` vs `0.376 m/s` (+5.5% error), yaw-rate RMSE `0.613` vs `0.548 rad/s` (+11.8%), and heading trajectory RMSE `0.470` vs `0.410 rad` (+14.8%). At 5 seconds, position RMSE was `12.663` vs `11.643 m` (+8.8%), `u` RMSE `0.878` vs `0.781 m/s` (+12.4%), and yaw-rate RMSE `0.907` vs `0.665 rad/s` (+36.3%).

On the two practice diagnostics, 2-second position RMSE rose from `1.246` to `1.584 m` (+27.1%), yaw-rate RMSE from `1.052` to `1.555 rad/s` (+47.8%); 5-second position rose from `8.008` to `11.947 m` (+49.2%), `u` from `1.187` to `1.465 m/s` (+23.4%), and yaw rate from `1.225` to `1.655 rad/s` (+35.1%). Practice sample size is only two runs, but both runs show the same direction of position regression.

One limited positive result: first crossing of `0.5 m` position error occurred later for the candidate on 4/6 OpenPlane runs; median crossing moved from `1.094 s` to `1.406 s`. This does not override the worse full-trajectory error, the mixed paired interval, or the practice regressions. The Level-1 material gate failed. The checkpoint was not promoted and the handoff forbids more seeds after this outcome.

## WP22 failure diagnosis and stop decision

The scoring-only analyzer [`diagnose_wp22_candidate_failure.py`](../../tools/vehicle_dynamics_learning/diagnose_wp22_candidate_failure.py) reused only the same frozen validation/practice captures; it performed no fitting and launched no simulator. At 250 ms, validation trajectory position errors were still close (candidate `0.0258 m`, parent `0.0243 m` in one representative run); substantial recursive degradation emerges later. During training, mean loss rose from `0.420` (0.5 s) to `3.019` (2 s) to `8.184` (5 s). Mean pre-clip gradient norm rose from `3.43` to `71.1` to `308.2` while the norm was clipped at 1. These figures point to a long-rollout optimization/state-consistency problem, but do not by themselves identify a unique cause.

The rollout's median WP20 kNN support distance across the six OpenPlane validation runs was `0.753–0.963`, far above the weak-support boundary `0.16416`; the two practice medians were `0.681` and `0.711`. Thus the candidate rapidly visits feature combinations outside the region in which WP20 calibrated its support/error relationship. The scoring-only support/error Spearman correlation was positive on 5/6 OpenPlane runs (macro-run mean `0.298`), but this reuses the same validation set and is diagnostic only—not independent support recalibration. Practice rollouts also show a yaw-rate suppression symptom: predicted maximum absolute yaw rate was about `0.16 rad/s` versus `0.97 rad/s` for the parent at 2 seconds, accompanying worse yaw and pose error. That is an observed failure signature, not yet an isolated mechanism.

These results triggered a return to state/target/support diagnosis. Per the handoff's WP23 section, only its minimal structural ablations were then run: A0 (no latent state) and A1 (latent state without nominal anchoring), compared with existing A2 (latent + stable nominal + residual). A3 was omitted because WP17 did not establish rear-wheel feedback as a necessary recursive physical state. The scoring-only WP22 report is [`wp22_failure_diagnosis.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/wp22_failure_diagnosis.json); full matched-run scores are in [`wp22_training_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/wp22_training_report.json).

## WP23 — minimal structural ablations

The runner [`run_wp23_structural_ablations.py`](../../tools/vehicle_dynamics_learning/run_wp23_structural_ablations.py) trained A0 and A1 once each with the same seed 101, 12-run training split, A/B/C horizons, 40/40/40 optimizer-step budget, validation windows, and practice diagnostics as A2. Its first launch stopped before any optimizer step because a new training-only bound calculation retained inconsistent per-run array dimensions; that shape issue was fixed, and the complete A0/A1 runs then finished. No test/final-test data were opened, no future truth/feedback entered rollouts, and the simulator was not launched. A2 is the existing WP22 seed-101 checkpoint; it was not retrained.

| Variant | OpenPlane 2 s position RMSE | OpenPlane 5 s position RMSE | Practice 2 s position RMSE | 0.5 m crossing median (10 s rollout) | Gate |
|---|---:|---:|---:|---:|---|
| A0: no latent | 2.401 m | 13.383 m | 1.314 m | 1.188 s | fail; 0/6 primary wins |
| A1: no stable nominal | 3.869 m | 9.803 m | 1.193 m | 0.250 s | fail; 1/6 primary wins |
| A2: latent + stable nominal | 2.212 m | 12.663 m | 1.584 m | 1.406 s | fail; 4/6 primary wins |
| WP19 parent | 2.055 m | 11.643 m | 1.246 m | 1.094 s | comparator |

A0's OpenPlane 2-second position error is 16.8% worse than the parent and it wins none of the six independent validation runs. A1's position is 88.3% worse at 2 seconds; first 0.5 m divergence is only 0.25 seconds (0/6 later than the parent). A1's lower practice pose error at 2/5 seconds does not offset its severe OpenPlane speed/trajectory failures; it is not a global plant. A2 is the strongest of these three on the primary OpenPlane metric, but remains 7.6% worse than the parent and regresses yaw/practice. No variant qualifies for WP24.

Machine report: [`wp23_structural_ablation_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp23_structural_ablations_seed101_v1/wp23_structural_ablation_report.json). Per-variant reports and retained checkpoints are in its sibling `A0/` and `A1/` directories. The decision record is [`WP22_AUGMENTED_STATE_SPACE_DECISION_20261003.md`](WP22_AUGMENTED_STATE_SPACE_DECISION_20261003.md).

## Verification and handoff status

The focused WP21/WP20/WP19 regression selection previously passed 30 tests. This status was extended by the post-2129427 handoff work below; it remains true that no plant is an accurate full-lap offline simulator or eligible for raceline/MPC optimization.

## Post-2129427 handoff — WP24 to WP26

WP24 froze source commit `2129427838101eee277e17e91727810bbe2df687`, refreshed the registry, and added explicit WP19/WP20/WP22/WP23 artifact and checkpoint-hash indexing. The exact WP22 validation/practice starts and comparator hashes are stored in `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/frozen_eval_starts.json` and `frozen_comparators.json`. Replaying A2 after the freeze reproduced 460 reported run/horizon/channel values exactly (maximum absolute difference 0; identical material-gate decision). After adding the WP26 model option, a second parity check matched 1,012 values exactly; default A2 behavior is unchanged.

WP25 performed scoring only: no simulator, no training, and no optimizer steps. Eight per-run archives record the 25 ms M0 internal rollout. The frozen A2 still has 2 s OpenPlane position/yaw/`u` errors of 2.212 m / 0.613 rad/s / 0.397 m/s and 5 s errors of 12.663 m / 0.907 rad/s / 0.878 m/s. Its latent transition disagrees with causal re-encoding by 5.92 training-standard-deviations per dimension after one step and about 13.6 by 250 ms. The support-loss gradient is negligible after short horizons; support escape is an early warning, not an established cause. The 99.88%-valid future-encoder oracle does not improve motion prediction. Full decision and limitations: [`WP25_RECURSIVE_FAILURE_LOCALIZATION_20261003.md`](WP25_RECURSIVE_FAILURE_LOCALIZATION_20261003.md).

WP26 trained only the fixed-budget D1/D2/D3 variants (seed 101; 40/40/40 updates at 0.5/2/5 s; identical sampler draw hash; same frozen runs and starts). All stayed finite and none passed the seven-part gate against A2:

| Variant | OpenPlane 2 s position RMSE | Change vs A2 | 2 s yaw RMSE | 2 s `u` RMSE | 0.5 m crossing later on | Gate |
|---|---:|---:|---:|---:|---:|---|
| A2 frozen | 2.212 m | — | 0.613 rad/s | 0.397 m/s | reference | fail |
| D1: support loss off | 2.212 m | 0.0% | 0.613 rad/s | 0.396 m/s | 0/6 | fail |
| D2: generated measurement removed from latent transition | 2.314 m | +4.6% | 0.626 rad/s | 0.307 m/s | 0/6 | fail |
| D3: both mechanisms off | 2.325 m | +5.1% | 0.630 rad/s | 0.308 m/s | 0/6 | fail |

D1 is effectively unchanged on the primary metric, consistent with the measured support gradient being tiny. D2/D3 reduce `u` error but worsen position and yaw; none delay the 0.5 m crossing on four runs. Practice, finiteness, and run-spread checks passed, but the primary position, divergence-time, and for D2/D3 the three-run improvement gates failed. Machine detail and per-variant packages are in [`wp26_mechanism_ablation_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/wp26_mechanism_ablation_report.json).

Per the handoff's explicit stop condition, no WP26 candidate qualifies for WP27. Do not continue training this A2 family, integrate any candidate, or use one for raceline/MPC optimization. The completed work is a stronger failure localization and a controlled rejection of these mechanisms—not a correct offline simulator. Resume only with a handoff-authorized simpler nonlinear transition formulation; do not repeat A0/A1 or add seeds.

## 2026-10-04 — multirate and no-latent high-steering follow-up

This follow-up addresses whether 1 kHz internal simulation with 40 Hz held
commands materially improves prediction, and whether the simpler no-autonomous-
latent acceleration model has a safe regime-specific gain. It is offline only;
no simulator, production odometry, MPC, or active container was changed.

### 1 kHz substeps: measured outcome

The controlled study is [`multirate_study_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/multirate_vector_field_v1/multirate_study_report.json). It trained a direct 40 Hz endpoint model and a continuous vector field on the same data/draws, held each command constant for one 25 ms control interval, and evaluated the ODE with 25 ms, 5 ms, and 1 ms internal steps. Only 40 Hz state labels exist; there are no measured intermediate 1 ms states.

On 2 s whole-run development validation, the continuous model's mean radial
position RMSE was 3.90032 m at 40 Hz and 3.90026 m at 1 kHz: a 0.000063 m
numerical difference. The direct 40 Hz model was 3.89092 m. The paired 1 kHz
minus direct-40-Hz position delta was +0.00934 m (run-cluster 95% CI
`[+0.00692,+0.01218]`); heading delta was +0.00338 rad
(`[+0.00168,+0.00510]`). On the two practice runs, 1 kHz was effectively
identical and slightly worse in position (+0.00023 m). Thus finer integration
did not improve this learned plant; its prediction error dwarfs the integration
resolution effect. This is evidence about the tested model/data, not a claim
that substeps can never help a correctly identified continuous-time plant.

The focused high-steering check reuses the same frozen field and compares its
40/200/1000 Hz internal rates on three independent 7–9 m/s,
`|steering| >= 0.30 rad` capture runs. At 2 s, position RMSE was 4.23405 m at
40 Hz and 4.23389 m at 1 kHz (only 0.000153 m better); heading RMSE changed
slightly worse, 1.42080 to 1.42083 rad. The 40 Hz model's absolute high-steer
errors remain very large. The numerical benefit is immaterial relative to the
dynamics-model error. Machine report:
[`highsteer_substep_diagnostic_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/multirate_vector_field_v1/highsteer_substep_diagnostic_20261004.json);
reproducible evaluator:
[`analyze_multirate_highsteer_substeps.py`](../../tools/vehicle_dynamics_learning/analyze_multirate_highsteer_substeps.py).

### A useful but narrow no-latent improvement

[`rigid_acceleration_highsteer_residual_gate_10s_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/rigid_acceleration_highsteer_residual_gate_10s_20261004.json)
compares a frozen rigid-acceleration model, its high-steering specialist, and a
smooth speed/steering-gated blend. The blend uses current predicted speed and
steering feedback only; it is not integrated into production.

On three 7.5 m/s high-steering development captures, 10 s position RMSE was
8.789 m for the frozen model, 6.733 m for the specialist, and 6.826 m for the
gated blend. All three runs improved; the paired gate-minus-parent run-cluster
95% interval was `[-2.463,-1.561] m`. At 5 s the gated score was 2.589 m vs
3.730 m; at 2 s, 0.467 m vs 0.808 m. However, the full specialist regressed
ordinary 10 s development position from 4.828 m to 8.379 m and practice from
2.362 m to 3.458 m. The gate avoided those regressions by remaining inactive
outside its narrow measured support; it leaves practice unchanged. This is a
real, localized gain, not a generally accurate offline simulator or a
validation on untouched runs.

Practice source captures divide laps into fragments. The evaluation joins
fragments only where run/reset/condition IDs match, rows and packet IDs are
consecutive, the cadence is 25 ms, and boundary labels are finite. Ten lap
boundaries met those physical-continuity checks; the resulting 10 s practice
score uses both complete unseen practice captures rather than stopping at a
lap-fragment boundary.

### Error attribution and current accuracy status

The oracle pose-integration diagnostic
[`truth_pose_integration_diagnostic_v1.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/truth_pose_integration_diagnostic_v1.json)
has only 0.052 m radial-position trajectory RMSE on the 5 s high-steering
windows and 0.069 m on practice when supplied true 40 Hz body motion. This
isolates the dominant long-horizon error to predicted body motion, not the pose
integrator or a need for 1 kHz pose steps. Teacher-forced one-step diagnostics
show nonlinear-region acceleration errors remain material; for example, on one
high-steering validation run the frozen model's `a_x`, `a_y`, and yaw-acceleration
RMSEs were 0.312 m/s², 0.763 m/s², and 3.36 rad/s² respectively. These are
one-run diagnostics, not population estimates.

**Status: high accuracy is not achieved.** The strongest verified change is
the gated high-steering improvement above. General 5–10 s motion prediction
still drifts by metres, and no candidate is accepted as an accurate full-lap
offline simulator. The A2 latent architecture remains stopped as required;
the next viable work is a genuinely different, no-autonomous-latent nonlinear
transition, judged on recursive whole-run results rather than timestep size or
teacher-forced fit. Existing high-steering and practice data used here are
development data, not an untouched final confirmation.

Coverage remains the bottleneck for that next step: the added high-steering
specialist has two independent 7.5 m/s training captures and three 7.5 m/s
development captures. This does not span a feasible 0–12 m/s speed/steering
surface. A broader high-steering specialist should wait for whole-run captures
at additional *feasible* speeds, with turn direction and transition phase
represented; there is no evidence here for extrapolating it across that full
surface. The existing `recursing_pasteur` container was still running the ROS
competition launch, bridge, odometry, EKF, and actuator when checked (37 hours
uptime). It was left untouched and no simulator was launched to avoid
interfering with that live stack.

### Held-out sensor-residual check: wheel-speed mismatch and roll

The frozen rigid-acceleration plant's teacher-forced 25 ms acceleration
residuals were tested for incremental explanatory value from rear-wheel
surface-speed/body-speed mismatch and IMU roll/roll-rate. The analysis uses
current-sample covariates only, whole-run cross-validation, and a bootstrap over
independent captures (not autocorrelated frames). The plant checkpoint is
unchanged. See
[`rigid_acceleration_sensor_residual_value_with_run_bootstrap_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/rigid_acceleration_sensor_residual_value_with_run_bootstrap_20261004.json)
and the reproducible diagnostic
[`diagnose_rigid_acceleration_sensor_residual_value.py`](../../tools/vehicle_dynamics_learning/diagnose_rigid_acceleration_sensor_residual_value.py).

On six independent open-plane dynamic validation captures, adding rear-wheel
mismatch reduced macro longitudinal-acceleration residual RMSE by only
0.0087 m/s² (run-cluster 95% CI `[-0.0109,-0.0067]`); lateral and yaw changes
were not distinguishable from zero. Adding roll and roll rate on top of wheel
mismatch reduced lateral residual RMSE by 0.0261 m/s² (95% CI
`[-0.0422,-0.0130]`, about 2.5%), but increased yaw-acceleration residual
RMSE by 0.0127 rad/s² (95% CI `[+0.0092,+0.0166]`, about 0.5%). Longitudinal
change was small and uncertain (`-0.0020 m/s²`, CI `[-0.0046,+0.0013]`). The
sign of the lateral and yaw changes was consistent across all six runs. This
is a genuine, limited predictive signal: roll carries information about
lateral acceleration error, but a naive shared correction trades that gain
against yaw.

The two unseen-practice captures show the same-direction lateral benefit
(`-0.0128 m/s²`, about 3.8%) and a small yaw benefit, but both runs show worse
longitudinal residuals (`+0.0091 m/s²`, about 1.5%). Two runs are not enough
for a dependable practice-domain uncertainty estimate; the bootstrap interval
there only resamples those two captures and must not be read as broad
generalization evidence. Rear-wheel mismatch alone improved practice
longitudinal residuals (`-0.0188 m/s²`) but barely changed lateral/yaw.

Neither measured wheel speed nor measured roll is a valid rollout input unless
the plant predicts that internal state itself. These are one-step,
teacher-forced residual diagnoses; they do not establish recursive improvement
or full-lap accuracy. The next targeted plant experiment is therefore to add
internally propagated roll/roll-rate, train roll dynamics only on training
runs, and compare recursive whole-run motion against the frozen parent. The
candidate must consume its own predicted roll after initialization and must
not read future IMU samples. Given the consistent yaw-residual penalty, roll
must enter the lateral-acceleration path selectively rather than be used as a
generic all-channel correction. Wheel-speed mismatch is a weaker, separate
longitudinal clue and will not be bundled into that first roll ablation.

The practice data currently available for this comparison contain only two
independent full-run captures, so final acceptance still needs additional
unseen whole runs. The high-steering holdout set has been used for development
comparisons and is no longer an untouched final confirmation set.

### 2026-10-04 — integration-rate check and yaw-refinement rejection

The proposed 40 Hz control / 1 kHz plant stepping was tested rather than
assumed. The captured plant labels are exactly 25 ms apart; there are no
measured 1 ms states to supervise intermediate motion. In the paired
40/200/1000 Hz continuous-vector-field study, raising the internal integration
rate did not materially change held-out motion prediction. On the difficult
high-steering windows, 2 s position RMSE changed from 4.234045 m at 40 Hz to
4.233892 m at 1 kHz (0.000153 m), while 300 training updates took 0.45 s for
the direct 40 Hz model and 4.33 s for the 1 kHz ODE model. This is numerical
resolution far below the multi-metre model error, not a useful accuracy gain.
Substeps can help only after the continuous vector field is right; substepping
the present 25 ms learned transition would invent unsupervised intermediate
states. The rate comparison and limits are recorded in
[`multirate_study_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/multirate_vector_field_v1/multirate_study_report.json)
and [`highsteer_substep_diagnostic_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/multirate_vector_field_v1/highsteer_substep_diagnostic_20261004.json).

A targeted yaw/pose-head refinement was then trained from the strongest
wheel/actuator teacher, with the shared encoder and other acceleration heads
frozen (514 trainable parameters, 1,200 updates). Its development-validation
selection score improved from 1.182 at initialization to 1.097, but the score
did not transfer. On the two independent whole-capture practice validations,
recursive position RMSE was 4.114 m and 3.751 m, and heading RMSE was 0.851
rad and 0.788 rad. The frozen parent on the same captures was substantially
better: position RMSE 1.167 m and 2.195 m, heading RMSE 0.145 rad and 0.126
rad. Parent isolated-lap position RMSE averaged 0.60 m and 0.45 m across those
captures; this is useful progress over the rejected refinement, but it is not
high accuracy.

The error breakdown explains the rejected trade: on both practice captures,
the parent's signed turn-window yaw-rate bias was about +0.022 rad/s in
negative-yaw turns and -0.075 rad/s in positive-yaw turns. The yaw-refined
candidate changed the negative-turn bias to about +0.06 rad/s while leaving
positive-turn bias near -0.075 rad/s. Its yaw-specific fit made one side
worse rather than correcting the repeatable directional imbalance. Full-run
position drift began within the first few seconds even though speed errors
were modest, and the 5 s high-steering position score also regressed. Keep the
parent as the best comparator; do not promote the yaw candidate or integrate
either into MPC/odometry.

Two older `unseen_practice` data views were checked for a broader independent
10 s comparison, but their capture is split into short branches (maximum
continuous segments 310 frames / 7.75 s and 244 frames / 6.10 s). They cannot
support the required 80-frame history plus 400-step recursive window, so the
scorer correctly rejected the attempted 10 s evaluation. They are not
whole-lap confirmation data and were not relabelled as such.

Candidate and replay artifacts are persistent under
[`history_context_sufficiency_v1`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/):
the yaw/pose checkpoint SHA-256 is
`432f604dc1200a991d5c362ff121a6d8d2e2182ea59e617603828fca8fc4672a`; the
independent practice replay is
[`effective_teacher_yaw_pose_practice_minus1_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/effective_teacher_yaw_pose_practice_minus1_20261004.json); the high-steering replay is
[`effective_teacher_yaw_pose_highsteer_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/effective_teacher_yaw_pose_highsteer_20261004.json).

**Status remains: no high-accuracy offline plant.** This continuation ran only
offline analyses/training; it did not launch or stop the simulator or alter
production behavior. The next evidence-led iteration is to diagnose the
repeatable signed-turn yaw residual on training-only sequences while matching
speed, steering magnitude/rate, and throttle, then test one small causal yaw
residual revision against both practice captures and all independent dynamic
runs. A new real-sim capture is still needed for untouched whole-lap
confirmation; do not interrupt the active controller publisher to gather it.

### 2026-10-04 — roll-path causal checks and signed-turn attribution

The current frozen full-practice comparator (checkpoint SHA-256
`8c3b50980e3f7d5d0fa1b946542672be25e032bf84f37d6ad13e5d8325cc2636`) already
has an internally propagated roll state and a learned roll-conditioned
acceleration residual. Before fitting another roll candidate, the existing
path was tested on the same frozen 5 s windows, with preceding-frame command
alignment and no model weights changed. The reproducible report is
[`roll_state_oracle.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/roll_state_oracle_causal_test_20261004/roll_state_oracle.json); implementation and focused tests are
[`diagnose_roll_state_oracle.py`](../../tools/vehicle_dynamics_learning/diagnose_roll_state_oracle.py)
and [`test_diagnose_roll_state_oracle.py`](../../tools/vehicle_dynamics_learning/test_diagnose_roll_state_oracle.py).

The causal baseline scored 131 valid dynamic starts from six captures and 135
practice starts from two captures. On practice, the ordinary recursive
5-second window scores were position `0.604 m`, heading `0.080 rad`, forward
speed `0.191 m/s`, yaw rate `0.111 rad/s`, and rear-wheel-pair speed `0.876
m/s`. These are window-level scores, not a full-lap claim. Turning off only
the trained roll-conditioned acceleration correction at inference worsened
position to `1.489 m` in practice and `3.030 m` on dynamic validation (from
`0.604 m` and `2.282 m`); heading also worsened to `0.232 rad` and `0.436
rad`. The dynamic position regression was consistent at run level (bootstrap
95% interval for relative improvement when disabling: `[-68.1%, -7.8%]`); both
practice captures regressed. This inference ablation is off the trained
pathway and is not a candidate, but it is evidence not to simply delete the
existing roll residual.

A separate, explicitly noncausal upper-bound arm replaced roll and roll rate
with measured IMU values at every source frame while all other states, latent,
and pose remained recursively predicted. On dynamic validation position
changed `2.282 -> 2.110 m` (7.5% apparent gain, run-cluster 95% interval
`[-7.8%, +18.4%]`, therefore uncertain); speed, lateral-speed, and wheel
errors worsened. On practice, position worsened `0.604 -> 0.734 m` (21.4%),
heading worsened `0.080 -> 0.118 rad`, and wheel-pair error worsened `0.876 ->
0.968 m/s`, even though speed RMSE improved `0.191 -> 0.172 m/s`. This is not
a deployable observer result and the two practice captures are a weak sample.
It shows that feeding measured roll into this model's learned correction is
not a reliable route to better path prediction; it does not establish that
roll physics are irrelevant. The oracle also leaves latent state untouched,
so it isolates the explicit roll-state path rather than every possible
roll/history interaction. Do not train/promote a roll revision from this
evidence alone.

The follow-up signed-turn analysis uses the frozen command-only recursive
rollouts and groups yaw-rate/acceleration errors post-hoc by measured steering
sign, speed, and rollout age. Results are in
[`signed_turn_yaw_error_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/signed_turn_yaw_error_20261004.json), with
[`analyze_signed_turn_yaw_error.py`](../../tools/vehicle_dynamics_learning/analyze_signed_turn_yaw_error.py)
and [`test_analyze_signed_turn_yaw_error.py`](../../tools/vehicle_dynamics_learning/test_analyze_signed_turn_yaw_error.py).
At 5 s on practice, yaw-rate bias was `+0.023 rad/s` during left steering
(two independent captures), `-0.079 rad/s` during right steering (only one
capture has eligible windows), and `-0.012 rad/s` near center (two captures).
Dynamic direction-conditioned errors change with rollout age and speed; the
high-speed turn cells often have only one or two independent runs. This
supports a turn-direction/phase diagnosis, but does not provide enough
independent support to fit an asymmetric yaw correction. The one-head yaw fit
already failed whole-capture transfer, so it is not repeated.

**Conclusion:** 40 Hz control with 1 kHz substeps does not improve accuracy
for the available 40 Hz-labelled plant; the measured high-steering position
change was only `0.000153 m` against `4.234 m` RMSE. The existing roll residual
should remain frozen: ablating it is worse, and measured-roll injection
regresses practice position/heading. No candidate passed a held-out transfer
gate, and full-capture position errors remain `1.167 m` and `2.195 m` over the
two five-lap practice captures (with isolated-lap scores around `0.60 m` and
`0.45 m`). The offline plant is **not high accuracy** and must not be used to
claim a reliable full-lap simulation or tune a raceline. Next useful work is
not another substep-rate or generic roll experiment: increase independent
practice-capture coverage for both turn directions, then test a causal yaw
model against whole-run dynamic and practice data with its signed bias,
speed, wheel and position metrics all gated separately. No active simulator,
production odometry/MPC, or physics were modified in this continuation.
Focused math/alignment regression checks for these diagnostics pass (10 tests);
both analysis scripts run against the persistent `apex-plant-eval` environment.

### 2026-10-04 — whole-practice test of explicit rigid-acceleration candidates

The 5 s and 10 s direct-acceleration checkpoints were replayed on both complete
held-out practice captures, alongside the selected WP28 direct-state parent.
Each run used one measured 2 s history/state/pose initialization; after that,
the only inputs were recorded commands and recursively predicted state/history.
Separate isolated-lap replays reinitialized from the measured state and prior
history at each lap boundary. Future state/pose were scoring labels only. The
report is
[`rigid_acceleration_history_direct_supervision_5s_vs_10s_full_practice_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/rigid_acceleration_history_direct_supervision_10s_v1/rigid_acceleration_history_direct_supervision_5s_vs_10s_full_practice_20261004.json)
and the evaluator is
[`evaluate_rigid_acceleration_full_practice.py`](../../tools/vehicle_dynamics_learning/evaluate_rigid_acceleration_full_practice.py).

The acceleration-integrated models did **not** pass the practice-transfer test.
Their uninterrupted 35.9 s position RMSEs were 5.79/7.96 m (5 s checkpoint)
and 6.07/6.59 m (10 s checkpoint) on runs r02/r03, versus 3.68/3.22 m for
WP28. Isolated-lap position RMSEs were 1.66/2.06 m and 3.17/3.59 m,
respectively, versus WP28's 1.26/1.48 m. The 10 s fine-tune improved the 5 s
checkpoint on one continuous capture but made isolated-lap errors worse on
both; it is not a general improvement. Neither model predicts wheel speed, so
neither can be called a complete vehicle plant.

One-step teacher-forced practice diagnostics explain why short transition
scores were misleading. The 10 s model's per-step position RMSE was about
3.2–3.7 mm and forward-speed RMSE 0.034–0.035 m/s, but its free-running
practice state developed about −0.54 m/s forward-speed bias and −0.25 m/s
lateral-speed bias. Position error first reached 0.5 m at 0.925 s, about the
same as WP28 (0.900 s), then grew far more over the full capture. Thus its
local transition fit is not evidence of stable recursive prediction. The
candidate's recursive speed/lateral/yaw errors appear across both steering
signs, 3–6 and 6–9 m/s, and low as well as high wheel/body mismatch; available
data do not isolate a single high-slip or throttle-slew cause. High throttle
slew had only 41–44 scored samples per run. Full regime results and the
one-step comparison are in
[`full_practice_one_step_and_recursive_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/rigid_acceleration_history_direct_supervision_10s_v1/full_practice_one_step_and_recursive_20261004.json)
and [`full_practice_capture_regime_diagnostics_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/rigid_acceleration_history_direct_supervision_10s_v1/full_practice_capture_regime_diagnostics_20261004.json).

The best existing full-state effective teacher remains materially better on
these same two captures: continuous position RMSE 1.17 m and 2.19 m; isolated
lap means 0.60 m and 0.44 m. Its forward-speed RMSE is about 0.20 m/s, lateral
speed 0.017 m/s, yaw rate 0.105–0.106 rad/s, and rear-wheel-pair RMSE about
0.83 m/s. This is useful progress, but not negligible full-lap error; wheel
prediction remains a substantial weakness. Its separate three-run 7–9 m/s,
`|steering| >= 0.30 rad` holdout still has 5 s position radial RMSE 1.73 m,
heading RMSE 0.127 rad, rear-left wheel RMSE 2.08 m/s, and rear-right wheel
RMSE 3.14 m/s (see
[`effective_teacher_highsteer_transfer_20261004.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/roll_coupled_multistep_candidate_v2_20261004/effective_teacher_highsteer_transfer_20261004.json)).

Coverage caveat: neither practice capture reaches 0.30 rad steering; maximum
absolute feedback steering is 0.2827 rad in each. They therefore say nothing
about full-steering lap accuracy. The dedicated open-plane high-steering holdout
is the relevant evidence for that region, and its current full-state errors
remain large. The practice regime report also contains no adequately sampled
high-throttle-slew test.

A proposed feature-semantics explanation was checked and falsified for these
Explore captures: the first three recorded odometry channels match the aligned
simulator body labels to floating-point precision in the loaded train,
validation, and practice runs. The observed recursive gap is therefore not
caused by an odometry-vs-truth mismatch in this dataset; adding a separate
observation model on that premise is not justified.

**Decision:** retain the existing effective teacher as the strongest full-lap
comparator; do not promote either explicit rigid-acceleration checkpoint and
do not tune the raceline with them. The 1 kHz substep decision is unchanged:
the observed numerical difference is negligible relative to model error and
there are no 1 ms labels. The next model work must directly address the
current teacher's high-steering body/wheel errors and recursive state drift,
using training runs only and then whole-run dynamic plus practice validation.
No new model has yet achieved high accuracy, and no production code or
simulator physics was changed.
