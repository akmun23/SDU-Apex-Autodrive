# Black-box offline plant progress — 2026-10-03

Scope is limited to the ordered work packages in `SDU_APEX_BLACK_BOX_OFFLINE_PLANT_HANDOFF_2026-10-03.md`. This is an evidence record, not a model promotion. No simulator physics, production odometry, or MPC behavior has been changed. No new plant has been trained or integrated.

## Current status

WP15–WP23 are implemented and evaluated. WP19 freezes body-state increment plus explicit acceleration consistency (B-B), and direct fixed-25-ms encoder angle increment (W-D), as the representation carried into replacement-plant research. WP20 narrowly calibrated run-balanced kNN support; WP21 passed its structural tests; the single WP22 seed and both authorized WP23 structural ablations failed the material-improvement gate. No candidate is promoted, the offline simulator is not ready, and WP24 onward is blocked. WP18 did not support a history-length plateau under its tested run-balanced neighbor estimator, so the 2-second history used here remains a controlled setting, not a claim of sufficiency. No production odometry, MPC, simulator physics, or active simulator process was changed.

## WP15 — experiment and provenance registry

The registry indexes existing training metadata, comparison artifacts, dataset/checkpoint hashes, whole-run splits, and unresolved document references:

- Script: [`build_experiment_registry.py`](../../tools/vehicle_dynamics_learning/build_experiment_registry.py)
- Registry: `live_runs/derived_dynamics_learning_20260928/experiment_registry_20261003.json`
- Human summary: [`OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md`](OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md)

The refreshed registry finds 112 training metadata files, 93 model runs, 47 diagnostic artifacts, and verifies 48 checkpoint hashes. It records 129 document artifact references (2 unresolved literal paths and 1 unmatched historical glob) and no diagnostic hash mismatches. It includes the WP16–WP18 reports and does not promote a default plant.

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

The saved candidate checkpoint is `live_runs/.../wp22_augmented_state_space_seed101_smoke_v1/seed101_candidate.pt` (SHA-256 `312a3146e0f8a9ec9e860bdd80dd24de5a80883efb23fcd9ac8c17c44abcbde2`). It is an unpromoted research artifact, not a usable offline simulator.

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

The focused WP21/WP20/WP19 regression selection passes: 30 tests, including A0/A1 variant math and state-shape checks. The completed work is a supported diagnosis and tested research architectures, **not** an accurate full-lap offline simulator. Current candidate 2-second errors remain far outside the handoff's engineering targets, and no plant is eligible for raceline or MPC optimization. WP23's no-variant-passed gate returns the project to state/target/support diagnosis; do not train more seeds or run WP24–WP30 from these branches.
