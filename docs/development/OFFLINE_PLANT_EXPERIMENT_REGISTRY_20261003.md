# Offline Plant Experiment Registry — 2026-10-03

Repository HEAD: `5b26c5ac829acf56d77bdb71a979b7e494703ac5`. Worktree was dirty at generation: `True`.

This is an evidence index, not a model promotion. The registry has no default candidate because no plant has passed the required recursive and task-level gates.

## Coverage

- Training metadata files found: 135; indexed model runs: 101.
- Indexed diagnostic/comparison artifacts: 51.
- `live_runs/` references in development documents: 152; missing literal paths: 3; unmatched globs: 1.
- Checkpoint hashes checked: 60; unresolved dataset/checkpoint references: 0/0; recorded diagnostic hash mismatches: 0.

## Current branch decisions

- **plain_gru:** preserved as comparator; no evidence of a validated recursive full-lap plant.
- **high_steering_gru:** preserved as comparator; regime gain did not establish whole-run plant accuracy.
- **raw_encoder_state:** diagnostic; mixed domain tradeoff; not promoted.
- **raw_encoder_history:** diagnostic; not a consistent global win; not promoted.
- **wheel_innovation_history:** diagnostic; not promoted.
- **contact_slip:** rejected on paired recursive dynamic and practice comparisons.
- **fixed_25ms_encoder:** separate causal measurement view; WP16 target audit complete and WP17 classifies measured processed wheel rate as measurement/output only.
- **current_parent_edssm:** frozen comparator only; recursive drift remains unacceptable.
- **moe_candidate:** rejected for plant use: pose gains accompanied by wheel/speed and practice-transfer regressions.
- **four_wheel_greybox:** rejected as race-domain nominal plant; prior grey-box relaxation did not remove recursive mismatch.
- **direct_transformer_and_direct_predictors:** diagnostic or incomplete; no accepted autonomous multi-step plant.
- **reflection_symmetry:** historical diagnostic branch only; no promoted whole-run result.
- **roll_conditioning:** not promoted; prior held-out practice evidence did not show material explanatory gain.
- **sensor_observer_gru:** separate sensor-only observer, not a plant; no full-pose drift acceptance.
- **wp14_blind_final:** already consumed for development interpretation; prohibited for further tuning.
- **wp16_signal_and_wheel_audit:** complete.
- **wp17_current_parent_causal_intervention:** complete; rear processed wheel rate is B: measurement/output only.
- **wp18_history_sufficiency:** no supported history plateau; retain no fixed-history claim.
- **serious_replacement_training:** WP19/WP20 closed; WP22/A0/A1 and WP26 D1/D2/D3 remain unpromoted; the A2 family stop gate was reached after WP26.

## Active black-box work-package artifacts

- WP19 selected checkpoint: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask/07_body_state_increment__encoder_angle_increment/model.pt` — SHA-256 `045e98e1d0aafd9ec369bc6fdb1bdbce4e94491dec101a764f33195fe09e7541`.
- WP19 report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask/wp19_target_ablation_report.json` — SHA-256 `4b8adfa6ee055a744d5fa1c68411870f5bd9ac662eff48d32e7d31a2bcb654c4`.
- WP20 report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp20_support_calibration_v2_model_train_runs/wp20_support_calibration_report.json` — SHA-256 `469ff17bf6039b9eb54f05a437370c0fa39a5e35bd5bfaa8085559a17146c03c`.
- WP22 A2 checkpoint: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/seed101_candidate.pt` — SHA-256 `312a3146e0f8a9ec9e860bdd80dd24de5a80883efb23fcd9ac8c17c44abcbde2`.
- WP22 training report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/wp22_training_report.json` — SHA-256 `a37ee70aed3c46acc4788616e9edb8a2be67785f4bec4c17a74141d2c4302f41`.
- WP22 failure diagnosis: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/wp22_failure_diagnosis.json` — SHA-256 `5248b3e9ecfef51a911c8599c6bb71812f3648f7d1a19a0b434fe2f7c9e09433`.
- WP23 A0 checkpoint: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp23_structural_ablations_seed101_v1/A0/seed101_A0_candidate.pt` — SHA-256 `eb3ccacc81566c1b4e13608ce379dd8a388e865b2cefc49ed2f08a958adb5e40`.
- WP23 A0 training report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp23_structural_ablations_seed101_v1/A0/wp23_A0_training_report.json` — SHA-256 `8fc53f337cf35228eaa8f559098a574bef936115d7489927015b1162cab6e2ad`.
- WP23 A1 checkpoint: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp23_structural_ablations_seed101_v1/A1/seed101_A1_candidate.pt` — SHA-256 `3c606e50432dc1aa834c9b9bb5ac503bdff82330d6728570df1255753429688a`.
- WP23 A1 training report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp23_structural_ablations_seed101_v1/A1/wp23_A1_training_report.json` — SHA-256 `a0c54c6179b07cbaa5df528358c8d136162f793fd297aecba462026d778bd862`.
- WP23 ablation report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp23_structural_ablations_seed101_v1/wp23_structural_ablation_report.json` — SHA-256 `dc5aefc757d46bec1ddd1a971e83912dd46a368ff260ce9f82a19308c3256a35`.
- WP24 frozen comparators: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/frozen_comparators.json` — SHA-256 `bb11de2b29c3d0285e89b915e469ef150568ff3196321b19740ecb7931569ed6`.
- WP24 frozen evaluation starts: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/frozen_eval_starts.json` — SHA-256 `d01d18b14ffa6d16536638db5400f8ee9049bb671ea6c6ad23e009944248defe`.
- WP24 A2 reproduction: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp25_failure_localization/wp24_reproduction_report.json` — SHA-256 `dedd395563b1d0bfc2d5bf0d3b6b7866e61ce714f4c9161efa34e864a56aaf3f`.
- WP25 scoring report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp25_failure_localization/wp25_score_only_diagnostics.json` — SHA-256 `be012433bfeb3aca46489b99cc9d6c72430f23a99cf15e4b01c7e11d4eb1f4db`.
- WP25 loss-gradient attribution: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp25_failure_localization/wp25_loss_gradient_attribution.json` — SHA-256 `18777845d0e00df21ff7ce5b2ddae674a661ab614f6948a01c07258682f850bf`.
- WP26 mechanism report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/wp26_mechanism_ablation_report.json` — SHA-256 `0c9a47939539f5a7425357d37ba46921a804052b083d58ad95fda4e1f364b950`.
- WP26 D1 checkpoint: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/D1/checkpoint.pt` — SHA-256 `8135ae7fe275f552a044a4eee0d8d6f9aa09dee2138f927afa97c0832ced4602`.
- WP26 D1 training report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/D1/training_report.json` — SHA-256 `88a6382c43d31f666a98e9e1e94d181a00b428138fb587fa2a073278a37e3b9c`.
- WP26 D2 checkpoint: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/D2/checkpoint.pt` — SHA-256 `3ac41218918cc7aeb78f8e02287772eef96ee8ae207b81fee3864741dbe6b19e`.
- WP26 D2 training report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/D2/training_report.json` — SHA-256 `b8e99fd8354088a01c7d0a211da771e74d5fafd07bf67cceb7f30d21460ef554`.
- WP26 D3 checkpoint: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/D3/checkpoint.pt` — SHA-256 `390b2342be1106f93998026d1d40ca47e5b6bfbc6747b224792b2bad1c1c049c`.
- WP26 D3 training report: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/D3/training_report.json` — SHA-256 `0544143163880d8d37e4fb71c0ffcd473f3db3bd129fc096f2d898a38a967916`.

## Current interpretation

The raw-wheel, raw-history, wheel-innovation, and contact-slip branches do not provide a universally better recursively coherent model. Contact-slip is rejected. Raw encoder state/history are mixed tradeoffs; neither is accepted as a production state. The fixed-40-Hz view is a separate, causally aligned measurement representation and must not be conflated with the source-stamp-derived sidecar. The two-expert candidate improved some dynamic pose scores while degrading other state channels and practice transfer. The frozen EDSSM parent remains a comparator, not a usable offline plant.

WP17 is complete: the fixed-25-ms and stored-filtered wheel-rate oracles predominantly regress the frozen parent on held-out dynamic recursive endpoints, so processed rear-wheel rate is measurement/output only. This does not reject an internally inferred latent traction state.

WP18 is complete with no supported history-length plateau under the tested run-balanced neighbor estimator. This is not evidence that physical history is useless: the tested trailing-mean/slope summaries and changing match dimensions limit that claim. WP19 and WP20 are closed; WP22/A0/A1 and WP26 D1/D2/D3 are indexed below but none passed their material gates. The A2 family stop gate is reached; conditional WP27 is not run.

## Reading the machine registry

`live_runs/derived_dynamics_learning_20260928/experiment_registry_20261003.json` contains per-run metadata, split IDs, checkpoint/dataset SHA-256 values when resolved, all discovered comparison-artifact references, and missing metadata explicitly. Experiments default to `diagnostic`, `incomplete`, `rejected`, or `unknown`; none are promoted by filename or a single score.

Older evidence is retained as historical context. Where it conflicts with the 2026-10-02 replacement-plant report or the 2026-10-03 handoff, the later whole-run recursive validation and current handoff's gates take precedence; no earlier local/conditional yaw result is treated as a validated black-box plant.

## Reviewed source documents

- `docs/development/REPLACEMENT_OFFLINE_SIM_RACELINE_PROGRESS_20261002.md` (present): Current baseline and whole-run recursive metrics; authoritative current plant status.
- `docs/development/MASTER_VEHICLE_MODEL_ODOM_MPC_PROGRESS_20261001.md` (present): Historical production model/odom record; not evidence that an offline plant passed recursive validation.
- `docs/development/FULL_MODELING_RESET_PROGRESS_20261001.md` (present): Earlier modeling reset and experiment history; superseded where later whole-run results conflict.
- `docs/development/EXTERNAL_REVIEW_OFFLINE_PLANT_HANDOFF_20261001.md` (present): External review evidence and caveats; retained as historical review, not a current model promotion.
- `docs/development/RIGID_BODY_TEACHER_PROGRESS_20261001.md` (present): Earlier rigid-body teacher experiments; no accepted replacement plant established.
- `docs/development/NONLINEAR_MODEL_DISCOVERY_UPDATE_20260928.md` (present): Earlier throttle-slew/roll/GRU tests; weak or negative evidence, not a general plant.
- `docs/development/OPEN_PLANE_PER_WHEEL_DYNAMICS.md` (present): Historical narrow per-wheel/yaw analyses; proxies do not identify tire-force truth or a full plant.
- `docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md` (present): Historical data inventory; its schemas/counts predate the replacement dataset and are not current totals.
- `docs/development/OPENPLANE_THROTTLE_SURFACE_RESULTS_20260930.md` (present): Throttle response-surface coverage; useful input-response data, not recursive whole-vehicle validation.
- `docs/development/OFFLINE_PLANT_NEXT_STEPS_20260930.md` (present): Earlier proposed sequence; superseded by the 2026-10-03 black-box handoff order.
- `docs/development/OFFLINE_PLANT_TRAINING_UPDATE_20260930.md` (present): Historical training report; later recursive validation controls current conclusions.
- `docs/development/SENSOR_OBSERVER_PROGRESS_20260929.md` (present): Sensor-only observer is a separate estimator, not the offline plant; prior GRU is not a full pose/drift observer.
- `docs/development/ENGINEERING_STATE.md` (present): Broad historical workspace state; use the dated plant report and handoff for current scope/status.
- `docs/development/BLACK_BOX_OFFLINE_PLANT_PROGRESS_20261003.md` (present): Current WP15–WP18 evidence and gates for the black-box offline plant handoff.

## Referenced artifacts not present

- Missing literal reference `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1/rigid_acceleration_history_direct_supervision_10s_v1/rigid_acceleration_history_direct_supervision_5s_vs_10s_full_practice_20261004.json` from docs/development/BLACK_BOX_OFFLINE_PLANT_PROGRESS_20261003.md; it is not a present indexed checkpoint/dataset/comparison artifact.
- Missing literal reference `live_runs/derived_dynamics_learning_20260928/model_eval_venv` from docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md; it is not a present indexed checkpoint/dataset/comparison artifact.
- Missing literal reference `live_runs/vehicle_model_training_env` from docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md; it is not a present indexed checkpoint/dataset/comparison artifact.
- Unmatched historical glob `live_runs/openplane_throttle_slew_holdout_{45,65}_rNN/run/run_0.db3` from docs/development/NONLINEAR_MODEL_DISCOVERY_UPDATE_20260928.md.

The historical `model_eval_venv` and `vehicle_model_training_env` references are absent local directories, not missing training outputs. Wildcard/brace references are expanded and counted separately; they are not silently reported as resolved files.
