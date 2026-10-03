# Offline Plant Experiment Registry — 2026-10-03

Repository HEAD: `d4312323d183176674898fa01bea5cd12c416b9b`. Worktree was dirty at generation: `True`.

This is an evidence index, not a model promotion. The registry has no default candidate because no plant has passed the required recursive and task-level gates.

## Coverage

- Training metadata files found: 112; indexed model runs: 93. 
- Indexed diagnostic/comparison artifacts: 47. 
- `live_runs/` references in development documents: 129; missing literal paths: 2; unmatched globs: 1.
- Checkpoint hashes checked: 48; unresolved dataset/checkpoint references: 0/0; recorded diagnostic hash mismatches: 0.

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
- **serious_replacement_training:** not authorized until WP19 and WP20 close.

## Current interpretation

The raw-wheel, raw-history, wheel-innovation, and contact-slip branches do not provide a universally better recursively coherent model. Contact-slip is rejected. Raw encoder state/history are mixed tradeoffs; neither is accepted as a production state. The fixed-40-Hz view is a separate, causally aligned measurement representation and must not be conflated with the source-stamp-derived sidecar. The two-expert candidate improved some dynamic pose scores while degrading other state channels and practice transfer. The frozen EDSSM parent remains a comparator, not a usable offline plant.

WP17 is complete: the fixed-25-ms and stored-filtered wheel-rate oracles predominantly regress the frozen parent on held-out dynamic recursive endpoints, so processed rear-wheel rate is measurement/output only. This does not reject an internally inferred latent traction state.

WP18 is complete with no supported history-length plateau under the tested run-balanced neighbor estimator. This is not evidence that physical history is useless: the tested trailing-mean/slope summaries and changing match dimensions limit that claim. Serious replacement training remains unauthorized until WP19 target representation and WP20 calibrated support gates close.

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

- Missing literal reference `live_runs/derived_dynamics_learning_20260928/model_eval_venv` from docs/development/OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md, docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md; it is not a present indexed checkpoint/dataset/comparison artifact.
- Missing literal reference `live_runs/vehicle_model_training_env` from docs/development/OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md, docs/development/VEHICLE_DYNAMICS_DATA_CATALOG_20260930.md; it is not a present indexed checkpoint/dataset/comparison artifact.
- Unmatched historical glob `live_runs/openplane_throttle_slew_holdout_{45,65}_rNN/run/run_0.db3` from docs/development/NONLINEAR_MODEL_DISCOVERY_UPDATE_20260928.md, docs/development/OFFLINE_PLANT_EXPERIMENT_REGISTRY_20261003.md.

The historical `model_eval_venv` and `vehicle_model_training_env` references are absent local directories, not missing training outputs. Wildcard/brace references are expanded and counted separately; they are not silently reported as resolved files.
