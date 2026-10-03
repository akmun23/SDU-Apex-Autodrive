# WP22/WP23 augmented state-space decision — 2026-10-03

## Decision

The seed-101 candidate **fails the handoff's WP22 material-improvement gate**. It is not promoted and must not be used as an offline simulator, production odometry/MPC model, raceline model, or weight-search model. Follow the handoff failure path: stop this candidate's seed/training branch and return to state/target/support diagnosis. The handoff's bounded WP23 structural diagnosis was then run; no variant qualified to continue.

This is a complete WP21/WP22 gate decision, not completion of the larger offline-simulator objective. The existing WP19 parent remains the comparator; neither model meets the handoff's accuracy targets.

## Evidence and protocol

- The WP20-calibrated support estimator is run-balanced kNN over the seven observable operating features. It narrowly passed on six independent OpenPlane validation runs: normalized-body-error Spearman run-bootstrap 95% CI `[0.00365, 0.42733]`, positive within-run correlation on 4/6 runs, and high-minus-low distance quintile error-difference CI `[0.01044, 0.43094]`. Local Mahalanobis did not pass. Thresholds frozen for this research stage: supported distance `<=0.0417018`; weak support `>0.0417018` through `0.1641602`; unsupported `>0.1641602`. This support score is a composite 2-second body-error indicator, not a per-channel confidence interval.
- WP21 architecture is isolated in [`augmented_state_space_plant.py`](../../tools/vehicle_dynamics_learning/augmented_state_space_plant.py). It uses the WP19 B-B/W-D target choice, observable `[u,v,r,steering_feedback,throttle_feedback]`, a causal 2-second history encoder, latent dimension 8, explicit actuator transitions, bounded learned residuals, a measurement head, exact pose integration, and the frozen WP20 adapter. No production files were changed.
- All 15 focused WP21 structural tests passed. The combined focused WP21/WP20/WP19 selection passes 28 tests.
- WP22 trained one seed (101), only on the frozen 12 whole-run training split. A/B/C used 0.5/2/5-second recursive windows with 40 optimizer updates per stage. Six OpenPlane whole runs were validation; two practice whole runs were transfer diagnostics. The two short full-input runs had no complete 5-second training windows and were excluded from Stage C only. No test/final-test capture was opened, no future measured truth/feedback entered rollouts, and no simulator was launched.
- Frozen candidate: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/seed101_candidate.pt`, SHA-256 `312a3146e0f8a9ec9e860bdd80dd24de5a80883efb23fcd9ac8c17c44abcbde2`.

## Matched recursive scores

Errors below are run-macro means over the stated independent runs. Lower is better.

| Split / horizon | Metric | WP19 parent | Candidate | Candidate error change |
|---|---|---:|---:|---:|
| OpenPlane, 2 s (6 runs) | radial-position trajectory RMSE | 2.055 m | 2.212 m | +7.6% |
| OpenPlane, 2 s | forward-speed RMSE | 0.376 m/s | 0.397 m/s | +5.5% |
| OpenPlane, 2 s | yaw-rate RMSE | 0.548 rad/s | 0.613 rad/s | +11.8% |
| OpenPlane, 2 s | heading trajectory RMSE | 0.410 rad | 0.470 rad | +14.8% |
| OpenPlane, 5 s (6 runs) | radial-position trajectory RMSE | 11.643 m | 12.663 m | +8.8% |
| OpenPlane, 5 s | forward-speed RMSE | 0.781 m/s | 0.878 m/s | +12.4% |
| OpenPlane, 5 s | yaw-rate RMSE | 0.665 rad/s | 0.907 rad/s | +36.3% |
| Practice, 2 s (2 runs) | radial-position trajectory RMSE | 1.246 m | 1.584 m | +27.1% |
| Practice, 2 s | forward-speed RMSE | 0.832 m/s | 0.810 m/s | −2.6% |
| Practice, 2 s | yaw-rate RMSE | 1.052 rad/s | 1.555 rad/s | +47.8% |
| Practice, 5 s (2 runs) | radial-position trajectory RMSE | 8.008 m | 11.947 m | +49.2% |
| Practice, 5 s | forward-speed RMSE | 1.187 m/s | 1.465 m/s | +23.4% |
| Practice, 5 s | yaw-rate RMSE | 1.225 rad/s | 1.655 rad/s | +35.1% |

The primary pre-registered gate was 2-second OpenPlane radial-position trajectory RMSE. Parent-minus-candidate paired mean was `−0.15695 m` (so the candidate was worse); the run-bootstrap 95% interval was `[−0.45666, 0.12130] m`. The candidate won on 4/6 runs but did not achieve the required 20% gain, and the interval did not favor it. The required no->10% systematic regression condition also failed. Practice sample size is only two, but both runs show the same direction of position loss.

The 2-second candidate errors are also nowhere near the handoff's research targets: position `<=0.30 m`, heading `<=0.05 rad`, forward speed `<=0.15 m/s`, and yaw rate `<=0.15 rad/s` (the `0.30 m` position threshold is the maximum research gate, not the preferred `0.20 m`). The candidate's corresponding OpenPlane run-macro values are `2.212 m`, `0.470 rad`, `0.397 m/s`, and `0.613 rad/s`.

One limited positive signal does not change the decision: the first crossing of 0.5 m position error is later on 4/6 OpenPlane runs; the median changes from `1.09375 s` (parent) to `1.40625 s` (candidate). The paired total trajectory remains worse. These divergence values are now included in the machine report's gate object.

## Failure diagnosis

The scoring-only report [`wp22_failure_diagnosis.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/wp22_failure_diagnosis.json) reused the frozen validation/practice starts and did not train. It provides clues, not a new independent evaluation:

1. **The short transition is close; recursive error grows.** In one representative OpenPlane run, 250 ms position trajectory RMSE is `0.0258 m` candidate vs `0.0243 m` parent. At 2 seconds the corresponding run is `4.057 m` vs `3.309 m`; errors have already separated materially.
2. **The rollout quickly leaves calibrated joint support.** Median candidate support distances over the first 2 seconds range from `0.753` to `0.963` across the six OpenPlane validation runs, and `0.681–0.711` on the two practice runs. All are well above the WP20 weak-support boundary `0.16416`; support confidence is therefore zero through most such predictions. The initial support/error correlation on these same validation windows is positive on 5/6 runs (macro-run mean `0.298`), but it is not independent recalibration and cannot authorize confidence outside the frozen calibration domain.
3. **Longer recursive training becomes difficult.** Mean training loss rises from `0.420` at Stage A to `3.019` at B and `8.184` at C. Mean pre-clip gradient norm rises from `3.43` to `71.1` to `308.2`, with clipping at norm 1. This is consistent with difficult long-rollout optimization, but does not prove whether the primary cause is state definition, latent reconstruction, measurement coupling, rollout distribution shift, or a combination.
4. **Practice yaw prediction is suppressed.** On practice 2-second starts, the candidate's maximum predicted absolute yaw rate is about `0.162 rad/s` versus `0.975 rad/s` for the parent; candidate yaw and position errors are substantially worse. This is a concrete failure signature, not an isolated causal explanation.
5. **Encoder measurement prediction remains poor in difficult regimes.** The scoring-only region report shows substantially higher candidate encoder-increment error than parent in several high-speed/transition slices. This supports retaining the measurement head as a separate diagnostic channel; it does not justify adding encoder measurements back into recursive body state.

Therefore the evidence locates the failure mainly in the recursive transition/state consistency and support escape, while leaving the precise responsible modeling choice unresolved. It does **not** show that more capacity, more seeds, or a larger undirected training run will solve it.

## WP23 minimal structural ablations

After the WP22 scoring diagnosis, only the three structures named by WP23 were compared: existing A2, plus a single seed-101 training of A0 and A1 each with the exact same training data, run-balanced sampler, curriculum, stage budget, frozen validation and practice windows. The run used the same 120 updates per variant (40 each at 0.5/2/5 seconds). No test/final-test data was opened, no simulator was launched, and the A2 candidate was not retrained. A3 was omitted because WP17's causal interventions did not show processed rear-wheel feedback to be necessary recursive body state.

| Variant | OpenPlane position RMSE, 2 s | OpenPlane position RMSE, 5 s | Practice position RMSE, 2 s / 5 s | Primary wins | 0.5 m crossing median / later runs | Decision |
|---|---:|---:|---:|---:|---:|---|
| A0: no latent state | 2.401 m | 13.383 m | 1.314 / 9.299 m | 0/6 | 1.188 s / 3 of 6 | reject |
| A1: no nominal anchoring | 3.869 m | 9.803 m | 1.193 / 2.690 m | 1/6 | 0.250 s / 0 of 6 | reject |
| A2: latent + stable nominal + residual | 2.212 m | 12.663 m | 1.584 / 11.947 m | 4/6 | 1.406 s / 4 of 6 | reject |
| WP19 parent | 2.055 m | 11.643 m | 1.246 / 8.008 m | comparator | 1.094 s | — |

All errors are run-macro means. A0 is 16.8% worse than the parent on the primary 2-second position metric; its paired parent-minus-A0 interval is wholly negative `[-0.6271, -0.0856] m`, and it wins none of six runs. It reduces 2-second forward-speed error by 19.6%, but increases yaw error by 20.5%; at 5 seconds yaw error is 53.9% worse. Its modestly later median position threshold crossing does not overcome the full-trajectory loss.

A1's primary 2-second position error is 88.3% worse than the parent; its 95% paired interval is `[-2.975, -0.583] m` for parent-minus-candidate, and it wins only 1/6 runs. The first 0.5 m crossing is at 0.25 s median, earlier than parent on all six runs. A1's 5-second position and practice-position values can look better in isolation, but it has OpenPlane `u` RMSE `2.865 m/s` at 2 s and `2.896 m/s` at 5 s (parent `0.376` and `0.781 m/s`), so this is not coherent movement prediction or a transferable plant.

A2 remains the best of these three on the pre-registered primary OpenPlane metric, but is still 7.6% worse than the parent, its paired interval crosses zero, and its practice/yaw regressions remain. The WP23 aggregate report therefore selects no architecture. Full per-run data and gates are in [`wp23_structural_ablation_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp23_structural_ablations_seed101_v1/wp23_structural_ablation_report.json), with individual A0/A1 reports and checkpoints alongside it.

## Gate disposition

- WP20: closed; run-balanced kNN support definition frozen narrowly for research use.
- WP21: closed; architecture and required structural tests implemented/passed.
- WP22: closed as a failed candidate branch; seed 101 retained only as an unpromoted comparator.
- WP23: closed; the explicitly allowed A0/A1 comparisons were run once and no A0/A1/A2 variant passed. No additional seed or training-budget escalation is authorized by the handoff.
- WP24 fixed validation suite, WP25 independent AutoDRIVE confirmation, WP26 historical configuration ranking, and WP27–WP30 optimizer work: **blocked**. No architecture was selected for full validation.

The exact machine reports are [`wp22_training_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/wp22_training_report.json) and [`wp22_failure_diagnosis.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp22_augmented_state_space_seed101_smoke_v1/wp22_failure_diagnosis.json). The ongoing progress record is [`BLACK_BOX_OFFLINE_PLANT_PROGRESS_20261003.md`](BLACK_BOX_OFFLINE_PLANT_PROGRESS_20261003.md).
