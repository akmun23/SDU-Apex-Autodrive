# WP26 — Mechanism Ablations and Stop Decision (2026-10-03)

## Outcome

None of the three handoff-authorized variants D1/D2/D3 passed the predeclared WP26 mechanism gate against frozen A2. Per the handoff, stop this A2 architecture family here. Do not start WP27, add seeds, integrate a checkpoint, or run a raceline/optimizer test with these models.

This is a negative but useful result: removing the WP20 support training penalty had essentially no primary-metric effect; removing generated encoder feedback reduced forward-speed error but made pose/heading accuracy worse. Neither mechanism explains or fixes A2's dominant recursive failure.

## Frozen protocol

- Source commit: `2129427838101eee277e17e91727810bbe2df687`.
- A2 comparator SHA-256: `312a3146e0f8a9ec9e860bdd80dd24de5a80883efb23fcd9ac8c17c44abcbde2`.
- Seed: 101 for all variants; no seed expansion.
- Each variant received 120 updates: 40 updates at each of 0.5 s, 2 s, and 5 s.
- All used the same train runs, six OpenPlane development-validation runs, two practice-diagnostic runs, exact WP24 start set, batch/sampler, and identical sampled-ref sequence (SHA-256 `63384d9cfa13ab9dc521b5bbccb73b00d430ac167662e0deead6900fa7d7accd`).
- D1 set only support-loss weight to zero. D2 removed only generated encoder measurement from the latent-transition input. D3 removed both. The measurement head remained present and scored for D2/D3.
- No simulator or test/final-test split was used. All outputs remained research artifacts.

## Seven-gate results

The gate compares macro-run metrics to A2; crossing time is paired per validation run. A positive position-error change is worse.

| Criterion | D1 | D2 | D3 |
|---|---:|---:|---:|
| OpenPlane 2 s position RMSE change vs A2 | 0.0% | +4.61% | +5.11% |
| OpenPlane 2 s yaw within +5% | pass | pass | pass |
| OpenPlane 2 s `u` regression ≤5% | pass (slight improvement) | pass (improves) | pass (improves) |
| 0.5 m crossing later on ≥4/6 runs | fail (0/6) | fail (0/6) | fail (0/6) |
| No practice position+yaw regression >20% at same horizon | pass | pass | pass |
| All recursive outputs finite | pass | pass | pass |
| Primary position improves on ≥3 independent runs | pass (4/6, but aggregate gain is 0.0%) | fail (0/6) | fail (0/6) |
| **Overall** | **fail** | **fail** | **fail** |

| Candidate | OpenPlane 2 s position RMSE | 2 s yaw RMSE | 2 s `u` RMSE | Practice+finiteness | Decision |
|---|---:|---:|---:|---|---|
| Frozen A2 | 2.212 m | 0.613 rad/s | 0.397 m/s | Baseline | Failed earlier material gate |
| D1 | 2.212 m | 0.613 rad/s | 0.396 m/s | No combined >20% practice regression; finite | No material position or divergence-time gain |
| D2 | 2.314 m | 0.626 rad/s | 0.307 m/s | No combined >20% practice regression; finite | Better `u`, worse pose; no position wins |
| D3 | 2.325 m | 0.630 rad/s | 0.308 m/s | No combined >20% practice regression; finite | Better `u`, worse pose; no position wins |

The 5% yaw allowance and 5% `u` regression allowance were satisfied by all three. These isolated passes do not offset failure of the primary position requirement and the 0.5 m divergence-time gate.

## What this says about the model

- **Support regularization is not a useful fix here.** D1 is effectively identical to A2 in aggregate position/yaw and only changes `u` by about 0.001 m/s. This agrees with WP25's support-gradient norm, which was below 0.14% of the trajectory gradient at 0.25 s and fell rapidly with horizon.
- **The measurement feedback path trades speed error against pose error.** D2 lowers 2 s `u` RMSE by about 23%, but increases position RMSE by 4.6% and yaw RMSE by about 2.2%; D2's position was worse on all six OpenPlane runs. D3 is slightly worse still on position/yaw. Removing that path is not a plant improvement.
- **The dominant latent inconsistency remains.** None of these variants addressed the observed mismatch between rolled latent state and causal re-encoding. The pre-registered WP26 budget was for mechanism isolation, not a long-horizon training redesign; it does not authorize continuing training after all three variants fail.

## Handoff gate and next state

WP27 is conditional on a WP26 mechanism candidate qualifying. Since the eligible set is empty, the branch stop is final for these A2 variants. This package did not produce an offline simulator suitable for full-lap simulation, raceline generation, MPC tuning, or controller integration. The next work requires the handoff's simpler nonlinear transition formulation, not more training on D1/D2/D3 or another A2 seed.

## Artifacts

- Machine gate: [`wp26_mechanism_ablation_report.json`](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp26_mechanism_ablations/wp26_mechanism_ablation_report.json)
- Candidate reports/checkpoints: sibling `D1/`, `D2/`, and `D3/` folders. Each contains `metadata.json`, `training_report.json`, `evaluation_report.json`, `checkpoint.pt`, `checkpoint.sha256`, `dataset_manifest.json`, and `source_commit.txt`.
- Trainer: [`run_wp26_mechanism_ablations.py`](../../tools/vehicle_dynamics_learning/run_wp26_mechanism_ablations.py)
- Model variant: [`augmented_state_space_plant.py`](../../tools/vehicle_dynamics_learning/augmented_state_space_plant.py)
- Full WP25 reasoning: [`WP25_RECURSIVE_FAILURE_LOCALIZATION_20261003.md`](WP25_RECURSIVE_FAILURE_LOCALIZATION_20261003.md)
