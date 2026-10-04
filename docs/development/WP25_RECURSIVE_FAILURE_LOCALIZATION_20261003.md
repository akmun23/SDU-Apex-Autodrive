# WP25 — Recursive Failure Localization (2026-10-03)

## Decision in brief

The WP22 A2 checkpoint is reproducible but is not an accurate offline plant. The clearest structural failure is that its recursively advanced latent state does not agree with the same model's causal history encoder—even after one 25 ms step. Encoder feedback and WP20 support do not explain the large rollout error by themselves: the future-encoder oracle does not help, and support-loss gradients are negligible after the first short horizon. Support escape is an early warning of subsequent error on these starts, but it is not evidence that the support penalty caused the error.

The WP25 evidence authorizes only the fixed WP26 mechanism-isolation comparisons D1, D2, and D3 in the handoff. D4 is not indicated: neutralizing measurement feedback degrades prediction, and encoder prediction has not been shown to be dispensable during fitting. No production integration, raceline use, or WP27 redesign is authorized unless a WP26 candidate passes all seven gates.

## Frozen base and reproduction

- Source commit: `2129427838101eee277e17e91727810bbe2df687`.
- The refreshed experiment registry now hashes the WP19 selected checkpoint, WP20 report, WP22 A2 checkpoint/report/failure diagnosis, and WP23 A0/A1 checkpoints/reports. Each associated checkpoint hash was checked against its recorded training report.
- `frozen_comparators.json` records WP19, A2, A0, A1, their SHA-256 values, dataset hashes, registry hash, and source commit.
- `frozen_eval_starts.json` records every original WP22 development-validation and practice-diagnostic start, including run, sequence, row, condition, canonical initial regime labels, and available contiguous horizon. The diagnostic reused these exact starts.
- A2 reproduction compared 460 saved run/horizon/channel values. Maximum absolute difference was **0.0**; the saved and reproduced material-gate decisions match exactly. WP25 therefore passed the WP24 provenance gate.
- No simulator was launched. WP25 performed **zero optimizer steps** and did not change any checkpoint.

The six OpenPlane runs are development validation, not untouched confirmation. The two practice runs remain diagnostic only.

## How poor is the frozen rollout?

Run-macro mean trajectory errors for A2 (the stored WP22 candidate, unchanged):

| Split / horizon | Position radial RMSE | Heading RMSE | `u` RMSE | `v` RMSE | Yaw-rate RMSE | Encoder-increment RMSE |
|---|---:|---:|---:|---:|---:|---:|
| OpenPlane / 0.25 s | 0.015 m | 0.019 rad | 0.046 m/s | 0.020 m/s | 0.197 rad/s | 4.765 rad/sample |
| OpenPlane / 0.75 s | 0.234 m | 0.128 rad | 0.153 m/s | 0.041 m/s | 0.411 rad/s | 4.666 rad/sample |
| OpenPlane / 2 s | 2.212 m | 0.470 rad | 0.397 m/s | 0.064 m/s | 0.613 rad/s | 4.657 rad/sample |
| OpenPlane / 5 s | 12.663 m | 1.164 rad | 0.878 m/s | 0.101 m/s | 0.907 rad/s | 4.700 rad/sample |
| Practice / 2 s | 1.584 m | 1.078 rad | 0.810 m/s | 0.009 m/s | 1.555 rad/s | 2.116 rad/sample |
| Practice / 5 s | 11.947 m | 2.142 rad | 1.465 m/s | 0.024 m/s | 1.655 rad/s | 2.551 rad/sample |

These are errors on the handoff's frozen overlapping windows, averaged within each independent run and then across runs. The two-run practice values are not a population-level estimate. The large multi-second errors are incompatible with using A2 as a trusted full-lap offline simulator.

## WP25.2 — Measurement-feedback interventions

Paired differences below are intervention minus native A2; positive means worse. OpenPlane intervals resample the six runs, not the overlapping windows.

| Intervention | 2 s position RMSE delta (95% run CI) | 2 s yaw RMSE delta (95% run CI) | 5 s `u` RMSE delta (95% run CI) | 5 s yaw RMSE delta (95% run CI) |
|---|---:|---:|---:|---:|
| M1 neutral measurement input | +0.02084 m (+0.00526, +0.03572) | +0.00626 (+0.00511, +0.00755) | +0.14498 m/s (+0.10791, +0.17324) | +0.01395 (+0.01224, +0.01582) |
| M2 hold last generated measurement | +0.00039 m (+0.00021, +0.00057) | +0.000054 (+0.000050, +0.000059) | +0.00094 m/s (+0.00064, +0.00120) | +0.000025 (+0.000002, +0.000043) |
| M3 zero left/right differential, preserve mean | −0.00040 m (−0.00070, −0.00007) | −0.000087 (−0.000172, −0.000004) | +0.00038 m/s (+0.00034, +0.00043) | −0.000112 (−0.000233, −0.000019) |
| M4 future measured encoder oracle | +0.01898 m (+0.00579, +0.03207) | +0.00596 (+0.00510, +0.00671) | +0.13787 m/s (+0.11068, +0.16432) | +0.01317 (+0.01219, +0.01423) |

The oracle had **99.88%** valid encoder-label coverage on the OpenPlane intervention windows; invalid labels were replaced with the training-run mean. Supplying nearly all actual next encoder measurements did not improve the rollout. Completely neutralizing the generated output did worsen it, while holding the output or removing its differential had effects near zero in absolute terms. The most defensible reading is that the learned generated-output feedback has become an internal recurrent feature, not that accurate real encoder feedback is a useful substitute. M4 is intentionally non-causal and diagnostic only.

**Decision:** the feedback path is questionable but not proven to be the main error source. D2 is retained as a retrained structural ablation because the oracle gives no benefit and the real output is extremely noisy; the inference interventions alone do not justify deleting the path from A2. The small M3 benefit is outweighed by its small `u` regression and is not a material dynamics improvement.

## WP25.3 — Support/error chronology

The first time a rollout exceeded the WP20 unsupported threshold preceded the `0.25 m/s` `u` error and `0.25 m` position error on all 6/6 OpenPlane runs and both practice runs. It preceded the `0.25 rad/s` yaw-rate error on 97.7% of OpenPlane starts (run-cluster 95% interval 95.3–99.7%) and 100% of the practice starts. Nine of 384 OpenPlane starts tied unsupported-support crossing with the yaw threshold; the rest had support crossing first. The boundary was often crossed in the first 25 ms, sometimes after a weak-support initial condition.

The support score was calibrated against WP19, not A2. Consequently this establishes an early OOD indicator for these A2 trajectories, not that low support physically caused the later error. The independent gradient result below is important: support's training gradient becomes vanishingly small as rollout horizon grows.

## WP25.4 — Frozen loss-gradient attribution

Gradients were computed on one deterministic, run-balanced, 10-run batch from frozen training windows. `support` is the actual weighted objective contribution (`0.02 × support loss`); no optimizer was constructed or stepped.

| Horizon | Trajectory gradient L2 | Support gradient L2 | Support / trajectory norm | Cosine: trajectory vs support | Cosine on yaw-output row |
|---|---:|---:|---:|---:|---:|
| 0.25 s | 0.1256 | 0.000173 | 1.38e−3 | −0.228 | −0.883 |
| 0.5 s | 0.4028 | 0.000201 | 4.98e−4 | −0.088 | −0.365 |
| 1 s | 2.6615 | 0.000208 | 7.82e−5 | +0.224 | +0.860 |
| 2 s | 23.8199 | 0.000211 | 8.85e−6 | +0.211 | +0.919 |
| 5 s | 260.4848 | 0.000183 | 7.02e−7 | +0.254 | +0.973 |

The short-horizon yaw-row directions oppose one another, but support's contribution is already only 0.14% of the trajectory-gradient norm at 0.25 s and falls below one-millionth by 5 s. This is not a materially competing training gradient and does not explain the large recursive drift. D1 remains in the small fixed WP26 comparison as a confirmation of the isolated mechanism, not as a presumed fix.

## WP25.5 — Latent transition versus causal re-encoding

The frozen model was rolled with recorded commands only. At each horizon its latent was compared with the same history encoder applied to the actual causal sensor/command history ending at that future time. Latent RMSE is normalized per dimension by training-only encoded-state standard deviation.

| Horizon (K) | Training normalized RMSE | OpenPlane validation normalized RMSE | Validation latent L2 RMSE | Validation cosine similarity |
|---|---:|---:|---:|---:|
| 25 ms (1) | 5.84 | 5.92 | 0.663 | +0.160 |
| 100 ms (4) | 11.28 | 11.71 | 1.347 | −0.089 |
| 250 ms (10) | 13.17 | 13.56 | 1.584 | −0.113 |
| 500 ms (20) | 13.47 | 13.87 | 1.621 | −0.115 |
| 1 s (40) | 13.51 | 13.76 | 1.610 | −0.114 |
| 2 s (80) | 13.41 | 13.61 | 1.596 | −0.103 |

Adjacent real-window encoder states move only 0.00319 latent-L2 on average (median 0.00189). The rolled latent is therefore not tracking the representation that the model itself reconstructs from causal history. The mismatch is already severe after one model step, grows rapidly over 100–250 ms, and then remains very large through 2 s. It is present on training as well as held-out validation windows, so it is not simply a practice-domain shift.

The machine report also contains per-run, per-horizon values for all eight latent dimensions, L2 error, cosine similarity, and within-run growth slope, stratified by canonical start bins for speed, steering, steering rate, throttle command/slew, wheel/body mismatch, and combined regimes. At 2 s, the run-macro validation summaries for selected difficult combined regimes were:

| Development-validation start regime | Runs with starts | Starts | Mean normalized latent RMSE/dimension | Latent L2 RMSE |
|---|---:|---:|---:|---:|
| 7–9 m/s high steering | 2 | 2 | 13.68 | 1.763 |
| High-speed moderate steering | 3 | 29 | 14.91 | 1.857 |
| Steering turn-in | 2 | 9 | 11.67 | 1.375 |
| Large wheel/body mismatch | 6 | 10 | 13.09 | 1.584 |

The two-run high-steering and turn-in groups are weak evidence, not population estimates. These results show that latent inconsistency is also present in the available difficult-region starts, but do not establish a regime-specific cause.

**This is the strongest WP25 structural diagnosis.** A latent transition that drifts away from its own encoder's state representation cannot serve as a coherent persistent hidden state for free-running simulation.

## WP25.6 — Encoder error lead/lag

Correlations are computed per rollout and averaged within runs over lags 0–1.5 s; they are associative, not causal. OpenPlane's largest absolute correlations were modest: differential error versus future `u` error −0.315 at 1.1 s, pair-mean error versus future `u` error +0.269 at 0.2 s, and either encoder error versus position-error growth at most about 0.18. Practice had larger maxima (up to 0.75) but only two runs, so those are weak transfer diagnostics. These results do not support encoder error as the primary OpenPlane divergence driver. M0 encoder-increment RMSE itself is still very poor (4.66 rad/sample over the 2 s OpenPlane windows); the large one-step training residual distribution also makes the train-only 99th-percentile event threshold conservative, so “no threshold crossing” must not be read as accurate encoder prediction.

## WP25 answers and authorized next step

1. **Does generated encoder feedback help?** Its effect is mixed: neutralizing the entire output worsens A2; holding it produces nearly unchanged behavior; zeroing its left/right differential yields a tiny position/yaw improvement with a tiny `u` regression. The future-measurement oracle, with 99.88% valid coverage, is worse than native A2. The generated channel functions as a learned recurrence, but measured wheel output has not been shown to improve the body model.
2. **Does support regularization materially fight trajectory fitting?** No. There is a negative short-horizon yaw-row cosine, but the weighted gradient is too small to account for the rollout failure. Support is an early warning, not an established cause.
3. **Is latent transition consistent with causal re-encoding?** No. Normalized latent error is 5.8 standard deviations/dimension after 25 ms and about 13.5 by 250 ms, on both training and validation.
4. **What fails first?** Support usually leaves its WP19-calibrated domain first (often within 25 ms); then `u` error, then position error, with yaw error often later. The train-only encoder-divergence threshold is not crossed on these runs, but the encoder RMSE and threshold scale make that result inconclusive for measurement accuracy.

WP26 is limited to the handoff's seed-101 D1/D2/D3 mechanism comparisons, using the same 120 updates, same frozen data/splits/starts, and 0.5/2/5 s horizons. D4 is not indicated. If no variant passes all seven WP26 criteria, stop the A2 family; do not proceed to WP27, controller integration, or raceline optimization on this model.

## Artifacts

- Machine result: `live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/next_phase_after_2129427/wp25_failure_localization/wp25_score_only_diagnostics.json`
- Frozen loss gradients: `.../wp25_failure_localization/wp25_loss_gradient_attribution.json`
- Exact starts and checkpoint provenance: `.../next_phase_after_2129427/frozen_eval_starts.json` and `frozen_comparators.json`
- Per-step M0 traces: `.../wp25_failure_localization/traces/M0_*.npz` (eight independent run archives; full 25 ms state, body residual, latent, encoder, support, pose and regime arrays)
- Diagnostic implementations: `tools/vehicle_dynamics_learning/trace_wp22_internal_rollout.py` (full WP25 runner; `--latent-only` refreshes the six-horizon, start-region consistency analysis without rerunning interventions) and `tools/vehicle_dynamics_learning/diagnose_wp22_loss_gradients.py`

All results use existing data. There are no untouched confirmation runs in this work package.
