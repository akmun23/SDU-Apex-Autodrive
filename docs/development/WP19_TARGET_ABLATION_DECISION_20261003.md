# WP19 target/state ablation decision — 2026-10-03

This closes WP19 of the 2026-10-03 black-box offline-plant handoff. It selects representations for the next research stage only. It does **not** promote a checkpoint, establish an accurate offline simulator, or authorize use in odometry/MPC.

## Controlled comparison

The corrected run trained all 12 body/wheel target combinations with the same 12 independent OpenPlane training runs, same run-uniform window sampling, GRU hidden size 32, fixed 2-second history, same 300 AdamW optimizer steps, batch size 8, and seed `20261019`. The first target was body-state increment and the preferred direct encoder-angle output. All candidates used the same transition windows and the same adjacent-valid encoder labels. All six OpenPlane validation runs and both unseen-practice runs contributed 64 evenly spaced starts each. Metrics are macro-averaged by independent run; test/final-test data were not loaded.

The history input was observable bridge odometry `u/v/r`, actuator feedback, and commands. Simulator rigid-body state was used as a label and as the known state at each rollout start, never as future rollout input. Recursive prediction used model-predicted body and actuator states plus the known future command sequence. The fixed encoder sidecar's row/sequence identity was checked against the source view; its inherited parent SHA-256 was independently matched to the actual OpenPlane and practice parent datasets. Its fixed-cadence wheel labels are intentionally not byte/value-equal to the variable-source-rate view.

The initial v1 result is explicitly superseded. A post-run audit found that 213 sampled rows had a valid next encoder sample but invalid current sample. W-C/W-D had therefore received slightly more wheel-output supervision than W-A/W-B. The corrected v2 masks every target to the same current-and-next valid transitions. The first report remains only as an audit record and must not be used for selection.

Corrected machine report: [WP19 report](../../live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask/wp19_target_ablation_report.json)

## Result and representation freeze

Freeze **B-B body-state increment with soft acceleration-consistency loss** and **W-D encoder angle increment** for WP20/WP21 research. Keep B-A, B-C, and the frozen parent as comparators. This is a cautious representation choice: B-B is the strongest compromise on the six dynamic validation runs, not a universally superior plant.

| Body target | OpenPlane 2 s trajectory RMSE `u / v / yaw-rate` | Practice 2 s trajectory RMSE `u / v / yaw-rate` | One-step `u`, OpenPlane / practice |
| --- | ---: | ---: | ---: |
| B-A effective acceleration | 0.716 / 0.318 / 0.705 | 1.323 / 0.147 / 1.346 | 0.027 / 0.055 |
| B-B state increment | **0.680 / 0.163 / 0.616** | 1.483 / **0.079** / 1.510 | **0.023 / 0.054** |
| B-C next state | 1.173 / 0.193 / **0.550** | **0.895 / 0.085 / 0.720** | 1.038 / 0.830 |

For B-B/W-D, OpenPlane run-bootstrap 95% CIs for the 2-second trajectory RMSE are `u` 0.376–1.015 m/s, `v` 0.045–0.387 m/s, and yaw rate 0.413–0.792 rad/s (6 runs). Relative to B-A, B-B macro errors are lower by 5%, 49%, and 13%; it wins per-run on `u` 3/6, `v` 6/6, and yaw rate 4/6. The intervals overlap, so do not describe this as a proven accuracy gain.

The two practice runs show a real tradeoff: B-B is about 12% worse than B-A in 2-second `u` and yaw-rate trajectory error, while about 46% better in `v`. B-C wins all three 2-second practice channels on these two runs, but has very poor first-step `u` error and worse OpenPlane validation `u` prediction. Its low practice error is not enough evidence to select it as the global representation. Two independent practice runs cannot establish robust transfer.

The selected B-B/W-D branch had encoder-angle-increment RMSE 0.540 rad/sample on the six OpenPlane validation runs (run-bootstrap 95% CI 0.509–0.578), equivalent to approximately 1.28 m/s at the canonical 25-ms interval and 0.059-m wheel radius. This is not yet a sufficiently accurate encoder simulator.

All candidate gradients and recursive outputs were finite. The selected candidate's pre-clip gradient norm was p50 0.475, p90 1.393, maximum 3.845; finite output is only a stability check, not an accuracy criterion. Train-only run-balanced B-B target scales were `[0.0654, 0.0476, 0.1354]` for `[Delta_u, Delta_v, Delta_r]`.

At fixed 25-ms cadence and fixed wheel radius, W-A vs W-B and W-C vs W-D are affine unit changes. After per-target standardization their predictions are nearly equivalent; W-D is selected because it emits the direct measured encoder quantity required downstream and follows WP17's “wheel rate is measurement/output only” decision—not because the unit conversion itself adds information.

## Artifacts and verification

- Corrected trainer: [`run_wp19_target_ablation.py`](../../tools/vehicle_dynamics_learning/run_wp19_target_ablation.py)
- Corrected report: `.../wp19_target_ablation_v2_common_encoder_mask/wp19_target_ablation_report.json`
- Selected diagnostic checkpoint: `.../wp19_target_ablation_v2_common_encoder_mask/07_body_state_increment__encoder_angle_increment/model.pt`
- Selected checkpoint SHA-256: `045e98e1d0aafd9ec369bc6fdb1bdbce4e94491dec101a764f33195fe09e7541`
- Superseded first-pass output: `.../wp19_target_ablation_v1/`; its report records the mask issue and points to corrected v2.
- Focused WP19/WP18/region tests: 17 passed. The tests cover rigid-body equivalence of A/B/C representations, acceleration-consistency inversion, target conversions and masks, future-command alignment, observable-only history, run-balanced normalization, and sequence/window boundaries.

## Gate status

WP19 is complete with a provisional representation freeze. No architecture has been integrated or promoted. Proceed to WP20 only: compare run-balanced kNN distance and local Mahalanobis support, calibrate both against corrected held-out recursive errors, and expose no support confidence unless low-support regions demonstrably have higher errors. WP21 serious training remains gated on WP20.
