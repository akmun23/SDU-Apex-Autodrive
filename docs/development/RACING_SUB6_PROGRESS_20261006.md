# Practice-track under-6 progress — 2026-10-06

## Bottom line

The short practice circuit has been driven below 6 seconds per lap in complete,
collision-free simulator sessions. The strongest full-session result found in
the saved runs is `practice_9g_runtime_matched_full_r01_20261005`: 10/10 scored
laps, best 5.7448 s, mean 5.7893 s, and zero collisions. Every scored lap in
that run was under 6 seconds. It completed the warmup and extra lap as well.

This is real simulator evidence, not an optimizer estimate. However, the run
has a bag and analysis report but no `run_manifest.json`, so its exact runtime
image/source/configuration provenance is incomplete. Treat it as a strong
measured result, not yet as a fully reproducible promoted build. The separate
2026-10-06 current-image screen has complete input hashes but only one scored
lap; it is supporting evidence, not a repeatability test.

The sub-5-second target has **not** been achieved. The best complete measured
mean is still about 0.79 s above 5 seconds (about 0.85 s above the external
4.94 s reference); optimizer estimates below that threshold are not simulator
results.

## Measured progression

| Evidence | Scored laps | Best / mean (s) | Collisions | Interpretation |
|---|---:|---:|---:|---|
| Fresh R0 reference, two runs combined | 20/20 | 6.0087 best / 6.0670 mean | 0 | Reproducible safe parent; one warmup + 10 scored + extra per run. |
| Spatial-minimum-time candidate, saved run labeled legacy-yaw | 10/10 | 5.8808 / 5.9222 | 0 | Complete under-6 result with a run manifest; separate candidate/configuration, not a controlled A/B against R0. |
| 8.4 m/s² envelope, runtime-matched candidate | 10/10 | 5.8618 / 5.9095 | 0 | Complete safe test; all scored laps under 6 seconds and a saved run manifest. |
| 8.4 candidate with current-source path | 10/10 | 5.9558 / 5.9908 | 0 | Mean under 6, but some individual laps exceeded 6; useful evidence that a candidate/config change is not automatically faster. |
| 9.0 m/s² envelope, runtime-matched candidate | 10/10 | **5.7448 / 5.7893** | **0** | Best complete measured run located; all scored laps under 6. Manifest is missing. |
| Current-image 9g screen (2026-10-06) | 1/10 | 5.8008 / 5.8008 | 0 | One scored lap only; full manifest exists, but not enough laps to establish repeatability. |

The strongest full run improved mean time by 0.2777 s (4.6%) versus the two-run
R0 mean. Its best lap was 0.2639 s (4.4%) faster than the best R0 lap. The
8.4-to-9.0 candidate comparison improved the measured mean by a further
0.1202 s (2.0%), but this comparison is not perfectly isolated: the old 9g
run lacks a manifest tying every runtime input to hashes.

Reports and source artifacts:

- [Fresh R0 comparison and baseline details](RACING_OPTIMIZATION_PAUSE_20261005.md)
- [9.0 m/s² full-run analysis](../../live_runs/practice_9g_runtime_matched_full_r01_20261005/analysis/race_report/summary.md)
- [9.0 m/s² optimizer configuration](../../live_runs/raceline_candidates/practice_9g_runtime_matched_wallmargin010_20261005/optimizer_config.yaml), [resolved model/configuration](../../live_runs/raceline_candidates/practice_9g_runtime_matched_wallmargin010_20261005/output/resolved_config.yaml), and [converged 0.18 m solution report](../../live_runs/raceline_candidates/practice_9g_runtime_matched_wallmargin010_20261005/output/checkpoints/mesh_0p180/report.json)
- [8.4 m/s² full-run analysis](../../live_runs/practice_8g_runtime_matched_lat84_default_full_r01_20261005/analysis/race_report/summary.md)
- [First spatial-minimum-time full-run analysis](../../live_runs/practice_spatialmintime_legacyyaw_r01_20261005/analysis/race_report/summary.md) and [its run manifest](../../live_runs/practice_spatialmintime_legacyyaw_r01_20261005/run_manifest.json)
- [8.4 m/s² run manifest](../../live_runs/practice_8g_runtime_matched_lat84_default_full_r01_20261005/run_manifest.json)
- [Current-image one-lap screen](../../live_runs/practice_proven9g_current_image_screen_r01_20261006/analysis/race_report/summary.md)
- [Current-image screen manifest](../../live_runs/practice_proven9g_current_image_screen_r01_20261006/run_manifest.json)

## What produced the faster measured laps

The substantial change was in the *candidate raceline and its speed profile*,
not a demonstrated AMCL or odometry breakthrough. The optimizer was made
practice-track/AutoDRIVE-specific and used an explicit Frenet spatial
minimum-time formulation instead of simply asking the existing MPC to track a
more aggressive version of the old reference.

For the strongest 9g candidate, the saved resolved optimizer configuration
shows the following design choices. The full-session 9g run lacks a manifest,
so these are the saved candidate settings associated with that run name, not
an independently hash-verified reconstruction of every runtime input:

- Raised the optimizer's lateral-acceleration envelope from the 8.0 m/s²
  runtime profile to a 9.0 m/s² candidate envelope (scale 1.125). The preceding
  8.4 m/s² candidate was tested first; 9.0 was a measured-data-informed further
  step, not an arbitrary unlimited grip assumption.
- Included longitudinal/lateral combined-acceleration coupling with exponent 2,
  so the speed plan could not independently spend the full longitudinal and
  lateral limits at once.
- Used the odometry lateral-velocity term in the lateral dynamics and a
  high-steering yaw-gain reduction beginning around 0.41 rad and reaching its
  configured end around 0.46 rad.
- Kept explicit steering magnitude/rate, acceleration, braking, target-speed
  slew, heading-error, progress, and vehicle-footprint/map-wall constraints.
  The candidate configuration requested 0.40 m center-to-wall clearance,
  accounting for the planning footprint plus extra margin.
- Warm-started from the measured 8.4 candidate and used mesh continuation. The
  0.25 m and 0.18 m solves converged; refinement at 0.14 m failed, so the
  optimizer correctly retained the last converged 0.18 m result. This is a
  converged usable candidate, but not a fully refined mesh solution.

The optimizer report estimates 5.5961 s for that 9g line (5.5950 s after its
export-time recomputation). It reports a 28.16 m optimized path, maximum
planned speed about 8.10 m/s, and peak planned steering about 0.451 rad. The
actual full-run mean was 5.7893 s, roughly 0.193 s slower than the estimate.
Thus the optimizer was useful for finding a faster feasible line, but its time
estimate is still optimistic and is not accurate enough to substitute for a
closed-loop simulator run.

In the measured 9g run, the report gives p95 path cross-track error of about
0.111 m, p95 one-step speed prediction error of 0.189 m/s, and p95 yaw-rate
prediction error of 0.117 rad/s. These describe that run only; they do not
prove full-lap plant accuracy or a general localization improvement. No saved
evidence isolates an AMCL, odometry, or MPC code change as the cause of the
under-6 result.

## What did not work, and why it matters

- Not every optimizer candidate improved real performance. The 8.4
  current-source run averaged 5.9908 s, while a yaw-surface-matched run on a
  related line averaged 6.0590 s and recorded a collision. Keep each result
  classified by the exact line/configuration; do not promote based on a single
  best lap or an optimizer objective.
- A separate faster candidate collided after the optimizer and runtime used
  different yaw models: the optimizer had the yaw response surface enabled,
  while the controller run had no corresponding overlay and used the default
  surface-disabled model. At the recorded corner, predicted yaw rate was about
  -3.47 rad/s versus simulator truth about -1.37 rad/s. This is a strong
  contributing mismatch, not proof of the only cause of collision.
- The static empirical acceleration envelope failed independent validation
  and practice transfer, so its fitted limits were not promoted into the
  optimizer. Raising an envelope is only defensible as a bounded candidate
  followed by real simulator evaluation; it is not evidence that the car has a
  universal 9 m/s² grip limit.
- A previously quoted 5.40 s optimizer candidate was invalid: it used the wrong
  yaw-model setup and a wall-clearance acceptance that was too small. It is
  excluded from performance claims.
- The latest expanded surface/multistart search and geometry-consistent export
  have not established a faster real lap. The best geometry-consistent
  candidate estimate is about 5.720 s and it passes the saved map raycast
  clearance check, but has not been run in the simulator. A static replay over
  a prior bag showed additional MPC corridor/residual rejections; because the
  changed line reprojects the same recorded poses differently, that replay is
  diagnostic only, not a closed-loop result.

  Supporting artifacts: [global seed-scan optimizer report](../../live_runs/raceline_candidates/highspeed_yaw_surface_supportgate_v01_20261006/raceline_global_seed_scan/report.json), [geometry-consistent candidate report](../../live_runs/raceline_candidates/highspeed_yaw_surface_supportgate_v01_20261006/raceline_geometry_consistent/report.json), and [static counterfactual replay](../../live_runs/raceline_candidates/highspeed_yaw_surface_supportgate_v01_20261006/counterfactual_replay_geometry_consistent.json).

The latest work therefore advances the optimizer/model diagnostics, but does
not supersede the 5.7893 s measured full-run result. The current candidate is
not integrated, the reference trajectory remains preserved, and no simulator
or test process is running.

## Current status and next reproducibility gate

1. Preserve `practice_9g_runtime_matched_full_r01_20261005` as the best
   complete measured sub-6 result, while labeling its missing manifest clearly.
2. Use the 2026-10-06 current-image run only as a one-lap screen; do not call it
   a repeatability confirmation.
3. Before comparing another raceline, save a run manifest that hashes the map,
   exact trajectory, MPC/odom/EKF/AMCL configs and source, and simulator and
   controller images. Ensure optimizer and runtime use the same model and
   settings.
4. Validate the geometry-consistent candidate in the established batch-mode
   simulator procedure. Score a complete warmup + 10 laps + extra only if
   collision-free; stop immediately at the first collision. Compare mean,
   spread, best lap, wall clearance, tracking error, MPC rejections, and model
   residuals against the measured 9g result.
5. Do not claim sub-5 until a complete, collision-free simulator run records
   it. The current best optimizer estimates and offline replays do not meet
   that bar.

The detailed run ledger predates some of the full candidate reports above; this
note records the reconciliation so the complete 9g result is not lost or
mistaken for a fully provenance-complete promotion.
