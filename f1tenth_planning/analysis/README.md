# Live-data baseline and acceptance campaign

Only source-timestamped simulator recordings are retained here. Ground truth
is diagnostics-only: it is never supplied to odometry, EKF, AMCL, or a
controller.

The canonical workflow is:

1. record a graphical Unity run with `-batchmode`, the ground-truth monitor,
   and the telemetry recorder;
2. score the monitor CSV with `score_localization_run`, passing the raw
   telemetry CSV so collision callbacks cannot be missed;
3. use scan matching only on scans from that same recording.

Native track sensor streams are approximately 20 Hz. The bridge's 40 Hz name
is its request clock, not a fabricated sensor rate; every report uses source
timestamps and records duplicates/gaps.

## Current retained evidence

The slow runs are diagnostic baselines. The high-speed runs are acceptance
evidence and are intentionally retained even when rejected, because they show
which runtime failure must be fixed next:

| Run | Speed cap | Result | Meaning |
| --- | ---: | --- | --- |
| `amcl_clean_baseline_20260909` | 0.5 m/s | collision-free | slow production baseline |
| `amcl_tuning_campaign_20260909/candidate_c` | 0.5 m/s | collision-free | 4 cm AMCL gate candidate; not promoted |
| `amcl_tuning_campaign_20260909/control_a_2mps` | 2.0 m/s | collision | pre-turn-entry-fix control; launch odometry lost about 1.5 m |
| `amcl_tuning_campaign_20260909/ekf_cov_2mps_retry` | 2.0 m/s | collision | EKF covariance candidate; launch odometry still failed |
| `amcl_tuning_campaign_20260909/turn_entry_fix_2mps` | 2.0 m/s | collision | turn-entry odometry fixed; estimator stayed near 5--12 cm, but the start bend/steering response failed |

The 2 m/s runs are not acceptance passes. Do not lower the speed cap and call
that acceptance; the next controller/path change must be retested at the real
operating speed and must complete the track without collision while remaining
within the localization threshold.
