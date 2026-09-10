# Clean live baseline

This is a real graphical AutoDRIVE ICRA-compete simulator run. Unity was
started with `-batchmode` and without `-no-graphics`; the bridge disabled
camera decode/publication. Ground truth was recorded only for offline scoring.
It was never supplied to odometry, EKF, AMCL, Pure Pursuit, or the actuator.

Run configuration:

- active map: `autodrive_track_ftg_commit_20260909_025m.yaml`
- active trajectory: `autodrive_track_ftg_commit_20260909_025m_mintime_raceline.csv`
- controller: Pure Pursuit, explicit 0.5 m/s cap
- recording window: 60 s telemetry; monitor continued to 89.377 s
- scoring alignment: relative to the first clean GT/AMCL pair
- maximum collision count: 0; no post-collision rows

Quantitative result from `localization_score.json`:

- 1,666 aligned samples over 42.324 m of GT travel
- AMCL error: mean 0.0483 m, p95 0.0960 m, maximum 0.1957 m
- current-map pose error: mean 0.0438 m, p95 0.1147 m, maximum 0.1597 m
- EKF error: mean 0.0606 m, p95 0.1176 m, maximum 0.1508 m
- odom error: mean 0.0596 m, p95 0.1170 m, maximum 0.1208 m
- no estimator jumps above the 0.25 m diagnostic threshold

Measured timing from the 60 s telemetry CSV:

- LiDAR, IMU, encoders, GT odometry: median 19.77 Hz
- odom/EKF/AMCL: median 19.77 Hz, 19.78 Hz, and 19.78 Hz respectively
- steering/throttle commands: approximately 40 Hz
- sensor gaps above 75 ms: 2--3 per stream; maximum observed gap 99.25 ms
- no duplicate or synthetic sensor samples were used

Files:

- `amcl_vs_ground_truth.csv`: full monitor comparison
- `telemetry/sensor_record_20260909_171454.csv`: lossless callback telemetry
- `localization_source_time_diagnostic.csv`: source-time offline reconstruction
- `localization_score.json`: path-normalized estimator score
- `source_time_diagnostic.json`: event counts and pre-collision diagnostics
- `controller.log`: exact launch/runtime log

This is a slow baseline only. It does not prove high-speed behavior or claim
that the absolute position error is within 2% at racing speed. Any model
change must be tested against a new live recording with the same fields.
